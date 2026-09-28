import { pickFile } from "./actions";
import type { CookCase } from "../api/status";
import { useCookInputs } from "../state/cookInputs";
import { msg, say } from "../state/say";
import { uploadedLocal } from "../transfer/local";
import { sendPlanes } from "../transfer/planes";
import type { BlockContext } from "./nodes";
import { onDeclared, onUploaded, restoreUploads, sendUpload, uploadBlocker, useUploads } from "../transfer/uploads";
import { sizeText } from "../platform/format";

/** Uploads and the graph: a finished upload goes into its node's parameter when that graph is the one open (this tab
 * or another tab of the same graph), and the uploads a reload interrupted come back, paused, on their parameters. */
export function startUploads(): void {
  // 申报完成后即将引用写入参数：此时尚未传输任何字节，但该引用已是最终引用
  // （id 仅由文件名与内容计算，`lab2shot/transfer/uploads.py set_id`）。参数写入后，服务器即可按申报中的
  // 图层生成输出口（`nodes/core/input.py made_ports` 的 `_declared` 回退），使用者随后才能连线。
  onDeclared((task, ref) => {
    const ci = useCookInputs.getState();
    if (ci.meta.name !== task.graph || !ci.nodes[task.node]) {
      say(msg("N-UPLOAD-GRAPHGONE", { file: task.name, graph: task.graph }));
      return;
    }
    if (ci.nodes[task.node].params[task.param] !== ref)
      pickFile(task.node, task.param, { ref, stem: "", files: task.files.length, bytes: task.bytes,
                                        first: task.frames[0] ?? null, last: task.frames.at(-1) ?? null, folder: task.folder });
  });
  onUploaded((task, up) => {
    if (!task) return;
    const ci = useCookInputs.getState();
    const node = ci.nodes[task.node];
    if (ci.meta.name !== task.graph || !node) {
      say(msg("N-UPLOAD-GRAPHGONE", { file: task.name, graph: task.graph }));
      return;
    }
    const { name: _, ...picked } = up;
    uploadedLocal(task.key, task.node, task.param, up.ref);
    if (node.params[task.param] !== up.ref) pickFile(task.node, task.param, picked);
  });
  void restoreUploads();
}

/** 本次实际计算的节点中，哪些素材仍在使用者本机上（`picked`：已申报、字节未传输）。
 *
 * 本次计算涉及的节点由服务器给出（状态回复的 `policy.computes` 与 `deliver.computes`，
 * `lab2shot/engine/evaluation.py _case`）：结果已缓存的读取节点不在其中，其素材无须传输任何字节。
 * 网页不按连线自行反推，否则会产生第二份答案。 */
export function pickedFor(_ctx: BlockContext, cook: CookCase) {
  const here = new Set(cook.computes);
  return Object.values(useUploads.getState().tasks).filter((t) => t.state === "picked" && here.has(t.node));
}

const EXR = (name: string) => name.toLowerCase().endsWith(".exr");

/** 上传本次所需、仍在使用者本机上的素材。每份素材有两种方式，由服务器决定（只上传用到的通道）：
 *   - 该读取节点的状态中带有 `channels`（`{take, write}`：已连线端口对应的通道），且文件均为 EXR
 *     → 只上传这些通道的原始像素（`transfer/planes.ts sendPlanes`）；
 *   - 没有 `channels`（PNG / JPG 不区分通道，需要全部通道）→ 上传整个文件（`transfer/declare.ts sendUpload`）。
 * 网页不检查连线，也不解释端口名：端口与通道名的对应关系仅在服务器定义（`ReadSequence.upload_channels`）。
 * 无法解出通道时（解码器不支持该压缩方式或文件损坏）→ 给出提示（`N-UPLOAD-WHOLEFILE`）并改为上传整个文件，数据完整。 */
export async function sendPicked(ctx: BlockContext, cook: CookCase): Promise<boolean> {
  const mine = pickedFor(ctx, cook);
  if (!mine.length) return true;
  const ci = useCookInputs.getState();
  const plans = mine.map((t) => {
    const channels = ctx.reply.nodes[t.node]?.channels;
    const ref = String(ci.nodes[t.node]?.params[t.param] ?? "");
    return { t, ref, channels: channels && ref.startsWith("upload:") && t.files.every((f) => EXR(f.name)) ? channels : undefined };
  });
  const whole = plans.filter((p) => !p.channels);
  const partial = plans.filter((p) => p.channels);
  if (whole.length)
    say(msg("N-UPLOAD-BEFORECOOK", { files: whole.reduce((n, p) => n + p.t.files.length, 0), size: sizeText(whole.reduce((n, p) => n + p.t.bytes, 0)) }));
  if (partial.length)
    say(msg("N-UPLOAD-CHANNELS", { files: partial.reduce((n, p) => n + p.t.files.length, 0),
                                    channels: [...new Set(partial.flatMap((p) => p.channels!.write))].join(" "),
                                    whole: sizeText(partial.reduce((n, p) => n + p.t.bytes, 0)) }));
  for (const p of plans) {
    let ok: boolean;
    if (p.channels) {
      const got = await sendPlanes(p.t.key, p.ref, p.channels);
      if ("whole" in got) {
        say(msg("N-UPLOAD-WHOLEFILE", { file: got.whole.file, reason: got.whole.reason }));
        ok = await sendUpload(p.t.key);
      } else {
        ok = got.ok;
        if (got.ok) say(msg("I-UPLOAD-CHANNELSSENT", { file: p.t.name, channels: p.channels.write.join(" "), sent: sizeText(got.sent), whole: sizeText(p.t.bytes) }));
      }
    } else ok = await sendUpload(p.t.key);
    if (!ok) {
      const now = useUploads.getState().tasks[p.t.key];
      say(now ? uploadBlocker(now, p.t.node) : msg("E-UPLOAD-NOANSWER", { status: 0 }));
      return false;
    }
  }
  return true;
}
