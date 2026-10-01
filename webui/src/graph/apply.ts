/** 上传与节点图的衔接：上传的引用写进节点参数、提交前把本次计算所需、仍在使用者本机上的素材传上去（sendPicked），
 * 以及「这次提交已被取消」的标记。上传本身在 transfer/，提交在 graph/actions.ts。 */

import { pickFile } from "./edit";

// 顶栏「取消」了这次提交（graph/actions.ts cancelSubmit）：上传停下之后、真正提交之前的每一步见到它就不再往下走；
// cook / deliverAll 占闩时清掉。放在这里（上传这一侧）：上传的循环最常问它，actions 也从这里读写
let abandoned = false;
/** 这次提交已被顶栏取消：提交路上每一步据此收手，不再另说什么。 */
export const submitAbandoned = (): boolean => abandoned;
export function abandonSubmit(on: boolean): void {
  abandoned = on;
}
import type { CookCase } from "../api/status";
import { useCookInputs } from "../state/cookInputs";
import { msg, say } from "../state/say";
import { uploadedLocal, uploadedShown } from "../transfer/local";
import { sendPlanes } from "../transfer/planes";
import type { BlockContext } from "./nodes";
import { followGraphId, onDeclared, onUploaded, restoreUploads, sendUpload, uploadBlocker, useUploads } from "../transfer/uploads";
import { sizeText } from "../platform/format";

/** 上传与节点图：传完的上传在那张图正开着时（本标签页，或同一张图的另一个标签页）写进节点的参数；刷新打断的上传
 * 以暂停状态回到各自的参数上。 */
export function startUploads(): void {
  // 申报完成后即将引用写入参数：此时尚未传输任何字节，但该引用已是最终引用
  // （id 仅由文件名与内容计算，`lab2shot/transfer/uploads.py set_id`）。参数写入后，服务器即可按申报中的
  // 图层生成输出口（`nodes/core/input.py made_ports` 的 `_declared` 回退），使用者随后才能连线。
  onDeclared((task, ref) => {
    const ci = useCookInputs.getState();
    if (ci.graphId !== task.graphId || !ci.nodes[task.node]) {
      say(msg("N-UPLOAD-GRAPHGONE", { file: task.name, graph: task.graph }));
      return;
    }
    if (ci.nodes[task.node].params[task.param] !== ref)
      pickFile(task.node, task.param, { ref, stem: "", files: task.files.length, bytes: task.bytes,
                                        first: task.frames[0] ?? null, last: task.frames.at(-1) ?? null, folder: task.folder });
  });
  // 按文档（graphId）判定：同一个模板打开两次，图名相同而 graphId 不同，旧的那份的上传不得写进新文档
  onUploaded((task, up, elsewhere) => {
    if (!task) return;
    const ci = useCookInputs.getState();
    const node = ci.nodes[task.node];
    if (ci.graphId !== task.graphId || !node) {
      // 另一个标签页传完的：那个标签页自己会说；这里开着的是别的文档，不打扰
      if (!elsewhere) say(msg("N-UPLOAD-GRAPHGONE", { file: task.name, graph: task.graph }));
      return;
    }
    const { name: _, ...picked } = up;
    uploadedLocal(task.key, task.graphId, task.node, task.param, up.ref);
    if (node.params[task.param] !== up.ref) pickFile(task.node, task.param, picked);
    uploadedShown(task.graphId, task.node, task.param); // 参数已是这个地址：之后再变空是使用者清的
  });
  followGraphId(); // 「另存为」换了 graphId：任务跟着文档走（transfer/uploads.ts）
  void restoreUploads();
}

/** 本次实际计算的节点中，哪些素材仍在使用者本机上（`picked`：已申报、字节未传输）。
 *
 * 本次计算涉及的节点由服务器给出（状态回复的 `policy.computes` 与 `deliver.computes`，
 * `lab2shot/engine/evaluation.py _case`）：结果已缓存的读取节点不在其中，其素材无须传输任何字节。
 * 网页不按连线自行反推，否则会产生第二份答案。 */
export function pickedFor(_ctx: BlockContext, cook: CookCase) {
  const here = new Set(cook.computes);
  const graph = useCookInputs.getState().graphId; // 只算这份文档的：另一份文档里同 id 的读取节点的素材不上传
  return Object.values(useUploads.getState().tasks).filter((t) => t.state === "picked" && t.graphId === graph && here.has(t.node));
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
    if (submitAbandoned()) return false; // 顶栏取消了这次提交：后面的素材不再开始传
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
      if (submitAbandoned()) return false; // 顶栏取消了这次提交：是使用者停下的，不当成上传出了问题再说一遍
      const now = useUploads.getState().tasks[p.t.key];
      say(now ? uploadBlocker(now, p.t.node) : msg("E-UPLOAD-NOANSWER", { status: 0 }));
      return false;
    }
  }
  return true;
}
