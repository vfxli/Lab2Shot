import { ApiError, json } from "../platform/http";
import { useUploads } from "../state/uploads";
import type { PlanesAnswer } from "./exrWorker";
import { exrPlanes } from "./exr";
import { completeSet, lineWait, patchTask, sendBytes, sending, sentHere } from "./uploads";
import { t } from "../i18n/t";

/** 通道级上传：点击「计算」时，一份 EXR 素材只上传已连线通道的原始像素，而非整个文件；
 * N 个通道未全部使用时，下游计算只上传所用的通道。
 *
 * 每份素材是否走此路径由服务器决定：状态回复中该读取节点带有 `channels`（`{take, write}`，
 * `lab2shot/nodes/core/input.py ReadSequence.upload_channels`）。端口到通道名的对应关系只在该处定义，本模块不作任何解释，
 * 照单执行；没有该项时上传整个文件（PNG / JPG、需要全部通道的情况，分支位于 `graph/apply.ts sendPicked`）。
 *
 * 单帧流程：
 *   1. 向服务器查询该帧（按原文件的 sha256）缺少哪些通道（`POST /api/uploads/planes/have`），已上传的通道不重传；
 *   2. 在 worker 中使用页面唯一的 EXR 解码器（`exr/decode.ts`）解出缺少通道的原始平面：half 保持 Uint16、
 *      float 保持 Float32、uint 保持 Uint32，不经过显示变换、不缩放；
 *   3. 平面按顺序首尾相接后 gzip（浏览器自带的 `CompressionStream`，无损），按现有分段协议作为一个 blob 上传
 *      （`uploads.ts sendBytes`：断线后续传）；
 *   4. `POST /api/uploads/planes`：服务器还原平面，使用唯一的 EXR 写入器写成只含这些通道的 EXR，
 *      与原文件已有的通道合并，按内容寻址存储（`lab2shot/transfer/uploads.py add_planes`）。
 * 所有帧到达后照常调用 `POST /api/uploads` 组装该输入（`uploads.ts completeSet`）；参数中的引用与整份上传
 * 相同（清单记录的是原文件的 sha，`set_id` 据此计算），因此下游无需任何修改。
 *
 * 无法解码时改为整份上传并给出提示（`N-UPLOAD-WHOLEFILE`）：解码器不支持该压缩方式、文件损坏、所需通道不在文件中，
 * 均不得静默降级。某一帧无法解码时，整个任务改为整份上传（`sendWith`），字节完整。 */

const AT_ONCE = 2; // 同时解码的帧数（解码在 worker 中进行，单帧平面达数十 MB，避免同时占用过多内存）
const BYTES = { half: 2, float: 4, uint: 4 } as const;

class Undecodable extends Error {
  constructor(readonly file: string, reason: string) {
    super(reason);
  }
}

/** 对一段字节做 gzip（浏览器自带的无损压缩），返回 Blob。 */
async function gzip(bytes: Uint8Array): Promise<Blob> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Response(stream).blob();
}

/** 处理一帧：解出缺少的通道，组装为一个容器（平面按 `take` 的顺序首尾相接），然后 gzip。 */
async function planesOf(file: File, take: string[]): Promise<{ blob: Blob; decoded: PlanesAnswer; types: string[] }> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let decoded: PlanesAnswer;
  try {
    decoded = await exrPlanes(bytes, take);
  } catch (e) {
    throw new Undecodable(file.name, (e as Error).message);
  }
  const found = new Map(decoded.channels.map((c) => [c.name, c]));
  const planes = take.map((name) => {
    const p = found.get(name);
    if (!p) throw new Undecodable(file.name, t("ui.upload.no_channel", { name }));
    if (p.width !== decoded.width || p.height !== decoded.height) throw new Undecodable(file.name, t("ui.upload.channel_size", { name }));
    return p;
  });
  const total = planes.reduce((n, p) => n + p.width * p.height * BYTES[p.type], 0);
  const container = new Uint8Array(total);
  let at = 0;
  for (const p of planes) {
    const raw = new Uint8Array(p.data.buffer, p.data.byteOffset, p.data.byteLength);
    if (raw.byteLength !== p.width * p.height * BYTES[p.type]) throw new Undecodable(file.name, t("ui.upload.channel_bytes", { name: p.name }));
    container.set(raw, at);
    at += raw.byteLength;
  }
  return { blob: await gzip(container), decoded, types: planes.map((p) => p.type) };
}

/** 处理结果：上传成功（`sent` 为压缩后经网络传输的字节数）、失败（原因记录在任务上），或该数据无法解码须整份上传
 * （`whole`：哪一帧及原因；调用方给出提示并改为整份上传，见 `graph/apply.ts sendPicked`）。 */
type PlanesResult = { ok: true; sent: number } | { ok: false } | { whole: { file: string; reason: string } };

/** 该素材（一个任务：一段序列或一张图）只上传已连线的通道。`ref`：参数中的引用 `upload:<id>/…`（服务器按 id 查找申报的文件头）；
 * `channels`：服务器给出的通道清单。上传成功时该输入已组装完成（`completeSet`）。 */
export async function sendPlanes(key: string, ref: string, channels: { take: string[]; write: string[] }): Promise<PlanesResult> {
  const task = useUploads.getState().tasks[key];
  const here = sentHere.get(key);
  if (!task || !here) return { ok: false };
  const files = here.files;
  const shaOf = new Map(task.files.map((f) => [f.name, f.sha]));
  if (files.some((f) => !shaOf.get(f.name))) return { whole: { file: task.name, reason: t("ui.upload.no_digest") } };
  const sid = ref.slice("upload:".length).split("/")[0];
  let stopped = false; // 已取消（cancelUpload）
  let failed = false; // 有一帧失败：其余工作者随之停下
  const inflight = new Set<XMLHttpRequest>(); // 在途的分段：取消时立刻中止，不再把余下的发完
  const wakes = new Set<() => void>(); // 等线路恢复的（lineWait）：取消时立刻叫醒
  sentHere.set(key, { files, stop: () => ((stopped = true), inflight.forEach((x) => x.abort()), wakes.forEach((w) => w())) });
  patchTask(key, { state: "sending", error: "", sent: 0, done: 0 });
  const cancelled = () => stopped || !useUploads.getState().tasks[key];
  const isStopped = () => failed || cancelled();
  const sizes = new Map<string, number>(); // 每帧压缩后的大小：进度条据此计算，而非按整个文件计算
  let sent = 0;
  let done = 0;
  // 在途字节按帧分开记：两路（AT_ONCE）同时在发，各报各的，合计才是进度
  const flying = new Map<string, number>();
  // 与整份上传同一套簿记（uploads.ts sending）：心跳、每秒量速度与进度、广播给别的标签页、定期落盘
  const book = sending(key, () => {
    const going = sent + [...flying.values()].reduce((a, b) => a + b, 0);
    const known = [...sizes.values()];
    const avg = known.length ? known.reduce((a, b) => a + b, 0) / known.length : 0;
    return { sent: going, done, bytes: Math.max(1, Math.round(going + avg * (files.length - known.length))) };
  });
  const report = book.report;
  // 与服务器的一问一答（查缺哪些通道、登记一帧的平面）：断线、服务器忙、登录过期都等线路恢复后续问（同 completeSet），
  // 不一断就把整份任务判为失败；服务器明确拒绝的照样报错
  let attempt = 0;
  const ask = async <T,>(what: () => Promise<T>): Promise<T> => {
    for (;;) {
      try {
        const got = await what();
        attempt = 0;
        return got;
      } catch (e) {
        if (isStopped()) throw e;
        const status = e instanceof ApiError ? e.status : 0;
        if (status && status < 500 && status !== 401 && status !== 408 && status !== 429) throw e;
        book.reset();
        await lineWait(key, attempt++, (e as Error).message, isStopped, wakes);
        if (isStopped()) throw e;
      }
    }
  };
  try {
    const shas = files.map((f) => shaOf.get(f.name)!);
    const said = await ask(() => json<{ missing?: Record<string, string[]> }>("POST", "/api/uploads/planes/have", { shas, channels: channels.write }));
    const missing = said.missing ?? {}; // 服务器回复缺这一项时按「全都缺」
    const queue = [...files];
    const one = async (f: File) => {
      const sha = shaOf.get(f.name)!;
      const lacking = new Set(missing[sha] ?? channels.write);
      if (!lacking.size) {
        done++;
        return; // 该帧所需的通道服务器均已具备
      }
      const take = channels.take.filter((_, i) => lacking.has(channels.write[i]));
      const write = channels.write.filter((w) => lacking.has(w));
      const { blob, decoded, types } = await planesOf(f, take);
      sizes.set(f.name, blob.size);
      const got = await sendBytes(key, blob, (n) => void flying.set(f.name, n), isStopped, inflight);
      flying.delete(f.name);
      if (isStopped()) return;
      sent += blob.size;
      await ask(() => json("POST", "/api/uploads/planes", {
        sid, sha, blob: got, width: decoded.width, height: decoded.height, compression: decoded.compression,
        channels: take.map((t, i) => ({ take: t, write: write[i], type: types[i] })),
        display: decoded.displayWindow ?? null, data: decoded.dataWindow ?? null,
      }));
      if (isStopped()) return;
      done++;
      report();
    };
    const worker = async () => {
      try {
        for (let f = queue.shift(); f && !isStopped(); f = queue.shift()) await one(f);
      } catch (e) {
        failed = true; // 整个任务统一改走一条路（整份上传或判失败）：本轮的工作者都不再发送
        throw e;
      }
    };
    await Promise.all(Array.from({ length: Math.min(AT_ONCE, queue.length) }, worker));
    if (isStopped()) return { ok: false };
    await completeSet(task, files, Object.fromEntries(files.map((f) => [f.name, shaOf.get(f.name)!])), isStopped);
    return useUploads.getState().tasks[key] ? { ok: false } : { ok: true, sent }; // 上传完成时任务即被删除：任务仍存在表示未成功
  } catch (e) {
    if (cancelled()) return { ok: false };
    if (e instanceof Undecodable) {
      // 整份上传前将任务恢复为 `picked` 状态：字节数为整个文件的大小，进度从零开始（`sendUpload` 随后执行整份上传）
      patchTask(key, { state: "picked", error: "", sent: 0, done: 0, bytes: files.reduce((n, f) => n + f.size, 0) });
      sentHere.set(key, { files, stop: () => undefined });
      return { whole: { file: e.file, reason: e.message } };
    }
    patchTask(key, { state: "failed", error: (e as Error).message, rate: 0 });
    return { ok: false };
  } finally {
    book.end();
  }
}
