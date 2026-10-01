// 上传任务的文字表述：进度、状态行与节点注记，以及阻止计算的原因（不估计剩余时间：按瞬时速度推算的时间不准）。

import type { UploadTask } from "../state/uploads";
import { msg, type Message } from "../messages/message";
import { rateText } from "../platform/format";

export const item = (t: UploadTask) => t.name;

// ------------------------------------------------------------------ 文字表述

/** 进度：序列按帧数计，单个文件按已传比例计。 */
export const howFar = (t: UploadTask) => (t.sequence ? `${t.done}/${t.files.length} 帧` : `${Math.floor((t.sent / Math.max(t.bytes, 1)) * 100)}%`);

/** 任务的单行描述，用于参数栏（如「上传中 37/124 帧 · 0.24 MB/s」）。 */
export function uploadLine(t: UploadTask): string {
  switch (t.state) {
    case "reading":
      // 读取本机文件需要时间（9 GB 的序列需数十秒），必须给出说明
      return `读取本机文件 ${t.done}/${t.files.length} · 算内容指纹，还没有任何字节上传`;
    case "picked":
      return "素材在你机器上，还没上传 · 点「计算」时自动传上去";
    case "sending":
      return ["上传中", howFar(t), t.rate ? rateText(t.rate) : ""].filter(Boolean).join(" · ").replace("上传中 · ", "上传中 ");
    case "waiting":
      return `网络断了，恢复后自动继续 · 已传 ${howFar(t)}`;
    case "finishing":
      return "传完了，服务器在整理";
    case "paused":
      // 尚未传输任何字节的情况须单独表述：选择文件后不会立即上传（先申报、后传字节），
      // 此时刷新页面，提示「从中断处继续」并不准确，因为上传尚未开始
      return t.sent
        ? `上传没传完 · 已传 ${howFar(t)} · 再选一次同样的文件，从断开的地方接着传`
        : "素材还没上传 · 刷新之后浏览器不再让网页读它：再选一次同样的文件";
    case "elsewhere":
      return `另一个标签页在上传 · ${howFar(t)}`;
    case "failed":
      return `没传完：${t.error}`;
  }
}

/** 同一信息的简短形式，用于节点主体。 */
export function uploadNote(t: UploadTask): string {
  return { reading: `读取本机文件 ${t.done}/${t.files.length}`, picked: "待上传", sending: `上传中 ${howFar(t)}`,
           waiting: `等网络 · ${howFar(t)}`, finishing: "上传整理中", paused: t.sent ? "上传没传完" : "要再选一次",
           elsewhere: `上传中 ${howFar(t)}`, failed: "上传失败" }[t.state];
}

/** 素材仍在本机不构成拦截：点击「计算」时先上传，完成后自动继续计算（`graph/apply.ts sendPicked`），
 * 因此 `picked` / `reading` 不视为 blocker。页面刷新后浏览器已失去文件访问权的情况为 `paused`，仍须拦截
 * （使用者需重新选择同一文件）。 */
export const uploadStopsClick = (t: UploadTask): boolean => t.state !== "picked" && t.state !== "reading";

/** 文件上传期间节点无法计算的原因（B-UPLOAD-*）；`node`：节点名。 */
export const uploadBlocker = (t: UploadTask, node: string): Message =>
  t.state === "paused"
    ? msg(t.sent ? "B-UPLOAD-PAUSED" : "B-UPLOAD-NOTSENT", { node })
    : t.state === "failed"
      ? msg("B-UPLOAD-FAILED", { node, reason: t.error })
      // 文件已全部发送、服务器仍在整理的阶段：若提示「还在上传（90/90 帧）」，会与同屏文件参数行的
      // 「传完了，服务器在整理」（ui/UploadState.tsx 的 finishing）相矛盾，且看似停滞。
      : t.state === "finishing"
        ? msg("B-UPLOAD-FINISHING", { node })
        : msg("B-UPLOAD-GOING", { node, progress: howFar(t) });
