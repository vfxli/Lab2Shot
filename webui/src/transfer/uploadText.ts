// 上传任务的文字表述：进度、状态行与节点注记，以及阻止计算的原因（不估计剩余时间：按瞬时速度推算的时间不准）。

import type { UploadTask } from "../state/uploads";
import { msg, type Message } from "../messages/message";
import type { Said } from "../messages/format";
import { rateText } from "../platform/format";
import { t as tr } from "../i18n/t";

export const item = (t: UploadTask) => t.name;

// ------------------------------------------------------------------ 文字表述

/** 进度：序列按帧数计，单个文件按已传比例计。 */
export const howFar = (t: UploadTask) => (t.sequence ? tr("ui.upload.frames_done", { done: t.done, count: t.files.length }) : `${Math.floor((t.sent / Math.max(t.bytes, 1)) * 100)}%`);

/** 任务的单行描述，用于参数栏（如「上传中 37/124 帧 · 0.24 MB/s」）。 */
export function uploadLine(t: UploadTask): string {
  switch (t.state) {
    case "reading":
      // 读取本机文件需要时间（9 GB 的序列需数十秒），必须给出说明
      return tr("ui.upload.line_reading", { done: t.done, count: t.files.length });
    case "picked":
      return tr("ui.upload.line_picked");
    case "sending":
      return t.rate ? tr("ui.upload.line_sending_rate", { far: howFar(t), rate: rateText(t.rate) }) : tr("ui.upload.line_sending", { far: howFar(t) });
    case "waiting":
      return tr("ui.upload.line_waiting", { far: howFar(t) });
    case "finishing":
      return tr("ui.upload.line_finishing");
    case "paused":
      // 尚未传输任何字节的情况须单独表述：选择文件后不会立即上传（先申报、后传字节），
      // 此时刷新页面，提示「从中断处继续」并不准确，因为上传尚未开始
      return t.sent ? tr("ui.upload.line_paused", { far: howFar(t) }) : tr("ui.upload.line_notsent");
    case "elsewhere":
      return tr("ui.upload.line_elsewhere", { far: howFar(t) });
    case "failed":
      return tr("ui.upload.line_failed", { error: t.error });
  }
}

/** 同一信息的简短形式，用于节点主体。 */
export function uploadNote(t: UploadTask): string {
  switch (t.state) {
    case "reading": return tr("ui.upload.note_reading", { done: t.done, count: t.files.length });
    case "picked": return tr("ui.upload.note_picked");
    case "sending": return tr("ui.upload.note_sending", { far: howFar(t) });
    case "waiting": return tr("ui.upload.note_waiting", { far: howFar(t) });
    case "finishing": return tr("ui.upload.note_finishing");
    case "paused": return t.sent ? tr("ui.upload.note_paused") : tr("ui.upload.note_repick");
    case "elsewhere": return tr("ui.upload.note_sending", { far: howFar(t) });
    case "failed": return tr("ui.upload.note_failed");
  }
}

/** 素材仍在本机不构成拦截：点击「计算」时先上传，完成后自动继续计算（`graph/apply.ts sendPicked`），
 * 因此 `picked` / `reading` 不视为 blocker。页面刷新后浏览器已失去文件访问权的情况为 `paused`，仍须拦截
 * （使用者需重新选择同一文件）。 */
export const uploadStopsClick = (t: UploadTask): boolean => t.state !== "picked" && t.state !== "reading";

/** 文件上传期间节点无法计算的原因（B-UPLOAD-*）；`node`：节点名。 */
export const uploadBlocker = (t: UploadTask, node: Said | string): Message =>
  t.state === "paused"
    ? msg(t.sent ? "B-UPLOAD-PAUSED" : "B-UPLOAD-NOTSENT", { node })
    : t.state === "failed"
      ? msg("B-UPLOAD-FAILED", { node, reason: t.error })
      // 文件已全部发送、服务器仍在整理的阶段：若提示「还在上传（90/90 帧）」，会与同屏文件参数行的
      // 「传完了，服务器在整理」（ui/UploadState.tsx 的 finishing）相矛盾，且看似停滞。
      : t.state === "finishing"
        ? msg("B-UPLOAD-FINISHING", { node })
        : msg("B-UPLOAD-GOING", { node, progress: howFar(t) });
