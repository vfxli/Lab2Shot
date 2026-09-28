import "./upload.css";
import { useEffect, useState } from "react";
import { eta, howFar, retryUpload, type UploadTask } from "../transfer/uploads";
import { rateText } from "../platform/format";
import { Button } from "./Button";

/** An upload on its file parameter, in three short lines (the panel is narrow; nothing is cut): the name and how far
 * (帧 or %), a bar, and what is happening — the speed and the time left, 「断网了」 and when it tries again, 「没传完」
 * after a reload (pick the same files again), or that the server refused it, with 重试. It is drawn only in the
 * parameter panel, which shows no tips (platform/tips.ts). */
export function UploadState({ task }: { task: UploadTask }) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (task.state !== "waiting") return;
    const t = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(t);
  }, [task.state]);
  const pct = Math.floor((task.sent / Math.max(task.bytes, 1)) * 100);
  const wait = task.state === "waiting" ? Math.max(0, Math.ceil((task.retryAt - Date.now()) / 1000)) : 0;
  const line = {
    reading: `${task.done}/${task.files.length} 个文件`,
    picked: "点「计算」时上传",
    sending: task.rate ? [rateText(task.rate), eta(task)].filter(Boolean).join(" · ") : "开始上传",
    waiting: wait ? `断网了，${wait} 秒后重连` : "断网了，正在重连",
    finishing: "传完了，服务器在整理",
    paused: task.sent ? "没传完，重选即续传" : "还没传，重选一次",
    elsewhere: "另一个标签页在传",
    failed: "服务器没收下",
  }[task.state];
  return (
    <>
      <span className="fp-line">
        <span className="fp-name" data-user-data>{task.name}</span>
        <span className="fp-meta fp-far tnum">{howFar(task)}</span>
      </span>
      <span className={`fp-bar fp-${task.state}`}>
        <i style={{ width: `${pct}%` }} />
      </span>
      <span className={`fp-line fp-up fp-${task.state}`}>
        <span className="fp-meta tnum" data-user-data>{line}</span>
        {task.state === "failed" && (
          <Button tone="ghost" layout="fp-retry" onClick={() => retryUpload(task.key)}>
            重试
          </Button>
        )}
      </span>
    </>
  );
}
