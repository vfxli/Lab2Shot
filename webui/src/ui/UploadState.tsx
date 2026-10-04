import "./upload.css";
import { useEffect, useState } from "react";
import { howFar, retryUpload, type UploadTask } from "../transfer/uploads";
import { rateText } from "../platform/format";
import { Button } from "./Button";
import { t } from "../i18n/t";

/** An upload on its file parameter, in three short lines (the panel is narrow; nothing is cut): the name and how far
 * (帧 or %), a bar, and what is happening — the speed (no estimate of the time left), 「断网了」 and when it tries again, 「没传完」
 * after a reload (pick the same files again), or that the server refused it, with 重试. It is drawn only in the
 * parameter panel, which shows no tips (platform/tips.ts). */
export function UploadState({ task }: { task: UploadTask }) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (task.state !== "waiting") return;
    const timer = window.setInterval(() => tick((n) => n + 1), 1000);
    return () => window.clearInterval(timer);
  }, [task.state]);
  const pct = Math.floor((task.sent / Math.max(task.bytes, 1)) * 100);
  const wait = task.state === "waiting" ? Math.max(0, Math.ceil((task.retryAt - Date.now()) / 1000)) : 0;
  const line = {
    reading: () => t("ui.upload.state_reading", { done: task.done, count: task.files.length }),
    picked: () => t("ui.upload.state_picked"),
    sending: () => (task.rate ? rateText(task.rate) : t("ui.upload.state_starting")),
    waiting: () => (wait ? t("ui.upload.state_waiting_in", { seconds: wait }) : t("ui.upload.state_waiting")),
    finishing: () => t("ui.upload.line_finishing"),
    paused: () => (task.sent ? t("ui.upload.state_paused") : t("ui.upload.state_notsent")),
    elsewhere: () => t("ui.upload.state_elsewhere"),
    failed: () => t("ui.upload.state_failed"),
  }[task.state]();
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
            {t("ui.common.retry")}
          </Button>
        )}
      </span>
    </>
  );
}
