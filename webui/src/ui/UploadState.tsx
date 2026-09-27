import "./upload.css";
import { useEffect, useState } from "react";
import { eta, howFar, retryUpload, type UploadTask } from "../transfer/uploads";
import { sizeText } from "../platform/format";
import { rateText } from "../platform/format";
import { Button } from "./Button";

/** An upload on its file parameter, in three short lines (the panel is narrow; nothing is cut): the name and how far
 * (帧 or %), a bar, and what is happening — the speed and the time left, 「断网了」 and when it tries again, 「没传完」
 * after a reload (pick the same files again), or that the server refused it, with 重试. The tooltip says the rest. */
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
  const tip = {
    // 选完文件一个字节都不传：先在本机把每个文件的内容指纹算出来，报给服务器，节点上的口就长出来了；
    // 字节要到点「计算」才传
    reading: `正在读你机器上的 ${task.files.length} 个文件，算它们的内容指纹（sha256）。\n**还没有任何字节上传**：算完就知道节点上有哪几个输出口，字节要到点「计算」时才传`,
    picked: `${task.name} 还在你的机器上，**一个字节都没上传**（${sizeText(task.bytes)}）。\n点「计算」的时候会先把它传上去，传完自动接着算——不用在这儿等`,
    sending: `正在上传 ${task.name}：已传 ${sizeText(task.sent)} / ${sizeText(task.bytes)}。服务器上已经有的文件不再传；网络断了会自己接着传，不用管`,
    waiting: `连不上服务器（${task.error}）。已经传上去的部分都在服务器上，网络一恢复就从断开的地方接着传，不用重新选`,
    finishing: "所有文件都传上去了，服务器正在把它们整理成一份输入",
    paused: task.sent
      ? `页面刷新过，浏览器不再让网页读这些文件。已传 ${howFar(task)}（${sizeText(task.sent)} / ${sizeText(task.bytes)}）：点左边的按钮再选一次同样的文件，从断开的地方接着传`
      // 选完文件本来就一个字节都不传（先申报、后传字节）：那时候刷新，说「没传完」是假话
      : `这份素材还没上传过（${sizeText(task.bytes)}），而页面刷新过，浏览器不再让网页读它：点左边的按钮再选一次同样的文件。\n节点上的输出口还在——申报那一份记在服务器上，刷新不会丢`,
    elsewhere: "这份文件正在这个浏览器的另一个标签页里上传：传完以后这里也会用上",
    failed: `服务器没有收下：${task.error}\n已经传上去的部分还在服务器上：点「重试」或者再选一次，都从断开的地方接着传`,
  }[task.state];
  return (
    <>
      <span className="fp-line">
        <span className="fp-name" data-user-data data-tip={task.name}>{task.name}</span>
        <span className="fp-meta fp-far tnum">{howFar(task)}</span>
      </span>
      <span className={`fp-bar fp-${task.state}`}>
        <i style={{ width: `${pct}%` }} />
      </span>
      <span className={`fp-line fp-up fp-${task.state}`} data-tip={tip}>
        <span className="fp-meta tnum" data-user-data data-tip={line}>{line}</span>
        {task.state === "failed" && (
          <Button tip="从断开的地方接着传" tone="ghost" layout="fp-retry" onClick={() => retryUpload(task.key)}>
            重试
          </Button>
        )}
      </span>
    </>
  );
}
