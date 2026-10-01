import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { clearLog, entryTime, levelText, logText, markSeen, useLog } from "../state/log";
import { useCookInputs } from "../state/cookInputs";
import { useViewer } from "../state/viewer";
import { copyText } from "../platform/util";
import { Sheet } from "../ui/Sheet";
import { msg, type Message } from "../messages/message";
import { Button } from "../ui/Button";

/** 页面的日志窗口：发生过什么，可复制给协助排查的人，或发送到服务器日志。
 *
 * 这是「说过的话」唯一的落点（没有单独的消息栏），所以每行都画出消息编号：用户转述时带上它，开发者按它搜索。编号存在 `state/log.ts` 的 `LogEntry.code` 里、「复制全部」也带着它；
 * 画出来用的是全站那个编号的样子（`ui/message.css` 的 `.msg-code`），不新造一种。 */
export function LogSheet() {
  const open = useViewer((s) => s.logOpen);
  const setOpen = useViewer((s) => s.setLogOpen);
  const meta = useCookInputs((s) => s.meta);
  const file = useViewer((s) => s.file);
  const entries = useLog();
  const list = useRef<HTMLDivElement>(null);
  const [note, setNote] = useState<Message | null>(null);

  useEffect(() => {
    if (!open) return;
    markSeen();
    list.current?.scrollTo({ top: list.current.scrollHeight });
  }, [open, entries]);

  if (!open) return null;
  const text = () => logText(`${meta.name}（${file?.name ?? "未保存"}）`);

  const copy = async () => {
    await copyText(text());
    setNote(msg("I-LOG-COPIED"));
  };

  const send = async () => {
    const sent = await api.sendLog(text()).then(() => true, () => false);
    setNote(msg(sent ? "I-LOG-SENT" : "E-LOG-NOTSENT"));
  };

  return (
    <Sheet title="日志" width={900} solid onClose={() => setOpen(false)}>
      <div className="log-list" ref={list}>
        {entries.length ? (
          entries.map((e, i) => (
            <div key={i} className={`log-row ${e.level}`} data-code={e.code}>
              <span className="log-time tnum">{entryTime(e)}</span>
              <span className="log-level">{levelText(e.level)}</span>
              <span className="log-text">
                {e.text}
                {e.code && <span className="msg-code">{e.code}</span>}
              </span>
            </div>
          ))
        ) : (
          <div className="q-empty">还没有记录：计算、保存和出错都会写到这里</div>
        )}
      </div>
      <div className="dialog-row log-actions">
        <span className="tpl-desc" data-code={note?.code}>{note?.text || "页面上出现过的提示、每次计算的经过和页面自己的错误都记在这里，刷新页面也还在"}</span>
        <Button tip="清空这个浏览器里记下的页面日志（服务器上的日志不动）" tone="ghost" onClick={() => (clearLog(), setNote(msg("I-LOG-CLEARED")))}>
          清空
        </Button>
        <Button tip="写进服务器的日志，附上当前账号" onClick={() => void send()}>
          发给管理员
        </Button>
        <Button tip="把全部记录复制到剪贴板，可以贴给管理员" tone="primary" onClick={() => void copy()}>
          复制全部
        </Button>
      </div>
    </Sheet>
  );
}
