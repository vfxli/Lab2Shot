import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { clearLog, entryText, entryTime, keepAgain, levelText, logText, markSeen, useLog, wantedAgain } from "../state/log";
import { useCookInputs } from "../state/cookInputs";
import { useViewer } from "../state/viewer";
import { copyText } from "../platform/util";
import { Sheet } from "../ui/Sheet";
import { msg, type Message } from "../messages/message";
import { Button } from "../ui/Button";
import { pick, t, useT } from "../i18n/t";
import { useLang } from "../i18n/lang";
import { tipOf } from "../platform/tips";

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
  useT();
  const lang = useLang((s) => s.lang);

  useEffect(() => {
    if (!open) return;
    markSeen();
    list.current?.scrollTo({ top: list.current.scrollHeight });
  }, [open, entries]);

  // what the server said in another language, said again in this one (state/log.ts): asked once per message
  useEffect(() => {
    if (!open) return;
    const wanted = wantedAgain(entries, lang);
    if (!wanted.length) return;
    api.me.said(wanted).then((got) => keepAgain(wanted, got.texts, lang), () => undefined); // not said again: its words as they were
  }, [open, entries, lang]);

  if (!open) return null;
  const text = () => logText(t("ui.editor.log_graph", { name: pick(meta.name) || t("ui.common.unnamed"), file: file?.name ?? t("ui.editor.log_unsaved") }));

  const copy = async () => {
    await copyText(text());
    setNote(msg("I-LOG-COPIED"));
  };

  const send = async () => {
    const sent = await api.sendLog(text()).then(() => true, () => false);
    setNote(msg(sent ? "I-LOG-SENT" : "E-LOG-NOTSENT"));
  };

  return (
    <Sheet title={t("ui.editor.log")} width={900} solid onClose={() => setOpen(false)}>
      <div className="log-list" ref={list}>
        {entries.length ? (
          entries.map((e, i) => (
            <div key={i} className={`log-row ${e.level}`} data-code={e.code}>
              <span className="log-time tnum">{entryTime(e)}</span>
              <span className="log-level">{levelText(e.level)}</span>
              <span className="log-text">
                {entryText(e, lang)}
                {e.code && <span className="msg-code">{e.code}</span>}
              </span>
            </div>
          ))
        ) : (
          <div className="q-empty">{t("ui.editor.log_empty")}</div>
        )}
      </div>
      <div className="dialog-row log-actions">
        <span className="tpl-desc" data-code={note?.code}>{note?.text || t("ui.editor.log_about")}</span>
        <Button tip={tipOf("consequence", t("ui.editor.log_clear_tip"))} tone="ghost" onClick={() => (clearLog(), setNote(msg("I-LOG-CLEARED")))}>
          {t("ui.editor.log_clear")}
        </Button>
        <Button onClick={() => void send()}>
          {t("ui.editor.log_send")}
        </Button>
        <Button tone="primary" onClick={() => void copy()}>
          {t("ui.editor.log_copy")}
        </Button>
      </div>
    </Sheet>
  );
}
