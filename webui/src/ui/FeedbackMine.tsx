/** 我的反馈: the account's own feedback with the administrator's replies, the new ones marked. */

import { useEffect, useState } from "react";
import { stampText } from "../platform/format";
import { Button } from "./Button";
import { useMine } from "./Feedback";
import { Loading } from "./Loading";
import { t } from "../i18n/t";

export function MyFeedbackList({ onWrite }: { onWrite: () => void }) {
  const items = useMine((s) => s.items);
  const load = useMine((s) => s.load);
  const read = useMine((s) => s.read);
  const problem = useMine((s) => s.problem);
  const rule = useMine((s) => s.rule);
  const [fresh, setFresh] = useState<Set<string> | null>(null); // unread when the list opened: marked while it shows
  const [open, setOpen] = useState<Set<string>>(new Set());

  useEffect(() => {
    load().then(
      () => {
        setFresh(new Set((useMine.getState().items ?? []).filter((f) => f.unread).map((f) => f.id)));
        void read();
      },
      () => setFresh(new Set()), // said below; what was read before (if anything) still shows
    );
  }, [load, read]);

  if (items === null) return problem ? <p className="adm-empty">{t("ui.feedback.mine_failed", { reason: problem })}</p> : <Loading what={t("ui.feedback.mine")} />;
  if (fresh === null) return <Loading what={t("ui.feedback.mine")} />;
  if (!items.length)
    return (
      <div className="fb-mine-empty">
        <p className="adm-empty">{t("ui.feedback.mine_empty")}</p>
        {rule && <p className="tpl-desc">{t("ui.feedback.rule", { rule })}</p>}
        <Button onClick={onWrite}>
          {t("ui.feedback.write")}
        </Button>
      </div>
    );
  return (
    <div className="fb-mine">
      <p className="tpl-desc">{t("ui.feedback.mine_about")}{rule && t("ui.feedback.rule", { rule })}</p>
      {items.map((f) => {
        const long = f.text.length > 160 || f.text.split("\n").length > 3;
        const whole = open.has(f.id);
        return (
          <article key={f.id} className={`fb-card${fresh.has(f.id) ? " fresh" : ""}`}>
            <div className="fb-meta">
              <span className="tnum">{stampText(f.at)}</span>
              {f.category_label && <span className="chip">{f.category_label}</span>}
              <span className={`chip fb-status ${f.status}`}>
                {f.status_label}
              </span>
              {f.images > 0 && <span>{t("ui.feedback.images", { count: f.images })}</span>}
              {fresh.has(f.id) && <span className="fb-new-mark">{f.reply && f.replied && f.replied >= (f.changed ?? 0) ? t("ui.feedback.new_reply") : f.reward ? t("ui.feedback.rewarded") : t("ui.feedback.status_changed")}</span>}
            </div>
            <div className={`fb-card-text${long && !whole ? " clipped" : ""}`}>{f.text}</div>
            {long && !whole && (
              <Button tone="ghost" size="sm" layout="fb-more" onClick={() => setOpen((s) => new Set(s).add(f.id))}>
                {t("ui.feedback.read_all")}
              </Button>
            )}
            {f.reward && <div className="fb-reward">{f.reward}</div>}
            {f.reply ? (
              <div className="fb-reply">
                <span className="fb-reply-head">{t("ui.feedback.reply")}{f.replied ? ` · ${stampText(f.replied)}` : ""}</span>
                {f.reply}
              </div>
            ) : (
              <div className="tpl-desc">{f.status === "new" ? t("ui.feedback.not_seen") : t("ui.feedback.no_reply")}</div>
            )}
          </article>
        );
      })}
    </div>
  );
}
