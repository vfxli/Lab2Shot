/** 我的反馈: the account's own feedback with the administrator's replies, the new ones marked. */

import { useEffect, useState } from "react";
import { stampText } from "../platform/format";
import { Button } from "./Button";
import { useMine, type MyFeedback } from "./Feedback";
import { Loading } from "./Loading";

export function MyFeedbackList({ onWrite }: { onWrite: () => void }) {
  const items = useMine((s) => s.items);
  const load = useMine((s) => s.load);
  const read = useMine((s) => s.read);
  const [fresh, setFresh] = useState<Set<string> | null>(null); // unread when the list opened: marked while it shows
  const [open, setOpen] = useState<Set<string>>(new Set());

  useEffect(() => {
    void load().then(() => {
      setFresh(new Set((useMine.getState().items ?? []).filter((f) => f.unread).map((f) => f.id)));
      void read();
    });
  }, [load, read]);

  if (items === null || fresh === null) return <Loading what="我的反馈" />;
  if (!items.length)
    return (
      <div className="fb-mine-empty">
        <p className="adm-empty">还没有提交过反馈：提交以后，这里列出提过的反馈和管理员的回复</p>
        <Button tip="写一条新的反馈" onClick={onWrite}>
          写反馈
        </Button>
      </div>
    );
  return (
    <div className="fb-mine">
      <p className="tpl-desc">当前账号提交过的反馈。管理员回复或者改了状态，「提交反馈」按钮上会出现一个点。</p>
      {items.map((f) => {
        const long = f.text.length > 160 || f.text.split("\n").length > 3;
        const whole = open.has(f.id);
        return (
          <article key={f.id} className={`fb-card${fresh.has(f.id) ? " fresh" : ""}`}>
            <div className="fb-meta">
              <span className="tnum">{stampText(f.at * 1000 / 1000)}</span>
              {f.category_label && <span className="chip">{f.category_label}</span>}
              <span className={`chip fb-status ${f.status}`} data-tip={STATUS_TIP[f.status]}>
                {f.status_label}
              </span>
              {f.images > 0 && <span>{f.images} 张图</span>}
              {fresh.has(f.id) && <span className="fb-new-mark">{f.reply && f.replied && f.replied >= (f.changed ?? 0) ? "新回复" : "状态有变化"}</span>}
            </div>
            <div className={`fb-card-text${long && !whole ? " clipped" : ""}`}>{f.text}</div>
            {long && !whole && (
              <Button tip="显示这条反馈的全文" tone="ghost" size="sm" layout="fb-more" onClick={() => setOpen((s) => new Set(s).add(f.id))}>
                展开全文
              </Button>
            )}
            {f.reply ? (
              <div className="fb-reply">
                <span className="fb-reply-head">管理员回复{f.replied ? ` · ${stampText(f.replied * 1000 / 1000)}` : ""}</span>
                {f.reply}
              </div>
            ) : (
              <div className="tpl-desc">{f.status === "new" ? "管理员还没看到" : "管理员还没有回复"}</div>
            )}
          </article>
        );
      })}
    </div>
  );
}

const STATUS_TIP: Record<MyFeedback["status"], string> = {
  new: "新：管理员还没看到",
  seen: "已看：管理员看过了，还在处理",
  solved: "已解决：问题解决了，或者建议处理了",
};
