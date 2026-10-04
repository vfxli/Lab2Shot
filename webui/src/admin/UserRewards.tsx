import { useEffect, useState } from "react";
import { adminApi, type UserRewards as Rewards } from "../api/admin";
import { stampText } from "../platform/format";
// the ledger folds like a feedback's diagnostics (ui/feedback.css fb-group)
import "../ui/feedback.css";
import { t as tr } from "../i18n/t";

/** 用户 / 一个账号: its 有效反馈奖励 (lab2shot/site/feedback.py rewards_of) — how many of its feedback are rated valid,
 * the days its grants still standing added, how many wait for the next grant, and the ledger of every grant and
 * revoke. Shown when the server says this login may read it (`account.rewards`). */
export function UserRewards({ user }: { user: number }) {
  const [data, setData] = useState<Rewards | null>(null);
  useEffect(() => {
    let alive = true;
    adminApi.userRewards(user).then(
      (d) => alive && setData(d),
      () => alive && setData(null),
    );
    return () => {
      alive = false;
    };
  }, [user]);
  if (!data) return null;
  const day = (t: number | null) => (t ? stampText(t).slice(0, 10) : tr("ui.admin.rewards.never_expires"));
  return (
    <div className="usr-rewards">
      <div className="adm-tiles">
        <div className="adm-tile">
          <span className="adm-tile-label">{tr("ui.admin.rewards.valid")}</span>
          <b className="tnum">{tr("ui.admin.rewards.items", { n: data.valid })}</b>
        </div>
        <div className="adm-tile">
          <span className="adm-tile-label">{tr("ui.admin.rewards.total")}</span>
          <b className="tnum">{data.recorded && !data.days ? tr("ui.admin.rewards.recorded_days", { n: data.recorded }) : tr("ui.admin.rewards.days", { n: data.days })}</b>
        </div>
        <div className="adm-tile">
          <span className="adm-tile-label">{tr("ui.admin.rewards.next")}</span>
          <b className="tnum">{data.per_days ? tr("ui.admin.rewards.to_go", { n: Math.max(data.per - data.pending, 0) }) : tr("ui.admin.rewards.none")}</b>
        </div>
      </div>
      {data.ledger.length > 0 && (
        <details className="fb-group">
          <summary>
            {tr("ui.admin.rewards.ledger")}<span className="fb-count">{tr("ui.admin.rewards.entries", { n: data.ledger.length })}</span>
          </summary>
          <div className="q-table-wrap">
            <table className="q-table">
              <thead>
                <tr>
                  <th>{tr("ui.admin.rewards.col_time")}</th>
                  <th>{tr("ui.admin.rewards.col_action")}</th>
                  <th>{tr("ui.admin.rewards.col_feedback")}</th>
                  <th>{tr("ui.admin.rewards.col_days")}</th>
                  <th>{tr("ui.admin.rewards.col_old")}</th>
                  <th>{tr("ui.admin.rewards.col_new")}</th>
                  <th>{tr("ui.admin.rewards.col_by")}</th>
                </tr>
              </thead>
              <tbody>
                {data.ledger.map((g) => (
                  <tr key={g.id} className={g.undone ? "dim" : undefined}>
                    <td className="q-time tnum">{stampText(g.at)}</td>
                    <td>
                      {g.kind_label}
                      {g.undone ? tr("ui.admin.rewards.undone") : ""}
                      {g.applied ? "" : tr("ui.admin.rewards.recorded_only")}
                    </td>
                    <td className="mono">{g.feedback.join(tr("list.sep"))}</td>
                    <td className="tnum">{g.kind === "revoke" ? `-${g.days}` : `+${g.days}`}</td>
                    <td className="tnum">{day(g.old)}</td>
                    <td className="tnum">{day(g.new)}</td>
                    <td>{g.by || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}
