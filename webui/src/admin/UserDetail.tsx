import { useEffect, useState } from "react";
import { adminApi, type UserResourceTab, type UserRow } from "../api/admin";
import { Button, Segmented } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { Loading } from "../ui/Loading";
import { reasonOf } from "../messages/message";
import { ResourceTable } from "./Resources";
import { shown, usable, why } from "../api/applies";
import { LoginsSummary } from "./UserLogins";
import { UserQuota } from "./UserQuota";
import { loginText, ROLE_TIP } from "./Users";

/** 用户 / 一个账号: the account's own page — who it is and what may be done to it at the
 * top, then one tab per kind of resource it owns, with how many, and one shared table under whichever is open.
 *
 * The tabs come from the server's registry (GET /api/admin/users/{id}/resources, lab2shot/resources.py): this page
 * does not name a kind of resource anywhere, and shows only what this login may look at. What may be done to the
 * account comes from the same availability answer the list uses (`u.applies`), never from a role. */

export function UserDetail({ u, onBack, onEdit, onReset, onDelete }: {
  u: UserRow;
  onBack: () => void;
  onEdit: () => void;
  onReset: () => void;
  onDelete: () => void;
}) {
  const [tabs, setTabs] = useState<UserResourceTab[] | null>(null);
  const [problem, setProblem] = useState("");
  const [kind, setKind] = useState("");

  useEffect(() => {
    let live = true;
    adminApi.userResources(u.id).then(
      (r) => live && (setTabs(r.tabs), setKind((k) => (r.tabs.some((t) => t.kind === k) ? k : (r.tabs[0]?.kind ?? "")))),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [u.id]);

  return (
    <section className="adm-sec usr-detail">
      <nav className="usr-crumb" aria-label="回到用户列表">
        <Button tip="回到全部账号的列表" tone="link" onClick={onBack}>
          用户
        </Button>
        <span className="dim">/</span>
        <span data-user-data data-tip={`${u.name} · ${u.username}`}>{u.name || u.username}</span>
      </nav>
      <header className="usr-head">
        <span className="usr-face" aria-hidden>
          {[...(u.name || u.username)][0]}
        </span>
        <div className="usr-who">
          <h2 data-user-data data-tip={u.name}>{u.name || u.username}</h2>
          <div className="usr-meta dim">
            <span className="mono" data-user-data data-tip={u.username}>
              {u.username}
            </span>
            <span>{u.department || "没有部门"}</span>
            <span data-tip={ROLE_TIP}>{u.role_label}</span>
            <span data-tip={u.state_tip}>{u.state}</span>
            <span data-tip="最近一次输密码登录">{loginText(u.last_login)}</span>
          </div>
        </div>
        <div className="usr-acts">
          {usable(u.applies, "account.edit") && (
            <Button tip="改中文名、部门、到期时间这些" onClick={onEdit}>
              改
            </Button>
          )}
          {usable(u.applies, "account.password") && (
            <Button tip="忘了密码时设一个新的：他所有的登录马上退出，用新密码重新登录" tone="ghost" onClick={onReset}>
              重设密码
            </Button>
          )}
          {shown(u.applies, "account.delete") && (
            <Button tip={why(u.applies, "account.delete") || "删除这个账号（要再确认一次）"} tone="ghost" danger disabled={!usable(u.applies, "account.delete")} onClick={onDelete}>
              删除
            </Button>
          )}
        </div>
      </header>
      <UserQuota user={u.id} applies={u.applies} />
      {usable(u.applies, "account.logins") && <LoginsSummary user={u.id} />}
      {problem ? (
        <Empty title={problem} hint="刷新页面再试一次。" />
      ) : !tabs ? (
        <Loading what="这个账号名下的记录" />
      ) : !tabs.length ? (
        <Empty title="这个登录看不到这个账号名下的记录" hint="任务、上传和反馈要各自的权限。" />
      ) : (
        <>
          <Segmented
            label="这个账号名下的记录"
            tabs
            size="md"
            value={kind}
            options={tabs.map((t) => ({
              value: t.kind,
              label: (
                <>
                  {t.label}
                  <b className="usr-tab-count tnum">{t.count}</b>
                </>
              ),
              tip: `这个账号名下的${t.label}：${t.count} 条`,
              field: `tab-${t.kind}`,
            }))}
            onChange={setKind}
            layout="usr-tabs"
          />
          {kind && <ResourceTable key={kind} user={u.id} kind={kind} />}
        </>
      )}
    </section>
  );
}
