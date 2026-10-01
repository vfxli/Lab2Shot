import { useCallback, useEffect, useState } from "react";
import { copyText } from "../platform/util";
import { adminApi, type Presence as PresenceView, type UserChange, type UserRow, type UsersView } from "../api/admin";
import { TAGS_TIP } from "./userFields";
import { useSignedIn } from "../state/session";
import { shown, usable, why } from "../api/applies";
import { activeText, onlineMeaning, onlineTip, placeText, Section, useAdmin } from "./common";
import { fullTimeText, whenText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { DeleteDialog, EditDialog, NewUserDialog, PurgeDialog, ResetDialog } from "./UserDialogs";
import { trafficColumns } from "./traffic";
import { Empty } from "../ui/Empty";
import { Loading } from "../ui/Loading";
import { Table, type Column } from "../ui/Table";
import { UserDetail } from "./UserDetail";
import { Rights } from "./Rights";

import "./users.css";

/** 用户: the accounts (lab2shot/accounts.py, server/users.py). The administrator creates one for each user (a username,
 * an initial password to hand over, a Chinese name and department for the statistics, an expiry date, what they may use),
 * extends or disables it, resets a forgotten password, deletes it. Which rows are listed and what each row, the new-user
 * form and the edit form offer is the server's answer (server/available.py, read through applies.ts): a
 * 二级管理员 sees only 普通用户 and no tags, roles or deleting. Every rule the server checks is checked here first,
 * so a mistake shows before anything is sent. */

/** What a role lets one do, in the server's words (lab2shot/roles.py ROLES): the role `id`'s, or every role this login
 * sees, one per line. */
export const roleTip = (roles: UsersView["roles"], id?: string): string =>
  roles.filter((r) => id === undefined || r.id === id).map((r) => `${r.label}：${r.tip}`).join("\n");

export const DAY_S = 86400;
const pad = (n: number) => String(n).padStart(2, "0");
/** A date input's value for a moment (local time). */
export const dateValue = (t: number) => {
  const d = new Date(t * 1000);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
};
/** The end of a chosen day (local time): an account expiring on a date can be used all that day. */
export const endOf = (value: string) => {
  const [y, m, d] = value.split("-").map(Number);
  return new Date(y, m - 1, d, 23, 59, 59).getTime() / 1000;
};
const dayText = (t: number | null) => {
  if (!t) return "—";
  const d = new Date(t * 1000);
  // never 9/13 (read as "9 个"); the year only when it is not this one, so the list fits a narrow window
  return `${d.getFullYear() === new Date().getFullYear() ? "" : `${d.getFullYear()}年`}${d.getMonth() + 1}月${d.getDate()}日`;
};
export const loginText = (t: number | null) => (t ? whenText(t) : "从没登录");
export const now = () => Date.now() / 1000;

export function UsersSection() {
  const { problem } = useAdmin();
  const state = useSignedIn();
  const { data: view, reload: reload } = usePoll(adminApi.users, 15000, { onError: (e) => problem(reasonOf(e)) });
  const [dialog, setDialog] = useState<{ kind: "new" } | { kind: "edit" | "reset" | "delete" | "purge"; user: UserRow } | null>(null);
  const [made, setMade] = useState<{ username: string; password: string } | null>(null);
  const [open, setOpen] = useState(detailOf); // the account whose own page is open (/admin#users/<id>)

  useEffect(() => {
    const follow = () => setOpen(detailOf());
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

  /** A change went through: the list again from the server. */
  const done = useCallback(
    (_: UsersView | null) => {
      problem(null);
      reload();
    },
    [problem, reload],
  );

  const change = async (u: UserRow, c: UserChange) => {
    try {
      done(await adminApi.changeUser(u.id, c));
    } catch (e) {
      problem((e as Error).message);
    }
  };

  const live = view?.users.filter((u) => !u.deleted) ?? [];
  const gone = view?.users.filter((u) => u.deleted) ?? [];
  const tags = view?.tags ?? {};
  const shownUser = view?.users.find((u) => u.id === open) ?? null;

  const dialogs = (
    <>
      {view && dialog?.kind === "new" && (
        <NewUserDialog view={view} a={state?.applies ?? null} onClose={() => setDialog(null)} onMade={(v, m) => (done(v), setMade(m), setDialog(null))} />
      )}
      {view && dialog?.kind === "edit" && <EditDialog view={view} u={dialog.user} onClose={() => setDialog(null)} onDone={(v) => (done(v), setDialog(null))} />}
      {dialog?.kind === "reset" && <ResetDialog u={dialog.user} onClose={() => setDialog(null)} onDone={(v, m) => (done(v), setMade(m), setDialog(null))} />}
      {dialog?.kind === "delete" && (
        <DeleteDialog u={dialog.user} onClose={() => setDialog(null)} onDone={(v) => (done(v), setDialog(null), openList())} />
      )}
      {dialog?.kind === "purge" && (
        <PurgeDialog u={dialog.user} onClose={() => setDialog(null)} onDone={(v) => (done(v), setDialog(null), openList())} />
      )}
    </>
  );

  if (shownUser)
    return (
      <>
        {made && <Made made={made} onClose={() => setMade(null)} />}
        <UserDetail
          key={shownUser.id} // another account: a page of its own (its tabs, the one open, a reason shown), nothing carried over
          u={shownUser}
          roleTip={roleTip(view?.roles ?? [], shownUser.role)}
          onBack={openList}
          onEdit={() => setDialog({ kind: "edit", user: shownUser })}
          onReset={() => setDialog({ kind: "reset", user: shownUser })}
          onDelete={() => setDialog({ kind: "delete", user: shownUser })}
          onChange={(c) => void change(shownUser, c)}
        />
        {dialogs}
      </>
    );

  const columns: Column<UserRow>[] = [
    {
      id: "username",
      label: "用户名",
      tip: "登录用的名字：点开这个账号名下的一切（任务、上传、反馈、交付、登录记录）",
      className: "mono",
      cell: (u) => (
        <Button tip="打开这个账号的页面：名下的任务、上传的素材、反馈、交付、登录记录" tone="link" layout="usr-name" onClick={() => openDetail(u.id)}>
          {u.username}
        </Button>
      ),
    },
    { id: "name", label: "中文名", tip: "统计和队列里显示的中文名", cell: (u) => <span className="usr-cut" data-user-data data-tip={u.name}>{u.name}</span> },
    { id: "role", label: "角色", tip: roleTip(view?.roles ?? []), cell: (u) => <span data-tip={roleTip(view?.roles ?? [], u.role)}>{u.role_label}</span> },
    { id: "department", label: "环节", tip: "账号所属的制作环节；使用统计按它分环节", cell: (u) => <span className="usr-cut" data-user-data data-tip={u.department}>{u.department || "—"}</span> },
    {
      id: "online",
      label: "在线",
      tip: `现在在不在线；不在线时是最后一次活动的时间。${ONLINE_TIP}`,
      cell: (u) => <Presence p={u.presence} windowS={view?.online.window_s ?? 90} />,
    },
    {
      id: "last_login",
      label: "最近登录",
      tip: "最近一次输密码登录的时间",
      className: "tnum",
      cell: (u) => <span data-tip={u.last_login ? dayText(u.last_login) : "还没有登录过"}>{loginText(u.last_login)}</span>,
    },
    ...trafficColumns(),  // the three traffic columns (admin/traffic.tsx): public access through frp is billed by traffic, so each account's use must be visible
    { id: "state", label: "状态", tip: "现在能不能用，不能用的写着为什么", cell: (u) => <span data-tip={u.state_tip}>{u.state}</span> },
    {
      id: "expires",
      label: "到期",
      tip: "这天过完就登录不了，已经登录的也马上退出",
      className: "tnum",
      cell: (u) => <Expiry u={u} />,
    },
    { id: "tags", label: "可用", tip: TAGS_TIP, cell: (u) => <UserTags u={u} tags={tags} /> },
  ];

  return (
    <Section
      title="用户"
      lede="每个人都用自己的账号登录，只看得到自己的任务、结果、上传的素材和反馈。在这里给朋友新建账号，把用户名和初始密码告诉他；到期后登录不了、已经登录的也马上退出，可以延期。点用户名打开这个账号名下的一切。"
      actions={
        shown(state?.applies, "users.create") ? (
          <Button tip="给朋友新建一个账号：用户名、初始密码、中文名、环节、到期时间" tone="primary" disabled={!view} onClick={() => setDialog({ kind: "new" })}>
            新建用户
          </Button>
        ) : undefined
      }
    >
      {made && <Made made={made} onClose={() => setMade(null)} />}
      <div className="adm-tiles">
        <div className="adm-tile" data-tip="这里列出的还在用的账号（不算删掉的）">
          <span className="adm-tile-label">账号</span>
          <b>{view ? live.length : "…"}</b>
        </div>
        <div className="adm-tile" data-tip={onlineTip(view?.online)}>
          <span className="adm-tile-label">在线</span>
          <b>{view ? `浏览器 ${view.online.browser} · 插件 ${view.online.client}` : "…"}</b>
        </div>
      </div>
      {!view ? (
        <Loading what="账号" />
      ) : (
        // what may be done to an account (改, 延期, 停用, 重设密码, 删除) is on its own page, which a click on its row
        // opens: the list is an overview that fits a narrow window without scrolling sideways
        <Table rows={live} columns={columns} rowKey={(u) => String(u.id)} onPick={(u) => openDetail(u.id)} dim={(u) => !u.usable_now} className="usr-table"
          empty={<Empty title="还没有账号" hint="点右上角的「新建用户」建一个。" />} />
      )}
      {gone.length > 0 && (
        <details className="usr-gone">
          <summary data-tip="删掉的账号：登录不了，任务记录还在，使用统计里算作「已删除的用户」">已删除 {gone.length} 个</summary>
          <Table
            rows={gone}
            columns={[
              { id: "username", label: "用户名", tip: "登录用的名字", className: "mono", cell: (u) => u.username },
              { id: "name", label: "中文名", tip: "统计里显示的中文名", cell: (u) => u.name },
              { id: "department", label: "环节", tip: "删除时所在的环节", cell: (u) => u.department || "—" },
              { id: "deleted", label: "删除", tip: "什么时候删的", className: "tnum", cell: (u) => dayText(u.deleted) },
              { id: "jobs", label: "任务", tip: "提交过的任务数", className: "tnum", cell: (u) => u.jobs },
              // 永久删除 takes two steps: 删除 moves the account here, then 永久删除 on its row. Its task records stay, shown as 「已删除的用户」
              { id: "purge", label: "", tip: "把这个账号彻底删掉，用户名可以给新人重用", className: "usr-acts",
                cell: (u) => shown(u.applies, "account.purge") && (
                  <Button tip={why(u.applies, "account.purge") || "彻底删掉这个账号：用户名可以给新人重用，任务记录留着显示成「已删除的用户」"} tone="ghost" size="sm" danger
                    disabled={!usable(u.applies, "account.purge")} onClick={() => setDialog({ kind: "purge", user: u })}>
                    永久删除
                  </Button>
                ) },
            ]}
            rowKey={(u) => String(u.id)}
            dim={() => true}
            className="usr-table"
          />
        </details>
      )}
      {/* 二级管理员权限: only a login that manages other administrators (admins.manage) sees this band */}
      <Rights applies={state?.applies} />
      {dialogs}
    </Section>
  );
}

const ONLINE_TIP = "浏览器、DCC 插件各算一处；点用户名看这个账号名下的一切";

/** The 在线 column (lab2shot/accounts.py presence()): a green dot and 在线 while the account made a request within the
 * last `windowS` seconds; otherwise, in grey, when it was last active. The tooltip says where. The only coloured
 * column of the list. */
function Presence({ p, windowS }: { p: PresenceView; windowS: number }) {
  if (p.online.length)
    return (
      <span className="usr-online" data-tip={`在线：\n${p.online.map(placeText).join("\n")}\n${onlineMeaning(windowS)}`}>
        <i className="usr-online-dot" aria-hidden="true" />
        在线
      </span>
    );
  const where = p.where ? `\n在 ${placeText(p.where)}` : "";
  return (
    <span className="usr-offline tnum" data-tip={p.active ? `不在线：最后一次活动 ${fullTimeText(p.active)}${where}\n${onlineMeaning(windowS)}` : "从没用过"}>
      {activeText(p.active)}
    </span>
  );
}

/** When an account expires, in the list and on its own page (where 延期 is). */
export function Expiry({ u }: { u: UserRow }) {
  const past = u.expires !== null && u.expires <= now();
  return (
    <span
      className={past ? "usr-past" : undefined}
      data-tip={u.expires === null ? "这个账号不会过期" : past ? "已经过期：登录不了，在这个账号的页面上点「+30 天」延期" : `${dayText(u.expires)}过完就登录不了`}
    >
      {u.expires === null ? "不过期" : dayText(u.expires)}
    </span>
  );
}

/** Which account's own page is open: /admin#users/<id> (nothing after the section: the list). */
function detailOf(): number | null {
  const rest = window.location.hash.slice(1).split("/")[1];
  return rest && /^\d+$/.test(rest) ? Number(rest) : null;
}

const openDetail = (id: number) => (window.location.hash = `users/${id}`);
const openList = () => (window.location.hash = "users");

/** What an account may use: every node, the tags it has, or the basic ones only. */
function UserTags({ u, tags }: { u: UserRow; tags: UsersView["tags"] }) {
  return (
    <span className="usr-tags">
      {u.all_nodes ? (
        <span className="chip" data-tip="什么节点都能用，不看标签">
          全部
        </span>
      ) : u.tags.length ? (
        u.tags.map((t) => (
          <span key={t} className="chip" data-tip={tags[t]?.tip ?? t}>
            {tags[t]?.label ?? t}
          </span>
        ))
      ) : (
        <span className="chip" data-tip="只能用基础节点：Lab2Shot 自己的节点和读写文件格式的模块">
          只有基础
        </span>
      )}
    </span>
  );
}

/** The account created (or its new password): what to hand to the user, shown once. */
function Made({ made, onClose }: { made: { username: string; password: string }; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const text = `Lab2Shot：${location.origin}\n用户名：${made.username}\n密码：${made.password}`;
  return (
    <div className="usr-made" role="status">
      <div className="usr-made-text">
        <b>把这些告诉他：</b>
        <span>
          地址 <code className="mono">{location.origin}</code>
        </span>
        <span>
          用户名 <code className="mono">{made.username}</code>
        </span>
        <span>
          密码 <code className="mono">{made.password}</code>
        </span>
      </div>
      <span className="usr-acts">
        <Button tip="复制地址、用户名和密码，发给他；密码只在这里显示这一次" onClick={() => void copyText(text).then(() => setCopied(true))}>
          {copied ? "已复制" : "复制"}
        </Button>
        <Button tip="关掉这个提示：之后再也看不到这个密码" tone="ghost" onClick={onClose}>
          关掉
        </Button>
      </span>
    </div>
  );
}
