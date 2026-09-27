import { useCallback, useEffect, useState } from "react";
import { copyText } from "../platform/util";
import { adminApi, type UserChange, type UserRow, type UsersView } from "../api/admin";
import { TAGS_TIP } from "./userFields";
import { useSignedIn } from "../state/session";
import { shown, usable, why } from "../api/applies";
import { onlineTip, Section, useAdmin } from "./common";
import { whenText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button, Switch } from "../ui/Button";
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

export const ROLE_TIP = "角色：管理员什么都能做；二级管理员管普通用户、队列、反馈和基准素材，不碰服务器、安装、数据库和标签；普通用户只用编辑器";

export const DAY_S = 86400;
const USERNAME = /^[a-z][a-z0-9_.-]{2,31}$/; // lab2shot/accounts.py USERNAME
const HAN = "\\p{Script=Han}";
const NAME = new RegExp(`^${HAN}+(·${HAN}+)*$`, "u"); // lab2shot/accounts.py _NAME
const DOTS = /[·•・‧∙]/g;

export const tidyName = (name: string) => name.trim().replace(DOTS, "·");

export function usernameProblem(raw: string): string {
  if (!raw) return "";
  return USERNAME.test(raw) ? "" : "用户名要 3 到 32 个字符：小写英文字母开头，只用小写字母、数字和 _ . -";
}

export function nameProblem(raw: string): string {
  const name = tidyName(raw);
  if (!name) return "要填中文名：2 到 6 个字";
  if (/\s/.test(name)) return `名字「${name}」里有空格：少数民族名字用 · 隔开，如 阿依·买买提`;
  if (!NAME.test(name)) return /[A-Za-z]/.test(name) ? `名字「${name}」要写中文名，不是拼音或英文名` : `名字「${name}」只能是中文字，名字之间可以用 · 隔开`;
  const n = [...name.replaceAll("·", "")].length;
  return n < 2 || n > 6 ? `名字「${name}」有 ${n} 个字：中文名要 2 到 6 个字` : "";
}

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
  return `${d.getFullYear()}年${d.getMonth() + 1}月${d.getDate()}日`; // never 9/13 (read as "9 个")
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
          u={shownUser}
          onBack={openList}
          onEdit={() => setDialog({ kind: "edit", user: shownUser })}
          onReset={() => setDialog({ kind: "reset", user: shownUser })}
          onDelete={() => setDialog({ kind: "delete", user: shownUser })}
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
    { id: "name", label: "中文名", tip: "统计和队列里显示的中文名", cell: (u) => u.name },
    { id: "role", label: "角色", tip: ROLE_TIP, cell: (u) => <span data-tip={ROLE_TIP}>{u.role_label}</span> },
    { id: "department", label: "部门", tip: "使用统计按它分部门", cell: (u) => u.department || "—" },
    {
      id: "expires",
      label: "到期",
      tip: "这天过完就登录不了，已经登录的也马上退出",
      className: "tnum",
      cell: (u) => {
        const past = u.expires !== null && u.expires <= now();
        return (
          <span
            className={past ? "usr-past" : undefined}
            data-tip={u.expires === null ? "这个账号不会过期" : past ? "已经过期：登录不了，点「延期」" : `${dayText(u.expires)}过完就登录不了`}
          >
            {u.expires === null ? "不过期" : dayText(u.expires)}
          </span>
        );
      },
    },
    { id: "state", label: "状态", tip: "现在能不能用，不能用的写着为什么", cell: (u) => <span data-tip={u.state_tip}>{u.state}</span> },
    { id: "tags", label: "可用", tip: TAGS_TIP, cell: (u) => <UserTags u={u} tags={tags} /> },
    {
      id: "last_login",
      label: "最近登录",
      tip: "最近一次输密码登录的时间",
      className: "tnum",
      cell: (u) => <span data-tip={u.last_login ? dayText(u.last_login) : "还没有登录过"}>{loginText(u.last_login)}</span>,
    },
    { id: "jobs", label: "任务", tip: "提交过的任务数", className: "tnum", cell: (u) => u.jobs },
    ...trafficColumns(),  // 流量三列（admin/traffic.tsx）：公网经 frp 按流量计费，需要能够查看各账号的用量
    {
      id: "online",
      label: "在线",
      tip: "现在在不在线：浏览器、DCC 插件各算一处；点用户名看这个账号名下的一切",
      cell: (u) => (u.online.browser && u.online.client ? "浏览器、插件" : u.online.browser ? "浏览器" : u.online.client ? "插件" : "不在线"),
    },
    {
      id: "acts",
      label: "",
      tip: "对这个账号的操作",
      cell: (u) => (
        <UserActs u={u} onEdit={() => setDialog({ kind: "edit", user: u })} onReset={() => setDialog({ kind: "reset", user: u })}
          onDelete={() => setDialog({ kind: "delete", user: u })} onChange={(c) => void change(u, c)} />
      ),
    },
  ];

  return (
    <Section
      title="用户"
      lede="每个人都用自己的账号登录，只看得到自己的任务、结果、上传的素材和反馈。在这里给朋友新建账号，把用户名和初始密码告诉他；到期后登录不了、已经登录的也马上退出，可以延期。点用户名打开这个账号名下的一切。"
      actions={
        shown(state?.applies, "users.create") ? (
          <Button tip="给朋友新建一个账号：用户名、初始密码、中文名、部门、到期时间" tone="primary" disabled={!view} onClick={() => setDialog({ kind: "new" })}>
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
        <Table rows={live} columns={columns} rowKey={(u) => String(u.id)} dim={(u) => !u.usable_now} className="usr-table"
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
              { id: "department", label: "部门", tip: "删除时所在的部门", cell: (u) => u.department || "—" },
              { id: "deleted", label: "删除", tip: "什么时候删的", className: "tnum", cell: (u) => dayText(u.deleted) },
              { id: "jobs", label: "任务", tip: "提交过的任务数", className: "tnum", cell: (u) => u.jobs },
              // 永久删除分两步：先「删除」移至此处，再在该行执行永久删除。任务记录保留，显示为「已删除的用户」
              { id: "purge", label: "", tip: "把这个账号彻底删掉，用户名可以给新人重用", className: "usr-acts",
                cell: (u) => (
                  <Button tip="彻底删掉这个账号：用户名可以给新人重用，任务记录留着显示成「已删除的用户」" tone="ghost" size="sm" danger
                    onClick={() => setDialog({ kind: "purge", user: u })}>
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
      {/* 二级管理员权限：只有能管理其他管理员的登录（admins.manage）才能看到此区域 */}
      <Rights applies={state?.applies} />
      {dialogs}
    </Section>
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

/** What may be done to one account, as the server resolved it for this login (`u.applies`). */
function UserActs({ u, onEdit, onReset, onDelete, onChange }: {
  u: UserRow;
  onEdit: () => void;
  onReset: () => void;
  onDelete: () => void;
  onChange: (c: UserChange) => void;
}) {
  const extend = (days: number) => onChange({ expires: Math.max(u.expires ?? now(), now()) + days * DAY_S });
  return (
    <span className="usr-acts">
      {usable(u.applies, "account.edit") && (
        <Button tip="改中文名、部门、到期时间这些" tone="ghost" onClick={onEdit}>
          改
        </Button>
      )}
      {shown(u.applies, "account.expiry") && (
        <>
          <Button tip={why(u.applies, "account.expiry") || "到期时间往后推 30 天（已经过期的从今天算起）"} tone="ghost" disabled={!usable(u.applies, "account.expiry")} onClick={() => extend(30)}>
            +30 天
          </Button>
          <Button tip={why(u.applies, "account.expiry") || "到期时间往后推 90 天（已经过期的从今天算起）"} tone="ghost" disabled={!usable(u.applies, "account.expiry")} onClick={() => extend(90)}>
            +90 天
          </Button>
        </>
      )}
      {shown(u.applies, "account.enable") && (
        <Switch
          on={u.enabled}
          label={`${u.username} 能不能用`}
          disabled={!usable(u.applies, "account.enable")}
          tip={why(u.applies, "account.enable") || (u.enabled ? "停用：登录不了，已经登录的马上退出；任务和结果都留着，可以再启用" : "启用：又能登录了")}
          onChange={(enabled) => onChange({ enabled })}
        />
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
