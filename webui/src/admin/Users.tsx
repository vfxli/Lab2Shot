import { useCallback, useEffect, useState } from "react";
import { copyText } from "../platform/util";
import { adminApi, type Presence as PresenceView, type UserChange, type UserRow, type UsersView } from "../api/admin";
import { useSignedIn } from "../state/session";
import { shown, usable, why } from "../api/applies";
import { activeText, onlineTip, placeText, Section, useAdmin } from "./common";
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
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 用户: the accounts (lab2shot/accounts.py, server/users.py). The administrator creates one for each user (a username,
 * an initial password to hand over, a Chinese name and department for the statistics, an expiry date, what they may use),
 * extends or disables it, resets a forgotten password, deletes it. Which rows are listed and what each row, the new-user
 * form and the edit form offer is the server's answer (server/available.py, read through applies.ts): a
 * 二级管理员 sees only 普通用户 and no tags, roles or deleting. Every rule the server checks is checked here first,
 * so a mistake shows before anything is sent. */

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
const dayText = (at: number | null) => {
  if (!at) return "—";
  const d = new Date(at * 1000);
  // never 9/13 (read as "9 个"); the year only when it is not this one, so the list fits a narrow window
  const md = { month: d.getMonth() + 1, day: d.getDate() };
  return d.getFullYear() === new Date().getFullYear() ? t("ui.admin.usage.month_day", md) : t("ui.admin.usage.year_month_day", { year: d.getFullYear(), ...md });
};
export const loginText = (at: number | null) => (at ? whenText(at) : t("ui.admin.users.never_logged_in"));
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
      label: t("ui.admin.auth.username"),
      className: "mono",
      cell: (u) => (
        <Button tone="link" layout="usr-name" onClick={() => openDetail(u.id)}>
          {u.username}
        </Button>
      ),
    },
    { id: "name", label: t("ui.admin.users.name"), cell: (u) => <span className="usr-cut" data-user-data {...tipAttrs(tipOf("truncated", u.name))}>{u.name}</span> },
    { id: "role", label: t("ui.admin.users.role"), cell: (u) => <span>{u.role_label}</span> },
    { id: "department", label: t("ui.admin.users.department"), cell: (u) => <span className="usr-cut" data-user-data {...tipAttrs(tipOf("truncated", u.department_label))}>{u.department_label || "—"}</span> },
    {
      id: "online",
      label: t("ui.admin.security.online"),
      cell: (u) => <Presence p={u.presence} />,
    },
    {
      id: "last_login",
      label: t("ui.admin.users.last_login"),
      className: "tnum",
      cell: (u) => loginText(u.last_login),
    },
    ...trafficColumns(),  // the three traffic columns (admin/traffic.tsx): public access through frp is billed by traffic, so each account's use must be visible
    { id: "state", label: t("ui.admin.resources.state"), cell: (u) => <span {...tipAttrs(tipOf("truncated", u.state_tip))}>{u.state}</span> },
    {
      id: "expires",
      label: t("ui.admin.users.expiry"),
      className: "tnum",
      cell: (u) => <Expiry u={u} />,
    },
    { id: "tags", label: t("ui.admin.users.allowed"), cell: (u) => <UserTags u={u} tags={tags} /> },
  ];

  return (
    <Section
      title={t("ui.admin.nav.users")}
      lede={t("ui.admin.users.lede")}
      actions={
        shown(state?.applies, "users.create") ? (
          <Button tone="primary" disabled={!view} onClick={() => setDialog({ kind: "new" })}>
            {t("ui.admin.users.new_title")}
          </Button>
        ) : undefined
      }
    >
      {made && <Made made={made} onClose={() => setMade(null)} />}
      <div className="adm-tiles">
        <div className="adm-tile">
          <span className="adm-tile-label">{t("ui.admin.users.accounts")}</span>
          <b>{view ? live.length : "…"}</b>
        </div>
        <div className="adm-tile" {...tipAttrs(onlineTip(view?.online))}>
          <span className="adm-tile-label">{t("ui.admin.security.online")}</span>
          <b>{view ? t("ui.admin.security.online_value", { browser: view.online.browser, client: view.online.client }) : "…"}</b>
        </div>
      </div>
      {!view ? (
        <Loading what={t("ui.admin.users.accounts")} />
      ) : (
        // what may be done to an account (改, 延期, 停用, 重设密码, 删除) is on its own page, which a click on its row
        // opens: the list is an overview that fits a narrow window without scrolling sideways
        <Table rows={live} columns={columns} rowKey={(u) => String(u.id)} onPick={(u) => openDetail(u.id)} dim={(u) => !u.usable_now} className="usr-table"
          empty={<Empty title={t("ui.admin.users.none")} hint={t("ui.admin.users.none_hint")} />} />
      )}
      {gone.length > 0 && (
        <details className="usr-gone">
          <summary>{t("ui.admin.users.deleted", { n: gone.length })}</summary>
          <Table
            rows={gone}
            columns={[
              { id: "username", label: t("ui.admin.auth.username"), className: "mono", cell: (u) => u.username },
              { id: "name", label: t("ui.admin.users.name"), cell: (u) => u.name },
              { id: "department", label: t("ui.admin.users.department"), cell: (u) => u.department_label || "—" },
              { id: "deleted", label: t("ui.admin.users.deleted_on"), className: "tnum", cell: (u) => dayText(u.deleted) },
              { id: "jobs", label: t("ui.admin.users.jobs"), className: "tnum", cell: (u) => u.jobs },
              // 永久删除 takes two steps: 删除 moves the account here, then 永久删除 on its row. Its task records stay, shown as 「已删除的用户」
              { id: "purge", label: "", className: "usr-acts",
                cell: (u) => shown(u.applies, "account.purge") && (
                  <Button tip={why(u.applies, "account.purge") ? tipOf("disabled", why(u.applies, "account.purge")) : tipOf("consequence", t("ui.admin.users.purge_tip"))} tone="ghost" size="sm" danger
                    disabled={!usable(u.applies, "account.purge")} onClick={() => setDialog({ kind: "purge", user: u })}>
                    {t("ui.admin.users.purge")}
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

/** The 在线 column (lab2shot/accounts.py presence()): a green dot and 在线 while the account made a request within the
 * server's online window (lab2shot/accounts.py ONLINE_S); otherwise, in grey, when it was last active. The tooltip says
 * where. The only coloured column of the list. */
function Presence({ p }: { p: PresenceView }) {
  if (p.online.length)
    return (
      <span className="usr-online" {...tipAttrs(tipOf("value", p.online.map(placeText).join("\n")))}>
        <i className="usr-online-dot" aria-hidden="true" />
        {t("ui.admin.security.online")}
      </span>
    );
  const where = p.where ? t("ui.admin.users.where", { place: placeText(p.where) }) : "";
  return (
    <span className="usr-offline tnum" {...tipAttrs(p.active ? tipOf("value", t("ui.admin.users.offline_tip", { at: fullTimeText(p.active), where })) : undefined)}>
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
      {...tipAttrs(past ? tipOf("error", t("ui.admin.users.expired_tip")) : undefined)}
    >
      {u.expires === null ? t("ui.admin.users.no_expiry") : dayText(u.expires)}
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
        <span className="chip">
          {t("ui.admin.resources.all")}
        </span>
      ) : u.tags.length ? (
        u.tags.map((t) => (
          <span key={t} className="chip">
            {tags[t]?.label ?? t}
          </span>
        ))
      ) : (
        <span className="chip">
          {t("ui.admin.users.basic_only")}
        </span>
      )}
    </span>
  );
}

/** The account created (or its new password): what to hand to the user, shown once. */
function Made({ made, onClose }: { made: { username: string; password: string }; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const text = t("ui.admin.users.made_text", { origin: location.origin, username: made.username, password: made.password });
  return (
    <div className="usr-made" role="status">
      <div className="usr-made-text">
        <b>{t("ui.admin.users.made_tell")}</b>
        <span>
          {t("ui.admin.users.made_address")} <code className="mono">{location.origin}</code>
        </span>
        <span>
          {t("ui.admin.auth.username")} <code className="mono">{made.username}</code>
        </span>
        <span>
          {t("ui.admin.auth.password")} <code className="mono">{made.password}</code>
        </span>
      </div>
      <span className="usr-acts">
        <Button onClick={() => void copyText(text).then(() => setCopied(true))}>
          {copied ? t("ui.admin.common.copied") : t("ui.admin.common.copy")}
        </Button>
        <Button tip={tipOf("consequence", t("ui.admin.users.made_close_tip"))} tone="ghost" onClick={onClose}>
          {t("ui.admin.users.made_close")}
        </Button>
      </span>
    </div>
  );
}
