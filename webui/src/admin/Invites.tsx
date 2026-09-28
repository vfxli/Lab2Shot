import { useCallback, useState } from "react";
import { adminApi, type Invite, type InvitesView, type Registered, type RegisteredFilter } from "../api/admin";
import { shown, usable, why } from "../api/applies";
import { useAdmin } from "./common";
import { dateValue, endOf, now, DAY_S } from "./Users";
import { Row } from "./userFields";
import { usePoll } from "../platform/poll";
import { copyText } from "../platform/util";
import { reasonOf } from "../messages/message";
import { whenText } from "../platform/format";
import { useSignedIn } from "../state/session";
import { Button, Switch } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { Sheet } from "../ui/Sheet";
import { Table, type Column } from "../ui/Table";
import "./users.css";

/** The invite codes of 自行注册 (lab2shot/registration.py, server/invites.py), the lower part of 注册设置 (below the
 * registration settings, admin/sections.tsx), and switching off in bulk the accounts that registered themselves with
 * one code or within a time. The list shows each code in full, with a button to copy it (logs and the audit log name
 * it by its first characters only). Shown to a login that may read the codes (invites.list).
 *
 * What registering did lately is said once on the page: in the registration settings' card for a login that reads the
 * settings (server/settings.py status), otherwise here, above the codes. */

const usesText = (i: { uses_max: number | null; used: number }) => (i.uses_max === null ? `${i.used} 次（不限）` : `${i.used} / ${i.uses_max} 次`);

export function InvitesPart() {
  const state = useSignedIn();
  if (!shown(state?.applies, "invites.list")) return null;
  return <Invites />;
}

function Invites() {
  const { problem } = useAdmin();
  const state = useSignedIn();
  const { data: view, reload } = usePoll(adminApi.invites, 10000, { onError: (e) => problem(reasonOf(e)) });
  const [dialog, setDialog] = useState<{ kind: "new" } | { kind: "edit"; invite: Invite } | { kind: "disable"; filter: RegisteredFilter; label: string } | null>(null);
  const [open, setOpen] = useState<number | null>(null);
  const edits = usable(state?.applies, "invites.edit");
  const done = () => (problem(null), reload());

  const remove = async (i: Invite) => {
    if (!window.confirm(`删除邀请码 ${i.code}？以后不能再用它注册；用它注册的 ${i.accounts.length} 个账号照旧。`)) return;
    try {
      await adminApi.deleteInvite(i.id);
      done();
    } catch (e) {
      problem((e as Error).message);
    }
  };
  const flip = async (i: Invite) => {
    try {
      await adminApi.changeInvite(i.id, { note: i.note, uses_max: i.uses_max, expires: i.expires, enabled: !i.enabled });
      done();
    } catch (e) {
      problem((e as Error).message);
    }
  };

  const columns: Column<Invite>[] = [
    { id: "code", label: "邀请码", tip: "发给要来注册的人的码；点「复制」拿去发", cell: (i) => <CodeCell code={i.code} /> },
    { id: "note", label: "备注", tip: "新建时写的备注：给谁的、用在哪次活动", cell: (i) => <span className="inv-note" data-user-data data-tip={i.note}>{i.note || "—"}</span> },
    { id: "used", label: "用了", tip: "用它注册了几个账号 / 最多几个", cell: (i) => usesText(i) },
    { id: "expires", label: "到期", tip: "过了这个时间就不能再用", cell: (i) => (i.expires ? whenText(i.expires) : "不过期") },
    { id: "state", label: "状态", tip: "现在能不能用它注册", cell: (i) => i.state },
    { id: "by", label: "谁建的", tip: "谁、什么时候建的", cell: (i) => <span data-tip={whenText(i.created)}>{i.created_by}</span> },
    {
      id: "acts",
      label: "",
      tip: "",
      cell: (i) => (
        <span className="usr-acts" onClick={(e) => e.stopPropagation()}>
          <Switch on={i.enabled} label={i.enabled ? "停用" : "启用"} disabled={!edits} onChange={() => void flip(i)} />
          <Button tip={why(state?.applies, "invites.edit") || "改备注、可用次数和到期"} tone="ghost" disabled={!edits} onClick={() => setDialog({ kind: "edit", invite: i })}>
            改
          </Button>
          <Button tip={why(state?.applies, "invites.edit") || "删除这个码"} tone="ghost" danger disabled={!edits} onClick={() => void remove(i)}>
            删除
          </Button>
        </span>
      ),
    },
  ];
  const shownInvite = view?.invites.find((i) => i.id === open) ?? null;

  return (
    <section className="set-card set-part">
      <header className="set-part-head">
        <h3>邀请码</h3>
        {shown(state?.applies, "invites.edit") && (
          <Button tip={why(state?.applies, "invites.edit") || "新建一个邀请码：自己填，或者随机生成"} tone="primary" disabled={!edits} onClick={() => setDialog({ kind: "new" })}>
            新建邀请码
          </Button>
        )}
        {shown(state?.applies, "registrations.disable") && (
          <Button tip={why(state?.applies, "registrations.disable") || "按注册时间段批量停用自己注册的账号"} tone="ghost" warn disabled={!usable(state?.applies, "registrations.disable")}
            onClick={() => setDialog({ kind: "disable", filter: { since: now() - DAY_S }, label: "一段时间里自己注册的账号" })}>
            按时间段停用…
          </Button>
        )}
      </header>
      <p className="adm-lede">开了「邀请码验证」时，自己注册要填一个能用的邀请码。点一行看谁用它注册了。</p>
      {view && !shown(state?.applies, "settings.values") && <Registering view={view} />}
      {!view ? (
        <p className="adm-lede">读取中…</p>
      ) : (
        <Table rows={view.invites} columns={columns} rowKey={(i) => String(i.id)} picked={open === null ? null : String(open)}
          onPick={(i) => setOpen(open === i.id ? null : i.id)} dim={(i) => !i.usable}
          empty={<Empty title="还没有邀请码" hint="点「新建邀请码」建一个，把它发给要来注册的人" />} />
      )}
      {shownInvite && (
        <Accounts invite={shownInvite}
          onDisable={() => setDialog({ kind: "disable", filter: { invite: shownInvite.id }, label: `用邀请码 ${shownInvite.code} 注册的账号` })} />
      )}
      {view && dialog?.kind === "new" && (
        <InviteDialog view={view} onClose={() => setDialog(null)} onDone={() => (setDialog(null), done())} />
      )}
      {view && dialog?.kind === "edit" && <InviteDialog view={view} invite={dialog.invite} onClose={() => setDialog(null)} onDone={() => (setDialog(null), done())} />}
      {dialog?.kind === "disable" && <DisableDialog filter={dialog.filter} label={dialog.label} onClose={() => setDialog(null)} onDone={() => (setDialog(null), done())} />}
    </section>
  );
}

/** What registering did lately, against the site-wide limits, and whether that paused it. */
function Registering({ view }: { view: InvitesView }) {
  const r = view.registering;
  const state = !r.open ? "没开放注册" : r.paused ? `注册已自动暂停（到了每${r.paused === "hour" ? "小时" : "天"}上限）` : r.invite ? "开放注册，要邀请码" : "开放注册，不要邀请码";
  return (
    <div className="adm-tiles">
      <div className="adm-tile" data-tip="全站自己注册的账号数和两个上限；到了上限注册自动暂停，过了那一小时 / 那一天自己恢复">
        <span className="adm-tile-label">注册</span>
        <b>{state}</b>
        <span className="adm-tile-sub">最近一小时 {r.hour} / {r.per_hour} 个 · 最近一天 {r.day} / {r.per_day} 个</span>
      </div>
    </div>
  );
}

/** A code, and a button to copy it. */
function CodeCell({ code }: { code: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <span className="usr-acts" onClick={(e) => e.stopPropagation()}>
      <code className="mono" data-user-data>{code}</code>
      <Button tip="复制这个邀请码，发给要来注册的人" tone="ghost" onClick={() => void copyText(code).then(() => setCopied(true))}>
        {copied ? "已复制" : "复制"}
      </Button>
    </span>
  );
}

/** The accounts registered with one code, as they are now, and switching off at once those still on. The button is
 * here rather than on the code's row so the list of codes fits the width of the settings' cards. */
function Accounts({ invite, onDisable }: { invite: Invite; onDisable: () => void }) {
  const state = useSignedIn();
  const columns: Column<Invite["accounts"][number]>[] = [
    { id: "username", label: "用户名", tip: "用这个码注册的账号", cell: (a) => <span className="mono" data-user-data>{a.username}</span> },
    { id: "name", label: "中文名", tip: "注册时填的中文名（后来改过就是改后的）", cell: (a) => <span data-user-data>{a.name}</span> },
    { id: "at", label: "注册时间", tip: "什么时候注册的", cell: (a) => whenText(a.at) },
    { id: "ip", label: "来源", tip: "注册时的地址（经可信代理时是它转告的地址）", cell: (a) => <span className="mono" data-user-data>{a.ip}</span> },
    { id: "state", label: "状态", tip: "账号现在能不能用；在「用户」里管", cell: (a) => (a.deleted ? "已删除" : a.enabled ? "开着" : "已停用") },
  ];
  return (
    <>
      <header className="set-part-head">
        <h3 className="adm-h3">用 {invite.code} 注册的账号</h3>
        {shown(state?.applies, "registrations.disable") && (
          <Button tip={why(state?.applies, "registrations.disable") || "把用这个码注册、现在还开着的账号全部停用"} tone="ghost" warn
            disabled={!usable(state?.applies, "registrations.disable") || !invite.accounts.some((a) => a.enabled && !a.deleted)}
            onClick={onDisable}>
            停用注册的账号
          </Button>
        )}
      </header>
      <Table rows={invite.accounts} columns={columns} rowKey={(a) => String(a.id)} dim={(a) => !a.enabled || !!a.deleted}
        empty={<Empty title="还没有人用它注册" />} />
    </>
  );
}

function InviteDialog({ view, invite, onClose, onDone }: { view: InvitesView; invite?: Invite; onClose: () => void; onDone: () => void }) {
  const [code, setCode] = useState("");
  const [note, setNote] = useState(invite?.note ?? "");
  const [uses, setUses] = useState(invite?.uses_max === null || invite?.uses_max === undefined ? "" : String(invite.uses_max));
  const [expires, setExpires] = useState(invite?.expires ? dateValue(invite.expires) : "");
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const c = view.rules;
  const typed = code.toUpperCase().replace(/[\s-]/g, "");
  const wrong = {
    code: !typed ? "" : typed.length < c.min || typed.length > c.max || !/^[A-Z0-9]+$/.test(typed) ? `自己填的邀请码要 ${c.min} 到 ${c.max} 个英文字母或数字` : "",
    uses: !uses.trim() ? "" : !/^\d+$/.test(uses.trim()) || +uses < 1 || +uses > c.uses_most ? `可用次数要填 1 到 ${c.uses_most} 的整数，留空表示不限` : invite && +uses < invite.used ? `已经用了 ${invite.used} 次：不能比这还少` : "",
    expires: !expires || (invite?.expires && expires === dateValue(invite.expires)) ? "" : endOf(expires) <= now() ? "到期时间要在以后" : "",
  };
  const bad = Object.values(wrong).some(Boolean);
  const submit = async () => {
    if (bad || busy) return;
    setBusy(true);
    const until = !expires ? null : invite?.expires && expires === dateValue(invite.expires) ? invite.expires : endOf(expires);
    const limit = uses.trim() ? +uses : null;
    try {
      if (invite) {
        await adminApi.changeInvite(invite.id, { note, uses_max: limit, expires: until, enabled: invite.enabled });
        onDone();
      } else {
        await adminApi.createInvite({ code: typed, note, uses_max: limit, expires: until });
        onDone();
      }
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={invite ? `改邀请码 ${invite.code}` : "新建邀请码"} width={520} onClose={onClose}>
      <div className="usr-form">
        {!invite && (
          <Row label="邀请码" tip={`留空就随机生成一个 16 位的（不会猜中，也不会有 0 和 O、1 和 I 这种看错的字）；自己填要 ${c.min} 到 ${c.max} 个英文字母或数字`} why={wrong.code}>
            <input className={`field${wrong.code ? " bad" : ""}`} value={code} autoFocus spellCheck={false} placeholder="留空：随机生成" data-tip="留空就随机生成"
              onChange={(e) => (setCode(e.target.value), setRefused(""))} />
          </Row>
        )}
        <Row label="备注" tip={`给谁的、用在哪次活动，最多 ${c.note_most} 个字；只在后台看得到`}>
          <input className="field" value={note} maxLength={c.note_most} placeholder="如 第一批测试" data-tip="只在后台看得到" onChange={(e) => (setNote(e.target.value), setRefused(""))} />
        </Row>
        <Row label="可用次数" tip="最多能用它注册几个账号；留空不限" why={wrong.uses}>
          <input className={`field${wrong.uses ? " bad" : ""}`} inputMode="numeric" value={uses} placeholder="留空：不限" data-tip="留空不限次数" onChange={(e) => (setUses(e.target.value), setRefused(""))} />
        </Row>
        <Row label="到期" tip="这天过完就不能再用；留空不过期" why={wrong.expires}>
          <input className={`field${wrong.expires ? " bad" : ""}`} type="date" value={expires} min={dateValue(now())} data-tip="留空不过期" onChange={(e) => (setExpires(e.target.value), setRefused(""))} />
        </Row>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不改了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={bad ? "先把上面紫框里的填好" : invite ? "保存" : "建好：它出现在列表最上面，点「复制」拿去发"} tone="primary" disabled={bad || busy} onClick={() => void submit()}>
            {busy ? "保存中…" : invite ? "保存" : "新建"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

/** Switch off, in one go, the accounts that registered themselves with one code, or within a time: first the list
 * the server finds, then the switch. */
function DisableDialog({ filter: first, label, onClose, onDone }: { filter: RegisteredFilter; label: string; onClose: () => void; onDone: () => void }) {
  const byTime = first.invite === undefined;
  const [since, setSince] = useState(first.since ? dateValue(first.since) : "");
  const [until, setUntil] = useState("");
  const filter: RegisteredFilter = byTime
    ? { ...(since ? { since: endOf(since) - DAY_S + 1 } : {}), ...(until ? { until: endOf(until) + 1 } : {}) }
    : first;
  const key = JSON.stringify(filter);
  const [refused, setRefused] = useState("");
  // one read per filter: the reader is keyed on the filter's text (a new function every render would read again at once)
  const read = useCallback(() => adminApi.registered(JSON.parse(key) as RegisteredFilter), [key]);
  const { data } = usePoll(read, null, { key, onError: (e) => setRefused(reasonOf(e)) });
  const [busy, setBusy] = useState(false);
  const rows: Registered[] = data?.accounts ?? [];
  const targets = rows.filter((r) => r.enabled && r.managed);
  const go = async () => {
    setBusy(true);
    try {
      await adminApi.disableRegistered(filter);
      onDone();
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const columns: Column<Registered>[] = [
    { id: "username", label: "用户名", tip: "自己注册的账号", cell: (r) => <span className="mono" data-user-data>{r.username}</span> },
    { id: "name", label: "中文名", tip: "中文名", cell: (r) => <span data-user-data>{r.name}</span> },
    { id: "at", label: "注册时间", tip: "什么时候注册的", cell: (r) => whenText(r.registered) },
    { id: "invite", label: "邀请码", tip: "注册时用的码的前几位", cell: (r) => (r.invite ? `${r.invite}…` : "—") },
    { id: "state", label: "状态", tip: "开着的、你管得着的才会停用", cell: (r) => (!r.enabled ? "已停用" : r.managed ? "会停用" : "你管不着") },
  ];
  return (
    <Sheet title={`停用${label}`} width={640} onClose={onClose}>
      <div className="usr-form">
        {byTime && (
          <>
            <Row label="从" tip="这天（含）以后注册的；留空从最早">
              <input className="field" type="date" value={since} data-tip="留空从最早" onChange={(e) => setSince(e.target.value)} />
            </Row>
            <Row label="到" tip="这天（含）以前注册的；留空到现在">
              <input className="field" type="date" value={until} data-tip="留空到现在" onChange={(e) => setUntil(e.target.value)} />
            </Row>
          </>
        )}
        <Table rows={rows} columns={columns} rowKey={(r) => String(r.id)} dim={(r) => !r.enabled || !r.managed} empty={<Empty title={data ? "没有这样的账号" : "读取中…"} />} />
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不停了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={targets.length ? `停用这 ${targets.length} 个账号：它们的登录马上失效，以后可以在「用户」里再启用` : "没有要停用的"} tone="primary" danger
            disabled={busy || !targets.length || (byTime && !since && !until)} onClick={() => void go()}>
            {busy ? "停用中…" : `停用 ${targets.length} 个`}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
