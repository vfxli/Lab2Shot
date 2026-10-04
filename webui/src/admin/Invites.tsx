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
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** The invite codes of 自行注册 (lab2shot/site/registration.py, server/invites.py), the lower part of 注册设置 (below the
 * registration settings, admin/sections.tsx), and switching off in bulk the accounts that registered themselves with
 * one code or within a time. The list shows each code in full, with a button to copy it (logs and the audit log name
 * it by its first characters only). Shown to a login that may read the codes (invites.list).
 *
 * What registering did lately is said once on the page: in the registration settings' card for a login that reads the
 * settings (server/settings.py status), otherwise here, above the codes. */

const usesText = (i: { uses_max: number | null; used: number }) => (i.uses_max === null ? t("ui.admin.invites.uses_unlimited", { n: i.used }) : t("ui.admin.invites.uses_of", { n: i.used, most: i.uses_max }));

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
    if (!window.confirm(t("ui.admin.invites.delete_confirm", { code: i.code, n: i.accounts.length }))) return;
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
    { id: "code", label: t("ui.admin.invites.code"), cell: (i) => <CodeCell code={i.code} /> },
    { id: "note", label: t("ui.admin.invites.note"), cell: (i) => <span className="inv-note" data-user-data {...tipAttrs(tipOf("truncated", i.note))}>{i.note || "—"}</span> },
    { id: "used", label: t("ui.admin.invites.used"), cell: (i) => usesText(i) },
    { id: "expires", label: t("ui.admin.users.expiry"), cell: (i) => (i.expires ? whenText(i.expires) : t("ui.admin.users.no_expiry")) },
    { id: "state", label: t("ui.admin.resources.state"), cell: (i) => i.state },
    { id: "by", label: t("ui.admin.invites.by"), cell: (i) => <span {...tipAttrs(tipOf("value", whenText(i.created)))}>{i.created_by}</span> },
    {
      id: "acts",
      label: "",
      cell: (i) => (
        <span className="usr-acts" onClick={(e) => e.stopPropagation()}>
          <Switch on={i.enabled} label={i.enabled ? t("ui.admin.invites.disable") : t("ui.admin.invites.enable")} disabled={!edits} onChange={() => void flip(i)} />
          <Button tip={tipOf("disabled", why(state?.applies, "invites.edit"))} tone="ghost" disabled={!edits} onClick={() => setDialog({ kind: "edit", invite: i })}>
            {t("ui.admin.user.edit")}
          </Button>
          <Button tip={tipOf("disabled", why(state?.applies, "invites.edit"))} tone="ghost" danger disabled={!edits} onClick={() => void remove(i)}>
            {t("ui.admin.user.delete")}
          </Button>
        </span>
      ),
    },
  ];
  const shownInvite = view?.invites.find((i) => i.id === open) ?? null;

  return (
    <section className="set-card set-part">
      <header className="set-part-head">
        <h3>{t("ui.admin.invites.title")}</h3>
        {shown(state?.applies, "invites.edit") && (
          <Button tip={tipOf("disabled", why(state?.applies, "invites.edit"))} tone="primary" disabled={!edits} onClick={() => setDialog({ kind: "new" })}>
            {t("ui.admin.invites.new")}
          </Button>
        )}
        {shown(state?.applies, "registrations.disable") && (
          <Button tip={tipOf("disabled", why(state?.applies, "registrations.disable"))} tone="ghost" warn disabled={!usable(state?.applies, "registrations.disable")}
            onClick={() => setDialog({ kind: "disable", filter: { since: now() - DAY_S }, label: t("ui.admin.invites.by_time_what") })}>
            {t("ui.admin.invites.by_time")}
          </Button>
        )}
      </header>
      <p className="adm-lede">{t("ui.admin.invites.lede")}</p>
      {view && !shown(state?.applies, "settings.values") && <Registering view={view} />}
      {!view ? (
        <p className="adm-lede">{t("ui.admin.common.reading")}</p>
      ) : (
        <Table rows={view.invites} columns={columns} rowKey={(i) => String(i.id)} picked={open === null ? null : String(open)}
          onPick={(i) => setOpen(open === i.id ? null : i.id)} dim={(i) => !i.usable}
          empty={<Empty title={t("ui.admin.invites.none")} hint={t("ui.admin.invites.none_hint")} />} />
      )}
      {shownInvite && (
        <Accounts invite={shownInvite}
          onDisable={() => setDialog({ kind: "disable", filter: { invite: shownInvite.id }, label: t("ui.admin.invites.with_code_what", { code: shownInvite.code }) })} />
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
  const state = !r.open ? t("ui.admin.invites.closed") : r.paused ? (r.paused === "hour" ? t("ui.admin.invites.paused_hour") : t("ui.admin.invites.paused_day")) : r.invite ? t("ui.admin.invites.open_code") : t("ui.admin.invites.open_free");
  return (
    <div className="adm-tiles">
      <div className="adm-tile">
        <span className="adm-tile-label">{t("ui.admin.recent.accounts")}</span>
        <b>{state}</b>
        <span className="adm-tile-sub">{t("ui.admin.invites.counts", { hour: r.hour, per_hour: r.per_hour, day: r.day, per_day: r.per_day })}</span>
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
      <Button tone="ghost" onClick={() => void copyText(code).then(() => setCopied(true))}>
        {copied ? t("ui.admin.common.copied") : t("ui.admin.common.copy")}
      </Button>
    </span>
  );
}

/** The accounts registered with one code, as they are now, and switching off at once those still on. The button is
 * here rather than on the code's row so the list of codes fits the width of the settings' cards. */
function Accounts({ invite, onDisable }: { invite: Invite; onDisable: () => void }) {
  const state = useSignedIn();
  const columns: Column<Invite["accounts"][number]>[] = [
    { id: "username", label: t("ui.admin.auth.username"), cell: (a) => <span className="mono" data-user-data>{a.username}</span> },
    { id: "name", label: t("ui.admin.users.name"), cell: (a) => <span data-user-data>{a.name}</span> },
    { id: "at", label: t("ui.admin.invites.registered"), cell: (a) => whenText(a.at) },
    { id: "ip", label: t("ui.admin.security.col_source"), cell: (a) => <span className="mono" data-user-data>{a.ip}</span> },
    { id: "state", label: t("ui.admin.resources.state"), cell: (a) => (a.deleted ? t("ui.admin.invites.deleted") : a.enabled ? t("ui.admin.invites.on") : t("ui.admin.invites.disabled")) },
  ];
  return (
    <>
      <header className="set-part-head">
        <h3 className="adm-h3">{t("ui.admin.invites.with_code", { code: invite.code })}</h3>
        {shown(state?.applies, "registrations.disable") && (
          <Button tip={tipOf("disabled", why(state?.applies, "registrations.disable"))} tone="ghost" warn
            disabled={!usable(state?.applies, "registrations.disable") || !invite.accounts.some((a) => a.enabled && !a.deleted)}
            onClick={onDisable}>
            {t("ui.admin.invites.disable_registered")}
          </Button>
        )}
      </header>
      <Table rows={invite.accounts} columns={columns} rowKey={(a) => String(a.id)} dim={(a) => !a.enabled || !!a.deleted}
        empty={<Empty title={t("ui.admin.invites.nobody")} />} />
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
    code: !typed ? "" : typed.length < c.min || typed.length > c.max || !/^[A-Z0-9]+$/.test(typed) ? t("ui.admin.invites.bad_code", { min: c.min, max: c.max }) : "",
    uses: !uses.trim() ? "" : !/^\d+$/.test(uses.trim()) || +uses < 1 || +uses > c.uses_most ? t("ui.admin.invites.bad_uses", { most: c.uses_most }) : invite && +uses < invite.used ? t("ui.admin.invites.uses_below", { n: invite.used }) : "",
    expires: !expires || (invite?.expires && expires === dateValue(invite.expires)) ? "" : endOf(expires) <= now() ? t("ui.admin.users.expiry_past") : "",
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
    <Sheet title={invite ? t("ui.admin.invites.edit_title", { code: invite.code }) : t("ui.admin.invites.new")} width={520} onClose={onClose}>
      <div className="usr-form lgrid">
        {!invite && (
          <Row label={t("ui.admin.invites.code")} why={wrong.code}>
            <input className={`field${wrong.code ? " bad" : ""}`} value={code} autoFocus spellCheck={false} placeholder={t("ui.admin.invites.code_placeholder")}
              onChange={(e) => (setCode(e.target.value), setRefused(""))} />
          </Row>
        )}
        <Row label={t("ui.admin.invites.note")}>
          <input className="field" value={note} maxLength={c.note_most} placeholder={t("ui.admin.invites.note_placeholder")} onChange={(e) => (setNote(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.invites.uses")} why={wrong.uses}>
          <input className={`field${wrong.uses ? " bad" : ""}`} inputMode="numeric" value={uses} placeholder={t("ui.admin.invites.uses_placeholder")} onChange={(e) => (setUses(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.users.expiry")} why={wrong.expires}>
          <input className={`field${wrong.expires ? " bad" : ""}`} type="date" value={expires} min={dateValue(now())} onChange={(e) => (setExpires(e.target.value), setRefused(""))} />
        </Row>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={bad ? tipOf("disabled", t("ui.admin.users.fix_first")) : undefined} tone="primary" disabled={bad || busy} onClick={() => void submit()}>
            {busy ? t("ui.admin.common.saving") : invite ? t("ui.admin.common.save") : t("ui.admin.users.create")}
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
    { id: "username", label: t("ui.admin.auth.username"), cell: (r) => <span className="mono" data-user-data>{r.username}</span> },
    { id: "name", label: t("ui.admin.users.name"), cell: (r) => <span data-user-data>{r.name}</span> },
    { id: "at", label: t("ui.admin.invites.registered"), cell: (r) => whenText(r.registered) },
    { id: "invite", label: t("ui.admin.invites.code"), cell: (r) => (r.invite ? `${r.invite}…` : "—") },
    { id: "state", label: t("ui.admin.resources.state"), cell: (r) => (!r.enabled ? t("ui.admin.invites.disabled") : r.managed ? t("ui.admin.invites.will_disable") : t("ui.admin.invites.not_yours")) },
  ];
  return (
    <Sheet title={t("ui.admin.invites.disable_title", { what: label })} width={640} onClose={onClose}>
      <div className="usr-form lgrid">
        {byTime && (
          <>
            <Row label={t("ui.admin.usage.from")}>
              <input className="field" type="date" value={since} onChange={(e) => setSince(e.target.value)} />
            </Row>
            <Row label={t("ui.admin.usage.to")}>
              <input className="field" type="date" value={until} onChange={(e) => setUntil(e.target.value)} />
            </Row>
          </>
        )}
        <Table rows={rows} columns={columns} rowKey={(r) => String(r.id)} dim={(r) => !r.enabled || !r.managed} empty={<Empty title={data ? t("ui.admin.invites.no_match") : t("ui.admin.common.reading")} />} />
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={targets.length ? tipOf("consequence", t("ui.admin.invites.disable_tip")) : tipOf("disabled", t("ui.admin.invites.nothing_to_disable"))} tone="primary" danger
            disabled={busy || !targets.length || (byTime && !since && !until)} onClick={() => void go()}>
            {busy ? t("ui.admin.invites.disabling") : t("ui.admin.invites.disable_n", { n: targets.length })}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
