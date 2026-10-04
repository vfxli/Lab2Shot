/** 用户's dialogs: 新建用户, 修改, 重设密码, 删除, 永久删除 (what each does is the server's: lab2shot/accounts.py). */

import { useState } from "react";
import { Sheet } from "../ui/Sheet";
import { adminApi, type UserChange, type UserRow, type UsersView } from "../api/admin";
import { nameProblem, passwordProblem, tidyName, usernameProblem } from "../platform/accountRules";
import { shown, usable, why, type Availability } from "../api/applies";
import { Button } from "../ui/Button";
import { DAY_S, dateValue, endOf, now } from "./Users";
import { Roles, Row, Tags } from "./userFields";
import { StageSelect } from "../ui/StageSelect";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

export function NewUserDialog({ view, a, onClose, onMade }: { view: UsersView; a: Availability | null; onClose: () => void; onMade: (v: UsersView, made: { username: string; password: string }) => void }) {
  const [pick, setPick] = useState(view.default_role);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [name, setName] = useState("");
  const [dept, setDept] = useState("");
  const [expires, setExpires] = useState(dateValue(now() + 30 * DAY_S));
  const [allowed, setAllowed] = useState<string[]>(view.allowed_new);
  const [touched, setTouched] = useState(false);
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const wrong = {
    username: username ? usernameProblem(username) : t("ui.admin.users.need_username"),
    password: password ? passwordProblem(password, again) : t("ui.admin.users.need_password"),
    again: password && again !== password ? t("ui.admin.users.mismatch") : "",
    name: nameProblem(name),
    dept: dept ? "" : t("ui.admin.users.need_department"),
    expires: !expires ? t("ui.admin.users.need_expiry") : endOf(expires) <= now() ? t("ui.admin.users.expiry_past") : "",
  };
  const bad = Object.values(wrong).some(Boolean);

  const submit = async () => {
    setTouched(true);
    if (bad || busy) return;
    setBusy(true);
    try {
      const r = await adminApi.createUser({
        username,
        password,
        name: tidyName(name),
        department: dept,
        expires: endOf(expires),
        ...(usable(a, "users.tags") ? { tags: allowed } : {}), // otherwise the server gives a new account the default
        ...(usable(a, "users.role") ? { role: pick } : {}),
      });
      onMade(r, { username: r.user.username, password });
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const show = (k: keyof typeof wrong) => (touched ? wrong[k] : "");

  return (
    <Sheet title={t("ui.admin.users.new_title")} width={560} onClose={onClose}>
      <div className="usr-form lgrid">
        <Row label={t("ui.admin.auth.username")} why={show("username")}>
          <input className={`field${show("username") ? " bad" : ""}`} value={username} autoFocus spellCheck={false} placeholder={t("ui.admin.users.username_placeholder")}
            onChange={(e) => (setUsername(e.target.value.toLowerCase().trim()), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.users.initial_password")} why={show("password")}>
          <input className={`field${show("password") ? " bad" : ""}`} type="password" autoComplete="new-password" value={password} placeholder={t("ui.admin.users.password_placeholder")} onChange={(e) => (setPassword(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.auth.again")} why={show("again")}>
          <input className={`field${show("again") ? " bad" : ""}`} type="password" autoComplete="new-password" value={again} onChange={(e) => setAgain(e.target.value)} />
        </Row>
        <Row label={t("ui.admin.users.name")} why={show("name")}>
          <input className={`field${show("name") ? " bad" : ""}`} value={name} placeholder={t("ui.admin.users.name_placeholder")} onChange={(e) => (setName(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.users.department")} why={show("dept")}>
          <StageSelect list={view.departments} value={dept} bad={!!show("dept")} onPick={(d) => (setDept(d), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.users.expiry")} why={show("expires")}>
          <input className={`field${show("expires") ? " bad" : ""}`} type="date" value={expires} min={dateValue(now())} onChange={(e) => setExpires(e.target.value)} />
        </Row>
        {usable(a, "users.role") && (
          <Row label={t("ui.admin.users.role")}>
            <Roles list={view.roles} value={pick} onPick={setPick} />
          </Row>
        )}
        {usable(a, "users.tags") && (
          <Row label={t("ui.admin.users.allowed")}>
            <Tags view={view} value={allowed} onChange={setAllowed} />
          </Row>
        )}
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={bad ? tipOf("disabled", t("ui.admin.users.fix_first")) : undefined} tone="primary" disabled={busy || (touched && bad)} onClick={() => void submit()}>
            {busy ? t("ui.admin.users.creating") : t("ui.admin.users.create")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function EditDialog({ view, u, onClose, onDone }: { view: UsersView; u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [name, setName] = useState(u.name);
  const [dept, setDept] = useState(u.department);
  const [expires, setExpires] = useState(u.expires ? dateValue(u.expires) : "");
  const [allowed, setAllowed] = useState<string[]>(u.tags);
  const [pick, setPick] = useState(u.role);
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const wrong = {
    name: nameProblem(name),
    expires: !usable(u.applies, "account.expiry") ? "" : !expires ? t("ui.admin.users.need_expiry") : "",
  };
  const bad = Object.values(wrong).some(Boolean);

  const submit = async () => {
    if (bad || busy) return;
    setBusy(true);
    const change: UserChange = { name: tidyName(name), department: dept };
    if (usable(u.applies, "account.expiry")) change.expires = endOf(expires);
    if (usable(u.applies, "account.tags")) change.tags = allowed;
    if (usable(u.applies, "account.role")) change.role = pick;
    try {
      onDone(await adminApi.changeUser(u.id, change));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={t("ui.admin.users.edit_title", { username: u.username })} width={560} onClose={onClose}>
      <div className="usr-form lgrid">
        <Row label={t("ui.admin.users.name")} why={wrong.name}>
          <input className={`field${wrong.name ? " bad" : ""}`} value={name} autoFocus onChange={(e) => (setName(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.users.department")} tip={tipOf("consequence", t("ui.admin.users.department_tip"))}>
          <StageSelect list={view.departments} value={dept} onPick={(d) => (setDept(d), setRefused(""))} />
        </Row>
        {shown(u.applies, "account.role") && (
          <Row label={t("ui.admin.users.role")}>
            {usable(u.applies, "account.role") ? <Roles list={view.roles} value={pick} onPick={setPick} /> : <span className="usr-static" {...tipAttrs(tipOf("disabled", why(u.applies, "account.role")))}>{u.role_label}</span>}
          </Row>
        )}
        {shown(u.applies, "account.expiry") && (
          <Row label={t("ui.admin.users.expiry")} why={wrong.expires}>
            {usable(u.applies, "account.expiry") ? (
              <input className={`field${wrong.expires ? " bad" : ""}`} type="date" value={expires} onChange={(e) => (setExpires(e.target.value), setRefused(""))} />
            ) : (
              <span className="usr-static" {...tipAttrs(tipOf("disabled", why(u.applies, "account.expiry")))}>{t("ui.admin.users.no_expiry")}</span>
            )}
          </Row>
        )}
        {shown(u.applies, "account.tags") && (
          <Row label={t("ui.admin.users.allowed")}>
            {usable(u.applies, "account.tags") ? <Tags view={view} value={allowed} onChange={setAllowed} /> : <span className="usr-static" {...tipAttrs(tipOf("disabled", why(u.applies, "account.tags")))}>{t("ui.admin.resources.all")}</span>}
          </Row>
        )}
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={bad ? tipOf("disabled", t("ui.admin.users.fix_first")) : tipOf("consequence", t("ui.admin.users.save_tip"))} tone="primary" disabled={bad || busy} onClick={() => void submit()}>
            {busy ? t("ui.admin.common.saving") : t("ui.admin.common.save")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function ResetDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView, made: { username: string; password: string }) => void }) {
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const rule = passwordProblem(next, again);
  const ready = !!next && next === again && !rule;

  const submit = async () => {
    if (!ready || busy) return;
    setBusy(true);
    try {
      onDone(await adminApi.resetPassword(u.id, next), { username: u.username, password: next });
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={t("ui.admin.users.reset_title", { username: u.username })} width={480} onClose={onClose}>
      <div className="usr-form lgrid">
        <p className="tpl-desc">{t("ui.admin.users.reset_lede")}</p>
        <Row label={t("ui.admin.auth.new_password")}>
          <input className="field" type="password" autoComplete="new-password" value={next} autoFocus onChange={(e) => (setNext(e.target.value), setRefused(""))} />
        </Row>
        <Row label={t("ui.admin.auth.again")} why={rule || refused}>
          <input className="field" type="password" autoComplete="new-password" value={again} onChange={(e) => (setAgain(e.target.value), setRefused(""))} />
        </Row>
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={ready ? undefined : tipOf("disabled", t("ui.admin.users.reset_first"))} tone="primary" disabled={!ready || busy} onClick={() => void submit()}>
            {busy ? t("ui.admin.auth.setting") : t("ui.admin.users.reset_go")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

/** 永久删除 takes two steps: 删除 first moves the account to 已删除, and 永久删除 is done from there, so no single
 * slip is irreversible. What is kept and what is removed is decided by the server alone (lab2shot/accounts.py purge);
 * this dialog only says it. */
export function PurgeDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      onDone(await adminApi.purgeUser(u.id));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={t("ui.admin.users.purge_title", { username: u.username })} width={480} onClose={onClose}>
      <div className="usr-form lgrid">
        <p className="tpl-desc">
          {t("ui.admin.users.purge_lede", { name: u.name, username: u.username })}
        </p>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tone="primary" danger disabled={busy} onClick={() => void submit()}>
            {busy ? t("ui.admin.users.deleting") : t("ui.admin.users.purge")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function DeleteDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      onDone(await adminApi.deleteUser(u.id));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={t("ui.admin.users.delete_title", { username: u.username })} width={480} onClose={onClose}>
      <div className="usr-form lgrid">
        <p className="tpl-desc">
          {t("ui.admin.users.delete_lede", { name: u.name, username: u.username })}
        </p>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tone="primary" danger disabled={busy} onClick={() => void submit()}>
            {busy ? t("ui.admin.users.deleting") : t("ui.admin.user.delete")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
