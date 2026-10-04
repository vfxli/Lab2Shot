import { useEffect, useRef, useState } from "react";
import { LabelRow } from "../ui/LabelRow";
import { api, type Account, type AuthState } from "../api";
import { BrandMark } from "../ui/icons";
import { Loading } from "../ui/Loading";
import { pageAccess } from "../api/applies";
import { useSession, useSessionWatch, useSignedIn } from "../state/session";
import { Button, ButtonLink, Segmented } from "../ui/Button";
import { passwordProblem } from "../platform/accountRules";
import { changedAccount, sawAccount } from "../platform/http";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** Gate for the admin pages (/admin; lab2shot/server/auth.py). The user is already logged in through the site
 * gate; whether a page is available to this login is decided by the server (applies.ts pageAccess): open, a
 * password prompt (the resulting rights last three days, lab2shot/accounts.py ADMIN_S), or a notice for accounts whose
 * role has no access. When the site login itself expires or ends elsewhere, the site gate (gate.tsx) asks for it over
 * the page, so the current section and any unsaved setting are kept; when only the administrator rights run out, this
 * gate's password prompt takes the page's place. The administrator's 口令 is configured on the server only and is never
 * displayed. */

const ADMIN_NAME = "admin"; // lab2shot/accounts.py ADMIN_NAME: the built-in administrator of a new installation.

/** Why 「我的口令」 cannot be chosen: it is set on the server only. */
const passphraseUnset = () => t("ui.admin.auth.passphrase_unset");


const auth = api.auth;

/** Renders the children once the page is available to this login; otherwise the login form, or a notice for an
 * account without access. `what`: the guarded page; `page`: its id in the server's availability map (page.admin). */
export function AdminGate({ what, page, children }: { what: string; page: string; children: React.ReactNode }) {
  useSessionWatch();
  const state = useSession((s) => s.state);
  const set = useSession((s) => s.set);
  if (!state) return <Loading what={t("ui.admin.auth.state")} fill />;
  const access = pageAccess(state, page);
  if (access === "not-yours" && state.user) return <NotYours what={what} user={state.user} />;
  if (access !== "open") return <LoginPage what={what} known={state.user} onIn={(s) => signedIn(s, set)} />;
  return <>{children}</>;
}

/** A login on the admin page's own form (the password again, or a new one with the 口令): the other tabs of this browser
 * follow it, and a login of another account than the page's opens the page again (platform/http.ts). */
function signedIn(s: AuthState, set: (s: AuthState) => void): void {
  const id = s.user?.id ?? null;
  changedAccount(id);
  if (!sawAccount(id)) set(s);
}

function Top() {
  return (
    <header className="adm-top">
      <span className="help-brand">
        <BrandMark />
        {t("ui.admin.page.title")}
      </span>
    </header>
  );
}

/** Shown when another account opens an administrator page: no content, only a way back. */
function NotYours({ what, user }: { what: string; user: Account }) {
  useEffect(() => {
    document.title = "Lab2Shot";
  }, []);
  return (
    <div className="login-page">
      <Top />
      <main className="login-main">
        <section className="login-card login-plate" aria-label={t("ui.admin.auth.admins_only")}>
          <h1>{t("ui.admin.auth.not_yours_title")}</h1>
          <p className="login-lede">
            {t("ui.admin.auth.not_yours", { what, name: user.name, username: user.username })}
          </p>
          <div className="login-actions end">
            <ButtonLink tone="primary" href="/">
              {t("ui.admin.auth.back_to_editor")}
            </ButtonLink>
          </div>
        </section>
      </main>
    </div>
  );
}

function LoginPage({ what, known, onIn }: { what: string; known: Account | null; onIn: (s: AuthState) => void }) {
  const [forgot, setForgot] = useState(false);
  useEffect(() => {
    document.title = t("ui.admin.auth.page_title");
  }, []);
  return (
    <div className="login-page">
      <Top />
      <main className="login-main">
        <section className="login-card login-plate" aria-label={forgot ? t("ui.admin.auth.reset_title") : t("ui.admin.auth.login")}>
          {forgot ? <ResetForm onIn={onIn} onBack={() => setForgot(false)} /> : <LoginForm what={what} known={known} onIn={onIn} onForgot={() => setForgot(true)} />}
        </section>
      </main>
    </div>
  );
}

function LoginForm({ what, known, onIn, onForgot }: { what: string; known: Account | null; onIn: (s: AuthState) => void; onForgot: () => void }) {
  const [username, setUsername] = useState(known ? known.username : ADMIN_NAME);
  const [password, setPassword] = useState("");
  const [problem, setProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => field.current?.focus(), []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    try {
      onIn(await auth.login(username.trim(), password)); // For another account, the gate reports that the page is unavailable.
    } catch (err) {
      setProblem((err as Error).message);
      setPassword("");
      field.current?.focus();
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={(e) => void submit(e)}>
      <h1>{known ? t("ui.admin.auth.again_title") : t("ui.admin.auth.login_title")}</h1>
      <p className="login-lede">
        {known ? t("ui.admin.auth.lede_again", { what }) : t("ui.admin.auth.lede", { what })}
      </p>
      <label className="login-field">
        <span>{t("ui.admin.auth.username")}</span>
        <input
          className="field"
          name="username"
          autoComplete="username"
          value={username}
          readOnly={!!known}
          onChange={(e) => (setUsername(e.target.value), setProblem(""))}
        />
      </label>
      <label className="login-field">
        <span>{t("ui.admin.auth.password")}</span>
        <input
          ref={field}
          className="field"
          type="password"
          name="password"
          autoComplete="current-password"
          value={password}
          aria-invalid={!!problem}
          onChange={(e) => (setPassword(e.target.value), setProblem(""))}
        />
      </label>
      {problem && (
        <p className="login-problem" role="alert">
          {problem}
        </p>
      )}
      <div className="login-actions">
        <Button tone="ghost" type="button" onClick={onForgot}>
          {t("ui.admin.auth.forgot")}
        </Button>
        <Button tip={password ? undefined : tipOf("disabled", t("ui.admin.auth.password_first"))} tone="primary" type="submit" disabled={!username.trim() || !password || busy}>
          {busy ? t("ui.admin.auth.logging_in") : t("ui.admin.auth.login")}
        </Button>
      </div>
    </form>
  );
}

function ResetForm({ onIn, onBack }: { onIn: (s: AuthState) => void; onBack: () => void }) {
  const [phrase, setPhrase] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [problem, setProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const rule = passwordProblem(next, again);
  const ready = !!phrase && !!next && next === again && !rule;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    try {
      onIn(await auth.recover(phrase, next));
    } catch (err) {
      setProblem((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form onSubmit={(e) => void submit(e)}>
      <h1>{t("ui.admin.auth.reset_title")}</h1>
      <p className="login-lede">
        {t("ui.admin.auth.reset_lede")}
      </p>
      <label className="login-field">
        <span>{t("ui.admin.auth.passphrase")}</span>
        <input className="field" type="password" autoComplete="off" value={phrase} autoFocus onChange={(e) => (setPhrase(e.target.value), setProblem(""))} />
      </label>
      <NewPassword next={next} again={again} onNext={setNext} onAgain={setAgain} />
      {(rule || problem) && (
        <p className="login-problem" role="alert">
          {rule || problem}
        </p>
      )}
      <div className="login-actions">
        <Button tone="ghost" type="button" onClick={onBack}>
          {t("ui.admin.auth.back")}
        </Button>
        <Button tip={ready ? undefined : tipOf("disabled", t("ui.admin.auth.reset_first"))} tone="primary" type="submit" disabled={!ready || busy}>
          {busy ? t("ui.admin.auth.setting") : t("ui.admin.auth.reset_go")}
        </Button>
      </div>
    </form>
  );
}

function NewPassword({ next, again, onNext, onAgain }: { next: string; again: string; onNext: (v: string) => void; onAgain: (v: string) => void }) {
  return (
    <>
      <label className="login-field">
        <span>{t("ui.admin.auth.new_password")}</span>
        <input className="field" type="password" autoComplete="new-password" value={next} onChange={(e) => onNext(e.target.value)} />
      </label>
      <label className="login-field">
        <span>{t("ui.admin.auth.again")}</span>
        <input className="field" type="password" autoComplete="new-password" value={again} onChange={(e) => onAgain(e.target.value)} />
      </label>
    </>
  );
}

/** 账号设置 → 管理员密码: changes the administrator's password, authorised by the current password or the server-side 口令 (「我的口令」). */
export function PasswordCard() {
  const state = useSignedIn();
  const set = useSession((s) => s.set);
  const [how, setHow] = useState<"current" | "passphrase">("current");
  const [proof, setProof] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [problem, setProblem] = useState("");
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  const rule = passwordProblem(next, again);
  const ready = !!proof && !!next && next === again && !rule;

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    try {
      set(how === "current" ? await auth.change(proof, next) : await auth.recover(proof, next));
      setProof("");
      setNext("");
      setAgain("");
      setProblem("");
      setDone(true);
    } catch (err) {
      setProblem((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="set-card pw-card lgrid" data-group="password" onSubmit={(e) => void submit(e)}>
      <h3>{t("ui.admin.auth.card_title")}</h3>
      <LabelRow className="set-row" labelClass="set-label" label={t("ui.admin.auth.proof")} ctlClass="set-ctl">
        <Segmented label={t("ui.admin.auth.proof")} value={how} options={[{ value: "current", label: t("ui.admin.auth.current") }, { value: "passphrase", label: t("ui.admin.auth.passphrase"), disabled: state.passphrase ? false : passphraseUnset() }]} onChange={(h) => (setHow(h), setProof(""), setProblem(""))} />
      </LabelRow>
      <LabelRow className="set-row" labelAs="label" labelClass="set-label" label={how === "current" ? t("ui.admin.auth.current") : t("ui.admin.auth.passphrase")} ctlClass="set-ctl">
        <input className="field" type="password" autoComplete={how === "current" ? "current-password" : "off"} value={proof} aria-label={how === "current" ? t("ui.admin.auth.current") : t("ui.admin.auth.passphrase")} onChange={(e) => (setProof(e.target.value), setProblem(""), setDone(false))} />
      </LabelRow>
      <LabelRow className="set-row" labelAs="label" labelClass="set-label" label={t("ui.admin.auth.new_password")} ctlClass="set-ctl">
        <input className="field" type="password" autoComplete="new-password" value={next} aria-label={t("ui.admin.auth.new_password")} onChange={(e) => (setNext(e.target.value), setDone(false))} />
      </LabelRow>
      <LabelRow className="set-row" labelAs="label" labelClass="set-label" label={t("ui.admin.auth.again")} ctlClass="set-ctl">
        <input className="field" type="password" autoComplete="new-password" value={again} aria-label={t("ui.admin.auth.again_label")} onChange={(e) => (setAgain(e.target.value), setDone(false))} />
      </LabelRow>
      {(rule || problem) && (
        <p className="set-why bad lrow-under" role="alert">
          {rule || problem}
        </p>
      )}
      {done && <p className="set-why ok lrow-under">{t("ui.admin.auth.changed")}</p>}
      <LabelRow className="set-row" labelClass="set-label" label="" ctlClass="set-ctl">
        <Button tip={ready ? tipOf("consequence", t("ui.admin.auth.change_tip")) : tipOf("disabled", t("ui.admin.auth.change_first"))} tone="primary" type="submit" disabled={!ready || busy}>
            {busy ? t("ui.admin.auth.changing") : t("ui.admin.auth.change")}
          </Button>
          <span className="pw-hint">
            {t("ui.admin.auth.passphrase_hint")}
          </span>
      </LabelRow>
    </form>
  );
}
