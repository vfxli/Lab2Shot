import { useEffect, useRef, useState } from "react";
import { api, type Account, type AuthState } from "../api";
import { BrandMark } from "../ui/icons";
import { Loading } from "../ui/Loading";
import { pageAccess } from "../api/applies";
import { useSession, useSessionWatch, useSignedIn } from "../state/session";
import { Button, ButtonLink, Segmented } from "../ui/Button";
import { MIN_CHARS, passwordProblem } from "../platform/accountRules";
import { changedAccount, sawAccount } from "../platform/http";

/** Gate for the admin pages (/admin; lab2shot/server/auth.py). The user is already logged in through the site
 * gate; whether a page is available to this login is decided by the server (applies.ts pageAccess): open, a
 * password prompt (the resulting rights last three days, lab2shot/accounts.py ADMIN_S), or a notice for accounts whose
 * role has no access. When the site login itself expires or ends elsewhere, the site gate (gate.tsx) asks for it over
 * the page, so the current section and any unsaved setting are kept; when only the administrator rights run out, this
 * gate's password prompt takes the page's place. The administrator's 口令 is configured on the server only and is never
 * displayed. */

const ADMIN_NAME = "admin"; // lab2shot/accounts.py ADMIN_NAME: the built-in administrator of a new installation.

export const PASSPHRASE_TIP =
  "「我的口令」是主人自己的备用钥匙：只能在服务器上执行 uv run lab2shot admin passphrase 设置，网页上永远看不到、也改不了。" +
  "忘了管理员密码时用它设一个新密码；在服务器上执行 uv run lab2shot admin password 也能重设密码。";


const auth = api.auth;

/** Renders the children once the page is available to this login; otherwise the login form, or a notice for an
 * account without access. `what`: the guarded page; `page`: its id in the server's availability map (page.admin). */
export function AdminGate({ what, page, children }: { what: string; page: string; children: React.ReactNode }) {
  useSessionWatch();
  const state = useSession((s) => s.state);
  const set = useSession((s) => s.set);
  if (!state) return <Loading what="登录状态" fill />;
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
        Lab2Shot 管理
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
        <section className="login-card login-plate" aria-label="只给管理员">
          <h1>这个页面只给管理员用</h1>
          <p className="login-lede">
            {what}只有管理员能打开。当前登录的是 {user.name}（{user.username}），编辑器照常能用。
          </p>
          <div className="login-actions end">
            <ButtonLink tip="回到节点编辑器" tone="primary" href="/">
              回到编辑器
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
    document.title = "Lab2Shot 管理 · 登录";
  }, []);
  return (
    <div className="login-page">
      <Top />
      <main className="login-main">
        <section className="login-card login-plate" aria-label={forgot ? "用口令设新密码" : "登录"}>
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
      <h1>{known ? "再输一次密码" : "管理登录"}</h1>
      <p className="login-lede">
        {what}要用有管理权限的账号。{known ? "管理权限三天后要再输一次密码；" : ""}已经登录的编辑器不受影响。
      </p>
      <label className="login-field">
        <span data-tip="管理员或二级管理员的用户名">用户名</span>
        <input
          className="field"
          name="username"
          autoComplete="username"
          value={username}
          readOnly={!!known}
          data-tip="管理员或二级管理员的用户名"
          onChange={(e) => (setUsername(e.target.value), setProblem(""))}
        />
      </label>
      <label className="login-field">
        <span data-tip="管理员的密码。忘了可以点下面的「忘记密码」用口令重设，或在服务器上执行 lab2shot admin password">密码</span>
        <input
          ref={field}
          className="field"
          type="password"
          name="password"
          autoComplete="current-password"
          value={password}
          aria-invalid={!!problem}
          data-tip="输入密码，按回车登录"
          onChange={(e) => (setPassword(e.target.value), setProblem(""))}
        />
      </label>
      {problem && (
        <p className="login-problem" role="alert">
          {problem}
        </p>
      )}
      <div className="login-actions">
        <Button tip={`用「我的口令」设一个新的管理员密码。${PASSPHRASE_TIP}`} tone="ghost" type="button" onClick={onForgot}>
          忘记密码
        </Button>
        <Button tip={password ? "登录，三天内这个浏览器不用再输" : "先输入密码"} tone="primary" type="submit" disabled={!username.trim() || !password || busy}>
          {busy ? "登录中…" : "登录"}
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
      <h1>用口令设新密码</h1>
      <p className="login-lede" data-tip={PASSPHRASE_TIP}>
        输入服务器上设好的「我的口令」，再给管理员设一个新密码。设好后管理员在别处的登录都会退出，这个浏览器随即登录。
      </p>
      <label className="login-field">
        <span data-tip={PASSPHRASE_TIP}>我的口令</span>
        <input className="field" type="password" autoComplete="off" value={phrase} autoFocus data-tip={PASSPHRASE_TIP} onChange={(e) => (setPhrase(e.target.value), setProblem(""))} />
      </label>
      <NewPassword next={next} again={again} onNext={setNext} onAgain={setAgain} />
      {(rule || problem) && (
        <p className="login-problem" role="alert">
          {rule || problem}
        </p>
      )}
      <div className="login-actions">
        <Button tip="回到用密码登录" tone="ghost" type="button" onClick={onBack}>
          返回
        </Button>
        <Button tip={ready ? "用口令设新密码并登录" : "先填好口令和两次一样的新密码"} tone="primary" type="submit" disabled={!ready || busy}>
          {busy ? "设置中…" : "设新密码并登录"}
        </Button>
      </div>
    </form>
  );
}

function NewPassword({ next, again, onNext, onAgain }: { next: string; again: string; onNext: (v: string) => void; onAgain: (v: string) => void }) {
  return (
    <>
      <label className="login-field">
        <span data-tip={`新的管理员密码，至少 ${MIN_CHARS} 个字符；字母、数字、符号、中文都可以`}>新密码</span>
        <input className="field" type="password" autoComplete="new-password" value={next} data-tip={`至少 ${MIN_CHARS} 个字符`} onChange={(e) => onNext(e.target.value)} />
      </label>
      <label className="login-field">
        <span data-tip="再输一次新密码，防止输错">再输一次</span>
        <input className="field" type="password" autoComplete="new-password" value={again} data-tip="和上面的新密码一样" onChange={(e) => onAgain(e.target.value)} />
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
    <form className="set-card pw-card" data-group="password" onSubmit={(e) => void submit(e)}>
      <h3>管理员密码</h3>
      <div className="set-row">
        <span className="set-label" data-tip={`改密码要先验明身份：输入现在的密码，或者「我的口令」。${PASSPHRASE_TIP}`}>
          用什么证明
        </span>
        <span className="set-ctl">
          <Segmented label="用什么证明" value={how} options={[{ value: "current", label: "现在的密码", tip: "输入现在的管理员密码" }, { value: "passphrase", label: "我的口令", tip: `输入「我的口令」。${PASSPHRASE_TIP}`, disabled: state.passphrase ? false : `还没有设口令。${PASSPHRASE_TIP}` }]} onChange={(h) => (setHow(h), setProof(""), setProblem(""))} />
        </span>
        <span />
      </div>
      <label className="set-row">
        <span className="set-label" data-tip={how === "current" ? "现在的管理员密码" : PASSPHRASE_TIP}>
          {how === "current" ? "现在的密码" : "我的口令"}
        </span>
        <span className="set-ctl">
          <input className="field" type="password" autoComplete={how === "current" ? "current-password" : "off"} value={proof} aria-label={how === "current" ? "现在的密码" : "我的口令"} data-tip={how === "current" ? "现在的管理员密码" : "服务器上设好的口令"} onChange={(e) => (setProof(e.target.value), setProblem(""), setDone(false))} />
        </span>
        <span />
      </label>
      <label className="set-row">
        <span className="set-label" data-tip={`新的管理员密码，至少 ${MIN_CHARS} 个字符`}>
          新密码
        </span>
        <span className="set-ctl">
          <input className="field" type="password" autoComplete="new-password" value={next} aria-label="新密码" data-tip={`至少 ${MIN_CHARS} 个字符`} onChange={(e) => (setNext(e.target.value), setDone(false))} />
        </span>
        <span />
      </label>
      <label className="set-row">
        <span className="set-label" data-tip="再输一次新密码，防止输错">
          再输一次
        </span>
        <span className="set-ctl">
          <input className="field" type="password" autoComplete="new-password" value={again} aria-label="再输一次新密码" data-tip="和上面的新密码一样" onChange={(e) => (setAgain(e.target.value), setDone(false))} />
        </span>
        <span />
      </label>
      {(rule || problem) && (
        <p className="set-why bad" role="alert">
          {rule || problem}
        </p>
      )}
      {done && <p className="set-why ok">密码已改好：管理员在别处的登录都已退出，这个浏览器继续登录着。</p>}
      <div className="set-row">
        <span className="set-label" />
        <span className="set-ctl">
          <Button tip={ready ? "改成新密码；管理员在别处的登录都要重新登录" : "先填好证明和两次一样的新密码"} tone="primary" type="submit" disabled={!ready || busy}>
            {busy ? "修改中…" : "改密码"}
          </Button>
          <span className="pw-hint" data-tip={PASSPHRASE_TIP}>
            口令只能在服务器上设
          </span>
        </span>
      </div>
    </form>
  );
}
