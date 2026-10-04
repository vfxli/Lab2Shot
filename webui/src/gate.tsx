import { messageOf, msg, type Message } from "./messages/message";
import { lazyRetry } from "./platform/lazyRetry";
import { Suspense, useEffect, useRef, useState } from "react";
import { deviceId } from "./platform/client";
import { COPYRIGHT, PROJECT_URL } from "./platform/brand";
import { ApiError, belongsTo, changedAccount, fromGate, json, loggedIn, NEED_LOGIN, NEED_TERMS, sawAccount } from "./platform/http";
import { BrandMark } from "./ui/icons";
import { Loading } from "./ui/Loading";
import { ErrorBoundary } from "./ui/ErrorBoundary";
import { whenText } from "./platform/format";
import { Button } from "./ui/Button";
import { RegisterForm, registerInfo, type RegisterInfo } from "./register";
import { TermsCard } from "./terms";
import { setLang, useLang } from "./i18n/lang";
import { t, useT } from "./i18n/t";
import { tipOf } from "./platform/tips";

/** The gate: all anyone gets before logging in (lab2shot/server/auth.py). It asks the server whether this browser is
 * logged in; if it is, it loads the page (site.tsx, a file the server hands out only then), if not it asks for the
 * username and password of the account the administrator made, and while the server lets people register themselves
 * it also offers 注册 (register.tsx). When the login runs out while a page is open (the
 * account was disabled or expired, the password changed elsewhere, or the account logged in on another browser:
 * one place online per account, per kind, lab2shot/accounts.py), the page stays as it is underneath and the login is
 * asked for over it, so nothing being edited is lost. An administrator who forgot the password sets a new one with the
 * server-side 口令 on the admin page's own login (admin/Auth.tsx) or from the command line, not here, where every other
 * user would read 「忘记密码」 as something they can do. An account that has not agreed to the current 用户协议 and
 * 隐私政策 (one an administrator made, or anyone after the texts changed: the login state's `terms`, lab2shot/terms)
 * is asked to before the page loads, or over it when the server refuses the page's requests for it (NEED_TERMS).
 * Kept small on purpose: no other part of the page's code is in here. */

const Site = lazyRetry(() => import("./site"));

/** What was in it, when the login this browser had was ended by another one (lab2shot/accounts.py, server/auth.py):
 * when, from where, what device. */
interface Kicked {
  at: number;
  ip: string;
  device: string;
  detail: string;
  kind?: string; // "embedded": a DCC plugin's embedded window whose plugin login ended (server/auth.py kicked_detail)
}



type Door = "asking" | "in" | "out" | "terms";

/** What the gate reads of a login state (the server's /api/auth/state, and the answer of logging in or registering). */
interface Said {
  user?: { id: number } | null;
  kicked?: Kicked | null;
  lang?: string; // what the server speaks to this browser (server/lang.py: the account's choice first)
  terms?: number | null; // the version of the 用户协议 and 隐私政策 this account must agree to first
}

const doorOf = (s: Said | null): Door => (!s?.user ? "out" : s.terms ? "terms" : "in");

export function Gate() {
  useLang((s) => s.lang); // the language changed: everything here renders again in it
  const [door, setDoor] = useState<Door>("asking");
  const [again, setAgain] = useState(false);
  const [owes, setOwes] = useState(false); // the page is open and the server asks for the agreement over it
  const [problem, setProblem] = useState<Message | null>(null);
  const [kicked, setKicked] = useState<Kicked | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const s = await fromGate(() => json<Said>("GET", "/api/auth/state", undefined, { cache: "no-store" }));
        belongsTo(s?.user?.id ?? null);
        setLang(s?.lang);
        if (alive) setDoor(doorOf(s));
        if (alive && s?.kicked) setKicked(s.kicked);
      } catch (e) {
        // the server answered but refused (its status said), or never answered at all
        if (alive) (setDoor("out"), setProblem(e instanceof ApiError ? msg("E-GATE-NOANSWER", { status: e.status }) : msg("E-GATE-UNREACHABLE")));
      }
    })();
    const need = (e: Event) => {
      setAgain(true);
      setKicked((e as CustomEvent<Kicked | null>).detail ?? null);
    };
    const terms = () => setOwes(true);
    window.addEventListener(NEED_LOGIN, need);
    window.addEventListener(NEED_TERMS, terms);
    return () => {
      alive = false;
      window.removeEventListener(NEED_LOGIN, need);
      window.removeEventListener(NEED_TERMS, terms);
    };
  }, []);

  if (door === "asking") return <Loading what={t("ui.gate.loading_state")} fill />;
  if (door === "terms") return <TermsPage over={false} onAgreed={() => setDoor("in")} />;
  return (
    <>
      {door === "in" && (
        <ErrorBoundary name={t("ui.gate.loading_page")}>
          <Suspense fallback={<Loading what={t("ui.gate.loading_page")} fill />}>
            <Site />
          </Suspense>
        </ErrorBoundary>
      )}
      {(door === "out" || again) && (
        <LoginPage
          over={door === "in"}
          problem={problem}
          kicked={kicked}
          onLoginAgain={() => setKicked(null)}
          onIn={(s) => {
            const id = s?.user?.id ?? null;
            changedAccount(id); // the other tabs of this browser follow (platform/http.ts)
            // another account logging in over the page (or one not known: null) gets a page of its own: nothing of the
            // one underneath (its graph, results, jobs, working copy) is carried over to it
            if (door === "in" && id === null) return window.location.reload();
            if (door === "in" && sawAccount(id)) return;
            if (door !== "in") belongsTo(id);
            if (door === "in") loggedIn(); // the page underneath goes on, as it was
            setAgain(false);
            setKicked(null);
            setProblem(null);
            if (door === "in") setOwes(!!s.terms);
            else setDoor(doorOf(s));
          }}
        />
      )}
      {door === "in" && owes && !again && (
        <TermsPage
          over
          onAgreed={() => {
            setOwes(false);
            loggedIn(); // the login state is read again, and the page's next requests go through
          }}
        />
      )}
    </>
  );
}

/** The login page's frame around the card that asks for the agreement (terms.tsx); 退出登录 ends the login and
 * starts again from the login page. */
function TermsPage({ over, onAgreed }: { over: boolean; onAgreed: () => void }) {
  const leave = () => void fromGate(() => json<unknown>("POST", "/api/auth/logout")).finally(() => (changedAccount(null), window.location.assign("/")));
  return (
    <div className={`login-page${over ? " over" : ""}`}>
      <main className="login-main">
        {!over && (
          <div className="login-mark" aria-hidden>
            <BrandMark hero />
          </div>
        )}
        <TermsCard over={over} onAgreed={onAgreed} onLeave={leave} />
      </main>
      {!over && <footer className="login-legal">{COPYRIGHT}</footer>}
    </div>
  );
}


/** The full-page notice for a browser whose login another login of the same account ended (one place online per
 * account, per kind): when, from where, what device. It never says "要重新登录" as if this browser had simply timed out,
 * so the viewer understands why. 重新登录 reveals the ordinary login form (submitting it, in turn, ends whichever place
 * is logged in now). */
function KickedNotice({ kicked, onLoginAgain }: { kicked: Kicked; onLoginAgain: () => void }) {
  // a DCC plugin's embedded window: it never offers a password login (one typed here would be a browser's and end the
  // account's own browser login); it is opened again from the DCC, with a new ticket
  if (kicked.kind === "embedded")
    return (
      <div className="login-card glass clear" role="alert">
        <h1>{t("ui.gate.embedded_out")}</h1>
        <p className="login-lede">{kicked.detail}</p>
      </div>
    );
  return (
    <div className="login-card glass clear" role="alert">
      <h1>{t("ui.gate.kicked_title")}</h1>
      <p className="login-lede">
        {t("ui.gate.kicked_body", { when: whenText(kicked.at), ip: kicked.ip, device: kicked.device })}
      </p>
      <div className="login-actions">
        <Button tip={tipOf("consequence", t("ui.gate.login_again_tip"))} tone="primary" type="button" onClick={onLoginAgain}>
          {t("ui.gate.login_again")}
        </Button>
      </div>
    </div>
  );
}

function LoginPage({
  over,
  problem: first,
  kicked,
  onLoginAgain,
  onIn,
}: {
  over: boolean;
  problem: Message | null;
  kicked: Kicked | null;
  onLoginAgain: () => void;
  onIn: (s: Said) => void;
}) {
  const [register, setRegister] = useState<RegisterInfo | null>(null); // 注册 is offered only while the server says it is open
  const [registering, setRegistering] = useState(false);
  useEffect(() => {
    if (!over) document.title = "Lab2Shot";
    if (!over) void registerInfo().then(setRegister);
  }, [over]);
  return (
    <div className={`login-page${over ? " over" : ""}`}>
      {!over && (
        <div className="login-scene" aria-hidden>
          {BOKEH.map((style, n) => (
            <i key={n} style={style} />
          ))}
        </div>
      )}
      <main className={`login-main${registering && register?.open ? " tall" : ""}`}>
        {!over && (
          <div className="login-mark" aria-hidden>
            <BrandMark hero />
          </div>
        )}
        {kicked ? (
          <KickedNotice kicked={kicked} onLoginAgain={onLoginAgain} />
        ) : registering && register?.open ? (
          <RegisterForm info={register} onIn={onIn} onBack={() => setRegistering(false)} />
        ) : (
          <LoginForm over={over} first={first} onIn={onIn} onRegister={register?.open ? () => setRegistering(true) : null} />
        )}
        {!over && <About />}
      </main>
      {!over && (
        <footer className="login-legal">
          {COPYRIGHT} <LanguageLink />
        </footer>
      )}
    </div>
  );
}

/** What Lab2Shot is, in a few lines under the login card, with a link to the project's page (opens in a new tab). */
function About() {
  const tr = useT(); // follows the language link beside the copyright
  return (
    <section className="login-about" aria-label={tr("ui.gate.about_link")}>
      <p>{tr("ui.gate.about")}</p>
      <a href={PROJECT_URL} target="_blank" rel="noopener noreferrer">{tr("ui.gate.about_link")}</a>
    </section>
  );
}

/** The login page's light (styles/login.css): many out-of-focus discs in the mark's colours, each with its own place,
 * size, brightness and pace. The same every time (a fixed seed): decoration, not state. */
const BOKEH: React.CSSProperties[] = (() => {
  let seed = 7;
  const r = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const tones = ["cyan", "blue", "magenta", "orange"];
  return Array.from({ length: 96 }, () => {
    const size = 2 + r() * r() * 24;
    const x = r();
    const y = r();
    const lit = 0.12 + 0.88 * ((x + y) / 2) ** 1.6; // dark at the top left, bright at the bottom right
    return {
      left: `${x * 110 - 5}%`,
      top: `${y * 110 - 5}%`,
      width: `${size}vmax`,
      height: `${size}vmax`,
      color: `var(--brand-${tones[Math.floor(r() * 4)]})`,
      opacity: (0.06 + r() * r() * 0.6) * lit,
      animationDuration: `${18 + r() * 40}s`,
      animationDelay: `${-r() * 40}s`,
    };
  });
})();

function LoginForm({ over, first, onIn, onRegister }: { over: boolean; first: Message | null; onIn: (s: Said) => void; onRegister: (() => void) | null }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [problem, setProblem] = useState(first);
  const [busy, setBusy] = useState(false);
  const field = useRef<HTMLInputElement>(null);
  useEffect(() => field.current?.focus(), []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true);
    try {
      onIn(await fromGate(() => json<Said>("POST", "/api/auth/login", { username: username.trim(), password, device_id: deviceId() })));
    } catch (err) {
      setProblem(messageOf(err));
      setPassword("");
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="login-card glass clear" aria-label={t("ui.gate.login")} onSubmit={(e) => void submit(e)}>
      {over && (
        <>
          <h1>{t("ui.gate.over_title")}</h1>
          <p className="login-lede">{t("ui.gate.over_body")}</p>
        </>
      )}
      <label className="login-field">
        <span>{t("ui.gate.username")}</span>
        <input
          ref={field}
          className="field lg"
          name="username"
          autoComplete="username"
          spellCheck={false}
          value={username}
          aria-invalid={!!problem}
          onChange={(e) => (setUsername(e.target.value), setProblem(null))}
        />
      </label>
      <label className="login-field">
        <span>{t("ui.gate.password")}</span>
        <input
          className="field lg"
          type="password"
          name="password"
          autoComplete="current-password"
          value={password}
          aria-invalid={!!problem}
          onChange={(e) => (setPassword(e.target.value), setProblem(null))}
        />
      </label>
      {problem && (
        <p className="login-problem" role="alert" data-code={problem.code}>
          {problem.text}
        </p>
      )}
      <div className="login-stack">
        <Button tip={username && password ? undefined : tipOf("disabled", t("ui.gate.need_both"))} tone="primary" size="lg" layout="login-go" type="submit" disabled={!username.trim() || !password || busy}>
          {busy ? t("ui.gate.logging_in") : t("ui.gate.login")}
        </Button>
        {onRegister && (
          <Button tone="ghost" type="button" onClick={onRegister}>
            {t("ui.gate.register")}
          </Button>
        )}
      </div>
    </form>
  );
}

/** The login page's language switch (before a login there is no account to keep it: this browser's cookie does,
 * server/lang.py; after it, the account menu's). It names the other language in that language. */
function LanguageLink() {
  const lang = useLang((s) => s.lang);
  const other = lang === "zh" ? "en" : "zh";
  return (
    <Button tone="ghost" layout="login-lang" onClick={() => setLang(other, true)}>
      {t(`lang.${other}`)}
    </Button>
  );
}
