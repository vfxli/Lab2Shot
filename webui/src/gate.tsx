import { messageOf, msg, type Message } from "./messages/message";
import { lazyRetry } from "./platform/lazyRetry";
import { Suspense, useEffect, useRef, useState } from "react";
import { deviceId } from "./platform/client";
import { COPYRIGHT } from "./platform/brand";
import { ApiError, belongsTo, changedAccount, fromGate, json, loggedIn, NEED_LOGIN, NEED_TERMS, sawAccount } from "./platform/http";
import { BrandMark } from "./ui/icons";
import { Loading } from "./ui/Loading";
import { whenText } from "./platform/format";
import { Button } from "./ui/Button";
import { RegisterForm, registerInfo, type RegisterInfo } from "./register";
import { TermsCard } from "./terms";

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
}



type Door = "asking" | "in" | "out" | "terms";

/** What the gate reads of a login state (the server's /api/auth/state, and the answer of logging in or registering). */
interface Said {
  user?: { id: number } | null;
  kicked?: Kicked | null;
  terms?: number | null; // the version of the 用户协议 and 隐私政策 this account must agree to first
}

const doorOf = (s: Said | null): Door => (!s?.user ? "out" : s.terms ? "terms" : "in");

export function Gate() {
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

  if (door === "asking") return <Loading what="登录状态" fill />;
  if (door === "terms") return <TermsPage over={false} onAgreed={() => setDoor("in")} />;
  return (
    <>
      {door === "in" && (
        <Suspense fallback={<Loading what="页面" fill />}>
          <Site />
        </Suspense>
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
  return (
    <div className="login-card glass clear" role="alert">
      <h1>这个账号在别处登录了</h1>
      <p className="login-lede">
        {whenText(kicked.at)}，{kicked.ip}，{kicked.device}。同一个账号同时只能有一处在线；不是本人操作的，先改密码。
      </p>
      <div className="login-actions">
        <Button tip="重新登录：会顶掉刚才那一处的登录" tone="primary" type="button" onClick={onLoginAgain}>
          重新登录
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
      </main>
      {!over && <footer className="login-legal">{COPYRIGHT}</footer>}
    </div>
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
    <form className="login-card glass clear" aria-label="登录" onSubmit={(e) => void submit(e)}>
      {over && (
        <>
          <h1>要重新登录</h1>
          <p className="login-lede">登录过期了，或者账号在别处改了密码、被停用。重新登录就能接着用，页面上正在做的都还在。</p>
        </>
      )}
      <label className="login-field">
        <span data-tip="管理员在管理页面「用户」里建好的用户名，或者自己注册的用户名">用户名</span>
        <input
          ref={field}
          className="field lg"
          name="username"
          autoComplete="username"
          spellCheck={false}
          value={username}
          aria-invalid={!!problem}
          data-tip="用户名，小写字母开头"
          onChange={(e) => (setUsername(e.target.value), setProblem(null))}
        />
      </label>
      <label className="login-field">
        <span data-tip="密码：第一次的密码向管理员要，登录后可以在右上角自己改">密码</span>
        <input
          className="field lg"
          type="password"
          name="password"
          autoComplete="current-password"
          value={password}
          aria-invalid={!!problem}
          data-tip="输入密码，按回车登录"
          onChange={(e) => (setPassword(e.target.value), setProblem(null))}
        />
      </label>
      {problem && (
        <p className="login-problem" role="alert" data-code={problem.code}>
          {problem.text}
        </p>
      )}
      <div className="login-stack">
        <Button tip={username && password ? "登录" : "先输入用户名和密码"} tone="primary" size="lg" layout="login-go" type="submit" disabled={!username.trim() || !password || busy}>
          {busy ? "登录中…" : "登录"}
        </Button>
        {onRegister && (
          <Button tip="还没有账号：自己注册一个" tone="ghost" type="button" onClick={onRegister}>
            注册
          </Button>
        )}
      </div>
    </form>
  );
}

