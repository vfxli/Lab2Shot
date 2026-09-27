import { messageOf, msg, type Message } from "./messages/message";
import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { deviceId } from "./platform/client";
import { COPYRIGHT } from "./platform/brand";
import { ApiError, fromGate, json, loggedIn, NEED_LOGIN } from "./platform/http";
import { BrandMark } from "./ui/icons";
import { Loading } from "./ui/Loading";
import { whenText } from "./platform/format";
import { Button } from "./ui/Button";

/** The gate: all anyone gets before logging in (lab2shot/server/auth.py). It asks the server whether this browser is
 * logged in; if it is, it loads the page (site.tsx, a file the server hands out only then), if not it asks for the
 * username and password of the account the administrator made. When the login runs out while a page is open (the
 * account was disabled or expired, the password changed elsewhere, or the account logged in on another browser:
 * one place online per account, per kind, lab2shot/accounts.py), the page stays as it is underneath and the login is
 * asked for over it, so nothing being edited is lost. An administrator who forgot the password sets a new one with the
 * server-side 口令 on the admin page's own login (admin/Auth.tsx) or from the command line, not here, where every other
 * user would read 「忘记密码」 as something they can do. Kept small on purpose: no other part of the page's code is in here. */

const Site = lazy(() => import("./site"));

/** What was in it, when the login this browser had was ended by another one (lab2shot/accounts.py, server/auth.py):
 * when, from where, what device. */
export interface Kicked {
  at: number;
  ip: string;
  device: string;
  detail: string;
}



type Door = "asking" | "in" | "out";

export function Gate() {
  const [door, setDoor] = useState<Door>("asking");
  const [again, setAgain] = useState(false);
  const [problem, setProblem] = useState<Message | null>(null);
  const [kicked, setKicked] = useState<Kicked | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const s = await fromGate(() => json<{ user?: unknown; kicked?: Kicked | null }>("GET", "/api/auth/state", undefined, { cache: "no-store" }));
        if (alive) setDoor(s?.user ? "in" : "out");
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
    window.addEventListener(NEED_LOGIN, need);
    return () => {
      alive = false;
      window.removeEventListener(NEED_LOGIN, need);
    };
  }, []);

  if (door === "asking") return <Loading what="登录状态" fill />;
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
          onIn={() => {
            if (door === "in") loggedIn(); // the page underneath goes on, as it was
            setAgain(false);
            setKicked(null);
            setProblem(null);
            setDoor("in");
          }}
        />
      )}
    </>
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
  onIn: () => void;
}) {
  useEffect(() => {
    if (!over) document.title = "Lab2Shot";
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
      <main className="login-main">
        {!over && (
          <div className="login-mark" aria-hidden>
            <BrandMark hero />
          </div>
        )}
        {kicked ? (
          <KickedNotice kicked={kicked} onLoginAgain={onLoginAgain} />
        ) : (
          <LoginForm over={over} first={first} onIn={onIn} />
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

function LoginForm({ over, first, onIn }: { over: boolean; first: Message | null; onIn: () => void }) {
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
      await fromGate(() => json<unknown>("POST", "/api/auth/login", { username: username.trim(), password, device_id: deviceId() }));
      onIn();
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
        <span data-tip="管理员在管理页面「用户」里建好的用户名">用户名</span>
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
      </div>
    </form>
  );
}

