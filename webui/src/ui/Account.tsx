import { useState } from "react";
import { api } from "../api";
import { useSession } from "../state/session";
import { shown } from "../api/applies";
import { Sheet } from "./Sheet";
import "./account.css";
import { dayText } from "../platform/format";
import { Button } from "./Button";
import { Menu, type MenuRow } from "./Menu";

/** The account this page is logged in with, at the end of the top bar: its name and department; a click opens a
 * small menu — change the password (with the current one), log out, and the admin page when it is this login's.
 *
 * `extra`: rows only the page it sits on can offer (the editor's 「录入模板」 needs the graph that is open). They go
 * above 退出登录, in the order given; whether each is there at all is the page's own call, from the same one
 * availability answer (api/applies.ts) — this component looks at no role either.
 *
 * 我的占用 is not here: it lives in the 队列 window, next to the jobs whose results take the room. */

const MIN_CHARS = 8; // lab2shot/accounts.py MIN_CHARS
const MENU_WIDTH = 220;

export function AccountChip({ extra = [] }: { extra?: MenuRow[] }) {
  const state = useSession((s) => s.state);
  const logout = useSession((s) => s.logout);
  // where the menu opens (under the chip, its right edge on the chip's): null while it is closed. The site's one Menu
  // (ui/Menu.tsx) floats over the page in fixed coordinates — a layer of this component's own, positioned inside the
  // top bar, was cut by the bar (its overflow is hidden so a long graph name never pushes the buttons out)
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const [changing, setChanging] = useState(false);
  const user = state?.user;
  if (!user) return null;
  const until = user.expires ? `账号到 ${dayText(user.expires)}` : "账号不过期";
  const toggle = (e: React.MouseEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setAt(at ? null : { x: Math.max(8, r.right - MENU_WIDTH), y: r.bottom + 6 });
  };
  const rows: MenuRow[] = [
    { key: "who", label: <span data-user-data data-tip={user.username}>{user.username}</span>, desc: until, tip: until, off: true },
    { key: "password", label: "改密码", tip: "改自己的密码：要现在的密码；改完别处的登录都会退出", run: () => setChanging(true) },
    ...(shown(state?.applies, "page.admin")
      ? [{ key: "admin", label: "管理页面", tip: "管理页面：用户、队列、设置……（新标签页打开）", run: () => void window.open("/admin", "_blank", "noreferrer") }]
      : []),
    ...extra,
    { key: "logout", label: "退出登录", tip: "退出登录：这个浏览器要重新登录才能用；没保存的节点图留在这个浏览器里，登录后还在", run: () => void logout() },
  ];
  return (
    <div className="acct">
      <Button
        tip={`登录的账号：${user.name}（${user.username}）${user.department ? ` · ${user.department}` : ""}；${until}。点击改密码或退出`}
        layout="who-chip acct-chip" data-user-data
        data-initial={[...user.name][0]}
        onClick={toggle}
      >
        <span>
          {user.name}
          {user.department && <span className="who-dept"> · {user.department}</span>}
        </span>
      </Button>
      {at && <Menu at={at} rows={rows} label="账号" width={MENU_WIDTH} layout="acct-menu" onClose={() => setAt(null)} />}
      {changing && <PasswordSheet onClose={() => setChanging(false)} />}
    </div>
  );
}

function PasswordSheet({ onClose }: { onClose: () => void }) {
  const set = useSession((s) => s.set);
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [problem, setProblem] = useState("");
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  const rule = !next ? "" : next.length < MIN_CHARS ? `新密码至少要 ${MIN_CHARS} 个字符` : !next.trim() ? "新密码不能全是空格" : again && again !== next ? "两次输入的新密码不一样" : "";
  const ready = !!current && !!next && next === again && !rule;
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    try {
      set(await api.auth.change(current, next));
      setDone(true);
    } catch (err) {
      setProblem((err as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title="改密码" width={420} onClose={onClose}>
      <form className="acct-form" onSubmit={(e) => void submit(e)}>
        {done ? (
          <p className="pw-note ok">密码改好了：别处的登录都已退出，这个浏览器接着用。</p>
        ) : (
          <>
            <label className="login-field">
              <span data-tip="现在的密码">现在的密码</span>
              <input className="field" type="password" autoComplete="current-password" value={current} autoFocus data-tip="现在的密码" onChange={(e) => (setCurrent(e.target.value), setProblem(""))} />
            </label>
            <label className="login-field">
              <span data-tip={`新密码，至少 ${MIN_CHARS} 个字符；字母、数字、符号、中文都可以`}>新密码</span>
              <input className="field" type="password" autoComplete="new-password" value={next} data-tip={`至少 ${MIN_CHARS} 个字符`} onChange={(e) => (setNext(e.target.value), setProblem(""))} />
            </label>
            <label className="login-field">
              <span data-tip="再输一次新密码，防止输错">再输一次</span>
              <input className="field" type="password" autoComplete="new-password" value={again} data-tip="和上面的新密码一样" onChange={(e) => (setAgain(e.target.value), setProblem(""))} />
            </label>
            {(rule || problem) && (
              <p className="login-problem" role="alert">
                {rule || problem}
              </p>
            )}
          </>
        )}
        <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
          <Button tip={done ? "关掉" : "不改了"} tone="ghost" type="button" onClick={onClose}>
            {done ? "关闭" : "取消"}
          </Button>
          {!done && (
            <Button tip={ready ? "改成新密码；别处的登录都会退出" : "先填好现在的密码和两次一样的新密码"} tone="primary" type="submit" disabled={!ready || busy}>
              {busy ? "修改中…" : "改密码"}
            </Button>
          )}
        </div>
      </form>
    </Sheet>
  );
}
