import { useState } from "react";
import { api } from "../api";
import { useSession } from "../state/session";
import { shown } from "../api/applies";
import { Sheet } from "./Sheet";
import "./account.css";
import { dayText } from "../platform/format";
import { Button } from "./Button";
import { Menu, type MenuRow } from "./Menu";
import { MIN_CHARS, passwordProblem } from "../platform/accountRules";
import { useLang, type Lang } from "../i18n/lang";
import { t } from "../i18n/t";
import { switchLanguage } from "../state/language";
import { say } from "../state/say";
import { messageOf } from "../messages/message";
import { tipAttrs, tipOf } from "../platform/tips";

/** The account this page is logged in with, at the end of the top bar: its name and department; a click opens a
 * small menu — change the password (with the current one), log out, and the admin page when it is this login's.
 *
 * 我的占用 is not here: it lives in the 队列 window, next to the jobs whose results take the room. */

const MENU_WIDTH = 220;

export function AccountChip() {
  const state = useSession((s) => s.state);
  const logout = useSession((s) => s.logout);
  // where the menu opens (under the chip, its right edge on the chip's): null while it is closed. The site's one Menu
  // (ui/Menu.tsx) floats over the page in fixed coordinates — a layer of this component's own, positioned inside the
  // top bar, would be cut by the bar (its overflow is hidden so a long graph name never pushes the buttons out)
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const [changing, setChanging] = useState(false);
  const lang = useLang((s) => s.lang);
  const user = state?.user;
  if (!user) return null;
  const until = user.expires ? t("ui.account.expires", { day: dayText(user.expires) }) : t("ui.account.no_expiry");
  const toggle = (e: React.MouseEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setAt(at ? null : { x: Math.max(8, r.right - MENU_WIDTH), y: r.bottom + 6 });
  };
  const other: Lang = lang === "zh" ? "en" : "zh";
  const rows: MenuRow[] = [
    { key: "who", label: <span data-user-data {...tipAttrs(tipOf("truncated", user.username))}>{user.username}</span>, desc: until, off: true },
    // the interface language (lab2shot/i18n): the account keeps it; the page switches at once, without a reload
    {
      key: "lang",
      label: `${t("ui.account.language")} · ${t(`lang.${lang}`)}`,
      desc: t("ui.account.language_switch", { other: t(`lang.${other}`) }),
      run: () => void switchLanguage(other).catch((e) => say(messageOf(e))),
    },
    { key: "password", label: t("ui.account.change_password"), tip: tipOf("consequence", t("ui.account.logs_out_others")), run: () => setChanging(true) },
    ...(shown(state?.applies, "page.admin")
      ? [{ key: "admin", label: t("ui.account.admin_page"), run: () => void window.open("/admin", "_blank", "noreferrer") }]
      : []),
    { key: "logout", label: t("ui.account.log_out"), tip: tipOf("consequence", t("ui.account.log_out_tip")), run: () => void logout() },
  ];
  return (
    <div className="acct">
      <Button
        tip={tipOf("value", t("ui.account.chip_tip", { name: user.name, username: user.username, department: user.department_label ? ` · ${user.department_label}` : "", until }))}
        layout="who-chip acct-chip" data-user-data
        data-initial={[...user.name][0]}
        onClick={toggle}
      >
        <span>
          {user.name}
          {user.department_label && <span className="who-dept"> · {user.department_label}</span>}
        </span>
      </Button>
      {at && <Menu at={at} rows={rows} label={t("ui.account.account")} width={MENU_WIDTH} layout="acct-menu" onClose={() => setAt(null)} />}
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
  const rule = passwordProblem(next, again);
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
      say(messageOf(err)); // every error goes to the log too (the dialog says it where it is typed)
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={t("ui.account.change_password")} width={420} onClose={onClose}>
      <form className="acct-form" onSubmit={(e) => void submit(e)}>
        {done ? (
          <p className="pw-note ok">{t("ui.account.password_changed")}</p>
        ) : (
          <>
            <label className="login-field">
              <span>{t("ui.account.current_password")}</span>
              <input className="field" type="password" autoComplete="current-password" value={current} autoFocus onChange={(e) => (setCurrent(e.target.value), setProblem(""))} />
            </label>
            <label className="login-field">
              <span>{t("ui.account.new_password")}</span>
              <input className="field" type="password" autoComplete="new-password" value={next} onChange={(e) => (setNext(e.target.value), setProblem(""))} />
            </label>
            <label className="login-field">
              <span>{t("ui.account.again")}</span>
              <input className="field" type="password" autoComplete="new-password" value={again} onChange={(e) => (setAgain(e.target.value), setProblem(""))} />
            </label>
            {(rule || problem) && (
              <p className="login-problem" role="alert">
                {rule || problem}
              </p>
            )}
          </>
        )}
        <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
          <Button tone="ghost" type="button" onClick={onClose}>
            {done ? t("ui.common.close") : t("ui.common.cancel")}
          </Button>
          {!done && (
            <Button tip={ready ? tipOf("consequence", t("ui.account.logs_out_others")) : tipOf("disabled", t("ui.account.fill_first", { count: MIN_CHARS }))} tone="primary" type="submit" disabled={!ready || busy}>
              {busy ? t("ui.account.changing") : t("ui.account.change_password")}
            </Button>
          )}
        </div>
      </form>
    </Sheet>
  );
}
