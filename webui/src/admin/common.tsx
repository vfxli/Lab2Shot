import { createContext, useContext } from "react";
import type { QueueView } from "../api";
import type { OnlineSummary, Overview, Place, SettingsView } from "../api/admin";
import { agoText, clockText, fullTimeText, whenText } from "../platform/format";
import { t, t as tr } from "../i18n/t";
import { tipOf, type Tip } from "../platform/tips";

/** What the admin page's sections share: the page-wide notice, going to another section, the queue, the overview and
 * the settings (each read once for the whole page) and asking for a restart. Switching cards is the 显卡 section's alone (Cards.tsx). */
export interface AdminContext {
  problem: (text: string | null) => void;
  go: (section: string) => void;
  queue: QueueView | null;
  refreshQueue: () => void;
  overview: Overview | null;
  refreshOverview: () => void;
  settings: SettingsView | null; // read while a section that shows settings is open (常驻模型, the settings pages)
  settingsSaved: (view: SettingsView) => void; // what saving answered: every section reads the same copy
  askRestart: () => void;
}

export const Admin = createContext<AdminContext | null>(null);

export function useAdmin(): AdminContext {
  return useContext(Admin)!;
}

/** One section of the admin page: its title with the section's buttons, what it is for, its content. */
export function Section({ title, lede, actions, className, children }: {
  title: string;
  lede?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={`adm-sec${className ? ` ${className}` : ""}`}>
      <header className="adm-sec-head">
        <h2>{title}</h2>
        {actions && <div className="adm-sec-actions">{actions}</div>}
      </header>
      {lede && <p className="adm-lede">{lede}</p>}
      {children}
    </section>
  );
}

/** Who is online and where, for the 在线 tile's tooltip (lab2shot/accounts.py online()); empty when nobody is. A
 * browser and a DCC plugin/command line count and are shown separately (each account may have one of each at once). */
export function onlineTip(online?: OnlineSummary): Tip | undefined {
  if (!online) return undefined;
  const lines = [...online.who.browser.map((w) => t("ui.admin.common.browser_at", { where: w })), ...online.who.client.map((w) => t("ui.admin.common.plugin_at", { where: w }))];
  return lines.length ? tipOf("value", t("ui.admin.common.online", { lines: lines.join("\n") })) : undefined;
}

/** Where a request came from, in words: 浏览器 · Chrome · Windows, 插件 · maya · 电脑名. */
export const placeText = (w: Place) =>
  w.kind === "web" ? t("ui.admin.common.browser_at", { where: w.device }) : t("ui.admin.common.plugin_at", { where: `${w.device}${w.hostname ? ` · ${w.hostname}` : ""}` });

/** When an account was last active, in words: 刚刚, 3 分钟前, 今天 14:05, 昨天 18:20, 9/14 18:20, 从未. */
export function activeText(t: number | null): string {
  if (!t) return tr("ui.admin.common.never");
  const s = Date.now() / 1000 - t;
  if (s < 3600) return agoText(t);
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(new Date()) - day(new Date(t * 1000))) / 86_400_000);
  if (days === 0) return tr("ui.admin.common.today_at", { time: clockText(t) });
  if (days === 1) return tr("ui.admin.common.yesterday_at", { time: clockText(t) });
  return new Date(t * 1000).getFullYear() === new Date().getFullYear() ? whenText(t) : fullTimeText(t);
}
