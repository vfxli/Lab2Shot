import { createContext, useContext } from "react";
import type { QueueView } from "../api";
import type { OnlineSummary, Overview, Place } from "../api/admin";
import { agoText, clockText, fullTimeText, whenText } from "../platform/format";

/** What the admin page's sections share: the page-wide notice, going to another section, the queue and the overview
 * (read once for the whole page) and asking for a restart. Switching cards is the 显卡 section's alone (Cards.tsx). */
export interface AdminContext {
  problem: (text: string | null) => void;
  go: (section: string) => void;
  queue: QueueView | null;
  refreshQueue: () => void;
  overview: Overview | null;
  refreshOverview: () => void;
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

/** What 在线 means, for every tooltip that shows it (lab2shot/accounts.py ONLINE_S): real presence, not a login. */
export const onlineMeaning = (windowS: number) =>
  `最近 ${windowS} 秒内向服务器发过请求就算在线：开着的页面至少每分钟问一次服务器，关掉页面（或浏览器把标签页藏到后台）一两分钟后就不算了`;

/** Who is online and where, for the 在线 tile's tooltip (lab2shot/accounts.py online()): a browser and a DCC
 * plugin/command line count and are shown separately (each account may have one of each at once). */
export function onlineTip(online?: OnlineSummary): string {
  if (!online) return "";
  const lines = [...online.who.browser.map((w) => `浏览器 · ${w}`), ...online.who.client.map((w) => `插件 · ${w}`)];
  return `${lines.length ? `在线：\n${lines.join("\n")}` : "现在没有人在线"}\n${onlineMeaning(online.window_s)}`;
}

/** Where a request came from, in words: 浏览器 · Chrome · Windows, 插件 · maya · 电脑名. */
export const placeText = (w: Place) =>
  w.kind === "web" ? `浏览器 · ${w.device}` : `插件 · ${w.device}${w.hostname ? ` · ${w.hostname}` : ""}`;

/** When an account was last active, in words: 刚刚, 3 分钟前, 今天 14:05, 昨天 18:20, 9/14 18:20, 从未. */
export function activeText(t: number | null): string {
  if (!t) return "从未";
  const s = Date.now() / 1000 - t;
  if (s < 3600) return agoText(t);
  const day = (d: Date) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((day(new Date()) - day(new Date(t * 1000))) / 86_400_000);
  if (days === 0) return `今天 ${clockText(t)}`;
  if (days === 1) return `昨天 ${clockText(t)}`;
  return new Date(t * 1000).getFullYear() === new Date().getFullYear() ? whenText(t) : fullTimeText(t);
}
