import { createContext, useContext } from "react";
import type { QueueView } from "../api";
import type { Overview } from "../api/admin";

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

/** Who is online and where, for the 在线 tile's tooltip (lab2shot/accounts.py online()): a browser and a DCC
 * plugin/command line count and are shown separately (each account may have one of each at once). */
export function onlineTip(online?: { who: { browser: string[]; client: string[] } }): string {
  if (!online) return "";
  const lines = [...online.who.browser.map((w) => `浏览器 · ${w}`), ...online.who.client.map((w) => `插件 · ${w}`)];
  return lines.length ? `在线：\n${lines.join("\n")}` : "现在没有人在线";
}
