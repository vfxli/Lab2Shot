import "./admin.css";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type SettingsPageEntry } from "../api";
import { BrandMark } from "../ui/icons";
import { useServer } from "../state/server";
import { adminApi, type SettingsView } from "../api/admin";
import { useSession, useSignedIn } from "../state/session";
import { shown, visible } from "../api/applies";
import { Admin, type AdminContext } from "./common";
import { RestartBanner, RestartDialog, RestartVeil } from "./Restart";
import { bandsOf, sectionsFor, type AdminSection } from "./sections";
import { lastedText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

/** The /admin page. User pages link to it only from an administrator's account menu; it is served only after the
 * administrator login (Auth.tsx AdminGate, via site.tsx). A side list of sections (sections.tsx) shows one section at
 * a time; the address /admin#<id> preserves the selection across reloads and logins. The header shows the server state, the
 * restart button and logout, and a banner while a restart is pending. */

const TITLE = "Lab2Shot 管理";

const noOverview = async () => null; // Logins without server.status do not read the overview.
const noQueue = async () => null; // Logins without queue.manage do not read the whole queue.
const noSettings = async () => null; // no open section shows settings
const QUEUE_WATCHED = 1500; // the 队列 section or the restart dialog is open: what runs now, as it changes
const QUEUE_BADGE = 15_000; // any other section: only the count on the side list

/** The section the address names (/admin#<id>); the first one when it names none of them. */
const sectionOf = (sections: readonly AdminSection[]) => {
  const id = window.location.hash.slice(1).split("/")[0]; // A section id may be followed by further path segments (e.g. #users/<id>).
  return sections.some((s) => s.id === id) ? id : sections[0].id;
};

export default function AdminPage() {
  const state = useSignedIn();
  const logout = useSession((s) => s.logout);
  // the fixed sections and the 设置 band (the server's pages); keyed by what the pages say, so the login state read
  // again does not make every section a new component (which would reset the one open)
  const pages = JSON.stringify(state?.settings_pages ?? []);
  const all = useMemo(() => sectionsFor(JSON.parse(pages) as SettingsPageEntry[]), [pages]);
  const bands = useMemo(() => bandsOf(all), [all]);
  const sections = visible(all, state?.applies); // Sections visible to this login (applies.ts).
  const [section, setSection] = useState(() => sectionOf(all));
  const [problem, setProblem] = useState<string | null>(null);
  const [queueVersion, setQueueVersion] = useState(0);
  const [asking, setAsking] = useState(false);
  const queueShown = shown(state?.applies, "queue");
  const { data: queue, error: queueFailed } = usePoll(queueShown ? api.admin.queue : noQueue, section === "queue" || asking ? QUEUE_WATCHED : QUEUE_BADGE, {
    key: queueVersion,
  });
  const queueError = queueFailed ? reasonOf(queueFailed) : null;
  const { data: overview, reload: refreshOverview } = usePoll(shown(state?.applies, "server.status") ? adminApi.overview : noOverview, 5000, { onError: (e) => setProblem(reasonOf(e)) });
  const { info } = useServer();
  // the settings, one copy for the page: the settings pages edit them, 常驻模型 shows its policy from them
  const wantsSettings = section === "resident" || section.startsWith("settings-");
  const { data: settingsRead } = usePoll(wantsSettings ? adminApi.settings : noSettings, null, { key: wantsSettings, onError: (e) => setProblem(reasonOf(e)) });
  const [settings, settingsSaved] = useState<SettingsView | null>(null);
  useEffect(() => void (settingsRead && settingsSaved(settingsRead)), [settingsRead]);

  useEffect(() => {
    document.title = TITLE;
    const follow = () => setSection(sectionOf(all));
    follow(); // the settings pages arrive with the login state: an address naming one is taken once they are known
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, [all]);

  const go = useCallback((id: string) => {
    setProblem(null);
    window.location.hash = id;
  }, []);

  const ctx: AdminContext = useMemo(
    () => ({
      problem: setProblem,
      go,
      queue,
      refreshQueue: () => setQueueVersion((v) => v + 1),
      overview,
      refreshOverview,
      settings,
      settingsSaved,
      askRestart: () => setAsking(true),
    }),
    [go, queue, overview, refreshOverview, settings],
  );

  const current = sections.find((s) => s.id === section) ?? sections[0];
  const Current = current.Component;
  const restart = info?.restart;
  return (
    <Admin.Provider value={ctx}>
      <div className="adm">
        <header className="adm-top">
          <a className="help-brand" href="/admin#overview" aria-label={TITLE}>
            <BrandMark />
          </a>
          <div className="adm-top-right">
            {overview && (
              <span
                className="chip adm-server"
                data-tip={`地址 ${overview.server.address}\n版本 ${overview.server.version} · 进程 ${overview.server.pid}\n启动命令：${overview.server.command}`}
              >
                <i style={{ background: restart ? "var(--orange)" : "var(--green)" }} />
                {restart ? (restart.state === "draining" ? "等任务算完后重启" : "正在重启") : `运行中 · 已运行 ${lastedText(overview.server.started)}`}
              </span>
            )}
            {shown(state?.applies, "server.restart") && (
              <Button
                tip="重启这个服务：让要重启才生效的设置生效，或者服务不正常时重新开始。先问一句，有任务在算时可以等它们算完"
                disabled={restart?.state === "restarting"}
                onClick={() => setAsking(true)}
              >
                重启服务
              </Button>
            )}
            <Button tip="这个浏览器退出登录，编辑器也一起退出，回到登录页" tone="ghost" onClick={logout}>
              退出登录
            </Button>
          </div>
        </header>
        <RestartBanner />
        <div className="adm-body">
          <nav className="adm-nav" aria-label="管理页面的各部分">
            {sections.map((s, i) => {
              const badge = s.badge?.(ctx);
              // Band heading: shown only when the previous visible section belongs to another band, so a band hidden
              // from this login leaves no empty heading.
              const mine = bands[s.id];
              const band = mine && mine !== (i ? bands[sections[i - 1].id] : "") ? mine : "";
              return (
                <div key={s.id} style={{ display: "contents" }}>
                  {band && <div className="adm-nav-group">{band}</div>}
                  <a href={`#${s.id}`} className={s.id === current.id ? "on" : undefined} data-tip={s.tip} aria-current={s.id === current.id ? "page" : undefined}>
                    <span>{s.label}</span>
                    {badge && <span className={`adm-badge${badge === "!" ? " warn" : ""}`}>{badge}</span>}
                  </a>
                </div>
              );
            })}
          </nav>
          <main className="adm-main admin" data-section={current.id}>
            {(problem || queueError) && (
              <div className="notice adm-notice">
                <span>{problem ?? queueError}</span>
                {problem && (
                  <Button tip="关掉这条提示" tone="ghost" onClick={() => setProblem(null)}>
                    知道了
                  </Button>
                )}
              </div>
            )}
            <Current key={current.id} />
          </main>
        </div>
        {asking && <RestartDialog onClose={() => setAsking(false)} />}
        <RestartVeil />
      </div>
    </Admin.Provider>
  );
}
