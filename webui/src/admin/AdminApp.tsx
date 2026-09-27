import "./admin.css";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { BrandMark } from "../ui/icons";
import { useServer } from "../state/server";
import { adminApi } from "../api/admin";
import { useSession, useSignedIn } from "../state/session";
import { shown, visible } from "../api/applies";
import { Admin, type AdminContext } from "./common";
import { RestartBanner, RestartDialog, RestartVeil } from "./Restart";
import { BAND_OF, SECTIONS } from "./sections";
import { lastedText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

/** The /admin page. It is not linked from user pages and is served only after the administrator login
 * (Auth.tsx AdminGate, via site.tsx). A side list of sections (sections.tsx) shows one section at a time; the
 * address /admin#<id> preserves the selection across reloads and logins. The header shows the server state, the
 * restart button and logout, and a banner while a restart is pending. */

const TITLE = "Lab2Shot 管理";

const noOverview = async () => null; // Logins without server.status do not read the overview.

const sectionOf = () => {
  const id = window.location.hash.slice(1).split("/")[0]; // A section id may be followed by further path segments (e.g. #benchmarks/<kind>).
  return SECTIONS.some((s) => s.id === id) ? id : SECTIONS[0].id;
};

export default function AdminPage() {
  const state = useSignedIn();
  const logout = useSession((s) => s.logout);
  const sections = visible(SECTIONS, state?.applies); // Sections visible to this login (applies.ts).
  const [section, setSection] = useState(sectionOf);
  const [problem, setProblem] = useState<string | null>(null);
  const [queueVersion, setQueueVersion] = useState(0);
  const { data: queue, error: queueFailed } = usePoll(api.admin.queue, 1500, { key: queueVersion });
  const queueError = queueFailed ? reasonOf(queueFailed) : null;
  const { data: overview, reload: refreshOverview } = usePoll(shown(state?.applies, "server.status") ? adminApi.overview : noOverview, 5000, { onError: (e) => setProblem(reasonOf(e)) });
  const [asking, setAsking] = useState(false);
  const { info } = useServer();

  useEffect(() => {
    document.title = TITLE;
    const follow = () => setSection(sectionOf());
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

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
      askRestart: () => setAsking(true),
    }),
    [go, queue, overview, refreshOverview],
  );

  const current = sections.find((s) => s.id === section) ?? sections[0];
  const Current = current.Component;
  const restart = info?.restart;
  return (
    <Admin.Provider value={ctx}>
      <div className="adm">
        <header className="adm-top">
          <a className="help-brand" href="/admin#overview">
            <BrandMark />
            {TITLE}
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
              const mine = BAND_OF[s.id];
              const band = mine && mine !== (i ? BAND_OF[sections[i - 1].id] : "") ? mine : "";
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
