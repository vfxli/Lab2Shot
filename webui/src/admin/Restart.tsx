import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { InstallTask } from "../api";
import { Sheet } from "../ui/Sheet";
import type { RestartState } from "../api";
import { nextOrigin, pollServer, useServer } from "../state/server";
import { adminApi } from "../api/admin";
import { useAdmin } from "./common";
import { useSignedIn } from "../state/session";
import { usable } from "../api/applies";
import { Button } from "../ui/Button";

/** Server restart from the admin page (lab2shot/server/restart.py): a confirmation stating whether running jobs are
 * awaited or stopped and what is lost; the banner shown while settings await a restart or a restart awaits running
 * jobs; and the overlay shown while the server is unavailable. The page reloads automatically once the server is back. */

export function RestartDialog({ onClose }: { onClose: () => void }) {
  const { queue, overview, problem } = useAdmin();
  const state = useSignedIn();
  const [installing, setInstalling] = useState<InstallTask | null>(null);
  const [busy, setBusy] = useState(false);
  // an install under way is only this login's to see with the right to install (the same routes as 扩展包); without
  // it the list is not asked for at all: the refusal would count against the session
  const installs = usable(state.applies, "extensions");
  useEffect(() => {
    if (installs) api.installs.list().then((r) => setInstalling(r.jobs.find((j) => j.state === "running") ?? null), () => undefined);
  }, [installs]);
  const running = queue?.jobs.filter((j) => j.state === "running") ?? [];
  const waiting = queue?.jobs.filter((j) => j.state === "queued") ?? [];
  const pending = overview?.pending ?? [];

  const go = async (mode: "drain" | "now") => {
    setBusy(true);
    try {
      await adminApi.restart(mode);
      pollServer();
      problem(null);
      onClose();
    } catch (e) {
      problem((e as Error).message);
      setBusy(false);
    }
  };

  const works = running.length > 0 || installing !== null;
  const kept = waiting.length ? `排队的 ${waiting.length} 个任务留在队列里，重启后接着算。` : "";
  return (
    <Sheet title="重启服务" width={600} onClose={onClose}>
      <div className="rs-body">
        <p>
          重启一般十几秒：先卸载常驻的模型，再停止监听，然后用同样的命令重新启动。正在用网页的人会看到「服务器正在重启」，回来后自动接上；DCC
          插件和命令行重试一次即可。任务记录、显卡授权、使用统计和设置都不受影响。
        </p>
        {pending.length > 0 && <p className="rs-note">重启后生效：{pending.map((p) => p.label).join("、")}。</p>}
        {works && (
          <ul className="rs-list">
            {running.map((j) => (
              <li key={j.id}>
                计算中：{j.client?.who ?? ""} ·「{j.title}」{!!j.cards?.length && ` · ${j.cards.join("、")}`}
              </li>
            ))}
            {installing && <li>正在安装：{installing.title.text}</li>}
            {waiting.length > 0 && <li>排队：{waiting.length} 个任务</li>}
          </ul>
        )}
        {works ? (
          <div className="rs-choices">
            <button className="rs-choice" data-tip="队列里的任务全部算完才重启，一件也不丢" disabled={busy} onClick={() => void go("drain")} autoFocus>
              <b>等当前任务算完再重启</b>
              <span>
                不再开始新的任务，新提交的先排队；{running.length ? `计算中的 ${running.length} 个任务` : ""}
                {running.length && installing ? "和" : ""}
                {installing ? `${installing.title.text} 的安装` : ""}完成后自动重启。{kept}什么都不会丢。
              </span>
            </button>
            <button className="rs-choice danger" data-tip="马上重启：正在算的任务会停下，重新提交要从头算" disabled={busy} onClick={() => void go("now")}>
              <b>立即重启</b>
              <span>
                {running.length > 0 && `马上停下计算中的 ${running.length} 个任务：记为已取消，已经算完的节点留在缓存里，再算时从那里接着，没算完的那部分要重新算。`}
                {installing && `${installing.title.text} 的安装中断，再点安装会接着装。`}
                {kept}
              </span>
            </button>
          </div>
        ) : (
          <p className="rs-note">{waiting.length ? kept : "现在没有任务在算，什么都不会丢。"}</p>
        )}
        <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
          <Button tip="不重启，关掉这个窗口" tone="ghost" onClick={onClose}>
            取消
          </Button>
          {!works && (
            <Button tip="现在没有任务在算：马上重启服务" tone="primary" disabled={busy} onClick={() => void go("drain")} autoFocus>
              重启服务
            </Button>
          )}
        </div>
      </div>
    </Sheet>
  );
}

/** Banner below the top bar: a restart waiting for running jobs (stop waiting or cancel it), or saved settings
 * awaiting a restart. */
export function RestartBanner() {
  const { overview, askRestart, problem, refreshQueue } = useAdmin();
  const { info } = useServer();
  const r = info?.restart;
  const act = (what: Promise<unknown>) =>
    what.then(
      () => (pollServer(), refreshQueue(), problem(null)),
      (e: Error) => problem(e.message),
    );
  if (r?.state === "draining") {
    const what = [r.running ? `${r.running} 个计算中的任务` : "", ...r.tasks.map((task) => task.text)].filter(Boolean).join("和");
    return (
      <div className="adm-banner draining" role="status">
        <i className="spin" />
        <span>{what ? `等${what}完成后重启` : "马上重启"}：不再开始新任务，新提交的先排队，重启后接着算</span>
        <Button tip="马上停下计算中的任务（记为已取消，算完的节点留在缓存里）再重启" onClick={() => void act(adminApi.restart("now"))}>
          立即重启
        </Button>
        <Button tip="不重启了：队列照常继续" tone="ghost" onClick={() => void act(adminApi.callOff())}>
          不重启了
        </Button>
      </div>
    );
  }
  if (!overview?.pending.length || r) return null;
  return (
    <div className="adm-banner pending" role="status">
      <span>
        {overview.pending.map((p) => p.label).join("、")} 改了，重启服务后生效
      </span>
      <Button tip="改过的设置要重启服务才生效：先看会影响哪些任务再决定" tone="primary" onClick={askRestart}>
        重启服务
      </Button>
    </div>
  );
}

/** Overlay shown while the server restarts or does not respond; reloads the page once it is back. A restart that
 * moves the server (another port, HTTPS switched) cannot be followed from here: the page's own security policy
 * (connect-src 'self', lab2shot/server/access.py) lets it ask nothing of another address, so the new address is
 * given as a link to open once the server is up. */
export function RestartVeil() {
  const { info, down, restarted } = useServer();
  const last = useRef<RestartState | null>(null);
  if (info?.restart) last.current = info.restart;
  const away = down || info?.restart?.state === "restarting";
  const [since, setSince] = useState(0);
  const [now, setNow] = useState(Date.now());
  const moved = nextOrigin(last.current);

  useEffect(() => {
    if (!away) return setSince(0);
    setSince((s) => s || Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(t);
  }, [away]);

  // Server is back: reload the page (the server may serve a new build, and every section re-reads its data).
  useEffect(() => {
    if (restarted && !down) window.location.reload();
  }, [restarted, down]);

  if (!away) return null;
  const seconds = since ? Math.round((now - since) / 1000) : 0;
  const restarting = !!last.current;
  return (
    <div className="rs-veil veil" role="alertdialog" aria-label={restarting ? "正在重启" : "连不上服务器"}>
      <div className="rs-veil-card glass">
        <i className="spin big" />
        <h2>{restarting ? "正在重启…" : "连不上服务器"}</h2>
        <p>
          {!restarting ? "服务可能正在重启或已经停了。回来后这一页自动刷新。"
            : moved ? "卸载常驻的模型、停止监听、换到新地址重新启动，一般十几秒。原来的地址不会再回来，这一页不会自己刷新。"
            : "卸载常驻的模型、停止监听、重新启动，一般十几秒。回来后这一页自动刷新。"}
          {seconds > 0 && ` 已等 ${seconds} 秒。`}
        </p>
        {moved && (
          <p>
            重启后的地址是 <a href={`${moved}/admin`}>{moved}</a>：等十几秒服务起来以后点这个地址（HTTPS 的证书要先装好）
          </p>
        )}
        {!moved && seconds > 60 && <p className="rs-note">超过一分钟还没回来：到服务器上查看 lab2shot ui 的输出或服务日志 work/logs/lab2shot.log。</p>}
      </div>
    </div>
  );
}
