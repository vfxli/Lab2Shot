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
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

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
  const kept = waiting.length ? t("ui.admin.restart.kept", { n: waiting.length }) : "";
  const awaited = [running.length ? t("ui.admin.restart.running_jobs", { n: running.length }) : "", installing ? t("ui.admin.restart.install_of", { title: installing.title.text }) : ""]
    .filter(Boolean)
    .join(t("ui.admin.restart.and"));
  return (
    <Sheet title={t("ui.admin.page.restart")} width={600} onClose={onClose}>
      <div className="rs-body">
        <p>
          {t("ui.admin.restart.lede")}
        </p>
        {pending.length > 0 && <p className="rs-note">{t("ui.admin.restart.pending", { labels: pending.map((p) => p.label).join(t("list.sep")) })}</p>}
        {works && (
          <ul className="rs-list">
            {running.map((j) => (
              <li key={j.id}>
                {t("ui.admin.restart.cooking", { who: j.client?.who ?? "", title: j.title })}{!!j.cards?.length && ` · ${j.cards.join(t("list.sep"))}`}
              </li>
            ))}
            {installing && <li>{t("ui.admin.restart.installing", { title: installing.title.text })}</li>}
            {waiting.length > 0 && <li>{t("ui.admin.restart.queued", { n: waiting.length })}</li>}
          </ul>
        )}
        {works ? (
          <div className="rs-choices">
            <button className="rs-choice" disabled={busy} onClick={() => void go("drain")} autoFocus>
              <b>{t("ui.admin.restart.drain")}</b>
              <span>
                {t("ui.admin.restart.drain_says", { awaited, kept })}
              </span>
            </button>
            <button className="rs-choice danger" disabled={busy} onClick={() => void go("now")}>
              <b>{t("ui.admin.restart.now")}</b>
              <span>
                {running.length > 0 && t("ui.admin.restart.now_running", { n: running.length })}
                {installing && t("ui.admin.restart.now_install", { title: installing.title.text })}
                {kept}
              </span>
            </button>
          </div>
        ) : (
          <p className="rs-note">{waiting.length ? kept : t("ui.admin.restart.idle")}</p>
        )}
        <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          {!works && (
            <Button tone="primary" disabled={busy} onClick={() => void go("drain")} autoFocus>
              {t("ui.admin.page.restart")}
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
    const what = [r.running ? t("ui.admin.restart.running_jobs", { n: r.running }) : "", ...(r.tasks ?? []).map((task) => task.text)].filter(Boolean).join(t("ui.admin.restart.and"));
    return (
      <div className="adm-banner draining" role="status">
        <i className="spin" />
        <span>{what ? t("ui.admin.restart.draining", { what }) : t("ui.admin.restart.draining_now")}</span>
        <Button tip={tipOf("consequence", t("ui.admin.restart.now_tip"))} onClick={() => void act(adminApi.restart("now"))}>
          {t("ui.admin.restart.now")}
        </Button>
        <Button tone="ghost" onClick={() => void act(adminApi.callOff())}>
          {t("ui.admin.restart.call_off")}
        </Button>
      </div>
    );
  }
  if (!overview?.pending.length || r) return null;
  return (
    <div className="adm-banner pending" role="status">
      <span>
        {t("ui.admin.overview.pending", { settings: overview.pending.map((p) => p.label).join(t("list.sep")) })}
      </span>
      <Button tone="primary" onClick={askRestart}>
        {t("ui.admin.page.restart")}
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
    const timer = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(timer);
  }, [away]);

  // Server is back: reload the page (the server may serve a new build, and every section re-reads its data).
  useEffect(() => {
    if (restarted && !down) window.location.reload();
  }, [restarted, down]);

  if (!away) return null;
  const seconds = since ? Math.round((now - since) / 1000) : 0;
  const restarting = !!last.current;
  return (
    <div className="rs-veil veil" role="alertdialog" aria-label={restarting ? t("ui.admin.page.restarting") : t("ui.admin.restart.unreachable")}>
      <div className="rs-veil-card glass">
        <i className="spin big" />
        <h2>{restarting ? t("ui.admin.restart.restarting_dots") : t("ui.admin.restart.unreachable")}</h2>
        <p>
          {!restarting ? t("ui.admin.restart.down")
            : moved ? t("ui.admin.restart.moving")
            : t("ui.admin.restart.restarting_says")}
          {seconds > 0 && t("ui.admin.restart.waited", { n: seconds })}
        </p>
        {moved && (
          <p>
            {t("ui.admin.restart.moved_before")}<a href={`${moved}/admin`}>{moved}</a>{t("ui.admin.restart.moved_after")}
          </p>
        )}
        {!moved && seconds > 60 && <p className="rs-note">{t("ui.admin.restart.long")}</p>}
      </div>
    </div>
  );
}
