import { useEffect, useState, type ReactNode } from "react";
import { msg, type Message } from "../messages/message";
import { MessageText } from "./MessageText";
import { readNotice } from "../api/notice";
import { nextOrigin, useServer } from "../state/server";
import { Button } from "./Button";
import "./banner.css";
import { t } from "../i18n/t";
import { tipAttrs, tipOf, type Tip } from "../platform/tips";

/** The site's single notice bar: the administrator's notice, the server restarting or not answering, this graph being
 * open in another tab. One component, distinguished only by its tone. It never covers anything a click needs and never
 * closes by itself: its content remains true until it changes.
 *
 * `tone` carries the meaning: info / notice (blue), warn (orange), error (purple), risk (red, only for what can cause a
 * production accident), ok (green). `float`: over the page, centred under the top bar. */

export type BannerTone = "info" | "notice" | "warn" | "error" | "risk" | "ok";

export function Banner({ tone, children, tip, aside, float, spin, actions }: {
  tone: BannerTone;
  children: ReactNode;
  tip?: Tip | null;
  aside?: ReactNode; // the author, or the address the server moves to
  float?: boolean;
  spin?: boolean; // an operation is in progress (a restart)
  actions?: ReactNode;
}) {
  return (
    <div className={`banner ${tone}${float ? " float glass" : ""}`} role="status" {...tipAttrs(tip)} data-tone={tone}>
      <i className={spin ? "spin" : undefined} />
      <span className="banner-text">{children}</span>
      {aside && <span className="banner-aside">{aside}</span>}
      {actions}
    </div>
  );
}

/** The administrator's notice: read once, and again whenever the shared poll of the server's state reports a change
 * (state/server.ts `notice`); it has no poll of its own. */
function AdminNotice() {
  const at = useServer().info?.notice ?? 0;
  const [said, setSaid] = useState<{ text: string; tone: BannerTone; on: boolean; by?: string } | null>(null);
  useEffect(() => {
    let live = true;
    readNotice()
      .then((n) => live && setSaid(n))
      .catch(() => live && setSaid(null)); // not logged in, or the server is unavailable: the restart notice covers that case
    return () => {
      live = false;
    };
  }, [at]);
  if (!said?.on || !said.text) return null;
  return (
    <Banner tone={said.tone} float aside={t("ui.misc.notice")} tip={tipOf("value", said.by ? t("ui.misc.notice_by", { by: said.by }) : t("ui.misc.notice_admin"))}>
      {said.text}
    </Banner>
  );
}

/** A short notice while the server restarts (triggered from the admin page) or does not answer: the page stays as it
 * is and continues when the server is back; the editor keeps the graph and resumes following its job. After a restart
 * that brought a new build of the page, it offers to reload (the graph is restored from the autosave). */
function RestartNotice() {
  const { info, down, restarted, newPage } = useServer();
  const [shown, setShown] = useState<string | null>(null); // the server process whose restart was announced
  const boot = info?.boot ?? null;
  const back = restarted && !down && !info?.restart && shown !== boot;
  useEffect(() => {
    if (!back || newPage) return;
    const timer = window.setTimeout(() => setShown(boot), 6000);
    return () => window.clearTimeout(timer);
  }, [back, newPage, boot]);

  const moved = nextOrigin(info?.restart);
  let said: Message | null = null;
  let tip: Tip | undefined;
  let tone: BannerTone = "warn";
  let spin = false;
  if (info?.restart?.state === "draining") {
    said = msg("N-SERVER-DRAINING");
    tip = tipOf("consequence", t("ui.misc.draining_tip"));
  } else if (info?.restart || down) {
    said = msg(info?.restart ? "N-SERVER-RESTARTING" : "N-SERVER-OFFLINE");
    tip = tipOf("consequence", t("ui.misc.offline_tip"));
    spin = true;
  } else if (back) {
    said = msg(newPage ? "N-SERVER-BACKNEWPAGE" : "N-SERVER-BACK");
    tip = tipOf("consequence", newPage ? t("ui.misc.new_page_tip") : t("ui.misc.back_tip"));
    tone = "ok";
  }
  if (!said) return null;
  return (
    <Banner
      tone={tone}
      float
      spin={spin}
      tip={tip}
      aside={moved && (info?.restart || down) ? t("ui.misc.moved", { address: moved }) : undefined}
      actions={
        back && newPage ? (
          <Button size="sm" onClick={() => window.location.reload()}>
            {t("ui.common.refresh")}
          </Button>
        ) : undefined
      }
    >
      <MessageText message={said} />
    </Banner>
  );
}

/** What every page of the site shows at its top, stacked (site.tsx renders it once): the administrator's notice, then
 * the server's own state. The stack sits in the page's flow at the very top, full width (ui/banner.css .site-notices),
 * so it pushes the page down instead of covering a control, and is only as tall as its visible content.
 *
 * `restart={false}` omits the server's own state: the admin page is where the server is restarted, and it reports that
 * itself (admin/Restart.tsx). The administrator's notice is shown on every page regardless. */
export function SiteNotices({ restart = true }: { restart?: boolean } = {}) {
  return (
    <div className="site-notices">
      <AdminNotice />
      {restart && <RestartNotice />}
    </div>
  );
}
