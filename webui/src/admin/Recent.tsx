import { adminApi, type RecentPeriod, type RecentView } from "../api/admin";
import { countText, sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { onlineTip, useAdmin } from "./common";
import { t } from "../i18n/t";
import { tipAttrs, type Tip } from "../platform/tips";

/** 概览's 「今天和最近」: what happened today and lately, one card per subject (访问, 注册, 任务, 流量, 反馈), each opening
 * its section like the machine tiles above. Every card is the same row of columns — 现在, 今日, 近 7 天 (本周 for
 * 注册), 本月 — so a period's numbers stand under each other from card to card; a card leaves a column it has no
 * number for empty. The server sends only the groups this login may open the section of (server/available.py
 * RECENT), and counts by its own local time (lab2shot/periods.py). Read more slowly than the tiles: these are counts
 * of a day, not what runs now. */

const EVERY = 30_000;

/** A period's name before what is counted in it: 今日登录 / Logins Today (a table per subject: written out whole). */
const PERIOD_LABEL = {
  access: { today: () => t("ui.admin.recent.access_today"), days7: () => t("ui.admin.recent.access_days7"), month: () => t("ui.admin.recent.access_month") },
  accounts: { today: () => t("ui.admin.recent.accounts_today"), week: () => t("ui.admin.recent.accounts_week"), month: () => t("ui.admin.recent.accounts_month") },
  tasks: { today: () => t("ui.admin.recent.tasks_today"), days7: () => t("ui.admin.recent.tasks_days7"), month: () => t("ui.admin.recent.tasks_month") },
  traffic: { today: () => t("ui.admin.recent.traffic_today"), days7: () => t("ui.admin.recent.traffic_days7"), month: () => t("ui.admin.recent.traffic_month") },
  feedback: { today: () => t("ui.admin.recent.feedback_today"), days7: () => t("ui.admin.recent.feedback_days7"), month: () => t("ui.admin.recent.feedback_month") },
};

interface Cell {
  label: string;
  value: string;
  sub?: string;
  tip?: Tip | null;
}

interface Group {
  id: string;
  title: string;
  page: string; // the section it opens, by name
  section: string;
  now?: Cell;
  periods: [RecentPeriod, Cell][]; // 今日, then 近 7 天 or 本周, then 本月
}

const PAGE: Record<string, () => string> = {
  security: () => t("ui.admin.nav.security"),
  users: () => t("ui.admin.nav.users"),
  queue: () => t("ui.admin.nav.queue"),
  feedback: () => t("ui.admin.nav.feedback"),
};

function groups(r: RecentView): Group[] {
  const out: Group[] = [];
  const group = (id: string, title: string, section: string, periods: Group["periods"], now?: Cell) =>
    out.push({ id, title, section, page: PAGE[section]?.() ?? section, periods, now });

  if (r.access) {
    const a = r.access;
    group("access", t("ui.admin.recent.access"), a.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: PERIOD_LABEL.access[p](),
      value: t("ui.admin.recent.people", { n: countText(a[p].people) }),
      sub: t("ui.admin.recent.access_sub", { logins: countText(a[p].logins), failed: countText(a[p].failed) }),
    }]), {
      label: t("ui.admin.recent.online"),
      value: t("ui.admin.recent.people", { n: countText(a.online.count) }),
      sub: t("ui.admin.security.online_value", { browser: a.online.browser, client: a.online.client }),
      tip: onlineTip(a.online),
    });
  }
  if (r.accounts) {
    const u = r.accounts;
    group("accounts", t("ui.admin.recent.accounts"), u.section, (["today", "week", "month"] as const).map((p) => [p, {
      label: PERIOD_LABEL.accounts[p](),
      value: t("ui.admin.recent.accounts_value", { n: countText(u[p].all) }),
      sub: t("ui.admin.recent.accounts_sub", { self: countText(u[p].self), admin: countText(u[p].all - u[p].self) }),
    }]));
  }
  if (r.tasks) {
    const k = r.tasks;
    group("tasks", t("ui.admin.recent.tasks"), k.section, (["today", "days7", "month"] as const).map((p) => {
      const c = k[p];
      const parts = [
        t("ui.admin.recent.done", { n: countText(c.done) }),
        ...(c.partial ? [t("ui.admin.recent.partial", { n: countText(c.partial) })] : []),
        t("ui.admin.recent.failed", { n: countText(c.failed) }),
        t("ui.admin.recent.cancelled", { n: countText(c.cancelled) }),
      ];
      return [p, {
        label: PERIOD_LABEL.tasks[p](),
        value: t("ui.admin.recent.tasks_value", { n: countText(c.all) }),
        sub: parts.join(" · "),
      }];
    }));
  }
  if (r.traffic) {
    const k = r.traffic;
    group("traffic", t("ui.admin.recent.traffic"), k.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: PERIOD_LABEL.traffic[p](),
      value: sizeText(k[p]),
    }]));
  }
  if (r.feedback) {
    const f = r.feedback;
    group("feedback", t("ui.admin.recent.feedback"), f.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: PERIOD_LABEL.feedback[p](),
      value: t("ui.admin.recent.feedback_value", { n: countText(f[p].came) }),
    }]), {
      label: t("ui.admin.recent.open"),
      value: t("ui.admin.recent.feedback_value", { n: countText(f.open) }),
      sub: t("ui.admin.recent.unread", { n: countText(f.new) }),
    });
  }
  return out;
}

function CellView({ cell }: { cell?: Cell }) {
  if (!cell) return <span className="rc-cell" />;
  return (
    <span className="rc-cell" {...tipAttrs(cell.tip)}>
      <span className="adm-tile-label">{cell.label}</span>
      <b className="tnum">{cell.value}</b>
      {cell.sub && (
        // each "label N" part stays whole: a narrow window breaks the line between parts, never inside one
        <span className="adm-tile-sub rc-sub">
          {cell.sub.split(" · ").map((part) => (
            <span key={part}>{part}</span>
          ))}
        </span>
      )}
    </span>
  );
}

export function RecentPart() {
  const { go } = useAdmin();
  const { data } = usePoll(adminApi.recent, EVERY);
  if (!data) return null;
  const shown = groups(data);
  if (!shown.length) return null;
  return (
    <div className="rc">
      <h3>{t("ui.admin.recent.title")}</h3>
      <p className="adm-lede">
        {t("ui.admin.recent.lede")}
      </p>
      <div className="rc-cards">
        {shown.map((g) => (
          <button key={g.id} className="rc-card" onClick={() => go(g.section)}>
            <span className="rc-title">
              <b>{g.title}</b>
              <span className="adm-tile-sub">{g.page}</span>
            </span>
            <CellView cell={g.now} />
            {g.periods.map(([p, c]) => (
              <CellView key={p} cell={c} />
            ))}
          </button>
        ))}
      </div>
    </div>
  );
}
