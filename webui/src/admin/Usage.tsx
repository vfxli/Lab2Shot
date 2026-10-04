import { useCallback, useEffect, useRef, useState } from "react";
import { useSignedIn } from "../state/session";
import { shown } from "../api/applies";
import { api, type UsageCounts, type UsageStats } from "../api";
import { Section } from "./common";
import { GroupView } from "./UsageGroups";
import { durationText, fullTimeText, hoursText } from "../platform/format";
import { Button, Chip, Segmented } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { msg } from "../messages/message";
import { DailyChart, ProjectBars, UsageTable } from "./UsageCharts";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

export { DailyChart, ProjectBars } from "./UsageCharts";
export type { BarRow } from "./UsageCharts";

/** 使用统计 on /admin: how much each third-party project, and each of its nodes, was used in a time range (so the
 * administrator can see which projects earn their place), and by which department and which account (按环节, 按人,
 * UsageGroups.tsx). A run computed;
 * a reuse was answered without computing (a cached result, or a worker's raw results from before). In 按项目 the tiles
 * and charts count third-party projects; the core's nodes come last in the table. */

// the marks' colors, chosen together to stay distinct on the page's dark surface: runs, reuses, hours
export const RUNS = "#0a84ff";
export const REUSES = "#199e70";
export const HOURS = "#d95926";

type View = "projects" | "departments" | "people";
const VIEWS: [View, () => string][] = [
  ["projects", () => t("ui.admin.usage.by_project")],
  ["departments", () => t("ui.admin.usage.by_department")],
  ["people", () => t("ui.admin.usage.by_person")],
];

type Preset = "7" | "30" | "90" | "all" | "custom";
const PRESETS: [Preset, () => string][] = [
  ["7", () => t("ui.admin.usage.days7")],
  ["30", () => t("ui.admin.usage.days30")],
  ["90", () => t("ui.admin.usage.days90")],
  ["all", () => t("ui.admin.usage.all")],
  ["custom", () => t("ui.admin.usage.custom")],
];

const dateText = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

/** Local midnight of a YYYY-MM-DD, `days` later, in seconds. */
function midnight(text: string, days = 0): number {
  const d = new Date(`${text}T00:00`);
  d.setDate(d.getDate() + days);
  return d.getTime() / 1000;
}

/** The range a choice means: [since, until] (null: open), or why it can't be asked for. */
function rangeOf(preset: Preset, from: string, to: string): [number | null, number | null] | string {
  if (preset === "all") return [null, null];
  if (preset !== "custom") return [midnight(dateText(new Date()), 1 - Number(preset)), null];
  if (!from || !to) return t("ui.admin.usage.custom_pick");
  if (from > to) return t("ui.admin.usage.custom_order");
  return [midnight(from), midnight(to, 1)];
}

export const times = (n: number) => t("ui.admin.usage.times", { n: n.toLocaleString("zh-CN") });

export const dayLabel = (day: string) => t("ui.admin.usage.month_day", { month: Number(day.slice(5, 7)), day: Number(day.slice(8)) });

export function lastText(at: number | null): string {
  if (!at) return t("ui.admin.usage.never_used");
  const d = new Date(at * 1000);
  const md = { month: d.getMonth() + 1, day: d.getDate() };
  return d.getFullYear() === new Date().getFullYear() ? t("ui.admin.usage.month_day", md) : t("ui.admin.usage.year_month_day", { year: d.getFullYear(), ...md });
}


/** What a project's or node's numbers say, for its tooltip. */
export function countsTip(title: string, c: UsageCounts): string {
  const time = c.seconds
    ? t("ui.admin.usage.cooked_for", { time: c.gpu_seconds ? t("ui.admin.usage.seconds_gpu", { total: durationText(c.seconds), gpu: durationText(c.gpu_seconds) }) : t("ui.admin.usage.seconds_cpu", { total: durationText(c.seconds) }) })
    : t("ui.admin.usage.not_cooked");
  return [title, t("ui.admin.usage.runs_reuses", { runs: times(c.runs), reuses: times(c.reuses) }), time, c.users.length ? t("ui.admin.usage.used_by", { n: c.users.length }) : "", c.last ? t("ui.admin.usage.last_used", { at: fullTimeText(c.last) }) : ""]
    .filter(Boolean)
    .join("\n");
}

export const used = (c: UsageCounts) => c.runs + c.reuses > 0;

/** The width a chart has (follows the page). */
export function UsageSection() {
  const [ask, confirmSheet] = useConfirm();
  const state = useSignedIn();
  const [preset, setPreset] = useState<Preset>("30");
  const [from, setFrom] = useState(() => dateText(new Date(Date.now() - 29 * 86400000)));
  const [to, setTo] = useState(() => dateText(new Date()));
  const [data, setData] = useState<UsageStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [view, setView] = useState<View>("projects");
  const range = rangeOf(preset, from, to);

  // Only the answer to the latest request is shown: switching the range while an earlier one is still on its way
  // (a long range answers slower than a short one) must not have the earlier answer land last and stay
  const latest = useRef(0);
  const load = useCallback(async () => {
    const asked = rangeOf(preset, from, to); // a preset counts back from today, whenever it is read
    if (typeof asked === "string") return;
    const n = ++latest.current;
    setLoading(true);
    try {
      const got = await api.admin.usage(...asked);
      if (n !== latest.current) return;
      setData(got);
      setError(null);
    } catch (e) {
      if (n === latest.current) setError((e as Error).message);
    } finally {
      if (n === latest.current) setLoading(false);
    }
  }, [preset, from, to]);
  useEffect(() => void load(), [load]);

  const reset = async () => {
    if (!data) return;
    const ok = await ask({ title: t("ui.admin.usage.reset_title"), say: msg("N-USAGE-RESET"), yes: t("ui.admin.usage.reset_yes"), danger: true });
    if (!ok) return;
    try {
      await api.admin.resetUsage();
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const undoReset = async () => {
    try {
      await api.admin.undoReset();
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const third = data?.projects.filter((p) => !p.core) ?? [];
  const usedThird = third.filter(used);
  const chosen = data?.projects.find((p) => p.name === selected) ?? null;
  const daily = chosen?.daily ?? {
    runs: (data?.days ?? []).map((_, i) => third.reduce((s, p) => s + p.daily.runs[i], 0)),
    reuses: (data?.days ?? []).map((_, i) => third.reduce((s, p) => s + p.daily.reuses[i], 0)),
    seconds: (data?.days ?? []).map((_, i) => third.reduce((s, p) => s + p.daily.seconds[i], 0)),
  };
  const total = (key: "runs" | "reuses" | "seconds") => third.reduce((s, p) => s + p[key], 0);
  const people = new Set(third.flatMap((p) => p.users)).size;
  const select = (name: string) => setSelected((s) => (s === name ? null : name));

  return (
    <Section
      title={t("ui.admin.nav.usage")}
      className="usage"
      actions={
        <>
          <Button tone="ghost" onClick={() => void load()}>
            {t("ui.admin.common.refresh")}
          </Button>
          {shown(state?.applies, "usage.reset") && (
            <Button tip={tipOf("consequence", t("ui.admin.usage.reset_tip"))} tone="ghost" disabled={!data} onClick={() => void reset()}>
              {t("ui.admin.usage.reset")}
            </Button>
          )}
          {data?.undo && shown(state?.applies, "usage.reset") && (
            <Button tone="ghost" onClick={() => void undoReset()}>
              {t("ui.admin.usage.undo_reset")}
            </Button>
          )}
        </>
      }
      lede={
        <>
          {view === "projects"
            ? t("ui.admin.usage.lede_projects")
            : view === "departments"
              ? t("ui.admin.usage.lede_departments")
              : t("ui.admin.usage.lede_people")}
          {data?.start ? t("ui.admin.usage.since_reset", { at: fullTimeText(data.start) }) : ""}
        </>
      }
    >
      <div className="u-filters">
        <Segmented label={t("ui.admin.usage.view_by")} layout="u-views" value={view} options={VIEWS.map(([id, label]) => ({ value: id, label: label() }))} onChange={(id) => (setView(id), setSelected(null))} />
        <Segmented label={t("ui.admin.usage.range")} value={preset} options={PRESETS.map(([id, label]) => ({ value: id, label: label() }))} onChange={setPreset} />
        {preset === "custom" && (
          <>
            <label htmlFor="usage-from">{t("ui.admin.usage.from")}</label>
            <input id="usage-from" className="field" type="date" value={from} max={to || undefined} onChange={(e) => setFrom(e.target.value)} />
            <label htmlFor="usage-to">{t("ui.admin.usage.to")}</label>
            <input id="usage-to" className="field" type="date" value={to} min={from || undefined} onChange={(e) => setTo(e.target.value)} />
          </>
        )}
      </div>
      {typeof range === "string" && <div className="notice">{range}</div>}
      {error && <div className="notice">{error}</div>}
      {data === null ? (
        <p className="help-muted">{t("ui.admin.common.reading")}</p>
      ) : view !== "projects" ? (
        <div className={`u-body${loading ? " u-loading" : ""}`}>
          <GroupView data={data} by={view} />
        </div>
      ) : (
        <div className={`u-body${loading ? " u-loading" : ""}`}>
          <div className="u-tiles">
            <div className="u-tile">
              <span>{t("ui.admin.usage.used_third")}</span>
              <b>
                {usedThird.length} / {third.length}
              </b>
            </div>
            <div className="u-tile">
              <span>{t("ui.admin.usage.col_runs")}</span>
              <b>{times(total("runs"))}</b>
            </div>
            <div className="u-tile">
              <span>{t("ui.admin.usage.col_reuses")}</span>
              <b>{times(total("reuses"))}</b>
            </div>
            <div className="u-tile">
              <span>{t("ui.admin.usage.cook_time")}</span>
              <b>{hoursText(total("seconds"))}</b>
            </div>
            <div className="u-tile">
              <span>{t("ui.admin.usage.col_users")}</span>
              <b>{t("ui.admin.usage.people", { n: people })}</b>
            </div>
          </div>
          <div className="u-charts">
            <div className="u-panel">
              <div className="u-chart-head">
                {t("ui.admin.usage.per_project")}
                <span className="u-legend">
                  <span>
                    <i style={{ background: RUNS }} />
                    {t("ui.admin.usage.col_runs")}
                  </span>
                  <span>
                    <i style={{ background: REUSES }} />
                    {t("ui.admin.usage.col_reuses")}
                  </span>
                  <span>
                    <i style={{ background: HOURS }} />
                    {t("ui.admin.usage.cook_time")}
                  </span>
                </span>
              </div>
              {usedThird.length ? (
                <ProjectBars projects={usedThird} selected={selected} onSelect={select} />
              ) : (
                <p className="help-muted">{t("ui.admin.usage.none_used")}</p>
              )}
              {third.length > usedThird.length && (
                <div className="u-unused">
                  <span>{t("ui.admin.usage.unused")}</span>
                  {third
                    .filter((p) => !used(p))
                    .map((p) => (
                      <Chip key={p.name} layout="u-chip" on={selected === p.name} onClick={() => select(p.name)}>
                        {p.title}
                      </Chip>
                    ))}
                </div>
              )}
            </div>
            <div className="u-panel">
              <div className="u-chart-head">
                {t("ui.admin.usage.daily_head", { what: chosen ? chosen.title : t("ui.admin.usage.all_third") })}
                {chosen && (
                  <Button tone="ghost" size="sm" onClick={() => setSelected(null)}>
                    {t("ui.admin.usage.show_all")}
                  </Button>
                )}
              </div>
              <DailyChart days={data.days} daily={daily} />
            </div>
          </div>
          <UsageTable projects={data.projects} selected={selected} onSelect={select} />
        </div>
      )}
      {confirmSheet}
    </Section>
  );
}
