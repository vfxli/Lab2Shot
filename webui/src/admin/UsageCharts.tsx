/** 使用统计's charts and table: bars per project, runs per day, the sortable table with its node rows. */

import { useRef, useState, type RefObject } from "react";
import type { UsageCounts, UsageProject } from "../api";
import { durationText, fullTimeText, hoursText } from "../platform/format";
import { useRefSize } from "../platform/size";
import { HOURS, REUSES, RUNS, countsTip, dayLabel, lastText, times, used } from "./Usage";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

function useWidth(): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  return [ref, useRefSize(ref).w];
}

/** A round top for an axis: 1, 2, 5 × a power of ten. */
function niceMax(v: number): number {
  if (v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const m = v / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

const tick = (v: number) => (v >= 100 || Number.isInteger(v) ? v.toLocaleString("zh-CN") : String(Number(v.toPrecision(2))));

/** A bar growing from x (or up from y when `up`), its data end rounded. */
function bar(x: number, y: number, w: number, h: number, up = false): string {
  const r = Math.min(4, up ? w / 2 : h / 2, up ? h : w);
  if (up)
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

const LABEL_CHARS = 22;
const short = (text: string) => (text.length > LABEL_CHARS ? `${text.slice(0, LABEL_CHARS - 1)}…` : text);

/** A row of the bar chart: a project, a department, a person. */
export type BarRow = UsageCounts & { name: string; title: string };

/** Per row (project, department, person): how often it was used (runs, then reuses, one bar) and how long it computed. */
export function ProjectBars({ projects, selected, onSelect, what = t("ui.admin.usage.project_lower") }: { projects: BarRow[]; selected: string | null; onSelect: (name: string) => void; what?: string }) {
  const [ref, width] = useWidth();
  const ROW = 24;
  const THICK = 12;
  const HEAD = 26;
  const VALUE = 72; // room for the number at a bar's end
  const labelW = Math.min(180, Math.max(110, width * 0.3));
  const panel = Math.max(40, (width - labelW - 2 * VALUE - 16) / 2);
  const hoursX = labelW + panel + VALUE + 16;
  const maxUses = Math.max(...projects.map((p) => p.runs + p.reuses), 1);
  const maxSeconds = Math.max(...projects.map((p) => p.seconds), 1);
  const height = HEAD + projects.length * ROW;
  return (
    <div ref={ref} className="u-chart">
      {width > 0 && (
        <svg className="u-svg" width={width} height={height} role="img" aria-label={t("ui.admin.usage.bars_label", { what })}>
          <text x={labelW} y={12} className="u-axis-title">
            {t("ui.admin.usage.uses")}
          </text>
          <text x={hoursX} y={12} className="u-axis-title">
            {t("ui.admin.usage.cook_time")}
          </text>
          {projects.map((p, i) => {
            const y = HEAD + i * ROW;
            const top = y + (ROW - THICK) / 2;
            const runsW = (p.runs / maxUses) * panel;
            const reusesW = (p.reuses / maxUses) * panel;
            const gap = runsW > 0 && reusesW > 0 ? 2 : 0;
            const hoursW = (p.seconds / maxSeconds) * panel;
            return (
              <g key={p.name} className={`u-row${selected === p.name ? " on" : ""}`} onClick={() => onSelect(p.name)} {...tipAttrs(tipOf("value", countsTip(p.title, p)))}>
                <rect className="u-row-bg" x={0} y={y} width={width} height={ROW} rx={5} />
                <text x={labelW - 10} y={y + ROW / 2} textAnchor="end" dominantBaseline="central" className="u-label">
                  {short(p.title)}
                </text>
                {runsW > 0 && <path d={reusesW > 0 ? `M${labelW},${top}h${runsW}v${THICK}h${-runsW}Z` : bar(labelW, top, runsW, THICK)} fill={RUNS} />}
                {reusesW > 0 && <path d={bar(labelW + runsW + gap, top, Math.max(reusesW - gap, 1), THICK)} fill={REUSES} />}
                <text x={labelW + runsW + reusesW + 6} y={y + ROW / 2} dominantBaseline="central" className="u-value">
                  {times(p.runs + p.reuses)}
                </text>
                {hoursW > 0 && <path d={bar(hoursX, top, Math.max(hoursW, 1), THICK)} fill={HOURS} />}
                <text x={hoursX + hoursW + 6} y={y + ROW / 2} dominantBaseline="central" className="u-value">
                  {hoursText(p.seconds)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

/** Per day of the range: uses (runs, then reuses) and compute hours, two charts on one time line. */
export function DailyChart({ days, daily }: { days: string[]; daily: UsageProject["daily"] }) {
  const [ref, width] = useWidth();
  const GUTTER = 44;
  const H = 104; // a chart, its unit on top
  const PAD = 22; // room for the unit above the top gridline
  const GAP = 18;
  const AXIS = 20;
  const plot = Math.max(40, width - GUTTER - 6);
  const band = plot / Math.max(days.length, 1);
  const barW = Math.max(1, Math.min(24, band * 0.7));
  const uses = days.map((_, i) => daily.runs[i] + daily.reuses[i]);
  const hours = daily.seconds.map((s) => s / 3600);
  const usesTop = niceMax(Math.max(...uses, 0));
  const hoursTop = niceMax(Math.max(...hours, 0));
  const every = Math.max(1, Math.ceil(days.length / Math.max(1, Math.floor(plot / 58))));
  const charts: [string, number, number, boolean][] = [
    [t("ui.admin.usage.unit_times"), 0, usesTop, true],
    [t("ui.admin.usage.unit_hours"), H + GAP, hoursTop, false],
  ];
  const height = 2 * H + GAP + AXIS;
  const x = (i: number) => GUTTER + i * band + (band - barW) / 2;
  return (
    <div ref={ref} className="u-chart">
      {width > 0 && (
        <svg className="u-svg" width={width} height={height} role="img" aria-label={t("ui.admin.usage.daily_label")}>
          {charts.map(([unit, top, max, counts]) => (
            <g key={unit}>
              {[0, 0.5, 1].map((f) => (
                <g key={f}>
                  <line x1={GUTTER} x2={width} y1={top + PAD + (1 - f) * (H - PAD)} y2={top + PAD + (1 - f) * (H - PAD)} className="u-grid" />
                  {(!counts || Number.isInteger(max * f)) && (
                    <text x={GUTTER - 6} y={top + PAD + (1 - f) * (H - PAD)} textAnchor="end" dominantBaseline="central" className="u-tick">
                      {tick(max * f)}
                    </text>
                  )}
                </g>
              ))}
              <text x={GUTTER - 6} y={top + 4} textAnchor="end" dominantBaseline="hanging" className="u-axis-title">
                {unit}
              </text>
            </g>
          ))}
          {days.map((day, i) => {
            const scale = (v: number, max: number) => (v / max) * (H - PAD);
            const runsH = scale(daily.runs[i], usesTop);
            const reusesH = scale(daily.reuses[i], usesTop);
            const gap = runsH > 0 && reusesH > 0 ? 2 : 0;
            const hoursH = scale(hours[i], hoursTop);
            const base = H;
            return (
              <g key={day}>
                {runsH > 0 && <path d={reusesH > 0 ? `M${x(i)},${base - runsH}h${barW}v${runsH}h${-barW}Z` : bar(x(i), base - runsH, barW, runsH, true)} fill={RUNS} />}
                {reusesH > 0 && <path d={bar(x(i), base - runsH - reusesH, barW, Math.max(reusesH - gap, 1), true)} fill={REUSES} />}
                {hoursH > 0 && <path d={bar(x(i), 2 * H + GAP - Math.max(hoursH, 1), barW, Math.max(hoursH, 1), true)} fill={HOURS} />}
                {(i % every === 0 || i === days.length - 1) && (days.length - 1 - i >= every || i === days.length - 1) && (
                  <text x={GUTTER + (i + 0.5) * band} y={height - 4} textAnchor="middle" className="u-tick">
                    {`${Number(day.slice(5, 7))}/${Number(day.slice(8))}`}
                  </text>
                )}
                <rect
                  className="u-col"
                  x={GUTTER + i * band}
                  y={0}
                  width={Math.max(band, 1)}
                  height={2 * H + GAP}
                  {...tipAttrs(tipOf("value", t("ui.admin.usage.day_tip", { day: dayLabel(day), runs: times(daily.runs[i]), reuses: times(daily.reuses[i]), hours: hoursText(daily.seconds[i]) })))}
                />
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

type SortKey = "title" | "runs" | "reuses" | "seconds" | "frames" | "users" | "last";
const COLUMNS: [SortKey, () => string][] = [
  ["title", () => t("ui.admin.usage.col_project")],
  ["runs", () => t("ui.admin.usage.col_runs")],
  ["reuses", () => t("ui.admin.usage.col_reuses")],
  ["seconds", () => t("ui.admin.usage.cook_time")],
  ["frames", () => t("ui.admin.usage.col_frames")],
  ["users", () => t("ui.admin.usage.col_users")],
  ["last", () => t("ui.admin.usage.col_last")],
];

const sortValue = (p: UsageCounts & { title?: string; subtitle?: string }, key: SortKey): number | string =>
  key === "title" ? (p.title ?? p.subtitle ?? "").toLowerCase() : key === "users" ? p.users.length : key === "last" ? (p.last ?? 0) : p[key];

function CountCells({ c }: { c: UsageCounts }) {
  const zero = (n: number) => (n ? "tnum" : "tnum u-zero");
  return (
    <>
      <td className={zero(c.runs)}>{times(c.runs)}</td>
      <td className={zero(c.reuses)}>{times(c.reuses)}</td>
      <td className={zero(c.seconds)} {...tipAttrs(tipOf("value", c.seconds ? (c.gpu_seconds ? t("ui.admin.usage.seconds_gpu", { total: durationText(c.seconds), gpu: durationText(c.gpu_seconds) }) : t("ui.admin.usage.seconds_cpu", { total: durationText(c.seconds) })) : undefined))}>
        {hoursText(c.seconds)}
      </td>
      <td className={zero(c.frames)}>{c.frames.toLocaleString("zh-CN")}</td>
      <td className={zero(c.users.length)} {...tipAttrs(tipOf("value", c.users.length ? c.users.join("\n") : undefined))}>
        {t("ui.admin.usage.people", { n: c.users.length })}
      </td>
      <td className={c.last ? "tnum" : "tnum u-zero"} {...tipAttrs(tipOf("value", c.last ? fullTimeText(c.last) : undefined))}>
        {lastText(c.last)}
      </td>
    </>
  );
}

export function UsageTable({ projects, selected, onSelect }: { projects: UsageProject[]; selected: string | null; onSelect: (name: string) => void }) {
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "runs", desc: true });
  const order = (rows: UsageProject[]) =>
    [...rows].sort((a, b) => {
      const va = sortValue(a, sort.key);
      const vb = sortValue(b, sort.key);
      return (va < vb ? -1 : va > vb ? 1 : 0) * (sort.desc ? -1 : 1) || b.runs + b.reuses - a.runs - a.reuses;
    });
  const third = order(projects.filter((p) => !p.core));
  const core = projects.filter((p) => p.core);
  return (
    <div className="q-table-wrap">
      <table className="q-table u-table">
        <thead>
          <tr>
            {COLUMNS.map(([key, label]) => (
              <th
                key={key}
                className={`u-sort${sort.key === key ? " on" : ""}`}
                onClick={() => setSort((s) => (s.key === key ? { key, desc: !s.desc } : { key, desc: key !== "title" }))}
              >
                {label()}
                {sort.key === key ? (sort.desc ? " ↓" : " ↑") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...third, ...core].map((p) => (
            <ProjectRows key={p.name} project={p} open={selected === p.name} onSelect={onSelect} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ProjectRows({ project: p, open, onSelect }: { project: UsageProject; open: boolean; onSelect: (name: string) => void }) {
  return (
    <>
      <tr className={`expandable${p.core ? " u-core" : ""}${open ? " u-open" : ""}`} onClick={() => onSelect(p.name)}>
        <td>
          <span className="u-caret">{open ? "▾" : "▸"}</span>
          <span className={used(p) ? undefined : "u-zero"}>{p.title}</span>
          {!p.installed && (
            <span className="chip u-chip">
              {t("ui.admin.usage.not_installed")}
            </span>
          )}
        </td>
        <CountCells c={p} />
      </tr>
      {open &&
        p.nodes.map((n) => (
          <tr key={n.id} className="u-node">
            <td>
              <span className={used(n) ? undefined : "u-zero"}>{n.subtitle}</span>
              <span className="u-node-id mono">{n.id}</span>
            </td>
            <CountCells c={n} />
          </tr>
        ))}
    </>
  );
}
