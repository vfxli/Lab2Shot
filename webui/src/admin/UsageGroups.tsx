import { Fragment, useState } from "react";
import type { UsageCounts, UsageDepartment, UsagePerson, UsageShare, UsageStats } from "../api";
import { DailyChart, HOURS, lastText, ProjectBars, REUSES, RUNS, times, used } from "./Usage";
import { durationText, fullTimeText, hoursText } from "../platform/format";
import { Button } from "../ui/Button";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 使用统计 按环节 and 按人 (lab2shot/farm/usage.py): the same range, tiles and charts as 按项目, and a table that
 * drills down department → accounts → projects, or account → projects. Everything counts, the core's nodes too. */

type Group = UsageDepartment | UsagePerson;

const sum = (rows: UsageCounts[], key: "runs" | "reuses" | "seconds" | "gpu_seconds") => rows.reduce((s, r) => s + r[key], 0);

export function GroupView({ data, by }: { data: UsageStats; by: "departments" | "people" }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const rows: Group[] = by === "departments" ? data.departments : data.people;
  const busy = rows.filter(used);
  const chosen = rows.find((r) => r.name === selected) ?? null;
  const daily = chosen?.daily ?? {
    runs: data.days.map((_, i) => rows.reduce((s, r) => s + r.daily.runs[i], 0)),
    reuses: data.days.map((_, i) => rows.reduce((s, r) => s + r.daily.reuses[i], 0)),
    seconds: data.days.map((_, i) => rows.reduce((s, r) => s + r.daily.seconds[i], 0)),
  };
  const what = by === "departments" ? t("ui.admin.usage.department_lower") : t("ui.admin.usage.person_lower");
  const select = (name: string) => setSelected((s) => (s === name ? null : name));
  const toggle = (key: string) =>
    setOpen((o) => {
      const next = new Set(o);
      if (!next.delete(key)) next.add(key);
      return next;
    });
  const listed = by === "departments" ? (rows as UsageDepartment[]).filter((d) => d.listed).length : rows.length;

  return (
    <>
      <div className="u-tiles">
        <div className="u-tile">
          <span>{by === "departments" ? t("ui.admin.usage.used_departments") : t("ui.admin.usage.used_people")}</span>
          <b>{by === "departments" ? `${(busy as UsageDepartment[]).filter((d) => d.listed).length} / ${listed}` : t("ui.admin.usage.people", { n: busy.length })}</b>
        </div>
        <div className="u-tile">
          <span>{t("ui.admin.usage.col_runs")}</span>
          <b>{times(sum(rows, "runs"))}</b>
        </div>
        <div className="u-tile">
          <span>{t("ui.admin.usage.col_reuses")}</span>
          <b>{times(sum(rows, "reuses"))}</b>
        </div>
        <div className="u-tile">
          <span>{t("ui.admin.usage.cook_time")}</span>
          <b>{hoursText(sum(rows, "seconds"))}</b>
        </div>
        <div className="u-tile">
          <span>{t("ui.admin.usage.gpu_time")}</span>
          <b>{hoursText(sum(rows, "gpu_seconds"))}</b>
        </div>
      </div>
      <div className="u-charts">
        <div className="u-panel">
          <div className="u-chart-head">
            {t("ui.admin.usage.per_what", { what })}
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
          {busy.length ? (
            <ProjectBars projects={busy.map((r) => ({ ...r, title: r.name }))} selected={selected} onSelect={select} what={what} />
          ) : (
            <p className="help-muted">{t("ui.admin.usage.nobody_cooked")}</p>
          )}
        </div>
        <div className="u-panel">
          <div className="u-chart-head">
            {t("ui.admin.usage.daily_head", { what: chosen ? chosen.name : by === "departments" ? t("ui.admin.usage.all_departments") : t("ui.admin.usage.everyone") })}
            {chosen && (
              <Button tone="ghost" size="sm" onClick={() => setSelected(null)}>
                {t("ui.admin.usage.show_all")}
              </Button>
            )}
          </div>
          <DailyChart days={data.days} daily={daily} />
        </div>
      </div>
      <div className="q-table-wrap">
        <table className={`q-table u-table u-groups u-by-${by}`}>
          <thead>
            <tr>
              <th>{by === "departments" ? t("ui.admin.usage.col_department") : t("ui.admin.usage.col_account")}</th>
              {by === "people" && <th>{t("ui.admin.usage.col_department")}</th>}
              <th>{t("ui.admin.usage.col_runs")}</th>
              <th>{t("ui.admin.usage.col_reuses")}</th>
              <th>{t("ui.admin.usage.cook_time")}</th>
              <th>{t("ui.admin.usage.gpu_time")}</th>
              <th>{t("ui.admin.usage.col_frames")}</th>
              {by === "departments" && <th>{t("ui.admin.usage.col_people")}</th>}
              <th>{t("ui.admin.usage.col_last")}</th>
            </tr>
          </thead>
          <tbody>
            {by === "departments"
              ? (rows as UsageDepartment[]).map((d) => (
                  <DepartmentRows key={d.name} d={d} nobody={data.nobody} open={open} toggle={toggle} />
                ))
              : (rows as UsagePerson[]).map((p) => (
                  <PersonRows key={p.name} p={p} nobody={data.nobody} open={open} toggle={toggle} />
                ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Cells({ c, people }: { c: UsageCounts; people?: boolean }) {
  const zero = (n: number) => (n ? "tnum" : "tnum u-zero");
  return (
    <>
      <td className={zero(c.runs)}>{times(c.runs)}</td>
      <td className={zero(c.reuses)}>{times(c.reuses)}</td>
      <td className={zero(c.seconds)} {...tipAttrs(tipOf("value", c.seconds ? durationText(c.seconds) : undefined))}>
        {hoursText(c.seconds)}
      </td>
      <td className={zero(c.gpu_seconds)} {...tipAttrs(tipOf("value", c.gpu_seconds ? durationText(c.gpu_seconds) : undefined))}>
        {hoursText(c.gpu_seconds)}
      </td>
      <td className={zero(c.frames)}>{c.frames.toLocaleString("zh-CN")}</td>
      {people && (
        <td className={zero(c.users.length)} {...tipAttrs(tipOf("value", c.users.length ? c.users.join("\n") : undefined))}>
          {t("ui.admin.usage.people", { n: c.users.length })}
        </td>
      )}
      <td className={c.last ? "tnum" : "tnum u-zero"} {...tipAttrs(tipOf("value", c.last ? fullTimeText(c.last) : undefined))}>
        {lastText(c.last)}
      </td>
    </>
  );
}

function Name({ name, nobody }: { name: string; nobody: string }) {
  return name === nobody ? (
    <span className="u-zero">
      {name}
    </span>
  ) : (
    <span>{name}</span>
  );
}

function ProjectRow({ p, depth, extra, people }: { p: UsageShare; depth: number; extra: number; people?: boolean }) {
  return (
    <tr className={`u-node u-depth-${depth}`}>
      <td>
        <span className={used(p) ? undefined : "u-zero"}>{p.title}</span>
        <span className="u-node-id mono">{p.name}</span>
      </td>
      {Array.from({ length: extra }, (_, i) => (
        <td key={i} />
      ))}
      <Cells c={p} people={people} />
    </tr>
  );
}

function DepartmentRows({ d, nobody, open, toggle }: { d: UsageDepartment; nobody: string; open: Set<string>; toggle: (key: string) => void }) {
  const shown = open.has(d.name);
  return (
    <>
      <tr className={`expandable u-dept${shown ? " u-open" : ""}`} onClick={() => toggle(d.name)}>
        <td>
          <span className="u-caret">{shown ? "▾" : "▸"}</span>
          <span className={used(d) ? undefined : "u-zero"}>
            <Name name={d.name} nobody={nobody} />
          </span>
          {!d.listed && d.name !== nobody && (
            <span className="chip u-chip">
              {t("ui.admin.usage.unlisted")}
            </span>
          )}
        </td>
        <Cells c={d} people />
      </tr>
      {shown &&
        (d.people.length ? (
          d.people.map((m) => {
            const key = `${d.name}/${m.name}`;
            const mine = open.has(key);
            return (
              <Fragment key={key}>
                <tr className={`expandable u-node u-member${mine ? " u-open" : ""}`} onClick={() => toggle(key)}>
                  <td>
                    <span className="u-caret">{mine ? "▾" : "▸"}</span>
                    <Name name={m.name} nobody={nobody} />
                  </td>
                  <Cells c={m} people />
                </tr>
                {mine && m.projects.map((p) => <ProjectRow key={p.name} p={p} depth={2} extra={0} people />)}
              </Fragment>
            );
          })
        ) : (
          <tr className="u-node">
            <td colSpan={8} className="u-zero">
              {t("ui.admin.usage.nobody_submitted")}
            </td>
          </tr>
        ))}
    </>
  );
}

function PersonRows({ p, nobody, open, toggle }: { p: UsagePerson; nobody: string; open: Set<string>; toggle: (key: string) => void }) {
  const shown = open.has(p.name);
  return (
    <>
      <tr className={`expandable u-person${shown ? " u-open" : ""}`} onClick={() => toggle(p.name)}>
        <td>
          <span className="u-caret">{shown ? "▾" : "▸"}</span>
          <Name name={p.name} nobody={nobody} />
        </td>
        <td>{p.departments.join(t("list.sep"))}</td>
        <Cells c={p} />
      </tr>
      {shown && p.projects.map((x) => <ProjectRow key={x.name} p={x} depth={1} extra={1} />)}
    </>
  );
}
