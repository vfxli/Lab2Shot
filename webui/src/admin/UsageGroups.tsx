import { Fragment, useState } from "react";
import type { UsageCounts, UsageDepartment, UsagePerson, UsageShare, UsageStats } from "../api";
import { countsTip, DailyChart, HOURS, lastText, ProjectBars, REUSES, RUNS, times, used } from "./Usage";
import { durationText, fullTimeText, hoursText } from "../platform/format";
import { Button } from "../ui/Button";

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
  const what = by === "departments" ? "环节" : "人";
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
        <div className="u-tile" data-tip={by === "departments" ? "这段时间有人提交过计算的环节 / 设置里的环节" : "这段时间提交过计算的人"}>
          <span>{by === "departments" ? "用过的环节" : "用过的人"}</span>
          <b>{by === "departments" ? `${(busy as UsageDepartment[]).filter((d) => d.listed).length} / ${listed}` : `${busy.length} 人`}</b>
        </div>
        <div className="u-tile">
          <span>计算</span>
          <b>{times(sum(rows, "runs"))}</b>
        </div>
        <div className="u-tile">
          <span>复用</span>
          <b>{times(sum(rows, "reuses"))}</b>
        </div>
        <div className="u-tile">
          <span>计算时长</span>
          <b>{hoursText(sum(rows, "seconds"))}</b>
        </div>
        <div className="u-tile" data-tip="计算时长里在显卡上的部分">
          <span>显卡时长</span>
          <b>{hoursText(sum(rows, "gpu_seconds"))}</b>
        </div>
      </div>
      <div className="u-charts">
        <div className="u-panel">
          <div className="u-chart-head">
            各{what}
            <span className="u-legend">
              <span>
                <i style={{ background: RUNS }} />
                计算
              </span>
              <span>
                <i style={{ background: REUSES }} />
                复用
              </span>
              <span>
                <i style={{ background: HOURS }} />
                计算时长
              </span>
            </span>
          </div>
          {busy.length ? (
            <ProjectBars projects={busy.map((r) => ({ ...r, title: r.name }))} selected={selected} onSelect={select} what={what} />
          ) : (
            <p className="help-muted">这段时间没有人提交过计算</p>
          )}
        </div>
        <div className="u-panel">
          <div className="u-chart-head">
            每天 · {chosen ? chosen.name : by === "departments" ? "全部环节" : "所有人"}
            {chosen && (
              <Button tip="不再只看选中的一组，列出全部" tone="ghost" size="sm" onClick={() => setSelected(null)}>
                看全部
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
              <th data-tip={by === "departments" ? "账号的环节；点开看这个环节的人" : "账号：中文名和用户名；点开看这个账号用的项目"}>{by === "departments" ? "环节" : "账号"}</th>
              {by === "people" && <th data-tip="账号的环节">环节</th>}
              <th data-tip="真正计算的次数">计算</th>
              <th data-tip="没有计算就给出结果的次数：用了缓存，或者用了以前算好的原始结果">复用</th>
              <th data-tip="真正计算用的时间，排队和等内存不算">计算时长</th>
              <th data-tip="计算时长里在显卡上的部分">显卡时长</th>
              <th data-tip="计算处理的帧数">帧数</th>
              {by === "departments" && <th data-tip="这段时间这个环节里提交过计算的账号数">人数</th>}
              <th data-tip="重置以来最近一次使用，不限于所选区间">最近使用</th>
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
      <td className={zero(c.seconds)} data-tip={c.seconds ? durationText(c.seconds) : undefined}>
        {hoursText(c.seconds)}
      </td>
      <td className={zero(c.gpu_seconds)} data-tip={c.gpu_seconds ? durationText(c.gpu_seconds) : undefined}>
        {hoursText(c.gpu_seconds)}
      </td>
      <td className={zero(c.frames)}>{c.frames.toLocaleString("zh-CN")}</td>
      {people && (
        <td className={zero(c.users.length)} data-tip={c.users.length ? c.users.join("\n") : undefined}>
          {c.users.length} 人
        </td>
      )}
      <td className={c.last ? "tnum" : "tnum u-zero"} data-tip={c.last ? fullTimeText(c.last) : undefined}>
        {lastText(c.last)}
      </td>
    </>
  );
}

function Name({ name, nobody, tip }: { name: string; nobody: string; tip: string }) {
  return name === nobody ? (
    <span className="u-zero" data-tip={`没选环节的账号（比如管理员自己）：在「用户」里给它选一个环节\n\n${tip}`}>
      {name}
    </span>
  ) : (
    <span data-tip={tip}>{name}</span>
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
            <Name name={d.name} nobody={nobody} tip={countsTip(d.name, d)} />
          </span>
          {!d.listed && d.name !== nobody && (
            <span className="chip u-chip" data-tip="这个环节已经不在设置的环节列表里，以前的记录照旧列出">
              已不在列表
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
                    <Name name={m.name} nobody={nobody} tip={countsTip(m.name, m)} />
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
              这段时间没有人提交
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
          <Name name={p.name} nobody={nobody} tip={countsTip(p.name, p)} />
        </td>
        <td>{p.departments.join("、")}</td>
        <Cells c={p} />
      </tr>
      {shown && p.projects.map((x) => <ProjectRow key={x.name} p={x} depth={1} extra={1} />)}
    </>
  );
}
