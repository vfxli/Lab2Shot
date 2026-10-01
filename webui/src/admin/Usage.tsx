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
const VIEWS: [View, string, string][] = [
  ["projects", "按项目", "每个三方项目和它的每个节点用了多少"],
  ["departments", "按环节", "每个环节用了多少，点开看它的人和他们用的项目"],
  ["people", "按人", "每个账号用了多少，点开看他用的项目"],
];

type Preset = "7" | "30" | "90" | "all" | "custom";
const PRESETS: [Preset, string][] = [
  ["7", "近 7 天"],
  ["30", "近 30 天"],
  ["90", "近 90 天"],
  ["all", "全部"],
  ["custom", "自定义"],
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
  if (!from || !to) return "自定义区间：选好开始和结束的日期";
  if (from > to) return "自定义区间：开始的日期要早于结束的日期";
  return [midnight(from), midnight(to, 1)];
}

export const times = (n: number) => `${n.toLocaleString("zh-CN")} 次`;

export const dayLabel = (day: string) => `${Number(day.slice(5, 7))}月${Number(day.slice(8))}日`;

export function lastText(t: number | null): string {
  if (!t) return "没用过";
  const d = new Date(t * 1000);
  const md = `${d.getMonth() + 1}月${d.getDate()}日`;
  return d.getFullYear() === new Date().getFullYear() ? md : `${d.getFullYear()}年${md}`;
}


/** What a project's or node's numbers say, for its tooltip. */
export function countsTip(title: string, c: UsageCounts): string {
  const time = c.seconds ? `计算 ${durationText(c.seconds)}${c.gpu_seconds ? `，显卡上 ${durationText(c.gpu_seconds)}` : "，都在 CPU 上"}` : "没有计算";
  return [title, `计算 ${times(c.runs)} · 复用 ${times(c.reuses)}`, time, c.users.length ? `${c.users.length} 人用过` : "", c.last ? `最近使用 ${fullTimeText(c.last)}` : ""]
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
    const ok = await ask({ title: "重置使用统计", say: msg("N-USAGE-RESET"), yes: "清零", tip: "使用统计从现在起重新算，可以撤销", danger: true });
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
      title="使用统计"
      className="usage"
      actions={
        <>
          <Button tip="重新读取使用统计" tone="ghost" onClick={() => void load()}>
            刷新
          </Button>
          {shown(state?.applies, "usage.reset") && (
            <Button tip="统计清零，从现在起重新算；任务记录一个不删，可以撤销" tone="ghost" disabled={!data} onClick={() => void reset()}>
              重置
            </Button>
          )}
          {data?.undo && shown(state?.applies, "usage.reset") && (
            <Button tip="回到上一次重置之前的起点" tone="ghost" onClick={() => void undoReset()}>
              撤销重置
            </Button>
          )}
        </>
      }
      lede={
        <>
          {view === "projects"
            ? "每个三方项目和它的每个节点被用了多少：真正算了一遍记一次「计算」，沿用以前的结果记一次「复用」，时间按任务结束的时候算。" +
              "装了的项目都列出来，这段时间没用过的是 0。上面的数字和图只算三方项目，核心节点在表格最后。每一次计算都算在内。"
            : view === "departments"
              ? "每个环节用了多少，点一个环节看它的人，再点一个人看他用的项目。环节是账号的环节（在「用户」里改），核心节点也算在内；设置里的环节都列出来，已经不在列表里的环节排在后面。" +
                "没选环节的账号（比如管理员自己）记在「未分环节」一行。"
              : view === "people"
                ? "每个账号用了多少，点一个账号看他用的项目。核心节点也算在内。删掉的账号合在「已删除的用户」一行；有账号以前的任务都算在管理员名下。"
                : "每张模板卡片被用来提交了几次任务：节点图从哪张模板打开，任务就记在哪张名下，不管后来改了多少；不是从模板打开的记在「自己搭的」。" +
                  "时间按任务提交的时候算；从这一版起才开始记，以前的任务不算。模板改了名按现在的名字，删掉的按最后一次用时的名字并标「已删除」。"}
          {data?.start ? `上次重置在 ${fullTimeText(data.start)}，统计从那时算起。` : ""}
        </>
      }
    >
      <div className="u-filters">
        <Segmented label="按什么看" layout="u-views" value={view} options={VIEWS.map(([id, label, tip]) => ({ value: id, label, tip }))} onChange={(id) => (setView(id), setSelected(null))} />
        <Segmented label="时间范围" value={preset} options={PRESETS.map(([id, label]) => ({ value: id, label, tip: id === "custom" ? "自己选开始和结束的日期" : `统计${label}` }))} onChange={setPreset} />
        {preset === "custom" && (
          <>
            <label htmlFor="usage-from">从</label>
            <input id="usage-from" className="field" type="date" data-tip="从这一天起算（含这一天）" value={from} max={to || undefined} onChange={(e) => setFrom(e.target.value)} />
            <label htmlFor="usage-to">到</label>
            <input id="usage-to" className="field" type="date" data-tip="算到这一天为止（含这一天）" value={to} min={from || undefined} onChange={(e) => setTo(e.target.value)} />
          </>
        )}
      </div>
      {typeof range === "string" && <div className="notice">{range}</div>}
      {error && <div className="notice">{error}</div>}
      {data === null ? (
        <p className="help-muted">读取中…</p>
      ) : view !== "projects" ? (
        <div className={`u-body${loading ? " u-loading" : ""}`}>
          <GroupView data={data} by={view} />
        </div>
      ) : (
        <div className={`u-body${loading ? " u-loading" : ""}`}>
          <div className="u-tiles">
            <div className="u-tile">
              <span>用过的三方项目</span>
              <b>
                {usedThird.length} / {third.length}
              </b>
            </div>
            <div className="u-tile">
              <span>计算</span>
              <b>{times(total("runs"))}</b>
            </div>
            <div className="u-tile">
              <span>复用</span>
              <b>{times(total("reuses"))}</b>
            </div>
            <div className="u-tile">
              <span>计算时长</span>
              <b>{hoursText(total("seconds"))}</b>
            </div>
            <div className="u-tile">
              <span>用户</span>
              <b>{people} 人</b>
            </div>
          </div>
          <div className="u-charts">
            <div className="u-panel">
              <div className="u-chart-head">
                各项目
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
              {usedThird.length ? (
                <ProjectBars projects={usedThird} selected={selected} onSelect={select} />
              ) : (
                <p className="help-muted">这段时间没有用过任何三方项目</p>
              )}
              {third.length > usedThird.length && (
                <div className="u-unused">
                  <span>这段时间没用过</span>
                  {third
                    .filter((p) => !used(p))
                    .map((p) => (
                      <Chip key={p.name} layout="u-chip" tip={countsTip(p.title, p)} on={selected === p.name} onClick={() => select(p.name)}>
                        {p.title}
                      </Chip>
                    ))}
                </div>
              )}
            </div>
            <div className="u-panel">
              <div className="u-chart-head">
                每天 · {chosen ? chosen.title : "全部三方项目"}
                {chosen && (
                  <Button tip="不再只看选中的一组，列出全部" tone="ghost" size="sm" onClick={() => setSelected(null)}>
                    看全部
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
