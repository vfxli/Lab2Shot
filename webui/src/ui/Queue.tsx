import "./queue.css";
import "./queueRow.css";
import { useState } from "react";
import type { CacheMark, JobClient, JobProgress, JobState, Lane, QueueJob, QueueView as QueueData } from "../api";
import { PHASE_TEXT, progressTip } from "../api";
import type { Delivery, DeliveryState } from "../api/deliveries";
import { downloadBatch, downloadDelivery } from "../files/transfer";
import { clockText, durationText, roughlyText } from "../platform/format";
import { api } from "../api";
import { Button, Switch } from "./Button";
import { useConfirm } from "./Confirm";
import { msg } from "../messages/message";
import { say } from "../state/say";
import { sizeText } from "../platform/format";
import { pausedLanes } from "../state/pause";
import { JobTable, fromRecord as recordJob } from "./QueueTables";
import { StoragePanel } from "./Storage";
import { render } from "../messages/format";

export { JobTable, fromRecord } from "./QueueTables";

/** The farm's queue, implemented once: the editor's 队列 window and the admin page (/admin) both render it with this
 * component. One table, never two sections: a running, waiting or finished job is the same entity in a different 状态,
 * so all jobs form one row set; the editor's own finished jobs carry 缓存 and 加载 in the same table. The admin page
 * shows every job and adds 谁, 位置, the GPU switches and everything known about who started each. */

// where a running job without a GPU runs (lab2shot/engine/policy.py lanes)
export const LANE_WHERE: Record<Lane, string> = { light: "立即计算", heavy: "CPU 队列", gpu: "在服务器上算" };

const STATE: Record<JobState, [string, string]> = {
  running: ["计算中", "var(--accent)"],
  queued: ["排队", "var(--orange)"],
  done: ["完成", "var(--green)"],
  failed: ["出错", "var(--error)"],
  cancelled: ["已取消", "var(--text-3)"],
  interrupted: ["中断", "var(--text-3)"],
};

export function StateChip({ state }: { state: JobState }) {
  const [label, color] = STATE[state];
  return (
    <span className="chip q-state">
      <i style={{ background: color }} />
      {label}
    </span>
  );
}


/** When a running job is expected to finish, or a waiting one to start ("" when unknown). */
export function etaText(job: QueueJob, now: number): string {
  if (!job.eta || (job.state !== "running" && job.state !== "queued")) return "";
  const left = job.eta.at - now;
  if (left < 1) return job.state === "running" ? "快算完了" : "马上开始";
  const about = job.eta.partial ? "至少" : "约";
  return job.state === "running" ? `剩余${about} ${roughlyText(left)}` : `${about} ${roughlyText(left)}后开始`;
}

/** 提交时刻与耗时，分为两列。 */
export const submittedAt = (j: QueueJob): string => clockText(j.submitted);

export function elapsed(j: QueueJob, now: number): string {
  if (j.state === "queued") return `已等 ${durationText(now - j.submitted)}`;
  if (j.state === "running") return `已算 ${durationText(now - (j.started ?? now))}`;
  // 未运行的任务没有耗时可言：状态列已显示「中断」，此处再写「没开始」属于重复
  return j.started && j.finished ? durationText(j.finished - j.started) : "";
}

/** 该行的结果：只显示几种状态，详情放在悬停提示中。交付物仍存在时提供「下载」。 */
export function Outcome({ job }: { job: QueueJob }) {
  if (job.state === "failed") return <span className="q-out bad" data-tip={job.error ?? "没写原因"}>失败</span>;
  const outs = (job.outputs ?? []).filter((d) => d.state !== "expired");
  const all = job.outputs ?? [];
  // 没有交付物的行不显示任何内容：状态列已显示「完成」「中断」，此处再写「算完了」「已取消」
  // 属于重复。只有存在交付物时才显示下方的标记（待取回 / 已保存 / 部分）
  if (!all.length) return null;
  const worst = all.find((d) => (d.state ?? "pending") === "pending") ?? all.find((d) => d.state === "expired") ?? all[0];
  const [label, color, tip] = DELIVERY_STATE[worst.state ?? "pending"];
  const detail = all.map((d) => `「${d.label}」${d.mode === "folder" ? `${d.name}/` : d.name}：${d.files.length} 个文件 · ${DELIVERY_STATE[d.state ?? "pending"][0]}`).join("\n");
  return (
    <span className="q-out">
      <span className="chip q-state" data-tip={`${all.length} 项交付：\n${detail}\n\n${tip}`}>
        <i style={{ background: color }} />
        {all.length > 1 ? `${label} · ${all.length} 项` : label}
      </span>
      {outs.length > 0 && (
        <Button tip={outs.length > 1 ? `下载全部 ${outs.length} 项（合成一个包，每条一个子文件夹）` : outs[0].mode === "folder" ? "下载（打成一个 tar）" : "下载"}
                tone="ghost" size="xs" onClick={(e) => (e.stopPropagation(), downloadHere(outs))}>
          下载
        </Button>
      )}
    </span>
  );
}

/** 取回一个任务的交付物：逐项处理产生的 N 个包合并为一次下载（浏览器不允许连续询问 N 次），
 * 其余逐个下载。一个「输出」的 N 条按 run+node 归为一组。 */
function downloadHere(outs: Delivery[]): void {
  const groups = new Map<string, Delivery[]>();
  for (const d of outs) {
    const k = `${d.run}/${d.node}`;
    groups.set(k, [...(groups.get(k) ?? []), d]);
  }
  for (const group of groups.values()) {
    if (group.length > 1 && group.every((d) => !!d.item)) downloadBatch(group);
    else group.forEach(downloadDelivery);
  }
}

const ETA_TIP = "按以往同样节点的用时，按帧数和分辨率换算；计算中的任务再按当前进度修正。“至少”表示有节点还没有用时记录";

const DELIVERY_STATE: Record<DeliveryState, [string, string, string]> = {
  pending: ["待取回", "var(--orange)", "还没有存到提交它的电脑上：打开提交它的那个浏览器，或者在这里下载"],
  saved: ["已保存", "var(--green)", "提交它的电脑已经存好了"],
  downloaded: ["已下载", "var(--green)", "整个包已经作为浏览器下载发出去了"],
  dismissed: ["不要了", "var(--text-3)", "提交它的人说不要了"],
  expired: ["已过期", "var(--text-3)", "服务器只保留几天，已经删掉了：要的话再算一次（有缓存很快）"],
};

/** What a job's 「输出」 delivered, what became of each, and a download while the server still has it. */
function Deliveries({ outputs }: { outputs: Delivery[] }) {
  return (
    <div className="q-deliveries">
      {outputs.map((d) => {
        const [label, color, tip] = DELIVERY_STATE[d.state ?? "pending"];
        return (
          <span key={d.node} className="q-delivery">
            <span className="mono q-dname" data-user-data data-tip={`「${d.label}」：${d.files.length} 个文件`}>
              {d.mode === "folder" ? `${d.name}/` : d.name}
            </span>
            <span className="chip q-state" data-tip={tip}>
              <i style={{ background: color }} />
              {label}
            </span>
            {d.state !== "expired" && (
              <Button tip={d.mode === "folder" ? "下载（打成一个 tar）" : "下载"} tone="ghost" size="xs" onClick={(e) => (e.stopPropagation(), downloadDelivery(d))}>
                下载
              </Button>
            )}
          </span>
        );
      })}
    </div>
  );
}

export function Progress({ job, now }: { job: QueueJob; now: number }) {
  if (job.state === "failed") return <span className="q-error" data-tip={job.error ?? ""}>{job.error}</span>;
  if (job.outputs?.length && (job.state === "done" || job.state === "cancelled")) return <Deliveries outputs={job.outputs} />;
  if (job.state === "cancelled" && job.reason) return <span className="q-muted" data-tip={job.reason}>{job.reason}</span>;
  const eta = etaText(job, now);
  if (job.state === "queued") return eta ? <span className="q-eta tnum" data-tip={ETA_TIP}>{eta}</span> : null;
  if (job.state !== "running") return null;
  // 计算进度只有一套（api/progress.ts）：节点读取的也是同一份，服务器只发送这一份
  const live = "phase" in job.now ? (job.now as JobProgress) : null;
  // a job of a 逐项处理 block is cooked in 计算单元 (farm/units.py): how many of its items are finished, and how many
  // cards it is currently spread over
  const units = job.units && job.units.items ? job.units : null;
  return (
    <div className="q-progress">
      {/* 「节点 · 阶段 · 解算器报告的步骤」，例如：「SAM 3D Body 全身动作 · 计算中 · 检测人物」 */}
      <span className="q-now">{job.stopping ? "正在停止…" : (live ? [live.label, PHASE_TEXT[live.phase], live.note].filter(Boolean).join(" · ") : "") || "准备"}</span>
      {units && (
        <span className="q-units tnum" data-tip={render("I-UNITS-TIP", { total: units.items, done: units.done, running: units.running })}>
          {render("I-UNITS-ITEMS", { done: units.done, total: units.items })}
        </span>
      )}
      {!!job.cards && job.cards > 1 && (
        <span className="q-units tnum" data-tip={render("I-UNITS-CARDSTIP", { count: job.cards })}>
          {render("I-UNITS-CARDS", { count: job.cards })}
        </span>
      )}
      {/* 进度条只依据 `at`：整个任务的完成度，服务器保证其单调不减（lab2shot/progress.py）。
          不使用解算器当前步骤的 done / total 计算宽度：分母在每个阶段都会变化，进度条会归零重来。
          无法估计时（任务中有节点首次计算，没有历史记录）绘制一条无刻度的进度条。 */}
      {live && (
        <span className="q-bar" data-tip={progressTip(live)}>
          <i className={live.at == null ? "indeterminate" : ""} style={live.at == null ? { width: "30%" } : { width: `${live.at * 100}%` }} />
        </span>
      )}
      {eta && <span className="q-eta tnum" data-tip={ETA_TIP}>{eta}</span>}
    </div>
  );
}

// what the request revealed and what clients report about themselves (lab2shot/server/auth.py details,
// webui/src/client.ts, lab2shot/client.py)
const DETAIL_LABELS: Record<string, string> = {
  ip: "IP 地址", user_agent: "User-Agent", hostname: "计算机名", user: "系统用户", platform: "平台", language: "语言",
  timezone: "时区", screen: "屏幕", python: "Python 版本", pid: "进程号",
};

/** Everything known about who started a job (the administrator's view): the account, then what the request revealed. */
export function ClientDetail({ client }: { client: JobClient }) {
  const rows: [string, unknown][] = [
    ["账号", client.username],
    ["中文名", client.name],
    ["部门", client.department],
    ["应用", client.app],
    ...Object.entries(client.details ?? {}).map(([k, v]): [string, unknown] => [DETAIL_LABELS[k] ?? k, v]),
  ];
  return (
    <dl className="q-detail">
      {rows
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd className={k === "User-Agent" || k === "客户端 id" ? "mono" : undefined}>{String(v)}</dd>
          </div>
        ))}
    </dl>
  );
}

function SwitchesRow({ switches, admin, onSwitch }: { switches: QueueData["switches"]; admin: boolean; onSwitch?: (key: "gpu" | "compute", on: boolean) => void }) {
  const paused = pausedLanes(switches);
  if (!onSwitch && !paused.length) return null;
  return (
    <div className="q-switches">
      {admin && onSwitch ? (
        <>
          {switches.gpu !== undefined && (
            <label className="q-switch-row" data-tip="关：所有显卡都不再接任务；正在算的算完为止，排队的等重新打开后接着算。导入、读取序列和看已算好的结果不用显卡，不受影响">
              <Switch on={switches.gpu} label="显卡任务" onChange={(on) => onSwitch?.("gpu", on)} />
              显卡任务
            </label>
          )}
          <label className="q-switch-row" data-tip="关：服务器不再接受新的计算任务，要显卡的、CPU 队列和交付都会被拒绝，界面上说明现在只能查看；正在算的算完为止">
            <Switch on={switches.compute} label="计算任务" onChange={(on) => onSwitch?.("compute", on)} />
            计算任务
          </label>
        </>
      ) : (
        <>
          {paused.includes("compute") && <span className="chip q-paused">计算任务已暂停：现在只能查看</span>}
          {paused.includes("gpu") && <span className="chip q-paused">显卡任务已暂停：要显卡的任务先排队等着</span>}
        </>
      )}
    </div>
  );
}

export function QueueView({
  data,
  admin = false,
  onCancel,
  onLoad,
  onSwitch,
  graphUrl,
  onReorder,
  onRefresh,
}: {
  data: QueueData;
  admin?: boolean;
  onCancel: (id: string) => void;
  onLoad?: (id: string) => void; // the editor: open one of the account's jobs again
  // re-read the queue immediately (the editor's poll): after cleanup, the 我的占用 bar and the row's 缓存 mark
  // must be correct immediately rather than at the next tick
  onRefresh?: () => void;
  onSwitch?: (key: "gpu" | "compute", on: boolean) => void;
  graphUrl?: (id: string) => string;
  // 拖拽插队, admin page only: the dragged job and the position within its own lane where it was dropped (1-based)
  onReorder?: (job: QueueJob, position: number) => void;
}) {
  const active = data.jobs.filter((j) => j.state === "queued" || j.state === "running");
  // 单一表格，不分上下两部分：计算中、排队中、已完成的任务属于同一类，区别仅在「状态」列。
  // 编辑器中只列出自己的任务：账号之间相互隔离，本就看不到他人的任务，列出「别人」没有意义。
  // 后台列出全部任务，因为管理员需要查看全局。
  const done = new Set<string>();
  const rows: { job: QueueJob; cache?: CacheMark | null }[] = onLoad
    ? [
        ...active.filter((j) => j.mine).map((job) => ({ job, cache: null })),
        // 已完成的任务接在后面，最新的在前；上方已列出的不再重复
        ...(data.history ?? [])
          .filter((r) => !active.some((j) => j.mine && j.id === r.id) && !done.has(r.id) && (done.add(r.id) || true))
          .map((r) => ({ job: recordJob(r, true), cache: r.cache })),
      ]
    : [...active, ...data.jobs.filter((j) => !active.includes(j))].map((job) => ({ job }));
  const waiting = rows.some(({ job }) => job.state === "queued");
  const ahead = active.filter((j) => j.state === "queued" && !j.mine).length;  // 排在前面的他人任务数
  // 腾出空间后（删除一条任务或「删除全部」）：占用条自行重读，队列也重读一次，表中各行
  // 以及顶栏「提交」是否置灰随之更新，无需等待下一次轮询
  const [cleaned, setCleaned] = useState(0);
  const cleanedOnce = () => {
    setCleaned((n) => n + 1);
    onRefresh?.();
  };
  // 「删除全部」：删除自己所有已结束的任务（排队和计算中的任务须先取消，因此不计入）
  const finished = rows.filter(({ job }) => job.state !== "queued" && job.state !== "running");
  const [wiping, setWiping] = useState(false);
  const [askWipe, wipeSheet] = useConfirm();
  const wipe = async () => {
    if (!(await askWipe({
      title: "删除全部任务",
      say: msg("N-QUEUE-FORGETALL", { count: finished.length }),
      yes: "删除全部",
      tip: "删掉全部已经结束的任务，连同它们占的空间",
      danger: true,
    }))) return;
    setWiping(true);
    try {
      const got = await api.forgetAllJobs();
      say(msg("I-QUEUE-FORGOTALL", { jobs: got.jobs, size: sizeText(got.bytes) }));
      cleanedOnce();
    } finally {
      setWiping(false);
    }
  };
  return (
    <div className="q-view">
      <SwitchesRow switches={data.switches} admin={admin} onSwitch={onSwitch} />
      {onLoad && (
        <>
          {/* 我的占用：放在队列中，与占用空间的任务相邻 */}
          <div className="sec-title">
            我的占用
            <span className="q-hint">自己在服务器上占的硬盘；删掉任务就腾出它占的空间，删完接着算</span>
          </div>
          <StoragePanel again={cleaned} />
        </>
      )}
      {/* 服务器配置（核数、CPU、内存、硬盘）不在此处：机器状态见后台的「概览」页。 */}
      {/* 编辑器中此行只显示前方的任务数。后台显示「任务 N」，因为管理员需要总量。 */}
      <div className="sec-title">
        {onLoad ? (
          <>
            <span className="tnum" data-tip="别人排在你前面的任务数。算什么、是谁都看不到——账号之间是隔离的">前面任务：{ahead}</span>
            {/* 一键腾出空间。腾出空间的操作均在任务上：此处删除全部，行上删除单条，「我的占用」一段只显示数值。
                控件不得时隐时现：没有已结束的任务时置灰并注明原因，位置不变。 */}
            {/* 不使用 ghost 样式：透明无边框时显示为一行灰字，与旁边「前面任务：0」的说明难以区分，置灰后更难辨认。
                与行上的「取消 / 删除 / 加载」使用同一种按钮，因为它们属于同一类可点击的操作 */}
            <Button
              size="sm"
              layout="q-forget-all"
              disabled={!finished.length || wiping}
              tip={finished.length
                ? `删掉你全部 ${finished.length} 条已经结束的任务，连同它们占的空间（交付包、只有它们用到的缓存和素材）。正在排队和计算的不动`
                : "你现在没有已经结束的任务可以删（正在排队和计算的要先取消）"}
              onClick={() => void wipe()}
            >
              {wiping ? "删除中…" : "删除全部"}
            </Button>
          </>
        ) : (
          <>
            任务 <span className="q-count tnum">{rows.length}</span>
            {waiting && onReorder && <span className="q-hint">拖左边的手柄可以改顺序</span>}
          </>
        )}
      </div>
      {wipeSheet}
      {rows.length ? (
        <JobTable jobs={rows} admin={admin} onCancel={onCancel} onForgotten={cleanedOnce} onLoad={onLoad} graphUrl={graphUrl} onReorder={onReorder} />
      ) : (
        // 不写「有人提交计算后在此列出」：账号之间相互隔离，他人的任务本就不会出现在此处。
        // 为空时只显示一项有用的信息：当前排队的任务数。
        <div className="q-empty">
          {onLoad ? "你还没有提交过计算：点右上角「计算」以后，这里列出来" : "队列里没有任务"}
          {ahead > 0 ? `；前面排队的有 ${ahead} 个` : ""}
        </div>
      )}
    </div>
  );
}
