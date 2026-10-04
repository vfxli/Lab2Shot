import "./queue.css";
import "./queueRow.css";
import { useState } from "react";
import type { CacheMark, JobClient, JobProgress, JobRecord, JobState, QueueJob, QueueView as QueueData } from "../api";
import { PHASE_TEXT, progressTip } from "../api";
import type { Output } from "../api/files";
import type { Availability } from "../api/applies";
import { OutputDownload, outputTip } from "./OutputDownload";
import { clockText, durationText } from "../platform/format";
import { api } from "../api";
import { Button, Switch } from "./Button";
import { useConfirm } from "./Confirm";
import { msg, reasonOf } from "../messages/message";
import { say } from "../state/say";
import { sizeText } from "../platform/format";
import { pausedLanes } from "../state/pause";
import { JobTable, fromRecord as recordJob } from "./QueueTables";
import { StoragePanel, useStorageUsage } from "./Storage";
import { t } from "../i18n/t";
import { listSep } from "../i18n/words";
import { tipAttrs, tipOf } from "../platform/tips";

export { JobTable, fromRecord } from "./QueueTables";

/** 计算农场的队列，只实现一次：编辑器的「队列」窗口和后台（/admin）都用这个组件绘制。
 * 单一表格，不分两部分：计算中、排队中、已完成的任务是同一类东西，区别只在「状态」，因此所有任务是同一组行；
 * 编辑器中自己已完成的任务在同一张表里带「缓存」和「加载」。后台列出全部任务，任务记录按同样方式并入，
 * 并增加「谁」、显卡开关，以及关于每个任务提交者的全部已知信息。 */

/** 计算中的任务在哪里算：其节点当前所在的显卡（仅对有权查看显卡的人），否则只写「在服务器上算」。 */
export const whereOf = (job: QueueJob): string => (job.cards?.length ? job.cards.join(listSep()) : t("ui.queue.on_server"));

const STATE: Record<JobState, [string, string]> = {
  running: ["ui.queue.state_running", "var(--accent)"],
  queued: ["ui.queue.state_queued", "var(--orange)"],
  done: ["ui.queue.state_done", "var(--green)"],
  partial: ["ui.queue.state_partial", "var(--orange)"],
  failed: ["ui.queue.state_failed", "var(--error)"],
  cancelled: ["ui.queue.state_cancelled", "var(--text-3)"],
  interrupted: ["ui.queue.state_interrupted", "var(--text-3)"],
};

export function StateChip({ state }: { state: JobState }) {
  const [labelKey, color] = STATE[state];
  return (
    <span className="chip q-state">
      <i style={{ background: color }} />
      {t(labelKey)}
    </span>
  );
}


/** 提交时刻与耗时，分为两列。 */
export const submittedAt = (j: QueueJob): string => clockText(j.submitted);

export function elapsed(j: QueueJob, now: number): string {
  if (j.state === "queued") return t("ui.queue.waited", { time: durationText(now - j.submitted) });
  if (j.state === "running") return t("ui.queue.cooked_for", { time: durationText(now - (j.started ?? now)) });
  // 未运行的任务没有耗时可言：状态列已显示「中断」，此处再写「没开始」属于重复
  return j.started && j.finished ? durationText(j.finished - j.started) : "";
}

/** 该行的结果：失败时说原因；这个任务的「输出」打包好的 zip 还在服务器上时，每个一个「下载」（浏览器自己下载，
 * ui/OutputDownload.tsx）。随任务过了保留天数、已删掉的只说一声，不给按钮。 */
export function Outcome({ job }: { job: QueueJob }) {
  if (job.state === "failed") return <span className="q-out bad" {...tipAttrs(tipOf("error", job.error ?? t("ui.queue.no_reason")))}>{t("ui.queue.failed")}</span>;
  const all = job.outputs ?? [];
  // 部分失败：出错的节点标红，用不到它的照常算完；打包好的「输出」照常能下载，没有时说一声哪里出错
  if (job.state === "partial" && !all.length) return <span className="q-out bad" {...tipAttrs(tipOf("error", job.error ?? t("ui.queue.no_reason")))}>{t("ui.queue.state_partial")}</span>;
  // 没有打包结果的行不显示任何内容：状态列已显示「完成」「中断」，此处再写「算完了」「已取消」属于重复
  if (!all.length) return null;
  const kept = all.filter((o) => !o.gone);
  if (!kept.length) return <span className="q-out" {...tipAttrs(tipOf("consequence", t("ui.queue.expired_tip")))}>{t("ui.queue.expired")}</span>;
  return (
    <span className="q-out">
      {kept.map((o) => <OutputDownload key={o.pkg} output={o} />)}
    </span>
  );
}

/** 已完成任务的「输出」打包出的内容：每个 zip 的名称，以及任务保留期间的下载。 */
function Outputs({ outputs }: { outputs: Output[] }) {
  return (
    <div className="q-outputs">
      {outputs.map((o) => (
        <span key={o.pkg} className="q-output">
          <span className="mono q-dname" data-user-data {...tipAttrs(tipOf("value", o.gone ? t("ui.queue.output_gone", { label: o.label, name: o.name }) : t("ui.queue.output_tip", { label: o.label, tip: outputTip(o).text })))}>
            {o.name}
          </span>
          <OutputDownload output={o} />
        </span>
      ))}
    </div>
  );
}

// 不显示预计还要多久、多久后开始：按以往用时推算的时间不准。排队的行只有「已等」（elapsed），计算中的行是进度条和「已算」。
export function Progress({ job }: { job: QueueJob }) {
  if (job.outputs?.length && (job.state === "done" || job.state === "partial" || job.state === "cancelled")) return <Outputs outputs={job.outputs} />;
  if (job.state === "failed" || job.state === "partial") return <span className="q-error" {...tipAttrs(tipOf("truncated", job.error))}>{job.error}</span>;
  if (job.state === "cancelled" && job.reason) return <span className="q-muted" {...tipAttrs(tipOf("truncated", job.reason))}>{job.reason}</span>;
  if (job.state !== "running") return null;
  // 计算进度只有一套（api/progress.ts）：节点读取的也是同一份，服务器只发送这一份
  const live = "phase" in job.now ? (job.now as JobProgress) : null;
  return (
    <div className="q-progress">
      {/* 「节点 · 阶段 · 解算器报告的步骤」，例如：「SAM 3D Body 全身动作 · 计算中 · 检测人物」 */}
      <span className="q-now">{job.stopping ? t("ui.queue.stopping_now") : (live ? [live.label, PHASE_TEXT[live.phase], live.note].filter(Boolean).join(" · ") : "") || t("ui.queue.preparing")}</span>
      {/* 进度条只依据 `at`：整个任务的完成度，服务器保证其单调不减（lab2shot/progress.py）。
          不使用解算器当前步骤的 done / total 计算宽度：分母在每个阶段都会变化，进度条会归零重来。
          无法估计时（任务中有节点首次计算，没有历史记录）绘制一条无刻度的进度条。 */}
      {live && (
        <span className="q-bar" {...tipAttrs(tipOf("value", progressTip(live)))}>
          <i className={live.at == null ? "indeterminate" : ""} style={live.at == null ? { width: "30%" } : { width: `${live.at * 100}%` }} />
        </span>
      )}
    </div>
  );
}

// 请求本身透露的信息，以及客户端自报的信息（lab2shot/server/auth.py details、
// webui/src/platform/client.ts、lab2shot/client.py）
const DETAIL_LABELS: Record<string, string> = {
  ip: "ui.queue.detail.ip", user_agent: "ui.queue.detail.user_agent", hostname: "ui.queue.detail.hostname", user: "ui.queue.detail.user",
  platform: "ui.queue.detail.platform", language: "ui.queue.detail.language", timezone: "ui.queue.detail.timezone",
  screen: "ui.queue.detail.screen", python: "ui.queue.detail.python", pid: "ui.queue.detail.pid",
};

/** 关于任务提交者的全部已知信息（管理员视图）：先是账号，然后是请求透露的信息。 */
export function ClientDetail({ client }: { client: JobClient }) {
  const rows: [string, unknown][] = [
    [t("ui.queue.detail.account"), client.username],
    [t("ui.queue.detail.name"), client.name],
    [t("ui.queue.detail.department"), client.department],
    [t("ui.queue.detail.app"), client.app],
    ...Object.entries(client.details ?? {}).map(([k, v]): [string, unknown] => [DETAIL_LABELS[k] ? t(DETAIL_LABELS[k]) : k, v]),
  ];
  return (
    <dl className="q-detail">
      {rows
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd className={k === "User-Agent" ? "mono" : undefined}>{String(v)}</dd>
          </div>
        ))}
    </dl>
  );
}

function SwitchesRow({ switches, admin, onSwitch, diskLow }: { switches: QueueData["switches"]; admin: boolean; onSwitch?: (key: "gpu" | "compute", on: boolean) => void; diskLow?: boolean }) {
  const paused = pausedLanes(switches);
  if (!onSwitch && !paused.length && !diskLow) return null;
  return (
    <div className="q-switches">
      {admin && onSwitch ? (
        <>
          {switches.gpu !== undefined && (
            <label className="q-switch-row" {...tipAttrs(tipOf("consequence", t("ui.queue.gpu_switch_tip")))}>
              <Switch on={switches.gpu} label={t("ui.queue.gpu_jobs")} onChange={(on) => onSwitch?.("gpu", on)} />
              {t("ui.queue.gpu_jobs")}
            </label>
          )}
          <label className="q-switch-row" {...tipAttrs(tipOf("consequence", t("ui.queue.compute_switch_tip")))}>
            <Switch on={switches.compute} label={t("ui.queue.compute_jobs")} onChange={(on) => onSwitch?.("compute", on)} />
            {t("ui.queue.compute_jobs")}
          </label>
        </>
      ) : (
        <>
          {paused.includes("compute") && <span className="chip q-paused">{t("ui.queue.compute_paused")}</span>}
          {paused.includes("gpu") && <span className="chip q-paused">{t("ui.queue.gpu_paused")}</span>}
        </>
      )}
      {/* 数据盘低于「暂停新计算的剩余空间」（lab2shot/farm/policy.py space）：所有账号的新计算暂停，腾出空间后自动恢复 */}
      {diskLow && <span className="chip q-paused">{t("ui.queue.disk_low")}</span>}
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
  onFirst,
  onRefresh,
  history,
  applies,
}: {
  data: QueueData;
  admin?: boolean;
  // 后台：任务记录（GET /api/admin/history），并入同一张表，排在队列自身的任务之后
  history?: JobRecord[] | null;
  onCancel: (id: string) => void;
  onLoad?: (id: string) => void; // 编辑器：重新打开本账号的某个任务
  // 立即重读队列（编辑器的轮询）：腾出空间后，「我的占用」条和该行的「缓存」标记必须立即正确，而非等到下一次轮询
  onRefresh?: () => void;
  onSwitch?: (key: "gpu" | "compute", on: boolean) => void;
  graphUrl?: (id: string) => string;
  // 插队，仅后台：把尚未结束的任务挪到队首（lab2shot/farm/queue.py Farm.first）
  onFirst?: (job: QueueJob) => void;
  // 后台：本次登录的可用性答复，说明能否删除他人的任务、为他人的组改名
  // （server/available.py ACTIONS queue.forget / queue.forgetgroup / queue.rename）
  applies?: Availability | null;
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
    : [
        ...active.map((job) => ({ job })),
        // 后台：已结束的任务接在后面，最新的在前。队列中仍保留的（刚结束的，状态最新）优先，任务记录补上更早的；
        // 同一任务只列一次，进行中的任务不会再以记录的形式出现
        ...[...data.jobs.filter((j) => !active.includes(j)), ...(history ?? []).map((r) => recordJob(r))]
          .filter((j) => !active.some((a) => a.id === j.id) && !done.has(j.id) && (done.add(j.id) || true))
          .sort((a, b) => b.submitted - a.submitted)
          .map((job) => ({ job })),
      ];
  const waiting = rows.some(({ job }) => job.state === "queued");
  const ahead = active.filter((j) => j.state === "queued" && !j.mine).length;  // 排在前面的他人任务数
  // 腾出空间后（删除一条任务或「删除全部」）：占用条自行重读，队列也重读一次，表中各行
  // 以及「计算」是否置灰（graph/actions.ts cookHold）随之更新，无需等待下一次轮询
  const [cleaned, setCleaned] = useState(0);
  const cleanedOnce = () => {
    setCleaned((n) => n + 1);
    onRefresh?.();
  };
  // 自己的占用：「我的占用」一段和表格里每个已结束任务的占用（删掉它腾出多少）读同一份，删除之后重读（后台不读）
  const { usage, problem } = useStorageUsage(onLoad ? cleaned : null);
  // 「删除全部」：删除自己所有已结束的任务（排队和计算中的任务须先取消，因此不计入）
  const finished = rows.filter(({ job }) => job.state !== "queued" && job.state !== "running");
  const [wiping, setWiping] = useState(false);
  const [askWipe, wipeSheet] = useConfirm();
  const wipe = async () => {
    if (!(await askWipe({
      title: t("ui.queue.wipe_title"),
      say: msg("N-QUEUE-FORGETALL", { count: finished.length }),
      yes: t("ui.queue.wipe"),
      tip: tipOf("consequence", t("ui.queue.wipe_tip")),
      danger: true,
    }))) return;
    setWiping(true);
    try {
      const got = await api.forgetAllJobs();
      say(msg("I-QUEUE-FORGOTALL", { jobs: got.jobs, size: sizeText(got.bytes) }));
      cleanedOnce();
    } catch (e) {
      say(msg("E-JOB-FORGETFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setWiping(false);
    }
  };
  return (
    <div className="q-view">
      <SwitchesRow switches={data.switches} admin={admin} onSwitch={onSwitch} diskLow={!!onLoad && !!data.disk_low} />
      {onLoad && (
        <>
          {/* 我的占用：放在队列中，与占用空间的任务相邻 */}
          <div className="sec-title">
            {t("ui.queue.my_usage")}
            <span className="q-hint">{t("ui.queue.my_usage_hint")}</span>
          </div>
          <StoragePanel usage={usage} problem={problem} onFreed={cleanedOnce} />
        </>
      )}
      {/* 服务器配置（核数、CPU、内存、硬盘）不在此处：机器状态见后台的「概览」页。 */}
      {/* 编辑器中此行只显示前方的任务数。后台显示「任务 N」，因为管理员需要总量。 */}
      <div className="sec-title">
        {onLoad ? (
          <>
            <span className="tnum">{t("ui.queue.ahead", { count: ahead })}</span>
            {/* 一键腾出空间。腾出空间的操作均在任务上：此处删除全部，行上删除单条，「我的占用」一段只显示数值。
                控件不得时隐时现：没有已结束的任务时置灰并注明原因，位置不变。 */}
            {/* 不使用 ghost 样式：透明无边框时显示为一行灰字，与旁边「前面任务：0」的说明难以区分，置灰后更难辨认。
                与行上的「取消 / 删除 / 加载」使用同一种按钮，因为它们属于同一类可点击的操作 */}
            <Button
              size="sm"
              layout="q-forget-all"
              disabled={!finished.length || wiping}
              tip={finished.length
                ? tipOf("consequence", t("ui.queue.wipe_space"))
                : tipOf("disabled", t("ui.queue.wipe_none"))}
              onClick={() => void wipe()}
            >
              {wiping ? t("ui.queue.deleting") : t("ui.queue.wipe")}
            </Button>
          </>
        ) : (
          <>
            {t("ui.queue.jobs")} <span className="q-count tnum">{rows.length}</span>
            {waiting && onFirst && <span className="q-hint">{t("ui.queue.first_hint")}</span>}
          </>
        )}
      </div>
      {wipeSheet}
      {rows.length ? (
        <JobTable jobs={rows} admin={admin} applies={applies} onCancel={onCancel} onForgotten={cleanedOnce} onLoad={onLoad} graphUrl={graphUrl} onFirst={onFirst}
                  sizes={onLoad ? usage?.tasks : undefined} groupSizes={onLoad ? usage?.groups : undefined} />
      ) : (
        // 不写「有人提交计算后在此列出」：账号之间相互隔离，他人的任务本就不会出现在此处。
        // 为空时只显示一项有用的信息：当前排队的任务数。
        <div className="q-empty">
          {onLoad ? t("ui.queue.empty_mine") : t("ui.queue.empty")}
          {ahead > 0 ? t("ui.queue.empty_ahead", { count: ahead }) : ""}
        </div>
      )}
    </div>
  );
}
