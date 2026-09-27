/** The queue's tables: a job row with its client, the job log, the account's finished jobs with their cache marks. */

import { useEffect, useState } from "react";
import { api } from "../api";
import type { CacheMark, JobRecord, QueueJob } from "../api";
import { fullTimeText, roughlyText } from "../platform/format";
import { Button, ButtonLink } from "./Button";
import { useConfirm } from "./Confirm";
import { msg, reasonOf, textOf } from "../messages/message";
import { say } from "../state/say";
import { ClientDetail, LANE_WHERE, Outcome, Progress, StateChip, elapsed, submittedAt } from "./Queue";
import { DRAG_QUEUED_ONLY, DRAG_QUEUE_TIP, useRowDrag } from "./rowDrag";

/** 该任务计算的帧范围。目标节点属于节点图内部信息，查看者并不关心，因此放在节点图一格的悬停提示中。 */
const framesText = (j: { frames?: [number, number] | null }): string =>
  j.frames ? (j.frames[0] === j.frames[1] ? `${j.frames[0]}` : `${j.frames[0]}–${j.frames[1]}`) : "全部";

function JobRow({ job, now, admin, onCancel, onForgotten, onLoad, cache, graphUrl, grip }: { job: QueueJob; now: number; admin: boolean; onCancel: (id: string) => void; onForgotten?: () => void; onLoad?: (id: string) => void; cache?: CacheMark | null; graphUrl?: (id: string) => string; grip?: GripProps }) {
  const [open, setOpen] = useState(false);
  const [ask, confirmSheet] = useConfirm();
  const active = job.state === "queued" || job.state === "running";
  const canDrag = job.state === "queued";
  // 任务记录 keeps a job's state but not its position in the line (fromRecord: position null): a job that was still
  // waiting when the record was written must not display 「第 null 位」
  const where = job.state === "queued" ? `${job.lane === "heavy" ? "CPU 队列" : job.lane === "light" ? "等空位" : ""}${job.position == null ? "排队中" : `第 ${job.position} 位`}${[textOf(job.waiting), textOf(job.waiting_detail)].filter(Boolean).map((w) => ` · ${w}`).join("")}` : LANE_WHERE[job.lane];
  const client = job.client;
  const [forgetting, setForgetting] = useState(false);
  const forget = async () => {
    if (!(await ask({ title: "删除任务", say: msg("N-JOB-FORGET", { title: job.title || "这个任务" }), yes: "删除", danger: true,
                      tip: "删了找不回来" }))) return;
    setForgetting(true);
    try {
      await (admin ? api.admin.forget(job.id) : api.forgetJob(job.id));
      say(msg("I-JOB-FORGOTTEN", { title: job.title || "这个任务" }));
      onForgotten?.();
    } catch (e) {
      say(msg("E-JOB-FORGETFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setForgetting(false);
    }
  };
  return (
    <>
      <tr className={`${job.mine ? "mine" : ""}${admin ? " expandable" : ""}${grip?.over ? " drag-over" : ""}`} onClick={admin ? () => setOpen(!open) : undefined} {...(grip?.row ?? {})}>
        {grip && (
          // 拖拽插队：手柄始终存在，不可拖动时置灰并注明原因，位置不变
          <td className="q-grip-cell">
            <span className={`q-grip${canDrag ? "" : " off"}`} data-tip={canDrag ? DRAG_QUEUE_TIP : DRAG_QUEUED_ONLY} onClick={(e) => e.stopPropagation()} {...(canDrag ? grip.handle : {})}>
              ⠿
            </span>
          </td>
        )}
        <td>
          <StateChip state={job.state} />
        </td>
        <td className="q-body">
          {/* 每一行固定为两行：上方为名称，下方为一排数值。名称与数值不能放在同一个会换行的 flex 中：
              内容多的行会折到第二行，内容少的行（中断且无耗时、无交付、无缓存）则挤在名称右侧不换行，
              同一列内容在各行中位置不一致，无法对齐。名称单独一行、数值单独一行，数值行为固定的四列网格
              （帧范围 | 提交 | 用时 | 剩余），各列跨行对齐；缺少的值留空，位置不变。 */}
          <div className="q-line-name">
            {/* 名称：用户自己的数据，是唯一可截断的项，占用整行宽度 */}
            <span className="q-title" data-user-data={job.title ? "" : undefined} data-tip={job.title
              ? [job.title, (job.targets ?? []).length ? `算到这些节点为止：${(job.targets ?? []).join("、")}` : ""].filter(Boolean).join("\n")
              : "别人的任务：算的是什么只有提交的人和管理员看得到"}>
              {job.title || "别人的任务"}
            </span>
            {admin && (
              <span className="q-facts">
                {client ? (
                  <>
                    <span className="q-who">{client.who}</span>
                    {client.department && <span className="chip q-dept" data-tip="账号的部门">{client.department}</span>}
                    {job.mine && <span className="chip q-mine">我的</span>}
                    {client.app !== "web" && <span className="q-app">{client.app}</span>}
                  </>
                ) : (
                  <span className="q-who q-muted" data-tip="别人的任务：只看得到它排在哪、大概多久，看不到是谁、算什么">别人</span>
                )}
              </span>
            )}
          </div>
          <div className="q-line-facts tnum">
            <span className="q-f" data-tip="算的是哪几帧">{framesText(job)}</span>
            {/* 悬停提示中显示年月日：表中只显示时分（列窄且跨行对齐），但最近 30 条可能跨越多天，
                仅凭「17:26」无法判断日期 */}
            <span className="q-f" data-tip={`提交时间：${fullTimeText(job.submitted)}`}>{submittedAt(job)}</span>
            <span className="q-f" data-tip="用了多久">{elapsed(job, now)}</span>
            {/* 已完成的行：交付标记和缓存标记各占一格，因此跨行时也对齐。
                计算中的行进度区域较宽，占据后两格（.q-run 的 grid-column: 4 / -1），
                每行只有一个进度区域 */}
            {active ? (
              <span className="q-f q-run">
                {/* 排队位置只在排队中的行显示。后台表格有独立的「位置」列，此处不重复 */}
                {!admin && job.state === "queued" && where && <span data-tip="排在第几位">{where}</span>}
                <Progress job={job} now={now} />
              </span>
            ) : (
              <>
                <span className="q-f q-deliv">
                  <Outcome job={job} />
                </span>
                <span className="q-f q-cached">{onLoad && <CacheCell cache={cache ?? null} />}</span>
              </>
            )}
          </div>
        </td>
        <td className="q-act">
          <Button
            tip={!active ? "这个任务已经结束了，没什么可取消的" : !(job.mine || admin) ? "别人的任务只有他自己和管理员能取消"
              : job.state === "queued" ? "移出队列" : "停下这个任务（已经算好的节点保留在缓存里）"}
            disabled={job.stopping || !active || !(job.mine || admin)}
            onClick={(e) => {
              e.stopPropagation();
              onCancel(job.id);
            }}
          >
            {job.stopping ? "停止中…" : "取消"}
          </Button>
          {/* 删除：仅对已结束的任务可用，计算中的任务须先「取消」。这是腾出空间的操作：该行记录、交付包、
              仅被该任务使用的缓存及上传素材一并删除；其他任务或模板仍在引用的一律保留
              （server/quota.py drop_for_job）。不提供单独清理缓存的按钮，腾出空间只有这一途径。 */}
          <Button
            tip={active ? "这个任务还在算，删不掉：先点「取消」，停下来之后再删"
              : !(job.mine || admin) ? "别人的任务只有他自己和管理员能删"
                : "把这一行从队列里删掉，连同它占的空间：交付包、只有它用到的缓存、只有它用到的上传素材（别的任务和模板还用得上的留着）"}
            disabled={active || !(job.mine || admin) || forgetting}
            onClick={(e) => (e.stopPropagation(), void forget())}
          >
            {forgetting ? "删除中…" : "删除"}
          </Button>
          {onLoad && (
            <Button
              tip={active ? "还在算，算完了才能把它当时的节点图打开" : "打开这个任务提交时的节点图（节点、参数、连线、视图），作为一张新的还没保存的节点图；结果还在缓存里的马上就能看"}
              disabled={active}
              onClick={(e) => (e.stopPropagation(), onLoad(job.id))}
            >
              加载
            </Button>
          )}
        </td>
      </tr>
      {confirmSheet}
      {admin && open && client && (
        <tr className="q-expanded">
          <td colSpan={grip ? 4 : 3}>
            <ClientDetail client={client} />
            {job.error && <pre className="q-error-full">{job.error}</pre>}
            {graphUrl && (
              <ButtonLink tip="在新标签页打开这个任务提交时的节点图（JSON）" tone="ghost" href={graphUrl(job.id)} target="_blank" rel="noreferrer">
                看节点图
              </ButtonLink>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

/** A job from the job log, shown like a finished job of the queue. */
export const fromRecord = (r: JobRecord, mine = false): QueueJob => ({
  ...r,
  outputs: (r.outputs ?? []).filter((o) => typeof o === "object" && Array.isArray(o?.files)), // records from before deliveries listed plain paths are skipped
  eta: null,
  position: null,
  waiting: null,
  now: {},
  stopping: false,
  // 编辑器的队列窗口只列出自己的任务（账号之间相互隔离，服务器只返回本人的 history），
  // 因此传入 true；后台「任务记录」列出所有人的任务，传入 false（管理员依靠权限，而非此标记）。
  // 若固定为 false，「删除」在自己已完成的任务上会一直置灰
  mine,
});

/** What a draggable row needs: the handle's props, the row's props, and whether the pointer is over it. */
interface GripProps {
  handle: Record<string, unknown>;
  row: Record<string, unknown>;
  over: boolean;
}

export function JobTable({ jobs, admin, onCancel, onForgotten, onLoad, graphUrl, onReorder }: {
  jobs: { job: QueueJob; cache?: CacheMark | null }[];
  admin: boolean;
  onCancel: (id: string) => void;
  onForgotten?: () => void; // 删除了一行：重新读取队列（该行已移除，占用也可能变化）
  // 编辑器：一张表即全部内容（计算中、排队中、已完成只是状态不同），因此行中增加「缓存」和「加载」
  onLoad?: (id: string) => void;
  graphUrl?: (id: string) => string;
  // 拖拽插队 (lab2shot/farm/queue.py Farm.reorder): the dragged job and the position within its own lane where it was
  // dropped (1-based). Omitted: the table is not draggable (the editor's 队列 window, the job log).
  onReorder?: (job: QueueJob, position: number) => void;
}) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, []);
  // a drag moves a job within its own lane, so it may only be dropped on that lane's other waiting jobs, and its new
  // position is counted among them. Any other target is refused during the drag (rowDrag's `can`: the row is not
  // highlighted and the pointer shows 禁止); a drop is never silently ignored.
  const at = (i: number) => jobs[i]?.job;
  const waitingWith = (moved: QueueJob) => jobs.filter(({ job: j }) => j.state === "queued" && j.lane === moved.lane).map((r) => r.job);
  const canDrop = (from: number, to: number) =>
    at(from)?.state === "queued" && at(to)?.state === "queued" && at(to)!.lane === at(from)!.lane;
  const drag = useRowDrag((from, to) => {
    const moved = at(from);
    if (moved) onReorder?.(moved, waitingWith(moved).indexOf(at(to)!) + 1);
  }, canDrop);
  return (
    <div className="q-table-wrap">
      <table className="q-table q-jobs">
        <thead>
          {/* 三格布局，而非每列一格：列数固定、界面文字不得换行截断、弹层宽度固定，三者同时成立时，
              多列表格必然在某些内容下溢出（出现横向滚动条），调整列宽只是将溢出转移到另一种内容。
              因此只设三格：状态 | 主体 | 操作。主体格内容可以换行，名称是唯一可截断的项
              （用户自己的数据），操作靠右且不压缩。任何内容、任何宽度下均不会溢出。 */}
          <tr>
            {onReorder && <th className="q-grip-cell" />}
            <th>状态</th>
            <th>任务</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {jobs.map(({ job: j, cache }, i) => (
            <JobRow
              key={j.id || `load-${i}`}
              job={j}
              cache={cache}
              now={now}
              admin={admin}
              onCancel={onCancel}
              onForgotten={onForgotten}
              onLoad={onLoad}
              graphUrl={graphUrl}
              grip={onReorder ? { handle: drag.grip(i), row: drag.row(i), over: drag.over === i } : undefined}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

const MARK: Record<CacheMark["mark"], [string, string]> = {
  all: ["全在", "var(--green)"],
  some: ["部分", "var(--orange)"],
  none: ["已清理", "var(--text-3)"],
};

/** 清理后重新计算所需时间的完整描述，统一在一处生成：缓存标记的悬停提示与「清缓存」确认框均原样使用
 * （拼接半句会产生「大概 多久还不知道」这类不通顺的文字）。 */
const againText = (cache: CacheMark): string =>
  cache.seconds > 0
    ? `重新算大概 ${roughlyText(cache.seconds)}${cache.unknown ? "，有节点还没有用时记录，只会更久" : ""}`
    : cache.unknown
      ? "重新算要多久还不知道：有节点还没有用时记录"
      : "重新算很快";

/** Whether a finished job's results are still cached (determined by the server from the job's graph), and 清理: frees
 * this job's own cached results on behalf of the account (lab2shot/server/quota.py clean_job). 清理后该节点图仍可正常
 * 加载和计算，只是需要重新计算。 */
function CacheCell({ cache }: { cache: CacheMark | null }) {
  // 不提供「清理」按钮：腾出空间的操作均在任务上，该行的「删除」已同时移除仅被它使用的缓存；
  // 再设一个仅清理缓存的按钮会使同一操作有两个入口，且清理后仍留下一行空任务，用户会误以为空间未释放。
  // 需要一次性清理时使用队列上方的「删除全部」。
  if (!cache) return null;
  const [label, color] = MARK[cache.mark];
  const again = againText(cache);
  const tip =
    cache.mark === "all"
      ? "结果都还在缓存里：加载后马上就能看，不用重新算"
      : cache.mark === "some"
        ? `${cache.nodes} 个节点里 ${cache.cached} 个的结果还在缓存里，别的要重新算；${again}`
        : `结果已经从缓存里清理了${cache.why ? `（${cache.why}）` : ""}：加载后要重新算；${again}`;
  return (
    <span className="q-cache">
      <span className="chip q-state" data-tip={tip}>
        <i style={{ background: color }} />
        {label}
      </span>
    </span>
  );
}

