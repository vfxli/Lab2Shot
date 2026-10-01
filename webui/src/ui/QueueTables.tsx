/** 队列的表格：任务行及其提交端信息、任务记录、本账号已完成的任务及其缓存标记。 */

import { useEffect, useState, type ReactNode } from "react";
import { api } from "../api";
import type { CacheMark, JobRecord, JobState, QueueJob, TaskGroup } from "../api";
import { clockText, fullTimeText, sizeText } from "../platform/format";
import { Button, ButtonLink } from "./Button";
import { useConfirm } from "./Confirm";
import { msg, reasonOf, textOf } from "../messages/message";
import { say } from "../state/say";
import { ClientDetail, Outcome, Progress, StateChip, elapsed, submittedAt, whereOf } from "./Queue";
import { NameSheet } from "./NameSheet";
import { shown, why, type Availability } from "../api/applies";

/** 某一行是否提供对其任务的操作（删除、删除组、组改名），以及当前不可用的原因。编辑器中账号通过自己的路由操作
 * 自己的任务：始终提供。后台通过管理路由操作，以服务器为本次登录解析的结果为准（`actionId`，server/available.py
 * ACTIONS）：登录不具备该路由的权限时不提供，权限已用尽时置灰并说明原因。 */
function offer(admin: boolean, applies: Availability | null | undefined, actionId: string): { there: boolean; why: string } {
  if (!admin) return { there: true, why: "" };
  return { there: shown(applies, actionId), why: why(applies, actionId) };
}

/** 该任务计算的帧范围。目标节点属于节点图内部信息，查看者并不关心，因此放在名称的悬停提示中。 */
const framesText = (j: { frames?: [number, number] | null }): string =>
  j.frames ? (j.frames[0] === j.frames[1] ? `${j.frames[0]}` : `${j.frames[0]}–${j.frames[1]}`) : "全部";

function JobRow({ job, now, admin, applies, onCancel, onForgotten, onLoad, cache, graphUrl, onFirst, inGroup = false }: { job: QueueJob; now: number; admin: boolean; applies?: Availability | null; onCancel: (id: string) => void; onForgotten?: () => void; onLoad?: (id: string) => void; cache?: CacheMark | null; graphUrl?: (id: string) => string; onFirst?: (job: QueueJob) => void; inGroup?: boolean }) {
  const [open, setOpen] = useState(false);
  const [ask, confirmSheet] = useConfirm();
  const active = job.state === "queued" || job.state === "running";
  // 任务记录保存任务的状态，但不保存其排队位置（fromRecord：position 为 null）：写记录时仍在排队的任务
  // 不得显示「第 null 位」
  const where = job.state === "queued" ? `${job.position == null ? "排队中" : `第 ${job.position} 位`}${[textOf(job.waiting), textOf(job.waiting_detail)].filter(Boolean).map((w) => ` · ${w}`).join("")}` : whereOf(job);
  // 插队（lab2shot/farm/queue.py Farm.first）：排队中或计算中的任务挪到队首。
  // 按钮始终在，不能插队时置灰并写明原因，位置不变
  const firstWhy = !active ? "这个任务已经结束了" : job.stopping ? "这个任务正在停止" : "";
  const client = job.client;
  const [forgetting, setForgetting] = useState(false);
  const may = offer(admin, applies, "queue.forget");
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
      <tr className={[admin && "expandable", inGroup && "q-member"].filter(Boolean).join(" ") || undefined} onClick={admin ? () => setOpen(!open) : undefined}>
        <td>
          <StateChip state={job.state} />
        </td>
        <td className="q-body">
          {/* 每一行固定为两行：上方为名称，下方为一排数值。名称与数值不能放在同一个会换行的 flex 中：
              内容多的行会折到第二行，内容少的行（中断且无耗时、无交付、无缓存）则挤在名称右侧不换行，
              同一列内容在各行中位置不一致，无法对齐。名称单独一行、数值单独一行，数值行为固定的五列网格
              （帧范围 | 提交 | 用时 | 交付 | 缓存，queueRow.css .q-line-facts），各列跨行对齐；缺少的值留空，位置不变。 */}
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
                    {client.department && <span className="chip q-dept" data-tip="账号的环节">{client.department}</span>}
                    {job.mine && <span className="chip q-mine">我的</span>}
                    {client.app !== "web" && <span className="q-app">{client.app}</span>}
                  </>
                ) : (
                  <span className="q-who q-muted" data-tip="别人的任务：只看得到它排在哪、算到哪，看不到是谁、算什么">别人</span>
                )}
              </span>
            )}
          </div>
          <div className="q-line-facts tnum">
            <span className="q-f" data-tip="算的是哪几帧">{framesText(job)}</span>
            {/* 悬停提示中显示年月日：表中只显示时分（列窄且跨行对齐），但列出的任务可能跨越多天，
                仅凭「17:26」无法判断日期 */}
            <span className="q-f" data-tip={`提交时间：${fullTimeText(job.submitted)}`}>{submittedAt(job)}</span>
            <span className="q-f" data-tip="用了多久">{elapsed(job, now)}</span>
            {/* 已完成的行：交付标记和缓存标记各占一格，因此跨行时也对齐。
                计算中的行进度区域较宽，占据后两格（.q-run 的 grid-column: 4 / -1），
                每行只有一个进度区域 */}
            {active ? (
              <span className="q-f q-run">
                {/* 排队位置只在编辑器中排队的行显示；后台表格不显示它 */}
                {!admin && job.state === "queued" && where && <span data-tip="排在第几位">{where}</span>}
                <Progress job={job} />
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
          {onFirst && (
            <Button
              tip={firstWhy || "插队：把这个任务挪到队首，之后空出来的显卡和 CPU 名额先给它；它正在算的和别的任务正在算的节点都不受影响"}
              disabled={!!firstWhy}
              onClick={(e) => (e.stopPropagation(), onFirst(job))}
            >
              插队
            </Button>
          )}
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
          {/* 删除：仅对已结束的任务可用，计算中的任务须先「取消」。这是腾出空间的操作：该行记录、输出的文件夹和 zip、
              仅被该任务使用的缓存及上传素材一并删除；其他任务或模板仍在引用的一律保留
              （server/quota.py drop_for_job）。不提供单独清理缓存的按钮，腾出空间只有这一途径。 */}
          {may.there && (
            <Button
              tip={active ? "这个任务还在算，删不掉：先点「取消」，停下来之后再删"
                : !(job.mine || admin) ? "别人的任务只有他自己和管理员能删"
                  : may.why || "把这一行从队列里删掉，连同它占的空间：输出的文件夹和 zip、只有它用到的缓存、只有它用到的上传素材（别的任务和模板还用得上的留着）"}
              disabled={active || !(job.mine || admin) || !!may.why || forgetting}
              onClick={(e) => (e.stopPropagation(), void forget())}
            >
              {forgetting ? "删除中…" : "删除"}
            </Button>
          )}
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
          <td colSpan={3}>
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

/** 任务记录中的一条，按队列中已完成的任务显示。 */
export const fromRecord = (r: JobRecord, mine = false): QueueJob => ({
  ...r,
  outputs: (r.outputs ?? []).filter((o) => typeof o === "object" && !!o?.pkg), // 只保留已打包进该任务的输出（lab2shot/transfer/outputs.py）
  position: null,
  waiting: null,
  now: {},
  stopping: false,
  // 编辑器的队列窗口只列出自己的任务（账号之间相互隔离，服务器只返回本人的 history），
  // 因此传入 true；后台「任务记录」列出所有人的任务，传入 false（管理员依靠权限，而非此标记）。
  // 若固定为 false，「删除」在自己已完成的任务上会一直置灰
  mine,
});

export function JobTable({ jobs, admin, applies, onCancel, onForgotten, onLoad, graphUrl, onFirst }: {
  jobs: { job: QueueJob; cache?: CacheMark | null }[];
  admin: boolean;
  applies?: Availability | null; // 后台：本次登录对他人任务拥有哪些操作（offer）
  onCancel: (id: string) => void;
  onForgotten?: () => void; // 删除了一行：重新读取队列（该行已移除，占用也可能变化）
  // 编辑器：一张表即全部内容（计算中、排队中、已完成只是状态不同），因此行中增加「缓存」和「加载」
  onLoad?: (id: string) => void;
  graphUrl?: (id: string) => string;
  // 插队（lab2shot/farm/queue.py Farm.first）：把尚未结束的任务挪到队首。不传时任何行都不提供
  // （编辑器的「队列」窗口、无权调整任务顺序的登录）。
  onFirst?: (job: QueueJob) => void;
}) {
  // 各行「用时」和进度读取的时钟：只在有任务排队或计算中时走动，因此全是已结束任务的表格不会每秒重绘
  const [now, setNow] = useState(Date.now() / 1000);
  const live = jobs.some(({ job }) => isActive(job));
  useEffect(() => {
    if (!live) return;
    setNow(Date.now() / 1000);
    const t = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(t);
  }, [live]);
  // 默认按组显示（lab2shot/transfer/groups.py：同一份素材，或没有素材时同一个模板的同一个两小时时段）：每组一行，
  // 点开才列出组里的每个任务。只有已结束的任务进组：进行中（排队、计算中）的不分组，按队列的顺序平铺在最上面，
  // 始终看得见、能插队；结束之后才归入它的组。组排在它最靠前的那个已结束任务的位置上（结束的按新到旧，顺序不变）。
  // 不是任务的行（别人的匿名任务）照常一行一个
  const blocks = blocksOf(jobs.map((r) => r.job));
  // 每组还有几个任务在进行中（平铺在上面的那些）：组那一行照样说出来
  const going = new Map<string, number>();
  for (const { job: j } of jobs) if (j.group && isActive(j)) going.set(groupId(j), (going.get(groupId(j)) ?? 0) + 1);
  const grouped = new Set(blocks.flatMap((b) => (b.group ? b.rows : [])));
  const [open, setOpen] = useState<ReadonlySet<string>>(new Set());
  const toggle = (id: string) => setOpen((was) => {
    const next = new Set(was);
    if (!next.delete(id)) next.add(id);
    return next;
  });
  const row = (i: number) => {
    const { job: j, cache } = jobs[i];
    return (
      <JobRow
        key={j.id || `load-${i}`}
        job={j}
        cache={cache}
        now={now}
        admin={admin}
        applies={applies}
        onCancel={onCancel}
        onForgotten={onForgotten}
        onLoad={onLoad}
        graphUrl={graphUrl}
        inGroup={grouped.has(i)}
        onFirst={onFirst}
      />
    );
  };
  return (
    <div className="q-table-wrap">
      <table className="q-table q-jobs">
        <thead>
          {/* 三格布局，而非每列一格：列数固定、界面文字不得换行截断、弹层宽度固定，三者同时成立时，
              多列表格必然在某些内容下溢出（出现横向滚动条），调整列宽只是将溢出转移到另一种内容。
              因此只设三格：状态 | 主体 | 操作。主体格内容可以换行，名称是唯一可截断的项
              （用户自己的数据），操作靠右且不压缩。任何内容、任何宽度下均不会溢出。 */}
          <tr>
            <th>状态</th>
            <th>任务</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {blocks.map((b) => b.group ? (
            <GroupRows key={b.id} group={b.group} jobs={b.rows.map((i) => jobs[i].job)} going={going.get(b.id) ?? 0} open={open.has(b.id)} onToggle={() => toggle(b.id)}
                       admin={admin} applies={applies} onForgotten={onForgotten}>
              {b.rows.map(row)}
            </GroupRows>
          ) : row(b.rows[0]))}
        </tbody>
      </table>
    </div>
  );
}

const isActive = (j: QueueJob) => j.state === "queued" || j.state === "running";

/** 组在页面上的 id：包含账号（组的 key 本身也已包含账号）。 */
const groupId = (j: QueueJob) => `group-${j.client?.user ?? ""}-${j.group?.key ?? ""}`;

/** 表格的绘制结构：一组已结束的行集中在该组的那一行下面，位于组内第一个已结束任务的位置；其余每行单独一块，
 * 包括排队或计算中的任务（它们从不进组：保持在队列给出的位置，始终可见，「插队」随手可用）。
 * `rows` 是表格自身列表中的下标。 */
function blocksOf(jobs: QueueJob[]): { id: string; group: TaskGroup | null; rows: number[] }[] {
  const out: { id: string; group: TaskGroup | null; rows: number[] }[] = [];
  const at = new Map<string, number>();
  jobs.forEach((j, i) => {
    if (!j.group || isActive(j)) {
      out.push({ id: `job-${j.id || i}`, group: null, rows: [i] });
      return;
    }
    const id = groupId(j);
    const found = at.get(id);
    if (found === undefined) {
      at.set(id, out.length);
      out.push({ id, group: j.group, rows: [i] });
    } else out[found].rows.push(i);
  });
  return out;
}

/** 没有素材的一组是哪个两小时时段：「9月28日 14:00–16:00」（按这台电脑的时区显示服务器分的时段）。 */
const slotText = (start: number): string => {
  const d = new Date(start * 1000);
  const end = new Date((start + 2 * 3600) * 1000);
  const hm = (t: Date) => `${String(t.getHours()).padStart(2, "0")}:00`;
  return `${d.getMonth() + 1}月${d.getDate()}日 ${hm(d)}–${end.getHours() === 0 ? "24:00" : hm(end)}`;
};

/** 一组的状态：组里只有已结束的任务（进行中的平铺在上面），即最新那个的状态。 */
const groupState = (jobs: QueueJob[]): JobState => jobs[0].state;

/** 组名。同一个账号有两个按素材分的组同名时，后面加上这组第一个任务的时间：「sh030_plate · 9月28日 14:05」（只在显示上区分）。
 * 改过名的组照原样显示用户起的名字（服务器不会给它 twin）。 */
const groupName = (group: TaskGroup): string => {
  const name = group.name || "未命名";
  if (!group.twin) return name;
  const d = new Date(group.first * 1000);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${name} · ${d.getMonth() + 1}月${d.getDate()}日 ${two(d.getHours())}:${two(d.getMinutes())}`;
};

/** 组的那一行（默认收起），展开时是组内已结束任务的各行（`children`，由表格绘制）；
 * `going`：组内有几个任务在排队或计算中（平铺在上方）。改名：设置组的显示名称（服务器的
 * PUT /api/task-groups/<key>/name，管理员可改任何人的组；留空则恢复自动名称），分组本身不变。
 * 删除组：删除组内每个已结束的任务，每个都与单独「删除」相同（服务器的 DELETE /api/task-groups：仍在排队或计算中的
 * 任务保留并告知）。`onForgotten`：两种操作之后重读表格。组名是用户自己的数据：只作纯文本。 */
function GroupRows({ group, jobs, going, open, onToggle, admin, applies, onForgotten, children }: {
  group: TaskGroup;
  jobs: QueueJob[];
  going: number;
  open: boolean;
  onToggle: () => void;
  admin: boolean;
  applies?: Availability | null;
  onForgotten?: () => void;
  children: ReactNode;
}) {
  const [ask, confirmSheet] = useConfirm();
  const [forgetting, setForgetting] = useState(false);
  const [renaming, setRenaming] = useState(false);
  const first = jobs[0];
  const client = first.client;
  const name = groupName(group);
  const listed = jobs.length + going;
  const owner = first.mine || admin;
  const mayForget = offer(admin, applies, "queue.forgetgroup");
  const mayRename = offer(admin, applies, "queue.rename");
  const latest = Math.max(...jobs.map((j) => j.submitted));
  const forget = async () => {
    if (!(await ask({ title: "删除这一组", say: msg("N-GROUP-FORGET", { name, count: group.count }), yes: "删除这一组", danger: true,
                      tip: "删了找不回来" }))) return;
    setForgetting(true);
    try {
      const got = admin && client?.user !== undefined ? await api.admin.forgetGroup(client.user, group.key) : await api.forgetGroup(group.key);
      say(msg("I-GROUP-FORGOTTEN", { name, jobs: got.jobs, size: sizeText(got.bytes) }));
      if (got.skipped) say(msg("W-GROUP-SKIPPED", { name, skipped: got.skipped }));
      onForgotten?.();
    } catch (e) {
      say(msg("E-JOB-FORGETFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setForgetting(false);
    }
  };
  const rename = async (to: string) => {
    try {
      const got = admin && client?.user !== undefined ? await api.admin.renameGroup(client.user, group.key, to) : await api.renameGroup(group.key, to);
      say(msg("I-GROUP-RENAMED", { name: got.name }));
      onForgotten?.();
    } catch (e) {
      say(msg("E-GROUP-RENAMEFAILED", { reason: reasonOf(e as Error) }));
    }
  };
  return (
    <>
      <tr className="q-group expandable" onClick={onToggle} aria-expanded={open}>
        <td>
          <StateChip state={groupState(jobs)} />
        </td>
        <td className="q-body">
          <div className="q-line-name">
            {/* 展开标记和组名同在一行：组名再长也只在自己那一格里切字，标记不会被挤到上一行 */}
            <span className="q-group-name">
              <span className="q-group-toggle" aria-hidden>{open ? "▾" : "▸"}</span>
              <span className="q-title" data-user-data=""
                    data-tip={[name, group.renamed ? "这个名字是改过的：分组还是原来的，之后进这一组的任务也用这个名字" : "",
                               group.slot == null ? "按素材分组：这些任务读的素材内容一样。换了素材、增删输入节点就是新的一组"
                                 : `没有素材的任务：同一个模板、同一个两小时时段（${slotText(group.slot)}）算一组`].filter(Boolean).join("\n")}>
                {name}
              </span>
            </span>
            {admin && client && (
              <span className="q-facts">
                <span className="q-who">{client.who}</span>
                {client.department && <span className="chip q-dept" data-tip="账号的环节">{client.department}</span>}
              </span>
            )}
          </div>
          <div className="q-group-facts tnum">
            <span data-tip={[listed < group.count ? `这里列出 ${listed} 个，这一组一共 ${group.count} 个任务` : "这一组的任务数",
                             going ? "进行中的不放进组里，平铺在最上面，结束了才归到这一组" : ""].filter(Boolean).join("\n")}>
              {group.count} 个任务{going ? ` · ${going} 个进行中` : ""}
            </span>
            {group.slot != null && <span data-tip="没有素材的任务按固定的两小时时段分组">{slotText(group.slot)}</span>}
            <span data-tip={`最近一次提交：${fullTimeText(latest)}`}>最近 {clockText(latest)}</span>
          </div>
        </td>
        <td className="q-act">
          {mayRename.there && (
            <Button
              tip={!owner ? "别人的组只有他自己和管理员能改名" : mayRename.why || "给这一组改个名字：只改显示的名字，分组不变；之后进这一组的任务也用这个名字"}
              disabled={!owner || !!mayRename.why}
              onClick={(e) => (e.stopPropagation(), setRenaming(true))}
            >
              改名
            </Button>
          )}
          {mayForget.there && (
            <Button
              tip={!owner ? "别人的任务只有他自己和管理员能删"
                : mayForget.why || `把这一组的 ${group.count} 个任务都删掉，连同它们占的空间（和一条一条删一样）；正在排队和计算的不动`}
              disabled={!owner || !!mayForget.why || forgetting}
              onClick={(e) => (e.stopPropagation(), void forget())}
            >
              {forgetting ? "删除中…" : "删除这一组"}
            </Button>
          )}
        </td>
      </tr>
      {confirmSheet}
      {renaming && (
        <NameSheet title="给这一组改名" label="组名" initial={group.renamed ? group.name : ""} max={64}
                   empty={group.renamed ? "留空：回到自动起的名字" : `留空：用自动起的名字（${group.name || "未命名"}）`}
                   save={rename} onClose={() => setRenaming(false)} />
      )}
      {open && children}
    </>
  );
}

const MARK: Record<CacheMark["mark"], [string, string]> = {
  all: ["全在", "var(--green)"],
  some: ["部分", "var(--orange)"],
  none: ["已清理", "var(--text-3)"],
};

/** 已完成任务的结果是否仍在缓存中（由服务器根据任务的节点图判定）。不估计重新计算需要多久：按以往用时推算的时间不可靠。 */
function CacheCell({ cache }: { cache: CacheMark | null }) {
  // 不提供「清理」按钮：腾出空间的操作均在任务上，该行的「删除」已同时移除仅被它使用的缓存；
  // 再设一个仅清理缓存的按钮会使同一操作有两个入口，且清理后仍留下一行空任务，用户会误以为空间未释放。
  // 需要一次性清理时使用队列上方的「删除全部」。
  if (!cache) return null;
  const [label, color] = MARK[cache.mark];
  const tip =
    cache.mark === "all"
      ? "结果都还在缓存里：加载后马上就能看，不用重新算"
      : cache.mark === "some"
        ? `${cache.nodes} 个节点里 ${cache.cached} 个的结果还在缓存里，别的加载后要重新算`
        : `结果已经从缓存里清理了${cache.why ? `（${cache.why}）` : ""}：加载后要重新算`;
  return (
    <span className="q-cache">
      <span className="chip q-state" data-tip={tip}>
        <i style={{ background: color }} />
        {label}
      </span>
    </span>
  );
}

