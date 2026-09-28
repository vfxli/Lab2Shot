import { msg, type Message } from "../messages/message";
import { ByUser } from "./Resources";
import { useSignedIn } from "../state/session";
import { shown } from "../api/applies";
import { useCallback, useEffect, useState } from "react";
import { Sheet } from "../ui/Sheet";
import { adminApi, type FeedbackDetail, type FeedbackStatus } from "../api/admin";
import { Section, useAdmin } from "./common";
import { stampText, whenText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button, ButtonLink, Segmented } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
// the feedback's own look (the detail's header, its groups, the status chips) is one file shared with 提交反馈 (ui/Feedback.tsx);
// the admin page does not load that component, so it brings the file itself
import "../ui/feedback.css";

/** 用户反馈 section: submissions from 提交反馈 (lab2shot/feedback.py): sender, time, text, attached screenshots
 * and diagnostics. Supports answering (a status and a reply visible in 我的反馈, plus a private note), downloading
 * everything, and deleting. */

const FEEDBACK_STATUS: Record<FeedbackStatus, { label: string; tip: string }> = {
  new: { label: "新", tip: "还没看过" },
  seen: { label: "已看", tip: "看过了，还没解决（打开一条新反馈就算看过）" },
  solved: { label: "已解决", tip: "问题解决了，或者建议处理了" },
};
const PERIODS = [
  { id: "all", label: "全部", days: 0 },
  { id: "day", label: "今天", days: 1 },
  { id: "week", label: "7 天", days: 7 },
  { id: "month", label: "30 天", days: 30 },
];

const full = (t: number | null | undefined) => (t ? stampText(t) : "—"); // the server's times: seconds
const pageTime = (ms: number) => stampText(ms / 1000); // the page's own diagnostics (its log, errors, requests): milliseconds

export function FeedbackSection() {
  const { problem, refreshOverview } = useAdmin();
  const [status, setStatus] = useState<FeedbackStatus | "">("");
  const [period, setPeriod] = useState("all");
  const [person, setPerson] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const read = useCallback(() => {
    const days = PERIODS.find((p) => p.id === period)?.days ?? 0;
    const since = days ? (days === 1 ? new Date().setHours(0, 0, 0, 0) / 1000 : Date.now() / 1000 - days * 86400) : undefined;
    return adminApi.feedback({ status, since, person });
  }, [status, period, person]);
  const { data: view, reload: reload } = usePoll(read, 20_000, { onError: (e) => problem(reasonOf(e)) });
  const changed = () => (reload(), refreshOverview());

  return (
    <Section
      title="用户反馈"
      lede="用户在编辑器顶栏点「提交反馈」发来的问题和建议，带着截图和查问题用的资料：浏览器和系统、当时的节点图、页面日志和错误、他最近的任务和出错节点的日志、服务日志里有关的行、扩展包和显卡内存的状态。一直留着，直到在这里删除；数据库备份也包括它们。"
      actions={
        <Button tip="重新读取反馈列表" tone="ghost" onClick={reload}>
          刷新
        </Button>
      }
    >
      <div className="fb-filters">
        <Segmented label="状态" value={status} options={[{ value: "", label: "全部", tip: "所有状态" }, ...(Object.keys(FEEDBACK_STATUS) as FeedbackStatus[]).map((s) => ({ value: s, label: `${FEEDBACK_STATUS[s].label}${view ? ` ${view.counts[s]}` : ""}`, tip: FEEDBACK_STATUS[s].tip }))]} onChange={setStatus} />
        <Segmented label="时间" value={period} options={PERIODS.map((p) => ({ value: p.id, label: p.label, tip: p.days ? `只看${p.label === "今天" ? "今天" : `最近 ${p.label}`}的反馈` : "不限时间" }))} onChange={setPeriod} />
        <input className="field fb-search" value={person} placeholder="按账号找" data-tip="只看中文名或用户名里有这几个字的账号提交的" onChange={(e) => setPerson(e.target.value)} />
      </div>
      {view === null ? (
        <p className="adm-lede">读取中…</p>
      ) : view.items.length ? (
        <div className="q-table-wrap">
          <table className="q-table fb-table">
            <thead>
              <tr>
                <th data-tip="提交的时间">时间</th>
                <th data-tip="提交反馈的账号（删掉的账号显示「已删除的用户」）">提交人</th>
                <th data-tip="账号的环节">环节</th>
                <th data-tip="用户选的类别（可以不选）">类别</th>
                <th data-tip="写的问题的第一行；点一行看全部">内容</th>
                <th data-tip="截图张数">截图</th>
                <th data-tip="新 / 已看 / 已解决">状态</th>
              </tr>
            </thead>
            <tbody>
              {view.items.map((f) => (
                <tr key={f.id} className={`expandable${f.status === "new" ? " fb-new" : ""}`} onClick={() => setOpen(f.id)}>
                  <td className="q-time tnum">{whenText(f.at)}</td>
                  <td className="q-who">{f.person}</td>
                  <td>{f.department}</td>
                  <td>{f.category_label || "—"}</td>
                  <td>
                    <span className="q-title">{f.title}</span>
                    {f.reply && <span className="q-targets">回复：{f.reply}</span>}
                    {f.note && <span className="q-targets">备注：{f.note}</span>}
                  </td>
                  <td className="tnum">{f.images.length || "—"}</td>
                  <td>
                    <StatusChip status={f.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-empty">{status || period !== "all" || person ? "没有符合条件的反馈：放宽上面的状态、时间和账号筛选" : "还没有用户反馈：用户在编辑器右下角点「提交反馈」以后，这里列出来"}</p>
      )}
      {open && <FeedbackSheet id={open} onClose={() => setOpen(null)} onChanged={changed} />}
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="feedback" />
    </Section>
  );
}

function StatusChip({ status }: { status: FeedbackStatus }) {
  return (
    <span className={`chip fb-status ${status}`} data-tip={FEEDBACK_STATUS[status].tip}>
      {FEEDBACK_STATUS[status].label}
    </span>
  );
}

/** Screenshots of one feedback entry. Shown as thumbnails by default (a large image at full size would be
 * clipped by the sheet); clicking one expands it to the sheet width and clicking again collapses it. The original
 * is available through a further click. */
function Shots({ id, names }: { id: string; names: string[] }) {
  const [big, setBig] = useState<string | null>(null);
  return (
    <div className="fb-shots">
      {names.map((name) => (
        <figure key={name} className={`fb-shot${big === name ? " open" : ""}`}>
          <img src={adminApi.feedbackFile(id, name)} alt={name} data-tip={big === name ? "点一下收回小图" : "点一下放大"} onClick={() => setBig((b) => (b === name ? null : name))} />
          {big === name && (
            <figcaption>
              <a href={adminApi.feedbackFile(id, name)} target="_blank" rel="noreferrer" data-tip="在新标签页看原图">看原图</a>
            </figcaption>
          )}
        </figure>
      ))}
    </div>
  );
}

/** One feedback entry: the text, screenshots, the answer (status, user-visible reply, private note) and the
 * diagnostics in collapsible groups. */
function FeedbackSheet({ id, onClose, onChanged }: { id: string; onClose: () => void; onChanged: () => void }) {
  const { problem } = useAdmin();
  const [f, setF] = useState<FeedbackDetail | null>(null);
  const [status, setStatus] = useState<FeedbackStatus>("seen");
  const [reply, setReply] = useState("");
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState<Message | null>(null);
  const dirty = !!f && (status !== f.status || reply.trim() !== f.reply || note.trim() !== f.note);

  useEffect(() => {
    let alive = true;
    adminApi.feedbackDetail(id).then(
      async (d) => {
        if (!alive) return;
        setF(d);
        setStatus(d.status === "new" ? "seen" : d.status);
        setReply(d.reply);
        setNote(d.note);
        if (d.status === "new") {
          // Opening marks the entry as seen; the user can see this status.
          const seen = await adminApi.answerFeedback(id, { status: "seen", reply: d.reply, note: d.note }).catch(() => null);
          if (alive && seen) setF({ ...d, ...seen });
          onChanged();
        }
      },
      (e: Error) => problem(e.message),
    );
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const save = async () => {
    try {
      const row = await adminApi.answerFeedback(id, { status, reply, note });
      setF((d) => (d ? { ...d, ...row } : d));
      setReply(row.reply);
      setNote(row.note);
      setSaved(msg(row.reply ? "I-FEEDBACK-REPLIED" : "I-FEEDBACK-STATUSSAVED"));
      problem(null);
      onChanged();
    } catch (e) {
      problem((e as Error).message);
    }
  };

  const [ask, confirmSheet] = useConfirm();
  const close = async () => {
    if (!dirty || (await ask({ title: "没保存", say: msg("N-FEEDBACK-DISCARD"), yes: "不保存", tip: "丢掉改了的回复、备注和状态，关掉", danger: true }))) onClose();
  };

  const state = useSignedIn();

  const remove = async () => {
    if (!f || !(await ask({ title: "删除反馈", say: msg("N-FEEDBACK-DELETE", { person: f.person, at: full(f.at) }), yes: "删除", tip: "删掉这条反馈和它的截图、诊断资料，不能恢复", danger: true }))) return;
    try {
      await adminApi.deleteFeedback(id);
      problem(null);
      onChanged();
      onClose();
    } catch (e) {
      problem((e as Error).message);
    }
  };

  const b = f?.bundle;
  const page = b?.page ?? {};
  const server = b?.server ?? {};
  return (
    <Sheet title="用户反馈" width={980} solid onClose={() => void close()}>
      {!f ? (
        <p className="adm-lede">读取中…</p>
      ) : (
        <div className="fb-detail">
          <div className="fb-meta">
            <span className="q-who">{f.person}</span>
            <span>{f.department}</span>
            <span className="tnum">{full(f.at)}</span>
            {f.category_label && <span className="chip">{f.category_label}</span>}
            <StatusChip status={f.status} />
            {f.updated && <span className="q-targets">{`${full(f.updated)} ${f.updated_by} 最后改过`}</span>}
          </div>
          <div className="fb-text-full">{f.text}</div>
          {f.images.length > 0 && <Shots id={id} names={f.images} />}
          <div className="fb-answer">
            <div className="fb-status-row">
              <label htmlFor="fb-reply" className="who-label" data-tip="写给提交的人：他在编辑器「提交反馈」的「我的反馈」里看到，按钮上会出现一个点">
                回复
              </label>
              <textarea id="fb-reply" className="field fb-note" value={reply} placeholder="给用户的回复：怎么解决、要他怎么做（可以不写）" data-tip="用户能看到" onChange={(e) => (setReply(e.target.value), setSaved(null))} />
            </div>
            <div className="fb-status-row">
              <label htmlFor="fb-note" className="who-label" data-tip="只有管理员看得到：用户永远看不到">
                内部备注
              </label>
              <textarea id="fb-note" className="field fb-note" value={note} placeholder="处理情况，只有管理员看得到（可以不写）" data-tip="用户看不到" onChange={(e) => (setNote(e.target.value), setSaved(null))} />
            </div>
            <div className="dialog-row fb-actions">
              <span className="tpl-desc">
                {saved?.text || (f.reply ? (f.unread ? `回复写于 ${full(f.replied)}：用户还没看到` : `回复写于 ${full(f.replied)}：用户已经看到`) : "用户能看到状态和回复，看不到内部备注")}
              </span>
              <Segmented label="状态" value={status} options={(Object.keys(FEEDBACK_STATUS) as FeedbackStatus[]).map((s) => ({ value: s, label: FEEDBACK_STATUS[s].label, tip: `${FEEDBACK_STATUS[s].tip}；点「保存」才生效` }))} onChange={(s) => (setStatus(s), setSaved(null))} />
              <Button tip="保存状态、回复和内部备注；状态或回复变了，用户那边会提示有新消息" tone="primary" disabled={!dirty} onClick={() => void save()}>
                保存
              </Button>
            </div>
          </div>
          {b ? (
            <div className="fb-groups">
              {page.help !== undefined && (
                <Group label="页面" tip="用户在看的帮助页面：哪个项目、安装的情况" count={String((page.help as { title?: string }).title ?? "")}>
                  <Pre value={page.help} />
                </Group>
              )}
              <Group label="节点图" tip="提交时的节点图（文件只有名字和位置）：选中和显示的节点、当前帧、计算范围、有问题的节点" count={graphCount(page.graph)}>
                <Pre value={page.editor ?? "这个页面没有节点图"} />
                {page.graph !== undefined && <Pre value={page.graph} />}
              </Group>
              <Group label="页面日志" tip="用户页面日志里最近的记录" count={`${(page.log ?? []).length} 条`}>
                <Pre value={(page.log ?? []).map((e) => `${pageTime(e.t)}${e.count && e.last ? `  ×${e.count}，到 ${pageTime(e.last)}` : ""}  [${e.level}]  ${e.text}`).join("\n") || "没有记录"} />
              </Group>
              <Group label="错误" tip="页面打开以来自己出的错，和服务器拒绝或没连上的请求" count={`${(page.errors ?? []).length + (page.requests ?? []).length} 个`}>
                <Pre value={(page.errors ?? []).map((e) => `${pageTime(e.t)}${e.count > 1 ? `  ×${e.count}，到 ${pageTime(e.last)}` : ""}  ${e.kind === "rejection" ? "未处理的异步错误" : "页面出错"}：${e.message}${e.where ? `（${e.where}）` : ""}${e.stack ? `\n${e.stack}` : ""}`).join("\n\n") || "页面没有出错"} />
                <Pre value={(page.requests ?? []).map((r) => `${pageTime(r.t)}  ${r.method} ${r.url} → ${r.status || "没连上"}（${r.ms} ms）  ${r.message}`).join("\n") || "没有失败的请求"} />
              </Group>
              <Group label="任务" tip="这个用户最近的任务：状态、报错、出错节点的日志、服务日志里和它有关的行" count={`${(server.jobs ?? []).length} 个`}>
                {(server.jobs ?? []).length ? (
                  (server.jobs ?? []).map((j) => (
                    <div key={j.id} className="fb-job">
                      <div className="fb-meta">
                        <span className="q-who">{j.title}</span>
                        <span className="chip">{j.state}</span>
                        <span className="tnum">{full(j.submitted)}</span>
                        <span className="q-targets">{j.id}</span>
                      </div>
                      {j.error && <div className="q-error">{j.error}</div>}
                      {j.error_log && <Pre value={`${j.error_log.file}\n\n${j.error_log.lines.join("\n")}`} />}
                      {j.server_log.length > 0 && <Pre value={j.server_log.join("\n")} />}
                    </div>
                  ))
                ) : (
                  <p className="adm-empty">这个用户最近没有任务：这个账号提交计算以后，这里列出来</p>
                )}
              </Group>
              <Group label="环境" tip="用户的浏览器和系统；服务器的版本、显卡、内存、队列、常驻模型；节点图用到的扩展包装好没有" count={String(server.revision ?? "")}>
                <Pre value={{ browser: page.browser, client: server.client, page: page.page }} />
                <Pre value={Object.fromEntries(Object.entries(server).filter(([k]) => !["jobs", "server_log", "client"].includes(k)))} />
              </Group>
              <Group label="服务日志" tip="提交反馈时服务日志的最后 200 行" count={`${(server.server_log ?? []).length} 行`}>
                <Pre value={(server.server_log ?? []).join("\n")} />
              </Group>
            </div>
          ) : (
            <p className="adm-empty">诊断资料读不出来：这条反馈提交时没有附上，或者已经删掉</p>
          )}
          <div className="dialog-row fb-actions">
            <ButtonLink tip="整条反馈打包成一个 zip：写的内容、谁、状态、回复、全部诊断资料和截图" href={adminApi.feedbackDownload(id)}>
              下载全部
            </ButtonLink>
            {shown(state?.applies, "feedback.delete") && (
              <Button tip="删除这条反馈（要再确认一次）" tone="ghost" onClick={() => void remove()}>
                删除
              </Button>
            )}
          </div>
        </div>
      )}
      {confirmSheet}
    </Sheet>
  );
}

const graphCount = (g: unknown) => {
  const nodes = (g as { nodes?: unknown[] } | undefined)?.nodes;
  return nodes ? `${nodes.length} 个节点` : "";
};

function Group({ label, tip, count, children }: { label: string; tip: string; count: string; children: React.ReactNode }) {
  return (
    <details className="fb-group">
      <summary data-tip={tip}>
        {label}
        {count && <span className="fb-count">{count}</span>}
      </summary>
      {children}
    </details>
  );
}

const Pre = ({ value }: { value: unknown }) => <pre className="server-log fb-pre">{typeof value === "string" ? value : JSON.stringify(value, null, 1)}</pre>;

