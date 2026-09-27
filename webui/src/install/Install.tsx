import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Checklist, type InstallTask, type InstallStep } from "../api";
import { shown, usable, why, type Availability } from "../api/applies";
import { controlKind, currentStep, installButton, taskLive } from "./installState";
import { Sheet } from "../ui/Sheet";
import { msg, reasonOf } from "../messages/message";
import { Loading } from "../ui/Loading";
import { Button } from "../ui/Button";
import { startPolling } from "../platform/poll";
import { useConfirm } from "../ui/Confirm";

/** 扩展包安装控件（lab2shot/installer），用于后台管理页的「扩展包」区域（admin/Extensions.tsx）。
 *
 * What a login may do here is the server's answer
 * (server/available.py extension: `actions`), never a role check: an account that does not install gets no actions, and
 * its uninstalled card shows only 「未安装」 (the corner mark); this component renders nothing for it.
 *
 * An install is a background task of the farm: 安装 opens the checklist (preflight: hand downloads, licences, Hugging
 * Face access, disk, cards), 开始安装 starts it when nothing blocks, and the sheet follows the task with a step bar and the
 * whole log, resuming from where it left off after a reload (the task is the server's: `p.job`). */

export interface Installable {
  name: string;
  title: string;
  installed: boolean;
  ready: boolean;
  label: string;
  actions: Availability;
  job: InstallTask | null;
}

export { taskLive };

// UI vocabulary: a job's and a step's state in one or two words
const JOB_STATE: Record<InstallTask["state"], string> = { queued: "排队中", running: "安装中", done: "已完成", failed: "失败", cancelled: "已取消" };
const STEP_STATE: Record<InstallStep["state"], string> = { waiting: "等待", running: "进行中", done: "完成", skipped: "跳过", failed: "失败", cancelled: "已取消" };

/** The card's and the project page's install control. `full`: the project page (reinstall, rollback, uninstall). */
export function InstallControl({ p, onChange, full = false }: { p: Installable; onChange: () => void; full?: boolean }) {
  const [open, setOpen] = useState<{ force: boolean } | null>(null);
  const [ask, confirmSheet] = useConfirm();
  const uninstall = async () => {
    // the page's one confirmation sheet: 卸载 deletes the environment and code, keeping the models
    if (!(await ask({ title: "卸载", say: msg("N-INSTALL-UNINSTALL", { title: p.title }), yes: "卸载", tip: "卸载后重装不用再下载模型", danger: true }))) return;
    act(api.installs.uninstall(p.name));
  };
  const [error, setError] = useState("");
  const a = p.actions;
  const kind = controlKind(p, full);
  const sheet = open && <InstallSheet p={p} force={open.force} onClose={() => setOpen(null)} onChange={onChange} />;
  // Not this login's, or nothing to do: nothing at all, except a sheet still open on a job that just ended (the card
  // lights up underneath it; the log stays until the administrator closes it). The sheet keeps one place in the tree
  // whatever the card shows, so it is never remounted (a remount would open the checklist instead of the finished job).
  if (kind === "none") return <>{false}{sheet}</>;
  const job = p.job;
  const active = kind === "progress";
  const failed = job?.state === "failed";
  const current = currentStep(job);
  const button = installButton(p);
  const act = (what: Promise<unknown>) => {
    setError("");
    what.then(onChange, (e) => setError(reasonOf(e)));
  };
  return (
    <>
    {confirmSheet}
    <div className="inst">
      {active && job && (
        <div className="inst-row">
          <StepBar steps={job.steps} compact />
          <span className="inst-state">{job.state === "queued" ? JOB_STATE.queued : current?.label ?? JOB_STATE.running}</span>
          <Button onClick={() => setOpen({ force: false })} tip="打开分步进度和完整日志">
            查看
          </Button>
        </div>
      )}
      {!active && (
        <div className="inst-row">
          {shown(a, "install") && (
            <Button
              tone={p.ready ? "default" : "primary"}
              disabled={!usable(a, "install")}
              onClick={() => setOpen({ force: button.force })}
              tip={why(a, "install") || (p.ready ? "所有步骤重新做，在旧环境旁边建新环境；自检通过才换上，旧的留着可以回退" : failed ? `从「${current?.label ?? ""}」这一步接着装，已完成的步骤不重做` : "先体检，齐全才开始：拉取代码、建环境、下载并校验模型、自检")}
            >
              {button.label}
            </Button>
          )}
          {failed && job && (
            <Button tone="ghost" onClick={() => setOpen({ force: false })} tip="看失败在哪一步和完整日志">
              日志
            </Button>
          )}
          {full && shown(a, "rollback") && (
            <Button tone="ghost" disabled={!usable(a, "rollback")} tip={why(a, "rollback") || "换回上一次安装之前的环境"} onClick={() => act(api.installs.rollback(p.name))}>
              回退
            </Button>
          )}
          {full && shown(a, "uninstall") && (
            <Button
              tone="ghost"
              disabled={!usable(a, "uninstall")}
              tip={why(a, "uninstall") || "删掉环境、代码和缓存；模型文件保留，重装不用再下载"}
              onClick={() => void uninstall()}
            >
              卸载
            </Button>
          )}
        </div>
      )}
      {failed && job?.result && <p className="inst-note err">{`${current?.label ?? ""}${current ? "：" : ""}${job.result.text}`}</p>}
      {error && <p className="inst-note err">{error}</p>}
    </div>
    {sheet}
    </>
  );
}

/** The steps as one bar: a segment per step, coloured by its state; `compact` (a card) without the step names. */
function StepBar({ steps, compact = false }: { steps: InstallStep[]; compact?: boolean }) {
  return (
    <div className={`inst-steps${compact ? " compact" : ""}`}>
      {steps.map((s) => (
        <span key={s.id} className={`inst-step ${s.state}`} data-tip={`${s.label}：${STEP_STATE[s.state]}${s.message ? `（${s.message.text}）` : ""}`}>
          <i />
          {!compact && s.label}
        </span>
      ))}
    </div>
  );
}

/** The checklist, then the job: an active or failed job of this extension is followed at once; otherwise the checklist
 * comes first and 开始安装 queues the job. */
function InstallSheet({ p, force, onClose, onChange }: { p: Installable; force: boolean; onClose: () => void; onChange: () => void }) {
  const [checklist, setChecklist] = useState<Checklist | null>(null);
  const [jobId, setJobId] = useState<string | null>(p.job && (taskLive(p.job) || (!force && p.job.state === "failed")) ? p.job.id : null);
  const [error, setError] = useState("");
  const [asking, setAsking] = useState(false);
  const check = useCallback(() => {
    setChecklist(null);
    setError("");
    api.installs.preflight(p.name).then(setChecklist, (e) => setError(reasonOf(e)));
  }, [p.name]);
  useEffect(() => {
    if (!jobId) check();
  }, [jobId, check]);
  const start = () => {
    setAsking(true);
    setError("");
    api.installs.start(p.name, force).then(
      (job) => (setJobId(job.id), setAsking(false), onChange()),
      (e) => (setError(reasonOf(e)), setAsking(false), check()),
    );
  };
  const blocked = checklist ? checklist.checks.filter((c) => c.state === "blocked").length : 0;
  return (
    <Sheet title={`${force ? "重新安装" : "安装"} ${p.title}`} width={760} onClose={onClose}>
      {jobId ? (
        <JobView id={jobId} onEnded={onChange} onRetry={() => setJobId(null)} />
      ) : (
        <div className="inst-sheet">
          <p className="inst-note">开始前先查一遍不能自动完成的步骤：全部准备好才能开始。</p>
          {checklist ? (
            <div className="inst-checks">
              {checklist.checks.map((c, i) => (
                <div key={i} className={`inst-check ${c.state}`}>
                  <span className="mark">{c.state === "ok" ? "✓" : c.state === "blocked" ? "✗" : c.state === "warning" ? "!" : "·"}</span>
                  <span className="kind">{c.label}</span>
                  <span className="text">
                    <Linked text={c.message.text} />
                  </span>
                </div>
              ))}
            </div>
          ) : (
            !error && <p className="inst-note">体检中…</p>
          )}
          {error && <p className="inst-note err">{error}</p>}
          <div className="inst-sheet-actions">
            <Button tone="ghost" onClick={check} tip="放好文件、批下权限、清出空间后点这里再查一遍">
              重新检查
            </Button>
            <Button tone="primary" disabled={!checklist || !checklist.ready || asking} onClick={start} tip={blocked ? `还有 ${blocked} 项没准备好` : "排队安装：同一时间只装一个"}>
              开始安装
            </Button>
          </div>
        </div>
      )}
    </Sheet>
  );
}

/** A log line that carries a message's code ("[W-INSTALL-RETRY] ..."): its level, for the colour. */
const level = (line: string): string => line.match(/^\[([EWNIB])-[A-Z0-9-]+\]/)?.[1] ?? "";

/** A sentence with its web addresses as links (a request page, a download page). */
function Linked({ text }: { text: string }) {
  const parts = text.split(/(https:\/\/[^\s，。；、）)」]+)/);
  return (
    <>
      {parts.map((part, i) =>
        part.startsWith("https://") ? (
          <a key={i} href={part} target="_blank" rel="noreferrer">
            {part}
          </a>
        ) : (
          part
        ),
      )}
    </>
  );
}

/** One install task, followed while it waits or runs: the step bar, its state, the whole log. */
function JobView({ id, onEnded, onRetry }: { id: string; onEnded: () => void; onRetry: () => void }) {
  const [job, setJob] = useState<InstallTask | null>(null);
  const [lines, setLines] = useState<string[]>([]);
  const [error, setError] = useState("");
  const next = useRef(0);
  const logRef = useRef<HTMLPreElement>(null);
  const ended = useRef(onEnded);
  ended.current = onEnded;

  useEffect(() => {
    next.current = 0;
    setLines([]);
    // the one polling hook (platform/poll.ts): ask again while the job runs or events are still coming, stop when
    // it ends
    const asking = startPolling({
      read: () => api.installs.job(id, next.current),
      every: (last, failed) => (failed ? 3000 : taskLive(last) || last?.lines.length ? 1000 : 0),
      until: (view) => !taskLive(view) && !view.lines.length,
      onValue: (view) => {
        next.current = view.next;
        setJob(view);
        setError("");
        if (view.lines.length) setLines((l) => [...l, ...view.lines].slice(-3000));
        if (!taskLive(view) && !view.lines.length) ended.current();
      },
      onError: (e) => setError(reasonOf(e)),
    });
    return asking.stop;
  }, [id]);

  useEffect(() => {
    const el = logRef.current;
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [lines]);

  if (!job) return <p className="inst-note">{error || <Loading what="安装进度" />}</p>;
  const failedStep = job.steps.find((s) => s.state === "failed");
  return (
    <div className="inst-sheet">
      <StepBar steps={job.steps} />
      <div className="inst-row">
        <span className={`chip inst-job-state ${job.state}`}>{JOB_STATE[job.state]}</span>
        {taskLive(job) && (
          <Button tone="ghost" onClick={() => api.installs.cancel(job.id).catch((e) => setError(reasonOf(e)))} tip="停在当前步骤；再点安装会从这一步接着装">
            取消
          </Button>
        )}
        {job.state === "failed" && (
          <Button tone="primary" onClick={onRetry} tip={`重新体检后从「${failedStep?.label ?? ""}」这一步接着装`}>
            从这一步重试
          </Button>
        )}
      </div>
      {job.result && job.state !== "done" && (
        <p className="inst-note err">
          {failedStep ? `${failedStep.label}：` : ""}
          <code>{job.result.code}</code> {job.result.text}
        </p>
      )}
      {error && <p className="inst-note err">{error}</p>}
      <pre className="server-log inst-log" ref={logRef}>
        {lines.map((line, i) => (
          <div key={i} className={level(line) ? `lv-${level(line)}` : undefined}>
            {line}
          </div>
        ))}
      </pre>
    </div>
  );
}
