import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Checklist, type InstallTask, type InstallStep } from "../api";
import { shown, usable, why, type Availability } from "../api/applies";
import { controlKind, currentStep, installButton } from "./installState";
import { taskLive } from "../api/tasks";
import { Sheet } from "../ui/Sheet";
import { msg, reasonOf } from "../messages/message";
import { CODE } from "../messages/format";
import { Loading } from "../ui/Loading";
import { Button } from "../ui/Button";
import { startPolling } from "../platform/poll";
import { useConfirm } from "../ui/Confirm";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** The extension install control (lab2shot/installer), used by the admin page's 「扩展包」 section
 * (admin/Extensions.tsx).
 *
 * What a login may do here is the server's answer
 * (server/available.py extension: `actions`), never a role check: an account that does not install gets no actions, and
 * its uninstalled row shows only 「未安装」 in the 状态 column; this component renders nothing for it.
 *
 * An install is a background task of the farm: 安装 opens the checklist (preflight: hand downloads, licences, Hugging
 * Face access, disk, cards), 开始安装 starts it when nothing blocks, and the sheet follows the task with a step bar and the
 * whole log, resuming from where it left off after a reload (the task is the server's: `p.job`). */

interface Installable {
  name: string;
  title: string;
  installed: boolean;
  ready: boolean;
  label: string;
  actions: Availability;
  job: InstallTask | null;
}


// UI vocabulary: a job's and a step's state in one or two words
// (keys: read with t() as they are shown)
const JOB_STATE: Record<InstallTask["state"], string> = { queued: "ui.install.job.queued", running: "ui.install.job.running", done: "ui.install.job.done", failed: "ui.install.job.failed", cancelled: "ui.install.job.cancelled" };
const STEP_STATE: Record<InstallStep["state"], string> = { waiting: "ui.install.step.waiting", running: "ui.install.step.running", done: "ui.install.step.done", skipped: "ui.install.step.skipped", failed: "ui.install.step.failed", cancelled: "ui.install.step.cancelled" };

/** The install control of one row of the admin page's 「扩展包」 table. `full` (the table passes it): also reinstall,
 * rollback and uninstall. */
export function InstallControl({ p, onChange, full = false }: { p: Installable; onChange: () => void; full?: boolean }) {
  const [open, setOpen] = useState<{ force: boolean } | null>(null);
  const [ask, confirmSheet] = useConfirm();
  const uninstall = async () => {
    // the page's one confirmation sheet: 卸载 deletes the environment and code, keeping the models
    if (!(await ask({ title: t("ui.install.uninstall"), say: msg("N-INSTALL-UNINSTALL", { title: p.title }), yes: t("ui.install.uninstall"), danger: true }))) return;
    act(api.installs.uninstall(p.name));
  };
  const [error, setError] = useState("");
  const a = p.actions;
  const kind = controlKind(p, full);
  const sheet = open && <InstallSheet p={p} force={open.force} onClose={() => setOpen(null)} onChange={onChange} />;
  // Not this login's, or nothing to do: nothing at all, except a sheet still open on a job that just ended (the row
  // updates underneath it; the log stays until the administrator closes it). The sheet keeps one place in the tree
  // whatever the row shows, so it is never remounted (a remount would open the checklist instead of the finished job).
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
          <span className="inst-state">{job.state === "queued" ? t(JOB_STATE.queued) : current?.label ?? t(JOB_STATE.running)}</span>
          <Button onClick={() => setOpen({ force: false })}>
            {t("ui.install.view")}
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
              tip={why(a, "install") ? tipOf("disabled", why(a, "install")) : p.ready ? tipOf("consequence", t("ui.install.reinstall_tip")) : undefined}
            >
              {button.label}
            </Button>
          )}
          {failed && job && (
            <Button tone="ghost" onClick={() => setOpen({ force: false })}>
              {t("ui.install.log")}
            </Button>
          )}
          {full && shown(a, "rollback") && (
            <Button tone="ghost" disabled={!usable(a, "rollback")} tip={tipOf("disabled", why(a, "rollback"))} onClick={() => act(api.installs.rollback(p.name))}>
              {t("ui.install.rollback")}
            </Button>
          )}
          {full && shown(a, "uninstall") && (
            <Button
              tone="ghost"
              disabled={!usable(a, "uninstall")}
              tip={why(a, "uninstall") ? tipOf("disabled", why(a, "uninstall")) : tipOf("consequence", t("ui.install.uninstall_tip"))}
              onClick={() => void uninstall()}
            >
              {t("ui.install.uninstall")}
            </Button>
          )}
        </div>
      )}
      {failed && job?.result && <p className="inst-note err">{current ? t("ui.install.at_step", { step: current.label, text: job.result.text }) : job.result.text}</p>}
      {error && <p className="inst-note err">{error}</p>}
    </div>
    {sheet}
    </>
  );
}

/** The steps as one bar: a segment per step, coloured by its state; `compact` (in the table row) without the step names. */
function StepBar({ steps, compact = false }: { steps: InstallStep[]; compact?: boolean }) {
  return (
    <div className={`inst-steps${compact ? " compact" : ""}`}>
      {steps.map((s) => (
        <span key={s.id} className={`inst-step ${s.state}`} {...tipAttrs(s.message ? tipOf("error", t("ui.install.step_tip_said", { step: s.label, state: t(STEP_STATE[s.state]), said: s.message.text })) : tipOf("value", t("ui.install.step_tip", { step: s.label, state: t(STEP_STATE[s.state]) })))}>
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
    <Sheet title={force ? t("ui.install.reinstall_title", { title: p.title }) : t("ui.install.install_title", { title: p.title })} width={760} onClose={onClose}>
      {jobId ? (
        <JobView id={jobId} onEnded={onChange} onRetry={() => setJobId(null)} />
      ) : (
        <div className="inst-sheet">
          <p className="inst-note">{t("ui.install.checklist_lede")}</p>
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
            !error && <p className="inst-note">{t("ui.install.checking")}</p>
          )}
          {error && <p className="inst-note err">{error}</p>}
          <div className="inst-sheet-actions">
            <Button tone="ghost" onClick={check}>
              {t("ui.install.check_again")}
            </Button>
            <Button tone="primary" disabled={!checklist || !checklist.ready || asking} onClick={start} tip={blocked ? tipOf("disabled", t("ui.install.blocked", { count: blocked })) : undefined}>
              {t("ui.install.start")}
            </Button>
          </div>
        </div>
      )}
    </Sheet>
  );
}

/** A log line that carries a message's code ("[W-INSTALL-RETRY] ..."): its level, for the colour. A code is what
 * messages/format.ts CODE says (the server's own rule, lab2shot/messages CODE, with its level letters). */
const level = (line: string): string => {
  const code = /^\[([^\]]+)\]/.exec(line)?.[1] ?? "";
  return CODE.test(code) ? code[0] : "";
};

/** A web address in a sentence: up to a space, or a closing punctuation mark of either language (written as escapes:
 * ，。；、）」 are U+FF0C U+3002 U+FF1B U+3001 U+FF09 U+300D). */
const URL_IN_TEXT = /(https:\/\/[^\s\uFF0C\u3002\uFF1B\u3001\uFF09)\u300D]+)/;

/** A sentence with its web addresses as links (a request page, a download page). */
function Linked({ text }: { text: string }) {
  const parts = text.split(URL_IN_TEXT);
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
      every: (last) => (taskLive(last) || last?.lines.length ? 1000 : 0),
      afterError: 3000,
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

  if (!job) return <p className="inst-note">{error || <Loading what={t("ui.install.loading_progress")} />}</p>;
  const failedStep = job.steps.find((s) => s.state === "failed");
  return (
    <div className="inst-sheet">
      <StepBar steps={job.steps} />
      <div className="inst-row">
        <span className={`chip inst-job-state ${job.state}`}>{t(JOB_STATE[job.state])}</span>
        {taskLive(job) && (
          <Button tone="ghost" onClick={() => api.installs.cancel(job.id).catch((e) => setError(reasonOf(e)))} tip={tipOf("consequence", t("ui.install.cancel_tip"))}>
            {t("ui.common.cancel")}
          </Button>
        )}
        {job.state === "failed" && (
          <Button tone="primary" onClick={onRetry}>
            {t("ui.install.retry_step")}
          </Button>
        )}
      </div>
      {job.result && job.state !== "done" && (
        <p className="inst-note err">
          {failedStep ? t("ui.install.step_prefix", { step: failedStep.label }) : ""}
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
