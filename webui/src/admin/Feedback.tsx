import { msg, type Message } from "../messages/message";
import { LabelRow } from "../ui/LabelRow";
import { ByUser } from "./Resources";
import { useSignedIn } from "../state/session";
import { shown } from "../api/applies";
import { useCallback, useEffect, useState } from "react";
import { Sheet } from "../ui/Sheet";
import { adminApi, type FeedbackDetail, type FeedbackRating, type FeedbackStatus } from "../api/admin";
import { Section, useAdmin } from "./common";
import { stampText, whenText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button, ButtonLink, Segmented } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
// the feedback's own look (the detail's header, its groups, the status chips) is one file shared with 提交反馈 (ui/Feedback.tsx);
// the admin page does not load that component, so it brings the file itself
import "../ui/feedback.css";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** 用户反馈 section: submissions from 提交反馈 (lab2shot/site/feedback.py): sender, time, text, attached screenshots
 * and diagnostics. Supports answering (a status and a reply visible in 我的反馈, plus a private note), downloading
 * everything, and deleting. */

const FEEDBACK_STATUS: Record<FeedbackStatus, { label: () => string }> = {
  new: { label: () => t("ui.admin.feedback.status_new") },
  seen: { label: () => t("ui.admin.feedback.status_seen") },
  solved: { label: () => t("ui.admin.feedback.status_solved") },
};
// the rating (评定), apart from the status: 有效 earns the sender's account time (lab2shot/site/feedback.py rate)
const RATINGS: { value: FeedbackRating; label: () => string }[] = [
  { value: "", label: () => t("ui.admin.feedback.unrated") },
  { value: "valid", label: () => t("ui.admin.feedback.valid") },
  { value: "invalid", label: () => t("ui.admin.feedback.invalid") },
];
const ratingFilter = (r: FeedbackRating) => r || "unrated"; // the listing's filter calls 未评定 "unrated"
const PERIODS = [
  { id: "all", label: () => t("ui.admin.resources.all"), days: 0 },
  { id: "day", label: () => t("ui.admin.traffic.today"), days: 1 },
  { id: "week", label: () => t("ui.admin.resources.days", { n: 7 }), days: 7 },
  { id: "month", label: () => t("ui.admin.resources.days", { n: 30 }), days: 30 },
];

const full = (at: number | null | undefined) => (at ? stampText(at) : "—"); // the server's times: seconds
const pageTime = (ms: number) => stampText(ms / 1000); // the page's own diagnostics (its log, errors, requests): milliseconds

export function FeedbackSection() {
  const { problem, refreshOverview } = useAdmin();
  const [status, setStatus] = useState<FeedbackStatus | "">("");
  const [rating, setRating] = useState(""); // "": every rating; else ratingFilter
  const [period, setPeriod] = useState("all");
  const [person, setPerson] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const read = useCallback(() => {
    const days = PERIODS.find((p) => p.id === period)?.days ?? 0;
    const since = days ? (days === 1 ? new Date().setHours(0, 0, 0, 0) / 1000 : Date.now() / 1000 - days * 86400) : undefined;
    return adminApi.feedback({ status, since, person, rating });
  }, [status, period, person, rating]);
  const { data: view, reload: reload } = usePoll(read, 20_000, { onError: (e) => problem(reasonOf(e)) });
  const changed = () => (reload(), refreshOverview());

  return (
    <Section
      title={t("ui.admin.nav.feedback")}
      lede={t("ui.admin.feedback.lede")}
      actions={
        <Button tone="ghost" onClick={reload}>
          {t("ui.admin.common.refresh")}
        </Button>
      }
    >
      <div className="fb-filters">
        <Segmented label={t("ui.admin.resources.state")} value={status} options={[{ value: "", label: t("ui.admin.resources.all") }, ...(Object.keys(FEEDBACK_STATUS) as FeedbackStatus[]).map((s) => ({ value: s, label: `${FEEDBACK_STATUS[s].label()}${view ? ` ${view.counts[s]}` : ""}` }))]} onChange={setStatus} />
        <Segmented label={t("ui.admin.feedback.rating")} value={rating} options={[{ value: "", label: t("ui.admin.resources.all") }, ...RATINGS.map((r) => ({ value: ratingFilter(r.value), label: `${r.label()}${view?.ratings ? ` ${view.ratings[r.value] ?? 0}` : ""}` }))]} onChange={setRating} />
        <Segmented label={t("ui.admin.resources.time")} value={period} options={PERIODS.map((p) => ({ value: p.id, label: p.label() }))} onChange={setPeriod} />
        <input className="field fb-search" value={person} placeholder={t("ui.admin.feedback.find_account")} onChange={(e) => setPerson(e.target.value)} />
      </div>
      {view === null ? (
        <p className="adm-lede">{t("ui.admin.common.reading")}</p>
      ) : view.items.length ? (
        <div className="q-table-wrap">
          <table className="q-table fb-table">
            <thead>
              <tr>
                <th>{t("ui.admin.security.col_time")}</th>
                <th>{t("ui.admin.feedback.col_person")}</th>
                <th>{t("ui.admin.users.department")}</th>
                <th>{t("ui.admin.feedback.col_category")}</th>
                <th>{t("ui.admin.feedback.col_text")}</th>
                <th>{t("ui.admin.feedback.col_shots")}</th>
                <th>{t("ui.admin.resources.state")}</th>
                <th>{t("ui.admin.feedback.rating")}</th>
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
                    {f.reply && <span className="q-targets">{t("ui.admin.feedback.reply_is", { text: f.reply })}</span>}
                    {f.note && <span className="q-targets">{t("ui.admin.feedback.note_is", { text: f.note })}</span>}
                  </td>
                  <td className="tnum">{f.images.length || "—"}</td>
                  <td>
                    <StatusChip status={f.status} />
                  </td>
                  <td>
                    <RatingChip rating={f.rating} label={f.rating_label} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-empty">{status || rating || period !== "all" || person ? t("ui.admin.feedback.none_match") : t("ui.admin.feedback.none")}</p>
      )}
      {open && <FeedbackSheet id={open} onClose={() => setOpen(null)} onChanged={changed} />}
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="feedback" />
    </Section>
  );
}

function StatusChip({ status }: { status: FeedbackStatus }) {
  return (
    <span className={`chip fb-status ${status}`}>
      {FEEDBACK_STATUS[status].label()}
    </span>
  );
}

function RatingChip({ rating, label }: { rating: FeedbackRating; label: string }) {
  return rating ? <span className={`chip fb-rating ${rating}`}>{label}</span> : <span className="dim">{label}</span>;
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
          <img src={adminApi.feedbackFile(id, name)} alt={name} onClick={() => setBig((b) => (b === name ? null : name))} />
          {big === name && (
            <figcaption>
              <a href={adminApi.feedbackFile(id, name)} target="_blank" rel="noreferrer">{t("ui.admin.feedback.original")}</a>
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
  const [rated, setRated] = useState(""); // what the last rating did, in the server's words
  const [rating, setRatingBusy] = useState(false);
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

  // the rating is saved at once, apart from the status and the reply (it gives or takes back the sender's time)
  const rate = async (r: FeedbackRating) => {
    if (!f || r === f.rating || rating) return;
    setRatingBusy(true);
    try {
      const done = await adminApi.rateFeedback(id, r);
      setF((d) => (d ? { ...d, ...done.feedback } : d));
      setRated(done.said);
      problem(null);
      onChanged();
    } catch (e) {
      problem((e as Error).message);
    } finally {
      setRatingBusy(false);
    }
  };

  const [ask, confirmSheet] = useConfirm();
  const close = async () => {
    if (!dirty || (await ask({ title: t("ui.admin.feedback.unsaved"), say: msg("N-FEEDBACK-DISCARD"), yes: t("ui.admin.feedback.discard"), danger: true }))) onClose();
  };

  const state = useSignedIn();

  const remove = async () => {
    if (!f || !(await ask({ title: t("ui.admin.feedback.delete_title"), say: msg("N-FEEDBACK-DELETE", { person: f.person, at: full(f.at) }), yes: t("ui.admin.user.delete"), danger: true }))) return;
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
    <Sheet title={t("ui.admin.nav.feedback")} width={980} solid onClose={() => void close()}>
      {!f ? (
        <p className="adm-lede">{t("ui.admin.common.reading")}</p>
      ) : (
        <div className="fb-detail">
          <div className="fb-meta">
            <span className="q-who">{f.person}</span>
            <span>{f.department}</span>
            <span className="tnum">{full(f.at)}</span>
            {f.category_label && <span className="chip">{f.category_label}</span>}
            <StatusChip status={f.status} />
            {f.updated && <span className="q-targets">{t("ui.admin.feedback.updated", { at: full(f.updated), by: f.updated_by })}</span>}
          </div>
          <div className="fb-text-full">{f.text}</div>
          {f.images.length > 0 && <Shots id={id} names={f.images} />}
          <div className="fb-answer lgrid">
            {shown(state?.applies, "feedback.rate") && (
              <LabelRow className="fb-status-row fb-rate-row" labelClass="who-label" label={t("ui.admin.feedback.rating")}>
                <div className="fb-rate">
                  <Segmented label={t("ui.admin.feedback.rating")} value={f.rating} options={RATINGS.map((r) => ({ value: r.value, label: r.label() }))} onChange={(r) => void rate(r)} />
                  <span className="tpl-desc">{rated || `${f.rated ? t("ui.admin.feedback.rated_by", { at: full(f.rated), by: f.rated_by }) : ""}${t("ui.admin.feedback.rating_says")}`}</span>
                </div>
              </LabelRow>
            )}
            <LabelRow className="fb-status-row top" labelAs="label" htmlFor="fb-reply" labelClass="who-label" label={t("ui.admin.feedback.reply")}>
              <textarea id="fb-reply" className="field fb-note" value={reply} placeholder={t("ui.admin.feedback.reply_placeholder")} onChange={(e) => (setReply(e.target.value), setSaved(null))} />
            </LabelRow>
            <LabelRow className="fb-status-row top" labelAs="label" htmlFor="fb-note" labelClass="who-label" label={t("ui.admin.feedback.note")}>
              <textarea id="fb-note" className="field fb-note" value={note} placeholder={t("ui.admin.feedback.note_placeholder")} onChange={(e) => (setNote(e.target.value), setSaved(null))} />
            </LabelRow>
            <div className="dialog-row fb-actions">
              <span className="tpl-desc">
                {saved?.text || (f.reply ? (f.unread ? t("ui.admin.feedback.reply_unread", { at: full(f.replied) }) : t("ui.admin.feedback.reply_read", { at: full(f.replied) })) : t("ui.admin.feedback.visible"))}
              </span>
              <Segmented label={t("ui.admin.resources.state")} value={status} options={(Object.keys(FEEDBACK_STATUS) as FeedbackStatus[]).map((s) => ({ value: s, label: FEEDBACK_STATUS[s].label() }))} onChange={(s) => (setStatus(s), setSaved(null))} />
              <Button tip={tipOf("consequence", t("ui.admin.feedback.save_tip"))} tone="primary" disabled={!dirty} onClick={() => void save()}>
                {t("ui.admin.common.save")}
              </Button>
            </div>
          </div>
          {b ? (
            <div className="fb-groups">
              {page.help !== undefined && (
                <Group label={t("ui.admin.feedback.g_page")} count={String((page.help as { title?: string }).title ?? "")}>
                  <Pre value={page.help} />
                </Group>
              )}
              <Group label={t("ui.admin.feedback.g_graph")} count={graphCount(page.graph)}>
                <Pre value={page.editor ?? t("ui.admin.feedback.no_graph")} />
                {page.graph !== undefined && <Pre value={page.graph} />}
              </Group>
              <Group label={t("ui.admin.feedback.g_log")} count={t("ui.admin.feedback.n_entries", { n: (page.log ?? []).length })}>
                <Pre value={(page.log ?? []).map((e) => `${pageTime(e.t)}${e.count && e.last ? t("ui.admin.feedback.repeat", { n: e.count, last: pageTime(e.last) }) : ""}  [${e.level}]  ${e.text}`).join("\n") || t("ui.admin.feedback.no_log")} />
              </Group>
              <Group label={t("ui.admin.feedback.g_errors")} count={t("ui.admin.feedback.n_items", { n: (page.errors ?? []).length + (page.requests ?? []).length })}>
                <Pre value={(page.errors ?? []).map((e) => `${pageTime(e.t)}${e.count > 1 ? t("ui.admin.feedback.repeat", { n: e.count, last: pageTime(e.last) }) : ""}  ${t(e.kind === "rejection" ? "ui.admin.feedback.rejection" : "ui.admin.feedback.page_error", { message: e.message })}${e.where ? t("ui.admin.feedback.where", { where: e.where }) : ""}${e.stack ? `\n${e.stack}` : ""}`).join("\n\n") || t("ui.admin.feedback.no_errors")} />
                <Pre value={(page.requests ?? []).map((r) => `${pageTime(r.t)}  ${r.method} ${r.url} → ${r.status || t("ui.admin.feedback.no_connection")} (${r.ms} ms)  ${r.message}`).join("\n") || t("ui.admin.feedback.no_failed")} />
              </Group>
              <Group label={t("ui.admin.users.jobs")} count={t("ui.admin.feedback.n_items", { n: (server.jobs ?? []).length })}>
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
                  <p className="adm-empty">{t("ui.admin.feedback.no_jobs")}</p>
                )}
              </Group>
              <Group label={t("ui.admin.feedback.g_env")} count={String(server.revision ?? "")}>
                <Pre value={{ browser: page.browser, client: server.client, page: page.page }} />
                <Pre value={Object.fromEntries(Object.entries(server).filter(([k]) => !["jobs", "server_log", "client"].includes(k)))} />
              </Group>
              <Group label={t("ui.admin.logs.title")} count={t("ui.admin.feedback.n_lines", { n: (server.server_log ?? []).length })}>
                <Pre value={(server.server_log ?? []).join("\n")} />
              </Group>
            </div>
          ) : (
            <p className="adm-empty">{t("ui.admin.feedback.no_bundle")}</p>
          )}
          <div className="dialog-row fb-actions">
            <ButtonLink href={adminApi.feedbackDownload(id)}>
              {t("ui.admin.feedback.download")}
            </ButtonLink>
            {shown(state?.applies, "feedback.delete") && (
              <Button tone="ghost" onClick={() => void remove()}>
                {t("ui.admin.user.delete")}
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
  return nodes ? t("ui.admin.cards.node_count", { n: nodes.length }) : "";
};

function Group({ label, count, children }: { label: string; count: string; children: React.ReactNode }) {
  return (
    <details className="fb-group">
      <summary>
        {label}
        {count && <span className="fb-count">{count}</span>}
      </summary>
      {children}
    </details>
  );
}

const Pre = ({ value }: { value: unknown }) => <pre className="server-log fb-pre">{typeof value === "string" ? value : JSON.stringify(value, null, 1)}</pre>;

