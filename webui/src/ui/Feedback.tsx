import "./feedback.css";
import "./log.css";
import { useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { json } from "../platform/http";
import { clientInfo } from "../platform/client";
import { collectDiagnostics, type Diagnostics } from "../state/diagnostics";
import { useSession } from "../state/session";
import { msg, reasonOf, say, type Message } from "../state/say";
import { snapshotPage } from "./snapshot";
import { readLocalJSON, writeLocal } from "../platform/util";
import { Sheet } from "./Sheet";
import { sizeText, stampText } from "../platform/format";
import { startPolling } from "../platform/poll";
import { Button, Segmented } from "./Button";
import { MyFeedbackList } from "./FeedbackMine";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

export { editorContext } from "./feedbackContext";

/** 提交反馈: a button in the top bar that opens a dialog where the user describes the problem. Attached are a screenshot
 * of the page (taken when the dialog opens, without asking for permission; or images the user attaches, pastes or
 * drops) and the diagnostics needed to locate the problem (diagnostics.ts, with the page's additions as `context`; the
 * server adds its own), all shown collapsed before sending. The text is kept as a draft until sent. The administrator
 * reads and answers it in the admin page's 用户反馈; 我的反馈, the dialog's other tab, shows users their own feedback
 * with its status and reply, and a dot on the button indicates new activity. */

const MAX_TEXT = 10_000; // the server's limits (lab2shot/site/feedback.py)
const MAX_IMAGES = 3;
const MAX_IMAGE = 5 << 20;
// the images the server takes (lab2shot/site/feedback.py _IMAGES, by their first bytes): anything else is refused here, before
// the whole feedback is sent and refused there
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const draftKey = (account: number | undefined) => `lab2shot.feedbackDraft.${account}`; // one per account: never shown to the next person on this browser

interface Picture {
  name: string;
  blob: Blob;
  url: string;
  page: boolean; // the screenshot taken when the dialog opened
}

interface JobPreview {
  id: string;
  title: string;
  state: string;
  submitted: number;
  error: string | null;
  error_log: boolean;
  server_log: number;
}

const toBase64 = (blob: Blob) =>
  new Promise<string>((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });

/** One item of the user's own feedback, as the user sees it (lab2shot/site/feedback.py PUBLIC). */
export interface MyFeedback {
  id: string;
  at: number;
  category: string;
  category_label: string;
  text: string;
  status: "new" | "seen" | "solved";
  status_label: string;
  reply: string;
  replied: number | null;
  changed: number | null;
  unread: boolean; // a status change or reply the user has not yet seen
  images: number;
  reward: string; // a valid one: what it earned the account, in the server's words ("" otherwise: 无效 is never said)
}

const POLL_MS = 60_000;

/** The account's own feedback: read when a page opens, every minute while the tab is visible, and when it becomes
 * visible again. */
export const useMine = create<{ items: MyFeedback[] | null; unread: number; rule: string; problem: string; load: () => Promise<void>; read: () => Promise<void> }>((set) => ({
  items: null,
  unread: 0,
  rule: "", // 有效反馈奖励 now, in the server's words ("": none)
  problem: "", // why the list could not be read (the last try); the polling asks again later
  load: async () => {
    try {
      set({ ...(await json<{ items: MyFeedback[]; unread: number; rule: string }>("GET", "/api/feedback/mine")), problem: "" });
    } catch (e) {
      set({ problem: reasonOf(e) });
      throw e; // the polling backs off (platform/poll.ts)
    }
  },
  read: async () => {
    try {
      set({ unread: (await json<{ unread: number }>("POST", "/api/feedback/mine/read", {})).unread });
    } catch {
      /* remains unread */
    }
  },
}));

function useMinePolling(): void {
  const load = useMine((s) => s.load);
  const who = useSession((s) => s.state?.user?.id);
  // only once the login is known: started before, it would read once for nobody and again for the account
  useEffect(() => (who === undefined ? undefined : startPolling({ read: load, every: POLL_MS }).stop), [load, who]);
}

export function FeedbackButton({ context, tone }: { context?: () => Record<string, unknown>; tone?: "ghost" }) {
  const [open, setOpen] = useState(false);
  const unread = useMine((s) => s.unread);
  useMinePolling();
  return (
    <>
      <Button
        tone={tone}
        entry // 和「模板」同一圈转动的彩虹亮边（ui/glow.css .btn.entry）：反馈入口要一眼找得到
        tip={unread ? tipOf("value", t("ui.feedback.unread_tip", { count: unread })) : undefined}
        layout="fb-button"
        onClick={() => setOpen(true)}
      >
        {t("ui.feedback.send")}
        {unread > 0 && <span className="fb-unread" aria-label={t("ui.feedback.unread", { count: unread })} />}
      </Button>
      {open && <FeedbackDialog context={context} onClose={() => setOpen(false)} />}
    </>
  );
}

/** The dialog: 写反馈 and 我的反馈 (the latter opens first when there is new activity). Each keeps its state while the
 * other is shown: the screenshot and the text are not lost. */
function FeedbackDialog({ context, onClose }: { context?: () => Record<string, unknown>; onClose: () => void }) {
  const unread = useMine((s) => s.unread);
  const count = useMine((s) => s.items?.length ?? 0);
  const [tab, setTab] = useState<"write" | "mine">(() => (useMine.getState().unread ? "mine" : "write"));
  const [seen, setSeen] = useState(tab === "mine"); // 我的反馈 has been shown: it stays mounted
  const show = (t: "write" | "mine") => (setTab(t), t === "mine" && setSeen(true));
  return (
    <Sheet title={tab === "mine" ? t("ui.feedback.mine") : t("ui.feedback.send")} width={680} onClose={onClose}>
      <Segmented
        label={t("ui.feedback.label")}
        tabs
        size="md"
        layout="fb-tabs"
        value={tab}
        options={[
          { value: "write", label: t("ui.feedback.write") },
          { value: "mine", label: <>{t("ui.feedback.mine")}{count ? ` ${count}` : ""}{unread > 0 && <span className="fb-unread" />}</> },
        ]}
        onChange={show}
      />
      <div className="fb-pane" hidden={tab !== "write"}>
        <WriteFeedback context={context} onClose={onClose} onMine={() => show("mine")} />
      </div>
      {seen && (
        <div className="fb-pane" hidden={tab !== "mine"}>
          <MyFeedbackList onWrite={() => show("write")} />
        </div>
      )}
    </Sheet>
  );
}

/** 写反馈: the text, the images and the accompanying diagnostics; the draft is kept until sent. */
function WriteFeedback({ context, onClose, onMine }: { context?: () => Record<string, unknown>; onClose: () => void; onMine: () => void }) {
  const draftAt = draftKey(useSession.getState().state?.user?.id);
  const draft = readLocalJSON<{ text?: string }>(draftAt, {});
  const [text, setText] = useState(draft.text ?? "");
  // there is no category picker; the server accepts an empty string as "none chosen", so an empty value is sent
  const category = "";
  const [pictures, setPictures] = useState<Picture[]>([]);
  const [shooting, setShooting] = useState(true);
  const [shotFailed, setShotFailed] = useState(false);
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const [jobs, setJobs] = useState<JobPreview[] | null>(null);
  const [server, setServer] = useState<Record<string, string>>({});

  const [touched, setTouched] = useState(false);
  const [sending, setSending] = useState(false);
  const [problem, setProblem] = useState<Message | null>(null);
  const [sent, setSent] = useState<{ id: string; at: number } | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const pics = useRef<Picture[]>([]);
  pics.current = pictures;

  const addPicture = (blob: Blob, name: string, page = false) => {
    if (!IMAGE_TYPES.includes(blob.type)) return setProblem(msg("B-FEEDBACK-NOTIMAGE", { name }));
    if (blob.size > MAX_IMAGE) return setProblem(msg("B-FEEDBACK-IMAGETOOBIG", { name, size: sizeText(blob.size), max: sizeText(MAX_IMAGE) }));
    if (pics.current.length >= MAX_IMAGES) return setProblem(msg("B-FEEDBACK-TOOMANY", { max: MAX_IMAGES }));
    setProblem(null);
    pics.current = [...pics.current, { name, blob, url: URL.createObjectURL(blob), page }]; // several at once
    setPictures(pics.current);
  };

  const shoot = async () => {
    setShooting(true);
    setShotFailed(false);
    const blob = await snapshotPage(".scrim, .tip").catch(() => null);
    setShooting(false);
    if (blob) addPicture(blob, t("ui.feedback.shot_file"), true);
    else setShotFailed(true);
  };

  useEffect(() => {
    // the page as it is (excluding the dialog), the data sent with the feedback, and what the server will add
    void shoot();
    const d = collectDiagnostics(context?.() ?? {});
    setDiag(d);
    json<{ jobs: JobPreview[]; environment: Record<string, string> }>("POST", "/api/feedback/jobs", { jobs: d.jobs.map((j) => j.id) })
      .then(
        (r) => (setJobs(r.jobs), setServer(r.environment)),
        () => setJobs([]),
      );
    return () => pics.current.forEach((p) => URL.revokeObjectURL(p.url));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!sent) writeLocal(draftAt, JSON.stringify({ text }));
  }, [text, sent]);

  useEffect(() => {
    const paste = (e: ClipboardEvent) => {
      for (const item of Array.from(e.clipboardData?.items ?? [])) {
        const f = item.kind === "file" ? item.getAsFile() : null;
        if (f && f.type.startsWith("image/")) { // an image of another format is said so (addPicture)
          e.preventDefault();
          addPicture(f, f.name || t("ui.feedback.pasted_file"));
        }
      }
    };
    window.addEventListener("paste", paste);
    return () => window.removeEventListener("paste", paste);
  }, []);

  const textWrong = !text.trim() ? t("ui.feedback.text_empty") : text.length > MAX_TEXT ? t("ui.feedback.text_long", { count: text.length, max: MAX_TEXT }) : "";

  const submit = async () => {
    setTouched(true);
    setProblem(null);
    if (textWrong || !diag) return;
    setSending(true);
    try {
      const images = await Promise.all(pictures.map(async (p) => ({ name: p.name, data: await toBase64(p.blob) })));
      const done = await json<{ id: string; at: number }>("POST", "/api/feedback", { text, category, client: clientInfo(), diagnostics: { ...diag, sent: Date.now() }, images });
      setSent(done);
      writeLocal(draftAt, "{}");
      useMine.getState().load().catch(() => undefined); // a failed read is kept in the store (problem) and asked again
      say(msg("I-FEEDBACK-SENT", { id: done.id, first: text.trim().split("\n")[0].slice(0, 60) }));
    } catch (e) {
      setProblem(msg("E-FEEDBACK-SENDFAILED", { reason: reasonOf(e) }));
    } finally {
      setSending(false);
    }
  };

  if (sent)
    return (
      <div className="fb-done">
        <div className="fb-done-mark">{t("ui.feedback.sent")}</div>
        <p className="tpl-desc">
          {t("ui.feedback.thanks", { id: sent.id, time: stampText(sent.at * 1000 / 1000) })}
        </p>
        <div className="dialog-row fb-actions">
          <Button onClick={() => (setSent(null), setText(""), setTouched(false), onMine())}>
            {t("ui.feedback.see_mine")}
          </Button>
          <Button tone="primary" onClick={onClose}>
            {t("ui.common.close")}
          </Button>
        </div>
      </div>
    );

  const groups: { label: string; count: string; body: unknown }[] = diag
    ? [
        { label: t("ui.feedback.group_environment"), count: `${server.version ? `Lab2Shot ${server.version} · ${server.revision}` : ""}`, body: { browser: diag.browser, client: diag.client, server } },
        ...(diag.help ? [{ label: t("ui.feedback.group_page"), count: String((diag.help as { title?: string }).title ?? ""), body: diag.help }] : []),
        ...(diag.graph ? [{ label: t("ui.feedback.group_graph"), count: t("ui.feedback.count_nodes", { count: (diag.graph as { nodes?: unknown[] }).nodes?.length ?? 0 }), body: { editor: diag.editor, graph: diag.graph } }] : []),
        { label: t("ui.feedback.group_log"), count: t("ui.feedback.count_entries", { count: diag.log.length }), body: diag.log },
        { label: t("ui.feedback.group_errors"), count: t("ui.feedback.count_errors", { count: diag.errors.length + diag.requests.length }), body: { errors: diag.errors, requests: diag.requests } },
        { label: t("ui.feedback.group_jobs"), count: jobs ? t("ui.feedback.count_errors", { count: jobs.length }) : "…", body: jobs ?? [] },
      ]
    : [];

  return (
    <>
      <div
        className="fb-form"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          Array.from(e.dataTransfer.files).forEach((f) => addPicture(f, f.name));
        }}
      >
        <textarea
          className={`field fb-text${touched && textWrong ? " bad" : ""}`}
          value={text}
          autoFocus
          placeholder={t("ui.feedback.placeholder")}
          onChange={(e) => (setText(e.target.value), setProblem(null))}
        />
        {touched && textWrong && <div className="fb-why">{textWrong}</div>}
        <div className="fb-pics">
          {pictures.map((p) => (
            <figure key={p.url} className="fb-pic">
              <img src={p.url} alt={p.name} />
              <figcaption>
                <span data-user-data {...tipAttrs(p.page ? undefined : tipOf("truncated", p.name))}>{p.page ? t("ui.feedback.shot") : p.name}</span>
                <Button tone="ghost" size="sm" onClick={() => (URL.revokeObjectURL(p.url), setPictures((all) => all.filter((x) => x !== p)))}>
                  {t("ui.common.remove")}
                </Button>
              </figcaption>
            </figure>
          ))}
          {shooting && <div className="fb-pic fb-pic-wait">{t("ui.feedback.shooting")}</div>}
        </div>
        <div className="fb-row">
          <Button tip={pictures.length >= MAX_IMAGES ? tipOf("disabled", t("ui.feedback.max_images", { max: MAX_IMAGES })) : undefined} disabled={pictures.length >= MAX_IMAGES} onClick={() => fileInput.current?.click()}>
            {t("ui.feedback.attach")}
          </Button>
          {!shooting && !pictures.some((p) => p.page) && (
            <Button tip={pictures.length >= MAX_IMAGES ? tipOf("disabled", t("ui.feedback.max_images", { max: MAX_IMAGES })) : undefined} disabled={pictures.length >= MAX_IMAGES} onClick={() => void shoot()}>
              {t("ui.feedback.shoot")}
            </Button>
          )}
          <span className="tpl-desc">{shotFailed ? t("ui.feedback.shot_failed") : t("ui.feedback.paste_hint")}</span>
          <input
            ref={fileInput}
            type="file"
            accept={IMAGE_TYPES.join(",")}
            multiple
            hidden
            onChange={(e) => {
              Array.from(e.target.files ?? []).forEach((f) => addPicture(f, f.name));
              e.target.value = "";
            }}
          />
        </div>
        <details className="fb-diag">
          <summary>
            {t("ui.feedback.attached", { groups: groups.map((g) => `${g.label}${g.count ? ` ${g.count}` : ""}`).join(" · ") })}
          </summary>
          <p className="tpl-desc">{t("ui.feedback.server_adds")}</p>
          {groups.map((g) => (
            <details key={g.label} className="fb-group">
              <summary>
                {g.label}
                {g.count && <span className="fb-count">{g.count}</span>}
              </summary>
              <pre className="fb-pre">{JSON.stringify(g.body, null, 1).slice(0, 60_000)}</pre>
            </details>
          ))}
        </details>
        {problem && <div className="fb-why" data-code={problem.code}>{problem.text}</div>}
        <div className="dialog-row fb-actions">
          <Button tip={tipOf("consequence", t("ui.feedback.cancel_tip"))} tone="ghost" onClick={onClose}>
            {t("ui.common.cancel")}
          </Button>
          <Button tone="primary" disabled={sending || !diag || shooting} onClick={() => void submit()}>
            {sending ? t("ui.feedback.sending") : shooting ? t("ui.feedback.shooting") : t("ui.feedback.submit")}
          </Button>
        </div>
      </div>
    </>
  );
}
