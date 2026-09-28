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

export { editorContext } from "./feedbackContext";

/** 提交反馈: a button in the top bar that opens a dialog where the user describes the problem. Attached are a screenshot
 * of the page (taken when the dialog opens, without asking for permission; or images the user attaches, pastes or
 * drops) and the diagnostics needed to locate the problem (diagnostics.ts, with the page's additions as `context`; the
 * server adds its own), all shown collapsed before sending. The text is kept as a draft until sent. The administrator
 * reads and answers it in the admin page's 用户反馈; 我的反馈, the dialog's other tab, shows users their own feedback
 * with its status and reply, and a dot on the button indicates new activity. */

const MAX_TEXT = 10_000; // the server's limits (lab2shot/feedback.py)
const MAX_IMAGES = 3;
const MAX_IMAGE = 5 << 20;
// the images the server takes (lab2shot/feedback.py _IMAGES, by their first bytes): anything else is refused here, before
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

/** One item of the user's own feedback, as the user sees it (lab2shot/feedback.py PUBLIC). */
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
}

const POLL_MS = 60_000;

/** The account's own feedback: read when a page opens, every minute while the tab is visible, and when it becomes
 * visible again. */
export const useMine = create<{ items: MyFeedback[] | null; unread: number; problem: string; load: () => Promise<void>; read: () => Promise<void> }>((set) => ({
  items: null,
  unread: 0,
  problem: "", // why the list could not be read (the last try); the polling asks again later
  load: async () => {
    try {
      set({ ...(await json<{ items: MyFeedback[]; unread: number }>("GET", "/api/feedback/mine")), problem: "" });
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
  useEffect(() => startPolling({ read: load, every: POLL_MS }).stop, [load, who]);
}

export function FeedbackButton({ context, tone }: { context?: () => Record<string, unknown>; tone?: "ghost" }) {
  const [open, setOpen] = useState(false);
  const unread = useMine((s) => s.unread);
  useMinePolling();
  return (
    <>
      <Button
        tone={tone}
        tip={`提交反馈：写下遇到的问题或建议，自动附上查问题需要的资料（发送前可以先看），管理员在管理页面能看到；「我的反馈」里看自己提过的和管理员的回复${unread ? `\n有 ${unread} 条反馈有新的回复或状态变化` : ""}`}
        layout="fb-button"
        onClick={() => setOpen(true)}
      >
        提交反馈
        {unread > 0 && <span className="fb-unread" aria-label={`${unread} 条新回复`} />}
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
    <Sheet title={tab === "mine" ? "我的反馈" : "提交反馈"} width={680} onClose={onClose}>
      <Segmented
        label="反馈"
        tabs
        size="md"
        layout="fb-tabs"
        value={tab}
        options={[
          { value: "write", label: "写反馈", tip: "写一条新的反馈" },
          { value: "mine", label: <>我的反馈{count ? ` ${count}` : ""}{unread > 0 && <span className="fb-unread" />}</>, tip: "当前账号提交过的反馈：处理到哪了、管理员的回复" },
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
  // 不提供类别选择器；服务器端空字符串是合法的「未选择」，因此发送空值
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
    if (blob) addPicture(blob, "页面截图.png", true);
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
          addPicture(f, f.name || "粘贴的图像.png");
        }
      }
    };
    window.addEventListener("paste", paste);
    return () => window.removeEventListener("paste", paste);
  }, []);

  const textWrong = !text.trim() ? "写清遇到了什么问题、在哪一步出现" : text.length > MAX_TEXT ? `写得太长了：${text.length} 个字，最多 ${MAX_TEXT} 个字` : "";

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
        <div className="fb-done-mark">已提交</div>
        <p className="tpl-desc">
          谢谢！管理员会在管理页面的「用户反馈」里看到这条反馈和附带的资料（编号 {sent.id}，{stampText(sent.at * 1000 / 1000)}）。管理员回复了，「提交反馈」按钮上会出现一个点，在「我的反馈」里看。
        </p>
        <div className="dialog-row fb-actions">
          <Button tip="看自己提过的反馈和管理员的回复" onClick={() => (setSent(null), setText(""), setTouched(false), onMine())}>
            看我的反馈
          </Button>
          <Button tip="关掉这个窗口" tone="primary" onClick={onClose}>
            关闭
          </Button>
        </div>
      </div>
    );

  const groups: { label: string; tip: string; count: string; body: unknown }[] = diag
    ? [
        { label: "环境", tip: "浏览器、系统、屏幕、语言、这台电脑的客户端编号，和服务器的版本", count: `${server.version ? `Lab2Shot ${server.version} · ${server.revision}` : ""}`, body: { browser: diag.browser, client: diag.client, server } },
        ...(diag.help ? [{ label: "页面", tip: "正在看的帮助页面，和它显示的扩展包、安装的情况", count: String((diag.help as { title?: string }).title ?? ""), body: diag.help }] : []),
        ...(diag.graph ? [{ label: "节点图", tip: "现在的节点图（只有文件的名字和位置，不含文件本身）、选中和显示的节点、当前帧和计算范围", count: `${(diag.graph as { nodes?: unknown[] }).nodes?.length ?? 0} 个节点`, body: { editor: diag.editor, graph: diag.graph } }] : []),
        { label: "日志", tip: "页面日志里最近的记录（最多 200 条）：出现过的提示、每次计算的经过", count: `${diag.log.length} 条`, body: diag.log },
        { label: "错误", tip: "打开页面以来页面自己出的错，和服务器拒绝或没连上的请求", count: `${diag.errors.length + diag.requests.length} 个`, body: { errors: diag.errors, requests: diag.requests } },
        { label: "任务", tip: "这个浏览器最近提交的计算任务：状态和报错；服务器还会附上出错节点的日志和服务日志里和它们有关的行", count: jobs ? `${jobs.length} 个` : "…", body: jobs ?? [] },
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
          placeholder="遇到了什么问题？当时在做什么？希望是什么样？"
          data-tip="必填：写得越具体，越容易找到问题；写的内容会自动留着，关掉窗口也不会丢"
          onChange={(e) => (setText(e.target.value), setProblem(null))}
        />
        {touched && textWrong && <div className="fb-why">{textWrong}</div>}
        <div className="fb-pics">
          {pictures.map((p) => (
            <figure key={p.url} className="fb-pic">
              <img src={p.url} alt={p.name} />
              <figcaption>
                <span data-user-data data-tip={p.page ? "页面截图" : p.name}>{p.page ? "页面截图" : p.name}</span>
                <Button tip="不附这张图" tone="ghost" size="sm" onClick={() => (URL.revokeObjectURL(p.url), setPictures((all) => all.filter((x) => x !== p)))}>
                  去掉
                </Button>
              </figcaption>
            </figure>
          ))}
          {shooting && <div className="fb-pic fb-pic-wait">正在截图…</div>}
        </div>
        <div className="fb-row">
          <Button tip={`附上自己截的图（最多 ${MAX_IMAGES} 张，每张 ${sizeText(MAX_IMAGE)} 以内）；也可以粘贴（Ctrl+V）或拖进这个窗口`} disabled={pictures.length >= MAX_IMAGES} onClick={() => fileInput.current?.click()}>
            附图像
          </Button>
          {!shooting && !pictures.some((p) => p.page) && (
            <Button tip="重新截一张当前页面（不含这个窗口）" disabled={pictures.length >= MAX_IMAGES} onClick={() => void shoot()}>
              截当前页面
            </Button>
          )}
          <span className="tpl-desc">{shotFailed ? "页面截图没成功：可以用「附图像」附上自己截的图" : "可以粘贴或拖进图像"}</span>
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
          <summary data-tip="提交时一起发给服务器的资料，点开可以逐项看；不含密码和令牌">
            附带的资料：{groups.map((g) => `${g.label}${g.count ? ` ${g.count}` : ""}`).join(" · ")}
          </summary>
          <p className="tpl-desc">服务器还会补上：出错节点的日志、服务日志里和这些任务有关的行、节点图用到的扩展包是否装好、显卡和内存的状态。密码和令牌一律去掉。</p>
          {groups.map((g) => (
            <details key={g.label} className="fb-group">
              <summary data-tip={g.tip}>
                {g.label}
                {g.count && <span className="fb-count">{g.count}</span>}
              </summary>
              <pre className="fb-pre">{JSON.stringify(g.body, null, 1).slice(0, 60_000)}</pre>
            </details>
          ))}
        </details>
        {problem && <div className="fb-why" data-code={problem.code}>{problem.text}</div>}
        <div className="dialog-row fb-actions">
          <Button tip="写的内容会留着，下次打开还在" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip="把反馈连同附上的截图和诊断资料发给管理员" tone="primary" disabled={sending || !diag || shooting} onClick={() => void submit()}>
            {sending ? "正在提交…" : shooting ? "正在截图…" : "提交"}
          </Button>
        </div>
      </div>
    </>
  );
}
