/** 编辑器顶栏：本模块拥有顶栏上的一切（文件菜单、撤销 / 重做、文档名与保存状态、模板入口、服务器负载、传输速率、
 * 队列 / 日志入口、提交被拒的说明）、页面对 /api/load 与队列的轮询，以及时间线上的「计算范围」控件（CookRange）。 */
import { useEffect, useRef, useState } from "react";
import { api, type ServerLoad } from "../api";
import type { Reading } from "../api/status";
import { askStatus } from "../graph/asking";
import { noteServer, useServer } from "../state/server";
import { useUnseenErrors } from "../state/log";
import { waitText } from "../graph/nodes";
import { syncJob } from "../graph/follow";
import { cancelCook, cancelSubmit, cookHold, cookHoldWhy, frameLimitProblemNow, redo, rangeProblemNow, setCookRange, undo } from "../graph/actions";
import { focusCompute, focusFinish, focusUncomputed } from "./focusJob";
import { useSubmitLine } from "../graph/submitLine";
import { Sheet } from "../ui/Sheet";
import { readOnlyWhy, useCookInputs, useReadOnly } from "../state/cookInputs";
import { planOf, useResults } from "../state/results";
import { useLook } from "../state/look";
import { addDir, authorisedDirs } from "../files/localDirs";
import { canReadFolder } from "../files/handles";
import { BrandMark, IconClose, IconGrid, IconRedo, IconUndo } from "../ui/icons";
import { useCatalog } from "../state/catalog";
import { useViewer } from "../state/viewer";
import { editorContext, FeedbackButton } from "../ui/Feedback";
import { PluginsButton } from "../ui/Plugins";
import { ReleasesButton } from "../ui/Releases";
import { AccountChip } from "../ui/Account";
import { usePoll } from "../platform/poll";
import { useSettled } from "../platform/settled";
import { useUploads } from "../transfer/uploads";
import { Rate } from "../transfer/rate";
import { rateParts, sizeText } from "../platform/format";
import { readPref, writePref } from "../platform/storage";
import { trafficTotals } from "../platform/traffic";
import { pushLost } from "../platform/events";
import { quietFor } from "../platform/http";
import { msg, textOf } from "../messages/message";
import { Button, IconButton } from "../ui/Button";
import { Kbd, Menu, type MenuRow } from "../ui/Menu";
import { QueueSheet } from "./ChromeSheets";
import { SaveToLibrarySheet } from "./MyTemplates";
import { shown } from "../api/applies";
import { useSession } from "../state/session";
import { ModeSwitch, useAppMode } from "./AppMode";
import { useWriteLock } from "../ui/writeLock";
import { nodeRef } from "../graph/naming";
import { pick, t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";
import { useFitBar } from "../ui/fitBar";

export { OPEN_GRAPH, TabBanner, UnsavedSheet } from "./ChromeSheets";
export { TemplatesSheet } from "./Templates";

// 空闲时不为队列单独发送请求：队列窗口关闭且当前账号没有任务时，页面只需要两个开关、帧数上限与存储占用，
// 这些随 `/api/load` 一并返回（lab2shot/farm/queue.py Queue.load）。一次往返的开销主要是请求头与会话 cookie，
// 因此合并请求比放慢轮询更有效。
const quietQueue = async () => null;

/** 编辑器顶栏，分两组。左边是文档：标志、节点模式 / 应用模式（editor/AppMode.tsx）、文件菜单、撤销 / 重做，以及图名与
 * 是否已保存。右边是其余一切：「模板」在前（页面唯一的入口，带连线配色的亮边），然后是「队列」（附任务状态与「取消」）、
 * 「日志」「提交反馈」「更新说明」和账号。这里没有「计算 / 提交」按钮：节点由其按钮参数或右键菜单计算，所有「输出」
 * 一起用 Ctrl+Shift+Enter；「计算范围」在时间线上，与帧在一起。 */
export function TopBar({ onNew, onOpen, onSave }: { onNew: () => void; onOpen: () => void; onSave: (saveAs: boolean) => void }) {
  const meta = useCookInputs((s) => s.meta);
  const file = useViewer((s) => s.file);
  const dirty = useViewer((s) => s.dirty);
  const setTemplatesOpen = useViewer((s) => s.setTemplatesOpen);
  const job = useResults((s) => s.job);
  const [queueOpen, setQueueOpen] = useState(false);
  const setLogOpen = useViewer((s) => s.setLogOpen);
  const errors = useUnseenErrors();
  const templates = useCatalog()?.templates ?? 0; // 只取数量：列表本身等打开「模板」时才取
  // 队列窗口打开时：每 1.5 秒一次，附带机器负载；关闭时：无变化则放慢，该图的任务开始或结束时立即请求。
  // 仅在队列窗口打开或当前账号有运行中的任务时请求完整的队列数据；其余时候所需的三项来自 /api/load。
  // 「排队 N · 计算 x/y」同样来自精简的 /api/load。
  const needQueue = queueOpen || !!job;
  const quiet = queueOpen ? 1500 : 30_000;
  const asked = usePoll(needQueue ? api.queue : quietQueue, needQueue ? (queueOpen ? 1500 : 5000) : null,
                        { key: job?.id ?? "", slowest: quiet });
  const queue = asked.data;
  // 队列里已看不到这张图的任务在排队 / 计算，而页面还记着它（事件流没说结束：断过、服务器重启过）：按服务器核对一次，
  // 以服务器为准（graph/follow.ts syncJob 重新查询后才下结论，刚提交、这一轮轮询还没带上的任务不会被误清）
  const jobId = job?.id;
  useEffect(() => {
    if (jobId && queue && !queue.jobs.some((j) => j.id === jobId && (j.state === "queued" || j.state === "running"))) void syncJob();
  }, [queue, jobId]);
  const server = useServerLoad();
  // /api/load 附带返回的服务状态写入共享状态后，服务状态自身的轮询退为每五分钟一次。
  useEffect(() => {
    if (server?.server) noteServer(server.server);
  }, [server?.server?.boot, server?.server?.ui, server?.server?.restart, server?.server?.notice, server?.server?.account]);
  const setQueueSwitches = useResults((s) => s.setQueueSwitches);
  const setMaxFrames = useResults((s) => s.setMaxFrames);
  const setStorage = useResults((s) => s.setStorage);
  const switches = queue?.switches ?? server?.switches;
  useEffect(() => {
    if (switches) setQueueSwitches(switches);
  }, [switches?.gpu, switches?.compute]); // eslint-disable-line react-hooks/exhaustive-deps
  // 单次提交允许的最大帧数（由管理员在「设置」中配置）：「计算范围」与每次提交据此在操作前进行限制
  const maxFrames = queue?.max_frames ?? server?.max_frames;
  useEffect(() => {
    if (maxFrames) setMaxFrames(maxFrames);
  }, [maxFrames]); // eslint-disable-line react-hooks/exhaustive-deps
  // 存储占用随队列数据返回（lab2shot/server/farm.py queue），无需额外请求。占满时所有「计算」入口（cookHold）
  // 变为不可用并说明原因（state/quota.ts）；服务器端同样进行限制。
  const storage = queue?.storage ?? server?.storage ?? null;
  useEffect(() => {
    setStorage(storage);
  }, [storage?.total, storage?.limit]); // eslint-disable-line react-hooks/exhaustive-deps
  // 结果全在缓存里的计算不到一秒就算完：任务状态和角标都等稳定 400 毫秒再显示，免得顶栏闪一下（platform/settled.ts）
  const busy = useSettled(queue?.jobs.filter((j) => j.state === "queued" || j.state === "running").length ?? 0, 400);
  const hasJob = useSettled(!!job, 400);
  // 点了「计算」、任务还没进队列（先上传素材、再提交）：同样在「队列」按钮上说，并给「取消」（graph/actions.ts
  // cancelSubmit）。同样稳定 400 毫秒再显示：素材已在服务器上时提交不到一秒
  const submitting = useSettled(!!useResults((s) => s.submitting), 400);
  const sending = useSubmitUpload();
  // 提交路上连不上服务器：写在等第几次、几秒后重连（graph/submitLine.ts），聚焦页的顶栏同样用这一句
  const line = useSubmitLine((s) => s.text);
  const submitText = !submitting || job ? "" : line ?? (sending === null ? t("ui.editor.submitting") : t("ui.editor.uploading", { pct: sending }));
  // 这张节点图的计算在「队列」按钮上显示状态（不显示任何预计时间：按以往用时推算的时间不准）
  const jobText = !job || !hasJob ? submitText : job.stopping ? t("ui.editor.stopping") : job.position == null ? t("ui.editor.cooking") : waitText(job);
  const viewer = useReadOnly(); // tabs.ts：这张图正在另一个标签页中编辑
  const viewerTip = readOnlyWhy();
  // 具有模板管理权限的账号可在「文件」菜单中保存预设模板
  const applies = useSession((st) => st.state)?.applies;
  const preset = shown(applies, "templates.create"); // 该登录是否管理模板：由服务器判定
  const saved = dirty ? t("ui.editor.unsaved") : file ? t("ui.editor.saved") : t("ui.editor.no_file");
  // 应用模式（editor/AppMode.tsx）：只用模板、不做模板——「模板」入口和存成模板的两项收起来
  const appMode = useAppMode((s) => s.mode === "app");
  // 聚焦模式（DCC 插件的内嵌窗口）：顶栏只剩这一步要的（FocusBar）；上面的轮询照旧（存储占用、开关、任务状态）
  const focus = useAppMode((s) => s.mode === "focus");
  // the bar never squeezes its cells (ui/fitBar.ts): the traffic and the server's load step out when there is no room
  // for them and the document's whole name
  const bar = useRef<HTMLDivElement>(null);
  useFitBar(bar, [focus, appMode, pick(meta.name)]);
  if (focus) return <FocusBar jobText={jobText} submitText={submitText} errors={errors} />;

  return (
    <div className="topbar" ref={bar}>
      <div className="brand" aria-label="Lab2Shot">
        <BrandMark />
      </div>
      <FileMenu onNew={onNew} onOpen={onOpen} onSave={onSave} viewer={viewer} viewerTip={viewerTip} preset={preset} appMode={appMode} />
      <UndoRedo />
      <span className="bar-sep" />
      <div className="doc-title">
        <span className="name" data-user-data data-fit-keep {...tipAttrs(tipOf("truncated", pick(meta.name)))}>{pick(meta.name) || t("ui.common.unnamed")}</span>
        <span className="doc-state" data-user-data {...tipAttrs(tipOf("value", `${saved}\n${file ? `${file.name}\n${file.handle ? t("ui.editor.file_writable") : t("ui.editor.file_download")}` : t("ui.editor.autosaved")}`))}>
          {file ? file.name : saved}
        </span>
        {dirty && <span className="dirty-dot" role="img" aria-label={t("ui.editor.unsaved")} />}
      </div>
      <div className="bar-actions">
        {/* 节点模式 / 应用模式：「模板」左边，两个按钮同一样式，被同一圈彩虹亮边框住，看得出是二选一 */}
        <ModeSwitch />
        {/* 应用模式也要能从模板新建：用的人就是靠它选一个做好的模板当应用；应用模式只是不做模板（存模板的菜单项收起） */}
        <Button entry layout="bar-templates" onClick={() => setTemplatesOpen(true)}>
          <IconGrid size={13} />
          <span className="bar-label">{t("ui.editor.templates")}</span>
          {templates > 0 && <span className="bar-count tnum">{templates}</span>}
        </Button>
        <span className="bar-sep" />
        {/* 服务器忙闲，然后是文字按钮「队列 / 日志 / 提交反馈 / 更新说明」（「队列」「日志」带计数） */}
        <LoadPill load={server} />
        <TransferRate />
        <Button tone="ghost" on={!!jobText} onClick={() => setQueueOpen(true)}>
          {jobText ? t("ui.editor.queue_job", { job: jobText }) : t("ui.editor.queue")}
          {/* the count only while the words do not already say how this graph's job stands (「队列 · 计算中」 needs no 1) */}
          {busy > 0 && !jobText && <span className="q-badge bar-count-badge tnum">{busy}</span>}
        </Button>
        <CancelButtons jobText={jobText} submitText={submitText} />
        <ReadingNow />
        <Button tone="ghost" onClick={() => setLogOpen(true)}>
          {t("ui.editor.log")}
          {errors > 0 && <span className="q-badge log-badge bar-count-badge tnum">{errors}</span>}
        </Button>
        <FeedbackButton tone="ghost" context={editorContext} />
        <ReleasesButton />
        <PluginsButton />
        {/* 这里没有帮助入口：没有帮助站点；扩展在管理页安装（admin/Extensions.tsx）。
            节点图角落的「操作说明」「?」是另一回事（editor/GraphHelp.tsx）。 */}
        <span className="bar-sep" />
        <AccountChip />
        <div className="bar-cook">
          <StorageNotice onOpen={() => setQueueOpen(true)} />
        </div>
      </div>
      {queueOpen && <QueueSheet data={queue} onRefresh={asked.reload} onClose={() => setQueueOpen(false)} />}
    </div>
  );
}

/** 聚焦模式（editor/AppMode.tsx）的顶栏：标志、聚焦的节点和任务名，任务状态（附「取消」），「日志」（出错时一切消息都在
 * 那里，没有别的地方能看），「计算」「完成」。没有文件菜单、模板、模式切换、反馈、账号这些入口：这一页只做 DCC 让做的一步。 */
function FocusBar({ jobText, submitText, errors }: { jobText: string; submitText: string; errors: number }) {
  const meta = useCookInputs((s) => s.meta);
  const node = useAppMode((s) => s.focus?.node ?? "");
  const label = useCookInputs((s) => nodeRef(node, s.nodes[node]?.typeId));
  const version = useCookInputs((s) => s.version);
  const port = useLook((s) => s.displayPort);
  const target = useAppMode((s) => (s.focus && !s.focus.delivers ? s.focus.targets[0] ?? s.focus.node : ""));
  // the same latch and the same words as every other 「计算」 (graph/actions.ts cookHold, cookHoldWhy): the server's own
  // reason when it says the focused node cannot be cooked now
  const at = target ? { node: target, version, port } : undefined;
  const hold = useResults((s) => cookHold(s, at));
  const why = useResults((s) => cookHoldWhy(s, cookHold(s, at), at));
  const setLogOpen = useViewer((s) => s.setLogOpen);
  const [asking, setAsking] = useState(false);
  const done = () => (focusUncomputed() ? setAsking(true) : focusFinish());
  return (
    <div className="topbar focus-bar">
      <div className="brand" aria-label="Lab2Shot">
        <BrandMark />
      </div>
      <div className="doc-title">
        <span className="name" data-user-data {...tipAttrs(tipOf("truncated", label))}>{label}</span>
        <span className="doc-state" data-user-data {...tipAttrs(tipOf("truncated", pick(meta.name)))}>{pick(meta.name) || t("ui.common.unnamed")}</span>
      </div>
      <div className="bar-actions">
        {jobText && <span className="focus-state">{jobText}</span>}
        <CancelButtons jobText={jobText} submitText={submitText} />
        <Button tone="ghost" onClick={() => setLogOpen(true)}>
          {t("ui.editor.log")}
          {errors > 0 && <span className="q-badge log-badge bar-count-badge tnum">{errors}</span>}
        </Button>
        <span className="bar-sep" />
        <Button tone="primary" disabled={hold !== null} tip={tipOf("disabled", why)} aria-label={t("ui.editor.cook")} onClick={focusCompute}>
          {t("ui.editor.cook")}
        </Button>
        <Button tip={tipOf("consequence", t("ui.editor.focus_done_tip"))} aria-label={t("ui.common.done")} onClick={done}>
          {t("ui.common.done")}
        </Button>
      </div>
      {asking && (
        <Sheet title={t("ui.editor.uncooked_title")} width={480} onClose={() => setAsking(false)}>
          <p className="tpl-desc" style={{ fontSize: 13 }}>
            {t("ui.editor.uncooked_text")}
          </p>
          <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
            <Button tone="ghost" onClick={() => setAsking(false)}>
              {t("ui.editor.go_cook")}
            </Button>
            <Button tip={tipOf("consequence", t("ui.editor.finish_uncooked_tip"))} onClick={() => (setAsking(false), focusFinish())}>
              {t("ui.editor.finish_uncooked")}
            </Button>
          </div>
        </Sheet>
      )}
    </div>
  );
}

/** 顶栏的「取消」：有任务在算时停下它，点了「计算」还在上传 / 提交时停下这次提交。普通顶栏和聚焦栏共用。 */
function CancelButtons({ jobText, submitText }: { jobText: string; submitText: string }) {
  const job = useResults((s) => s.job);
  if (job && jobText)
    return (
      <Button tip={tipOf("consequence", t("ui.editor.cancel_job_tip"))} tone="ghost" disabled={job.stopping} onClick={() => void cancelCook()}>
        {t("ui.common.cancel")}
      </Button>
    );
  if (!job && submitText)
    return (
      <Button tip={tipOf("consequence", t("ui.editor.cancel_submit_tip"))} tone="ghost" onClick={cancelSubmit}>
        {t("ui.common.cancel")}
      </Button>
    );
  return null;
}

/** 点了「完成」而窗口没关掉（在系统浏览器里打开的页面，脚本关不了它）：盖一层说可以回 DCC 了。 */
export function FocusDone() {
  const done = useAppMode((s) => !!s.focus?.done);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    if (!done) return setShown(false);
    const t = setTimeout(() => setShown(true), 400); // 内嵌窗口这会儿已经被插件关了
    return () => clearTimeout(t);
  }, [done]);
  if (!shown) return null;
  return (
    <div className="focus-done" role="status">
      <div className="focus-done-card glass">
        <h1>{t("ui.editor.back_to_dcc")}</h1>
        <p>{t("ui.editor.back_to_dcc_text")}</p>
      </div>
    </div>
  );
}

/** 「文件」：新建、打开与保存节点图文件，一个菜单代替三个按钮。「保存到我的模板」则把同一张图存到服务器、记在本账号名下：
 * 换一台机器、同一登录，它仍在。 */
function FileMenu({ onNew, onOpen, onSave, viewer, viewerTip, preset, appMode }: { onNew: () => void; onOpen: () => void; onSave: (saveAs: boolean) => void; viewer: boolean; viewerTip: string; preset: boolean; appMode: boolean }) {
  const [savingPreset, setSavingPreset] = useState(false);
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirs, setDirs] = useState<string[]>([]);
  // 「授权本机素材文件夹」一项始终显示；浏览器不支持文件夹对话框（Firefox、Safari）时显示为不可用并说明原因。
  useEffect(() => {
    if (!at) return;
    void authorisedDirs().then((all) => setDirs(all.map((one) => one.dir.name)));
  }, [at]);
  // 出现在打开它的按钮下方，无论顶栏把按钮放在哪里（键盘触发的点击没有指针位置）
  const open = (e: React.MouseEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setAt({ x: r.left, y: r.bottom + 4 });
  };
  return (
    <>
      {/* 「文件」是左边唯一的文字按钮，不带图标：旁边的标志已经是 logo */}
      <Button tone="ghost" on={!!at} onClick={(e) => (at ? setAt(null) : open(e))}>
        {t("ui.editor.file")}
      </Button>
      {at && (
        <Menu
          at={at}
          label={t("ui.editor.file")}
          width={230}
          onClose={() => setAt(null)}
          rows={[
            // 新建只在节点模式里有：空图在应用模式下没有可显示的参数
            ...(appMode ? [] : [{ key: "new", label: t("ui.editor.new_graph"), run: onNew } satisfies MenuRow]),
            { key: "open", label: t("ui.common.open"), desc: <Kbd>Ctrl+O</Kbd>, run: onOpen },
            { key: "save", label: t("ui.common.save"), tip: viewer ? tipOf("disabled", viewerTip) : undefined, desc: <Kbd>Ctrl+S</Kbd>, off: viewer, run: () => onSave(false) },
            { key: "saveas", label: t("ui.common.save_as"), tip: viewer ? tipOf("disabled", viewerTip) : undefined, desc: <Kbd>Ctrl+Shift+S</Kbd>, off: viewer, run: () => onSave(true) },
            // 应用模式不存模板（做模板在节点模式里）
            ...(appMode ? [] : [{
              key: "library",
              label: t("ui.editor.save_library"),
              tip: viewer ? tipOf("disabled", viewerTip) : undefined,
              off: viewer,
              run: () => setSaving(true),
            } satisfies MenuRow]),
            // 仅具有模板管理权限的账号可见：填写名称、简介与分类后保存为预设模板
            ...(preset && !appMode
              ? [{
                  key: "preset",
                  label: t("ui.editor.save_preset"),
                  tip: viewer ? tipOf("disabled", viewerTip) : undefined,
                  off: viewer,
                  run: () => setSavingPreset(true),
                } satisfies MenuRow]
              : []),
            {
              // 授权后，视图直接从本机文件夹读取素材的原件（`files/localDirs.ts`）：全精度、无需传输，
              // 刷新页面后授权仍然有效。浏览器不支持文件夹对话框时，该项显示为不可用。
              key: "originals",
              label: t("ui.editor.originals"),
              tip: canReadFolder
                ? tipOf("value", dirs.length ? t("ui.editor.originals_allowed", { dirs }) : "")
                : tipOf("disabled", t("ui.editor.originals_none")),
              desc: dirs.length ? <span className="tnum">{dirs.length}</span> : undefined,
              off: !canReadFolder,
              run: () => void addDir().then((name) => name && setDirs((had) => (had.includes(name) ? had : [...had, name]))),
            },
          ]}
        />
      )}
      {saving && <SaveToLibrarySheet onClose={() => setSaving(false)} />}
      {savingPreset && <SaveToLibrarySheet preset onClose={() => setSavingPreset(false)} />}
    </>
  );
}

/** 节点图的撤销 / 重做，各自说明将要做什么。 */
function UndoRedo() {
  const undoLabel = useViewer((s) => s.undoLabel);
  const redoLabel = useViewer((s) => s.redoLabel);
  const lock = useWriteLock(); // 撤销 / 重做也是写文档：只读时置灰并说明（闸本身在 graph/document.ts undo / redo）
  return (
    <div className="bar-history">
      <IconButton tip={lock ? tipOf("disabled", lock) : undoLabel ? tipOf("shortcut", t("ui.editor.undo_tip", { step: undoLabel() })) : tipOf("disabled", t("ui.editor.undo_none"))} aria-label={t("ui.common.undo")} tone="ghost" disabled={!!lock || !undoLabel} onClick={undo}>
        <IconUndo size={13} />
      </IconButton>
      <IconButton tip={lock ? tipOf("disabled", lock) : redoLabel ? tipOf("shortcut", t("ui.editor.redo_tip", { step: redoLabel() })) : tipOf("disabled", t("ui.editor.redo_none"))} aria-label={t("ui.common.redo")} tone="ghost" disabled={!!lock || !redoLabel} onClick={redo}>
        <IconRedo size={13} />
      </IconButton>
    </div>
  );
}

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v)}%`);

/** 服务器当前状态（/api/load）：负载、两个开关、帧数上限与存储占用。
 *
 * 顶栏只发起一次轮询：每次调用 `usePoll` 都会建立一条独立的轮询，因此该钩子仅在顶栏使用，
 * 状态标签通过参数接收同一份结果。 */
const useServerLoad = () =>
  usePoll(api.load, 5000, {
    slowest: 60_000,
    // 变化判断只看排队与计算位、两个开关、帧数上限、存储占用与服务状态；CPU、内存、显存百分比每秒变化，
    // 若纳入判断，轮询退避将永远不会生效。
    identity: (v) => [v.queue, v.slots, v.switches, v.max_frames, v.storage, v.server],
  }).data;

/** 服务器忙闲，位于「队列」左边：「排队 N · CPU a/b · GPU c/d」，按空闲 / 忙 / 满着色；悬停显示计算位的含义、
 * 机器的 CPU 与内存以及每张显卡。所有人都能看到全部内容，包括显卡。 */
function LoadPill({ load }: { load: ServerLoad | null | undefined }) {
  if (!load) return null;
  const { queue, slots } = load;
  const busy = slots.cpu.busy + slots.gpu.busy;
  const total = slots.cpu.total + slots.gpu.total;
  const state = busy >= total && total > 0 ? "full" : busy > 0 || queue.waiting > 0 ? "busy" : "idle";
  const lines = [
    t("ui.editor.load_waiting", { count: queue.waiting }),
    t("ui.editor.load_running", { count: queue.running }),
    "",
    t("ui.editor.load_slots"),
    t("ui.editor.load_cpu", { busy: slots.cpu.busy, total: slots.cpu.total }),
    t("ui.editor.load_gpu", { busy: slots.gpu.busy, total: slots.gpu.total }),
    "",
    t("ui.editor.load_server", { cpu: pct(load.cpu_pct), ram: pct(load.ram_pct) }),
    ...(load.cards ?? []).map((c, i) => t("ui.editor.load_card", { n: i + 1, state: t(c.busy ? "ui.editor.load_card_busy" : "ui.editor.load_card_idle"), mem: pct(c.mem_pct) })),
  ];
  return (
    <span className="load-pill tnum" data-state={state} data-fit="1" {...tipAttrs(tipOf("value", lines.join("\n")))}>
      {t("ui.editor.load_pill", { waiting: queue.waiting, cpu: slots.cpu.busy, cpus: slots.cpu.total, gpu: slots.gpu.busy, gpus: slots.gpu.total })}
    </span>
  );
}

/** 帧范围的一端：自由输入，离开输入框（或按 Enter）时生效。 */
function RangeEnd({ value, label, disabled, bad, onCommit }: { value: string; label: string; disabled: boolean; bad: boolean; onCommit: (text: string) => void }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <input
      className={`field num tl-field${bad ? " bad" : ""}`}
      value={text}
      aria-label={label}
      placeholder="—"
      inputMode="numeric"
      spellCheck={false}
      disabled={disabled}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => text !== value && onCommit(text)}
      onKeyDown={(e) => e.key === "Enter" && e.currentTarget.blur()}
    />
  );
}

/** 计算所取的帧（使用镜头自身的帧号）：使用者收窄之前为输入的完整范围，收窄之后只计算这些帧（上游也一样），
 * 每个范围各自缓存；按输入的原样随图保存（提交的是它与输入的交集：graph/nodes.ts cookSpan）。它位于时间线上
 * （帧所在之处），挨着外观相似的播放范围，并与时间线其余部分一样不显示提示：写错或与输入完全不相交的范围
 * 在此标红，并在计算拒绝它时说明。 */
export function CookRange() {
  const cookRange = useCookInputs((s) => s.cookRange);
  // 显示节点对上当前编辑的 plan 的素材范围（state/results.ts planOf）：改过、新回复到之前不知道
  const version = useCookInputs((s) => s.version);
  const display = useLook((s) => s.displayId);
  const port = useLook((s) => s.displayPort);
  const full = useResults((s) => planOf(s, { node: display, port, version })?.range ?? null);
  const problem = rangeProblemNow() ?? frameLimitProblemNow();
  // 有任务在算、或点了「计算」正在上传 / 提交（提交的是点下去那一刻的范围）：锁住
  const busy = useResults((s) => !!s.job || !!s.submitting);
  const whole: [string, string] | null = full && [String(full[0]), String(full[1])];
  const shown = cookRange ?? whole ?? ["", ""];
  const off = busy || (!cookRange && !whole); // 没有序列输入就无可收窄
  const commit = (end: 0 | 1, text: string) =>
    setCookRange(end ? [shown[0], text.trim() || whole?.[1] || ""] : [text.trim() || whole?.[0] || "", shown[1]]);
  return (
    <div className={`tl-group cook-range${problem ? " bad" : ""}`}>
      <span className="tl-label">{t("ui.editor.cook")}</span>
      <RangeEnd value={shown[0]} label={t("ui.editor.range_first")} disabled={off} bad={!!problem} onCommit={(t) => commit(0, t)} />
      <span className="tl-dash">–</span>
      <RangeEnd value={shown[1]} label={t("ui.editor.range_last")} disabled={off} bad={!!problem} onCommit={(t) => commit(1, t)} />
      {cookRange && (
        <Button tone="ghost" size="sm" disabled={busy} onClick={() => setCookRange(null)}>
          {t("ui.editor.range_all")}
        </Button>
      )}
    </div>
  );
}

/** 自己的占用到了 80%、满了（lab2shot/server/quota.py stage_of，随 /api/load 和队列的轮询来）：顶栏提醒一次，每个阶段一次——
 * 点「知道了」或点它打开队列窗口之后，这个阶段不再提醒；占用降到这个阶段以下再回来，才再提醒。记在这个浏览器里，按账号分开
 * （platform/storage.ts）。满了时「计算」本身也置灰并说明（state/quota.ts），这里只是提前说一声、给个去腾空间的入口。 */
function StorageNotice({ onOpen }: { onOpen: () => void }) {
  const storage = useResults((s) => s.storage);
  const user = useSession((st) => st.state?.user?.id ?? 0);
  const key = `storageSeen.${user}`;
  const stage = storage?.stage ?? 0;
  const [seen, setSeen] = useState(() => readPref<number>(key, 0));
  useEffect(() => setSeen(readPref<number>(key, 0)), [key]);
  useEffect(() => {
    if (storage && stage < seen) {  // 降回这个阶段以下：下次到了再提醒
      writePref(key, stage);
      setSeen(stage);
    }
  }, [storage, stage, seen, key]);
  if (!storage || !storage.limit || stage === 0 || stage <= seen) return null;
  const done = () => {
    writePref(key, stage);
    setSeen(stage);
  };
  const said = { used: sizeText(storage.total), limit: sizeText(storage.limit) };
  const pct = Math.floor((storage.total / storage.limit) * 100);
  const m = stage >= 2 ? msg("N-STORAGE-FULLBAR", said) : msg("N-STORAGE-WARN", { ...said, pct });
  // 顶栏挤：一行只写几个字（占了多少、点开去腾），整句是它的提示
  return (
    <span className="bar-storage" role="status">
      <button type="button" className="bar-storage-text" {...tipAttrs(tipOf("truncated", m.text))} data-code={m.code} onClick={() => (done(), onOpen())}>
        {stage >= 2 ? t("ui.editor.storage_full") : t("ui.editor.storage_pct", { pct })}
      </button>
      <IconButton tone="ghost" aria-label={t("ui.editor.got_it")} onClick={done}>
        <IconClose />
      </IconButton>
    </span>
  );
}

/** 点了「计算」、素材正在上传时，这张图这一次要传的素材传到了百分之几（各任务已传字节之和 / 总字节）；没有在传的为
 * null（只剩提交那一步）。读上传任务已有的字节进度（transfer/uploads.ts），不另算。 */
function useSubmitUpload(): number | null {
  const graph = useCookInputs((s) => s.graphId);
  return useUploads((s) => {
    const going = Object.values(s.tasks).filter((t) => t.graphId === graph && (t.state === "sending" || t.state === "waiting" || t.state === "finishing"));
    if (!going.length) return null;
    const bytes = going.reduce((a, t) => a + t.bytes, 0);
    return Math.floor((going.reduce((a, t) => a + t.sent, 0) / Math.max(bytes, 1)) * 100);
  });
}

/** 页面每秒收发的字节（含浏览器 HTTP 缓存直接给的：页面分不出，见 platform/traffic.ts），常驻在「排队 … GPU」胶囊旁边（先于「队列」）：↑ 上行、↓ 下行，每秒刷新，没有传输时
 * 照样在（状态轮询、推送的零星字节也算），单位按大小自动取 B/s、KB/s、MB/s。字节由网络经过的几处统一记
 * （platform/traffic.ts：fetch、推送 SSE、上传素材），速度用 transfer/rate.ts 唯一的算法。推送通道断开（platform/events.ts
 * pushLost）或服务器没有应答（state/server.ts down）时这一块变成错误色写「已断开」，接上后恢复。 */
function TransferRate() {
  const serverDown = useServer().down;
  const [shown, setShown] = useState<{ up: [string, string]; down: [string, string]; lost: boolean; paused: number }>({ up: rateParts(0), down: rateParts(0), lost: false, paused: 0 });
  useEffect(() => {
    const up = new Rate();
    const down = new Rate();
    const tick = () => {
      const t = trafficTotals();
      // 服务器说太频繁（429）时整站停着（platform/http.ts quietFor）：这一块倒计时，停完自动消失
      const next = { up: rateParts(up.add(t.up)), down: rateParts(down.add(t.down)), lost: pushLost(), paused: Math.ceil(quietFor() / 1000) };
      setShown((was) => (was.up.join() === next.up.join() && was.down.join() === next.down.join() && was.lost === next.lost && was.paused === next.paused ? was : next));
    };
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, []);
  const lost = shown.lost || serverDown;
  const paused = !lost && shown.paused > 0;
  return (
    <span className={`bar-rate tnum${lost || paused ? " lost" : ""}`} data-fit="2"
      {...tipAttrs(lost ? tipOf("error", t("ui.editor.rate_lost_tip")) : paused ? tipOf("error", t("ui.editor.rate_paused_tip")) : undefined)}>
      {lost ? <span className="rate-lost">{t("ui.editor.rate_lost")}</span> : paused ? <span className="rate-lost">{textOf(msg("N-ACCESS-PAUSED", { seconds: shown.paused }))}</span> : (
        <>
          <span className="rate-dir">↑</span><span className="rate-num">{shown.up[0]}</span><span className="rate-unit">{shown.up[1]}</span>
          <span className="rate-dir">↓</span><span className="rate-num">{shown.down[0]}</span><span className="rate-unit">{shown.down[1]}</span>
        </>
      )}
    </span>
  );
}

/** 一个文件正在后台读内容（导入节点列清单时自动读的，没点「计算」，所以没有任务、队列里取消不了）：顶栏说一句和用了
 * 多久，旁边「停止」。超过上限服务器自己停下（engine/external.py ask_worker），节点上说原因和怎么办。 */
const NO_READINGS: Reading[] = []; // one empty list: a selector returning a new [] each time would re-render for ever

function ReadingNow() {
  const readings = useResults((s) => s.reply?.readings ?? NO_READINGS);
  if (!readings.length) return null;
  const r = readings[0];
  const several = readings.length > 1;
  return (
    <span className="bar-reading" role="status">
      <span data-user-data {...tipAttrs(tipOf("truncated", `${r.label} · ${r.file}`))}>{several ? t("ui.editor.reading_many", { file: r.file || r.label, count: readings.length, seconds: r.seconds }) : t("ui.editor.reading", { file: r.file || r.label, seconds: r.seconds })}</span>
      <Button tone="ghost" tip={tipOf("consequence", t("ui.editor.reading_stop_tip", { minutes: Math.round(r.limit / 60) }))}
        onClick={() => void api.stopReadings().then(() => askStatus())}>
        {t("ui.editor.stop")}
      </Button>
    </span>
  );
}
