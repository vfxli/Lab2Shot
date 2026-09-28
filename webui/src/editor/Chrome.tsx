import { useEffect, useState } from "react";
import { api, type ServerLoad } from "../api";
import { noteServer } from "../state/server";
import { useUnseenErrors } from "../state/log";
import { waitText } from "../graph/nodes";
import { cancelCook, frameLimitProblemNow, redo, rangeProblemNow, setCookRange, undo } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { addDir, authorisedDirs } from "../files/localDirs";
import { canReadFolder } from "../files/handles";
import { BrandMark, IconGrid, IconRedo, IconUndo } from "../ui/icons";
import { useCatalog } from "../state/catalog";
import { useViewer } from "../state/viewer";
import { editorContext, FeedbackButton } from "../ui/Feedback";
import { ReleasesButton } from "../ui/Releases";
import { AccountChip } from "../ui/Account";
import { etaText } from "../ui/Queue";
import { usePoll } from "../platform/poll";
import { Button, IconButton } from "../ui/Button";
import { MessageText } from "../ui/MessageText";
import { Kbd, Menu, type MenuRow } from "../ui/Menu";
import { QueueSheet } from "./ChromeSheets";
import { SaveToLibrarySheet } from "./MyTemplates";
import { shown } from "../api/applies";
import { useSession } from "../state/session";

export { OPEN_GRAPH, TabBanner, UnsavedSheet } from "./ChromeSheets";
export { TemplatesSheet } from "./Templates";

// 空闲时不为队列单独发送请求：队列窗口关闭且当前账号没有任务时，页面只需要两个开关、帧数上限与存储占用，
// 这些随 `/api/load` 一并返回（lab2shot/farm/queue.py Queue.load）。一次往返的开销主要是请求头与会话 cookie，
// 因此合并请求比放慢轮询更有效。
const quietQueue = async () => null;

/** The editor's top bar, in two groups. Left is the document — the mark, the file menu, 撤销 / 重做 and the graph's
 * name with whether it is saved. Right is everything else, 模板 first (the page's one entry, with a rim in the wire
 * colours) and 提交 last (the page's one main button). There is no 计算 button here (a node is cooked from its own
 * menu); 计算范围 sits on the timeline, where the frames are. */
export function TopBar({ onOpen, onSave }: { onOpen: () => void; onSave: (saveAs: boolean) => void }) {
  const meta = useCookInputs((s) => s.meta);
  const file = useViewer((s) => s.file);
  const dirty = useViewer((s) => s.dirty);
  const setTemplatesOpen = useViewer((s) => s.setTemplatesOpen);
  const job = useResults((s) => s.job);
  const [queueOpen, setQueueOpen] = useState(false);
  const setLogOpen = useViewer((s) => s.setLogOpen);
  const errors = useUnseenErrors();
  const templates = useCatalog()?.templates ?? 0; // the count alone: the list itself waits until 模板 is opened
  // open: every 1.5 s with the machine's load; closed: less often while nothing changes, at once when a job of this
  // graph starts or ends.
  // 仅在队列窗口打开或当前账号有运行中的任务时请求完整的队列数据；其余时候所需的三项来自 /api/load。
  // 「排队 N · 计算 x/y」同样来自精简的 /api/load。
  const needQueue = queueOpen || !!job;
  const quiet = queueOpen ? 1500 : 30_000;
  const asked = usePoll(needQueue ? api.queue : quietQueue, needQueue ? (queueOpen ? 1500 : 5000) : null,
                        { key: job?.id ?? "", slowest: quiet });
  const queue = asked.data;
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
  // 单次提交允许的最大帧数（由管理员在「设置」中配置）：「计算范围」与「提交」据此在操作前进行限制
  const maxFrames = queue?.max_frames ?? server?.max_frames;
  useEffect(() => {
    if (maxFrames) setMaxFrames(maxFrames);
  }, [maxFrames]); // eslint-disable-line react-hooks/exhaustive-deps
  // 存储占用随队列数据返回（lab2shot/server/farm.py queue），无需额外请求。占满时「提交」与节点菜单的「计算」
  // 变为不可用并说明原因（state/quota.ts）；服务器端同样进行限制。
  const storage = queue?.storage ?? server?.storage ?? null;
  useEffect(() => {
    setStorage(storage);
  }, [storage?.total, storage?.limit]); // eslint-disable-line react-hooks/exhaustive-deps
  const busy = queue?.jobs.filter((j) => j.state === "queued" || j.state === "running").length ?? 0;
  const mine = job && queue?.jobs.find((j) => j.id === job.id);
  const eta = mine ? etaText(mine, Date.now() / 1000) : "";
  const jobText = !job
    ? ""
    : job.stopping
      ? "正在停止…"
      : [job.position == null ? "计算中" : waitText(job), eta].filter(Boolean).join(" · ");
  const viewer = useViewer((s) => s.role === "viewer"); // tabs.ts: this graph is being edited in another tab
  const viewerTip = "这张节点图在另一个标签页里编辑：这里改不了";
  // 具有模板管理权限的账号可在「文件」菜单中保存预设模板
  const applies = useSession((st) => st.state)?.applies;
  const preset = shown(applies, "templates.create"); // whether this login manages templates: the server says
  const saved = dirty ? "有未保存的修改" : file ? "已保存" : "还没存成文件";

  return (
    <div className="topbar">
      <div className="brand" aria-label="Lab2Shot">
        <BrandMark />
      </div>
      <FileMenu onOpen={onOpen} onSave={onSave} viewer={viewer} viewerTip={viewerTip} preset={preset} />
      <UndoRedo />
      <span className="bar-sep" />
      <div className="doc-title">
        <span className="name" data-user-data data-tip={meta.name}>{meta.name}</span>
        <span className="doc-state" data-user-data data-tip={`${saved}\n${file ? `${file.name}\n${file.handle ? "本机的节点图文件：「保存」写回它" : "本机的节点图文件：这个浏览器不能写回，「保存」会下载一份"}` : "节点图一直自动保存在这个浏览器里，刷新不会丢"}`}>
          {file ? file.name : saved}
        </span>
        {dirty && <span className="dirty-dot" data-tip="有未保存的修改" />}
      </div>
      <div className="bar-actions">
        <Button tip="从内置模板新建一张节点图：现成的流程，选好素材就能算" entry onClick={() => setTemplatesOpen(true)}>
          <IconGrid size={13} />
          <span className="bar-label">模板</span>
          {templates > 0 && <span className="bar-count tnum">{templates}</span>}
        </Button>
        <span className="bar-sep" />
        {/* how busy the server is, then 队列 / 日志 / 提交反馈 / 更新说明 as words (队列 and 日志 with their count) */}
        <LoadPill load={server} />
        <Button tip="队列：自己的任务排第几、大概多久，算完的也在同一张表里，「加载」打开当时的节点图" tone="ghost" onClick={() => setQueueOpen(true)}>
          队列
          {busy > 0 && <span className="q-badge bar-count-badge tnum">{busy}</span>}
        </Button>
        <Button tip="日志：出现过的提示、计算经过和错误；出问题时复制给技术人员" tone="ghost" onClick={() => setLogOpen(true)}>
          日志
          {errors > 0 && <span className="q-badge log-badge bar-count-badge tnum">{errors}</span>}
        </Button>
        <FeedbackButton tone="ghost" context={editorContext} />
        <ReleasesButton />
        {/* No help entry here: there is no help site; extensions are installed on the admin page
            (admin/Extensions.tsx). The 「操作说明」 "?" in the graph's corner is a different thing (editor/GraphHelp.tsx). */}
        <span className="bar-sep" />
        <AccountChip />
        <div className="bar-cook">
          {job ? (
            <>
              <Button tip="打开队列" on layout="job-state" onClick={() => setQueueOpen(true)}>
                {jobText}
              </Button>
              <Button tip="取消这次计算：排队的移出队列，计算中的停下（已经算好的节点留在缓存里）" tone="ghost" disabled={job.stopping} onClick={() => void cancelCook()}>
                取消
              </Button>
            </>
          ) : (
            <>
              <SubmitRefusal />
            </>
          )}
        </div>
      </div>
      {queueOpen && <QueueSheet data={queue} onRefresh={asked.reload} onClose={() => setQueueOpen(false)} />}
    </div>
  );
}

/** 文件: opening and saving the graph file, one menu instead of three buttons. 「保存到我的模板」 saves the same
 * graph on the server under this account instead: another machine, same login, it is still there. */
function FileMenu({ onOpen, onSave, viewer, viewerTip, preset }: { onOpen: () => void; onSave: (saveAs: boolean) => void; viewer: boolean; viewerTip: string; preset: boolean }) {
  const [savingPreset, setSavingPreset] = useState(false);
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const [saving, setSaving] = useState(false);
  const [dirs, setDirs] = useState<string[]>([]);
  // 「授权本机素材文件夹」一项始终显示；浏览器不支持文件夹对话框（Firefox、Safari）时显示为不可用并说明原因。
  useEffect(() => {
    if (!at) return;
    void authorisedDirs().then((all) => setDirs(all.map((one) => one.dir.name)));
  }, [at]);
  // under the button that opened it, wherever the bar put that button (a keyboard click has no pointer position)
  const open = (e: React.MouseEvent<HTMLButtonElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    setAt({ x: r.left, y: r.bottom + 4 });
  };
  return (
    <>
      {/* 文件 is the one word on the left, with no icon — the mark beside it is already the logo */}
      <Button tip="节点图文件：打开、保存、另存为" tone="ghost" on={!!at} onClick={(e) => (at ? setAt(null) : open(e))}>
        文件
      </Button>
      {at && (
        <Menu
          at={at}
          label="文件"
          width={230}
          onClose={() => setAt(null)}
          rows={[
            { key: "open", label: "打开", tip: "打开本机的节点图文件", desc: <Kbd>Ctrl+O</Kbd>, run: onOpen },
            { key: "save", label: "保存", tip: viewer ? viewerTip : "保存到节点图文件；还没存过就先选位置", desc: <Kbd>Ctrl+S</Kbd>, off: viewer, run: () => onSave(false) },
            { key: "saveas", label: "另存为", tip: viewer ? viewerTip : "存成另一个文件", desc: <Kbd>Ctrl+Shift+S</Kbd>, off: viewer, run: () => onSave(true) },
            {
              key: "library",
              label: "保存到我的模板",
              tip: viewer ? viewerTip : "存到服务器、记在这个账号名下：换台电脑登录也在，在「模板」的「我的模板」里打开。只存节点图和参数，素材每次自己选",
              off: viewer,
              run: () => setSaving(true),
            },
            // 仅具有模板管理权限的账号可见：填写名称、简介与分类后保存为预设模板
            ...(preset
              ? [{
                  key: "preset",
                  label: "保存为预设模板",
                  tip: viewer ? viewerTip : "把这张节点图存成一张预设模板卡片：所有人在「模板」里都看得到。只填名字，分类自动判定，之后在「模板」里拖到别的分类",
                  off: viewer,
                  run: () => setSavingPreset(true),
                } satisfies MenuRow]
              : []),
            {
              // 授权后，视图直接从本机文件夹读取素材的原件（`files/localDirs.ts`）：全精度、无需传输，
              // 刷新页面后授权仍然有效。浏览器不支持文件夹对话框时，该项显示为不可用。
              key: "originals",
              label: "授权本机素材文件夹…",
              tip: canReadFolder
                ? `指一个素材所在的文件夹给视图直接读，指过之后视图就从你自己那份素材画——满精度、不下载，刷新之后还认得。${dirs.length ? `\n现在指过的：${dirs.join("、")}\n再指一个会一起用；指过的文件夹里找不到对应的文件时，照旧从服务器取预览` : "\n还没指过：视图现在从服务器取预览"}`
                : "这个浏览器没有文件夹对话框（要 Chrome 或 Edge，地址是 localhost 或 https）：视图从服务器取预览",
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

/** 撤销 / 重做 of the node graph, each saying what it would do. */
function UndoRedo() {
  const undoLabel = useViewer((s) => s.undoLabel);
  const redoLabel = useViewer((s) => s.redoLabel);
  return (
    <div className="bar-history">
      <IconButton tip={undoLabel ? `撤销：${undoLabel}（Ctrl+Z）` : "没有可以撤销的修改"} aria-label="撤销" tone="ghost" disabled={!undoLabel} onClick={undo}>
        <IconUndo size={13} />
      </IconButton>
      <IconButton tip={redoLabel ? `重做：${redoLabel}（Ctrl+Shift+Z 或 Ctrl+Y）` : "没有可以重做的修改"} aria-label="重做" tone="ghost" disabled={!redoLabel} onClick={redo}>
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

/** How busy the server is, left of 队列 — 「排队 N · CPU a/b · GPU c/d」, in the colour of idle / busy / full, with
 * what a 计算位 is, the machine's CPU and memory and every card on hover. Everyone sees all of it, the
 * cards included. */
function LoadPill({ load }: { load: ServerLoad | null | undefined }) {
  if (!load) return null;
  const { queue, slots } = load;
  const busy = slots.cpu.busy + slots.gpu.busy;
  const total = slots.cpu.total + slots.gpu.total;
  const state = busy >= total && total > 0 ? "full" : busy > 0 || queue.waiting > 0 ? "busy" : "idle";
  const lines = [
    `排队：${queue.waiting} 个任务在等（所有人的）`,
    `正在算：${queue.running} 个任务`,
    "",
    "同时能算几个节点：",
    `CPU：${slots.cpu.busy} / ${slots.cpu.total} 在用`,
    `GPU：${slots.gpu.busy} / ${slots.gpu.total} 在用`,
    "",
    `服务器：CPU ${pct(load.cpu_pct)} · 内存 ${pct(load.ram_pct)}`,
    ...(load.cards ?? []).map((c, i) => `显卡 ${i + 1}：${c.busy ? "有任务" : "空闲"} · 显存 ${pct(c.mem_pct)}`),
  ];
  return (
    <span className="load-pill tnum" data-state={state} data-tip={lines.join("\n")}>
      排队 {queue.waiting} · CPU {slots.cpu.busy}/{slots.cpu.total} · GPU {slots.gpu.busy}/{slots.gpu.total}
    </span>
  );
}

/** One end of the frame range: typed freely, taken when the field is left (or Enter). */
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

/** The frames the cook takes (in the shot's own frame numbers): the inputs' whole range until the user narrows it, and
 * then only those frames are cooked, upstream too, each range cached on its own; it is saved with the graph. It sits
 * on the timeline (where the frames are), next to the playback range it looks like, and like the rest of the timeline
 * shows no tips: a range the inputs do not cover is marked red here and said when 提交 refuses it. */
export function CookRange() {
  const cookRange = useCookInputs((s) => s.cookRange);
  const full = useResults((s) => s.plan?.range ?? null);
  const problem = rangeProblemNow() ?? frameLimitProblemNow();
  const busy = useResults((s) => !!s.job);
  const whole: [string, string] | null = full && [String(full[0]), String(full[1])];
  const shown = cookRange ?? whole ?? ["", ""];
  const off = busy || (!cookRange && !whole); // nothing to narrow without a sequence input
  const commit = (end: 0 | 1, text: string) =>
    setCookRange(end ? [shown[0], text.trim() || whole?.[1] || ""] : [text.trim() || whole?.[0] || "", shown[1]]);
  return (
    <div className={`tl-group cook-range${problem ? " bad" : ""}`}>
      <span className="tl-label">计算</span>
      <RangeEnd value={shown[0]} label="计算起始帧" disabled={off} bad={!!problem} onCommit={(t) => commit(0, t)} />
      <span className="tl-dash">–</span>
      <RangeEnd value={shown[1]} label="计算结束帧" disabled={off} bad={!!problem} onCommit={(t) => commit(1, t)} />
      {cookRange && (
        <Button tone="ghost" size="sm" disabled={busy} onClick={() => setCookRange(null)}>
          全部
        </Button>
      )}
    </div>
  );
}

/** Why the server refused this graph (state/results.ts `refused`), in its own words, in the top bar for as long as it
 * stands: a graph it cannot read says so as soon as it is opened, and a refused submission right where it was made,
 * not only as a count on the log. Cut to the bar's width; the whole text is in its tip and in the log. */
function SubmitRefusal() {
  const refused = useResults((s) => s.refused);
  if (!refused) return null;
  return (
    <span className="bar-refused" role="alert" data-tip={refused.text}>
      <MessageText message={refused} />
    </span>
  );
}


/** How long this cook should take, from the records of earlier cooks, per node — inside the 提交 tooltip. */
