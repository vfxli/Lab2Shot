import "./editor.css";
import { OpenSheet } from "./ParamControls";
import { useCallback, useEffect, useRef, useState } from "react";
import { ReactFlowProvider } from "@xyflow/react";
import { api, type GraphJSON } from "../api";
import { cook, deliverAll, loadGraph, openHeld, redo, resumeJobs, undo } from "../graph/actions";
import { jump, step } from "../graph/playback";
import { catalogReady, setCatalog } from "../state/catalog";
import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { readOnly, useReadOnly } from "../state/cookInputs";
import { usePreferences } from "../state/preferences";
import { startUploads } from "../graph/apply";
import { NodeEditor } from "./NodeEditor";
import { ColorspaceFills } from "./ColorspaceFill";
import { NodeMenu } from "./NodeMenu";
import { ParamPanel } from "./ParamPanel";
import { Viewer } from "./Viewer";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { OPEN_GRAPH, TabBanner, TemplatesSheet, TopBar, UnsavedSheet } from "./Chrome";
import { msg, reasonOf, say } from "../state/say";
import { LogSheet } from "./LogSheet";
import { Welcome } from "./Welcome";
import { lastWorking, startAutosave } from "./autosave";
import { startTabSync } from "./tabs";
import { openGraphFile, saveGraphFile, type GraphFile } from "../graph/graphFile";
import { LOGGED_IN } from "../platform/http";
import { useShortcut } from "../platform/keys";
import { signedIn } from "../state/session";
import { followDrag } from "../platform/drag";
import { defaultName } from "../graph/naming";
import { graphHidden, useAppMode } from "./AppMode";
import { FocusDone } from "./Chrome";
import { focusCompute, openAddressedJob, restoresWorkingCopy, useFocusDocument } from "./focusJob";
import { t } from "../i18n/t";
import { useLang, type Lang } from "../i18n/lang";
import { tipAttrs, tipOf } from "../platform/tips";

/** 编辑器页面的根：本模块拥有工作区版面（视图、节点图、参数面板三格与两条分割线）以及页面级快捷键、
 * 打开 / 保存图的入口。工作区版面：
 *
 *     .workspace                先纵向分为两部分（列：左栏 | 1px 竖分割线 | 参数面板）
 *     ├── .wk-left              左栏上下分为两部分（行：视图 | 1px 横分割线 | 节点图）
 *     ├── .vsplitter            竖分割线，占满整个高度
 *     └── .panel (ParamPanel)   右侧，占满整个高度
 *
 * 不变量（CSS 是其直接实现；修改前须先阅读以下四条）：
 *   1. 参数面板宽度固定（`--insp-w`，默认 `INSPECTOR_WIDTH`）：在 `flex/grid` 中为 `0 0 auto`，
 *      宽度仅由使用者拖动 `.vsplitter` 改变，不随当前节点的参数数量撑开（见下方注释）；
 *   2. 左栏为弹性部分：`minmax(0, 1fr)`，无论被压缩到多窄都不得将 `.workspace` 撑出窗口
 *      （因此使用 `minmax(0, 1fr)` 而非 `1fr`：后者的最小值为内容最小宽度，会随内容变化）；
 *   3. 两条分割线各 1 px，为固定轨道，不参与弹性分配；左栏上半部（视图）为 `var(--split)` 百分比，
 *      下半部（节点图）占据剩余的 `minmax(0, 1fr)`；
 *   4. 滚动：视图与节点图自身不滚动（`.panel { overflow: hidden }`），参数面板在其内部滚动
 *      （`.inspector .insp-body`）。工作区本身不滚动，页面不得宽于窗口。
 *
 * 画布是布局中的固定区域，不得被挤压：此处只划分格子，不得出现针对画布被挤压的补偿代码。
 * 视图格的宽高仅由工作区尺寸、面板宽度与 `--split` 决定（宽 = 工作区宽 − 面板宽 − 1，
 * 高 = 工作区高 × `--split`），因此切换显示节点时画面的屏幕位置与缩放不变。
 *
 * 不设单独的「消息」栏：所有消息统一进入日志（顶栏的日志图标，出错时显示红色计数），另加节点自身的两处：
 * 底行的「注意 / 提醒」角标，以及右下角「数据信息」中的「提醒」组。不得在视图或节点图上添加第二个通知控件。 */

/** 参数面板的宽度：固定默认值，不随选中节点变化。
 *
 * 若面板宽度取「完整显示当前选中节点全部参数所需的宽度」（state/viewer.ts inspectorFit），
 * 每切换一个节点面板宽度就会变化，舞台随之变窄或变宽，画面被重新缩放；控件位置不得跳动，
 * 切换显示节点时 2D 的缩放与平移也不得改变。
 *
 * 因此与 Houdini / Nuke 相同：面板宽度固定，由使用者拖动分割线决定（拖动后记入首选项并持续使用）；
 * 双击分割线表示按当前节点撑开一次，这是使用者主动触发的操作，而非自动行为。`fit` 仅用于此。 */
// px：未拖动时的宽度，按界面语言（英文字比中文长：标签列 ui/labelRow.css 在英文下也放宽）。所有节点的参数在此宽度下
// 都必须能完整排列（界面文字不得换行或截断）
const INSPECTOR_WIDTH: Record<Lang, number> = { zh: 440, en: 520 };
const INSPECTOR_MIN = 400; // px：双击「按参数撑开」时的最小宽度

export default function App() {
  const [owner, setOwner] = useState<number | null>(null); // 本页保存其工作副本的账号（autosave.ts）；图载入后才设置
  const ready = owner !== null;
  const displayId = useLook((s) => s.displayId);
  const selectedId = useViewer((s) => s.selectedId);
  const viewer = useReadOnly();
  const [error, setError] = useState<string | null>(null);
  const split = usePreferences((s) => s.split);
  const setSplit = usePreferences((s) => s.setSplit);
  const [dragging, setDragging] = useState(false);
  const inspector = usePreferences((s) => s.inspectorWidth);
  const setInspector = usePreferences((s) => s.setInspectorWidth);
  const [resizing, setResizing] = useState(false);
  const fit = useViewer((s) => s.inspectorFit);
  const left = useRef<HTMLDivElement>(null); // 左栏：上下分割在其内部测量
  const workspace = useRef<HTMLDivElement>(null); // 整个工作区：左右分割在其内部测量
  // 应用模式（editor/AppMode.tsx）：左栏只有视图（节点图收起），右栏参数面板只有模板的参数界面树（「计算」「下载」
  // 也是树里公开的按钮参数，没有写死的块）。聚焦模式是它的变体：同样收起节点图，参数面板是聚焦的那个节点的
  const appMode = useAppMode((s) => graphHidden(s.mode));
  const focus = useAppMode((s) => s.mode === "focus");
  useFocusDocument();

  const openFailed = (e: unknown) => say(msg("E-GRAPH-OPENFAILED", { reason: reasonOf(e) }));
  const saveFailed = (e: unknown) => say(msg("E-GRAPH-SAVEFAILED", { reason: reasonOf(e) }));

  // 在有未保存修改时打开另一张图，先询问（保存 / 不保存 / 取消）；freshId：模板打开时取新 id，
  // 不与模板文件里写死的 id 共用
  const [pending, setPending] = useState<{ g: GraphJSON; file: GraphFile | null; freshId: boolean } | null>(null);
  // 有任务在算 / 在提交时不打开别的图（graph/actions.ts openHeld，与「计算」置灰同一判定）：所有打开入口都经过这里
  // 目录（节点类型）到了才读图（state/catalog.ts catalogReady）：登录后马上点模板 / 打开文件，目录还在路上时，
  // 图会按「没有任何节点类型」读进来——节点全成「未知节点」、参数界面是空的，目录到了也不会再读一遍
  const open = useCallback(
    (g: GraphJSON, file: GraphFile | null, freshId = false) =>
      catalogReady().then(() =>
        openHeld() ? undefined : useViewer.getState().dirty ? setPending({ g, file, freshId }) : loadGraph(g, file, false, undefined, { freshId })),
    [],
  );
  const resolvePending = async (choice: "save" | "discard" | "cancel") => {
    const next = pending;
    setPending(null);
    if (!next || choice === "cancel") return;
    if (choice === "save" && !(await saveGraphFile().catch((e) => (saveFailed(e), false)))) return;
    if (openHeld()) return; // 问「存不存」的这会儿点了「计算」
    loadGraph(next.g, next.file, false, undefined, { freshId: next.freshId });
  };
  const openFile = useCallback(() => {
    if (openHeld()) return; // 在选文件之前拦下：不能让人选完文件才被告知打不开
    // catch 而不是 then 的第二个参数：打开这一步（open → loadGraph）里抛的错也要说出来，不能无声无息
    openGraphFile().then((r) => r && open(r.graph, r.file)).catch(openFailed);
  }, [open]);
  // 从队列重新载入的任务：其图作为新文档打开，有未保存修改时必先询问
  useEffect(() => {
    const onOpen = (e: Event) => void open((e as CustomEvent<GraphJSON>).detail, null).catch(openFailed);
    window.addEventListener(OPEN_GRAPH, onOpen);
    return () => window.removeEventListener(OPEN_GRAPH, onOpen);
  }, [open]);
  const save = useCallback((saveAs: boolean) => void saveGraphFile(saveAs).catch(saveFailed), []); // eslint-disable-line react-hooks/exhaustive-deps
  // 新建节点图：一张只有输出节点的空图，按新文档打开（有未保存修改时同样先问）
  const newGraph = useCallback(() => {
    // meta 不写名字：打开时按「未命名」填（graph/document.ts filledIn）
    const blank = { schema: "lab2shot.graph/1", meta: {}, nodes: [{ id: defaultName("output", () => false), type: "output", ui: { x: 0, y: 0 } }], edges: [] } as unknown as GraphJSON;
    void open(blank, null, true).catch(openFailed);
  }, [open]);

  useEffect(() => {
    (async () => {
      try {
        const [catalog, me] = await Promise.all([api.catalog(), signedIn()]);
        // 地址要打开一个这一页还没打开过的任务（#job=…，editor/focusJob.ts）时不恢复工作副本，直接打开任务
        const saved = restoresWorkingCopy() ? lastWorking(me.id) : null;
        setCatalog(catalog);
        // 该账号在本机上次的工作状态（刷新后为本标签页自己的）原样恢复（未保存的修改仍标记为未保存）；
        // 首次打开为空图，上面覆盖欢迎页（模板、节点、图文件）
        if (saved) {
          loadGraph(saved.graph, saved.file, saved.dirty, saved.id);
          if (saved.dirty) say(msg("N-GRAPH-RESTORED"));
        }
        await openAddressedJob(saved?.graph ?? null);
        setOwner(me.id);
      } catch (e) {
        setError(String((e as Error).message));
      }
    })();
  }, []);

  // 图显示出来之后：每次修改都保存在本浏览器中（刷新不丢失），该图仍在排队或计算的任务重新跟踪，
  // 该图的「输出」打包好的结果提供下载
  useEffect(() => {
    if (owner === null) return;
    void resumeJobs();
    // 在页面上重新登录（任务中途登录过期，由登录门自身提示）：仍在排队或计算的任务重新跟踪；
    // 任务本身始终在服务器上（graph/follow.ts onlogin）。
    const again = () => void resumeJobs();
    window.addEventListener(LOGGED_IN, again);
    startUploads(); // 已完成的上传写入其参数；被刷新打断的上传处于暂停状态
    const stopAutosave = startAutosave(owner);
    const stopTabSync = startTabSync();
    return () => {
      window.removeEventListener(LOGGED_IN, again);
      stopAutosave();
      stopTabSync();
    };
  }, [owner]);

  // 从扩展包页面（另一个标签页）回来：已安装扩展的节点立即可用
  useEffect(() => {
    const onFocus = () => api.catalog().then(setCatalog, () => undefined);
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  // 页面快捷键（platform/keys.ts）：空格播放，← → 逐帧，↑ ↓ 跳到播放范围的起点 / 终点（时间线的播放范围，与
  // Houdini、Nuke 相同），ctrl+s 保存（+shift：另存为），ctrl+o 打开，ctrl+z 撤销，ctrl+shift+z / ctrl+y 重做（在文本
  // 输入框中这两者归输入框自己），ctrl+enter「计算」当前显示的节点，ctrl+shift+enter「提交」（所有「输出」）
  useShortcut({
    keys: ["mod+s", "mod+shift+s", "mod+o"],
    inText: true,
    run: (e) => {
      if (useAppMode.getState().mode === "focus") return; // 聚焦页只有「计算」「完成」：不打开别的图、不存文件
      // 「打开」替换整张图，只读查看者使用也无妨；「保存」写不出任何新内容（不可编辑），但同样拒绝，
      // 与按钮的理由相同：比一次什么也没保存的保存更容易解释。
      if (e.key.toLowerCase() === "o") openFile();
      else if (!readOnly()) save(e.shiftKey);
      else say(msg("B-GRAPH-OTHERTAB"));
    },
  });
  useShortcut({
    keys: ["mod+z", "mod+shift+z", "mod+y"],
    run: (e) => {
      if (e.key.toLowerCase() === "y" || e.shiftKey) redo(); // the read-only gate is in undo / redo themselves
      else undo();
    },
  });
  useShortcut({
    keys: ["mod+enter", "mod+shift+enter"],
    inText: true,
    run: (e) => {
      const disp = useLook.getState().displayId;
      if (useAppMode.getState().mode === "focus") focusCompute(); // 和聚焦页的「计算」同一件事
      else if (e.shiftKey) void deliverAll();
      else if (disp) void cook(disp);
      else say(msg("B-COOK-NODISPLAY"));
    },
  });
  useShortcut({
    keys: ["space", "arrowleft", "arrowright", "arrowup", "arrowdown"],
    run: (e) => {
      const viewerState = useViewer.getState();
      if (e.key === " ") viewerState.play(viewerState.playDir);
      else if (e.key === "ArrowLeft") step(-1);
      else if (e.key === "ArrowRight") step(1);
      else jump(e.key === "ArrowUp" ? 0 : 1);
    },
  });

  // 横分割线：左栏内部的上下分配（上为视图，下为节点图）。测量的是左栏自身的高度，而非整页高度
  useEffect(() => {
    if (!dragging) return;
    const move = (e: PointerEvent) => {
      const r = left.current!.getBoundingClientRect();
      setSplit(Math.min(80, Math.max(22, ((e.clientY - r.top) / r.height) * 100)));
    };
    return followDrag(move, () => setDragging(false));
  }, [dragging, setSplit]);

  // 竖分割线：整个工作区的左右分配（左栏 | 参数面板）。拖动设置宽度，双击按当前节点的参数撑开一次
  useEffect(() => {
    if (!resizing) return;
    const move = (e: PointerEvent) => {
      const r = workspace.current!.getBoundingClientRect();
      setInspector(Math.round(Math.min(r.width * 0.7, Math.max(280, r.right - e.clientX))));
    };
    return followDrag(move, () => setResizing(false));
  }, [resizing, setInspector]);
  const lang = useLang((s) => s.lang);
  const inspectorWidth = inspector ?? INSPECTOR_WIDTH[lang]; // 固定宽度；`fit` 仅用于双击分割线（见上方注释）

  if (error) {
    return (
      <div className="empty" style={{ height: "100%" }}>
        <div>
          <div style={{ fontSize: 15, color: "var(--text-2)", marginBottom: 6 }}>{t("ui.editor.no_server")}</div>
          <div>{error}</div>
          <div style={{ marginTop: 8 }}>{t("ui.editor.no_server_hint")}</div>
        </div>
      </div>
    );
  }

  return (
    <ReactFlowProvider>
      <div className={`app${viewer ? " viewer" : ""}${appMode ? " app-mode" : ""}${focus ? " focus-mode" : ""}`}>
        <TopBar onNew={newGraph} onOpen={openFile} onSave={save} />
        <TabBanner />
        {/* 整个工作区先纵向分为两部分：左栏（视图 + 节点图） | 竖分割线 | 参数面板（占满整个高度） */}
        <div className="workspace" ref={workspace} style={{ ["--insp-w" as string]: `${inspectorWidth}px` }}>
          {/* 左栏上下分为两部分：视图 | 横分割线 | 节点图 */}
          <div className="wk-left" ref={left} style={{ ["--split" as string]: `${split}%` }}>
            <div className="panel">
              {ready && (
                <ErrorBoundary name={t("ui.editor.part_viewport")} resetKey={displayId}>
                  <Viewer />
                </ErrorBoundary>
              )}
            </div>
            {/* 应用模式不画节点图（NodeEditor 不挂载）；「色彩空间」的自动填写与节点图无关，照常挂着 */}
            {!appMode && <div className={`splitter${dragging ? " active" : ""}`} onPointerDown={() => setDragging(true)} />}
            <div className="panel wk-graph">
              {ready && (
                <ErrorBoundary name={t("ui.editor.part_network")}>
                  {!appMode && <NodeEditor />}
                  {/* 仅在可编辑该图的标签页中填写：其他标签页正在编辑时此处为只读查看者，填写只会修改本地副本，与编辑方不一致 */}
                  {!viewer && <ColorspaceFills />}
                  {!appMode && <Welcome onOpen={openFile} />}
                </ErrorBoundary>
              )}
            </div>
          </div>
          <div
            className={`vsplitter${resizing ? " active" : ""}`}
            onPointerDown={() => setResizing(true)}
            /* 双击表示按当前节点的参数撑开一次（使用者主动触发的操作），而非「恢复自动宽度」 */
            onDoubleClick={() => setInspector(Math.max(INSPECTOR_MIN, Math.round(fit)))}
            {...tipAttrs(tipOf("shortcut", t("ui.editor.inspector_split")))}
          />
          <div className="panel">
            {ready && (
              <ErrorBoundary name={t("ui.editor.part_parameters")} resetKey={selectedId}>
                <ParamPanel />
              </ErrorBoundary>
            )}
          </div>
        </div>
        <NodeMenu />
        {/* 开着的编辑窗画在这里，不挂在参数面板的行上：换选中、读进新版本都不关它（ParamControls.tsx OpenSheet） */}
        <OpenSheet />
        <TemplatesSheet onOpen={(g) => void open(g, null, true).catch(openFailed)} />
        <LogSheet />
        {pending && <UnsavedSheet onChoice={resolvePending} />}
        {focus && <FocusDone />}
      </div>
    </ReactFlowProvider>
  );
}
