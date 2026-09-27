/** The state-ownership registry: every zustand store under webui/src/state/ has exactly one owner from these five,
 * and every key of every store's initial state (data fields and actions alike) is listed here exactly once.
 * `webui/tests/stateOwners.test.ts` imports each store, calls `.getState()`, and checks its key set against the
 * matching entry below; adding a store or a field without registering it here makes that test fail by design.
 *
 * The browser's schema cache (state/catalog.ts) is intentionally not a zustand `create()` store (see catalog.ts), so
 * it has no entry here and the test skips it by name (NON_ZUSTAND_STATE). */

export type Owner = "计算输入" | "文档外观" | "视图" | "计算结果" | "用户首选项";

export const OWNER_LABEL: Record<Owner, string> = {
  计算输入: "计算输入 (state/cookInputs.ts) — what the server's check of the graph reads; a change bumps its version and makes every result stale",
  文档外观: "文档外观 (state/look.ts) — saved with the graph, undoable, but never part of the cook: positions, group boxes, the display node, playback",
  视图: "视图 (state/viewer.ts) — this browser tab's own: selection, panels, menus, the 「消息」 panel open or shut, the 2D/3D view, canvas selection/drag ephemera; never saved, never undone",
  计算结果: "计算结果 (state/results.ts) — a cache of the server's last answer; trusted only while it answers the current 计算输入 (useTrustedResults)",
  用户首选项: "用户首选项 (state/preferences.ts) — this browser's remembered tool setup, the same across every graph, in localStorage",
};

/** store variable name -> { field/action name -> owner }. */
export const OWNERS: Record<string, Record<string, Owner>> = {
  useCookInputs: {
    graphId: "计算输入", meta: "计算输入", exposed: "计算输入", cookRange: "计算输入", nodes: "计算输入", order: "计算输入",
    edges: "计算输入", kept: "计算输入", version: "计算输入",
    load: "计算输入", setGraphId: "计算输入", setMeta: "计算输入", setExposed: "计算输入", setCookRange: "计算输入", setNode: "计算输入",
    insertNode: "计算输入", removeNodes: "计算输入", setEdges: "计算输入", setKept: "计算输入",
  },
  useLook: {
    positions: "文档外观", onNode: "文档外观", boxes: "文档外观", displayId: "文档外观", displayPort: "文档外观",
    playback: "文档外观", version: "文档外观",
    load: "文档外观", setPosition: "文档外观", removeNodes: "文档外观", setOnNode: "文档外观", setDisplay: "文档外观",
    setDisplayPort: "文档外观", setPlayback: "文档外观", addBox: "文档外观", setBox: "文档外观", moveBox: "文档外观",
    removeBoxes: "文档外观",
  },
  useViewer: {
    selectedId: "视图", expanded: "视图", panelTab: "视图", reveal: "视图", panTo: "视图",
    inspectorFit: "视图", menu: "视图", templatesOpen: "视图", logOpen: "视图", frames: "视图", frame: "视图", playing: "视图",
    playDir: "视图", fps: "视图", loads: "视图", docId: "视图", role: "视图", peerBanner: "视图", file: "视图", dirty: "视图",
    undoLabel: "视图", redoLabel: "视图", canvas: "视图", selectedEdgeIds: "视图", selectedBoxIds: "视图",
    select: "视图", toggleExpanded: "视图", setPanelTab: "视图", revealParam: "视图", requestPan: "视图",
    setInspectorFit: "视图", openMenu: "视图", setTemplatesOpen: "视图",
    setLogOpen: "视图", setFrame: "视图", setFrames: "视图", togglePlay: "视图", play: "视图", setPlaying: "视图",
    claimEditing: "视图", stayViewer: "视图", freshDoc: "视图", setFile: "视图", setSaveState: "视图",
    setCanvasNode: "视图", removeCanvasNodes: "视图",
    scrubbing: "视图", setScrubbing: "视图",  // 正在拖动时间线：拖动过程中不取帧，松开后才取
    setSelectedNodes: "视图", setSelectedEdges: "视图", setSelectedBoxes: "视图",
    setEdgeSelected: "视图", setBoxSelected: "视图", reset: "视图",
  },
  useView2D: {
    mode: "视图", setMode: "视图", right: "视图", setRight: "视图", left: "视图", setLeft: "视图", views: "视图", dropView: "视图",
    zoomPercent: "视图", navigable: "视图", fitAsk: "视图", oneAsk: "视图", goAsk: "视图", fit: "视图", one: "视图", goTo: "视图",
  },
  useViewOptions: { o: "视图", set: "视图", reset: "视图" },
  // 当前条目 per 逐项处理 block (state/items.ts): a view setting; changing it cooks nothing and makes no result stale
  useItems: { view: "视图", setItem: "视图", reset: "视图" },
  useViewCamera: { view: "视图", ortho: "视图", look: "视图", frameAsk: "视图", viewAsk: "视图", cameras: "视图", selected: "视图",
                     setView: "视图", setOrtho: "视图", setLook: "视图", frame: "视图", setCameras: "视图", setSelected: "视图" },
  useViewerNote: { note: "视图", why: "视图", n: "视图", say: "视图" },
  // catchingUp：本轮放弃实时播放（缓存未跟上，逐帧等待拉取，不跳帧）
  useViewLoads: { loaded: "视图", decoded: "视图", stale: "视图", catchingUp: "视图" },
  useRulerView: { zoom: "视图", setZoom: "视图" },
  useResults: {
    reply: "计算结果", results: "计算结果", forCookInputs: "计算结果", plan: "计算结果", statusProblem: "计算结果",
    now: "计算结果", job: "计算结果", queueSwitches: "计算结果", maxFrames: "计算结果", storage: "计算结果", byNode: "计算结果", blockedAt: "计算结果", deliveries: "计算结果",
    setReply: "计算结果", clearResults: "计算结果", setStatusProblem: "计算结果", applyNodeDone: "计算结果", setNow: "计算结果",
    setJob: "计算结果", patchJob: "计算结果", setQueueSwitches: "计算结果", setMaxFrames: "计算结果", setStorage: "计算结果", setNodeStatus: "计算结果", removeNodeStatus: "计算结果",
    setDelivery: "计算结果", deliveryFor: "计算结果", reset: "计算结果",
  },
  useUploads: { tasks: "计算结果" },
  // this browser's login (state/session.ts): a cache of the server's response, like the results
  useSession: { state: "计算结果", load: "计算结果", set: "计算结果", logout: "计算结果" },
  // 用户已授权的本机目录及其修改次数（state/localDirs.ts；句柄本身存于 IndexedDB）
  useLocalDirs: { version: "用户首选项", bump: "用户首选项" },
  usePreferences: {
    split: "用户首选项", inspectorWidth: "用户首选项", minimap: "用户首选项", recentNodes: "用户首选项", displayOptionsTab: "用户首选项",
    browseGroup: "用户首选项",
    timelineOpen: "用户首选项", timelineHeight: "用户首选项", playbackMode: "用户首选项",
    setSplit: "用户首选项", setInspectorWidth: "用户首选项", setMinimap: "用户首选项", toggleMinimap: "用户首选项",
    pushRecentNode: "用户首选项", setDisplayOptionsTab: "用户首选项", setBrowseGroup: "用户首选项", setTimelineStrip: "用户首选项", setPlayback: "用户首选项", 
  },
};

/** Stores that are intentionally not zustand `create()` calls, so the "no create( outside webui/src/state/" scan in
 * `stateOwners.test.ts` does not need to find one for them, and its per-store key check skips them as well. */
export const NON_ZUSTAND_STATE = ["catalog"];

/** Stores outside webui/src/state/ that are not one of the graph editor's five kinds of state: ui/Feedback.tsx's
 * useMine (the administrator's own submitted feedback, listed in a dialog) belongs to the auth / admin-feedback
 * subsystem, not to the document, a view of it or a preference. The "no create( outside state/" scan in
 * `stateOwners.test.ts` explicitly allows the files named here instead of silently missing them. */
export const OUT_OF_SCOPE_STORES = ["ui/Feedback.tsx"];
