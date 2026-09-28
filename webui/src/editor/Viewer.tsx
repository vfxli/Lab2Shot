import { msg, textOf } from "../messages/message";
import { useEffect, useRef, useState, lazy, Suspense } from "react";
import { countText } from "../platform/format";
import { useTypes } from "../state/catalog";
import { useLook } from "../state/look";
import { useResults, useTrustedResults } from "../state/results";
import { ValuesView } from "./ValuesView";
import { useDisplayPlan, type DisplayPlan, type Stage, type ViewItem } from "../view/plan";
import { Stage2D, useManifest } from "../view/Stage2D";
import { useCloudProxy, useSceneKinds } from "../view/kinds3d";
import { DisplayOptions } from "../ui/DisplayOptions";
import { KINDS } from "../view/kinds3d";
import { useViewCamera, useViewerNote } from "../state/viewer";
import { Empty } from "../ui/Empty";

import { HANDLE_HINT } from "../view/handles2d";
import { channelOptions, pickOf, pickValue, pickedIn, singleChannel, type Layer } from "../model/view2d";
import { type ViewFacts } from "../model/viewControls";
import { viewAvailable } from "../view/available";
import { usable } from "../api/applies";
import { channelsOf } from "../model/view2d";
import { useView2D, useViewOptions } from "../state/viewer";
import { ownReason } from "../platform/util";
import { useUploadHint } from "../view/localPick";
import { ViewButtons } from "./ViewPick";
import { Segmented } from "../ui/Button";
import { Loading } from "../ui/Loading";
import { usePartial } from "../view/partial";
import { ViewerFrame } from "./ViewerFrame";
import { CurveStrip, CurveToggle, Curves } from "./ViewerCurves";
import { BgPick, ChannelPick, GradePick, MergePick, ModePick, TintPick } from "./previewBar";

// the 3D stage (three.js and its renderers) loads when a 3D result is first shown, not with the editor
const Stage3D = lazy(() => import("../view/Stage3D").then((m) => ({ default: m.Stage3D })));

/** The viewer: the active node's display plan (view/plan.ts) on its 2D or 3D stage, with one control bar (the
 * stage, the picture, what is shown, the node's handles) and, for a node with curves, the curve editor: docked under
 * the stage when the node also has a picture or a scene (the 曲线 button shows it; it is never drawn over the stage),
 * or the whole view when curves are all the node gives.
 *
 * 本文件只负责选择舞台、舞台上的内容以及控制栏上的控件。外框与角标位于 editor/ViewerFrame.tsx，
 * 曲线编辑器位于 editor/ViewerCurves.tsx，二维预览链的控件位于 editor/previewBar.tsx，显示选项位于
 * ui/DisplayOptions.tsx。 */

const EMPTY_PLAN = { elements: [], pointMaps: [] } as unknown as DisplayPlan; // before a node is displayed

const TRANSFORM_MODES = { translate: "移动", rotate: "旋转", scale: "缩放" } as const;

/** The picture picker's option text: an output-settings node's results carry the node they were made from
 * ("FaceAnything · 规范坐标"), which only fits a select box for a node with a short name, and the current node
 * is already named in the pill to the left, so the port alone identifies it (unique unless two of the pictures share a
 * port name from different sources, in which case the full text is kept). */
function pictureOptionLabel(p: { label: string }, all: { label: string }[]): string {
  const port = (l: string) => (l.includes(" · ") ? l.slice(l.lastIndexOf(" · ") + 3) : l);
  const short = port(p.label);
  return all.filter((q) => port(q.label) === short).length > 1 ? p.label : short;
}

/** The picture the 2D stage draws: the port the user chose, else the node's main result, else one with a result, else
 * the first; never the first by position while the node declares which of its outputs is the main one. */
function shownOf(plan: DisplayPlan, port: string | null): ViewItem | null {
  const find = (want: (p: ViewItem) => boolean) => plan.pictures.find(want) ?? null;
  // 某一层是否有可绘制内容只由一处判定（`view/origin.ts` 的 `from.own`）
  return find((p) => p.port === port) ?? find((p) => p.port === plan.mainPort) ?? find((p) => p.from.own) ?? plan.pictures.at(0) ?? null;
}

export function Viewer() {
  const plan = useDisplayPlan();
  const job = useResults((s) => s.job);
  const types = useTypes();
  const displayPort = useLook((s) => s.displayPort);
  const setDisplayPort = useLook((s) => s.setDisplayPort);
  // The stage belongs to the viewer, not the node: the one last picked by hand, else the one shown last; another one only
  // when the node has nothing to show there (reported in a note).
  const [picked, setPicked] = useState<Stage | null>(null);
  const lastShown = useRef<Stage | null>(null);
  const look = useViewCamera((s) => s.look);
  const say = useViewerNote((s) => s.say);
  // 记录使用者切换过的叠加物，而非隐藏的叠加物：初始是否显示由节点决定（plan.overlaysOn），
  // 屏幕上是否显示 = 默认值 异或 是否切换过。记录「切换过」而非「隐藏」，默认值才能持续生效。
  const [flipped, setFlipped] = useState<Set<string>>(new Set());
  const [pointLabel, setPointLabel] = useState(0);
  const [transformMode, setTransformMode] = useState<keyof typeof TRANSFORM_MODES>("translate");
  const kinds = useSceneKinds(plan ?? EMPTY_PLAN);
  // 点云采用代理显示时始终标注。view 层只计算倍数，提示文字在此生成，
  // 内容为「显示了 30 万 / 共 100 万点」，而非倍数。
  const proxyNow = useCloudProxy(plan ?? EMPTY_PLAN);
  // 删减点数是默认行为，因此只有一种提示：点数过多，管理员可调整「点云上限」。
  const proxyWhy = "I-VIEW-CLOUDPROXYWHY";
  const cloudProxy = proxyNow.every > 1
    ? [{ kind: "proxy" as const, key: "proxy",
         text: textOf(msg("N-VIEW-CLOUDPROXY", { shown: countText(proxyNow.shown), total: countText(proxyNow.total) })),
         tip: textOf(msg(proxyWhy, { every: proxyNow.every, shown: countText(proxyNow.shown), total: countText(proxyNow.total) })) }]
    : [];
  // 边算边看的进度与已载入浏览器的帧不在画面角另行通知，时间线色带已有标示。
  // 二维预览链：黑白点、着色、运算、背景属于显示选项（跨节点保持，model/viewOptions.ts）；左右各自的通道与当前模式
  // 随视图保存（state/view2d.ts）；右侧显示的层即 displayPort（随节点图保存）。
  const o = useViewOptions((s) => s.o);
  const setOption = useViewOptions((s) => s.set);
  const mode = useView2D((s) => s.mode);
  const setMode = useView2D((s) => s.setMode);
  const leftIndex = useView2D((s) => s.left);
  const setLeft = useView2D((s) => s.setLeft);
  const rightIndex = useView2D((s) => s.right);
  const setRight = useView2D((s) => s.setRight);
  const shownPicture = plan ? shownOf(plan, displayPort) : null;
  // 两侧数据各自的描述：通道数、值域、类别表均取自此处（`image` 家族的输出口在目录中没有通道数，包中有）。
  const plateManifest = useManifest(plan?.plate ?? null);
  const rightManifest = useManifest(shownPicture?.fp ?? null);
  const leftChannels = channelsOf(String(plateManifest?.type ?? ""));
  const rightChannels = channelsOf(String(rightManifest?.type ?? ""));
  // 双击显示某个节点时，按该节点的预览标签切换一次模式。此后以使用者的选择为准，直到切换到下一个节点。
  const snapped = useRef<string | null>(null);
  useEffect(() => {
    if (!plan || snapped.current === plan.node.id) return;
    snapped.current = plan.node.id;
    setMode(plan.preview);
  }, [plan?.node.id, plan?.preview]); // eslint-disable-line react-hooks/exhaustive-deps

  // Why the node cannot be cooked yet, as checked by the server on the current graph (no file, no camera chosen, ...).
  const trustedResults = useTrustedResults();
  const blocked = plan ? trustedResults[plan.node.id]?.error : undefined;
  // 边算边看 (view/partial.ts): while this node is being cooked, the stage draws the frames it has already written,
  // until its own result arrives and the content address takes over.
  const partialPort = plan ? (shownOf(plan, displayPort)?.port ?? plan.mainPort) : "";
  const partial = usePartial(plan?.node.id ?? null, partialPort, !!plan && !!shownOf(plan, displayPort)?.from.own);
  // 使用者本机上的画面（`view/localPick.ts`）：该判断只在 `view/plan.ts` 中进行一次，
  // 此处读取其结果，避免两处判断得出不一致的结论。
  const local = plan?.file ?? null;
  const going = useUploadHint(plan?.node.id ?? null); // Or the progress of its file's upload, when there is nothing to draw.

  const wanted: Stage | null = plan ? (picked ?? lastShown.current ?? plan.defaultStage) : null;
  const otherStage: Stage = wanted === "2d" ? "3d" : "2d";
  const stage: Stage | null = !plan || !wanted ? null : plan.why[wanted] === null ? wanted : plan.why[otherStage] === null ? otherStage : plan.defaultStage;
  useEffect(() => {
    if (!plan || !stage) return;
    // The pill gives a short summary and shows the full sentence on hover (no wrapped or truncated text).
    if (lastShown.current && stage !== lastShown.current && stage !== wanted)
      say(msg("N-VIEW-STAGESWITCHED", { stage: stage.toUpperCase(), reason: plan.shortly[wanted!] ?? "" }), plan.why[wanted!] ?? "");
    lastShown.current = stage;
  }, [plan?.node.id, stage]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!plan || !stage) return <ViewerFrame body={<Empty title="没有显示节点" hint="双击一个节点，或点节点右上角的眼睛图标" />} />;

  const picture = shownOf(plan, displayPort);
  // 二维舞台右侧画的：选中的层；它还没有数据时，二维手柄作用的输入画面（view/plan.ts handleInput）
  const drawn = picture?.fp || !plan.handleInput ? picture : plan.handleInput;
  const toggle = (key: string) => setFlipped((f) => (f.has(key) ? new Set([...f].filter((k) => k !== key)) : new Set([...f, key])));
  // 三维种类默认全部显示，因此对它们而言「切换过」与「隐藏」等价，行为不变。
  const shows = (key: string, on = true) => on !== flipped.has(key);
  // 叠加显示按输出口划分：由一个列表拆出的多条（view/plan.ts 的 `expand`）是同一输出口的同一份结果，
  // 因此共用一个开关，同时显示或隐藏。
  const portOf = (o: ViewItem) => `${o.nodeId}.${o.port}`;
  const hidden = new Set<string>([...plan.overlays.filter((o) => !shows(portOf(o), plan.overlaysOn)).map((o) => o.key),
                                  ...kinds.filter((k) => !shows(k))]);
  // 边算边看 counts as having something to show: the frames written so far are the node's own result (view/partial.ts).
  const cooked = plan.items.some((it) => it.from.own) || !!partial;
  const handle2d = plan.handles.find((h) => h.stage === "2d");
  // 左右两侧当前显示的层与通道：层与通道列在同一下拉中，同时提供整体与单通道（与 Nuke 一致）。
  // 数据中不存在的层与通道不列出，这也是采用下拉而非按钮的原因。
  const leftLayers: Layer[] = plan.plate ? [{ port: "", label: plan.plateLabel, channels: leftChannels || 3 }] : [];
  // 每个输出口对应一层：下拉中选择的是层，层即输出口（displayPort 保存的也是口名）。由一个画面列表拆出的
  // 多条属于同一输出口，在此算作一层，二维舞台每次只显示一张（DisplayPlan.pictures：one shown at a time）。
  // 列出的是节点拥有的层，而非已计算的层（多层 EXR 须可选层），因此不得按是否已有结果过滤：
  // 读取节点持有本机原件时不参与计算（`graph/actions.ts`），始终没有结果，过滤后将无层可选。
  // cook 只决定是否有像素可绘制，不决定可选的层：层即输出口，输出口由文件中的图层生成
  // （`lab2shot/nodes/core/input.py made_ports`，字节尚未传输时为申报的图层）。
  // 选中的层尚无像素时，由舞台说明原因，不显示空白画面。
  const rightLayers: Layer[] = plan.pictures.filter((p2, i, all) => all.findIndex((q) => q.port === p2.port) === i).map((p2) => ({
    port: p2.port,
    // 尚无像素的层在名称上标注「还没算」，避免使用者在无提示的情况下等待。
    // 选中该层时，画面显示上游原图（不压暗）；下拉中的这段文字即为全部说明，不另设通知控件。
    label: pictureOptionLabel(p2, plan.pictures),
    // 可绘制的层不加此标注：主画面层即使未计算也能绘制（使用本机文件，无传输且保持完整精度），
    // 此时标注「还没算」与屏幕上显示的图像相矛盾。
    // 判定依据为 `view/origin.ts` 的 `from.own`：服务器算出的结果以及本节点读取的本机文件均属于节点自身的结果。
    note: p2.from.own ? undefined : "还没算",
    channels: (p2.fp === picture?.fp ? rightChannels : 0) || channelsOf(p2.type) || 1,
  }));
  const leftOptions = channelOptions(leftLayers);
  const rightOptions = channelOptions(rightLayers);
  const leftValue = pickedIn(pickValue({ port: "", index: leftIndex }), leftOptions);
  const rightValue = pickedIn(pickValue({ port: picture?.port ?? "", index: rightIndex }), rightOptions);
  // 右侧选择一项即同时切换层（displayPort，随节点图保存）与通道（视图状态）。
  const pickRight = (v: string) => {
    const got = pickOf(v);
    if (!got) return;
    if (got.port !== picture?.port) setDisplayPort(got.port);
    setRight(got.index);
  };
  const channels = mode === "plate" ? leftChannels : rightChannels;
  const shownIndex = mode === "plate" ? leftIndex : rightIndex;
  const facts: ViewFacts = { stage, shows: new Set<string>(), options: o, mode,
                             channels, single: singleChannel(shownIndex, channels), ready: !!(mode === "plate" ? plateManifest : rightManifest) };
  const preview = viewAvailable(facts);
  const off = (control: Parameters<typeof usable>[1]) => !usable(preview, control);
  // 「贴合」：按数据未裁切的真实最小值与最大值设置黑白点（包中的 range 为 1%–99%，本身已裁切；
  // full_range 由 data/payloads.py ExrWriter 另行记录）。黑白点以 range 内的 0 到 1 表示，因此需要换算。
  const fitRange = () => {
    const m = mode === "plate" ? plateManifest : rightManifest;
    const range = (m?.meta.range as number[] | undefined) ?? [0, 1];
    const full = (m?.meta.full_range as number[] | undefined) ?? range;
    const span = range[1] - range[0] || 1e-6;
    return { black: (full[0] - range[0]) / span, white: (full[1] - range[0]) / span };
  };
  // 二维舞台当前绘制的内容（每项为一个名称）。各控件是否生效由集中声明决定
  // （`model/viewControls.ts`：`lineWidth` 要求绘制框、跟踪点或手柄，`overlayByPerson` 要求绘制人物框等）。
  // 此处只如实列出绘制内容，不按类型再次过滤：再次过滤等于重复推断该声明，新增叠加物时需同步修改，
  // 且两处不一致时不会报错（控件会一直处于禁用状态）。
  const shows2d = new Set<string>([...plan.overlays.filter((it) => it.from.own && !hidden.has(it.key)).map((it) => it.type), ...(handle2d ? ["handle"] : [])]);
  const active = plan.handles.find((h) => h.stage === stage);
  const hint = active ? HANDLE_HINT[active.kind] : null; // How to use the node's handle.

  // A node that gives only values (「浮点」, 「拆分相机」) and has no picture upstream: the values are the stage. With a
  // picture upstream (「AnyCalib 镜头标定」: values derived from a plate) the 2D stage shows that picture, with the values in a
  // strip under it. This is the same display a solver with the same outputs and a 3D result (COLMAP) gets: one rule based on
  // the available data, rather than a separate board for this node.
  const nothingElse = !plan.pictures.length && !plan.overlays.length && !plan.elements.length && !plan.pointMaps.length;
  // Decided by whether a picture port is upstream (`plateUpstream`), not by a picture's fingerprint (`plate`): before a
  // file is chosen there is no fingerprint yet, and the node must then look as COLMAP does (the empty-state message, the
  // values in the strip under it).
  const onlyValues = plan.values.length > 0 && nothingElse && !plan.plateUpstream;
  const paramStrip = !!plan.def?.strip?.length; // 节点声明在值条上显示的参数（NodeDef.strip，如「LensDistortion」使用的镜头）。
  const curves = plan.strips.find((s) => s.fp) ?? null;
  const onlyCurves = !!curves && nothingElse && !plan.values.length; // Curves are the whole view.
  let body: React.ReactNode;
  // The timeline's zoom bar (适应 / 1:1 / %) is available on the 2D stage, whatever draws it (a picture, or a 3D result
  // held in its camera), and on the 3D stage when looking through a camera. It works while there is a picture to zoom
  // (state/view2d.ts navigable) and states the reason otherwise.
  // 本机文件只能替代主画面层（`view/origin.ts fileStandsFor`）：选中 depth、法线等层时无法提供像素，
  // 此时进入下方的空状态并说明原因，不显示空白画面，也不让使用者在无提示的情况下等待。
  // 尚无任何输出口时（文件刚选定、服务器状态回复尚未返回）没有 picture 可查询，此时改为判断当前画面是否持有本机文件。
  const localShows = picture ? !!picture.from.file : !!plan.file;
  const stage2d = stage === "2d" || (stage === "3d" && !!look);
  if (onlyValues) {
    body = <ValuesView plan={plan} board />;
  } else if (onlyCurves) {
    body = (
      <div className="curves-board">
        <Curves item={curves} />
      </div>
    );
  } else if (plan.emptyList) {
    // 空结果不是错误：an empty list is shown as itself, with what it means.
    body = <Empty title={textOf(msg("I-LIST-EMPTY"))} hint={textOf(msg("I-LIST-EMPTYWHY"))} />;
  } else if (!cooked && !plan.plate && !(localShows && stage === "2d")) {
    const waiting = job?.target === plan.node.id && job.position != null;
    const busy = !!job && plan.node.data.status === "cooking";
    const title = waiting ? `${plan.node.data.label} 在队列里排队` : busy ? `${plan.node.data.label} 计算中…` : `${plan.node.data.label} 还没有结果`;
    const hint = waiting || busy ? plan.node.data.note : going ?? (blocked ? ownReason(blocked.text, plan.node.data.label) : "在这个节点上右键选「计算」，或按 Ctrl+Enter，生成结果");
    body = <Empty title={title} hint={hint} />;
  } else if (stage === "3d") {
    // 二维舞台不通过节点自身的相机观看：只输出三维结果的节点计算完成后默认进入 3D 查看点云与相机，
    // 视角下拉中可选择相机；切换到 2D 时按原样播放上游序列。因此 3D 舞台仅在 `stage === "3d"` 时挂载。
    body = (
      <Suspense fallback={<Loading what="3D 视图" />}>
        <Stage3D plan={plan} hidden={hidden} transformMode={transformMode} hint={hint}
          partial={partial && job ? { info: partial.info, job: job.id, node: plan.node.id, port: partialPort } : null} />
      </Suspense>
    );
  } else {
    // 本机所选文件与服务器计算结果使用同一个二维舞台，显示无须等待上传与回传。二者必须位于 JSX 中的同一位置：
    // 更换组件，或将同一个 Stage2D 写在 if/else 的两个分支中，都会导致 React 卸载后重新挂载，中间出现空白帧（慢速网络下可达数百毫秒）。
    // 因此从选定文件到计算完成，始终是同一块画布，本机文件只是其帧源之一（transfer/sources.ts localFirst）。
    body = (
      <Stage2D plan={plan} picture={drawn} hidden={hidden} pointLabel={pointLabel} mode={mode}
        leftIndex={leftIndex} rightIndex={rightIndex} onChannel={mode === "plate" ? setLeft : setRight}
        local={local}
        partial={partial && job ? { info: partial.info, job: job.id, node: plan.node.id, port: partialPort } : null} />
    );
  }

  // 每个输出口一个 chip（与上方 portOf 同理）：由一个列表拆出的多条共用一个开关。
  const chips2d = plan.overlays.filter((o) => o.from.own)
    .filter((o, i, all) => all.findIndex((q) => portOf(q) === portOf(o)) === i)
    .map((o) => ({ key: portOf(o), label: o.context ? `${types[o.type]?.label ?? o.label} · 输入` : o.label }));
  const kinds3d = kinds;
  const tools = (
    <>
      <Segmented hud label="视图" value={stage} options={(["2d", "3d"] as const).map((s) => ({ value: s, label: s.toUpperCase(), disabled: plan.why[s] }))} onChange={setPicked} />
      {stage === "2d" && chips2d.length > 0 && (
        <Segmented hud label="叠加显示" value={new Set(chips2d.filter((c) => !hidden.has(c.key)).map((c) => c.key))} options={chips2d.map((c) => ({ value: c.key, label: c.label }))} onChange={toggle} />
      )}
      {stage === "2d" && (
        // 排列顺序：模式 · 左 · 中 · 右 · 背景。
        // 不包裹在单个 div 中：包裹后成为整体，空间不足时浏览器只能将其整体换行，
        // 导致 2D / 3D 与「视图设置」被挤到不同行。
        <>
          <ModePick mode={mode} onPick={setMode} />
          {/* 三种模式始终显示同一组控件，不适用的控件置灰并说明原因，不随模式显示或隐藏。控件数量变化会改变
              窄窗口下的换行数，导致工具栏高度与画布位置跳动，影响在两个节点间切换对比。
              空间不足时换行，不使用横向滚动；允许换行，但控件数量不得变化。 */}
          <ChannelPick side="left" value={leftValue} options={leftOptions} onPick={(v) => setLeft(pickOf(v)?.index ?? null)}
            off={mode === "result"} />
          <MergePick op={o.op} mix={o.mix} offOp={mode !== "over"} offMix={off("mix")}
            onOp={(op) => setOption({ op })} onMix={(mix) => setOption({ mix })} />
          <ChannelPick side="right" value={rightValue} options={rightOptions} onPick={pickRight}
            off={mode === "plate"} />
          <TintPick value={o.tint} off={off("tint")} onPick={(tint) => setOption({ tint })} />
          <GradePick black={o.black} white={o.white} off={off("black")} onSet={setOption} fit={fitRange} />
          <BgPick bg={o.bg} colour={o.bgColor} onPick={setOption} />
        </>
      )}
      {stage === "3d" && kinds3d.length > 0 && (
        <Segmented hud label="显示的种类" value={new Set(kinds3d.filter((k) => !hidden.has(k)))} options={kinds3d.map((k) => ({ value: k, label: KINDS[k] }))} onChange={toggle} />
      )}
      {/* 「点的种类」仅适用于 points 手柄（主体 / 排除）。火柴人手柄的 labels 是 18 个关节名称，
          不供使用者在此选择；若不排除，工具条上会多出一排超出画面的关节名。 */}
      {stage === "2d" && handle2d?.kind === "points" && handle2d.labels.length ? (
        <Segmented hud label="点的种类" value={String(pointLabel)} options={handle2d.labels.map((l, i) => ({ value: String(i), label: l }))} onChange={(i) => setPointLabel(Number(i))} />
      ) : null}
      {/* 视角与框显按钮位于此行：它们属于视图工具，与 2D / 3D、显示种类同处一行，不浮于画面上。
          透过相机观看时不显示（此时视角即该相机，见 state/viewTools.ts 的 look）。 */}
      {stage === "3d" && <ViewButtons />}
      {stage === "3d" ? <DisplayOptions stage="3d" shows={new Set(kinds3d)} /> : <DisplayOptions stage="2d" shows={shows2d} preview={facts} />}
      {curves && !onlyCurves && <CurveToggle />}
      {stage === "3d" && plan.handles.some((h) => h.kind === "transform") && (
        <Segmented hud label="变换手柄" value={transformMode} options={(Object.keys(TRANSFORM_MODES) as (keyof typeof TRANSFORM_MODES)[]).filter((m) => m !== "scale" || plan.handles.some((h) => h.kind === "transform" && h.params.scale)).map((m) => ({ value: m, label: TRANSFORM_MODES[m] }))} onChange={setTransformMode} />
      )}
    </>
  );
  const strip = curves && !onlyCurves ? <CurveStrip item={curves} /> : !onlyValues && (plan.values.length || paramStrip) ? <ValuesView plan={plan} board={false} /> : null;
  return (
    <ViewerFrame plan={plan} tools={tools} body={body} hint={stage === "3d" ? null : hint} strip={strip} stage2d={stage2d}
      proxy={cloudProxy}
      shown={plan.node.id} />
  ); // the 3D stage: in its foot
}
