import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import type { BoxesData, Manifest, TracksData } from "../api";
import { msg, textOf } from "../messages/message";
import { setParam } from "../graph/actions";
import { useStageNotes, useStagePicture, useViewer } from "../state/viewer";
import { clearLoads, reportLoads } from "../transfer/readiness";
import { drawLoading, useDecodedFrames, useLoadedFrames } from "../transfer/frames";
import { isPlane } from "../transfer/plane";
import type { LocalPicture } from "./localPick";
import { useDescribed } from "../transfer/described";
import { useDevicePixelRatio } from "../platform/size";
import { clickHandle, dragHandle, dragStart, drawHandle, moveDrag, personAt, pickedPeople, removeAt, type Drag, type Pt } from "./handles2d";
import { drawBackground, drawBoxes, drawTracks, type Frame } from "./overlays";
import { drawLook, useGpuState, type Box, type Merge, type Side } from "./look";
import type { DisplayPlan, ViewItem } from "./plan";
import { useStageSources } from "./stageSources";
import { useViewOptions } from "../state/viewer";
import { DEFAULTS } from "../model/viewOptions";
import { CHANNEL_KEYS, channelsOf, lookIndex, mixOf, pictureSize, singleChannel, type Mode, type Tint } from "../model/view2d";
import { bgColourOf } from "../platform/palette";
import { useView2DNav } from "../state/viewer";
import { followPress } from "../platform/drag";
import { useWriteLock } from "../ui/writeLock";
import type { Partial as PartialResult } from "./partial";

/** 二维舞台：左侧为原图、右侧为该节点的结果，按当前档位（仅原图 / 运算 / 仅结果）绘制，
 * 并叠加自带数据的叠加物（人物框、跟踪点）及节点的二维手柄。
 *
 * 主源（时间线跟随的一侧）在「仅结果」档中为右侧，其他档中为左侧；另一侧同样经取帧账本获取（view/stageSources.ts），
 * 两侧均到达后在 GPU 上合成（view/look.ts）。 */

interface Props {
  plan: DisplayPlan;
  picture: ViewItem | null; // 右侧显示该节点的哪一层（其一个输出端口）
  hidden: Set<string>; // 在工具栏中关掉的叠加物
  pointLabel: number; // 点一下给出的标签（带标签的手柄）
  mode: Mode; // 仅原图 / 运算 / 仅结果
  leftIndex: number | null; // 左侧选取的通道序号（null：整体）
  rightIndex: number | null; // 右侧选取的通道序号（null：整体）
  onChannel: (index: number | null) => void; // 键盘 R G B A 切换的是当前一侧的通道
  // 用户在本标签页中选择的文件（view/localPick.ts useLocalPicture）：画面优先使用这些文件，
  // 不等待服务器
  local?: LocalPicture | null;
  // 边算边看（view/partial.ts）：该节点计算期间，用它已写出的帧代替原图显示，结果本身一可用即撤下
  partial?: { info: PartialResult; job: string; node: string; port: string } | null;
}

/** 已计算的包的包说明（meta），每个包只载入一次：画面有没有 alpha、浮点图的显示范围是多少。 */
export const useManifest = (fp: string | null): Manifest | null => useDescribed<Manifest>("manifest", fp ? [fp] : [])[0] ?? null;

/** 列表拆开后的每一条（view/plan.ts 的 `expand`）各一份：都到了才给（一条没到时不画半份），没到时给常量空数组（作依赖）。 */
function useEach<T>(kind: "boxes" | "tracks", fps: readonly string[]): T[] {
  const all = useDescribed<T>(kind, fps);
  return useMemo(() => (all.length && all.every((v) => v !== null) ? (all as T[]) : EMPTY), [all]);
}
const EMPTY: never[] = [];

/** 将多份人物框合并为一份：列表拆出的各条本就是同一画面中的多个人物（宽高、帧均相同，
 * 编号保持不变：`lab2shot/data/items.py` `_boxes_split` 只挑选人物，不修改编号）。 */
const joinBoxes = (all: BoxesData[]): BoxesData | null =>
  all.length ? { ...all[0], people: all.flatMap((b) => b.people) } : null;

/** 将多组跟踪点合并为一份（同上：`_tracks_split` 按组挑选点，宽高和帧不变）。
 * 分数与平面四角为可选项：仅当每一份都具备时才拼接，否则两侧无法对应；宁可不绘制，也不绘制错误结果。 */
const joinTracks = (all: TracksData[]): TracksData | null => {
  if (!all.length) return null;
  const every = <T,>(pick: (t: TracksData) => T[] | undefined): T[] | undefined =>
    all.every((t) => pick(t)) ? all.flatMap((t) => pick(t)!) : undefined;
  return { ...all[0], points: all.flatMap((t) => t.points), names: all.flatMap((t) => t.names),
           confidence: every((t) => t.confidence), outline: every((t) => t.outline) };
};

export function Stage2D({ plan, picture, hidden, pointLabel, mode, leftIndex, rightIndex, onChannel, local, partial }: Props) {
  const frame = useViewer((s) => s.frame);
  const playDir = useViewer((s) => s.playDir);
  const locked = !!useWriteLock(); // 写不了（ui/writeLock.ts）：手柄照画，不接
  const [el, setEl] = useState<HTMLCanvasElement | null>(null);
  // 画布外面的一层（.canvas2d-box）：量尺寸的是它，画布本身按量到的尺寸用像素定宽高、贴在左上角（不用 CSS 100%）。
  // 这样拖参数栏的分割线时，新尺寸量到、重画之前的那一两帧，旧画面只会被裁掉或露出一条空白，不会被 CSS 横向拉伸
  // 成别的比例；量到新尺寸后立即按「适应 / 1:1」的变换（state/view2d.ts）重画（useLayoutEffect：在这一帧绘制之前）
  const [box, setBox] = useState<HTMLDivElement | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const line = useViewOptions((s) => s.o.lineWidth);
  const byPerson = useViewOptions((s) => s.o.overlayByPerson);
  const black = useViewOptions((s) => s.o.black);
  const white = useViewOptions((s) => s.o.white);
  const tint = useViewOptions((s) => s.o.tint);
  const op = useViewOptions((s) => s.o.op);
  const mix = useViewOptions((s) => s.o.mix);
  const bg = useViewOptions((s) => s.o.bg);
  const bgColor = useViewOptions((s) => s.o.bgColor);

  // 左侧为原图（上游最近的画面；无上游画面时，该节点自身输出的画面即为原图，二者是同一规则而非分支），
  // 右侧为该节点计算出的层。当前档位绘制哪两张图只在此处计算一次：
  //   仅原图 → 左；仅结果 → 右；运算 → 左 ⊕ 右。
  // 主源（`mainFp`）由时间线跟随；另一侧（`overFp`）与它共用取帧账本，两侧均到达后才合成。
  const rightFp = partial ? null : picture?.fp ?? null;
  const leftFp = plan.plate ?? rightFp;
  const mainFp = mode === "result" ? rightFp ?? leftFp : leftFp;
  const overFp = mode === "over" && rightFp && rightFp !== leftFp ? rightFp : null;
  const main = partial ? null : mainFp ? { fp: mainFp, type: mainFp === rightFp ? picture?.type ?? "" : "" } : null;
  const manifest = useManifest(main?.fp ?? null);
  const overManifest = useManifest(overFp);
  const overlays = plan.overlays.filter((o) => o.from.own && !hidden.has(o.key)); // 是否有可绘制的内容：见 view/origin.ts
  // 每种叠加物有几份即绘制几份：列表在视图中已拆分为各条（view/plan.ts 的 `expand`），
  // 在此层与「一个节点输出多份结果」没有区别，合并为一份绘制
  const boxesItems = overlays.filter((o) => o.type === "boxes" && !o.context);
  const sourceBoxesItems = plan.overlays.filter((o) => o.type === "boxes" && o.context && o.from.own);
  const sourceBoxesItem = sourceBoxesItems[0] ?? null;
  const tracksItems = overlays.filter((o) => o.type === "tracks2d");
  // 手柄跟随跟踪结果，不受「叠加物隐藏」开关影响：开关隐藏的是画面上的轮廓，四角手柄仍须位于跟踪到的位置，
  // 否则关闭叠加物后手柄会回到起始帧的像素位置。因此结果按该节点自身输出的每一份获取，绘制轮廓时才检查开关
  const ownTracksItems = plan.overlays.filter((o) => o.type === "tracks2d" && o.from.own);
  const boxesGot = useEach<BoxesData>("boxes", boxesItems.flatMap((o) => (o.fp ? [o.fp] : [])));
  const sourceGot = useEach<BoxesData>("boxes", sourceBoxesItems.flatMap((o) => (o.fp ? [o.fp] : [])));
  const tracksGot = useEach<TracksData>("tracks", ownTracksItems.flatMap((o) => (o.fp ? [o.fp] : [])));
  const boxes = useMemo(() => joinBoxes(boxesGot), [boxesGot]);
  const sourceBoxes = useMemo(() => joinBoxes(sourceGot), [sourceGot]);
  const tracks = useMemo(() => joinTracks(tracksGot), [tracksGot]);
  const tracksShown = tracksItems.length ? tracks : null; // 绘制在画面上的数据：叠加物隐藏时不绘制（手柄仍跟随 `tracks`）
  // 数据的通道数及其为数值还是画面：均取自包自带的信息（类型 id 即通道数），不依据类型名
  const channels = channelsOf(String(manifest?.type ?? main?.type ?? ""));
  const overChannels = channelsOf(String(overManifest?.type ?? ""));
  // 本机文件刚选择、服务器尚未读取时，尺寸取自解码得到的图像（见下方 localSize）：
  // 这样当前画面始终只有一个舞台，无需另设「本机画面」组件，因为两个组件之间切换必然出现一帧空白
  const [localSize, setLocalSize] = useState<{ w: number; h: number } | null>(null);
  // 画面在舞台上的尺寸只由该帧的画面范围决定，而非由图像的像素数决定
  // （规格与原因见 `model/view2d.ts pictureSize` 的注释）
  const { w: width, h: height } = pictureSize({ partial: partial?.info, meta: manifest?.meta,
                                                overlays: [boxes, sourceBoxes, tracks], decoded: localSize });
  const handleValues = (h: (typeof plan.handles)[number]) => (plan.node.data.params[Object.values(h.params)[0]] as string[] | undefined) ?? [];
  // 手柄输入上的人物框里高亮哪些人：节点自己的结果（选中的人）；没有与点选相符的结果时（还没算、已过期，view/plan.ts
  // underHandles），按点选当前的参数高亮点到的人（只是显示，不计算）
  const personHandle = plan.handles.find((h) => h.kind === "person") ?? null;
  const chosen = !sourceBoxesItem ? null : boxes ? new Set(boxes.people.map((p) => p.id))
    : personHandle ? pickedPeople(sourceBoxes, handleValues(personHandle)) : null;
  // 查看器的平移 / 缩放（state/view2d.ts：各节点共用，透过相机看三维结果时也是同一份）；
  // R G B A 切换当前一侧的通道，再按一次回到「整体」（与 Nuke 相同），仅在指针位于舞台上时生效
  // 这一次渲染的右键单击（在下面、手柄知道了以后定）：按下时取的就是它，绑着这次渲染的帧与点
  let rightClick: ((e: { clientX: number; clientY: number }) => void) | null = null;
  const nav = useView2DNav(box, width, height, {
    keys: (k) => {
      const want = CHANNEL_KEYS[k];
      const have = mode === "plate" ? channels : mode === "over" ? overChannels || channels : channels;
      if (want === undefined || !(want < have)) return false;
      onChannel((mode === "plate" ? leftIndex : rightIndex) === want ? null : want);
      return true;
    },
    // 右键单击（松开时没拖过：view2d.ts 判定）删掉点下的那个点；右键拖动是缩放
    onRightPress: () => rightClick,
  });
  const at = nav.at;

  // 画面自身的帧（时间线可能更长：只算了部分范围的原图）。包说明（manifest）到达之前不请求它的任何帧：人物框或跟踪点
  // 可能先给出画布的宽高（useDescribed 另行载入它们），下方的绘制 effect 因此可能先于画面的包说明运行；此时不知道范围就
  // 请求帧，可能请求到范围之外的帧（404）。
  // 本机文件有该帧时即绘制，不等待包说明（manifest）：上传刚完成、结果刚产生时 manifest 仍在传输中，
  // 若以其为准，画布会被清空一帧。
  const localHas = (f: number) => !!picked && picked.frames.includes(f);
  const shown = (f: number) => (partial ? partial.info.frames_done.includes(f) : localHas(f) || !main || (!!manifest && (!Array.isArray(manifest.meta.frames) || (manifest.meta.frames as number[]).includes(f))));

  // 分割图、物体编号等数据自带「编号到名称」的表（包的 meta.classes）。其值为整数，
  // 按灰度绘制时几乎全黑（0,1,2,3 在 0..1 范围内均为黑色），
  // 因此数据声明自身为编号时默认使用「编号」着色；用户手动选择过则以用户选择为准（视图设置属于视图）。
  const classesOf = (m: typeof manifest) => (Array.isArray(m?.meta.classes) ? (m.meta.classes as unknown[]) : null);
  const topOf = (m: typeof manifest) => { const c = classesOf(m); return c ? Number((m?.meta.range as number[] | undefined)?.[1] ?? c.length) : 0; };
  // 着色只在单看一条通道时生效：多条通道一起查看时为彩色画面，颜色即数据本身，不得再叠加着色。
  // 工具栏在多通道时将「着色」置灰，但其值仍保留（用户上次选择的红色），因此不能直接向下传递。
  const tintFor = (m: typeof manifest, index: number | null, ch: number): Tint =>
    (!singleChannel(index, ch) ? "grey" : classesOf(m) && tint === DEFAULTS.tint ? "id" : tint);
  // 各侧的画面框（包的 data_window）：右侧回到其原本位置，半分辨率的数据按框缩放对齐；数据中已包含该信息，
  // 因此不提供对齐方式的开关
  const boxOf = (m: typeof manifest, w: number, h: number): Box => {
    const dw = m?.meta.data_window as number[] | undefined;
    return dw && dw.length === 4 && dw[2] > 0 && dw[3] > 0 ? [dw[0], dw[1], dw[2], dw[3]] : [0, 0, w, h];
  };
  // 左侧不调整黑白点、不着色（这些是右侧的工具）；右侧的第四条通道在「运算」档中作为运算强度，其他档中作为其自身的透明度
  const leftSide: Side = { index: lookIndex(leftIndex, channels), black: 0, white: 1, tint: "grey", alpha: "data" };
  const rightCh = overFp ? overChannels : channels;
  const rightIdx = lookIndex(rightIndex, rightCh);
  const rightSide: Side = { index: rightIdx,
                            black, white,
                            tint: tintFor(overFp ? overManifest : manifest, rightIdx, rightCh),
                            top: topOf(overFp ? overManifest : manifest),
                            alpha: mode === "over" ? "weight" : "data" };
  const mainSide: Side = mode === "result" ? rightSide : leftSide;
  const merge: Merge | null = overFp && overManifest
    ? { width, height, left: boxOf(manifest, width, height), right: boxOf(overManifest, width, height), op, mix: mixOf(op, mix) }
    : null;
  const lookKey = JSON.stringify([mainSide, overFp ? rightSide : null]);
  const mergeKey = JSON.stringify(merge);
  // 当前画面所需数据的来源完整定义在 view/stageSources.ts（所需通道、路径、
  // 本机文件能否直接使用、多个源如何进入同一账本）。此处只负责获取后如何绘制。
  const { picked, plate, mainHas, overHas, sideSource, mainNames, overNames, loadedId, used } =
    useStageSources({ plan, main, manifest, overFp, overManifest, mainSide, rightSide, partial, local, frame, playDir, shownPort: picture?.port ?? plan.mainPort });

  // 显卡用不了（本机开不出 WebGL2）或中途丢了（等恢复）时给出提示，不静默处理。正常情况下该通知不会出现
  const note = useStageNotes((n) => n.put);
  const gpu = useGpuState(); // 变了才换通知（不是每次渲染都重写）；恢复后重画（下方绘制的依赖里有它）
  useEffect(() => {
    note("looked", gpu === "broken" ? { text: textOf(msg("N-VIEW-NOGPU")), tip: textOf(msg("I-VIEW-NOGPUWHY")) }
      : gpu === "lost" ? { text: textOf(msg("N-VIEW-GPULOST")), tip: textOf(msg("I-VIEW-GPULOSTWHY")) } : null);
    return () => note("looked", null);
  }, [gpu]); // eslint-disable-line react-hooks/exhaustive-deps
  // 所需数据在服务器上不存在时给出提示，不静默回退为空白画布。
  // 走通道路径时尤为重要：一条通道 404 后画面将没有任何内容，若不提示，用户只能看到背景。
  const missing = [mainHas.failed && (mainNames.join("、") || "这一帧的画面"), overHas.failed && (overNames.join("、") || "右边那一路")]
    .filter(Boolean).join(" / ");
  useEffect(() => {
    note("error", missing ? { text: textOf(msg("E-VIEW-NOPIXELS", { what: missing })), tip: textOf(msg("E-VIEW-NOPIXELS", { what: missing })) } : null);
    return () => note("error", null);
  }, [missing]); // eslint-disable-line react-hooks/exhaustive-deps
  // 「运算」档一侧本来没有这一帧（transfer/readiness.ts PairHold）：说明缺的是哪一侧、哪一帧，只画另一侧
  const absentSide = merge ? (overHas.absent ? "右边" : mainHas.absent ? "左边" : "") : "";
  useEffect(() => {
    const said = absentSide ? { side: absentSide, frame } : null;
    note("absent", said ? { text: textOf(msg("N-VIEW-SIDEABSENT", said)), tip: textOf(msg("I-VIEW-SIDEABSENTWHY", said)) } : null);
    return () => note("absent", null);
  }, [absentSide, frame]); // eslint-disable-line react-hooks/exhaustive-deps
  // 本机文件已解码而服务器尚未提供尺寸时，使用该图像的尺寸（见上方 localSize 的注释）
  useEffect(() => {
    if (!manifest && !partial && plate.image) {
      const [w, h] = [plate.image.width, plate.image.height];
      setLocalSize((was) => (was && was.w === w && was.h === h ? was : { w, h }));
    }
  }, [manifest, partial, plate.image]);
  // 时间线跟随本机选择的帧（服务器读取完成之前）
  useEffect(() => {
    if (manifest || partial || !picked?.frames.length) return;
    const now = useViewer.getState();
    if (now.frames.join() !== picked.frames.join()) now.setFrames(picked.frames, now.frame);
  }, [manifest, partial, picked]);

  // 二维时时间线的「已载入视图」一行：浏览器持有的该包的每一帧，不论由视图还是预取取回（同一个缓存，platform/cache.ts），
  // 因此这一行随预取推进而填满；二维舞台卸载时清除（三维舞台设置自己的）。
  // 主源的 id 由 view/stageSources.ts 提供（并非包指纹：它还包含代理档位等信息，transfer/frameKey.ts）。
  // 走通道路径时主源即第一条通道，「该帧是否在浏览器中」即指绘制该帧所需的数据是否已就绪。
  const loaded = useLoadedFrames(loadedId);
  const ready = useDecodedFrames(used); // 在用的格按各自的帧都已解码的帧：播放器等它（transfer/readiness.ts readyAcross）
  const stale = plan.stale; // 该节点是否绘制的是上一次的结果（view/plan.ts；随节点而非随主源）
  useEffect(() => {
    reportLoads({ loaded, ready, waiting: null, stale });
  }, [loaded, ready, stale]);
  useEffect(() => clearLoads, []);
  // 当前画面的尺寸（图像像素）：面板控件需要在画面中放置内容时读取（state/viewTools.ts useStagePicture）。
  // 舞台卸载时清除，避免其他模块使用一个已不在屏幕上的尺寸
  useEffect(() => {
    useStagePicture.setState({ size: { width, height } });
  }, [width, height]);
  useEffect(() => () => useStagePicture.setState({ size: null }), []);


  const dpr = useDevicePixelRatio(); // 换到 DPR 不同的屏上：画布缓冲按新比例重建（下方绘制的依赖里有它）
  useLayoutEffect(() => {
    if (!el || !at) return;
    const { w: cw, h: ch } = nav.box; // 外层量到的尺寸：画布的显示尺寸和绘制缓冲同时设成它，二者永远同比例
    if (cw <= 0 || ch <= 0) return;
    // 尺寸或设备像素比变了才重设（重设会重建画布的后备缓冲）；没变只清空重画
    if (el.width !== Math.round(cw * dpr) || el.height !== Math.round(ch * dpr)) {
      el.style.width = `${cw}px`;
      el.style.height = `${ch}px`;
      el.width = Math.round(cw * dpr);
      el.height = Math.round(ch * dpr);
    }
    const ctx = el.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false; // 任何缩放下都用最近邻、不平滑，才能判断抠像边缘
    const { s } = at;
    // 「运算」档主源（左）本来没有这一帧、右侧有：右侧单独画（与右侧缺帧时左侧照画同一规则，transfer/readiness.ts PairHold）
    const leftAbsent = !!merge && mainHas.absent && overHas.ready;
    const on = (main || partial || picked) && (shown(frame) || leftAbsent);

    ctx.clearRect(0, 0, cw, ch);
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.6)";
    ctx.shadowBlur = 24;
    ctx.fillStyle = "#000";
    ctx.fillRect(at.x, at.y, width * s, height * s);
    ctx.restore();
    // 先绘制背景：画面第四条通道透明处显示的即为背景。三种模式下均存在，与运算完全解耦
    drawBackground(ctx, bg, bgColourOf(bgColor), at.x, at.y, width * s, height * s);
    // 显示结果在 GPU 上实时计算（view/look.ts）：（取通道或按范围映射）→ 黑白点 → 着色 → 与右侧合成，
    // 一次完成。不存储处理后的图像，因此切换着色、拖动黑白点不产生缓存项、不发请求，也不存在
    // 「无法获取时静默绘制原图」的情况。
    const mainManifest = mode === "result" ? (overFp ? overManifest : manifest) : manifest;
    const drawn = on && leftAbsent
      ? drawLook({
          width, height,
          drawWidth: Math.min(width, width * s * dpr), drawHeight: Math.min(height, height * s * dpr),
          left: sideSource(5, rightSide, boxOf(overManifest, width, height), overManifest),
          right: null,
          merge: null,
        })
      : on && mainHas.ready
      ? drawLook({
          width, height,
          // GL 画布按屏幕上的大小建（画面在屏幕上的像素 × 设备像素比），不按图像尺寸
          drawWidth: Math.min(width, width * s * dpr), drawHeight: Math.min(height, height * s * dpr),
          left: sideSource(0, mainSide, boxOf(mainManifest, width, height), mainManifest),
          // 右侧数据全部到达后才合成：只到达一部分就绘制，等于将不完整的数据当作结果展示
          // 两侧成对地换（view/stageSources.ts PairHold）：拼不成时两侧一起停在上一对；从没有过一对时右侧算加载中
          right: merge && overHas.ready ? sideSource(5, rightSide, boxOf(overManifest, width, height), overManifest) : null,
          merge,
        })
      : null;
    if (drawn) {
      ctx.drawImage(drawn, at.x, at.y, width * s, height * s);
    }
    // 无法创建 WebGL2 上下文的浏览器：绘制服务器提供的原图，并在统一通知区说明，不静默回退。
    // 通道路径在此类环境下无法绘制（一条通道不是一张图），此时不绘制任何内容，由通知区的提示说明原因。
    if (!drawn && on) {
      const px = plate.image;
      if (px && !isPlane(px)) ctx.drawImage(px, at.x, at.y, width * s, height * s);
    }
    // 「加载中」看两侧真在路上的格（transfer/frames.ts coming）：「运算」档右侧在取也算，右侧本来没有这一帧不算
    const waiting = plate.loading || (!!merge && overHas.loading);
    if ((main || partial) && shown(frame) && waiting) drawLoading(ctx, at, width, height, frame, !drawn);
    el.dataset.frame = plate.loading ? "" : String(plate.frame ?? "");
    const f: Frame = { ctx, frame, at, width, height, quiet: false, line };
    if (sourceBoxes && !hidden.has(sourceBoxesItem!.key)) drawBoxes({ ...f, quiet: true }, sourceBoxes, hover, chosen, byPerson);
    else if (boxes) drawBoxes(f, boxes, hover, null, byPerson);
    if (tracksShown) drawTracks(f, tracksShown);
    for (const h of plan.handles) if (h.stage === "2d") drawHandle(h, f, handleValues(h), drag, tracks);
  }, [frame, width, height, main?.fp, overFp, mainHas.ready, overHas.ready, picture?.fp, partial?.info.frames_done.length, overlays.map((o) => o.key).join(), boxes, sourceBoxes, tracks, hover, drag, plan, line, lookKey, mergeKey, channels, bg, bgColor, byPerson, plate.image, plate.loading, plate.window.join(), mainHas.images, overHas.images, el, at?.x, at?.y, at?.s, nav.box, gpu, dpr]); // eslint-disable-line react-hooks/exhaustive-deps

  // 写不了（ui/writeLock.ts useWriteLock）：手柄照画，点、拖、右键都不接
  const handle = locked ? null : plan.handles.find((h) => h.stage === "2d") ?? null;
  const pointAt = (e: { clientX: number; clientY: number }): Pt & { inside: boolean } => {
    const r = el!.getBoundingClientRect();
    const { x, y, s } = at ?? { x: 0, y: 0, s: 1 };
    const p = { x: (e.clientX - r.left - x) / s, y: (e.clientY - r.top - y) / s };
    return { ...p, inside: p.x >= 0 && p.y >= 0 && p.x <= width && p.y <= height };
  };
  const edit = (next: string[] | null) => handle && next && setParam(plan.node.id, Object.values(handle.params)[0], next);
  if (handle) {
    const values = handleValues(handle);
    rightClick = (e) => edit(removeAt(handle, values, frame, pointAt(e), at?.s ?? 1, sourceBoxes, tracks));
  }

  return (
    <div className="canvas2d-box" ref={setBox}>
    <canvas
      ref={setEl}
      className="canvas2d"
      style={{
        cursor: nav.cursor ?? (handle?.kind === "points" || handle?.kind === "box" || handle?.kind === "corners" || handle?.kind === "canvas" ? "crosshair" : handle && hover !== null ? "pointer" : "default"),
      }}
      onMouseMove={(e) => {
        nav.onMouseMove(e);
        const p = pointAt(e);
        if (!drag && handle?.kind === "person") setHover(personAt(sourceBoxes, frame, p));
      }}
      onMouseLeave={() => {
        nav.onMouseLeave();
        setHover(null);
      }}
      // 每次按下都捕获指针直到松开（pointer capture；鼠标事件随之而来）：在画布外松开的拖动仍在这里结束，
      // 手柄不会粘在指针上
      onPointerDown={(e) => e.currentTarget.setPointerCapture(e.pointerId)}
      onMouseDown={(e) => {
        if (nav.onMouseDown(e)) return; // 中键拖动或 Alt + 左键拖动平移，右键拖动缩放（Houdini 式），不交给手柄
        if (e.button !== 0 || !handle) return;
        // 一次按下是点还是拖，由 platform/drag.ts followPress 一处判定：点 = 在按下处加（clickHandle），拖 = 挪 / 画 / 拉框
        // （dragHandle）；只点不拖不挪任何东西
        const p0 = pointAt(e);
        const values = handleValues(handle);
        const shift = e.shiftKey;
        let cur = dragStart(handle, values, frame, p0, at?.s ?? 1);
        followPress(
          e,
          (ev) => {
            if (!cur) return;
            cur = moveDrag(handle, cur, pointAt(ev));
            setDrag(cur);
          },
          (how, ev) => {
            setDrag(null);
            if (how === "drag" && cur && ev) edit(dragHandle(handle, values, frame, { ...cur, to: pointAt(ev) }));
            else if (how === "click" && p0.inside) edit(clickHandle(handle, values, frame, p0, shift ? 1 : pointLabel, sourceBoxes));
          },
        );
      }}
      onContextMenu={nav.onContextMenu}
    />
    </div>
  );
}
