import { useMemo, useRef } from "react";
import type { Manifest } from "../api";
import { channelFrames, localFirst, pickedFrames, serverFrames, stillHeld, tierOf, useFrames } from "../transfer/frames";
import { isPlane, type Pixels } from "../transfer/plane";
import { PairHold, SideHold } from "../transfer/readiness";
import { channelsFor, validFor } from "../transfer/route";
import { useViewer } from "../state/viewer";
import { partialFrames, type Partial as PartialResult } from "./partial";
import type { LocalPicture } from "./localPick";
import type { Box, Side, SideSource } from "./look";
import type { DisplayPlan } from "./plan";
import { fileStandsFor } from "./origin";
import { useGens } from "../transfer/gens";

/** 当前画面所需数据的来源：所需通道、传输路径、本机文件能否直接使用、
 * 十个源如何一起进入取帧账本。`view/Stage2D.tsx` 只负责获取后如何绘制。 */

interface StageSourcesAsk {
  plan: DisplayPlan;
  main: { fp: string; type: string } | null;
  manifest: Manifest | null;
  overFp: string | null;
  shownPort: string; // 当前查看的层（端口名）：尚无任何层计算过时，据此判断能否以本机文件代替

  overManifest: Manifest | null;
  mainSide: Side;
  rightSide: Side;
  partial?: { info: PartialResult; job: string; node: string; port: string } | null;
  local?: LocalPicture | null;
  frame: number;
  playDir: number;
}

export function useStageSources(ask: StageSourcesAsk) {
  const { plan, main, manifest, overFp, overManifest, mainSide, rightSide, partial, local, frame, playDir, shownPort } = ask;
  // 画面的当前帧，前后帧同时载入（transfer/frames.ts：不黑屏，下一帧载入期间保留上一帧）；节点计算中时，
  // 取它已写出的帧（view/partial.ts），经任务自身的地址
  // 本机文件优先（浏览器已持有的数据不重复请求）：
  // 用户在本标签页中选择的文件，帧到文件的对应关系由浏览器自行建立，不等待服务器的包说明；
  // PNG / JPEG 始终使用本机文件（服务器提供的显示图即为该文件本身），EXR 只在服务器数据到达前使用
  // （服务器须按 OCIO 执行显示变换，以所见即所得为优先）。
  const timeline = useViewer((s) => s.frames);
  const still = !!local && local.item.kind !== "sequence";
  const picked = useMemo(
    () => (local ? pickedFrames(plan.node.id, local.item, local.space, still ? timeline : []) : null),
    [local, plan.node.id, still ? timeline : null], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // 只有绘制的是原图时才使用本机文件：本机文件是用户选择的画面，而非节点计算的结果。
  // 若无此限制，在「仅结果」档查看遮罩时会绘制出原图。
  //
  // 节点尚无任何层计算过（`main` 为 null）时，按所选是否为主画面层判断：
  // 多层 EXR 的读取节点不自动计算，每层的 `fp` 均为空、`main` 也为 null；若无条件使用本机文件，
  // 选择「depth」时绘制出的是本机文件解码得到的彩色图，而标签显示为 depth，且没有任何提示。
  // 浏览器只从本机 EXR 中解出其颜色通道（`transfer/exr/decode.ts`，经本机代理或 `transfer/exr.ts`），
  // 不提供其他层，因此只有查看主画面层时才使用本机文件。
  const onPlate = main ? !plan.plate || main.fp === plan.plate : fileStandsFor(plan.filePort, shownPort);
  const exact = onPlate && !!local && !local.item.files.some((f) => f.name.toLowerCase().endsWith(".exr"));

  // ---------------------------------------------------------------- 该侧所需的通道及其传输方式
  //
  // 仅有两条路径：
  //   图片路径：彩色画面（整体查看，或单看其中一条颜色通道 R / G / B）。该图是 8 位有损 WebP 代理，
  //     是三条通道的最小载体。
  //   通道路径：数值图（alpha、遮罩、深度、编号，单看一条或整体查看）与视频的单条通道
  //     （数值图不经过色彩管理，每条通道一种颜色，相互独立，不需要三维查找表）。判定见 transfer/route.ts。按需发送所需通道，
  //     值为数据自身的值，范围映射、黑白点、着色、合成均在浏览器中计算（view/look.ts）。
  //
  // 无论解算器输出多少条通道，只传输用户查看的通道；客户端已有的 rgb 不重复发送。
  // 数值图走通道路径，浏览器持有全精度的值，拖动黑白点时即时计算，无需再向服务器请求图像。
  const framesOf = (m: Manifest | null) => (Array.isArray(m?.meta.frames) ? (m!.meta.frames as number[]) : []);
  const rangeOf = (m: Manifest | null): readonly [number, number] => {
    const r = m?.meta.range as number[] | undefined;
    return m?.meta.values && r && r.length === 2 ? [r[0], r[1]] : [0, 1];
  };
  // 边算边看使用任务自身的地址（view/partial.ts），不走上述两条路径
  const mainNames = partial ? [] : channelsFor(manifest, mainSide.index);
  const overNames = overFp ? channelsFor(overManifest, rightSide.index) : [];
  // 代理档位（管理员设置，包说明中的 `proxy.px`）：计入源的 id，从而计入缓存键。
  // 键须包含「数据包 · 帧 · 通道 · 代理档位」四项，切换离开再切回时才能命中缓存
  // 显示档位随包说明（同一份包说明同一个值，下面的帧源不因每次渲染重建）
  const tier = useMemo(() => tierOf(manifest), [manifest]);
  const overTier = useMemo(() => tierOf(overManifest), [overManifest]);
  // 生成号计入 memo 依赖（transfer/gens.ts）：同一指纹重算后帧源重建，键中含新生成号，旧帧不会再命中
  const mainGen = useGens((s) => (main ? s.gens[main.fp] ?? "" : ""));
  const overGen = useGens((s) => (overFp ? s.gens[overFp] ?? "" : ""));


  // 两侧的源（固定十格，`null` 表示本轮不需要该格）：图片路径占一格，通道路径最多三格加一格 valid。
  // 十格均使用同一账本（transfer/frames.ts useFrames）：同一个窗口、同一份预算、同一次取消，
  // 否则播放时部分在传输、部分被丢弃，各帧画面将不一致。
  const mainPicture = useMemo(() => {
    if (partial) return partialFrames(partial.job, partial.node, partial.port, partial.info.frames_done);
    if (mainNames.length) return null; // 该侧走通道路径：不请求图片
    const server = main && manifest ? serverFrames(main, manifest) : null;
    // 尚未计算时画面的来源为本标签页中选择的文件（`picked`）。
    // 此处不从已登记的本机原件创建帧源：登记表（`transfer/originals.ts`）的键为包指纹，
    // 未计算则没有指纹，`main` 也为 null。刷新后的本机原件经由 `serverFrames` 中逐帧替换字节的路径
    // （须先计算一次）。
    return localFirst(server, onPlate ? picked : null, exact);
    // `plan.sourceKey`：当前画面各数据来源合成的身份（`view/origin.ts`；其中包含本机原件的查找结果）。
    // 查找原件是异步的，若无此项，登记完成后该格仍为服务器数据，须切换离开再切回才会更新
  }, [partial?.job, partial?.node, partial?.port, partial?.info.frames_done.length, mainNames.join(), main?.fp, main?.type, manifest, tier, picked, onPlate, exact, plan.sourceKey, mainGen]); // eslint-disable-line react-hooks/exhaustive-deps
  const mainPlanes = useMemo(
    () => (main ? mainNames.map((n) => channelFrames(main.fp, n, framesOf(manifest), tier)) : []),
    [main?.fp, mainNames.join(), manifest, tier, mainGen], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const mainValid = useMemo(() => {
    const n = main && validFor(manifest, mainNames);
    return n ? channelFrames(main!.fp, n, framesOf(manifest), tier) : null;
  }, [main?.fp, mainNames.join(), manifest, tier, mainGen]); // eslint-disable-line react-hooks/exhaustive-deps
  // 「运算」档右侧一路同样使用取帧账本：直接按地址加载会绕过窗口、预算和取消
  // （播放时它总是显示为「已到达」，且会一次请求整段）。
  const overPicture = useMemo(
    () => (overFp && overManifest && !overNames.length ? serverFrames({ fp: overFp, type: "" }, overManifest) : null),
    [overFp, overManifest, overNames.join(), overTier, plan.sourceKey, overGen], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const overPlanes = useMemo(
    () => (overFp ? overNames.map((n) => channelFrames(overFp, n, framesOf(overManifest), overTier)) : []),
    [overFp, overNames.join(), overManifest, overTier, overGen], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const overValid = useMemo(() => {
    const n = overFp && validFor(overManifest, overNames);
    return n ? channelFrames(overFp!, n, framesOf(overManifest), overTier) : null;
  }, [overFp, overNames.join(), overManifest, overTier, overGen]); // eslint-disable-line react-hooks/exhaustive-deps

  // 十格均使用同一账本（`useFrames`）：已解码保留的帧数取决于预算，
  // 整段的压缩字节在选中后即开始取回（transfer/fill.ts fillWhole），不等待播放
  const sources = [
    mainPicture, mainPlanes[0] ?? null, mainPlanes[1] ?? null, mainPlanes[2] ?? null, mainValid,
    overPicture, overPlanes[0] ?? null, overPlanes[1] ?? null, overPlanes[2] ?? null, overValid,
  ];
  const scrubbing = useViewer((s) => s.scrubbing);
  const got = useFrames(sources, frame, playDir, scrubbing);
  // 单侧：要用的格都到了这一帧才换（transfer/readiness.ts）；没到齐时整侧沿用上一次完整合成的那一帧，
  // 不把各格停在不同帧的图拼在一起
  const mainHold = useRef(new SideHold<Pixels>()).current;
  const overHold = useRef(new SideHold<Pixels>()).current;
  const gather = (from: number, hold: SideHold<Pixels>) => {
    const on = [0, 1, 2, 3, 4].map((k) => from + k).filter((i) => sources[i]);
    const need = on.filter((i) => i !== from + 4).map((i) => i - from); // valid 没到不挡绘制（它只把无值处涂黑）
    const cells = [0, 1, 2, 3, 4].map((k) => got[from + k]);
    const side = hold.take(on.map((i) => sources[i]!.id).join("\n"), frame, cells, need,
      (k, f, image) => { const s = sources[from + k]; return !!s && stillHeld(s, f, image); });
    return {
      ...side,
      loading: on.some((i) => got[i].loading),
      absent: !!side.absent,
      failed: on.some((i) => got[i].failed),
      primary: on[0] ?? -1,
    };
  };
  const mainTake = gather(0, mainHold);
  const overTake = gather(5, overHold);
  // 两侧（「运算」档）：成对地换（transfer/readiness.ts PairHold）；拼不成时两侧一起停在上一对。一侧本来没有这一帧时
  // 另一侧照画当前帧，缺的一侧不画（Stage2D 说明）；「加载中」只看真在路上的格
  const pairHold = useRef(new PairHold<Pixels>()).current;
  const twoSided = sources.slice(5).some(Boolean);
  const heldSide = (side: 0 | 1, s: { images: (Pixels | null)[]; frame: number | null }) =>
    s.frame !== null && s.images.every((p, k) => { const src = sources[side * 5 + k]; return p === null || (!!src && stillHeld(src, s.frame!, p)); });
  const pair = twoSided ? pairHold.take(sources.map((s) => s?.id ?? "").join("\n"), mainTake, overTake, heldSide) : null;
  const mainHas = pair?.ready ? { ...mainTake, ...pair.a } : mainTake; // 没有一对（还没到，或右侧缺这一帧）：左侧照画自己的
  // 一侧本来没有这一帧：另一侧照它自己的画（两边同一规则：左侧缺时右侧单独画，右侧缺时左侧照画）
  const overHas = pair ? { ...overTake, ...pair.b, ready: pair.ready || (mainTake.absent && overTake.ready) } : overTake;
  const plate = mainHas.primary < 0 ? got[0] : got[mainHas.primary];
  const cellOf = (i: number) => (i < 5 ? mainHas.images[i] : overHas.images[i - 5]);
  const asPicture = (i: number) => { const px = sources[i] ? cellOf(i) : null; return px && !isPlane(px) ? px : null; };
  const asPlane = (i: number) => { const px = sources[i] ? cellOf(i) : null; return px && isPlane(px) ? px : null; };
  const sideSource = (from: number, look: Side, box: Box, m: Manifest | null): SideSource => ({
    box, side: look,
    picture: asPicture(from),
    planes: [asPlane(from + 1), asPlane(from + 2), asPlane(from + 3)],
    valid: asPlane(from + 4),
    range: rangeOf(m),
  });

  // 时间线的「已载入视图」跟随主源所在的格（而非包指纹：id 中还含有通道名和代理档位）
  const loadedId = sources[mainHas.primary]?.id ?? null;
  // 播放器要等的：所有在用的格（两侧、各条通道），每一格都解码了才算这一帧就绪
  const used = sources.flatMap((x) => (x ? [x] : []));
  return { picked, plate, mainHas, overHas, sideSource, mainNames, overNames, loadedId, used };
}
