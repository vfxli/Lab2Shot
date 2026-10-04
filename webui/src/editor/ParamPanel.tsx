/** 参数面板（工作区右栏）归本模块：选中节点时列出它的参数行（标签、三个标记、控件、来源与置灰原因）与标题信息；
 * 没选中节点或处于应用模式时画节点图自己的参数界面树（exposed）。控件本身的分发在 editor/ParamControls.tsx。 */

import { packetOf } from "../state/results";
import type { Availability } from "../api/applies";
import { licensedValues, registeredValues } from "../graph/rules";
import { optionView } from "../ui/controls";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { nodeCategory, type NodeTypeDef, type ParamDef, type SceneKind, type WritesPart } from "../api";
import { insertNode as insertNodeAction, setInterface, setParam as setParamAction, setParamsAcross, toggleExposed as toggleExposedAction, toggleOnNode as toggleOnNodeAction, togglePromoted as togglePromotedAction } from "../graph/actions";
import { useGraphDoc } from "../graph/snapshot";
import { getTypes, useCatalog } from "../state/catalog";
import { drives, entryLabel, exposedParams, firstTarget, isGroup, readOnlyWhy, targetsOf, useCookInputs } from "../state/cookInputs";
import { useWriteLock } from "../ui/writeLock";
import type { ExposedEntry, ExposedGroup, ExposedParam } from "../api";
import { conditionNames, parseCondition, condEqual, optionDisabled, shownOptions } from "../platform/conditions";
import { Select } from "../ui/Select";
import { Switch } from "../ui/Button";
import { SheetWindow } from "./ParamSheet";
import { IconButton } from "../ui/Button";
import { ParamInterface, INTERFACE_PARAM } from "./ParamInterfaceEditor";
import { useAppMode } from "./AppMode";
import { enterPicking } from "../state/viewPicking";
import { useRowDrag } from "../ui/rowDrag";
import { canDropAt, dropAt, entryDisabled, entryHidden, exposedValues, rowsOf, type Path } from "../graph/exposedTree";
import { groupSwitches, switchesState, switchesTo, type SwitchKind } from "../model/groupSwitches";
import { isBlocked, skippedBranches } from "../model/nodeOutcome";
import { useResults, useShownResults, useShownTarget } from "../state/results";
import { useViewer } from "../state/viewer";
import { chosenOnNode, nodeMessages, portColor, upstream, writesTable } from "../graph/nodes";
import { textOf } from "../messages/message";
import type { NodeStatus } from "../api/status";
import { commercialOf, costOf, wiredFrom } from "../graph/rules";
import { why, nodeUsable, nodeWhy } from "../api/applies";
import { Button } from "../ui/Button";
import { PasteFrom } from "./PasteFrom";
import { Control, fitWidth, opensSheet } from "./ParamControls";
import { IconOnNode, IconSliders } from "../ui/icons";
import { multiline } from "../ui/controls";
import { LabelGrid, LabelRow } from "../ui/LabelRow";
import { ParamCurve } from "./CurvesView";
import { useSession } from "../state/session";
import { webAddress } from "../platform/util";
import { NodeCommentField, NodeNameInput } from "./NodeNaming";
import { pick, t } from "../i18n/t";
import { listSep } from "../i18n/words";
import { tipOf } from "../platform/tips";

/** 一个参数：标签连同其公开标记与（可由连线驱动的参数才有的）「提升到节点」标记、控件；节点的连接或设置使其不起作用时
 * 置灰并说明原因（由服务器的图状态判定）。由连线驱动时控件置灰，并说明取值来源（"38.6 mm · 来自 AnyCalib 镜头标定 · Focal Length"）；
 * `said`：服务器状态给出的节点取值来源（Focal Length 覆盖相机的值，或取相机的值）。
 * `onNode`：节点本体上是否也显示该行（简单参数），以及切换它的点击。在节点上点击的参数会在此处滚动到可见并闪烁
 * （双击时聚焦其输入框）。
 *
 * 标签后的三个标记各司其职：对外参数（ExposePin）、提升到节点（PromotePin：增加一个输入口）、
 * 在节点上显示（OnNodePin：仅显示，不提供输入口）。
 *
 * 参数面板不显示悬停提示（整块面板带 data-no-tips，platform/tips.ts）。 */
function ParamRow({ nodeId, p, label, value, set, why, pinned, onPin, socket, said, over, onNode, reason, control, bare, quiet, notes, pending, note: authored }: {
  nodeId: string; p: ParamDef; label: string; value: unknown; set: (v: unknown) => void; why?: string; pinned: boolean; onPin: () => void;
  socket?: { on: boolean; fixed?: boolean; onClick: () => void }; said?: string; onNode?: { on: boolean; onClick: () => void };
  // the server's clause when the value is set over a connected input (status `overrides`), shown on its own line
  over?: string;
  // 参数界面（exposed 树）给这一项的：`reason` 条件为假时置灰的原因（「由『X』决定」），`control` 覆盖显示的控件（下拉 /
  // 复选框），`bare` 应用模式：不画三个标记（公开、提升、在节点上显示都是做模板的操作）
  reason?: string; control?: React.ReactNode; bare?: boolean;
  // 应用模式按钮下的安静一行（小号灰字，不置灰按钮）：这一步被开关关着（model/nodeOutcome.ts skippedBranches）
  quiet?: string;
  // 应用模式按钮下：这一步算出来时说的警告和说明（ExposedRow cookNotes），各一行
  notes?: { level: string; text: string }[];
  // 显示的是上一次回复、新回复还没到（state/results.ts Shown `pending`）：来源、接线值照写，淡色
  pending?: boolean;
  // 模板作者给这一项写的说明（含义、许可后果、限制）：行下常显（ui/LabelRow.tsx note），不是悬停
  note?: string;
}) {
  why = why || reason;
  // 只读标签页（editor/tabs.ts：这张图正在另一个标签页里编辑）：控件和三个标记真正禁用（disabled），键盘 Tab 进去也改不了；
  // 样式上的置灰在 styles/21-tooltip.css，只管外观
  const readOnly = !!useWriteLock();
  const snap = useGraphDoc();
  const wired = socket?.on ? wiredFrom({ ...snap, results: snap.shown }, nodeId, p.name) : null;
  // 接线时说明来自哪里；否则为某个输入提供该值时节点给出的说明（「Focal Length · 来自相机（ViPE 相机解算）」）。
  // 应用模式不说（说的是节点和线：卡片上没有节点）
  const note = bare ? "" : wired ? `← ${wired.from}${wired.value ? ` · ${wired.value}` : ""}` : said && !over ? said : "";
  const reveal = useViewer((s) => (s.reveal?.node === nodeId && s.reveal.param === p.name ? s.reveal : null));
  const row = useRef<HTMLDivElement>(null);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (!reveal || !row.current) return;
    // 等面板排好刚切换到的内容（另一个节点、该行上方很高的表格）后再滚动：同一时刻测量的滚动位置
    // 会因上方仍在增长的高度而停在该行之前。
    const el = row.current;
    let raf = requestAnimationFrame(() => {
      raf = requestAnimationFrame(() => {
        el.scrollIntoView({ block: "center", behavior: "smooth" });
        // 多行参数使用 <textarea> 而非 <input>：若遗漏此类型，在节点上双击「提示词」时无法聚焦面板中的输入框
        if (reveal.focus) el.querySelector<HTMLElement>(".ctl input, .ctl textarea, .ctl select, .ctl button")?.focus({ preventScroll: true });
      });
    });
    setFlash(true);
    const t = setTimeout(() => setFlash(false), 1300);
    return () => (cancelAnimationFrame(raf), clearTimeout(t));
  }, [reveal?.n]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    // 一行 = 通用的「标签 + 控件」行（ui/LabelRow.tsx）：三个标记、标签、控件各占一格，标签放不下省略号（悬停看全），
    // 从不压到标记或控件上；下面的说明从控件那一列起
    <LabelRow ref={row} className={`prow${multiline(p) || p.widget === "table" ? " multi top" : ""}${why ? " inactive" : ""}${wired && !why ? " wired" : ""}${flash ? " flash" : ""}${pending ? " pending" : ""}`} data-param={p.name}
      labelClass="plabel-text" label={label} note={authored}
      marks={bare ? undefined : (
        <fieldset className="plabel-marks" disabled={readOnly}>
          <ExposePin on={pinned} onClick={onPin} />
          {/* 三个标记各有固定的一格（没有的那一格空着），各行的标记上下对齐 */}
          {socket ? <PromotePin on={socket.on} wired={!!wired} fixed={socket.fixed} onClick={socket.onClick} />
            // 按钮参数也有三个标记：提升到节点对它没有意义（没有值可接），置灰
            : p.widget === "button" ? <PromotePin on={false} wired={false} fixed refused={t("ui.params.button_no_promote")} onClick={() => {}} />
            : <span className="pin-slot" aria-hidden />}
          {onNode ? <OnNodePin on={onNode.on || !!socket?.on} promoted={!!socket?.on} onClick={onNode.onClick} /> : <span className="pin-slot" aria-hidden />}
        </fieldset>
      )}
      below={<>
        {reason && <div className="pwhy lrow-under">{reason}</div>}
        {/* 值从哪里来、覆盖了什么，由下方的小号灰字说明 */}
        {!why && !wired && note && <div className="pwhy lrow-under">{note}</div>}
        {/* 接进来的口可能没有值（素材没记帧率）：这里填的值照样可改，素材有值用素材的，否则用这里填的。模板作者给这一项
            写了说明（note）就不再重复这一句；应用模式按 .app 说（不提节点名：卡片上没有节点，{from} 不用） */}
        {!why && wired?.fallback && !authored && (
          <div className="pwhy lrow-under">{t("ui.params.wired_fallback", { from: wired.value ? `${wired.from} · ${wired.value}` : wired.from })}</div>
        )}
        {!why && over && <div className="pwhy pover lrow-under">{over}</div>}
        {quiet && <div className="pwhy lrow-under">{quiet}</div>}
        {notes?.map((m) => <div key={m.text} className={`pwhy pnote-${m.level} lrow-under`}>{m.text}</div>)}
      </>}>
      {wired && !why && !wired.fallback ? (
        // 由连线驱动：以连线自身颜色的一块说明取值及其来源，而不是一个无法输入的输入框。应用模式（卡片上没有节点和线）
        // 只写值，还没算出时说「跟随输入」，用中性色（不是线的颜色，也不像出错）
        <span className={`ctl pwired${bare ? " app" : ""}`}>
          <b data-user-data>{wired.value || t("ui.params.wired_pending")}</b>
          {!bare && <small data-user-data>{t("ui.params.wired_from", { from: wired.from })}</small>}
        </span>
      ) : (
        // 打开窗的参数（「对应关系」「层级」）不放进禁用的组：打开窗是看，窗里按 lock 只锁写（editor/ParamSheet.tsx）
        <fieldset className="ctl" disabled={(!!why || readOnly) && !(!control && opensSheet(p))}>
          {control ?? <Control nodeId={nodeId} p={p} value={value} set={set} lock={readOnly ? readOnlyWhy() : why || ""} />}
        </fieldset>
      )}
    </LabelRow>
  );
}

/** 提升到节点（相当于 Houdini 的通道引用、ComfyUI 中把控件转为输入）：点一下把该参数放到节点本体上，成为一个独立的
 * 输入口（`param:<name>`），同一行带有可直接编辑的控件；再点一下则连同其连线一起去掉该行。接入连线后节点上的控件置灰，取值以连线传入的值为准
 * （editor/NodeParamRow.tsx）。 */
function PromotePin({ on, wired, fixed, refused, onClick }: { on: boolean; wired: boolean; fixed?: boolean; refused?: string; onClick: () => void }) {
  return (
    <button className={`socket-pin${on ? " on" : ""}${wired ? " wired" : ""}${fixed ? " fixed" : ""}${refused ? " refused" : ""}`}
      aria-label={refused ?? t(on ? "ui.params.unpromote" : "ui.params.promote")} aria-pressed={on} aria-disabled={fixed} onClick={fixed ? undefined : onClick}>
      <svg width={11} height={11} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.7}>
        <circle cx="10" cy="8" r="3.6" fill={on ? "currentColor" : "none"} />
        <path d="M1.5 8h5" strokeLinecap="round" />
      </svg>
    </button>
  );
}

/** 在节点上显示（NodeDef.on_node 是节点作者给定的默认值，该标记对应使用者自己的名单：graph/edit.ts toggleOnNode）：
 * 在节点上增加一行并可直接编辑，但不提供输入口；需要输入口时使用相邻的「提升到节点」（PromotePin）。
 * 提升后的参数本就显示在节点上（其输入口位于该行），因此该标记对其始终点亮且不可取消。 */
function OnNodePin({ on, promoted, onClick }: { on: boolean; promoted: boolean; onClick: () => void }) {
  return (
    <button className={`onnode-pin${on ? " on" : ""}${promoted ? " fixed" : ""}`} aria-label={t("ui.params.on_node")} aria-pressed={on} aria-disabled={promoted}
      onClick={promoted ? undefined : onClick}>
      <IconOnNode size={11} filled={on} />
    </button>
  );
}

const NO_KINDS: SceneKind[] = [];
const WRITES_MARK: Record<WritesPart["how"], string> = { full: "✓", static: "ui.params.writes_static", no: "✗" };
// 可以写出但部分内容无法保留（如 FBX 无法保留模型网格的分区）：标注「部分」。
// 宽度与「静止」相同，位置不跳动；格式丢失的内容必须对使用者可见
const WRITES_PARTLY = "ui.params.writes_partly";

/** The compute tier the server rates a node with (lab2shot/nodes/compute.py TIERS), in words; a tier this table does not
 * know is shown as the server wrote it. */
const TIER_KEY: Record<string, string> = {
  low: "ui.params.tier.low",
  medium: "ui.params.tier.medium",
  high: "ui.params.tier.high",
  very_high: "ui.params.tier.very_high",
};
const tierWord = (tier: string): string => (TIER_KEY[tier] ? t(TIER_KEY[tier]) : tier);

/** 支持的数据：三维输出设置节点的格式对每种三维数据能保留多少，来自其声明（OutputSettings.writes）：
 * 每种数据以其类型的颜色标出，✓ / 静止 / ✗ / 部分。 */
function WritesGroup({ def }: { def: NodeTypeDef }) {
  const catalog = useCatalog();
  const kinds = catalog?.scene_kinds ?? NO_KINDS;
  const types = getTypes();
  return (
    <div className="group">
      <div className="group-title">{t("ui.params.writes")}</div>
      <div className="writes">
        {writesTable(kinds, def).map((r) => (
          <span key={r.kind.id} className={`writes-item ${r.how}${r.lost ? " lost" : ""}`}>
            <i style={{ background: portColor(types, r.kind.type) }} />
            {r.kind.label}
            <b>{r.lost ? t(WRITES_PARTLY) : WRITES_MARK[r.how].startsWith("ui.") ? t(WRITES_MARK[r.how]) : WRITES_MARK[r.how]}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

/** 把参数公开到节点图的标记：模板会把它放在前面，lab2shot cook 与 DCC 插件可以设置它。 */
function ExposePin({ on, onClick }: { on: boolean; onClick: () => void }) {
  return (
    <button className={`expose-pin${on ? " on" : ""}`} aria-label={t("ui.params.expose")} aria-pressed={on} onClick={onClick}>
      <svg width={11} height={11} viewBox="0 0 16 16" fill={on ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.6} strokeLinejoin="round">
        <path d="M5.5 2.5h5l-.8 4 2.3 2.2H4l2.3-2.2z" />
        <path d="M8 8.7v4.8" strokeLinecap="round" />
      </svg>
    </button>
  );
}


// 面板不设选项卡，全部为参数；节点的数据信息位于节点右下角 ⓘ 的卡片中（NodeInfoCard）。

/** 该节点中由逐帧取值的连线驱动的参数，以及读取曲线所用的结果：只有帧间有变化的数值值得画曲线，服务器按事实列在
 * status 的 `curves`（data/values.py varies），不看文字。 */
function perFrameCurves(snap: ReturnType<typeof useGraphDoc>, id: string): { param: string; fp: string; title: string }[] {
  const out: { param: string; fp: string; title: string }[] = [];
  for (const e of snap.edges) {
    const param = e.target === id ? (e.targetHandle ?? "").match(/^param:(.+)$/)?.[1] : undefined;
    if (!param) continue;
    const port = e.sourceHandle ?? "";
    const status = snap.shown[e.source];
    const fp = packetOf(status, port) ?? ""; // only a port that has a packet (state/results.ts packetOf)
    if (!fp || !status?.curves?.includes(port)) continue;
    const def = snap.nodeDefs[snap.nodes.find((n) => n.id === id)?.data.typeId ?? ""];
    const label = def?.params.find((q) => q.name === param)?.label ?? param;
    out.push({ param, fp, title: label });
  }
  return out;
}

export function ParamPanel() {
  const applies = useSession((s) => s.state?.applies);
  // 聚焦模式（editor/AppMode.tsx）：面板一直是聚焦的那个节点的（它自己的参数行、「在视图里编辑」等手柄），不看选中了谁
  const focused = useAppMode((s) => (s.mode === "focus" ? s.focus?.node ?? null : null));
  const picked = useViewer((s) => s.selectedId);
  const selectedId = focused ?? picked;
  const snap = useGraphDoc();
  const node = snap.nodes.find((n) => n.id === selectedId);
  const def = node ? snap.nodeDefs[node.data.typeId] : undefined;
  const catalog = useCatalog();
  const cat = nodeCategory(catalog, def);
  const setParam = setParamAction;
  const meta = useCookInputs((s) => s.meta);
  const exposed = useCookInputs((s) => s.exposed);
  const appMode = useAppMode((s) => s.mode === "app");
  const viewerRole = !!useWriteLock();
  // 参数界面（分组、顺序、显示名、下拉、Hide / Disable When）谁都能编辑：改的是自己手上这张节点图；能存到哪里才分权限
  // （预设模板要「管理模板」，自己的存「我的模板」或 json）。应用模式和只读标签页不能改
  // 只读标签页也能打开来看（窗里只锁写，ParamSheet.tsx）；面板里的参数树照样只在能改时可拖
  const interfaceRight = !appMode;
  const [interfaceOpen, setInterfaceOpen] = useState(false);
  const nodes = snap.nodes;
  const nodeDefs = snap.nodeDefs;
  const toggleExposed = toggleExposedAction;
  const togglePromoted = togglePromotedAction;
  const toggleOnNode = toggleOnNodeAction;
  const { results, pending } = useShownResults();
  // 这一次打开的是哪份文档（每次 loadGraph 换一个 docId，同一文件再打开也换；「另存为」只换 graphId、不换它）：参数界面树按它重新挂上
  const loaded = useViewer((s) => s.docId);
  const insertNode = insertNodeAction;
  const setInspectorFit = useViewer((s) => s.setInspectorFit);
  const body = useRef<HTMLDivElement>(null);
  const commercial = commercialOf(snap, selectedId ?? "");
  const cost = costOf(snap, selectedId ?? "");
  // 服务器按该节点设置解析出的许可证，用目录自身的措辞表述
  const licence = (selectedId && results[selectedId]?.licence) || def?.at_defaults.licence;
  const licenceTags = (licence?.tags ?? []).filter((id) => !catalog?.tags[id]?.implied).map((id) => ({ id, label: catalog?.tags[id]?.label ?? id }));
  const params = node?.data.params;
  // 依赖中必须包含 `exposed`：未选中节点时面板只显示「对外参数」各行，标签来自 `exposed`。它变化时（打开
  // 模板卡、增减对外参数）标签随之改变，若不重新计算列宽，列宽会停留在 `fitWidth` 的下限 112px，长标签
  // （如「已知 Focal Length」加三个标记约 155px）会与标记重叠。
  const refit = () => body.current && setInspectorFit(fitWidth(body.current));
  useLayoutEffect(() => {
    if (body.current) setInspectorFit(fitWidth(body.current));
  }, [def, params, exposed, appMode, setInspectorFit]);

  // 「在节点上显示」标记及其点击行为（仅可显示在节点上的参数具备：p.simple）
  const onNodeOf = (nodeId: string, p: ParamDef) => {
    const n = nodes.find((m) => m.id === nodeId);
    const nd = n && nodeDefs[n.data.typeId];
    return p.simple && n && nd ? { on: chosenOnNode(nd, n.data.onNode).includes(p.name), onClick: () => toggleOnNode(nodeId, p.name) } : undefined;
  };

  // 参数界面里一项在面板左边写的标签（按钮项不写：名字在按钮上），面板的标签列按全部这些定宽（LabelGrid labels）
  const rowLabel = (x: ExposedParam): string => {
    const [nid, pname] = firstTarget(x);
    const n = nodes.find((m) => m.id === nid);
    return n && nodeDefs[n.data.typeId]?.params.find((q) => q.name === pname)?.widget === "button" ? "" : entryLabel(x);
  };

  // 没选中节点（节点模式）或应用模式（editor/AppMode.tsx，不看选中了哪个节点）：面板就是节点图自己——标题是 meta.name，
  // 下面是它的参数界面树（分组、顺序、显示方式、Hide / Disable When，InterfaceTree）。两种模式同一份渲染，只差：应用模式
  // 不画三个标记、不能拖、没有「编辑参数界面」入口（「计算」「下载」也是树里的按钮参数，模板作者公开进来、放在他想放的
  // 位置，没有任何写死的块），没有公开参数时说切回节点模式；节点模式末尾提示选中节点。
  if (focused && (!node || !def)) {
    return (
      <div className="inspector" data-no-tips>
        <div className="insp-body">
          <div className="empty" style={{ height: "auto", paddingTop: 28 }}>{t("ui.params.focused_none")}</div>
        </div>
      </div>
    );
  }
  if (appMode || !selectedId || !node || !def) {
    return (
      <div className="inspector" data-no-tips>
        <div className="insp-head">
          {/* 「编辑参数界面」是标题行最右的一个小按钮（Houdini「Edit Parameter Interface」），弹窗框架 editor/ParamSheet.tsx
              SheetWindow 挂编辑器 ParamInterface，「确定」写回 exposed */}
          <div className="insp-title insp-title-graph">
            <span style={{ fontSize: 15, fontWeight: 600 }}>{pick(meta.name) || t("ui.common.unnamed")}</span>
            {interfaceRight && (
              <IconButton tone="ghost" size="sm"
                          layout="insp-interface" aria-label={t("ui.interface.title")} onClick={() => setInterfaceOpen(true)}>
                <IconSliders />
              </IconButton>
            )}
          </div>
          {meta.description && <div className="insp-desc">{pick(meta.description)}</div>}
        </div>
        {/* 整个面板一列标签：宽度按所有对外参数的标签定（现在被 Hide When 藏起、组折叠着的也算），切换选项时控件不动 */}
        <LabelGrid className="insp-body" ref={body} onColumn={refit} labels={exposedParams(exposed).map(rowLabel)}>
          {interfaceOpen && interfaceRight && (
            <SheetWindow kind={ParamInterface} nodeId="" p={INTERFACE_PARAM} value={exposed} choices={{}}
              set={(v) => setInterface(v as ExposedEntry[])} close={() => setInterfaceOpen(false)} />
          )}
          {(appMode ? exposedParams(exposed).length > 0 : exposed.length > 0) && (
            <InterfaceTree key={loaded} entries={exposed} bare={appMode} editable={interfaceRight && !viewerRole} />
          )}
          {appMode ? (
            exposedParams(exposed).length === 0 && (
              <div className="empty" style={{ height: "auto", paddingTop: 28 }}>
                {t("ui.params.app_empty")}
              </div>
            )
          ) : (
            <div className="empty" style={{ height: "auto", paddingTop: 28 }}>
              {t("ui.params.pick_node")}
            </div>
          )}
        </LabelGrid>
      </div>
    );
  }

  // 面板只显示可填写的参数：panel=false 的参数没有可填写的值，仅由连线驱动，其来源在节点上逐条说明（GraphNode.tsx `said`）
  const panelParams = def.params.filter((p) => p.panel !== false);
  const groups: Record<string, ParamDef[]> = {};
  for (const p of panelParams) (groups[p.group || t("ui.params.group_default")] ??= []).push(p);
  const refused = nodeMessages(results, node.id).filter((w) => w.refused);  // 连接错误的连线在标题下方说明；其他提醒位于节点右下角 ⓘ 的卡片中

  return (
    <div className="inspector" data-no-tips>
      <div className="insp-head">
        <div className="insp-title">
          {/* 节点类别画成其颜色的小方块，而不是框里的图标 */}
          <i className="insp-swatch" style={{ background: cat.color }} />
          {/* 节点名（就是节点图里的 id，Houdini 的 node name）：回车或点到别处生效，不合规或重名就地说明（editor/NodeNaming.tsx） */}
          <NodeNameInput id={node.id} disabled={viewerRole || !!focused} />
        </div>
        {/* 名称下方以小号灰字显示类型名与它的副标题：类型名不随改名变化，命令行与 DCC 插件用的正是它 */}
        <div className="insp-type">
          {def.id}
          {def.subtitle && <span className="insp-type-desc">{def.subtitle}</span>}
        </div>
        {/* 备注：任意文字，可选；「显示备注」打开时写在节点旁边 */}
        <NodeCommentField id={node.id} disabled={viewerRole || !!focused} />
        {/* 节点类别、许可证与开销：动参数之前需要知道的三件事。运行环境不在标题行显示。 */}
        <div className="insp-sub">
          {cat.label && <span className="chip" style={{ ["--c" as string]: cat.color }}>{cat.label}</span>}
          {/* 许可证自身的措辞，来自目录的标签表（nodes/tags.py）：页面不自行书写 */}
          {licenceTags.map((tag) => (
            <span key={tag.id} className={commercial ? "chip ok" : "chip nc"}>
              {tag.label}
            </span>
          ))}
          {/* 开销，用标题区自己的小标签：「GPU · 中」 */}
          {cost?.rating && <span className="chip">{cost.gpu ? `GPU · ${tierWord(cost.rating.tier)}` : tierWord(cost.rating.tier)}</span>}
          {/* 三方项目的代码仓库，在新标签页中打开 */}
          {/* 显示的值是上一次回复的、刚改过还没回来（state/results.ts Shown）：与节点状态格同一个词 */}
          {pending.has(node.id) && <span className="chip">{t("ui.state.phase_pending")}</span>}
          {def.links && webAddress(def.links.repo || def.links.homepage) && (
            <a className="chip link" href={def.links.repo || def.links.homepage} target="_blank" rel="noopener noreferrer">GitHub</a>
          )}
        </div>
      </div>
      {!nodeUsable(applies, def.id) && <div className="notice">{nodeWhy(applies, def.id)}</div>}
      {refused.map((w) => (
        <div className="notice warn refused" key={(w.port ?? "") + w.text}>
          <span>{w.text}</span>
          {w.fix && (
            <Button size="sm" layout="notice-fix" disabled={viewerRole} onClick={() => insertNode(node.id, w.port ?? "", w.fix!.insert)}>
              {w.fix.label}
            </Button>
          )}
        </div>
      ))}
      <LabelGrid className="insp-body" ref={body} onColumn={refit} labels={panelParams.map((p) => p.label)}>
          {/* 从其他软件粘贴一组参数：仅声明了 paste 的节点具有此按钮，位于参数上方；
              粘贴的是整张表，先粘贴再逐项核对，与操作顺序一致 */}
          <PasteFrom nodeId={node.id} def={def} />
          {Object.keys(def.writes ?? {}).length > 0 && <WritesGroup def={def} />}
          {/* 「输出」的「下载」只有一处：它的按钮参数（lab2shot/nodes/core/output.py buttons，editor/buttonActions.tsx download），
              和「计算」一样在下面的参数组里一行，不另写死一块 */}
          {panelParams.length === 0 && <div className="empty" style={{ height: "auto", paddingTop: 28 }}>{t("ui.params.no_params")}</div>}
          {Object.entries(groups).map(([g, ps]) => (
            <div className="group lsub" key={g}>
              <div className="group-title">{g}</div>
              {ps.map((p) => (
                <ParamRow key={p.name} nodeId={node.id} p={p} label={p.label} value={node.data.params[p.name]} set={(v) => setParam(node.id, p.name, v)}
                  why={why(results[node.id]?.applies, p.name)} pinned={exposedParams(exposed).some((x) => drives(x, `${node.id}.${p.name}`))}
                  onPin={() => toggleExposed(node.id, p.name, p.label)}
                  socket={p.wire ? (def.wired_ports?.includes(p.name)
                    // 常驻口：节点类型声明其始终存在（NodeDef.wired_ports，如 Focal Length / Filmback，接镜头标定节点的数值输出），不可关闭
                    ? { on: true, fixed: true, onClick: () => {} }
                    : { on: !!node.data.promoted?.includes(p.name), onClick: () => togglePromoted(node.id, p.name) }) : undefined}
                  said={results[node.id]?.sources?.[p.name]} over={results[node.id]?.overrides?.[p.name]} onNode={onNodeOf(node.id, p)} bare={!!focused}
                  pending={pending.has(node.id)} />
              ))}
            </div>
          ))}
          {/* 逐帧取值的连线画成曲线，放在最后，使参数行位置不变；只带一个值的连线不画，参数行已写明该数值。 */}
          {perFrameCurves(snap, node.id).map((c) => (
            <ParamCurve key={c.param} fp={c.fp} title={c.title} />
          ))}
      </LabelGrid>
      {/* 三方项目的许可证，贴于面板下边缘。仅用于展示，不参与任何判断：名称、一句说明、原文链接 */}
      {def.links && (
        <div className="insp-licence">
          <span className="insp-licence-name">{def.project} · {def.links.licence.name}</span>
          {webAddress(def.links.licence.url) && <a href={def.links.licence.url} target="_blank" rel="noopener noreferrer">{t("ui.params.licence_text")}</a>}
        </div>
      )}
    </div>
  );
}

// ------------------------------------------------------------------ 参数界面树（exposed，api/catalog.ts ExposedEntry）

interface TreeCtx {
  bare?: boolean;
  values: Record<string, unknown>; // 表达式里的名字取的值：每个公开参数现在的值
  labelOf: (name: string) => string;
  hidden: (x: ExposedEntry) => boolean; // Hide When 为真（组：里面全都不显示）
  index: (path: Path) => number; // 这一行在 rowsOf 里的位置（拖动用）
  switchKind: (x: ExposedParam) => SwitchKind; // 这一项是不是开关（组头的「全开 / 全关」：model/groupSwitches.ts）
  valueOf: (x: ExposedParam) => unknown; // 这一项现在的值（节点没写的取默认值）
  drag: ReturnType<typeof useRowDrag> | null; // null：不能拖（应用模式、只读标签页）
}

/** 公开参数按参数界面树画出来：组可折叠（`collapsed` 是打开时的默认），子组缩进；参数项按它自己的显示名、控件覆盖
 * （下拉 / 复选框），Hide When 为真的不画（组里全不画时组也不画），Disable When 为真的置灰并说「由『X』决定」。控件是
 * 目标参数自己的（ParamRow → ParamControls.tsx Control：文件、层级选择、对应关系、表格……与选中节点时同一个分发），只多
 * 分组、条件和覆盖。节点模式和应用模式（`bare`：不画公开 / 提升 / 在节点上显示三个标记）共用这一份。
 *
 * `editable`（节点模式、不是只读标签页；谁都能改自己手上这张图的参数界面，存到哪里才分权限）：每行（参数和组）左边一个
 * 把手，拖到参数上 = 放到它前面，拖到组的标题上 = 放进这个组（空组照画，也能拖进去），拖到最下面的空条 = 放到最外层
 * 末尾；规则与「编辑参数界面」弹窗同一套（graph/exposedTree.ts dropAt），写回 exposed，撤销步骤叫「调整参数顺序」
 * （graph/history.ts）。折叠只是这一次看的状态，不写进节点图。
 *
 * 调用处按「这一次打开的文档」（state/viewer.ts docId）给它 key：打开另一份（或重新打开同一份）文档，整棵树重新挂上——
 * 各组的折叠回到新文档的默认。 */
function InterfaceTree({ entries, bare, editable }: { entries: ExposedEntry[]; bare?: boolean; editable?: boolean }) {
  const snap = useGraphDoc();
  const all = exposedParams(entries);
  const values = exposedValues(entries, (id) => snap.nodes.find((m) => m.id === id)?.data, snap.nodeDefs);
  const labelOf = (name: string) => { const x = all.find((y) => y.name === name); return (x && entryLabel(x)) || name; };
  const hidden = (x: ExposedEntry): boolean => entryHidden(x, values);
  const rows = rowsOf(entries);
  const keys = rows.map((r) => r.path.join("/"));
  const drag = useRowDrag((from, to) => setExposedNow(dropAt(entries, rows, from, to)[0]), (from, to) => canDropAt(rows, from, to));
  const paramOf = (x: ExposedParam) => {
    const [nid, pname] = firstTarget(x);
    const n = snap.nodes.find((m) => m.id === nid);
    const p = n && snap.nodeDefs[n.data.typeId]?.params.find((q) => q.name === pname);
    return n && p ? { n, p } : null;
  };
  const switchKind = (x: ExposedParam): SwitchKind => {
    const got = paramOf(x);
    if (!got || got.p.widget === "button") return null;
    if (got.p.type === "boolean") return "bool";
    return x.widget === "checkbox" && got.p.type === "integer" ? "int01" : null;
  };
  const valueOf = (x: ExposedParam) => values[x.name];
  const ctx: TreeCtx = { bare, values, labelOf, hidden, index: (p) => keys.indexOf(p.join("/")), drag: editable && !bare ? drag : null, switchKind, valueOf };
  return (
    <>
      <TreeEntries entries={entries} parent={[]} depth={0} ctx={ctx} />
      {ctx.drag && <div className={`ptree-end${ctx.drag.over === rows.length ? " drag-over" : ""}`} {...ctx.drag.row(rows.length)} />}
    </>
  );
}

const setExposedNow = (tree: ExposedEntry[]) => useCookInputs.getState().setExposed(tree);

/** 拖动把手（只在能拖时画）。 */
function Grip({ ctx, i }: { ctx: TreeCtx; i: number }) {
  return ctx.drag ? <span className="ptree-grip nodrag" aria-label={t("ui.params.drag_order")} {...ctx.drag.grip(i)}>⋮⋮</span> : null;
}

function TreeEntries({ entries, parent, depth, ctx }: { entries: ExposedEntry[]; parent: Path; depth: number; ctx: TreeCtx }) {
  // 根下连着的参数项放在一个没有标题的段里，组各自一段：与节点参数的「组」同一种排版（.group）
  const runs: { key: string; group?: { g: ExposedGroup; path: Path }; items: { x: ExposedParam; path: Path }[] }[] = [];
  entries.forEach((x, i) => {
    const path = [...parent, i];
    if (ctx.hidden(x)) return;
    if (isGroup(x)) runs.push({ key: path.join("/"), group: { g: x, path }, items: [] });
    else if (runs.length && !runs[runs.length - 1].group) runs[runs.length - 1].items.push({ x, path });
    else runs.push({ key: path.join("/"), items: [{ x, path }] });
  });
  return (
    <>
      {runs.map((r) =>
        r.group ? (
          <TreeGroup key={r.key} g={r.group.g} path={r.group.path} depth={depth} ctx={ctx} />
        ) : (
          <div className={depth ? "ptree-items lsub" : "group lsub"} key={r.key}>
            {r.items.map(({ x, path }) => {
              const i = ctx.index(path);
              return (
                <div key={`${x.name}:${targetsOf(x).join("+")}`} className={`ptree-row lsub${ctx.drag?.over === i ? " drag-over" : ""}`} {...(ctx.drag ? ctx.drag.row(i) : {})}>
                  <Grip ctx={ctx} i={i} />
                  <ExposedRow x={x} ctx={ctx} />
                </div>
              );
            })}
          </div>
        ),
      )}
    </>
  );
}

function TreeGroup({ g, path, depth, ctx }: { g: ExposedGroup; path: Path; depth: number; ctx: TreeCtx }) {
  const [open, setOpen] = useState(!g.collapsed);
  const i = ctx.index(path);
  return (
    <div className={depth ? "ptree-sub lsub" : "group lsub"}>
      <div className={`ptree-head${ctx.drag?.over === i ? " drag-over" : ""}`} {...(ctx.drag ? ctx.drag.row(i) : {})}>
        <Grip ctx={ctx} i={i} />
        <button type="button" className={`group-title ptree-title${open ? " open" : ""}`} aria-expanded={open} onClick={() => setOpen(!open)}>
          <span className="ptree-caret" aria-hidden>▸</span>
          <span data-user-data>{pick(g.label)}</span>
        </button>
        <GroupSwitches g={g} ctx={ctx} />
      </div>
      {open && <TreeEntries entries={g.children} parent={path} depth={depth + 1} ctx={ctx} />}
    </div>
  );
}

/** 组里全是开关时组头右边的「全开」「全关」（规则：model/groupSwitches.ts）：一起改组里现在显示、没置灰的开关，一步撤销；
 * 已经全开（全关）时那一个按钮置灰。 */
function GroupSwitches({ g, ctx }: { g: ExposedGroup; ctx: TreeCtx }) {
  const switches = groupSwitches(g, ctx.switchKind);
  if (!switches) return null;
  const shown = (x: ExposedParam) => !ctx.hidden(x);
  const enabled = (x: ExposedParam) => !entryDisabled(x, ctx.values);
  const state = switchesState(switches, ctx.switchKind, ctx.valueOf, shown, enabled);
  const set = (on: boolean) => setParamsAcross(t(on ? "ui.params.all_on_step" : "ui.params.all_off_step", { group: pick(g.label) }), switchesTo(on, switches, ctx.switchKind, ctx.valueOf, shown, enabled));
  return (
    <span className="ptree-switches nodrag">
      <Button size="xs" disabled={state.allOn} onClick={() => set(true)}>{t("ui.params.all_on")}</Button>
      <Button size="xs" disabled={state.allOff} onClick={() => set(false)}>{t("ui.params.all_off")}</Button>
    </span>
  );
}

/** 应用模式「计算」按钮下列出的：按钮所在节点和它上游已经算好的节点，计算时说的警告、说明与生产风险（W / N / P），按卡片的
 * 说法（textOf：.app，没有的按级别的通用句），同一句只一次。卡片上没有节点，这些原本只在节点的信息卡和日志里。 */
function cookNotes(nid: string, edges: { source: string; target: string }[], results: Record<string, NodeStatus>): { level: string; text: string }[] {
  const out = new Map<string, string>();
  for (const id of upstream(nid, edges)) {
    const st = results[id];
    if (!st?.cached) continue;
    for (const m of st.messages ?? []) if (m.level === "W" || m.level === "N" || m.level === "P") out.set(textOf(m), m.level);
  }
  return [...out].map(([text, level]) => ({ level, text }));
}

/** 参数界面里的一项参数：目标节点或参数已经不在的不画（「编辑参数界面」里标红）。 */
function ExposedRow({ x, ctx }: { x: ExposedParam; ctx: TreeCtx }) {
  const snap = useGraphDoc();
  const { results, pending } = useShownResults();
  const catalog = useCatalog(); // picked_in_view: the widgets a 2D handle works on (服务端 nodes/handles.py PICKED_IN_VIEW，唯一一份)
  // the node parameter it speaks for: its first target on now (state/results.ts useShownTarget, the one rule in
  // model/targets.ts shownTarget) — its greying, 「待更新」 and picking in the view; an entry driving several holds one
  // value of one kind, written into all of them (graph/edit.ts)
  const [nid, pname] = useShownTarget(x);
  const n = snap.nodes.find((m) => m.id === nid);
  const nd = n && snap.nodeDefs[n.data.typeId];
  const p = nd?.params.find((q) => q.name === pname);
  // 应用模式看不到节点：按钮所在节点被「阻断」关着，在按钮下安静说一句（节点模式由节点底行说）。
  // 认法与节点底行同一份状态（state/results.ts byNode，规则 model/nodeOutcome.ts）。实时计算状态不在 useGraphDoc 里：
  // 这一句自己按 byNode 订阅，只在这句话变了时重画这一行
  const says = ctx.bare && p?.widget === "button" && p.action === "cook"; // 只在「计算」「打包」下说一次（同一节点的「下载」不重复）
  const quiet = useResults((s) => (says ? skippedBranches(nid, (id) => isBlocked(s.byNode[id])) : ""));
  // 应用模式看不到节点，也就看不到节点上的提醒：这一步（按钮所在节点及其上游）算出来时说的警告和说明（降档、分段接缝……），
  // 在按钮下按 .app 说法列出，同一句只列一次
  const notes = says ? cookNotes(nid, snap.edges, results) : [];
  if (!n || !nd || !p) return null;
  const set = (v: unknown) => setParamAction(nid, pname, v);
  const value = n.data.params[pname];
  // Disable When 为真（能判为真，表达式就一定读得通、用到的名字都在）：说由哪几项决定。写错的不锁（编辑器里标红）
  const off = entryDisabled(x, ctx.values);
  const used = off ? conditionNames(parseCondition(x.disable_when)) : [];
  const reason = !off ? undefined : used.length ? t("ui.params.decided_by", { names: used.map((u) => ctx.labelOf(u)).join(listSep()) }) : t("ui.params.disable_holds", { when: x.disable_when ?? "" });
  const onNode = p.simple && !ctx.bare ? { on: chosenOnNode(nd, n.data.onNode).includes(pname), onClick: () => toggleOnNodeAction(nid, pname) } : undefined;
  const showOnChange = !!x.show_on_change && p.widget !== "button";
  return (
    <>
    {/* 按钮项：按钮上写的就是模板里给它起的名字（「解算人物」「打包」），左边不重复行标签。控件自己的名字（无障碍名、
        按钮上的字）也用这一项的显示名，和行标签一致，不用节点参数名 */}
    <ParamRow nodeId={nid} p={{ ...p, label: entryLabel(x) }} label={p.widget === "button" ? "" : entryLabel(x)}
      value={value} set={set} why={why(results[nid]?.applies, pname)} pinned
      onPin={() => toggleExposedAction(nid, pname, x.label ?? entryLabel(x))}
      // 常驻口（NodeDef.wired_ports，如「切换」的「走哪一路」）与提升了的参数一样：真接了线才按线显示（ParamRow wiredFrom），没接线照常可改
      socket={nd.wired_ports?.includes(pname) ? { on: true, fixed: true, onClick: () => {} }
        : n.data.promoted?.includes(pname) ? { on: true, onClick: () => togglePromotedAction(nid, pname) } : undefined}
      said={results[nid]?.sources?.[pname]} over={results[nid]?.overrides?.[pname]} onNode={onNode} reason={reason} bare={ctx.bare} quiet={quiet} notes={notes} pending={pending.has(nid)} note={pick(x.note)}
      control={overrideControl(x, p, value, set, ctx.values, results[nid]?.applies, licensedValues(nd, pname), registeredValues(nd, pname)) ?? (showOnChange && (catalog?.picked_in_view ?? []).includes(p.widget ?? "") && !off && !why(results[nid]?.applies, pname)
        ? <PickRow nodeId={nid} p={p} value={value} set={set} /> : undefined)} />
    </>
  );
}

/** 公开参数树里带手柄的参数（点选、手绘）且「修改后在视图里显示这个节点」（show_on_change）为真：碰到这一行（点 chip、
 * 点空白处）就进入点选，视图显示这个节点、手柄可用。显式的入口是框架给每个这种参数自动生成的按钮参数「在视图里点选」
 * （服务端 nodes/params.py pick_button，editor/buttonActions.tsx pick_in_view），在这个参数下一行。 */
function PickRow({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  return (
    <div className="ppick" onPointerDownCapture={(e) => !(e.target as HTMLElement).closest("button")
      && enterPicking(nodeId)}>
      <Control nodeId={nodeId} p={p} value={value} set={set} />
    </div>
  );
}

/** 参数界面给的控件覆盖：下拉（模板作者自填的值和显示名；当前值不在其中的照实列出、标明；某一项自己的 Hide When 成立时
 * 不列它；当前值不会停在被藏起的项上——条件一变就落到第一个列出的，graph/exposedTree.ts hiddenChoices——万一在那一刻
 * 之前画到了，照列它）、复选框（布尔：勾 = 真；整数 0 / 1：
 * 勾 = 1）。没有覆盖的用参数自己的控件（undefined）。`values`：条件里的名字取的值（InterfaceTree）。 */
function overrideControl(x: ExposedParam, p: ParamDef, value: unknown, set: (v: unknown) => void, values: Record<string, unknown>,
                         answer: Availability | null | undefined, licensed: Record<string, string>, registered: Record<string, true>): React.ReactNode | undefined {
  if (x.widget === "checkbox") {
    const on = p.type === "boolean" ? !!value : value === 1;
    return <Switch on={on} onChange={(v) => set(p.type === "boolean" ? v : v ? 1 : 0)} label={entryLabel(x)} />;
  }
  if (x.widget === "menu" && x.options?.length) {
    const opts = x.options;
    const at = opts.findIndex((o) => condEqual(o.value, value));
    const shown = new Set(shownOptions(opts, values));
    // 作者的显示名只替换名称；许可词与不能选的原因照参数自己的下拉加（ui/controls.tsx optionView）
    const rows = opts.flatMap((o, i) => {
      if (i !== at && !shown.has(o)) return [];
      const view = optionView(p, o.value, answer, licensed, pick(o.label), p.name, registered);
      // 作者给这一项写的 Disable When 成立：置灰，指针上说作者写的为什么（不能选的原因是允许的悬停提示）
      const authored = optionDisabled(o, values);
      return [{ value: String(i), label: view.label, tip: authored !== null ? tipOf("disabled", authored) : view.tip, off: !!view.off || authored !== null }];
    });
    const own = t("ui.params.not_listed", { value: value === null || value === undefined ? t("ui.params.empty_value") : JSON.stringify(value) });
    return (
      <Select value={at < 0 ? "own" : String(at)} label={entryLabel(x)}
        options={at < 0 ? [{ value: "own", label: own }, ...rows] : rows}
        onPick={(v) => v !== "own" && set(opts[Number(v)].value)} />
    );
  }
  return undefined;
}
