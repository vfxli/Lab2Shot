import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { nodeCategory, type NodeTypeDef, type ParamDef, type SceneKind, type WritesPart } from "../api";
import { insertNode as insertNodeAction, setLabel as setLabelAction, setParam as setParamAction, toggleExposed as toggleExposedAction, toggleOnNode as toggleOnNodeAction, togglePromoted as togglePromotedAction } from "../graph/actions";
import { useGraphSnapshot } from "../graph/snapshot";
import { getTypes, useCatalog } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import { chosenOnNode, nodeMessages, portColor, writesTable } from "../graph/nodes";
import { commercialOf, costOf, wiredFrom } from "../graph/rules";
import { why, nodeUsable, nodeWhy } from "../api/applies";
import { Button } from "../ui/Button";
import { PasteFrom } from "./PasteFrom";
import { OutputDownload } from "../ui/OutputDownload";
import { Control, fitWidth } from "./ParamControls";
import { IconOnNode } from "../ui/icons";
import { multiline } from "../ui/controls";
import { ParamCurve } from "./CurvesView";
import { useSession } from "../state/session";
import { webAddress } from "../platform/util";

/** One parameter: label with its expose pin and (for one a wire can drive) its 提升到节点 pin, control, and, when the
 * node's connections or settings make it irrelevant, greyed out with the reason (the server's graph status says which).
 * Driven by a wire, the control is greyed and says where the value comes from ("38.6 mm · 来自 AnyCalib 镜头标定 · Focal Length");
 * `said`: where the node gets its value, when the server's status says (a Focal Length over the camera's, or the camera's).
 * `onNode`: whether the node's body shows this row too (a simple parameter), and the
 * click that changes it. A parameter clicked on the node is brought into view here and flashes (its field focused on a
 * double click).
 *
 * 标签后的三个标记各司其职：对外参数（ExposePin）、提升到节点（PromotePin：增加一个输入口）、
 * 在节点上显示（OnNodePin：仅显示，不提供输入口）。
 *
 * 参数面板不显示悬停提示（整块面板带 data-no-tips，platform/tips.ts）。 */
function ParamRow({ nodeId, p, label, value, set, why, pinned, onPin, socket, said, onNode }: {
  nodeId: string; p: ParamDef; label: string; value: unknown; set: (v: unknown) => void; why?: string; pinned: boolean; onPin: () => void;
  socket?: { on: boolean; fixed?: boolean; onClick: () => void }; said?: string; onNode?: { on: boolean; onClick: () => void };
}) {
  const snap = useGraphSnapshot();
  const wired = socket?.on ? wiredFrom(snap, nodeId, p.name) : null;
  const over = said?.match(/（(覆盖[^）]*)）$/)?.[1];
  // wired: where from; else what the node says of it when an input gives it (「Focal Length · 来自相机（ViPE 相机解算）」)
  const note = wired ? `← ${wired.from}${wired.value ? ` · ${wired.value}` : ""}` : said && !over ? said : "";
  const reveal = useViewer((s) => (s.reveal?.node === nodeId && s.reveal.param === p.name ? s.reveal : null));
  const row = useRef<HTMLDivElement>(null);
  const [flash, setFlash] = useState(false);
  useEffect(() => {
    if (!reveal || !row.current) return;
    // After the panel has laid out what it just switched to (another node, a tall table above this row): a scroll
    // measured at the same moment stops short of the row by the height still growing above it.
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
    <div ref={row} className={`prow${multiline(p) ? " multi" : ""}${why ? " inactive" : ""}${wired && !why ? " wired" : ""}${flash ? " flash" : ""}`} data-param={p.name}>
      <div className="plabel">
        <span className="plabel-text">{label}</span>
        <ExposePin on={pinned} onClick={onPin} />
        {socket && <PromotePin on={socket.on} wired={!!wired} fixed={socket.fixed} onClick={socket.onClick} />}
        {onNode && <OnNodePin on={onNode.on || !!socket?.on} promoted={!!socket?.on} onClick={onNode.onClick} />}
      </div>
      {wired && !why ? (
        // Driven by a wire: a block in the wire's own colour stating the value and its source, instead of a
        // field that cannot be typed into.
        <span className="ctl pwired">
          <b data-user-data>{wired.value || "待计算"}</b>
          <small data-user-data>· 来自 {wired.from}</small>
        </span>
      ) : (
        <fieldset className="ctl" disabled={!!why}>
          <Control nodeId={nodeId} p={p} value={value} set={set} />
        </fieldset>
      )}
      {/* 值从哪里来、覆盖了什么，由下方的小号灰字说明 */}
      {!why && !wired && note && <div className="pwhy">{note}</div>}
      {!why && over && <div className="pwhy pover">{over}</div>}
    </div>
  );
}

/** 提升到节点 (Houdini's channel reference, ComfyUI's widget turned into an input): one click puts the parameter on
 * the node's body, as an input of its own (`param:<name>`) together with, on the same row, its control to edit it
 * directly; another click removes the row along with its wire. 接入连线后节点上的控件置灰，取值以连线传入的值为准
 * (editor/NodeParamRow.tsx)。 */
function PromotePin({ on, wired, fixed, onClick }: { on: boolean; wired: boolean; fixed?: boolean; onClick: () => void }) {
  return (
    <button className={`socket-pin${on ? " on" : ""}${wired ? " wired" : ""}${fixed ? " fixed" : ""}`}
      aria-label={on ? "取消提升" : "提升到节点"} aria-pressed={on} aria-disabled={fixed} onClick={fixed ? undefined : onClick}>
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
    <button className={`onnode-pin${on ? " on" : ""}${promoted ? " fixed" : ""}`} aria-label="在节点上显示" aria-pressed={on} aria-disabled={promoted}
      onClick={promoted ? undefined : onClick}>
      <IconOnNode size={11} filled={on} />
    </button>
  );
}

const NO_KINDS: SceneKind[] = [];
const WRITES_MARK: Record<WritesPart["how"], string> = { full: "✓", static: "静止", no: "✗" };
// 可以写出但部分内容无法保留（如 FBX 无法保留模型网格的分区）：标注「部分」。
// 宽度与「静止」相同，位置不跳动；格式丢失的内容必须对使用者可见
const WRITES_PARTLY = "部分";

/** 支持的数据: what a 3D output-settings node's format holds of each kind of 3D data, from its declaration
 * (OutputSettings.writes): each kind in its type's color, ✓ / 静止 / ✗ / 部分. */
function WritesGroup({ def }: { def: NodeTypeDef }) {
  const catalog = useCatalog();
  const kinds = catalog?.scene_kinds ?? NO_KINDS;
  const types = getTypes();
  return (
    <div className="group">
      <div className="group-title">支持的数据</div>
      <div className="writes">
        {writesTable(kinds, def).map((r) => (
          <span key={r.kind.id} className={`writes-item ${r.how}${r.lost ? " lost" : ""}`}>
            <i style={{ background: portColor(types, r.kind.type) }} />
            {r.kind.label}
            <b>{r.lost ? WRITES_PARTLY : WRITES_MARK[r.how]}</b>
          </span>
        ))}
      </div>
    </div>
  );
}

/** Pin that exposes a parameter on the graph: templates show it up front, lab2shot cook and DCC plugins can set it. */
function ExposePin({ on, onClick }: { on: boolean; onClick: () => void }) {
  return (
    <button className={`expose-pin${on ? " on" : ""}`} aria-label="对外参数" aria-pressed={on} onClick={onClick}>
      <svg width={11} height={11} viewBox="0 0 16 16" fill={on ? "currentColor" : "none"} stroke="currentColor" strokeWidth={1.6} strokeLinejoin="round">
        <path d="M5.5 2.5h5l-.8 4 2.3 2.2H4l2.3-2.2z" />
        <path d="M8 8.7v4.8" strokeLinecap="round" />
      </svg>
    </button>
  );
}


// 面板不设选项卡，全部为参数；节点的数据信息位于节点右下角 ⓘ 的卡片中（NodeInfoCard）。

/** The parameters of this node a wire drives with a value per frame, and the result to read the curve from
 * (data/values.py says 「（逐帧）」 for a value that changes and 「（逐帧，不变）」 for one that does not: only the
 * first is worth a curve). */
function perFrameCurves(snap: ReturnType<typeof useGraphSnapshot>, id: string): { param: string; fp: string; title: string }[] {
  const out: { param: string; fp: string; title: string }[] = [];
  for (const e of snap.edges) {
    const param = e.target === id ? (e.targetHandle ?? "").match(/^param:(.+)$/)?.[1] : undefined;
    if (!param) continue;
    const port = e.sourceHandle ?? "";
    const status = snap.results[e.source];
    const said = status?.values?.[port] ?? "";
    const fp = status?.outputs?.[port] ?? "";
    if (!fp || !said.includes("（逐帧）")) continue;
    const def = snap.nodeDefs[snap.nodes.find((n) => n.id === id)?.data.typeId ?? ""];
    const label = def?.params.find((q) => q.name === param)?.label ?? param;
    out.push({ param, fp, title: label });
  }
  return out;
}

export function ParamPanel() {
  const applies = useSession((s) => s.state?.applies);
  const selectedId = useViewer((s) => s.selectedId);
  const snap = useGraphSnapshot();
  const node = snap.nodes.find((n) => n.id === selectedId);
  const def = node ? snap.nodeDefs[node.data.typeId] : undefined;
  const catalog = useCatalog();
  const cat = nodeCategory(catalog, def);
  const setParam = setParamAction;
  const setLabel = setLabelAction;
  const meta = useCookInputs((s) => s.meta);
  const exposed = useCookInputs((s) => s.exposed);
  const nodes = snap.nodes;
  const nodeDefs = snap.nodeDefs;
  const toggleExposed = toggleExposedAction;
  const togglePromoted = togglePromotedAction;
  const toggleOnNode = toggleOnNodeAction;
  const results = useResults((s) => s.results);
  const graphId = useCookInputs((s) => s.graphId);
  // 「输出」: what it last packed (its download is its only control: ui/OutputDownload.tsx)
  const output = useResults((s) => (selectedId ? s.outputs[`${graphId}:${selectedId}`] : undefined));
  const insertNode = insertNodeAction;
  const setInspectorFit = useViewer((s) => s.setInspectorFit);
  const body = useRef<HTMLDivElement>(null);
  const commercial = commercialOf(snap, selectedId ?? "");
  const cost = costOf(snap, selectedId ?? "");
  // its licence as the server resolved it with this node's settings, said in the catalogue's own words
  const licence = (selectedId && results[selectedId]?.licence) || def?.at_defaults.licence;
  const licenceTags = (licence?.tags ?? []).filter((id) => !catalog?.tags[id]?.implied).map((id) => ({ id, label: catalog?.tags[id]?.label ?? id }));
  const params = node?.data.params;
  // 依赖中必须包含 `exposed`：未选中节点时面板只显示「对外参数」各行，标签来自 `exposed`。它变化时（打开
  // 模板卡、增减对外参数）标签随之改变，若不重新计算列宽，列宽会停留在 `fitWidth` 的下限 112px，长标签
  // （如「已知 Focal Length」加三个标记约 155px）会与标记重叠。
  useLayoutEffect(() => {
    if (body.current) setInspectorFit(fitWidth(body.current));
  }, [def, params, exposed, setInspectorFit]);

  // 「在节点上显示」标记及其点击行为（仅可显示在节点上的参数具备：p.simple）
  const onNodeOf = (nodeId: string, p: ParamDef) => {
    const n = nodes.find((m) => m.id === nodeId);
    const nd = n && nodeDefs[n.data.typeId];
    return p.simple && n && nd ? { on: chosenOnNode(nd, n.data.onNode).includes(p.name), onClick: () => toggleOnNode(nodeId, p.name) } : undefined;
  };

  if (!selectedId || !node || !def) {
    // nothing selected: the graph itself and its exposed parameters
    return (
      <div className="inspector" data-no-tips>
        <div className="insp-head">
          <div className="insp-title">
            <span style={{ fontSize: 15, fontWeight: 600 }}>{meta.name}</span>
          </div>
          {meta.description && <div className="insp-desc">{meta.description}</div>}
        </div>
        <div className="insp-body" ref={body}>
          {exposed.length > 0 && (
            <div className="group">
              <div className="group-title">对外参数</div>
              {exposed.map((x) => {
                const [nid, pname] = x.target.split(".");
                const n = nodes.find((m) => m.id === nid);
                const nd = n && nodeDefs[n.data.typeId];
                const p = nd?.params.find((q) => q.name === pname);
                if (!n || !nd || !p) return null;
                return (
                  <ParamRow key={x.name} nodeId={nid} p={p} label={x.label} value={n.data.params[pname]} set={(v) => setParam(nid, pname, v)}
                    why={why(results[nid]?.applies, pname)} pinned onPin={() => toggleExposed(nid, pname, x.label)}
                    socket={n.data.promoted?.includes(pname) ? { on: true, onClick: () => togglePromoted(nid, pname) } : undefined}
                    said={results[nid]?.sources?.[pname]} onNode={onNodeOf(nid, p)} />
                );
              })}
            </div>
          )}
          <div className="empty" style={{ height: "auto", paddingTop: 28 }}>
            选中一个节点来调整它的参数
          </div>
        </div>
      </div>
    );
  }

  // 面板只显示可填写的参数：panel=false 的参数没有可填写的值，仅由连线驱动，其来源在节点上逐条说明（GraphNode.tsx `said`）
  const panelParams = def.params.filter((p) => p.panel !== false);
  const groups: Record<string, ParamDef[]> = {};
  for (const p of panelParams) (groups[p.group || "参数"] ??= []).push(p);
  const refused = nodeMessages(results, node.id).filter((w) => w.refused);  // 连接错误的连线在标题下方说明；其他提醒位于节点右下角 ⓘ 的卡片中

  return (
    <div className="inspector" data-no-tips>
      <div className="insp-head">
        <div className="insp-title">
          {/* the node's category is a small square of its colour, not an icon in a box */}
          <i className="insp-swatch" style={{ background: cat.color }} />
          <input value={node.data.label} onChange={(e) => setLabel(node.id, e.target.value)} aria-label="节点名称" />
        </div>
        {/* 名称下方以小号灰字显示节点类型 id，与节点上的显示一致：
            名称可修改，修改后无法据此识别节点类型；id 不随改名变化，命令行与 DCC 插件使用的正是该 id */}
        <div className="insp-type">{def.id}</div>
        {/* The node's category, its licence and its cost: the three things to know before touching a
            parameter. Which environment it runs in is not shown in the title row. */}
        <div className="insp-sub">
          {cat.label && <span className="chip" style={{ ["--c" as string]: cat.color }}>{cat.label}</span>}
          {/* the licence's own word, from the catalogue's tag table (nodes/tags.py): the page never writes it */}
          {licenceTags.map((tag) => (
            <span key={tag.id} className={commercial ? "chip ok" : "chip nc"}>
              {tag.label}
            </span>
          ))}
          {/* what it costs, in the head's own small label: 「GPU · 中」 */}
          {cost?.rating && <span className="chip">{cost.gpu ? `GPU · ${cost.rating.tier}` : cost.rating.tier}</span>}
          {/* 三方项目的代码仓库，在新标签页中打开 */}
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
            <Button size="sm" layout="notice-fix" onClick={() => insertNode(node.id, w.port ?? "", w.fix!.insert)}>
              {w.fix.label}
            </Button>
          )}
        </div>
      ))}
      <div className="insp-body" ref={body}>
          {/* 从其他软件粘贴一组参数：仅声明了 paste 的节点具有此按钮，位于参数上方；
              粘贴的是整张表，先粘贴再逐项核对，与操作顺序一致 */}
          <PasteFrom nodeId={node.id} def={def} />
          {Object.keys(def.writes ?? {}).length > 0 && <WritesGroup def={def} />}
          {def.delivers && (
            <div className="group">
              <div className="group-title">下载</div>
              <OutputDownload output={output ?? null} size="sm" />
            </div>
          )}
          {panelParams.length === 0 && !def.delivers && <div className="empty" style={{ height: "auto", paddingTop: 28 }}>这个节点没有参数</div>}
          {Object.entries(groups).map(([g, ps]) => (
            <div className="group" key={g}>
              <div className="group-title">{g}</div>
              {ps.map((p) => (
                <ParamRow key={p.name} nodeId={node.id} p={p} label={p.label} value={node.data.params[p.name]} set={(v) => setParam(node.id, p.name, v)}
                  why={why(results[node.id]?.applies, p.name)} pinned={exposed.some((x) => x.target === `${node.id}.${p.name}`)}
                  onPin={() => toggleExposed(node.id, p.name, p.label)}
                  socket={p.wire ? (def.wired_ports?.includes(p.name)
                    // 常驻口：节点类型声明其始终存在（NodeDef.wired_ports，如 Focal Length / Filmback，接镜头标定节点的数值输出），不可关闭
                    ? { on: true, fixed: true, onClick: () => {} }
                    : { on: !!node.data.promoted?.includes(p.name), onClick: () => togglePromoted(node.id, p.name) }) : undefined}
                  said={results[node.id]?.sources?.[p.name]} onNode={onNodeOf(node.id, p)} />
              ))}
            </div>
          ))}
          {/* A wire carrying a value per frame is drawn as a curve, placed at the end so no parameter row
              ever moves; a wire carrying one value draws nothing, as the row already states the number. */}
          {perFrameCurves(snap, node.id).map((c) => (
            <ParamCurve key={c.param} fp={c.fp} title={c.title} />
          ))}
      </div>
      {/* 三方项目的许可证，贴于面板下边缘。仅用于展示，不参与任何判断：名称、一句说明、原文链接 */}
      {def.links && (
        <div className="insp-licence">
          <span className="insp-licence-name">{def.project} · {def.links.licence.name}</span>
          {webAddress(def.links.licence.url) && <a href={def.links.licence.url} target="_blank" rel="noopener noreferrer">许可证原文</a>}
        </div>
      )}
    </div>
  );
}
