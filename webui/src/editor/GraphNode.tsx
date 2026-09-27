import { memo, useEffect, useMemo } from "react";
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from "@xyflow/react";
import { nodeCategory, progressTip, type NodeTypeDef, type ParamDef, type ServerMessage } from "../api";
import type { PreparedGNode } from "../graph";
import { getNodeDefs, useCatalog, useTypes } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults, type NodeDelivery } from "../state/results";
import { useViewer } from "../state/viewer";
import { hiddenOnNode, isLive, nodeRows, portColor, promotedByHand } from "../graph/nodes";
import { noncommercialValues, PARAM, tableRows } from "../graph/rules";
import { IconChevron, IconEye } from "../ui/icons";
import { ownReason } from "../platform/util";
import { why, nodeUsable, nodeWhy } from "../api/applies";
import { ERROR_COLOR } from "../platform/palette";
import { useNodeUpload } from "../transfer/uploads";
import { useLocalProxies } from "../transfer/localProxy";
import { nodePhase, proxyOfNode } from "../state/phase";
import { useStaleNode } from "../state/stale";
import { agoText } from "../platform/format";
import { useSession } from "../state/session";
import { render } from "../messages/format";
import { NodeParam } from "./NodeParamRow";
import { NodeFoot } from "./NodeFoot";

const NO_MESSAGES: ServerMessage[] = [];

// 「输出」 never caches (it always re-delivers), so once a job that delivered through it ends, the usual cached/idle
// logic would say 未计算. Instead its footer states what its last delivery here became:
// 已交付 · <时间> while the server still has it (待取回), 已保存 / 已下载 once settled, 已过期 once gone. The title
// row's short word and colour for the same state are in state/phase.ts (the one table the state cell reads).
// "dismissed" (不要了) is not shown on the node: it is not a state worth pinning there.
const DELIVERY_TEXT = (d: NodeDelivery): string =>
  ({ pending: `已交付 · ${agoText(d.at)}`, saved: "已保存", downloaded: "已下载", expired: "已过期" })[d.state as Exclude<NodeDelivery["state"], "dismissed">];

const FILE_WIDGETS = ["file", "sequence", "deliver"];

/** The node's file (what it reads, or where 「输出」 saves) as its footer says it: a read-only mirror of the parameter,
 * which is in the panel; undefined when the node has none. */
function fileParam(def: NodeTypeDef): ParamDef | undefined {
  return def.params.find((p) => FILE_WIDGETS.includes(p.widget ?? ""));
}

/** The footer's words for it, with what was chosen from the file (the camera in its hierarchy): read-only, as
 * choices from a file never go on the node's body. */
function fileText(def: NodeTypeDef, p: ParamDef, params: Record<string, unknown>): string {
  const v = params[p.name];
  const file = (typeof v === "string" && v.split("/").pop()) || (p.widget === "deliver" ? "未选保存位置" : "未选择文件");
  const leaf = (path: string) => path.slice(path.lastIndexOf("/") + 1) || path;
  const chosen = def.params
    .filter((q) => (q.widget === "choice" || q.widget === "hierarchy") && q.choices_from.includes(p.name) && (Array.isArray(params[q.name]) ? (params[q.name] as string[]).length : params[q.name]))
    .map((q) => {
      const v = params[q.name];
      if (!Array.isArray(v)) return q.widget === "hierarchy" ? leaf(String(v)) : String(v);
      return v.length === 1 ? leaf(String(v[0])) : `${v.length} 个${q.label}`; // a file's thousands said as a count
    });
  return [file, ...chosen].join(" · ");
}


export const GraphNode = memo(function GraphNode({ id, data }: NodeProps<PreparedGNode>) {
  const def = getNodeDefs()[data.typeId];
  const oneOf = new Set(def?.needs_any ?? []);  // 多种接法之一中的端口：不标注「可选」（标注不正确，全部不接不可行）
  const outputs = data.outputs;
  // its inputs as the server resolved them (graph/index.ts: the declared ones, a ports_from table's rows, the promoted
  // parameters'); the promoted parameters' sit on their own rows below
  // 「提升到节点」产生的参数端口绘制在对应参数行上，因此从顶部端口列表中移除；常驻口（wired_ports）不在参数行上，
  // 它们即节点的输入口，保留在列表中
  const expanded = useViewer((s) => s.expanded.includes(id));
  // its parameters on its body: the ones its type puts there (all simple ones while 展开) and the ones 提升到节点, each with its input
  const rows = nodeRows(def, data, expanded);
  const rowNames = rows.map((p) => p.name).join(",");
  // Only a parameter that actually got a row keeps its input there: a table (畸变系数) never goes on the body, so its
  // input has to stay in the port list; otherwise 提升到节点 gives the artist a parameter with nowhere to plug a wire.
  const inputs = useMemo(() => data.inputs.filter((p) => !rowNames.split(",").some(
    (name) => name && !def?.wired_ports?.includes(name) && def?.param_ports[name]?.name === p.name)), [data.inputs, rowNames, def]);
  const rowsKey = rowNames;
  const outputsKey = outputs.map((p) => p.name).join(",");
  const inputsKey = inputs.map((p) => p.name).join(",");
  const updateInternals = useUpdateNodeInternals();
  useEffect(() => updateInternals(id), [outputsKey, inputsKey, rowsKey, id, updateInternals]); // handles came or went: wires re-attach
  const wired = data.wired;
  // its value outputs once known (a constant: at once), and where its parameters get their values
  const values = data.values;
  const sources = data.sources;
  const applies = data.applies;
  const commercial = data.commercial;
  const types = useTypes();
  const catalog = useCatalog();
  const cat = nodeCategory(catalog, def);
  const isDisplay = useLook((s) => s.displayId === id);
  // 计算进度只有一份数据源（api/progress.ts）：队列面板读取同一个 `now`，服务器只发送这一份
  const progress = useResults((s) => (s.now?.node === id ? s.now : null));
  const login = useSession((s) => s.state?.applies); // which node types this login may use now
  const nodeStatus = useResults((s) => s.byNode[id]);
  const graphId = useCookInputs((s) => s.graphId);
  const going = useNodeUpload(id, def); // its file going up (transfer/uploads.ts): said in its footer
  // 本机代理的处理进度（transfer/localProxy）：选取 store 自身的 `tasks`（稳定引用，原因见下方 rawMessages 的注释），
  // 在 useMemo 中按节点汇总
  const proxyTasks = useLocalProxies((s) => s.tasks);
  const proxy = useMemo(() => proxyOfNode(proxyTasks, id), [proxyTasks, id]);
  const delivery = useResults((s) => s.deliveries[`${graphId}:${id}`]); // only meaningful when def.delivers (「输出」)
  const stale = useStaleNode(id); // 参数已修改，但上一次结果仍可查看（state/stale.ts）
  const status = nodeStatus?.status ?? "idle";
  const note = nodeStatus?.note ?? "";
  // 端口由表格生成的节点（`ports_from_side === "inputs"`，目前仅「多层 EXR 输出设置」）在表格为空时
  // 没有任何输入口，向节点拖入一根连线即添加一行，即添加一个端口。底行对此给出一句说明（见下文）
  const needsFirstRow = def.ports_from_side === "inputs" && !tableRows(def, data.params).length;
  const blocked = nodeStatus?.blocked;
  // What the server says about the node (its usage checks, what it said when cooked: lab2shot/messages), used as the
  // marker's tooltip (a string: the same value while nothing changed). The selector reads only the store's own `messages`
  // array (a stable reference, unchanged unless the server answers again). Never use `.filter()` or `?? []` inline in a
  // zustand selector: that allocates a new array on every read and makes useSyncExternalStore treat the store as changed on
  // every render, indefinitely (React error #185: the node graph never mounts a single node). The actual filtering (a
  // refused wire is its own error, shown in the error colour, not a usage warning) happens in this component's own useMemo.
  const rawMessages = useResults((s) => s.results[id]?.messages) ?? NO_MESSAGES;
  // inside a 逐项处理 block (engine/scopes.py): how all its items stand, and which one the node is showing. Both come
  // from the status reply; the page never works out which block a node is in.
  const items = useResults((s) => s.results[id]?.summary);
  const itemNames = useResults((s) => s.results[id]?.item?.names);
  // a refused wire is the node's error (in the error colour); information (I) stays on the 信息 tab and in the log; a
  // notice the node shows as a production-risk mark on its bottom row (N-LENS-NEEDSUNDISTORTED) is said there, not again in the head
  const marked = def?.marks;
  const warnList = useMemo(
    () => rawMessages.filter((w) => !w.refused && (w.level === "W" || w.level === "N") && !marked?.some((m) => m.code === w.code)),
    [rawMessages, marked],
  );
  const warnings = warnList.some((w) => w.level === "W");
  const warned = warnList.length
    ? `${warnings ? "能算，但结果可能不是预期的" : "提醒"}：\n${warnList.map((w) => `· ${w.text} ${w.code}`).join("\n")}${warnList.some((w) => w.fix) ? "\n右边 ⓘ 的卡片里可以一键修正" : ""}\n详细看右边 ⓘ 的卡片`
    : "";
  const setDisplay = useLook((s) => s.setDisplay);
  const toggleExpanded = useViewer((s) => s.toggleExpanded);
  const reveal = useViewer((s) => s.revealParam);
  const select = useViewer((s) => s.select);

  if (!def) {
    return (
      <div className="gnode unavailable">
        <div className="gnode-head">
          <span className="gnode-title">{data.label}</span>
        </div>
        <div className="gnode-foot">
          <span>未知节点 {data.typeId}</span>
        </div>
      </div>
    );
  }

  const ports = Math.max(inputs.length, outputs.length);
  const color = (t: string) => portColor(types, t);
  const usableNow = nodeUsable(login, def.id);
  const tip = blocked ?? (usableNow ? def.description : nodeWhy(login, def.id));
  // What a port says is the server's sentence (nodes/base.py Port.tip: the type's name in the server's words
  // (PortDef.type_label) with its unit, what that type is, what this port means, what it does with a picture's alpha),
  // followed by what the node says. The page writes none of it; what a cooked result actually holds is in 数据信息
  // (editor/DataInfo.tsx), from the one summary data/summary.py makes.
  // `finding`: what a benchmark found about this input or the parameter behind it, which the catalogue carries only
  // for the accounts that run benchmarks (api/catalog.ts PortDef.finding): shown when present, never requested.
  const finding = (name: string) =>
    name.startsWith(PARAM) ? def.params.find((q) => q.name === name.slice(PARAM.length))?.finding : def.inputs.find((q) => q.name === name)?.finding;
  const portTip = (p: { name: string; tip?: string }) => [p.tip, finding(p.name), tip].filter(Boolean).join("\n\n");
  // 参数值的来源：规则唯一，接入几根连线就列出几条，全部列在下方区域，长度不限。
  // 不得按「参数是否显示在节点上（NodeDef.on_node）」或「同一上游有几个同类」改变显示形式：这两个量与来源无关，
  // 会使同样由连线驱动参数的两个节点外观完全不同。由连线驱动的参数不占参数行（其值不在本节点中，
  // 该行也没有可编辑内容），一律在此区域逐条说明；覆盖连线值的参数也在此说明。
  // 「提升到节点」产生的参数除外：其输入口绘制在该参数行上（`<Handle id={`param:…`}>`），
  // 行被移除则端口随之消失，连线立即断开。因此保留该行，由该行自行说明来源。
  const ownRows = promotedByHand(def, data.promoted);
  // 置灰的参数不进入此区域：对于因某个端口自带该值而置灰的参数，服务器同样会报告一句
  // 「Focal Length · 来自相机（ViPE 相机解算）」（engine/evaluation.py sources + applies.supplying_port）。
  // 该说明应出现在参数面板与交付的出处记录中：节点上该参数已置灰且悬停时说明原因，再增加一行属于重复，
  // 还会增加节点高度（排版检查限制节点高度）。
  const onBody = Object.fromEntries(Object.entries(sources ?? {}).filter(([name]) => !why(applies, name)));
  const paramRows = rows.filter((p) => ownRows.includes(p.name) || !(p.name in onBody));
  // 节点上使用简短形式：`参数 ← 值`（值尚未计算时显示来源节点），每行一条，不得换行。
  // 服务器的完整句子（如「镜头模型「OpenCV Brown」· 来自 AnyCalib 镜头标定」，在节点上需占两行）
  // 用于悬停提示与参数面板
  const said = Object.entries(onBody)
    // 「提升到节点」产生的参数在其所在行说明（端口位于该行），但覆盖了已连线输入的情况除外：
    // 「（覆盖相机的 Focal Length）」一句在行上放不下，且行上无法体现，因此在此另行说明
    .filter(([name, text]) => !ownRows.includes(name) || text.includes("（覆盖"))
    .map(([name, text]) => {
      const w = wired[name];
      const label = def.params.find((q) => q.name === name)?.label ?? name;
      // 覆盖其他输入的说明必须保留原文（简短形式无法说明覆盖对象），其余使用简短形式
      return { text: w && !text.includes("（覆盖") ? `${label} ← ${w.value || w.node}` : text, tip: text };
    });
  const hidden = hiddenOnNode(def, data);
  const file = fileParam(def);
  const gpu = data.cost?.lane === "gpu"; // its cost with its parameters, as the server resolved it
  const compute = usableNow && gpu ? data.cost?.rating ?? null : null;
  const simpleCount = def.params.filter((p) => p.simple).length;
  // A fresh cook (in progress or errored) always takes precedence over a stale delivery; otherwise a settled delivery record
  // (not 不要了) is more informative than the generic idle/cooked status a node that never caches would otherwise show.
  const deliveryShown = def.delivers && delivery && delivery.state !== "dismissed" && !isLive(status) && status !== "error" ? delivery : null;
  // 节点当前状态由一处确定（state/phase.ts 中的表）：右上角状态格只读取该结果，此处不拼接任何文字
  const phase = nodePhase({
    blocked: blocked ? ownReason(blocked, data.label) : undefined,
    unusable: usableNow ? undefined : nodeWhy(login, def.id),
    upload: going, proxy, delivery: deliveryShown ?? undefined, status, progress, stale,
  });

  return (
    <div className={`gnode${usableNow ? "" : " unavailable"}${blocked ? " blocked" : ""}${progress ? " cooking" : ""}`} data-tip={tip}>
      <div className="gnode-head">
        {/* the category's colour is a bar at the head's left edge, not a glyph badge */}
        <i className="gnode-kind" style={{ background: cat.color }} data-tip={cat.label || tip} />
        {/* 名称下方以小号灰字显示节点类型 id，所有节点均有：节点名称可修改，修改后无法再据此识别
            节点类型；类型 id 不随改名变化，且命令行与 DCC 插件使用的正是该字符串 */}
        <span className="gnode-name">
          <span className="gnode-title">{data.label}</span>
          <span className="gnode-type" data-tip={`节点类型：${def.label}（命令行和 DCC 插件里就用 ${data.typeId} 这个名字）`}>{data.typeId}</span>
        </span>
        {/* The state is a word at the right end of the title row, not a dot. 该格宽度固定
            （08a-node.css `--node-state-w`）：文字变化不改变节点宽度。显示内容、颜色与提示文字
            均由 state/phase.ts 中的表决定（上传进度、本机代理进度、服务器计算进度及其优先级）；
            此处只读取结果。格内不显示数字：解算器每个阶段重新计数，分母随阶段变化，数字会显得跳动；
            帧数与字节数显示在悬停提示中。 */}
        <span className={`gnode-state tnum ${phase.tone}`} data-tip={phase.tip}>
          {phase.word}
        </span>
        {(hidden > 0 || expanded) && (
          <button
            className={`expand-flag nodrag${expanded ? " on" : ""}`}
            data-tip={expanded ? "收起：节点上回到只显示选定的参数" : `展开：在节点上临时显示它全部 ${simpleCount} 个简单参数，方便对比。不存进节点图`}
            aria-label={expanded ? "收起" : "展开"}
            onClick={(e) => {
              e.stopPropagation();
              toggleExpanded(id);
            }}
          >
            <IconChevron size={12} up={expanded} />
          </button>
        )}
        <button
          className={`display-flag nodrag${isDisplay ? " on" : ""}`}
          data-tip="在视图中显示"
          onClick={(e) => {
            e.stopPropagation();
            setDisplay(id);
          }}
        >
          <IconEye size={12} />
        </button>
      </div>
      {ports > 0 && (
        <div className="gnode-ports">
          <div className="col">
            {inputs.map((p) => (
              <div className={`port-row in${p.inactive ? " inactive" : ""}`} key={p.name}
                data-tip={p.inactive ? `${portTip(p)}\n\n${p.inactive.text}` : portTip(p)}>
                <Handle type="target" position={Position.Left} id={p.name} className={`${p.multi ? "multi" : ""}${p.list ? " list" : ""}`.trim()} style={{ ["--c" as string]: color(p.type) }} />
                <span className="port-label">{p.label}</span>
                {p.optional && !oneOf.has(p.name) && <span className="opt">可选</span>}
              </div>
            ))}
          </div>
          <div className="col">
            {/* 输出口同样有可用与不可用之分：解算器接入相机后，其「相机」输出仅原样透传该相机，
                此时该端口置灰且无法连出，需要该相机时应从其来源连接。与输入口采用同一机制
                （由服务器计算 `inactive` 并下发，engine/graph.py ports()） */}
            {outputs.map((p) => (
              <div className={`port-row out${p.waits ? " waiting" : p.ghost ? " ghost" : p.inactive ? " inactive" : ""}`} key={p.name}
                data-tip={p.waits ? `${p.label}：还没有选。${p.waits}后，从这里接出的线就接上了` : p.ghost ? `没有「${p.name}」这个输出了：从这里接出的线要删掉或改接` : p.inactive ? `${portTip(p)}\n\n${p.inactive.text}` : portTip(p)}>
                {values?.[p.name] && <span className="port-value" data-user-data data-tip={`${p.label}：${values[p.name]}`}>{values[p.name]}</span>}
                <span className="port-label">{p.label}</span>
                <Handle type="source" position={Position.Right} id={p.name} className={p.list ? "list" : ""} isConnectableStart={!p.ghost && !p.inactive} style={{ ["--c" as string]: p.waits ? "rgba(235, 235, 245, 0.34)" : p.ghost ? ERROR_COLOR : color(p.type) }} />
              </div>
            ))}
          </div>
        </div>
      )}
      {paramRows.length > 0 && (
        <div className="gnode-params">
          {paramRows.map((p) => (
            <NodeParam key={p.name} nodeId={id} p={p} value={data.params[p.name]} promoted={promotedByHand(def, data.promoted).includes(p.name)}
              // 置灰的参数在节点上只说明不可用的原因，不再另加一行说明值的来源：两句含义相同，
              // 多出的一行还会增加节点高度。「Focal Length · 来自相机（ViPE 相机解算）」一句在参数面板、
              // 任务单与交付的出处记录中均有显示
              port={def.param_ports[p.name]} wired={wired[p.name] ?? null} why={why(applies, p.name)}
              said={why(applies, p.name) ? undefined : sources?.[p.name]}
              nc={noncommercialValues(def, p.name)} />
          ))}
        </div>
      )}
      {/* 计算进度：贴于节点底边的一条线。它不占布局（绝对定位，styles/12-preview-bar.css），因此出现、推进与
          消失均不改变节点高度。节点尺寸仅由其形状（标题、端口行数、参数行数）决定，
          计算进度不得改变尺寸。 */}
      {status === "cooking" && (
        <div className="gnode-progress" data-tip={progress ? progressTip(progress) : undefined}>
          {/* 进度条只依据 `at`（整个任务的完成量，服务器保证单调不减：lab2shot/progress.py）。不得用
              done / total 计算宽度，否则分母变化时进度条会归零重来；无法估计时绘制无刻度的进度条。 */}
          <div style={progress?.at == null ? { width: "30%" } : { width: `${progress.at * 100}%` }} className={progress?.at == null ? "indeterminate" : ""} />
        </div>
      )}
      {items && (
        <div className="gnode-items" data-tip={render("I-EACH-NODETIP")}>
          <span className="tnum">{render("I-EACH-NODEDONE", { done: items.cached ?? 0, total: items.total })}</span>
          {!!items.failed && <span className="items-failed tnum">{render("I-EACH-FAILED", { count: items.failed })}</span>}
          {!!itemNames?.length && (
            <span className="items-name" data-user-data data-tip={itemNames.join(" / ")}>
              {itemNames.join(" / ")}
            </span>
          )}
        </div>
      )}
      {said.length > 0 && (
        <div className="gnode-said" data-tip={said.map((t) => t.tip).join("\n")}>
          {said.map((t) => (
            <div key={t.text} data-tip={t.tip}>{t.text}</div>
          ))}
        </div>
      )}
      <NodeFoot id={id} def={def} data={data} commercial={commercial} warned={warned} warnings={warnings}
        compute={compute} status={status} note={note} going={going} file={file} deliveryShown={deliveryShown}
        outputs={outputs} rawMessages={rawMessages} needsFirstRow={needsFirstRow}
        deliveryText={DELIVERY_TEXT} fileText={fileText} select={select} reveal={reveal} />
    </div>
  );
});
