import type { NodeTypeDef, ParamDef, ServerMessage } from "../api";
import type { Output } from "../api/files";
import type { NodeStatus } from "../state/graph";
import type { UploadTask } from "../transfer/uploads";
import { NodeCopyToNuke } from "../ui/CopyToNuke";
import { OutputDownload } from "../ui/OutputDownload";
import { NodeMarks } from "../ui/nodeMarks";
import { NodeInfoButton, worstLevel } from "./NodeInfoCard";
import { isLive, STATUS_TEXT } from "../graph/nodes";
import { uploadNote } from "../transfer/uploads";
import { agoText } from "../platform/format";

/** 节点最下面那一行：许可标签、节点自己声明的标记、这一轮计算留下的提醒、计算量、给 Nuke 的复制、
 * 状态 / 上传 / 文件那一格，右边「下一步做什么」那句话，和右下角的「数据信息」。
 *
 * 底行和 `GraphNode.tsx` 里画口、画参数的部分没有共用状态，所以单独成文件，用到的东西全部从入参传进来。 */
interface NodeFootAsk {
  id: string;
  def: NodeTypeDef;
  data: { label: string; params: Record<string, unknown> };
  commercial: boolean;
  /** 这一轮计算留下了提醒 / 警告（「注意」/「提醒」那一格；内容在数据信息卡片里） */
  warned: boolean;
  warnings: number | boolean;
  compute: { tier: string } | null;
  status: NodeStatus;
  note: string;
  going: UploadTask | null | undefined;
  file: ParamDef | null | undefined;
  /** 「输出」: what it last packed (null: nothing yet, its download is greyed); undefined for every other node */
  output?: Output | null;
  outputs: unknown[];
  rawMessages: ServerMessage[];
  /** 口是从一张表长出来的、而表还空着（「多层 EXR 输出设置」）：底行那句「拖一根线进来加一层」 */
  needsFirstRow: boolean;
  fileText: (def: NodeTypeDef, p: ParamDef, params: Record<string, unknown>) => string;
  select: (id: string) => void;
  reveal: (id: string, param: string, edit?: boolean) => void;
}

export function NodeFoot({ id, def, data, commercial, warned, warnings, compute, status, note, going, file,
                           output, outputs, rawMessages, needsFirstRow, fileText,
                           select, reveal }: NodeFootAsk) {
  return (
      <div className="gnode-foot">
        {/* the licence in the footer, not the title row: a non-commercial node's official name keeps the room it needs.
            The word is the server's (ResolvedLicence.word = nodes/tags.py strictest): 「仅限研究」 is stricter than
            「非商用」 and must not be shown as it — the page never picks among the tags itself. */}
        {!commercial && (
          <span className="nc-badge">
            {def.at_defaults.licence.word || "非商用"}
          </span>
        )}
        {/* the standing marks of the node's own declaration (a crop that decides its own size: I-SHAPE-CROP), coloured by
            their level — red only for a production risk; the sentence is in 数据信息 */}
        <NodeMarks marks={def.marks} />
        {/* 这一轮计算留下的提醒 / 警告，和上面「节点自己声明的标记」是同一种东西（都是这个节点要说的话），
            所以排在一起、同一套样式，在文档流里排，不绝对定位（绝对定位会盖住标题行右端的状态字）。
            放在底行不放标题行：标题行留给双击（双击节点头 = 打开它），能点的角标摆在那儿会被双击点到、
            面板页签跟着跳。 */}
        {warned && (
          <span className="node-mark warn-mark" data-level={warnings ? "W" : "N"}
                onClick={(e) => (e.stopPropagation(), select(id))}>
            {warnings ? "注意" : "提醒"}
          </span>
        )}
        {compute && (
          <span className="gnode-compute" data-tier={compute.tier}>
            {compute.tier}
          </span>
        )}
        <NodeCopyToNuke node={id} def={def} what={data.label} />
        {/* 为什么不能算 is a whole sentence (an extension not installed, a licence this account may not use): it is the
            node's 数据信息 card and the parameter panel, never its bottom row — that row is one line, and our own words are
            never cut, so it holds no long text. The title row's state word says
            that it cannot be computed; the node itself is greyed. */}
        {isLive(status) ? (
          <span className="gnode-note">{note || STATUS_TEXT[status]}</span>
        ) : going ? (
          <span className={`gnode-file gnode-up ${going.state}`} data-user-data onPointerDown={() => reveal(id, going.param)}>
            {uploadNote(going)}
          </span>
        ) : output !== undefined ? (
          // 「输出」 has no parameters: its body's one control is its download (enabled once its own cook packed a zip)
          <span className="nodrag"><OutputDownload output={output} /></span>
        ) : file ? (
          <span className="gnode-file" data-user-data
            onPointerDown={() => reveal(id, file.name)} onDoubleClick={(e) => (e.stopPropagation(), reveal(id, file.name, true))}>
            {fileText(def, file, data.params)}
          </span>
        ) : (
          <span />
        )}
        <span className="gnode-note">
          {output && !output.gone && output.finished && !isLive(status)
            ? `${agoText(output.finished)}打包`
            : status === "cooked" && note
              ? note
              // 口是从一张表长出来的、而表还是空的（如「多层 EXR 输出设置」：一行「图层」一个输入口）：
              // 这种节点刚放下来时一个输入口都没有，看不出该往哪儿接，底行给一句提示。
              // 按声明判（`ports_from_side === "inputs"`），不写死某个节点。
              : needsFirstRow
                ? "拖一根线进来加一层"
                : !outputs.length && status === "idle"
                  ? def.delivers ? "右键「计算」整理打包" : "右键「计算」算出"
                  : ""}
        </span>
        {/* 数据信息: the small icon at the node's bottom-right corner is the one way in, no shortcut.
            It takes the colour of the loudest message the node has, so it is plain to see there is something to read. */}
        <NodeInfoButton node={id} label={data.label} level={worstLevel(rawMessages)} />
      </div>
  );
}

