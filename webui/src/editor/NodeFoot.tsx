import type { NodeTypeDef, ParamDef, ServerMessage } from "../api";
import type { Output } from "../api/files";
import type { NodeStatus } from "../state/graph";
import type { UploadTask } from "../transfer/uploads";
import { NodeCopyToNuke } from "../ui/CopyToNuke";
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
  /** 「输出」：上一次打包的结果（null：还没有，其「下载」置灰）；其他节点一律为 undefined */
  output?: Output | null;
  outputs: unknown[];
  rawMessages: ServerMessage[];
  /** 口是从一张表长出来的、而表还空着（「多层 EXR 输出设置」）：底行那句「点 ＋ 或把线拖到 ＋ 加一层」
   * （一行叫什么是节点声明的 `ports_from_word`：层、路） */
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
        {/* 许可标签放在底行而非标题行：非商用节点的正式名称需要保留足够的空间。
            文字由服务器给出（ResolvedLicence.word = nodes/tags.py strictest）：「仅限研究」比「非商用」更严格，
            不得显示为后者——页面从不自行在多个标签中挑选。 */}
        {!commercial && (
          <span className="nc-badge">
            {def.at_defaults.licence.word || "非商用"}
          </span>
        )}
        {/* 节点自身声明的常驻标记（自行决定尺寸的裁切：I-SHAPE-CROP），按级别着色——只有生产风险才是红色；
            完整的句子在「数据信息」中 */}
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
        {/* 「为什么不能算」是一整句话（扩展未安装、该账号不能使用的许可）：它出现在节点的「数据信息」卡片与参数面板中，
            从不出现在底行——底行只有一行，而页面自己的文字从不截断，因此底行不放长文本。标题行的状态字说明
            它不能计算；节点本身置灰。 */}
        {isLive(status) ? (
          <span className="gnode-note">{note || STATUS_TEXT[status]}</span>
        ) : going ? (
          <span className={`gnode-file gnode-up ${going.state}`} data-user-data onPointerDown={() => reveal(id, going.param)}>
            {uploadNote(going)}
          </span>
        ) : file ? (
          // 「输出」的「下载」不在底行：它是按钮参数，默认显示在节点体上（core.output on_node，NodeParamRow.tsx ButtonParam）
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
              // 这种节点刚放下来时只有输入口列表末尾的「＋」口（GraphNode.tsx AddRowPort），底行说一句它怎么用。
              // 按声明判（`ports_from_side === "inputs"`），不写死某个节点。
              : needsFirstRow
                ? `点 ＋ 或把线拖到 ＋ 加一${def.ports_from_word}`
                : !outputs.length && status === "idle"
                  ? def.delivers ? "右键「计算」整理打包" : "右键「计算」算出"
                  : ""}
        </span>
        {/* 数据信息：节点右下角的小图标是唯一入口，没有快捷键。
            它取节点消息中最高级别的颜色，一眼可见有内容可读。 */}
        <NodeInfoButton node={id} label={data.label} level={worstLevel(rawMessages)} />
      </div>
  );
}

