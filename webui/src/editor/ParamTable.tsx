/** 表格参数（widget "table"）的绘制与编辑归本模块：取值为一组条目的参数，如「读取多条序列」的各条序列、
 * 「多层 EXR 输出设置」的图层、相机的畸变参数（关节映射不是表格参数，见 editor/RigMap.tsx）。
 *
 * 「读取序列」没有图层表：输出口直接由文件中的图层生成（nodes/core/input.py made_ports）。 */

import { optionView } from "../ui/controls";
import type { ParamDef } from "../api";
import { deriveParams } from "../graph/actions";
import { addEmptyRow } from "../graph/edit";
import { rowsLeft } from "../graph/rules";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { greyed } from "../api/applies";
import { useRowDrag } from "../ui/rowDrag";
import { useChoices } from "../ui/choices";
import { IconClose } from "../ui/icons";
import { Button } from "../ui/Button";
import type { CellChoice, CellControl } from "./ParamControls";

type Entry = Record<string, unknown>;

/** 条目列表（P 的取值为某个模型的列表）：每个条目一行，列出服务器为其声明的每一列：可编辑的列画成控件，
 * 其余（`widget: "fixed"`）显示为纯文字（去掉只读列就看不出一行是哪条序列、有多少帧）。
 * 一行中哪些字段起作用由服务器判定（nodes/applies.py 逐行解析：节点的 applies 中的 `layers[2].scale`），此处不推算。
 * 条目可以去掉；由其他参数推出的参数（derived_from）可以据此重新列出。
 *
 * `name` 是行的身份而不是一列：它是键，没有别的字段为该行命名时也是该行自己的名字。 */

/** 只读格的文字：选项列显示该选项的名称（ui/controls.tsx optionView，与下拉同一处），其余列显示值本身。 */
function fixedText(f: ParamDef, v: unknown): string {
  return f.options && v !== null && v !== undefined && v !== "" ? optionView(f, v, null, {}).label : String(v ?? "");
}

/** `Control`：每个格子里画的控件——就是参数面板的那一个分发（editor/ParamControls.tsx Control），由它传进来，本文件不
 * 反过来 import 它（互相 import 会成环）。 */
export function TableParam({ nodeId, p, value, set, Control }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void; Control: CellControl }) {
  const rows = (value as Entry[] | null) ?? [];
  // 声明了 panel=false 的列不在面板中占列（读取节点的「图层」：它显示在「文件里」一列的最前面）
  const columns = (p.items ?? []).filter((f) => f.panel !== false);
  const answer = useResults((s) => s.results[nodeId]?.applies);
  // 控件不得显隐切换：行中不起作用的格仍占据所在列，置灰；否则同一张表各行列数不同，
  // 使用者无法记住各列含义
  const inactive = (f: ParamDef, i: number) => greyed(answer, `${p.name}[${i}].${f.name}`);
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const def = getNodeDefs()[typeId];
  // 行左侧的名称：文件给出的名称，否则为行自己的 id，且仅在两者都不可编辑时显示。名称由使用者填写的表
  // （「多层 EXR 输出设置」的图层名）或以自身某一列为行命名的表（「读取多条序列」的序列）没有名称格：
  // 每一列都是一个格，按服务器声明的顺序排列。
  const labelCol = columns.find((f) => f.name === "label");
  const nameCol = columns.find((f) => f.name === "name");
  const leads = labelCol?.widget === "fixed" ? labelCol : !labelCol && nameCol?.widget === "fixed" ? nameCol : null;
  // 不可编辑、又不作为行名显示的 `name` 是行的键（输出口的 id）：不作为可读的一列
  const fields = columns.filter((f) => f !== leads && !(f === nameCol && f.widget === "fixed"));
  const side = def?.ports_from === p.name ? def.ports_from_side : undefined;
  const isPortsFrom = side !== undefined;
  const choice = useChoices(nodeId, p);
  const cell = (row: Entry): CellChoice | null =>
    choice && { ...choice, auto: (choice.auto as Record<string, string> | undefined)?.[String(row.name)] ?? "" };
  const edit = (i: number, name: string, v: unknown) => set(rows.map((r, j) => (j === i ? { ...r, [name]: v } : r)));
  const drag = useRowDrag((fromIdx, toIdx) => {
    const next = rows.slice();
    const [moved] = next.splice(fromIdx, 1);
    next.splice(toIdx, 0, moved);
    set(next);
  });
  // 列头：单行表格仅凭内容无法区分各列（两项内容相邻）。行数较多时，列头也可避免每行重复标签。
  // 「重新列出」接在列头行的右端，不单独占一行；没有列头时（表为空或只有一列）才单独占一行，
  // 此时上方没有可衔接的行
  const again = p.derived_from.length > 0 && (
    <Button
      tone="ghost"
      layout="ptable-again"
      onClick={() => void deriveParams(nodeId).then((got) => got && p.name in got && set(got[p.name]))}
    >
      重新列出
    </Button>
  );
  // ---- 每张表使用一套列：若表头与每一行各自为 flex 行，列宽分别计算，无法对齐（相同 class 不能保证列宽一致，
  // 调整 gap、min-width 也无法解决）。列宽由该表计算一次（`template`），表头与每一行均通过 subgrid 使用同一套列，
  // 由结构保证对齐。
  const hasGrip = isPortsFrom && rows.length > 1;
  // 输入侧的表（一行一个输入口，「多层 EXR 输出设置」的图层、「切换」的各路）有「添加」，最后一行也能去掉：加错了的空行
  // 要能拿掉；节点自己给行起名的表满了（「切换」十路：graph/rules.ts rowsLeft）就不再有「添加」
  const addsRows = side === "inputs";
  const canAdd = addsRows && rowsLeft(def, { [p.name]: rows });
  const hasDrop = addsRows ? rows.length > 0 : hasGrip;
  // 列宽：只读列按自身文字宽度（3DE 的参数名不得截断），选项来自上游的列占据剩余宽度；
  // 没有此类列时由最后一个可编辑列占据剩余宽度，否则右侧会大片留空
  const wideAt = fields.findIndex((f) => f.widget === "choice");
  const lastEditable = fields.map((f) => f.widget !== "fixed").lastIndexOf(true);
  const columnOf = (f: ParamDef, i: number) =>
    f.widget === "fixed" ? "minmax(56px, max-content)"
      : i === (wideAt >= 0 ? wideAt : lastEditable) ? "minmax(96px, 1fr)"
        : "max-content";
  const template = [
    ...(hasGrip ? ["auto"] : []),
    ...(leads ? ["minmax(64px, max-content)"] : []),
    ...fields.map(columnOf),
    ...(hasDrop ? ["auto"] : []),
    ...(again ? ["max-content"] : []),
  ].join(" ");
  // 后两列在部分行中没有内容：为其保留空格以使列号对齐（位置不跳动）
  const blank = (key: string) => <span key={key} className="ptable-blank" aria-hidden />;

  const head = (leads || fields.length > 1) && rows.length > 0 && (
    <div className="ptable-row ptable-head">
      {hasGrip && <span className="ptable-grip" aria-hidden />}
      {leads && <span className="ptable-name" aria-hidden>{leads.label}</span>}
      {fields.map((f) => (
        <span key={f.name} aria-hidden className={f.widget === "fixed" ? "ptable-fixed" : `ptable-cell${f.widget === "choice" ? " wide" : ""}`}>
          {f.label}
        </span>
      ))}
      {hasDrop && blank("drop")}
      {again}
    </div>
  );

  return (
    <div className="ptable" style={{ gridTemplateColumns: template }}>
      {head}
      {rows.map((row, i) => (
        <div className={`ptable-row${drag.over === i ? " drag-over" : ""}`} key={String(row.name ?? i)} {...drag.row(i)}>
          {hasGrip && (
            <span className="ptable-grip nodrag" {...drag.grip(i)}>
              ⠿
            </span>
          )}
          {leads && (
            <span className="ptable-name" data-user-data>
              {String(row[leads.name] ?? "")}
            </span>
          )}
          {fields.map((f) => {
            const off = inactive(f, i); // 该格当前不起作用：置灰，位置不变
            return (
            // 服务器声明为只读的列：与其他列一样列出，按原样显示。
            f.widget === "fixed" ? (
              // 选项列使用其自身的名称（如「图层名」，而非 name）；其余为使用者数据（序列名、图层名），可截断
              <span className={`ptable-fixed${off ? " inactive" : ""}`} key={f.name} {...(f.options ? {} : { "data-user-data": true })}>
                {fixedText(f, row[f.name])}
              </span>
            ) : (
              <span className={`ptable-cell${f.widget === "choice" ? " wide" : ""}${off ? " inactive" : ""}`} key={f.name}>
                <fieldset className="ptable-field" disabled={off}>
                  <Control nodeId={nodeId} at={`${p.name}[${i}].${f.name}`} p={{ ...f, widget: f.options ? "select" : f.widget }} value={row[f.name]} set={(v) => edit(i, f.name, v)}
                    choice={f.widget === "choice" ? cell(row) ?? undefined : undefined} />
                </fieldset>
              </span>
            ));
          })}
          {hasDrop && (
            <button className="ptable-drop" aria-label="去掉" onClick={() => set(rows.filter((_, j) => j !== i))}>
              <IconClose size={9} />
            </button>
          )}
          {head && again && blank("again")}
        </div>
      ))}
      {p.choices_from.length > 0 && !choice && <span className="class-hint">上游算过以后，这里列出人物的关节和自动猜到的是哪个</span>}
      {!head && again}
      {/* 「添加」：与节点上点「＋」同一个动作（graph/edit.ts addEmptyRow），末尾加一空行 = 多一个输入口 */}
      {canAdd && (
        <Button tone="ghost" layout="ptable-add" onClick={() => addEmptyRow(nodeId)}>
          添加
        </Button>
      )}
    </div>
  );
}
