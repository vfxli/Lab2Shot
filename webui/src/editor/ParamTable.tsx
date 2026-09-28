/** 表格参数 (widget "table"): a parameter that is a list of entries, such as 「读取多条序列」's sequences,
 * 「多层 EXR 输出设置」's 图层, the camera's distortion parameters, or the joint mapping.
 *
 * 「读取序列」没有图层表：输出口直接由文件中的图层生成（nodes/core/input.py made_ports）。 */

import type { ParamDef } from "../api";
import { deriveParams } from "../graph/actions";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { greyed } from "../api/applies";
import { useRowDrag } from "../ui/rowDrag";
import { useChoices } from "../ui/choices";
import { IconClose } from "../ui/icons";
import { Button } from "../ui/Button";
import { Control, type CellChoice } from "./ParamControls";

type Entry = Record<string, unknown>;

/** A list of entries (P over a list of a model): one line per entry, with every column the server declares for it: the
 * editable ones as controls, the others (`widget: "fixed"`) as plain text (dropping a fixed column would hide
 * which sequence a row is and how many frames it has). Which
 * fields apply to a row is the server's to say (nodes/applies.py resolves each row: `layers[2].scale` among the node's
 * applies), never worked out here. Entries can be taken out; a parameter the node works out from others (derived_from)
 * can be listed again from them.
 *
 * `name` is the row's identity, not a column: it is the key, and the row's own name when nothing else names it. */

/** A read-only cell's text: a column of choices says the word its options are called by, anything else itself. */
function fixedText(f: ParamDef, v: unknown): string {
  const said = String(v ?? "");
  return f.options ? f.option_labels?.[said] ?? said : said;
}

export function TableParam({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const rows = (value as Entry[] | null) ?? [];
  // 声明了 panel=false 的列不在面板中占列（读取节点的「图层」：它显示在「文件里」一列的最前面）
  const columns = (p.items ?? []).filter((f) => f.panel !== false);
  const answer = useResults((s) => s.results[nodeId]?.applies);
  // 控件不得显隐切换：行中不起作用的格仍占据所在列，置灰；否则同一张表各行列数不同，
  // 使用者无法记住各列含义
  const inactive = (f: ParamDef, i: number) => greyed(answer, `${p.name}[${i}].${f.name}`);
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const def = getNodeDefs()[typeId];
  // What names the row on its left: a 名称 the file gave it, else the row's own id, but only when neither can be
  // edited. A table whose name is typed (「多层 EXR 输出设置」's 图层名) or that names its rows by a column of its own
  // (「读取多条序列」's 序列) has no name span: every column is a cell, in the order the server declares them.
  const labelCol = columns.find((f) => f.name === "label");
  const nameCol = columns.find((f) => f.name === "name");
  const leads = labelCol?.widget === "fixed" ? labelCol : !labelCol && nameCol?.widget === "fixed" ? nameCol : null;
  // a row's `name` it cannot edit and that does not name it is its key (an output's port id): not a column to read
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
  const hasDrop = hasGrip;
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
            // A column the server declares read-only: listed like any other, shown as it is. Its text is the user's
            // own data (a sequence's name, a layer's), so it may be truncated.
            f.widget === "fixed" ? (
              // 选项列使用其自身的名称（如「图层名」，而非 name）；其余为使用者数据（序列名、图层名），可截断
              <span className={`ptable-fixed${off ? " inactive" : ""}`} key={f.name} {...(f.options ? {} : { "data-user-data": true })}>
                {fixedText(f, row[f.name])}
              </span>
            ) : (
              <span className={`ptable-cell${f.widget === "choice" ? " wide" : ""}${off ? " inactive" : ""}`} key={f.name}>
                <fieldset className="ptable-field" disabled={off}>
                  <Control nodeId={nodeId} p={{ ...f, widget: f.options ? "select" : f.widget }} value={row[f.name]} set={(v) => edit(i, f.name, v)}
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
    </div>
  );
}
