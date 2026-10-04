/** 表情对应（widget "expression_map"，「表情重定向」的 mapping）：一张参数表格，每个表情槽一行，「曲线 → 形变」两个
 * 下拉。用参数表格的同一套排法（editor/styles/28-param-table.css 的 ptable：一张表一套列，表头与各行对齐）；选项是
 * 服务端给的（NodeDef.choices 里这个参数的那一项，lab2shot/nodes/kit/rig_map.py expression_choice：每个槽能用的源
 * 曲线、目标的全部形变、每个槽推测的那一对）。
 *
 * 值的约定与骨架的对应关系相同（服务端 data/expressions.py merged_rows）：参数里写了的槽以参数为准，没写的按推测；
 * 一行改得与推测一样就不写；一行也没有为 null（= 全部自动）。每个下拉的第一项「自动 · X」是这一栏推测的那个，「不配」
 * 是这一栏留空（这个槽不驱动，目标那个形变保持 0）。 */

import type { ExpressionSlot, ParamDef, RigRow } from "../api";
import { mapRowsOf } from "../model/rigPair";
import { useChoices } from "../ui/choices";
import { Select } from "../ui/Select";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const AUTO = "\u0001auto";
const NONE = "\u0001none";
const WORD = { src: "ui.params.expr.curve", dst: "ui.params.expr.shape" } as const;

export function ExpressionMap({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const choice = useChoices(nodeId, p);
  const slots: ExpressionSlot[] | undefined = choice?.rows;
  if (!slots) return <span className="class-hint">{choice?.empty || t("ui.params.expr.after_cook")}</span>;
  const shapes = choice?.shapes ?? [];
  const given = new Map(mapRowsOf(value).map((r) => [r.part, r] as [string, RigRow]));
  const auto = (q: ExpressionSlot): RigRow => q.auto ?? { part: q.id, src: [], dst: [] };
  const rowOf = (q: ExpressionSlot): RigRow => given.get(q.id) ?? auto(q);
  const same = (a: RigRow, b: RigRow) => a.src.join("\n") === b.src.join("\n") && a.dst.join("\n") === b.dst.join("\n");
  const write = (q: ExpressionSlot, col: "src" | "dst", names: string[]) => {
    const next = { ...rowOf(q), part: q.id, [col]: names };
    const rows = slots.flatMap((s) => {
      const r = s.id === q.id ? next : given.get(s.id);
      return r && !same(r, auto(s)) ? [r] : [];
    });
    // 表情槽以外的行（换过源以后留下的）原样保留：计算时由服务端报出
    const rest = [...given.values()].filter((r) => !slots.some((s) => s.id === r.part));
    const out = [...rows, ...rest];
    set(out.length ? out : null);
  };
  const cell = (q: ExpressionSlot, col: "src" | "dst") => {
    const names = col === "src" ? q.curves : shapes;
    const now = rowOf(q)[col];
    const guessed = auto(q)[col];
    const current = !given.has(q.id) ? AUTO : now.length ? now[0] : NONE;
    const missing = now.length > 0 && !names.includes(now[0]);
    const options = [
      { value: AUTO, label: t("ui.params.expr.auto", { guess: guessed[0] ?? t("ui.params.expr.none") }) },
      { value: NONE, label: t("ui.params.expr.none") },
      ...(missing ? [{ value: now[0], label: t("ui.params.choice_gone", { value: now[0] }), tip: tipOf("error", t("ui.params.choice_gone_tip")) }] : []),
      ...names.map((n) => ({ value: n, label: n })),
    ];
    return (
      <span className="ptable-cell wide">
        <Select className="choice names" data-user-data label={t("ui.params.expr.cell", { expression: q.label, what: t(WORD[col]) })} value={current} options={options}
          onPick={(v) => write(q, col, v === AUTO ? [...guessed] : v === NONE ? [] : [v])} />
      </span>
    );
  };
  return (
    <div className="ptable" style={{ gridTemplateColumns: "minmax(64px, max-content) minmax(96px, 1fr) minmax(96px, 1fr)" }}>
      <div className="ptable-row ptable-head">
        <span className="ptable-name" aria-hidden>{t("ui.params.expr.expression")}</span>
        <span className="ptable-cell wide" aria-hidden>{t(WORD.src)}</span>
        <span className="ptable-cell wide" aria-hidden>{t(WORD.dst)}</span>
      </div>
      {slots.map((q) => (
        <div className="ptable-row" key={q.id}>
          <span className="ptable-name" data-user-data {...tipAttrs(tipOf("truncated", q.label))}>{q.label}</span>
          {cell(q, "src")}
          {cell(q, "dst")}
        </div>
      ))}
    </div>
  );
}
