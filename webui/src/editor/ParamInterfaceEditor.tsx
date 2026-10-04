/** 「编辑参数界面」：模板的公开参数怎么摆给用模板的人（Houdini「Edit Parameter Interface」式）。弹窗框架
 * editor/ParamSheet.tsx 的一个编辑器，由参数面板没选中节点时标题行最右的小按钮打开（ParamPanel.tsx，ParamSheet.tsx
 * SheetWindow）；它编辑的不是某个节点的参数，而是节点图的 `exposed` 这棵树（api/catalog.ts ExposedEntry），所以不按 widget
 * 名登记在 SHEET_EDITORS 里。「确定」整棵写回，「取消」/ Esc / 点遮罩不写回。
 *
 * 左边是树：拖动排序（拖到参数上 = 放到它前面，拖到组上 = 放进这个组的末尾，拖到最下面一行 = 放到最外层末尾）；上移 /
 * 下移 / 移出组；新建组（在选中项后面）、新建子组（在选中的组里）、删除组（组里的东西留下，放回组原来的位置）。右边是
 * 选中项的属性：组的名字和是否默认折叠；参数的显示名、控件覆盖（默认 / 下拉 / 复选框）及下拉的选项表（值和显示名都由
 * 模板作者自己填）、条件表达式（platform/conditions.ts，即时解析）。有问题的项在树上和属性里标红并说明：规则与服务端
 * 保存模板时的校验（lab2shot/engine/templates.py check_exposed）逐条对应，权威在服务端。
 *
 * 公开 / 取消公开用参数旁的图钉（graph/edit.ts toggleExposed：新公开的进根下末尾）；这里不增删参数。 */

import { LabelRow } from "../ui/LabelRow";
import { Num, optionView, paramNum } from "../ui/controls";
import { licensedValues, optionName, registeredValues } from "../graph/rules";
import { useResults } from "../state/results";
import { useMemo, useState, useId } from "react";
import type { ExposedEntry, ExposedGroup, ExposedOption, ExposedParam, NodeTypeDef, ParamDef, Words } from "../api";
import { entryLabel, exposedParams, isGroup, splitTarget, targetsOf, useCookInputs } from "../state/cookInputs";
import { listSep } from "../i18n/words";
import { at, canDropAt, dropAt, insertAt, mergeInto, move, removeAt, replaceAt, rowsOf, samePath, type Path } from "../graph/exposedTree";
import { getNodeDefs } from "../state/catalog";
import { compared, conditionProblem, neverValue, parseCondition, condEqual, valueRefused, widgetRefused } from "../platform/conditions";
import { useRowDrag } from "../ui/rowDrag";
import { Button, Segmented, Switch } from "../ui/Button";
import { Select } from "../ui/Select";
import { SheetFoot, useDraft, Writes, type SheetEditor, type SheetEditorProps } from "./ParamSheet";
import { nodeRef } from "../graph/naming";
import { pick, t } from "../i18n/t";
import { edited, newWords } from "../state/cookInputs";
import { tipAttrs, tipOf } from "../platform/tips";

/** 弹窗框架的编辑器都拿一个参数定义（SheetEditorProps.p）：这棵树不是任何节点的参数，这里给一个只有名字的，编辑器不读它。 */
export const INTERFACE_PARAM: ParamDef = {
  name: "exposed", label: "", type: "array", nullable: false, minimum: null, maximum: null, open_minimum: false, open_maximum: false, multiple_of: null,
  options: null, option_labels: null, widget: "", parts: [], lines: 1, group: "", affects_result: false,
  placeholder: "", accept: [], unit: "", derived_from: [], unique: false, choices_from: [], wire: "", simple: "",
  per_frame: false, panel: true, overrides: [], items: null,
};

// ------------------------------------------------------------------ 问题（与服务端 check_exposed 逐条对应，权威在服务端）

interface Target {
  id: string; // 节点的 id
  node: string; // 节点在图上的名字
  def: NodeTypeDef;
  p: ParamDef;
}

/** One node parameter an entry drives (`key`: `node.param`; null when it is not there). */
function targetAt(key: string): Target | null {
  const [nid, pname] = splitTarget(key);
  const n = useCookInputs.getState().nodes[nid];
  const def = n && getNodeDefs()[n.typeId];
  const p = def?.params.find((q) => q.name === pname);
  return n && def && p ? { id: nid, node: nodeRef(nid, n.typeId), def, p } : null;
}

/** The node parameter the entry shows (its first target: state/cookInputs.ts firstTarget). */
const targetOf = (x: ExposedParam): Target | null => targetAt(targetsOf(x)[0] ?? "");

// what makes two node parameters one kind (the server's engine/templates.py _SHAPE_KEYS): one entry may drive both
const KIND_KEYS = ["type", "nullable", "minimum", "maximum", "open_minimum", "open_maximum", "multiple_of", "options", "widget", "unit", "items", "action", "target"] as const;
const kindOf = (p: ParamDef) => JSON.stringify(KIND_KEYS.map((k) => p[k] ?? null));

/** Why `other` cannot be merged into `x` (「合并」: one entry driving both one's and the other's parameters), null when it
 * can: a button holds no value to share; the two must be one kind of parameter (what the server checks: E-EXPOSED-TARGETS). */
function mergeRefused(x: ExposedParam, other: ExposedParam): string | null {
  const a = targetOf(x);
  const b = targetOf(other);
  if (!a || !b) return t("ui.interface.merge_gone");
  if (a.p.widget === "button" || b.p.widget === "button") return t("ui.interface.merge_button");
  if (kindOf(a.p) !== kindOf(b.p)) return t("ui.interface.merge_unlike");
  // the entries' own controls (a menu or checkbox over the parameter, with its own items) must be one too: merged, the
  // one entry left writes its value into the other's parameter, which would get a value its menu does not offer
  const own = (q: ExposedParam) => JSON.stringify([q.widget ?? null, q.widget ? (q.options ?? []).map((o) => o.value) : null]);
  return own(x) === own(other) ? null : t("ui.interface.merge_menu");
}


/** 公开参数 `name` 永远不会是 `value` 的原因（null：可能是，或说不准）：规则在 platform/conditions.ts neverValue（与服务端同一份）。 */
function never(name: string, value: unknown, all: ExposedParam[]): string | null {
  const x = all.find((q) => q.name === name);
  const tg = x ? targetOf(x) : null;
  return x && tg ? neverValue(name, value, x.options, tg.p) : null;
}

/** 一个 Hide When / Disable When 的问题（null：没有）：写法、用到的名字（conditionProblem），再看拿参数比的字面量它取不取得到。 */
function exprProblem(text: string | undefined, all: ExposedParam[]): string | null {
  const names = all.map((x) => x.name);
  const wrong = conditionProblem(text, names);
  if (wrong || !String(text ?? "").trim()) return wrong;
  for (const [n, v] of compared(parseCondition(text))) {
    const why = never(n, v, all);
    if (why) return why;
  }
  return null;
}

export function problemsOf(entry: ExposedEntry, all: ExposedParam[], dupes: Set<string>): string[] {
  if (isGroup(entry)) return pick(entry.label).trim() ? [] : [t("ui.interface.problem.group_name")];
  const out: string[] = [];
  if (!entry.name.trim()) out.push(t("ui.interface.problem.name_empty"));
  else if (dupes.has(entry.name)) out.push(t("ui.interface.problem.name_dupe", { name: entry.name }));
  const tg = targetOf(entry);
  for (const key of targetsOf(entry)) if (!targetAt(key)) out.push(t("ui.interface.problem.target_gone", { target: key }));
  if (tg && entry.widget) {
    const why = widgetRefused(entry.widget, tg.p, entry.options);
    if (why) out.push(t(entry.widget === "menu" ? "ui.interface.problem.no_menu" : "ui.interface.problem.no_checkbox", { why }));
  }
  const opts = entry.options ?? [];
  opts.forEach((o, i) => {
    const why = !pick(o.label).trim() ? t("ui.interface.problem.option_label")
      : (tg ? valueRefused(tg.p, o.value) : null) ?? (opts.slice(0, i).some((q) => condEqual(q.value, o.value)) ? t("ui.interface.problem.option_dupe") : null);
    // 这一项自己的 Hide When：规则同条目级（exprProblem），文字同服务端 check_exposed
    const hidden = why ? null : o.hide_when !== undefined ? exprProblem(o.hide_when, all) : null;
    if (why || hidden) out.push(t("ui.interface.problem.option", { n: i + 1, value: JSON.stringify(o.value), why: why ?? t("ui.interface.problem.option_hide", { when: o.hide_when ?? "", why: hidden ?? "" }) }));
    // 这一项自己的 Disable When 与它的原因：规则同服务端 check_exposed（有条件就要写为什么）
    const off = why || hidden ? null : o.disable_when !== undefined ? exprProblem(o.disable_when, all)
      ?? (!pick(o.disable_why).trim() ? t("ui.interface.problem.disable_why") : null) : null;
    if (off) out.push(t("ui.interface.problem.option", { n: i + 1, value: JSON.stringify(o.value), why: t("ui.interface.problem.option_disable", { when: o.disable_when ?? "", why: off }) }));
  });
  const hide = exprProblem(entry.hide_when, all);
  if (hide) out.push(t("ui.interface.problem.hide", { why: hide }));
  const disable = exprProblem(entry.disable_when, all);
  if (disable) out.push(t("ui.interface.problem.disable", { why: disable }));
  return out;
}

// ------------------------------------------------------------------ 编辑器

function InterfaceEditor({ value, set, cancel }: SheetEditorProps) {
  // 草稿：改不了（只读标签页）时改它不生效，已有的留着（ParamSheet.tsx useDraft）；选中哪一项是看，照常
  const [tree, setTree] = useDraft<ExposedEntry[]>(() => structuredClone((value as ExposedEntry[]) ?? []));
  const [sel, setSel] = useState<Path | null>(null);
  const rows = rowsOf(tree);
  const all = exposedParams(tree);
  const names = all.map((x) => x.name);
  const dupes = new Set(names.filter((n, i) => names.indexOf(n) !== i));
  const problems = new Map(rows.map((r) => [r.path.join("/"), problemsOf(r.entry, all, dupes)]));
  const bad = [...problems.values()].filter((p) => p.length).length;
  const chosen = sel ? at(tree, sel) : undefined;

  // 拖动：行 i 是 rows[i]；最后多一行「放到最外层末尾」（下标 rows.length）
  const moved = ([next, now]: [ExposedEntry[], Path]) => (setTree(next), setSel(now));
  const drag = useRowDrag((from, to) => moved(dropAt(tree, rows, from, to)), (from, to) => canDropAt(rows, from, to));
  const place = (src: Path, parent: Path, index: number) => moved(move(tree, src, parent, index));

  const parentOf = (p: Path) => p.slice(0, -1);
  const last = (p: Path) => p[p.length - 1];
  const siblings = (p: Path) => (p.length > 1 ? (at(tree, parentOf(p)) as ExposedGroup).children : tree);

  const newGroup = (sub: boolean) => {
    const g: ExposedGroup = { kind: "group", label: newWords(t(sub ? "ui.interface.new_subgroup_label" : "ui.interface.new_group_label")), collapsed: false, children: [] };
    if (sub && sel && chosen && isGroup(chosen)) {
      setTree(insertAt(tree, sel, chosen.children.length, [g]));
      setSel([...sel, chosen.children.length]);
    } else if (sel) {
      setTree(insertAt(tree, parentOf(sel), last(sel) + 1, [g]));
      setSel([...parentOf(sel), last(sel) + 1]);
    } else {
      setTree([...tree, g]);
      setSel([tree.length]);
    }
  };
  const shift = (d: -1 | 1) => {
    if (!sel) return;
    const i = last(sel) + d;
    if (i < 0 || i >= siblings(sel).length) return;
    place(sel, parentOf(sel), d > 0 ? i + 1 : i);
  };
  const outdent = () => {
    if (!sel || sel.length < 2) return;
    const g = parentOf(sel);
    place(sel, parentOf(g), last(g) + 1);
  };
  const dissolve = () => {
    if (!sel || !chosen || !isGroup(chosen)) return;
    setTree(insertAt(removeAt(tree, sel), parentOf(sel), last(sel), chosen.children));
    setSel(null);
  };
  const edit = (entry: ExposedEntry) => sel && setTree(replaceAt(tree, sel, entry));
  // 「合并」：选中的这一项留下，`name` 那一项并进来（graph/exposedTree.ts mergeInto）
  const merge = (name: string) => {
    const from = rows.find((r) => !isGroup(r.entry) && r.entry.name === name)?.path;
    if (sel && from) moved(mergeInto(tree, sel, from));
  };

  const canUp = !!sel && last(sel) > 0;
  const canDown = !!sel && last(sel) < siblings(sel).length - 1;
  return (
    <>
      <div className="pif">
        <div className="pif-left">
          <div className="pif-tools">
            <Writes>
            <Button size="sm" tone="ghost" onClick={() => newGroup(false)}>{t("ui.interface.new_group")}</Button>
            <Button size="sm" tone="ghost" disabled={!chosen || !isGroup(chosen)} onClick={() => newGroup(true)}>{t("ui.interface.new_subgroup")}</Button>
            <span className="pif-gap" />
            <Button size="sm" tone="ghost" disabled={!canUp} onClick={() => shift(-1)}>{t("ui.interface.move_up")}</Button>
            <Button size="sm" tone="ghost" disabled={!canDown} onClick={() => shift(1)}>{t("ui.interface.move_down")}</Button>
            <Button size="sm" tone="ghost" disabled={!sel || sel.length < 2} onClick={outdent}>{t("ui.interface.outdent")}</Button>
            <Button size="sm" tone="ghost" danger disabled={!chosen || !isGroup(chosen)} tip={tipOf("consequence", t("ui.interface.dissolve_tip"))} onClick={dissolve}>{t("ui.interface.dissolve")}</Button>
            </Writes>
          </div>
          <div className="pif-tree" role="tree" aria-label={t("ui.interface.title_short")}>
            {rows.length === 0 && <div className="pif-empty">{t("ui.interface.empty")}</div>}
            {rows.map((r, i) => {
              const key = r.path.join("/");
              const wrong = (problems.get(key) ?? []).length > 0;
              return (
                <div key={key} role="treeitem" aria-selected={samePath(sel, r.path)}
                  className={`pif-row${isGroup(r.entry) ? " is-group" : ""}${samePath(sel, r.path) ? " on" : ""}${wrong ? " wrong" : ""}${drag.over === i ? " drag-over" : ""}`}
                  style={{ paddingLeft: 8 + r.depth * 16 }} onClick={() => setSel(r.path)} {...drag.row(i)}>
                  <span className="pif-grip" {...drag.grip(i)} {...tipAttrs(tipOf("shortcut", t("ui.interface.drag_tip")))}>⋮⋮</span>
                  {isGroup(r.entry) ? (
                    <>
                      <span className="pif-kind">{t("ui.interface.kind.group")}</span>
                      <span className="pif-name" data-user-data>{pick(r.entry.label) || t("ui.interface.no_name")}</span>
                    </>
                  ) : (
                    <>
                      <span className="pif-name" data-user-data>{entryLabel(r.entry)}</span>
                      <span className="pif-detail" data-user-data>{r.entry.name}</span>
                      {r.entry.widget && <span className="pif-kind">{t(r.entry.widget === "menu" ? "ui.interface.kind.menu" : "ui.interface.kind.checkbox")}</span>}
                      {targetOf(r.entry)?.p.widget === "button" && <span className="pif-kind">{t("ui.interface.kind.button")}</span>}
                      {r.entry.hide_when && <span className="pif-kind">{t("ui.interface.kind.hide")}</span>}
                      {r.entry.disable_when && <span className="pif-kind">{t("ui.interface.kind.disable")}</span>}
                      {r.entry.show_on_change && <span className="pif-kind">{t("ui.interface.kind.show")}</span>}
                    </>
                  )}
                </div>
              );
            })}
            {rows.length > 0 && (
              <div className={`pif-row pif-end${drag.over === rows.length ? " drag-over" : ""}`} {...drag.row(rows.length)}>
                {t("ui.interface.drop_end")}
              </div>
            )}
          </div>
        </div>
        <div className="pif-right">
          {!chosen && <div className="pif-empty">{t("ui.interface.pick_one")}</div>}
          <Writes>
            {chosen && isGroup(chosen) && <GroupProps g={chosen} onChange={edit} problems={problems.get(sel!.join("/")) ?? []} />}
            {chosen && !isGroup(chosen) && <ParamProps key={sel!.join("/")} x={chosen} onChange={edit} onMerge={merge} all={all} problems={problems.get(sel!.join("/")) ?? []} />}
          </Writes>
        </div>
      </div>
      <SheetFoot ok={() => set(tree)} cancel={cancel}>
        {bad > 0 && <span className="pif-bad">{t("ui.interface.bad", { count: bad })}</span>}
      </SheetFoot>
    </>
  );
}

function Problems({ list }: { list: string[] }) {
  return list.length ? (
    <ul className="pif-problems">
      {list.map((p) => (
        <li key={p}>{p}</li>
      ))}
    </ul>
  ) : null;
}

function GroupProps({ g, onChange, problems }: { g: ExposedGroup; onChange: (g: ExposedGroup) => void; problems: string[] }) {
  const id = useId();
  return (
    <div className="pif-props lgrid">
      <LabelRow className="pif-field" labelAs="label" htmlFor={`${id}-name`} label={t("ui.interface.group_name")}>
        <input id={`${id}-name`} className="field" value={pick(g.label)} spellCheck={false} onChange={(e) => onChange({ ...g, label: edited(g.label, e.target.value) })} />
      </LabelRow>
      <LabelRow className="pif-field" label={t("ui.interface.collapsed")}>
        <Switch on={!!g.collapsed} onChange={(on) => onChange({ ...g, collapsed: on })} label={t("ui.interface.collapsed")} />
      </LabelRow>
      <Problems list={problems} />
    </div>
  );
}

type WidgetChoice = "default" | "menu" | "checkbox";

function ParamProps({ x, onChange, onMerge, all, problems }: {
  x: ExposedParam; onChange: (x: ExposedParam) => void; onMerge: (name: string) => void; all: ExposedParam[]; problems: string[];
}) {
  const names = all.map((q) => q.name);
  const keys = targetsOf(x).join("+");
  const tg = useMemo(() => targetOf(x), [keys]); // eslint-disable-line react-hooks/exhaustive-deps
  const driven = useMemo(() => targetsOf(x).map((k) => ({ k, tg: targetAt(k) })), [keys]); // eslint-disable-line react-hooks/exhaustive-deps
  const widget: WidgetChoice = x.widget ?? "default";
  const setWidget = (w: WidgetChoice) => {
    const { widget: _w, options: _o, ...rest } = x;
    void _w;
    void _o;
    if (w === "default") return onChange(rest);
    // 复选框接整数参数时，选项就是 0 和 1 两项；下拉先按参数自己的可选值（或 0 / 1）填好，作者再改显示名
    const seed = w === "checkbox"
      ? tg?.p.type === "integer" ? [{ value: 0, label: t("ui.common.off") }, { value: 1, label: t("ui.common.on") }] : undefined
      : x.options?.length ? x.options : tg?.p.options?.map((o) => ({ value: o as unknown, label: optionName(tg.p, o) }))
        ?? (tg?.p.type === "boolean" ? [{ value: true, label: t("ui.common.yes") }, { value: false, label: t("ui.common.no") }] : [{ value: tg ? sample(tg.p) : "", label: "" }]);
    onChange({ ...rest, widget: w, ...(seed ? { options: seed } : {}) });
  };
  const options = x.options ?? [];
  const setOptions = (o: ExposedOption[]) => onChange({ ...x, options: o });
  const button = tg?.p.widget === "button";
  const id = useId();
  return (
    <div className="pif-props lgrid">
      <LabelRow className="pif-field" labelAs="label" htmlFor={`${id}-label`} label={t("ui.interface.label")}>
        <input id={`${id}-label`} className="field" value={entryLabel(x)} spellCheck={false}
          // a label typed over a shared button word (「打包」) replaces it: the word goes, the text is the entry's own
          onChange={(e) => { const { word: _w, ...rest } = x; void _w; onChange({ ...rest, label: edited(x.word ? newWords(entryLabel(x)) : x.label, e.target.value) }); }} />
      </LabelRow>
      {/* 作者的说明：行下常显的一句（含义、许可后果、限制），不需要就空着 */}
      <LabelRow className="pif-field top" labelAs="label" htmlFor={`${id}-note`} label={t("ui.interface.note")}>
        <textarea id={`${id}-note`} className="field" rows={2} value={pick(x.note)} spellCheck={false}
          onChange={(e) => {
            const { note: _n, ...rest } = x;
            void _n;
            const next = edited(x.note, e.target.value); // the page's language; the other kept
            onChange(Object.values(next).some((v) => String(v).trim()) ? { ...x, note: next } : rest);
          }} />
      </LabelRow>
      <LabelRow className="pif-field" label={t("ui.interface.name")}>
        <span className="pif-note" data-user-data>{x.name}<small>{t("ui.interface.name_note", { name: x.name })}</small></span>
      </LabelRow>
      <LabelRow className="pif-field" label={t("ui.interface.target")}>
        {/* 一项驱动多个参数时逐个列出（值写进每一个，显示第一个的） */}
        <span className="pif-note" data-user-data>{driven.map(({ k, tg: one }) => (one ? t("ui.interface.target_of", { node: one.node, param: one.p.label }) : k)).join(listSep())}</span>
      </LabelRow>
      {!button && (
        // 「合并」：另一项并进这一项，一个值写进两边的参数（「确定」时另一边取这一项现在的值：graph/edit.ts setInterface）
        <LabelRow className="pif-field" label={t("ui.interface.merge")}
          below={<div className="pif-hint lrow-under">{t("ui.interface.merge_note", { label: entryLabel(x) })}</div>}>
          <Select label={t("ui.interface.merge")} value="" onPick={onMerge}
            options={[{ value: "", label: t("ui.interface.merge_pick"), off: true },
              ...all.filter((q) => q.name !== x.name).map((q) => {
                const why = mergeRefused(x, q);
                return { value: q.name, label: `${entryLabel(q)} · ${q.name}`, off: !!why, tip: why ? tipOf("disabled", why) : undefined };
              })]} />
        </LabelRow>
      )}
      {!button && <LabelRow className="pif-field" label={t("ui.interface.widget")}>
        <Segmented label={t("ui.interface.widget")} value={widget} onChange={setWidget}
          options={[
            { value: "default" as WidgetChoice, label: t("ui.interface.widget_default") },
            { value: "menu" as WidgetChoice, label: t("ui.interface.kind.menu"), disabled: tg && widgetRefused("menu", tg.p, [{ value: 0 }]) },
            { value: "checkbox" as WidgetChoice, label: t("ui.interface.kind.checkbox"), disabled: tg && tg.p.type !== "boolean" && tg.p.type !== "integer" ? t("ui.interface.checkbox_unfit") : false },
          ]} />
      </LabelRow>}
      {x.widget && (x.widget === "menu" || tg?.p.type === "integer") && (
        <LabelRow className="pif-field top" label={t(x.widget === "menu" ? "ui.interface.options" : "ui.interface.two_values")}>
          <div className="pif-options">
            {options.map((o, i) => (
              <div className="pif-option" key={i}>
                <OptionValue tg={tg} value={o.value} onChange={(v) => setOptions(options.map((q, j) => (j === i ? { ...q, value: v } : q)))} />
                <input className="field" placeholder={t("ui.interface.label")} value={pick(o.label)} spellCheck={false}
                  onChange={(e) => setOptions(options.map((q, j) => (j === i ? { ...q, label: edited(q.label, e.target.value) } : q)))} />
                {x.widget === "menu" && (
                  <Button size="sm" tone="ghost" onClick={() => setOptions(options.filter((_, j) => j !== i))}>{t("ui.common.remove")}</Button>
                )}
                {x.widget === "menu" && (
                  <OptionHide value={o.hide_when} all={all}
                    onChange={(v) => setOptions(options.map((q, j) => (j === i ? withOptionHide(q, v) : q)))} />
                )}
                {x.widget === "menu" && (
                  <OptionDisable o={o} all={all} onChange={(next) => setOptions(options.map((q, j) => (j === i ? next : q)))} />
                )}
              </div>
            ))}
            {x.widget === "menu" && (
              <Button size="sm" tone="ghost" onClick={() => setOptions([...options, { value: tg ? sample(tg.p) : "", label: "" }])}>{t("ui.common.add")}</Button>
            )}
          </div>
        </LabelRow>
      )}
      {!button && (
        <LabelRow className="pif-field" label={t("ui.interface.show_node")}>
          <label className="pif-check">
            <Switch on={!!x.show_on_change} label={t("ui.interface.show_on_change")}
              onChange={(on) => onChange(on ? { ...x, show_on_change: true } : (({ show_on_change: _s, ...rest }) => (void _s, rest))(x))} />
            <span>{t("ui.interface.show_on_change_note")}</span>
          </label>
        </LabelRow>
      )}
      <Expression label={t("ui.interface.hide_when")} value={x.hide_when} all={all} placeholder={t("ui.interface.hide_placeholder")}
        onChange={(v) => onChange(withExpr(x, "hide_when", v))} />
      <Expression label={t("ui.interface.disable_when")} value={x.disable_when} all={all} placeholder={t("ui.interface.disable_placeholder")}
        onChange={(v) => onChange(withExpr(x, "disable_when", v))} />
      <div className="pif-hint">
        {t("ui.interface.hint_head")}
        <code>==</code> <code>!=</code> <code>in [1, 2]</code> <code>not in</code>{" "}
        <code>&lt;</code> <code>&lt;=</code> <code>&gt;</code> <code>&gt;=</code> <code>and</code> <code>or</code> <code>not</code> {t("ui.interface.hint_parens")}
        {t("ui.interface.hint_tail", { names: names.filter((n) => n !== x.name).join(", ") || t("ui.interface.no_other_names") })}
      </div>
      <Problems list={problems} />
    </div>
  );
}

/** 一个表达式框（Hide When / Disable When）：即时解析，错了红框、下面说哪里错（写法、名字、比的值取不到：exprProblem）。 */
function Expression({ label, value, all, placeholder, onChange }: {
  label: string; value: string | undefined; all: ExposedParam[]; placeholder: string; onChange: (v: string) => void;
}) {
  const wrong = exprProblem(value, all);
  const id = useId();
  return (
    <LabelRow className="pif-field" labelAs="label" htmlFor={id} label={label} below={wrong && <div className="pif-problems lrow-under">{wrong}</div>}>
      <input id={id} className={`field mono${wrong ? " wrong" : ""}`} value={value ?? ""} spellCheck={false} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)} />
    </LabelRow>
  );
}

/** 下拉一项自己的 Hide When（成立时下拉里不列这一项，例如「G1 机器人时只留原骨架」）：一行表达式框，即时标错（exprProblem）。 */
function OptionHide({ value, all, onChange }: { value: string | undefined; all: ExposedParam[]; onChange: (v: string) => void }) {
  const wrong = exprProblem(value, all);
  return (
    <>
      <input className={`field mono pif-option-hide${wrong ? " wrong" : ""}`} value={value ?? ""} spellCheck={false}
        placeholder={t("ui.interface.option_hide_placeholder")} onChange={(e) => onChange(e.target.value)} />
      {wrong && <div className="pif-problems pif-option-hide">{wrong}</div>}
    </>
  );
}

/** 一项下拉选项自己的 Disable When 和置灰时说的为什么（两个一起写；条件空着 = 不置灰，为什么也一起去掉）。 */
function OptionDisable({ o, all, onChange }: { o: ExposedOption; all: ExposedParam[]; onChange: (o: ExposedOption) => void }) {
  const wrong = o.disable_when ? exprProblem(o.disable_when, all) : null;
  const set = (when: string, why: Words) => {
    const { disable_when: _w, disable_why: _y, ...rest } = o;
    void _w; void _y;
    onChange(when.trim() ? { ...rest, disable_when: when, disable_why: why } : rest);
  };
  return (
    <>
      <input className={`field mono pif-option-hide${wrong ? " wrong" : ""}`} value={o.disable_when ?? ""} spellCheck={false}
        placeholder={t("ui.interface.option_disable_placeholder")} onChange={(e) => set(e.target.value, o.disable_why ?? "")} />
      {o.disable_when && <input className="field pif-option-hide" value={pick(o.disable_why)} spellCheck={false}
        placeholder={t("ui.interface.option_why_placeholder")} onChange={(e) => set(o.disable_when ?? "", edited(o.disable_why, e.target.value))} />}
      {wrong && <div className="pif-problems pif-option-hide">{wrong}</div>}
    </>
  );
}

function withOptionHide(o: ExposedOption, v: string): ExposedOption {
  const { hide_when: _old, ...rest } = o;
  void _old;
  return v.trim() ? { ...rest, hide_when: v } : rest;
}

/** 写上一个表达式；空着的去掉这个字段（没有条件）。 */
function withExpr(x: ExposedParam, key: "hide_when" | "disable_when", v: string): ExposedParam {
  const { [key]: _old, ...rest } = x;
  void _old;
  return v.trim() ? { ...rest, [key]: v } : rest;
}

/** 一个新选项的初值：目标参数的类型里最简单的那个值。 */
function sample(p: ParamDef): unknown {
  if (p.options?.length) return p.options[0];
  return p.type === "boolean" ? true : p.type === "integer" || p.type === "number" ? (p.minimum ?? 0) : "";
}

/** 下拉一项的值：目标参数有自己的可选值、或是布尔时从中选；数字和文字自己填（填的不合法照样留着，标红说明）。
 * 可选值照参数自己的下拉写（ui/controls.tsx optionView：许可词、现在不能选的原因），名称前加上值本身（作者要写的是值）。 */
function OptionValue({ tg, value, onChange }: { tg: Target | null; value: unknown; onChange: (v: unknown) => void }) {
  const p = tg?.p ?? null;
  const answer = useResults((s) => (tg ? s.results[tg.id]?.applies : undefined));
  if (tg && p && (p.options?.length || p.type === "boolean")) {
    const all: unknown[] = p.options?.length ? p.options : [true, false];
    const i = all.findIndex((o) => condEqual(o, value));
    const licensed = licensedValues(tg.def, p.name);
    const registered = registeredValues(tg.def, p.name);
    const rows = all.map((o, j) => {
      const named = optionName(p, o);
      // 作者挑的是以后用的值：现在不能选的也能挑，原因写在悬停里
      const view = optionView(p, o, answer, licensed, named === String(o) ? named : `${String(o)} · ${named}`, p.name, registered);
      return { value: String(j), label: view.label, tip: view.tip };
    });
    return (
      <Select label={t("ui.interface.value")} value={i < 0 ? "own" : String(i)} onPick={(v) => v !== "own" && onChange(all[Number(v)])}
        options={i < 0 ? [{ value: "own", label: t("ui.interface.value_invalid", { value: JSON.stringify(value) }) }, ...rows] : rows} />
    );
  }
  // 数字参数：页面唯一的数字框（ui/controls.tsx Num），按目标参数的声明规范化
  if (p && (p.type === "integer" || p.type === "number"))
    return <Num value={typeof value === "number" ? value : null} placeholder={t("ui.interface.value")} onChange={onChange} {...paramNum(p)} />;
  const text = typeof value === "string" ? value : JSON.stringify(value) ?? "";
  return (
    <input className={`field mono${p && valueRefused(p, value) ? " wrong" : ""}`} placeholder={t("ui.interface.value")} value={text} spellCheck={false}
      onChange={(e) => onChange(e.target.value)} />
  );
}

/** 弹窗框架里的这一种编辑器（ParamPanel.tsx 用 ParamSheet.tsx SheetWindow 打开）。 */
export const ParamInterface: SheetEditor = {
  editor: InterfaceEditor,
  title: () => t("ui.interface.title"),
  width: "content", // 树的每一行、右边每一项（含「对应参数」）完整显示（styles/30-param-interface.css .pif）
};
