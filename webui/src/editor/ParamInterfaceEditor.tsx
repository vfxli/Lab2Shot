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

import { Num, optionView, paramNum } from "../ui/controls";
import { licensedValues, optionName } from "../graph/rules";
import { useResults } from "../state/results";
import { useMemo, useState } from "react";
import type { ExposedEntry, ExposedGroup, ExposedOption, ExposedParam, NodeTypeDef, ParamDef } from "../api";
import { exposedParams, isGroup, useCookInputs } from "../state/cookInputs";
import { at, canDropAt, dropAt, insertAt, move, removeAt, replaceAt, rowsOf, samePath, type Path } from "../graph/exposedTree";
import { getNodeDefs } from "../state/catalog";
import { compared, conditionProblem, neverValue, parseCondition, condEqual, valueRefused, widgetRefused } from "../platform/conditions";
import { useRowDrag } from "../ui/rowDrag";
import { Button, Segmented, Switch } from "../ui/Button";
import { Select } from "../ui/Select";
import { SheetFoot, useDraft, Writes, type SheetEditor, type SheetEditorProps } from "./ParamSheet";

/** 弹窗框架的编辑器都拿一个参数定义（SheetEditorProps.p）：这棵树不是任何节点的参数，这里给一个只有名字的，编辑器不读它。 */
export const INTERFACE_PARAM: ParamDef = {
  name: "exposed", label: "参数界面", type: "array", nullable: false, minimum: null, maximum: null, open_minimum: false, open_maximum: false, multiple_of: null,
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

function targetOf(x: ExposedParam): Target | null {
  const [nid, pname] = x.target.split(".");
  const n = useCookInputs.getState().nodes[nid];
  const def = n && getNodeDefs()[n.typeId];
  const p = def?.params.find((q) => q.name === pname);
  return n && def && p ? { id: nid, node: n.label, def, p } : null;
}


/** 公开参数 `name` 永远不会是 `value` 的原因（null：可能是，或说不准）：规则在 platform/conditions.ts neverValue（与服务端同一份）。 */
function never(name: string, value: unknown, all: ExposedParam[]): string | null {
  const x = all.find((q) => q.name === name);
  const t = x ? targetOf(x) : null;
  return x && t ? neverValue(name, value, x.options, t.p) : null;
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
  if (isGroup(entry)) return entry.label.trim() ? [] : ["组要有名字"];
  const out: string[] = [];
  if (!entry.name.trim()) out.push("对外名字不能空着");
  else if (dupes.has(entry.name)) out.push(`对外名字「${entry.name}」重复了：命令行和插件按它传值，要各不相同`);
  const t = targetOf(entry);
  if (!t) out.push(`对应的参数 ${entry.target} 不存在了（节点删了或者参数没了）：去掉公开，或者重新公开`);
  if (t && entry.widget) {
    const why = widgetRefused(entry.widget, t.p, entry.options);
    if (why) out.push(`不能显示成${entry.widget === "menu" ? "下拉" : "复选框"}：${why}`);
  }
  const opts = entry.options ?? [];
  opts.forEach((o, i) => {
    const why = !String(o.label ?? "").trim() ? "要有显示名"
      : (t ? valueRefused(t.p, o.value) : null) ?? (opts.slice(0, i).some((q) => condEqual(q.value, o.value)) ? "和前面一项的值重复" : null);
    // 这一项自己的 Hide When：规则同条目级（exprProblem），文字同服务端 check_exposed
    const hidden = why ? null : o.hide_when !== undefined ? exprProblem(o.hide_when, all) : null;
    if (why || hidden) out.push(`第 ${i + 1} 项 ${JSON.stringify(o.value)}：${why ?? `Hide When「${o.hide_when}」${hidden}`}`);
  });
  const hide = exprProblem(entry.hide_when, all);
  if (hide) out.push(`Hide When ${hide}`);
  const disable = exprProblem(entry.disable_when, all);
  if (disable) out.push(`Disable When ${disable}`);
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
    const g: ExposedGroup = { kind: "group", label: sub ? "子组" : "新组", collapsed: false, children: [] };
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

  const canUp = !!sel && last(sel) > 0;
  const canDown = !!sel && last(sel) < siblings(sel).length - 1;
  return (
    <>
      <div className="pif">
        <div className="pif-left">
          <div className="pif-tools">
            <Writes>
            <Button size="sm" tone="ghost" tip="新建一个组：放在选中的那一项后面（没选时放到最后）" onClick={() => newGroup(false)}>新建组</Button>
            <Button size="sm" tone="ghost" disabled={!chosen || !isGroup(chosen)} tip="在选中的组里新建一个子组" onClick={() => newGroup(true)}>新建子组</Button>
            <span className="pif-gap" />
            <Button size="sm" tone="ghost" disabled={!canUp} tip="在同一层里上移" onClick={() => shift(-1)}>上移</Button>
            <Button size="sm" tone="ghost" disabled={!canDown} tip="在同一层里下移" onClick={() => shift(1)}>下移</Button>
            <Button size="sm" tone="ghost" disabled={!sel || sel.length < 2} tip="移出所在的组，放到这个组的后面" onClick={outdent}>移出组</Button>
            <Button size="sm" tone="ghost" danger disabled={!chosen || !isGroup(chosen)} tip="删除这个组：组里的参数和子组留下，放回组原来的位置" onClick={dissolve}>删除组</Button>
            </Writes>
          </div>
          <div className="pif-tree" role="tree" aria-label="参数界面">
            {rows.length === 0 && <div className="pif-empty">还没有公开参数：关掉这个窗口，点参数旁的图钉公开，再回来摆放</div>}
            {rows.map((r, i) => {
              const key = r.path.join("/");
              const wrong = (problems.get(key) ?? []).length > 0;
              return (
                <div key={key} role="treeitem" aria-selected={samePath(sel, r.path)}
                  className={`pif-row${isGroup(r.entry) ? " group" : ""}${samePath(sel, r.path) ? " on" : ""}${wrong ? " wrong" : ""}${drag.over === i ? " drag-over" : ""}`}
                  style={{ paddingLeft: 8 + r.depth * 16 }} onClick={() => setSel(r.path)} {...drag.row(i)}>
                  <span className="pif-grip" {...drag.grip(i)} data-tip="拖动：放到参数上 = 放到它前面；放到组上 = 放进这个组">⋮⋮</span>
                  {isGroup(r.entry) ? (
                    <>
                      <span className="pif-kind">组</span>
                      <span className="pif-name" data-user-data>{r.entry.label || "（没有名字）"}</span>
                    </>
                  ) : (
                    <>
                      <span className="pif-name" data-user-data>{r.entry.label}</span>
                      <span className="pif-detail" data-user-data>{r.entry.name}</span>
                      {r.entry.widget && <span className="pif-kind">{r.entry.widget === "menu" ? "下拉" : "复选框"}</span>}
                      {targetOf(r.entry)?.p.widget === "button" && <span className="pif-kind">按钮</span>}
                      {r.entry.hide_when && <span className="pif-kind">Hide</span>}
                      {r.entry.disable_when && <span className="pif-kind">Disable</span>}
                      {r.entry.show_on_change && <span className="pif-kind">显示</span>}
                    </>
                  )}
                </div>
              );
            })}
            {rows.length > 0 && (
              <div className={`pif-row pif-end${drag.over === rows.length ? " drag-over" : ""}`} {...drag.row(rows.length)}>
                拖到这里：放到最外层末尾
              </div>
            )}
          </div>
        </div>
        <div className="pif-right">
          {!chosen && <div className="pif-empty">在左边选一项，这里改它的显示名、显示方式和条件</div>}
          <Writes>
            {chosen && isGroup(chosen) && <GroupProps g={chosen} onChange={edit} problems={problems.get(sel!.join("/")) ?? []} />}
            {chosen && !isGroup(chosen) && <ParamProps key={sel!.join("/")} x={chosen} onChange={edit} all={all} problems={problems.get(sel!.join("/")) ?? []} />}
          </Writes>
        </div>
      </div>
      <SheetFoot ok={() => set(tree)} cancel={cancel} okTip="按这个样子摆参数面板（写进节点图）">
        {bad > 0 && <span className="pif-bad">{bad} 项有问题（标红）：存成模板时服务器会拒绝</span>}
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
  return (
    <div className="pif-props">
      <label className="pif-field">
        <span>组名</span>
        <input className="field" value={g.label} spellCheck={false} onChange={(e) => onChange({ ...g, label: e.target.value })} />
      </label>
      <div className="pif-field">
        <span>默认折叠</span>
        <Switch on={!!g.collapsed} onChange={(on) => onChange({ ...g, collapsed: on })} label="默认折叠" />
      </div>
      <Problems list={problems} />
    </div>
  );
}

type WidgetChoice = "default" | "menu" | "checkbox";

function ParamProps({ x, onChange, all, problems }: { x: ExposedParam; onChange: (x: ExposedParam) => void; all: ExposedParam[]; problems: string[] }) {
  const names = all.map((q) => q.name);
  const t = useMemo(() => targetOf(x), [x.target]); // eslint-disable-line react-hooks/exhaustive-deps
  const widget: WidgetChoice = x.widget ?? "default";
  const setWidget = (w: WidgetChoice) => {
    const { widget: _w, options: _o, ...rest } = x;
    void _w;
    void _o;
    if (w === "default") return onChange(rest);
    // 复选框接整数参数时，选项就是 0 和 1 两项；下拉先按参数自己的可选值（或 0 / 1）填好，作者再改显示名
    const seed = w === "checkbox"
      ? t?.p.type === "integer" ? [{ value: 0, label: "关" }, { value: 1, label: "开" }] : undefined
      : x.options?.length ? x.options : t?.p.options?.map((o) => ({ value: o as unknown, label: optionName(t.p, o) }))
        ?? (t?.p.type === "boolean" ? [{ value: true, label: "是" }, { value: false, label: "否" }] : [{ value: t ? sample(t.p) : "", label: "" }]);
    onChange({ ...rest, widget: w, ...(seed ? { options: seed } : {}) });
  };
  const options = x.options ?? [];
  const setOptions = (o: ExposedOption[]) => onChange({ ...x, options: o });
  const button = t?.p.widget === "button";
  return (
    <div className="pif-props">
      <label className="pif-field">
        <span>显示名</span>
        <input className="field" value={x.label} spellCheck={false} onChange={(e) => onChange({ ...x, label: e.target.value })} />
      </label>
      <div className="pif-field">
        <span>对外名字</span>
        <span className="pif-note" data-user-data>{x.name}<small>命令行和插件按这个名字传值（lab2shot cook --set {x.name}=…），这里不改</small></span>
      </div>
      <div className="pif-field">
        <span>对应参数</span>
        <span className="pif-note" data-user-data>{t ? `「${t.node}」的「${t.p.label}」` : x.target}</span>
      </div>
      {!button && <div className="pif-field">
        <span>显示成</span>
        <Segmented label="显示成" value={widget} onChange={setWidget}
          options={[
            { value: "default" as WidgetChoice, label: "参数自己的" },
            { value: "menu" as WidgetChoice, label: "下拉", disabled: t && widgetRefused("menu", t.p, [{ value: 0 }]) },
            { value: "checkbox" as WidgetChoice, label: "复选框", disabled: t && t.p.type !== "boolean" && t.p.type !== "integer" ? "目标参数要是布尔或整数" : false },
          ]} />
      </div>}
      {x.widget && (x.widget === "menu" || t?.p.type === "integer") && (
        <div className="pif-field top">
          <span>{x.widget === "menu" ? "选项" : "两个值"}</span>
          <div className="pif-options">
            {options.map((o, i) => (
              <div className="pif-option" key={i}>
                <OptionValue t={t} value={o.value} onChange={(v) => setOptions(options.map((q, j) => (j === i ? { ...q, value: v } : q)))} />
                <input className="field" placeholder="显示名" value={o.label} spellCheck={false}
                  onChange={(e) => setOptions(options.map((q, j) => (j === i ? { ...q, label: e.target.value } : q)))} />
                {x.widget === "menu" && (
                  <Button size="sm" tone="ghost" tip="去掉这一项" onClick={() => setOptions(options.filter((_, j) => j !== i))}>去掉</Button>
                )}
                {x.widget === "menu" && (
                  <OptionHide value={o.hide_when} all={all}
                    onChange={(v) => setOptions(options.map((q, j) => (j === i ? withOptionHide(q, v) : q)))} />
                )}
              </div>
            ))}
            {x.widget === "menu" && (
              <Button size="sm" tone="ghost" tip="加一项：值和显示名都自己填" onClick={() => setOptions([...options, { value: t ? sample(t.p) : "", label: "" }])}>添加</Button>
            )}
          </div>
        </div>
      )}
      {!button && (
        <div className="pif-field">
          <span>显示节点</span>
          <label className="pif-check">
            <Switch on={!!x.show_on_change} label="修改后在视图里显示这个节点"
              onChange={(on) => onChange(on ? { ...x, show_on_change: true } : (({ show_on_change: _s, ...rest }) => (void _s, rest))(x))} />
            <span>修改后在视图里显示这个节点（点选类：碰到这一行就能在视图里点）</span>
          </label>
        </div>
      )}
      <Expression label="Hide When" value={x.hide_when} all={all} placeholder="空着 = 总显示；例如 cam_src != 1"
        onChange={(v) => onChange(withExpr(x, "hide_when", v))} />
      <Expression label="Disable When" value={x.disable_when} all={all} placeholder="空着 = 总能改；例如 not use_cam"
        onChange={(v) => onChange(withExpr(x, "disable_when", v))} />
      <div className="pif-hint">
        和 Houdini 一样：Hide When 为真时这一项不显示（组里全不显示时组也不显示），Disable When 为真时置灰不能改。写对外名字和值：
        <code>==</code> <code>!=</code> <code>in [1, 2]</code> <code>not in</code>{" "}
        <code>&lt;</code> <code>&lt;=</code> <code>&gt;</code> <code>&gt;=</code> <code>and</code> <code>or</code> <code>not</code> 括号；
        文字加引号，布尔写 true / false。可用的名字：{names.filter((n) => n !== x.name).join("、") || "（没有别的公开参数）"}
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
  return (
    <>
      <label className="pif-field">
        <span>{label}</span>
        <input className={`field mono${wrong ? " wrong" : ""}`} value={value ?? ""} spellCheck={false} placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)} />
      </label>
      {wrong && <div className="pif-problems">{wrong}</div>}
    </>
  );
}

/** 下拉一项自己的 Hide When（成立时下拉里不列这一项，例如「G1 机器人时只留原骨架」）：一行表达式框，即时标错（exprProblem）。 */
function OptionHide({ value, all, onChange }: { value: string | undefined; all: ExposedParam[]; onChange: (v: string) => void }) {
  const wrong = exprProblem(value, all);
  return (
    <>
      <input className={`field mono pif-option-hide${wrong ? " wrong" : ""}`} value={value ?? ""} spellCheck={false}
        placeholder="这一项的 Hide When：空着 = 总列出；例如 robot == 2" onChange={(e) => onChange(e.target.value)} />
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
function OptionValue({ t, value, onChange }: { t: Target | null; value: unknown; onChange: (v: unknown) => void }) {
  const p = t?.p ?? null;
  const answer = useResults((s) => (t ? s.results[t.id]?.applies : undefined));
  if (t && p && (p.options?.length || p.type === "boolean")) {
    const all: unknown[] = p.options?.length ? p.options : [true, false];
    const i = all.findIndex((o) => condEqual(o, value));
    const licensed = licensedValues(t.def, p.name);
    const rows = all.map((o, j) => {
      const named = optionName(p, o);
      // 作者挑的是以后用的值：现在不能选的也能挑，原因写在悬停里
      const view = optionView(p, o, answer, licensed, named === String(o) ? named : `${String(o)} · ${named}`);
      return { value: String(j), label: view.label, tip: view.tip };
    });
    return (
      <Select label="值" value={i < 0 ? "own" : String(i)} onPick={(v) => v !== "own" && onChange(all[Number(v)])}
        options={i < 0 ? [{ value: "own", label: `${JSON.stringify(value)} · 不合法` }, ...rows] : rows} />
    );
  }
  // 数字参数：页面唯一的数字框（ui/controls.tsx Num），按目标参数的声明规范化
  if (p && (p.type === "integer" || p.type === "number"))
    return <Num value={typeof value === "number" ? value : null} placeholder="值" onChange={onChange} {...paramNum(p)} />;
  const text = typeof value === "string" ? value : JSON.stringify(value) ?? "";
  return (
    <input className={`field mono${p && valueRefused(p, value) ? " wrong" : ""}`} placeholder="值" value={text} spellCheck={false}
      onChange={(e) => onChange(e.target.value)} />
  );
}

/** 弹窗框架里的这一种编辑器（ParamPanel.tsx 用 ParamSheet.tsx SheetWindow 打开）。 */
export const ParamInterface: SheetEditor = {
  editor: InterfaceEditor,
  title: () => "编辑参数界面",
  width: 920,
};
