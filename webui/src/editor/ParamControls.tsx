/** A parameter's control by its widget: numbers, sliders, options, colour spaces and views (OCIO), choices from a file or
 * a wire, class lists, tables. */

import { useEffect, useState } from "react";
import { api, emptyChoiceLabel, type OcioInfo, type ParamDef } from "../api";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { noncommercialValues } from "../graph/rules";
import { IconClose } from "../ui/icons";
import { DeliverParam, InputFileParam } from "./FileParam";
import { NumberField, optionOff, optionText, TextField, vecLabels } from "../ui/controls";
import { useChoices } from "../ui/choices";
import { HierarchyParam } from "./HierarchyPicker";
import { TableParam } from "./ParamTable";
import { Button, Chip, IconButton, Switch } from "../ui/Button";
import { useStagePicture, useViewer } from "../state/viewer";
import { addFigure, figureFrames } from "../view/figure2d";
import { Select, type SelectOption } from "../ui/Select";
import { useResults } from "../state/results";

let ocioCache: Promise<OcioInfo> | null = null;

function useOcio() {
  const [v, set] = useState<OcioInfo | null>(null);
  useEffect(() => {
    (ocioCache ??= api.ocio()).then(set).catch(() => {});
  }, []);
  return v;
}

/** 判断参数是否绘制为滑块：规则集中于此，无须每个参数单独声明 widget="slider"（有范围的数字不应提供可填入任意大值的输入框）。
 * 取值个数过多的参数除外：种子 0–21 亿、渲染宽高 16–16384 难以拖动定位，保留为输入框，范围写入提示。
 * 滑块右侧附带数字框，可输入精确值。 */
function slidable(p: ParamDef): p is ParamDef & { minimum: number; maximum: number } {
  if (p.minimum == null || p.maximum == null) return false;
  if (p.widget === "slider") return true;
  if (p.widget) return false; // 其他控件已明确声明类型，不覆盖
  if (p.type === "integer") return p.maximum - p.minimum <= 200;
  return p.type === "number";
}

export function Control({ nodeId, p, value, set, choice }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void; choice?: CellChoice }) {
  const ocio = useOcio();
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const nc = noncommercialValues(getNodeDefs()[typeId], p.name);
  // 选项是否可选与参数本身是否可用采用同一机制：服务器按声明计算（nodes/applies.py option_conditions），
  // 此处只按 id「<参数>=<选项>」查询。页面自身不对数据做任何判断
  const answer = useResults((s) => s.results[nodeId]?.applies);

  if (p.widget === "file" || p.widget === "sequence") return <InputFileParam nodeId={nodeId} p={p} value={(value as string) ?? ""} />;
  if (p.widget === "table") return <TableParam nodeId={nodeId} p={p} value={value} set={set} />;
  if (p.widget === "deliver") return <DeliverParam nodeId={nodeId} p={p} value={(value as string) ?? ""} set={set} />;

  if (p.type === "boolean") {
    return <Switch on={!!value} onChange={set} label={p.label} />;
  }
  if (p.options) {
    // a choice the node declares non-commercial says so the same way on every node
    const text = (o: string) => optionText(p, o, nc);
    // 多档选择一律使用下拉（不绘制分段控件），节点上与参数面板中外观一致；浏览器不支持的选项置灰并说明原因（option_needs）。
    // 空值一档必须有名称且可重新选中：值为空时触发器上不得将 null 显示为字面量 "null"，
    // 列表中也必须包含空值项，否则选中数字后将无法回到空值。参数声明了 placeholder
    // （P(placeholder=…) / measured_param(auto=…)）时以其作为该项名称并列于最前；未声明时不提供该项
    // （此类参数不应为空）。
    const blank = p.placeholder || "";
    const rows = p.options.map((o) => {
      const why = optionOff(p, answer, o); // 浏览器不支持，或接入的数据不符合该档要求（集中计算）
      // 不可用的选项仍然列出，但不可选择：置灰，原因写在其悬停提示中
      return { value: String(o), label: text(String(o)), tip: why ? `${text(String(o))}：${why}` : text(String(o)), off: !!why };
    });
    const empty = value === null || value === undefined;
    return (
      <Select value={empty ? "" : String(value)} label={p.label} tip={empty ? blank || p.label : text(String(value))}
        options={blank ? [{ value: "", label: blank, tip: blank }, ...rows] : rows}
        onPick={(v) => set(v === "" ? null : p.options!.find((o) => String(o) === v) ?? v)} />
    );
  }
  if (p.widget === "vec3") {
    const v = (value as number[]) ?? [0, 0, 0];
    return (
      <div className="vec3">
        {vecLabels(p).map(([axis, color], i) => (
          <label key={axis}>
            <span style={{ color }}>{axis}</span>
            <NumberField value={v[i]} onChange={(n) => set(v.map((x, j) => (j === i ? n ?? 0 : x)))} />
          </label>
        ))}
      </div>
    );
  }
  // 有上下界的数值参数使用滑块（`slidable`），取值过多的保留为输入框，范围写在提示中
  if (slidable(p)) {
    const v = Number(value ?? p.minimum);
    const pct = ((v - p.minimum) / (p.maximum - p.minimum)) * 100;
    return (
      <div className="slider">
        <input
          type="range"
          data-tip={`${p.label}：拖动，或在右边输入`}
          min={p.minimum}
          max={p.maximum}
          step={p.type === "integer" ? 1 : (p.maximum - p.minimum) / 100}
          value={v}
          style={{ ["--pct" as string]: `${pct}%` }}
          onChange={(e) => set(p.type === "integer" ? Math.round(Number(e.target.value)) : Number(Number(e.target.value).toFixed(3)))}
        />
        <NumberField value={v} onChange={(n) => set(n ?? p.minimum)} />
      </div>
    );
  }
  if (p.widget === "colorspace" && ocio) {
    return <ColorspaceParam nodeId={nodeId} p={p} ocio={ocio} value={value as string | null} set={set} />;
  }
  if (p.widget === "classes") return <ClassesParam nodeId={nodeId} p={p} value={(value as string) ?? ""} set={set} />;
  if (p.widget === "hierarchy") return <HierarchyParam nodeId={nodeId} p={p} value={value as string | string[]} set={set} />;
  if (p.widget === "choice") {
    return choice ? <ChoiceSelect p={p} choice={choice} value={value as string | null} set={set} /> : <ChoiceParam nodeId={nodeId} p={p} value={value as string | null} set={set} />;
  }
  if (p.widget === "picks" || p.widget === "canvas") return <DrawnList p={p} value={value} set={set} />;
  if (p.widget === "figure") return <FigureFrames p={p} value={value} set={set} />;
  if (p.type === "number" || p.type === "integer") {
    // 有上下界但难以拖动定位的参数（种子 0–21 亿、渲染宽高 16–16384）保留为输入框：为空时将范围写在占位提示中，
    // 超出范围的值由 NumberField 自动夹回范围内
    const range = p.minimum != null && p.maximum != null ? `${p.minimum}–${p.maximum}` : undefined;
    return <NumberField value={value as number | null} onChange={set} placeholder={p.placeholder || range} integer={p.type === "integer"} p={p} />;
  }
  // 自由文本。单行或多行由参数自身声明（`P(lines=…)`）：与节点上使用同一控件、同一份声明
  return <TextField lines={p.lines} value={(value as string) ?? ""} onChange={(v) => set(v === "" && p.nullable ? null : v)} placeholder={p.placeholder} />;
}

/** One control's options: the list and their names, what 自动 comes to here, the name of "" when it means "none",
 * what the empty value shows when there is no 自动, and the value to write in while it is empty. */
export interface CellChoice {
  options: string[];
  labels?: Record<string, string>;
  auto?: string;
  none?: string;
  empty?: string;
  default?: string;
}

/** 「色彩空间」没有「自动」档。值始终是实际的色彩空间名：节点的 choices 按格式给出 `default`
 * （读取器：文件本身的色彩空间，EXR 为 ACEScg，PNG / JPG 为 sRGB；输出：写出的色彩空间），使用者看到的即为该值，修改即生效。
 * 填写操作不在此处进行，而是在节点图根部进行，与节点是否选中无关（editor/ColorspaceFill.tsx）；若仅在面板打开时填写，
 * 先提交再打开节点会导致节点图变化、上游全部重算。
 * 值为空时（尚未选择文件）显示 choices 的 `empty` 或参数自身的 placeholder。 */
function ColorspaceParam({ nodeId, p, ocio, value, set }: { nodeId: string; p: ParamDef; ocio: OcioInfo; value: string | null; set: (v: unknown) => void }) {
  const choice = useChoices(nodeId, p);
  const empty = choice?.empty || p.placeholder || "按格式";
  return (
    <Select className="names" data-user-data label={p.label} value={value ?? ""} tip={value || empty}
      options={[...(value ? [] : [{ value: "", label: empty, tip: empty }]),
                ...ocio.colorspaces.map((c) => ({ value: c, label: c, tip: c === ocio.working ? `${c}（工作空间）` : c }))]}
      onPick={(v) => set(v || null)} />
  );
}

const EMPTY = "\u0001empty";
const NONE = "\u0001none";

/** A dropdown of options from a file or from what is wired in. The empty value first: 自动 showing what it resolves to
 * (the file's only camera, the guessed joint), or what it lacks (先选择相机文件, 选一台, the parameter's own placeholder
 * while the options are not known yet, see api/status.ts emptyChoiceLabel; once something is chosen that
 * entry is left out, as it would only undo the choice); "none" when allowed (不映射); then the options. A value no
 * longer among them stays listed, marked; before the options are known (the input not cooked yet) the value shows as
 * it is. The empty value is the parameter's own: null when it can be, else "". */
function ChoiceSelect({ p, choice, value, set }: { p: ParamDef; choice: CellChoice | null; value: string | null; set: (v: unknown) => void }) {
  const empty = p.nullable ? null : "";
  // "" is "none" where the parameter can be null (then null is 自动) or where the options say so (an import node's 相机)
  const current = value == null || value === "" ? (value === "" && (p.nullable || choice?.none) ? NONE : EMPTY) : value;
  const options = choice?.options ?? [];
  const name = (o: string) => choice?.labels?.[o] ?? o;
  const auto = emptyChoiceLabel(choice, p.placeholder, name); // api/status.ts: the single definition of how 「还没选」 is displayed
  const own = current !== EMPTY && current !== NONE && !options.includes(current); // not among the options (yet)
  const withEmpty = (current === EMPTY || choice === null || choice.auto !== undefined) && !(choice?.none && !p.nullable);
  const rows: SelectOption[] = [];
  if (withEmpty) rows.push({ value: EMPTY, label: auto, tip: auto });
  if (choice?.none || current === NONE) rows.push({ value: NONE, label: choice?.none ?? "无", tip: choice?.none ?? "不映射" });
  if (own) rows.push({ value: current, label: choice ? `${current} · 找不到了` : current, tip: choice ? `${current}：上游已经没有这一项了` : current });
  for (const o of options) rows.push({ value: o, label: name(o), tip: name(o) });
  return (
    <Select
      className="choice names"
      data-user-data
      label={p.label}
      value={current}
      disabled={!options.length && !choice?.none && current === EMPTY}
      tip={current === EMPTY ? auto : current === NONE ? choice?.none : name(current)}
      options={rows}
      onPick={(v) => set(v === EMPTY ? empty : v === NONE ? "" : v)}
    />
  );
}

function ChoiceParam({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: string | null; set: (v: unknown) => void }) {
  const choice = useChoices(nodeId, p);
  return <ChoiceSelect p={p} choice={choice && { ...choice, auto: choice.auto as string | undefined }} value={value} set={set} />;
}

/** What the user drew in the 2D view, kept as this node's parameter: widget "picks" (one click each) and widget
 * "canvas" (one hand-drawn outline each, nodes/handles.py). One row of chips for both: the drawing happens in the
 * view, and the panel only lists what is there and lets a wrong one be removed. An entry is "frame:x,y…" (nodes/base.py):
 * a click is short enough to show as it is; an outline is dozens of numbers, so its chip is the parameter's own name and
 * its number (「形状 1」, no second wording), with the frame and the point count on hover. Panel text never wraps, and is
 * never cut off. */
function DrawnList({ p, value, set }: { p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const list = (value as string[]) ?? [];
  const shapes = p.widget === "canvas";
  const points = (entry: string) => Math.floor((entry.split(":")[1] ?? "").split(",").length / 2);
  return (
    <div className="picks">
      {list.length === 0 && <span style={{ color: "var(--text-3)", fontSize: 12 }}>{p.placeholder}</span>}
      {list.map((k, i) => (
        <span className="chip" key={`${i}:${k}`} data-tip={shapes ? `${k.split(":")[0]} · ${points(k)} 个点` : k}>
          {shapes ? `${p.label} ${i + 1}` : k}
          <button onClick={() => set(list.filter((_, j) => j !== i))} aria-label="移除" data-tip="从列表里移除">
            <IconClose size={9} />
          </button>
        </span>
      ))}
    </div>
  );
}

/** 火柴人的关键姿势（widget「figure」，`lab2shot/nodes/core/sketch.py` 的 `poses`）：此处添加的是帧，
 * 视图中只负责拖动关节。不通过拖动创建（容易产生不合理的人体比例），而是按帧添加：
 * 1. 「添加帧」：在当前帧放置默认 T-pose（比例固定，`view/figure2d.ts FIGURE_TPOSE`）；
 * 2. 「基于前一帧」：原样复制前一帧的姿势；
 * 3. 已绘制的帧：每帧一个 chip，点击后时间线跳到该帧（便于切换检查连续性，
 *    比在时间线上定位更快），✕ 删除该帧的姿势。
 *
 * 不可用的按钮置灰并说明原因，位置不变，不隐藏。 */
function FigureFrames({ p, value, set }: { p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const list = (value as string[]) ?? [];
  // 画面尺寸（图像像素）由二维舞台提供，此处不另行计算（state/viewTools.ts useStagePicture）
  const size = useStagePicture((s) => s.size);
  const frames = figureFrames(list);
  // 该区域不得随当前帧重绘（webui/tests/playbackFrameTime.test.ts：参数面板不随时钟更新，
  // 否则播放时每秒重绘 24 次）。因此两个按钮是否可用只取决于已绘制的帧，与当前所在帧无关；
  // 当前帧在按下按钮时才读取（`useViewer.getState()`），随后跳到目标帧，使用者可直接看到添加的位置
  const why = !size ? "还不知道画面多大：接一段序列进「图像」口，或者先点「计算」，2D 视图里有画面了再添加" : "";
  const add = (basedOnPrevious: boolean) => {
    if (!size) return;
    const got = addFigure(list, useViewer.getState().frame, size, basedOnPrevious);
    if (!got) return;
    set(got.values);
    useViewer.getState().setFrame(got.frame);
  };
  return (
    <div className="fig-frames">
      <div className="picks">
        <Button size="sm" disabled={!!why} onClick={() => add(false)}
                tip={why || "在当前帧放一个站好的火柴人（T-pose，真人比例），再在 2D 视图里拖关节摆姿势。这一帧已经画过了就放到后面第一个没画的帧上"}>
          添加帧
        </Button>
        <Button size="sm" disabled={!!why || frames.length === 0} onClick={() => add(true)}
                tip={why || (frames.length === 0 ? "前面还没有画过的帧：第一帧只能是默认 T-pose"
                                                 : "把前一帧那个姿势原样复制到当前帧，再在它基础上改")}>
          基于前一帧
        </Button>
      </div>
      <div className="picks">
        {frames.length === 0 && <span className="fig-none">{p.placeholder}</span>}
        {/* 每帧一个 chip：点击跳到该帧，旁边的 ✕ 删除该帧的姿势。
            两者均由通用组件绘制（ui/Button.tsx 的 Chip 与 IconButton），此处只负责成对排列 */}
        {frames.map((f) => (
          <span className="fig-frame" key={f}>
            <Chip tip={`跳到第 ${f} 帧看这个姿势`} onClick={() => useViewer.getState().setFrame(f)}>第 {f} 帧</Chip>
            <IconButton size="xxs" tone="ghost" aria-label="移除" tip={`删掉第 ${f} 帧的姿势`}
                        onClick={() => set(list.filter((s) => Number(s.split(":")[0]) !== f))}>
              <IconClose size={9} />
            </IconButton>
          </span>
        ))}
      </div>
    </div>
  );
}

const splitList = (text: string) => text.split(/[,，、]/).map((t) => t.trim()).filter(Boolean);

/** A list of class names typed into a text field, plus the classes the wired segmentation carries (once it has been
 * cooked) as chips that add or remove themselves. Typing works before that: numbers and names are both accepted. */
function ClassesParam({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: string; set: (v: unknown) => void }) {
  const choice = useChoices(nodeId, p);
  const tokens = splitList(value);
  const names = (o: string) => [o, ...(choice?.aliases?.[o] ?? [])].map((k) => k.toLowerCase());
  const has = (o: string) => tokens.some((t) => names(o).includes(t.toLowerCase()));
  const toggle = (o: string) => set((has(o) ? tokens.filter((t) => !names(o).includes(t.toLowerCase())) : [...tokens, o]).join(", "));
  return (
    <div className="class-pick">
      <TextField value={value} onChange={set} placeholder={p.placeholder} />
      {choice && choice.options.length > 0 && (
        <div className="picks">
          {choice.options.map((o) => (
            <Chip key={o} tip={[...(choice.aliases?.[o] ?? []).slice(0, 1), o].join(" · ")} on={has(o)} onClick={() => toggle(o)}>
              {choice.labels?.[o] ?? o}
            </Chip>
          ))}
        </div>
      )}
      {!choice && <span className="class-hint">上游算过以后，这里列出能选的类别</span>}
    </div>
  );
}

let canvas: CanvasRenderingContext2D | null = null;
function textWidth(text: string, font: string): number {
  canvas ??= document.createElement("canvas").getContext("2d");
  if (!canvas) return text.length * 12;
  canvas.font = font;
  return canvas.measureText(text).width;
}

/** The width the panel needs for every label and control to show in full (the label column is sized here too):
 * a pull-down at its longest option, lists at their longest entry, other fields at a comfortable width. */
export function fitWidth(body: HTMLElement): number {
  const labels = [...body.querySelectorAll<HTMLElement>(".plabel-text")];
  // Room for the label and, left of it, the three marks: 对外参数 (.expose-pin, left -2), 提升到节点 (.socket-pin,
  // left 15) and 在节点上显示 (.onnode-pin, left 32), 16px each, ending at 48.
  //
  // 上限与下限都必须计入标记区域：206 / 112 指文字可用宽度，标记区域另行增加。若上限写为整列宽度 206，
  // 当其他因素使该列达到上限时，文字会被压缩并与标记重叠。界面文字不得换行或截断，因此必须预留足够宽度。
  const MARKS = 57; // 三个标记占到 48，另留 9 的间距
  const label = Math.min(206 + MARKS, Math.max(112 + MARKS, ...labels.map((l) => textWidth(l.textContent ?? "", getComputedStyle(l).font) + MARKS)));
  body.style.setProperty("--plabel-w", `${Math.ceil(label)}px`);
  let control = 170;
  for (const ctl of body.querySelectorAll<HTMLElement>(".prow .ctl")) {
    const select = ctl.querySelector<HTMLSelectElement>("select");
    const file = ctl.querySelector<HTMLElement>(".fp-line, .fp-where");
    if (select) {
      const font = getComputedStyle(select).font;
      control = Math.max(control, ...[...select.options].map((o) => textWidth(o.text, font) + 34));
    } else if (file) {
      // a file parameter: its buttons and the whole line saying what was chosen (an output: its folder and a name)
      const parts = file.classList.contains("fp-line") ? ([...file.children] as HTMLElement[]) : [file];
      const text = parts.reduce((w, el) => w + textWidth(el.textContent ?? "", getComputedStyle(el).font) + 12, 0);
      const buttons = [...ctl.querySelectorAll<HTMLElement>("button:not(.fp-where)")].reduce((w, b) => w + b.offsetWidth + 6, 0);
      control = Math.max(control, Math.min(480, text + buttons + (file.classList.contains("fp-where") ? 190 : 0)));
    }
  }
  return Math.ceil(label + 10 + control + 10 + 32 + 10); // row gaps, group padding, scrollbar
}
