/** 参数控件的分发归本模块：按参数的 widget 选出面板中画的控件——数字、滑块、选项、色彩空间与视图（OCIO）、
 * 来自文件或连线的选项、类别列表、表格，打开弹窗的编辑器（SHEET_EDITORS，editor/ParamSheet.tsx），以及在视图里编辑的
 * 手柄参数的摘要行（骨架姿势、双骨架编辑 rig_pair）。 */

import { useEffect, useRef, useState } from "react";
import { api, emptyChoiceLabel, type Choice, type OcioInfo, type ParamDef } from "../api";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { licensedValues, registeredValues } from "../graph/rules";
import { IconClose } from "../ui/icons";
import { InputFileParam } from "./FileParam";
import { coerce } from "../model/numbers";
import { Num, optionView, paramNum, TextField, VecField } from "../ui/controls";
import { useChoices, useChoiceSet } from "../ui/choices";
import { Hierarchy } from "./HierarchyPicker";
import { ExpressionMap } from "./ExpressionMap";
import { SheetParam, SheetWindow, TextSheet, useOpenSheet, type SheetEditor } from "./ParamSheet";
import { Sheet } from "../ui/Sheet";
import { setParams } from "../graph/actions";
import { indexAt, paramAt, placeAt, stableAt } from "./paramPath";
import { why as whyOff } from "../api/applies";
import { usePickBoxes } from "./pickedPeople";
import { pickChips } from "../model/pickChips";
import { ButtonParam } from "./buttonActions";
import { TableParam } from "./ParamTable";
import { Button, Chip, IconButton, Switch } from "../ui/Button";
import { useStagePicture, useViewer } from "../state/viewer";
import { addFigure, figureFrames } from "../view/figure2d";
import { Select, type SelectOption } from "../ui/Select";
import { useResults } from "../state/results";
import { useHandleView } from "../state/handleView";
import { useLook } from "../state/look";
import { useRigPairView, type RigMode } from "../state/rigPairView";
import { mappingSummary, namesOf, recordOf, withTouched, type RigRoles } from "../model/rigPair";
import type { RigPairData } from "../api";
import { peopleHandle } from "../view/handles2d";
import { role3d } from "../view/handleEditing";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** 弹窗编辑参数的注册表（editor/ParamSheet.tsx）：widget 名 → 编辑器。参数声明了这里的某个 widget，面板上就是
 * 「摘要 + 编辑…」，点开一个窗，确定写回、取消不写回。新增一种编辑器在这里加一行（步骤见 ParamSheet.tsx 顶部）。 */
const SHEET_EDITORS: Record<string, SheetEditor> = {
  hierarchy: Hierarchy, // 导入节点的 USD / FBX 层级选择（HierarchyPicker.tsx）
  // 框架的最小示例：没有节点用它，ParamSheet.tsx 顶部「新增一种弹窗编辑器」的说明以它为范例
  // （实际的使用者是上面的 hierarchy）
  sheet_text: TextSheet,
};

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
  // 可留空的数（空 = 自动 / 按默认，如「动作重定向」的「修正系数」留空当 1）：滑块表达不了「空」，会把空画成最小值
  // （0.01）误导人；用带 placeholder 的输入框
  if (p.nullable) return false;
  if (p.widget === "slider") return true;
  if (p.widget) return false; // 其他控件已明确声明类型，不覆盖
  if (p.type === "integer") return p.maximum - p.minimum <= 200;
  return p.type === "number";
}

/** `lock`：改不了的原因（只读标签页、参数不适用），只给打开窗的参数用：它们不放进禁用的 fieldset（打开窗是看）。 */
export const opensSheet = (p: ParamDef): boolean => !!SHEET_EDITORS[p.widget ?? ""];

/** `at`：这个控件的值在节点里的位置（editor/paramPath.ts）：参数是它的名字（默认），表格格子是 `表[i].列`。许可、选项能不能选、
 * 编辑窗、半填的草稿都按它，不按列名（列名可能与某个参数同名）。 */
export function Control({ nodeId, p, value, set, choice, lock = "", at = p.name }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void; choice?: CellChoice; lock?: string; at?: string }) {
  const ocio = useOcio();
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const def = getNodeDefs()[typeId];
  const nc = licensedValues(def, at);
  const reg = registeredValues(def, at);
  // 选项是否可选与参数本身是否可用采用同一机制：服务器按声明计算（nodes/applies.py option_conditions），
  // 此处只按 id「<参数>=<选项>」查询。页面自身不对数据做任何判断
  const answer = useResults((s) => s.results[nodeId]?.applies);

  // 按钮参数（没有值，只有动作：editor/buttonActions.tsx），占满参数列
  if (p.widget === "button") return <ButtonParam nodeId={nodeId} p={p} />;
  if (p.widget === "file" || p.widget === "sequence") return <InputFileParam nodeId={nodeId} p={p} value={(value as string) ?? ""} />;
  if (p.widget === "table") return <TableParam nodeId={nodeId} p={p} value={value} set={set} Control={Control} />;
  if (p.widget === "expression_map") return <ExpressionMap nodeId={nodeId} p={p} value={value} set={set} />;
  // 节点的双骨架编辑手柄（rig_pair）改的参数：对应关系、两侧姿势、两侧忽略，面板上只是摘要行与「在视图里编辑」
  const rig = at === p.name ? rigRoleOf(def, p.name) : null;
  if (rig) return <RigPairParam nodeId={nodeId} value={value} {...rig} />;
  const sheet = SHEET_EDITORS[p.widget ?? ""];
  if (sheet) return <SheetParam nodeId={nodeId} at={at} p={p} value={value} set={set} kind={sheet} lock={lock} />;

  if (p.type === "boolean") {
    return <Switch on={!!value} onChange={set} label={p.label} />;
  }
  if (p.options) {
    // 许可受限的选项（非商用、仅限研究）与不能选的原因，在每处以同一方式标明（ui/controls.tsx optionView）
    // 多档选择一律使用下拉（不绘制分段控件），节点上与参数面板中外观一致；不可用的选项置灰并说明原因（option_applies）。
    // 空值一档必须有名称且可重新选中：值为空时触发器上不得将 null 显示为字面量 "null"，
    // 列表中也必须包含空值项，否则选中数字后将无法回到空值。参数声明了 placeholder
    // （P(placeholder=…) / measured_param(auto=…)）时以其作为该项名称并列于最前；未声明时不提供该项
    // （此类参数不应为空）。
    const blank = p.placeholder || "";
    const rows = p.options.map((o) => {
      // 不可用的选项仍然列出，但不可选择：置灰，原因写在其悬停提示中
      const view = optionView(p, o, answer, nc, undefined, at, reg);
      return { value: String(o), label: view.label, tip: view.tip, off: !!view.off };
    });
    const empty = value === null || value === undefined;
    return (
      <Select value={empty ? "" : String(value)} label={p.label}
        options={blank ? [{ value: "", label: blank }, ...rows] : rows}
        onPick={(v) => set(v === "" ? null : p.options!.find((o) => String(o) === v) ?? v)} />
    );
  }
  // 空着时填了一半的分量属于这个节点的这个位置：按它作 key，换了选中的节点（面板不换组件）就不带过去
  // （格子按行名：前面删了一行，草稿不挪到别的行上，editor/paramPath.ts stableAt）
  if (p.widget === "vec3") return <VecField key={`${nodeId}.${stableAt(useCookInputs.getState().nodes[nodeId]?.params ?? {}, at)}`} p={p} value={value} set={set} />;
  // 有上下界的数值参数使用滑块（`slidable`），取值过多的保留为输入框，范围写在提示中
  if (slidable(p)) {
    const v = Number(value ?? p.minimum);
    const pct = ((v - p.minimum) / (p.maximum - p.minimum)) * 100;
    return (
      <div className="slider">
        <input
          type="range"
          aria-label={p.label}
          min={p.minimum}
          max={p.maximum}
          step={p.type === "integer" ? 1 : (p.maximum - p.minimum) / 100}
          value={v}
          style={{ ["--pct" as string]: `${pct}%` }}
          onChange={(e) => { const n = coerce(Number(e.target.value), paramNum(p)); if (n !== null && n !== v) set(n); }}
        />
        <Num value={v} onChange={set} {...paramNum(p)} />
      </div>
    );
  }
  if (p.widget === "colorspace" && ocio) {
    return <ColorspaceParam nodeId={nodeId} p={p} ocio={ocio} value={value as string | null} set={set} />;
  }
  if (p.widget === "classes") return <ClassesParam nodeId={nodeId} p={p} value={(value as string) ?? ""} set={set} />;
  if (p.widget === "choice") {
    return choice ? <ChoiceSelect p={p} choice={choice} value={value as string | null} set={set} /> : <ChoiceParam nodeId={nodeId} p={p} value={value as string | null} set={set} />;
  }
  if (p.widget === "picks" || p.widget === "canvas") return <DrawnList nodeId={nodeId} p={p} value={value} set={set} />;
  if (p.widget === "figure") return <FigureFrames p={p} value={value} set={set} />;
  if (p.widget === "skeleton_pose") return <PoseSummary nodeId={nodeId} p={p} value={value} set={set} />;
  if (p.type === "number" || p.type === "integer") {
    // 有上下界但难以拖动定位的参数（种子 0–21 亿、渲染宽高 16–16384）保留为输入框：为空时将范围写在占位提示中，
    // 超出范围的值由 Num 夹回范围内；清空只在参数可为空时写 null，否则退回原值
    const range = p.minimum != null && p.maximum != null ? `${p.minimum}–${p.maximum}` : undefined;
    return p.nullable
      ? <Num nullable value={value as number | null} onChange={set} placeholder={p.placeholder || range} {...paramNum(p)} />
      : <Num value={value as number | null} onChange={set} placeholder={p.placeholder || range} {...paramNum(p)} />;
  }
  // 自由文本。单行或多行由参数自身声明（`P(lines=…)`）：与节点上使用同一控件、同一份声明
  return <TextField lines={p.lines} value={(value as string) ?? ""} onChange={(v) => set(v === "" && p.nullable ? null : v)} placeholder={p.placeholder} />;
}

/** 表格格子里画控件用的那个组件（就是上面的 Control），传给 ParamTable.tsx。 */
export type CellControl = typeof Control;

/** 一个控件的选项：选项列表及其名称、「自动」在此处的结果、"" 表示「无」时的名称、
 * 没有「自动」时空值显示的文字，以及为空时写入的值。 */
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
  const empty = choice?.empty || p.placeholder || t("ui.params.by_format");
  return (
    <Select className="names" data-user-data label={p.label} value={value ?? ""}
      options={[...(value ? [] : [{ value: "", label: empty }]),
                ...ocio.colorspaces.map((c) => ({ value: c, label: c, tip: c === ocio.working ? tipOf("value", t("ui.params.working_space")) : undefined }))]}
      onPick={(v) => set(v || null)} />
  );
}

const EMPTY = "\u0001empty";
const NONE = "\u0001none";

/** 选项来自文件或接入数据的下拉。空值一项在最前：「自动」显示它解析到的结果（文件中唯一的相机、猜到的关节），
 * 或缺少的东西（先选择相机文件、选一台；选项尚未知道时为参数自身的 placeholder，见 api/status.ts emptyChoiceLabel；
 * 已选定某项后不列出该项，因为它只会撤销选择）；允许时接着是「无」（不映射）；然后是各选项。
 * 不在选项中的值仍列出并加标记；选项尚未知道（输入还没算）时按原样显示该值。
 * 空值是参数自身的空值：可为 null 时为 null，否则为 ""。 */
function ChoiceSelect({ p, choice, value, set }: { p: ParamDef; choice: CellChoice | null; value: string | null; set: (v: unknown) => void }) {
  const empty = p.nullable ? null : "";
  // 参数可为 null（此时 null 为「自动」）或选项声明了「无」（导入节点的「相机」）时，"" 表示「无」
  const current = value == null || value === "" ? (value === "" && (p.nullable || choice?.none) ? NONE : EMPTY) : value;
  const options = choice?.options ?? [];
  const name = (o: string) => choice?.labels?.[o] ?? o;
  const auto = emptyChoiceLabel(choice, p.placeholder, name); // api/status.ts：「还没选」如何显示的唯一定义
  const own = current !== EMPTY && current !== NONE && !options.includes(current); // （暂时）不在选项中
  const withEmpty = (current === EMPTY || choice === null || choice.auto !== undefined) && !(choice?.none && !p.nullable);
  const rows: SelectOption[] = [];
  if (withEmpty) rows.push({ value: EMPTY, label: auto });
  if (choice?.none || current === NONE) rows.push({ value: NONE, label: choice?.none ?? t("ui.common.none") });
  if (own) rows.push({ value: current, label: choice ? t("ui.params.choice_gone", { value: current }) : current, tip: choice ? tipOf("error", t("ui.params.choice_gone_tip")) : undefined });
  for (const o of options) rows.push({ value: o, label: name(o) });
  return (
    <Select
      className="choice names"
      data-user-data
      label={p.label}
      value={current}
      disabled={!options.length && !choice?.none && current === EMPTY}
      options={rows}
      onPick={(v) => set(v === EMPTY ? empty : v === NONE ? "" : v)}
    />
  );
}

function ChoiceParam({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: string | null; set: (v: unknown) => void }) {
  const choice = useChoices(nodeId, p);
  return <ChoiceSelect p={p} choice={choice && { ...choice, auto: choice.auto as string | undefined }} value={value} set={set} />;
}

/** 使用者在 2D 视图中画的内容，作为该节点的参数保存：widget "picks"（每次点击一项）与 widget "canvas"
 * （每个手绘轮廓一项，nodes/handles.py）。两者都画成一行 chip：绘制在视图中进行，面板只列出已有内容并允许去掉画错的。
 * 每项为 "frame:x,y…"（nodes/handles.py parse_picks / parse_shapes）：一次点击足够短，按原样显示；轮廓有几十个数，
 * 因此其 chip 为参数自身的名称加序号（「形状 1」，不另起说法）。面板文字不换行，也不截断。 */
function DrawnList({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const list = (value as string[]) ?? [];
  const shapes = p.widget === "canvas";
  const person = !shapes && !!peopleHandle(getNodeDefs()[useCookInputs.getState().nodes[nodeId]?.typeId ?? ""]?.handles, p.name);
  if (person) return <PeoplePicks nodeId={nodeId} p={p} list={list} set={set} />;
  return (
    <div className="picks">
      {list.length === 0 && <span style={{ color: "var(--text-3)", fontSize: 12 }}>{p.placeholder}</span>}
      {list.map((k, i) => (
        <span className="chip" key={`${i}:${k}`}>
          {shapes ? `${p.label} ${i + 1}` : k}
          <button onClick={() => set(list.filter((_, j) => j !== i))} aria-label={t("ui.common.remove")}>
            <IconClose size={9} />
          </button>
        </span>
      ))}
    </div>
  );
}

/** 点人（「选人」的点选，手柄 kind "person"）：chip 说点的是几号（editor/pickedPeople.tsx，与服务端同一条命中规则），
 * 同一个人点几次只一个 chip、标次数，✕ 去掉这个人的全部点击；上游人物框还没算出时退回「第 F 帧 (x, y)」并说明。 */
function PeoplePicks({ nodeId, p, list, set }: { nodeId: string; p: ParamDef; list: string[]; set: (v: unknown) => void }) {
  const boxes = usePickBoxes(nodeId, p.name);
  const chips = pickChips(list, boxes);
  return (
    <div className="picks">
      {list.length === 0 && <span style={{ color: "var(--text-3)", fontSize: 12 }}>{p.placeholder}</span>}
      {chips.map((c) => (
        <span className={`chip${c.known ? "" : " unknown"}`} key={c.key} data-user-data>
          {c.text}
          <button onClick={() => set(list.filter((k) => !c.picks.includes(k)))} aria-label={t("ui.common.remove")}>
            <IconClose size={9} />
          </button>
        </span>
      ))}
      {chips.some((c) => !c.known) && (
        <span className="picks-why">{t(boxes ? "ui.params.picks_outside" : "ui.params.picks_need_boxes")}</span>
      )}
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
 * 不可用的按钮置灰，位置不变，不隐藏：还不知道画面尺寸时两个都不可用，还没有画过的帧时「基于前一帧」不可用。 */
function FigureFrames({ p, value, set }: { p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const list = (value as string[]) ?? [];
  // 画面尺寸（图像像素）由二维舞台提供，此处不另行计算（state/viewTools.ts useStagePicture）
  const size = useStagePicture((s) => s.size);
  const frames = figureFrames(list);
  // 该区域不得随当前帧重绘（参数面板不随时钟更新，
  // 否则播放时每秒重绘 24 次）。因此两个按钮是否可用只取决于已绘制的帧，与当前所在帧无关；
  // 当前帧在按下按钮时才读取（`useViewer.getState()`），随后跳到目标帧，使用者可直接看到添加的位置
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
        <Button size="sm" disabled={!size} onClick={() => add(false)}>
          {t("ui.params.frame_add")}
        </Button>
        <Button size="sm" disabled={!size || frames.length === 0} onClick={() => add(true)}>
          {t("ui.params.frame_from_previous")}
        </Button>
      </div>
      <div className="picks">
        {frames.length === 0 && <span className="fig-none">{p.placeholder}</span>}
        {/* 每帧一个 chip：点击跳到该帧，旁边的 ✕ 删除该帧的姿势。
            两者均由通用组件绘制（ui/Button.tsx 的 Chip 与 IconButton），此处只负责成对排列 */}
        {frames.map((f) => (
          <span className="fig-frame" key={f}>
            <Chip onClick={() => useViewer.getState().setFrame(f)}>{t("ui.params.frame_n", { frame: f })}</Chip>
            <IconButton size="xxs" tone="ghost" aria-label={t("ui.common.remove")}
                        onClick={() => set(list.filter((s) => Number(s.split(":")[0]) !== f))}>
              <IconClose size={9} />
            </IconButton>
          </span>
        ))}
      </div>
    </div>
  );
}

const splitList = (text: string) => text.split(/[,\uFF0C\u3001]/).map((t) => t.trim()).filter(Boolean);

/** 在文本框中填写的类别名列表，另加接入的分割结果所带的类别（算过之后），以 chip 形式点击即可加入或去掉。
 * 算过之前也可直接填写：数字与名称都接受。 */
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
            <Chip key={o} on={has(o)} onClick={() => toggle(o)}>
              {choice.labels?.[o] ?? o}
            </Chip>
          ))}
        </div>
      )}
      {!choice && <span className="class-hint">{t("ui.params.classes_after_cook")}</span>}
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

/** 面板完整显示所有标签与控件所需的宽度：下拉按最长的选项、列表按最长的条目，
 * 其他输入框取适中宽度。 */
export function fitWidth(body: HTMLElement): number {
  // 标签列整个面板一列（ui/LabelRow.tsx LabelGrid：按全部标签定宽，上限 --lrow-label-max），这里只算面板要多宽才放得下：
  // 最长的标签（不超过上限）加上标记格的宽度（量出来的，没有写死的像素）
  const labels = [...body.querySelectorAll<HTMLElement>(".lrow-label")];
  const marks = Math.max(0, ...[...body.querySelectorAll<HTMLElement>(".lrow-marks")].map((m) => m.getBoundingClientRect().width));
  const capOf = (l: HTMLElement) => parseFloat(getComputedStyle(l).fontSize) * 16; // --lrow-label-max: 16em
  // 面板告诉了全部标签（含现在藏起的，ui/LabelRow.tsx LabelGrid labels）时，标签列就是那么宽
  const fixed = Number(body.dataset.labelW) || 0;
  const label = marks + Math.max(0, ...labels.map((l) => Math.min(capOf(l), Math.max(fixed, textWidth(l.textContent ?? "", getComputedStyle(l).font) + 8))));
  let control = 170;
  for (const ctl of body.querySelectorAll<HTMLElement>(".prow .ctl")) {
    const select = ctl.querySelector<HTMLSelectElement>("select");
    const file = ctl.querySelector<HTMLElement>(".fp-line");
    if (select) {
      const font = getComputedStyle(select).font;
      control = Math.max(control, ...[...select.options].map((o) => textWidth(o.text, font) + 34));
    } else if (file) {
      // 文件参数：其按钮，加上说明所选内容的整行文字
      const parts = [...file.children] as HTMLElement[];
      const text = parts.reduce((w, el) => w + textWidth(el.textContent ?? "", getComputedStyle(el).font) + 12, 0);
      const buttons = [...ctl.querySelectorAll<HTMLElement>("button")].reduce((w, b) => w + b.offsetWidth + 6, 0);
      control = Math.max(control, Math.min(480, text + buttons));
    }
  }
  return Math.ceil(label + 10 + control + 10 + 32 + 10); // 行间距、分组内边距、滚动条
}


/** 「骨架姿势」参数（widget "skeleton_pose"）：面板上只说改了几个关节、给一个清空；改在三维视图里做（舞台的骨架姿势手柄：
 * 显示这个节点时点关节、拖手柄或填数，view/skeletonPose.tsx）。 */
function PoseSummary({ nodeId, p, value, set }: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown) => void }) {
  const n = Array.isArray(value) ? value.length : 0;
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const handle = getNodeDefs()[typeId]?.handles.findIndex((h) => role3d(h) === "pose" && h.params.pose === p.name) ?? -1;
  const on = useHandleView((s) => s.editing?.node === nodeId && s.editing.handle === handle);
  // 主视图默认不画姿势手柄（与节点结果里的骨架重叠）：这里打开它，显示这个节点、只画这一副（state/handleView.ts editing）
  const toggle = () => {
    if (on) return useHandleView.getState().setEditing(null);
    if (useLook.getState().displayId !== nodeId) useLook.getState().setDisplay(nodeId);
    useHandleView.getState().setEditing({ node: nodeId, handle });
  };
  return (
    <div className="picks">
      <span className="dim">
        {n ? t("ui.params.joints_fixed", { count: n }) : t("ui.params.joints_unfixed")}
      </span>
      {handle >= 0 && (
        <Button size="sm" tone={on ? "primary" : "default"} on={on} onClick={toggle}>
          {t(on ? "ui.params.edit_exit" : "ui.params.edit_in_view")}
        </Button>
      )}
      <Button size="sm" disabled={!n} onClick={() => set([])}>{t("ui.params.clear")}</Button>
    </div>
  );
}

/** 双骨架编辑手柄（rig_pair）的哪些角色在面板上是摘要行（其余角色：忽略规则是普通下拉，auto_record 不在面板上）。 */
const RIG_ROWS: Record<string, RigMode> = { mapping: "map", src_pose: "pose", dst_pose: "pose", src_ignore: "ignore", dst_ignore: "ignore" };

/** 参数 `name` 是节点某个 rig_pair 手柄的哪个角色（在面板上画成摘要行的那几个），连同那个手柄的序号。 */
function rigRoleOf(def: { handles: { kind: string; params: Record<string, string> }[] } | undefined, name: string): { handle: number; role: string } | null {
  for (const [k, h] of (def?.handles ?? []).entries()) {
    if (role3d(h) !== "pair") continue;
    const role = Object.entries(h.params).find(([r, q]) => q === name && r in RIG_ROWS)?.[0];
    if (role) return { handle: k, role };
  }
  return null;
}

/** 双骨架编辑的参数行（对应关系、姿势、忽略）：面板上只写摘要（对应关系「手动 N 个部位，其余自动 · 有问题：…」，
 * 问题要手柄数据在手时才判得出），改在三维视图里做（view/rigPair.tsx）——「在视图里编辑」显示这个节点、打开这个手柄，
 * 并切到这一行对应的模式；同一行再点一下收起。姿势与忽略另有「清空」。 */
function RigPairParam({ nodeId, value, handle, role }: { nodeId: string; value: unknown; handle: number; role: string }) {
  const mode = RIG_ROWS[role];
  const on = useHandleView((s) => s.editing?.node === nodeId && s.editing.handle === handle);
  const here = useRigPairView((s) => s.mode === mode);
  const hd = useResults((s) => s.reply?.handle_data);
  const data = hd?.node === nodeId ? (hd.handles?.[String(handle)] as RigPairData | undefined) : undefined;
  const n = Array.isArray(value) ? value.length : 0;
  // 默认自动（model/rigPair.ts effective）：从没执行过这一项的「自动」、参数也空着时，计算按服务端的建议走
  const params = useCookInputs((s) => s.nodes[nodeId]?.params);
  const roles = (getNodeDefs()[useCookInputs.getState().nodes[nodeId]?.typeId ?? ""]?.handles[handle]?.params ?? {}) as RigRoles;
  const record = roles.auto_record ? recordOf(params?.[roles.auto_record]) : null;
  const ignoresEmpty = (["src_ignore", "dst_ignore"] as const).every((r) => !roles[r] || !namesOf(params?.[roles[r]!]).length);
  const auto = mode === "pose" ? !n && !record?.pose : mode === "ignore" ? ignoresEmpty && !record?.ignore : false;
  // 清空也是一次手动改：第一次手动改要补上 auto_record（model/rigPair.ts withTouched），之后空着就是空
  const paramName = roles[role as keyof RigRoles] ?? "";
  const cleared = mode === "map" || !paramName ? null : withTouched({ [paramName]: [] }, data ?? null, roles, params);
  const said = mode === "map" ? mappingSummary(value, data ?? null)
    : { text: auto ? t("ui.params.rig_auto") : mode === "pose" ? (n ? t("ui.params.joints_fixed", { count: n }) : t("ui.params.joints_unfixed")) : (n ? t("ui.params.joints_ignored", { count: namesOf(value).length }) : t("ui.params.joints_none_ignored")), bad: false };
  const open = () => {
    if (on && here) return useHandleView.getState().setEditing(null);
    if (useLook.getState().displayId !== nodeId) useLook.getState().setDisplay(nodeId);
    useRigPairView.getState().set({ mode, first: null, picked: null, menu: null });
    useHandleView.getState().setEditing({ node: nodeId, handle });
  };
  return (
    <div className="picks">
      <span className={said.bad ? "rpair-sum bad" : "dim"}>{said.text}</span>
      <Button size="sm" tone={on && here ? "primary" : "default"} on={on && here} onClick={open}>{t(on && here ? "ui.params.edit_exit" : "ui.params.edit_in_view_long")}</Button>
      {mode !== "map" && <Button size="sm" disabled={!n || !cleared} tip={n && !cleared ? tipOf("disabled", t("ui.params.clear_auto_tip")) : undefined}
        onClick={() => cleared && setParams(nodeId, cleared)}>{t("ui.params.clear")}</Button>}
    </div>
  );
}

// ------------------------------------------------------------------ the open editor window, at the page's top

/** The editor window that is open (editor/ParamSheet.tsx useOpenSheet), drawn at the page's top rather than on its
 * parameter row: another node selected, a newer version of the graph read in, the parameter panel redrawn — none of
 * them closes it or loses its draft; only the user does. When its node or parameter is gone (the graph read in no
 * longer has it) the window says so and waits to be closed. */
export function OpenSheet() {
  const open = useOpenSheet((s) => s.open);
  const lock = useOpenSheet((s) => s.lock);
  const typeId = useCookInputs((s) => (open ? s.nodes[open.nodeId]?.typeId : undefined));
  const params = useCookInputs((s) => (open ? s.nodes[open.nodeId]?.params : undefined));
  const answer = useResults((s) => (open ? s.results[open.nodeId]?.applies : undefined));
  if (!open) return null;
  const close = () => useOpenSheet.getState().close();
  const def = typeId ? getNodeDefs()[typeId] : undefined;
  const here = def && params ? paramAt(def, params, open.at) : null;
  const kind = here ? SHEET_EDITORS[here.p.widget ?? ""] : undefined;
  if (!here || !kind) {
    return (
      <Sheet title={t("ui.common.edit")} width={420} onClose={close}>
        <div className="psheet-none">{t(!params ? "ui.params.sheet_node_gone" : "ui.params.sheet_param_gone")}</div>
      </Sheet>
    );
  }
  const { nodeId, at } = open;
  // 写回按位置（editor/paramPath.ts placeAt）：格子写的是整张表、只改那一格，以写的那一刻节点的参数为准
  const set = (v: unknown, also: Record<string, unknown>) => {
    const now = useCookInputs.getState().nodes[nodeId]?.params ?? {};
    setParams(nodeId, { ...(v === undefined ? {} : placeAt(now, at, v)), ...also });
  };
  // 窗按行名记位置（paramPath stableAt）；服务器的回答按这一行现在的下标（indexAt）
  const x = { nodeId, p: here.p, value: here.value, set, kind, lock: lock || whyOff(answer, indexAt(params ?? {}, at) ?? at) };
  return here.p.choices_from.length ? <OpenSheetWithChoices {...x} /> : <SheetWindow {...x} choices={{}} close={close} />;
}

/** The window's data, asked the same way as the parameter row's (ui/choices.ts). While the window is open only data the
 * editor can use (its `ready`) replaces what it has: a failed or empty answer keeps the last usable one, and the window
 * says so, instead of handing the editor something it cannot draw. */
function OpenSheetWithChoices(x: { nodeId: string; p: ParamDef; value: unknown; set: (v: unknown, also: Record<string, unknown>) => void; kind: SheetEditor; lock: string }) {
  const choices = useChoiceSet(x.nodeId, x.p);
  const usable = (c: Record<string, Choice> | null) => !!c && (x.kind.ready ? x.kind.ready({ nodeId: x.nodeId, p: x.p, value: x.value, choices: c, set: (v) => x.set(v, {}) }) : true);
  const kept = useRef(usable(choices) ? choices : null);
  if (usable(choices)) kept.current = choices;
  const note = kept.current && choices !== kept.current ? t("ui.params.sheet_stale", { why: choices?.[""]?.empty ?? t("ui.params.sheet_rereading") }) : "";
  // nothing usable yet (the window's first frame, the file still being read, or a read that failed): the window opens
  // and says so, the editor is not mounted until an answer it can draw arrives (SheetWindow `pending`)
  if (!kept.current) return <SheetWindow {...x} choices={{}} pending={choices?.[""]?.empty ?? t("ui.params.sheet_reading")} close={() => useOpenSheet.getState().close()} />;
  return <SheetWindow {...x} choices={kept.current} note={note} close={() => useOpenSheet.getState().close()} />;
}
