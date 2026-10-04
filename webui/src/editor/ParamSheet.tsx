/** 弹窗编辑参数（框架能力）：参数面板上显示「当前值的摘要 + 『编辑…』」，点开一个 Sheet（ui/Sheet.tsx），在窗里操作，
 * 「确定」把结果写回参数并关上，「取消」/ Esc / 点遮罩关上、不写回。「弹一个窗、在里面操作、把结果拿回参数」归这里；
 * 窗里具体怎么操作归各自的编辑器组件（如 USD 层级选择 HierarchyPicker.tsx；本文件末尾的 TextSheet 是
 * 最小演示）。不挂在某个参数上的窗（参数面板标题行的「编辑参数界面」，ParamInterfaceEditor.tsx）直接用 SheetWindow。
 *
 * 新增一种弹窗编辑器，三步：
 *   1. 服务端给参数声明控件名：`P(..., widget="<名字>")`。窗里要用文件或上游给的数据时再加 `choices_from=[...]`，
 *      数据由节点的 `NodeDef.choices` 给出（POST /api/nodes/{type}/choices，ui/choices.ts），框架替编辑器取好。
 *      widget 名服务端不用登记：非简单控件自动只放参数面板（nodes/params.py simple_kind）。
 *   2. 写一个编辑器组件，入参是 `SheetEditorProps`：打开时的 `value` 与节点参数 `base`（窗的基准，开着时不跟别处的改动变）、`set(新值, 附带参数?)`（= 确定：写回并关窗）、`cancel()`、
 *      `nodeId`、参数定义 `p`、`choices`。窗里改的是组件自己的草稿（useDraft：只读标签页里改不动；改草稿的按钮包进 Writes），只有调 `set` 才写回；底部用 `<SheetFoot>` 放
 *      「取消 / 确定」。然后把它包成一个 `SheetEditor`（至少 `editor`；可选窗名、按钮字、摘要、何时可点、窗宽）。
 *   3. 在 editor/ParamControls.tsx 的 `SHEET_EDITORS` 里按 widget 名登记这一项。分发（`Control` 里的一个分支）已经接好。
 */

import { useWriteLock, WriteLock } from "../ui/writeLock";
import { createContext, Suspense, useContext, useEffect, useState } from "react";
import { create } from "zustand";
import { createPortal } from "react-dom";
import type { Choice, ParamDef } from "../api";
import { useChoiceSet } from "../ui/choices";
import { Sheet } from "../ui/Sheet";
import { Button } from "../ui/Button";
import { ErrorBoundary } from "../ui/ErrorBoundary";
import { composing } from "../platform/keys";
import { readOnlyWhy, useCookInputs, useReadOnly } from "../state/cookInputs";
import { same, type Json } from "../model/graphPatch";
import { t } from "../i18n/t";
import { stableAt } from "./paramPath";
import { tipOf, type Tip } from "../platform/tips";

/** 编辑器组件（窗里的内容）拿到的东西。 */
export interface SheetEditorProps {
  nodeId: string;
  p: ParamDef;
  /** 打开窗时参数的值（窗的基准，别处改了它也不跟着变，见 SheetWindow）：编辑器以它为草稿的初值 */
  value: unknown;
  /** 打开窗时这个节点的全部参数（同一基准）：用 `set` 的 `also` 带写别的参数的编辑器以它为那些参数草稿的初值、
   * 判断改没改，不读节点现在的参数 */
  base: Record<string, unknown>;
  /** 服务端 NodeDef.choices 给的数据（参数声明了 choices_from 时；没声明的为空表），与面板上其他控件同一份。编辑器挂上时
   * 它一定是 `ready` 认过的那份：没拿到能用的数据时窗不挂编辑器（SheetWindow `pending`），编辑器不用自己防 */
  choices: Record<string, Choice>;
  /** 确定：把新值写回参数，并关上窗。`also`：同一节点的别的参数与它一起写（一步撤销；「对应关系」连同两侧的姿势修正）。
   * `also` 的键是节点参数名；窗挂在表格格子上时（`表[i].列`，editor/paramPath.ts）值写回那一格，`also` 照样写节点参数 */
  set: (v: unknown, also?: Record<string, unknown>) => void;
  /** 取消：关上窗，不写回（Esc、点遮罩也是这个） */
  cancel: () => void;
}

/** 面板上摘要（以及按钮旁的附加控件）拿到的东西：`set` 直接写参数（例如「清掉」），不经过窗。 */
export interface SheetSummaryProps {
  nodeId: string;
  p: ParamDef;
  value: unknown;
  choices: Record<string, Choice> | null;
  set: (v: unknown) => void;
}

/** 一种弹窗编辑器：按 widget 名登记在 ParamControls.tsx 的 SHEET_EDITORS 里。 */
export interface SheetEditor {
  /** 窗里的内容。引入三维舞台（view/ 下的三维部件）的编辑器只能按需加载（React.lazy），three.js 不进主包 */
  editor: React.ComponentType<SheetEditorProps>;
  /** 窗的标题，默认「编辑<参数名>」 */
  title?: (p: ParamDef) => string;
  /** 打开按钮上的字，默认「编辑…」 */
  button?: () => string;
  /** 窗宽（px），默认 560；"content"：按内容宽（不超过窗口，ui/Sheet.tsx） */
  width?: number | "content";
  /** 面板上当前值的摘要，默认把值写成一行字（空值显示参数的 placeholder 或「未设置」） */
  summary?: React.ComponentType<SheetSummaryProps>;
  /** 按钮右边的附加控件（USD 层级的「清掉」） */
  actions?: React.ComponentType<SheetSummaryProps>;
  /** 现在能不能打开（例如还没选文件、没有可选的东西）：不能时按钮置灰，默认总能打开 */
  ready?: (x: SheetSummaryProps) => boolean;
  /** 这个编辑器用 `set` 的 `also` 连带写的节点参数（「对应关系」两侧的姿势修正）：窗的基准也包括它们，别处改了它们也在
   * 窗顶说（SheetWindow）。默认没有 */
  also?: (x: { nodeId: string; p: ParamDef; choices: Record<string, Choice> | null }) => string[];
}

/** `lock`：这个参数现在改不了的原因（只读标签页、条件不适用；"" 能改）。打开窗是看，从不因它置灰；窗里按它只锁写。 */
type SheetParamProps = { nodeId: string; at: string; p: ParamDef; value: unknown; set: (v: unknown) => void; kind: SheetEditor; lock?: string };
/** 参数面板里的「摘要 + 编辑…」；它打开的窗在页面顶层画（ParamControls.tsx OpenSheet）。参数声明了 choices_from 才去问
 * 服务端要数据（ui/choices.ts）。 */
export function SheetParam(props: SheetParamProps) {
  return props.p.choices_from.length ? <WithChoices {...props} /> : <SheetParamBody {...props} choices={null} />;
}

function WithChoices(props: SheetParamProps) {
  const choices = useChoiceSet(props.nodeId, props.p);
  return <SheetParamBody {...props} choices={choices} />;
}

/** 开着的那个编辑窗（节点、值在节点里的位置 `at`：参数名或表格格子 `表[i].列`，editor/paramPath.ts），与它现在改不了的原因（参数面板上它那一行在时由那一行给）。窗不挂在参数面板的行上：
 * 换了选中、读进另一个标签页写的新版本（参数面板整个重画）都不关它、不丢草稿；只由使用者关（ParamControls.tsx
 * OpenSheet 在页面顶层画它）。 */
export const useOpenSheet = create<{
  open: { nodeId: string; at: string } | null;
  lock: string;
  show: (nodeId: string, at: string) => void;
  setLock: (lock: string) => void;
  close: () => void;
}>((set) => ({
  open: null,
  lock: "",
  show: (nodeId, at) => set({ open: { nodeId, at }, lock: "" }),
  setLock: (lock) => set((s) => (s.lock === lock ? s : { lock })),
  close: () => set({ open: null, lock: "" }),
}));

function SheetParamBody({ nodeId, at, p, value, set, kind, choices, lock = "" }: SheetParamProps & { choices: Record<string, Choice> | null }) {
  // 开着的窗记的是行名形式的位置（editor/paramPath.ts stableAt）：按这一行现在的名字比
  const mine = useOpenSheet((s) => s.open?.nodeId === nodeId && s.open.at === stableAt(useCookInputs.getState().nodes[nodeId]?.params ?? {}, at));
  // 开着的是这一行的窗：这一行在时，锁的原因由它给（只读标签页、参数不适用、Disable When）
  useEffect(() => { if (mine) useOpenSheet.getState().setLock(lock); }, [mine, lock]);
  const x: SheetSummaryProps = { nodeId, p, value, choices, set };
  const ready = kind.ready ? kind.ready(x) : true;
  const Summary = kind.summary ?? PlainSummary;
  const Actions = kind.actions;
  return (
    <div className="psheet">
      <div className="psheet-sum">
        <Summary {...x} />
      </div>
      <div className="psheet-actions">
        <Button layout="psheet-open" disabled={!ready} onClick={() => useOpenSheet.getState().show(nodeId, stableAt(useCookInputs.getState().nodes[nodeId]?.params ?? {}, at))}>
          {kind.button ? kind.button() : t("ui.params.sheet_edit_button")}
        </Button>
        {Actions && <fieldset className="psheet-lock" disabled={!!lock}><Actions {...x} /></fieldset>}
      </div>
    </div>
  );
}

/** 窗本身：标题、宽度按编辑器登记的来，里面挂编辑器；「确定」= 写回再关，「取消」/ Esc / 点遮罩 = 关、不写回。参数面板上的
 * 「摘要 + 编辑…」（SheetParamBody）和不属于某个参数的入口（参数面板标题行的「编辑参数界面」，ParamPanel.tsx）都用它，
 * 这套约定只有这一处。 */
export function SheetWindow({ kind, nodeId, p, value, choices, set, close, lock = "", note = "", pending = "" }: {
  kind: SheetEditor; nodeId: string; p: ParamDef; value: unknown; choices: Record<string, Choice>; close: () => void;
  /** 写回：`v` 是新值（undefined：值本身没改，只写 `also`），`also` 是同一节点别的参数（一步撤销）。只在有改动时调 */
  set: (v: unknown, also: Record<string, unknown>) => void;
  lock?: string; // 这个参数现在改不了的原因（条件不适用）；只读标签页另由 useReadOnly 知道
  note?: string; // 窗顶的一句说明（数据暂时取不到、沿用打开时的那份）
  /** 编辑器要的数据一份能用的都还没有（刚打开、还在读，或读失败）：窗开着、只写这句，不挂编辑器——编辑器拿到的
   * `choices` 永远是它 `ready` 认过的那份，不用自己防 null */
  pending?: string;
}) {
  const Editor = kind.editor;
  // 改不了（只读标签页，或参数现在不适用）禁的是写，不是看、不是关：窗照常打开，写明原因；看的控件（页签、树 / 3D、
  // 搜索、展开、框显）照常能用；改草稿的一律不生效（useDraft），改草稿的按钮置灰（Writes）；「确定」写不回，「取消」
  // 照常关窗。编辑中途变成只读（另一个标签页接手）也一样：窗不关，草稿留着，只是锁住
  const readOnly = useReadOnly();
  const why = readOnly ? readOnlyWhy() : lock;
  const locked = !!why;
  const [footAt, setFootAt] = useState<HTMLDivElement | null>(null);
  // 窗的基准：打开时的值与节点参数。编辑器的草稿从它起步；它在窗开着时不跟着变（别的标签页接手又交回、撤销，都可能
  // 把参数改掉），改了就在窗顶说，由使用者选「以新值重来」（换基准、编辑器按新值重开，草稿丢掉）或照旧改。「确定」只写
  // 与基准不同的参数：没动过的不会把别处的新值盖回旧值
  const [base, setBase] = useState(() => ({ value, params: (nodeId && useCookInputs.getState().nodes[nodeId]?.params) || {}, n: 0 }));
  // 别处改过的：这个值，或编辑器连带写的那些参数（SheetEditor.also）
  const alsoKeys = kind.also ? kind.also({ nodeId, p, choices }) : [];
  const nowParams = useCookInputs((s) => (nodeId ? s.nodes[nodeId]?.params : undefined));
  const moved = !same(value as Json, base.value as Json) || alsoKeys.some((k) => !same(nowParams?.[k] as Json, base.params[k] as Json));
  const restart = () => setBase((b) => ({ value, params: (nodeId && useCookInputs.getState().nodes[nodeId]?.params) || {}, n: b.n + 1 }));
  return (
    <Sheet title={kind.title ? kind.title(p) : t("ui.params.sheet_title", { label: p.label })} width={kind.width ?? 560} onClose={close}>
      {locked && <div className="psheet-locked">{why}</div>}
      {note && <div className="psheet-locked">{note}</div>}
      {moved && (
        <div className="psheet-locked">
          {t("ui.params.sheet_moved")}
          <Button tone="ghost" size="sm" onClick={restart}>{t("ui.params.sheet_restart")}</Button>
        </div>
      )}
      <WriteLock.Provider value={why}>
      <FootSlot.Provider value={footAt}>
        {/* 编辑器可以按需加载（lazy：带三维舞台的「对应关系」），到之前窗里先写一句 */}
        {/* 没载进来时只这个窗报错、可重试（platform/lazyRetry.tsx），参数面板不受牵连 */}
        {pending ? <div className="psheet-none">{pending}</div> : (
        <ErrorBoundary name={kind.title ? kind.title(p) : t("ui.params.sheet_title", { label: p.label })}>
        <Suspense fallback={<div className="psheet-none">{t("ui.params.sheet_loading")}</div>}>
        <Editor key={base.n} nodeId={nodeId} p={p} value={base.value} base={base.params} choices={choices} cancel={close}
          set={(v, also) => {
            // 锁着时「确定」什么都不做，也不关窗（回车、双击这类不经按钮的确定也是）：关窗不丢草稿，只由使用者关
            if (locked) return;
            close();
            const rest = Object.entries(also ?? {}).filter(([k, x]) => !same(x as Json, base.params[k] as Json));
            const mine = !same(v as Json, base.value as Json);
            if (mine || rest.length) set(mine ? v : undefined, Object.fromEntries(rest));
          }} />
        </Suspense>
        </ErrorBoundary>
        )}
      </FootSlot.Provider>
      </WriteLock.Provider>
      <div ref={setFootAt} className="psheet-foot" />
    </Sheet>
  );
}

/** 默认摘要：值写成一行字（数组、对象写成 JSON），空值显示参数的 placeholder 或「未设置」。 */
function PlainSummary({ p, value }: SheetSummaryProps) {
  const empty = value === null || value === undefined || value === "" || (Array.isArray(value) && !value.length);
  if (empty) return <span className="psheet-none">{p.placeholder || t("ui.params.unset")}</span>;
  const text = typeof value === "string" ? value : JSON.stringify(value);
  return <span className="psheet-text" data-user-data>{text}</span>;
}

/** 窗的最后一行：左边放编辑器自己的东西（已选几个、全部不选…），右边「取消」「确定」。 */
/** SheetWindow 里放底行的地方（编辑器之后）。 */
const FootSlot = createContext<HTMLElement | null>(null);

/** 窗里现在改不了的原因（ui/writeLock.ts：SheetWindow 给只读标签页或参数不适用；窗外按只读标签页）；"" 能改。 */
export const useSheetLock = useWriteLock;

/** 编辑器的草稿（窗里改的东西，「确定」时写回）：改不了时（useSheetLock）改它不生效，已有的草稿留着。草稿用它，看的
 * 状态（页签、展开、搜索）用普通的 useState：禁写不禁看，分在这一处。 */
export function useDraft<T>(init: T | (() => T)): [T, React.Dispatch<React.SetStateAction<T>>] {
  const [value, set] = useState(init);
  const locked = !!useSheetLock();
  return [value, locked ? () => undefined : set];
}

/** 改草稿的控件（按钮、勾选）包在它里面：改不了时整组置灰。看的控件不包。菜单行这类不是表单控件的写入口，用
 * useSheetLock 自己置灰。 */
export function Writes({ children }: { children: React.ReactNode }) {
  return <fieldset className="psheet-lock" disabled={!!useSheetLock()}>{children}</fieldset>;
}

/** 窗的最后一行：左边放编辑器自己的东西（已选几个、全部不选…），右边「取消」「确定」。只读时「取消」照常能点（关窗
 * 不是写），「确定」与左边编辑器自己的东西（Writes）置灰。`okTip`：「确定」的后果里看不到的那部分（没有就不给）。 */
export function SheetFoot({ ok, cancel, okTip, layout, children }: {
  ok: () => void; cancel: () => void; okTip?: Tip; layout?: string; children?: React.ReactNode;
}) {
  const slot = useContext(FootSlot);
  const why = useSheetLock();
  const locked = !!why;
  const row = (
    <div className={`dialog-row${layout ? ` ${layout}` : ""}`}>
      <Writes>{children}</Writes>
      <span style={{ flex: 1 }} />
      <Button tone="ghost" onClick={cancel}>
        {t("ui.common.cancel")}
      </Button>
      <Button tip={locked ? tipOf("disabled", why) : okTip} tone="primary" disabled={locked} onClick={ok}>
        {t("ui.common.ok")}
      </Button>
    </div>
  );
  return slot ? createPortal(row, slot) : row;
}

/** 最小演示（widget「sheet_text」）：窗里一个文本框，改的是草稿；回车或「确定」写回，「取消」/ Esc / 点遮罩不写回。
 * 摘要用框架默认的一行字。任何节点给一个文本参数声明 `P("", label=…, widget="sheet_text")` 就得到这个弹窗。 */
function TextSheetEditor({ p, value, set, cancel }: SheetEditorProps) {
  const [text, setText] = useDraft(typeof value === "string" ? value : "");
  const ok = () => set(text === "" && p.nullable ? null : text);
  return (
    <>
      <input className="field" autoFocus value={text} placeholder={p.placeholder} spellCheck={false}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && !composing(e) && (e.preventDefault(), ok())} />
      <SheetFoot ok={ok} cancel={cancel} />
    </>
  );
}

export const TextSheet: SheetEditor = { editor: TextSheetEditor };
