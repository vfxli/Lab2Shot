import { mixOf, rampGradient, type Bg, type Mode, type Op, type Tint } from "../model/view2d";
import { CHOICES, RANGES } from "../model/viewOptions";
import { BG_CHOICES, bgColourOf } from "../platform/palette";
import { Button, Segmented } from "../ui/Button";
import { Select } from "../ui/Select";
import { Num } from "../ui/controls";
import { coerce } from "../model/numbers";
import { t } from "../i18n/t";

/** 二维预览链在控制栏上的组成部分，三种模式下是同一行控件：
 *
 *     [ 仅原图 │ 运算 │ 仅结果 ]  左：通道▾ │ 运算▾（加 / 乘 Alpha / 乘 RGBA）· mix │ 右：通道▾ │ 着色▾ │ 黑点 白点 自动 │ 背景▾
 *
 * 切换模式不增减控件，控件位置不变，不适用的控件置灰（是否适用由 model/viewControls.ts 声明，
 * 此处只读 view/available.ts 的回答）。这一行在视图的工具栏里，不显示悬停提示；下拉打开的菜单照常显示。
 *
 * 黑白点与着色位于该行，不在「视图设置」中。 */

/** 模式：仅原图 / 运算 / 仅结果。 */
export function ModePick({ mode, onPick }: { mode: Mode; onPick: (m: Mode) => void }) {
  return (
    <Segmented hud label={t("ui.view.preview")} value={mode}
      options={[
        { value: "plate" as Mode, label: t("ui.view.mode_plate") },
        { value: "over" as Mode, label: t("ui.view.mode_over") },
        { value: "result" as Mode, label: t("ui.view.mode_result") },
      ]}
      onChange={onPick} />
  );
}

/** 一侧的通道下拉：层与通道列在同一下拉中，同时提供整体与单通道（与 Nuke 一致）。数据中不存在的层与通道不列出。 */
export function ChannelPick({ side, value, options, onPick, off = false }:
  { side: "left" | "right"; value: string; options: { value: string; label: string }[]; onPick: (v: string) => void;
    off?: boolean }) {
  const empty = side === "left" ? t("ui.view.no_plate") : t("ui.view.no_result");
  // 只有一项时也必须可以打开：该控件用于选择右侧显示本节点的哪份结果及哪条通道，而非仅用于选择通道。
  // 单通道结果（各类抠像、遮罩、深度、置信度）始终只有一项，若「只有一项即置灰」，该控件对这些节点
  // 将始终不可用，使用者无法查看计算结果。只有一种情况应置灰：没有任何可选项（该链上尚无结果）。
  // 只剩箭头的下拉无法让使用者判断是故障还是没有内容：没有可选项时，触发器上显示「没有原图」「还没有结果」
  return (
    <Select className="port-pick fit sm" label={side === "left" ? t("ui.view.plate_channel") : t("ui.view.result_channel")} value={options.length ? value : ""} onPick={onPick} width={240}
      disabled={off || !options.length}
      options={options.length ? options.map((o) => ({ value: o.value, label: o.label })) : [{ value: "", label: empty }]} />
  );
}

/** 中间运算：一个四档下拉（加 · 盖上 · 乘 Alpha · 乘 RGBA）加一个强度控件。作用对象仅对「乘」有意义
 * （model/view2d.ts 的 `Op`），因此不单独设置 RGBA / 仅 Alpha 选项。 */
export function MergePick({ op, mix, offOp = false, offMix, onOp, onMix }:
  { op: Op; mix: number; offOp?: boolean; offMix: boolean;
    onOp: (v: Op) => void; onMix: (v: number) => void }) {
  // 滑块显示的是实际使用的强度：切换到「乘」时立即变为 1.00 且不可拖动，
  // 而非显示 0.5 却按 1 计算（屏幕显示与数据必须一致）
  const shown = mixOf(op, mix);
  return (
    <>
      <Select className="port-pick fit sm" label={t("ui.view.mode_over")} value={op} onPick={(v) => onOp(v as Op)} width={220}
        disabled={offOp}
        options={(Object.keys(CHOICES.op) as Op[]).map((v) => ({ value: v, label: t(CHOICES.op[v]) }))} />
      <label className="hud-mix">
        <span>mix</span>
        {/* `--pct` 表示填充进度：滑块的蓝色条据此绘制（styles/03-top-bar.css 的 runnable-track）。
            缺少该值时会停留在默认的 50%，出现数字为 1.00 而色条只有一半的情况 */}
        <input type="range" min={RANGES.mix[0]} max={RANGES.mix[1]} step={0.01} value={shown} disabled={offMix}
          style={{ ["--pct" as string]: `${((shown - RANGES.mix[0]) / (RANGES.mix[1] - RANGES.mix[0])) * 100}%` }}
          aria-label="mix" onChange={(e) => { const n = coerce(Number(e.target.value), { min: RANGES.mix[0], max: RANGES.mix[1] }); if (n !== null) onMix(n); }} />
        <b>{shown.toFixed(2)}</b>
      </label>
    </>
  );
}

/** 着色：单通道显示时的颜色方案（若干色标与纯色）。 */
export function TintPick({ value, off, onPick }: { value: Tint; off: boolean; onPick: (v: Tint) => void }) {
  return (
    <Select className="port-pick fit sm" label={t("ui.display.tint")} value={value} onPick={(v) => onPick(v as Tint)} width={200} disabled={off}
      options={(Object.keys(CHOICES.tint) as Tint[]).map((v) => ({
        value: v,
        label: (<span className="tint-row"><i className="tint-bar" style={{ background: rampGradient(v) }} />{t(CHOICES.tint[v])}</span>),
      }))} />
  );
}

/** 黑点 / 白点 / 自动：以数据自身范围的 0 到 1 表示，不裁切（参照 Nuke 的 Grade）。 */
export function GradePick({ black, white, off, onSet, fit }:
  { black: number; white: number; off: boolean; onSet: (p: { black?: number; white?: number }) => void; fit: () => { black: number; white: number } }) {
  const [lo, hi] = RANGES.black;
  // 黑点 / 白点：这份数据里映射成黑 / 白的那个值，用它自己范围的 0 到 1 表示（不是绝对值），超出的不裁切
  const num = (label: string, value: number, set: (v: number) => void, after?: React.ReactNode) => (
    <label className="hud-num">
      <span>{label}</span>
      <Num value={value} min={lo} max={hi} digits={3} disabled={off} label={label} onChange={set} />
      {after}
    </label>
  );
  return (
    <>
      {num(t("ui.view.black_point"), black, (v) => onSet({ black: v }))}
      {/* 「自动」并入「白点」格：其作用是同时设置黑点与白点，与这两个数字属于同一功能；
          单独设置按钮会多占约 50 px，在 1600 宽度下会使该行换到第二行。对应 Photoshop 色阶中的「自动」按钮 */}
      {/* 自动：按这份数据里真实的最小和最大值拉满（包里记的 1%–99% 本身就在裁切，这里用的是不裁切的那一份） */}
      {num(t("ui.view.white_point"), white, (v) => onSet({ white: v }),
           <Button tone="ghost" size="sm" layout="hud-num-auto" disabled={off} onClick={() => onSet(fit())}>
             {t("ui.view.auto")}
           </Button>)}
    </>
  );
}

/** 背景：透明区域的填充方式。与运算完全独立，三种模式下均可用。
 *
 * 使用单个下拉，而非「开关 + 色板」两个控件：棋盘格与各颜色是同一选项的不同取值，拆成两个控件需要点击两次
 * 才能完成一次设置，该行也会多占 150 px。 */
export function BgPick({ bg, colour, onPick }: { bg: Bg; colour: string; onPick: (p: { bg: Bg; bgColor?: string }) => void }) {
  const now = bg === "checker" ? "checker" : bgColourOf(colour);
  return (
    <Select className="port-pick fit sm" label={t("ui.display.background")} value={now} width={200}
      onPick={(v) => onPick(v === "checker" ? { bg: "checker" } : { bg: "solid", bgColor: v })}
      options={[
        { value: "checker", label: <span className="tint-row"><i className="tint-bar checker-bar" />{t(CHOICES.bg.checker)}</span> },
        ...BG_CHOICES.map((c) => ({
          value: c.value,
          label: <span className="tint-row"><i className="tint-bar" style={{ background: c.value }} />{t(c.name)}</span>,
        })),
      ]} />
  );
}

