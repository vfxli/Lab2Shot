import type { LoopMode } from "../state/preferences";
import { Select } from "../ui/Select";
import { t } from "../i18n/t";

/** 时间线「播放方式」下拉的唯一所在。播放方式：循环 / 往返 / 单次，属于该浏览器自身的设定（state/preferences.ts），不随节点图保存。 */
const MODES: Record<LoopMode, string> = { loop: "ui.timeline.loop", bounce: "ui.timeline.bounce", once: "ui.timeline.once" }; // keys of the words

/** 时间线最右侧只有播放方式一个下拉。逐帧与跳转使用快捷键（← → ↑ ↓），播放范围通过拖动标尺上的两个把手设置。
 * 「实时」不是开关：播放一律采用 Nuke 的方式（缓存未到达时等待，第一遍较慢，第二遍实时），因此此处不设其他选项。 */
export function MoreMenu({ prefs, prefer }: {
  prefs: { mode: LoopMode };
  prefer: (p: Partial<{ mode: LoopMode }>) => void;
}) {
  return (
    <Select
      label={t("ui.timeline.play_mode")}
      value={prefs.mode}
      className="tl-loopmode"  /* 该行只有标尺是弹性的，其余一律固定宽度：见 styles/07-timeline.css */
      options={(Object.keys(MODES) as LoopMode[]).map((m) => ({ value: m, label: t(MODES[m]) }))}
      onPick={(v) => prefer({ mode: v as LoopMode })}
    />
  );
}

/* 画质没有可切换的档位，所以时间线上也没有对应的下拉：二维只有视图代理一种方式（计算完成后按管理员设置的档位缩放、压缩并保存）；
   点云按后台「视图 · 点云上限」删减点数，始终生效且没有开关，删减数量由视图通知区的「显示了 N / 共 M 点」说明。 */
