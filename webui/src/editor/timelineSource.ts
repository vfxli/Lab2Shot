import { useEffect, useMemo } from "react";
import type { Manifest } from "../api";
import { useViewer } from "../state/viewer";
import { joinLayers, markLayers, type MarkLayer } from "../model/timelineMath";
import { useDisplayPlan } from "../view/plan";
import { useDescribed } from "../transfer/described";

/** 时间线为视图显示的节点跟随的内容（本模块是其唯一来源）：其第一个结果（或其处理的素材）的帧、其自身结果携带的标记
 * （"keys" 与 "marks"），以及正在绘制的节点上火柴人手柄已摆好姿势的帧。
 *
 * 最后一项用于在绘制一段后切换帧检查连续性。它读取的是参数而非计算结果，因此点击「添加帧」后
 * 标尺上立即出现标记，无须先计算（计算后结果自身带的标记与之合并，两者不冲突）。使用的是标尺上
 * 已有的标记机制（`tl-mark`，与 `markLayers` 同一路径），不增加新控件。 */
export function useSource(): { layers: MarkLayer[] } {
  const plan = useDisplayPlan();
  const follows = plan?.frames ?? null;
  const own = plan ? plan.items.filter((it) => it.fp && !it.context).map((it) => it.fp!) : [];
  const [followed] = useDescribed<Manifest>("manifest", follows ? [follows] : []);
  // 帧表属于读进来的这张图：读进另一张图时 viewer.reset 清空它，这里按新图重设（显示节点的包与旧图相同也要设）
  const loads = useViewer((s) => s.loads);
  const owned = useDescribed<Manifest>("manifest", own);

  useEffect(() => {
    // 切换到没有帧的节点（只输出数值或尚未计算）：清空帧列表，不沿用上一个节点的帧，否则标尺上仍显示旧节点的帧号，
    // 播放头会在其他节点的时间线上移动。包说明还没到时不动（到了、或重算后换了代次再到时设）
    if (!follows) {
      if (useViewer.getState().frames.length) useViewer.getState().setFrames([], useViewer.getState().frame);
      return;
    }
    if (followed) useViewer.getState().setFrames(followed.meta.frames ?? [], useViewer.getState().frame);
  }, [follows, followed, loads]);

  const mine = useMemo(() => joinLayers(owned.flatMap((m) => (m ? markLayers(m.meta) : []))), [owned]);
  // 当前火柴人手柄已绘制的帧（参数中每一条目都记录其所在帧，nodes/handles.py parse_figures）。
  // 仅处理火柴人：其他二维手柄（点、框、轮廓）在标尺上标记帧没有意义，手绘遮罩对整段镜头生效，
  // 点与框是提供给解算器的提示，不表示该帧已绘制完成
  const figure = plan?.handles.find((h) => h.stage === "2d" && h.kind === "figure") ?? null;
  const posed = figure ? ((plan!.node.data.params[Object.values(figure.params)[0]] as string[] | undefined) ?? []) : [];
  // 以 | 连接而非逗号：条目本身包含大量逗号（「1001:300,300,284,…」），
  // 按逗号拆分会将每个坐标都视为帧号（300 也是整数），标尺上会多出数十个错误标记
  const posedKey = posed.join("|");
  const drawn = useMemo((): MarkLayer[] => {
    const frames = [...new Set(posedKey ? posedKey.split("|").map((s) => Number(s.split(":")[0])) : [])]
      .filter((f) => Number.isInteger(f)).sort((a, b) => a - b);
    return frames.length ? [{ name: "关键姿势", frames }] : [];
  }, [posedKey]);
  return { layers: drawn.length ? joinLayers([...mine, ...drawn]) : mine };
}
