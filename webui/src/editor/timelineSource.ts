import { useEffect, useMemo, useState } from "react";
import { useViewer } from "../state/viewer";
import { joinLayers, markLayers, type MarkLayer } from "../model/timelineMath";
import { useDisplayPlan } from "../view/plan";
import { manifestOf } from "../transfer/frames";

/** What the timeline follows for the displayed node: the frames of its first result (or of the plate it works on),
 * the marks its own results carry (their "keys" and "marks") and, for a node being
 * drawn on right now, the frames its stick-figure handle already has a pose on.
 *
 * 最后一项用于在绘制一段后切换帧检查连续性。它读取的是参数而非计算结果，因此点击「添加帧」后
 * 标尺上立即出现标记，无须先计算（计算后结果自身带的标记与之合并，两者不冲突）。使用的是标尺上
 * 已有的标记机制（`tl-mark`，与 `markLayers` 同一路径），不增加新控件。 */
export function useSource(): { layers: MarkLayer[] } {
  const plan = useDisplayPlan();
  const follows = plan?.frames ?? null;
  const own = plan ? plan.items.filter((it) => it.fp && !it.context).map((it) => it.fp!) : [];
  const key = own.join("|");
  const [got, setGot] = useState<{ key: string; layers: MarkLayer[] }>({ key: "", layers: [] });

  useEffect(() => {
    // 切换到没有帧的节点（只输出数值或尚未计算）：清空帧列表，不沿用上一个节点的帧，否则标尺上仍显示旧节点的帧号，
    // 播放头会在其他节点的时间线上移动
    if (!follows) {
      if (useViewer.getState().frames.length) useViewer.getState().setFrames([], useViewer.getState().frame);
      return;
    }
    let alive = true;
    void manifestOf(follows).then((m) => alive && useViewer.getState().setFrames(m.meta.frames ?? [], useViewer.getState().frame), () => undefined);
    return () => {
      alive = false;
    };
  }, [follows]);

  useEffect(() => {
    let alive = true;
    void Promise.all(key ? key.split("|").map((fp) => manifestOf(fp).catch(() => null)) : []).then((all) => {
      const metas = all.flatMap((m) => (m ? [m.meta] : []));
      if (alive) setGot({ key, layers: joinLayers(metas.flatMap(markLayers)) });
    });
    return () => {
      alive = false;
    };
  }, [key]);
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
  const mine = got.key === key ? got.layers : [];
  return { layers: drawn.length ? joinLayers([...mine, ...drawn]) : mine };
}
