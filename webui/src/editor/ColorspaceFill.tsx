import { useEffect } from "react";
import type { ParamDef } from "../api";
import { setDerivedParam } from "../graph/edit";
import { useGraphDoc } from "../graph/snapshot";
import { useChoices } from "../ui/choices";

/** 「色彩空间」参数自动填写的唯一所在：该参数在选定文件时即确定，而非在打开节点面板时确定。
 *
 * 该参数没有「自动」档：为空时按文件格式填写服务器 choices 给出的 `default`（EXR 为 ACEScg，PNG / JPG 为 sRGB 等）。
 * 该步骤不能放在面板控件中（只有节点被选中、控件被绘制时才会执行）：否则先提交（参数仍为空）
 * 再点击读取节点（参数被填写）时，节点图会发生变化，指纹随之改变，上游全部重算；取值不得依赖使用者是否打开过节点。
 *
 * 因此图中每个带「色彩空间」参数、已选定文件而参数为空的节点，都在此处（挂载于节点图根部，与选中状态无关）
 * 查询一次 choices，取得 default 后写入。服务器端的指纹同样按生效值计算（nodes/base.py fingerprint_params），
 * 两端均不依赖执行时序。 */
export function ColorspaceFills() {
  const snap = useGraphDoc();
  const need: { nodeId: string; p: ParamDef }[] = [];
  for (const n of snap.nodes) {
    const def = snap.nodeDefs[n.data.typeId];
    if (!def) continue;
    for (const p of def.params) {
      if (p.widget !== "colorspace") continue;
      const v = n.data.params[p.name];
      // 尚未选定文件时（choices 会提示「先选择文件」）不查询；已选定且参数为空时才查询
      const picked = p.choices_from.length ? p.choices_from.some((from) => !!n.data.params[from]) : true;
      if ((v == null || v === "") && picked) need.push({ nodeId: n.id, p });
    }
  }
  return (
    <>
      {need.map(({ nodeId, p }) => (
        <Fill key={`${nodeId}:${p.name}`} nodeId={nodeId} p={p} />
      ))}
    </>
  );
}

function Fill({ nodeId, p }: { nodeId: string; p: ParamDef }) {
  const choice = useChoices(nodeId, p);
  const fill = choice?.default;
  useEffect(() => {
    // 这不是使用者的操作：若经由 setParam，撤销后参数变空并被此处再次填写，撤销永远无效，
    // 重做栈被清空，打开色彩空间为空的旧图时也会立即被标记为已修改。setDerivedParam 写入节点但不进入撤销历史、不标记为已修改
    if (fill) setDerivedParam(nodeId, p.name, fill);
  }, [fill]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}
