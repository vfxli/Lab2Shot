import { useRef } from "react";
import type { Item } from "../api/sequences";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { upstream } from "../graph/nodes";
import { drawable, pickedItem, uploadedItem } from "../transfer/local";
import { taskFor, uploadLine, useUploads } from "../transfer/uploads";
import { pictureKey } from "./origin";

/** 使用者在当前标签页中选择的本机文件：由哪个节点读取、各帧对应哪个文件、处于上传中还是服务器读取中。
 *
 * 绘制不在本模块进行：二维舞台（view/Stage2D.tsx）将其作为自身的一个帧源，始终只有一个舞台。
 * 显示不等待上传完成，也不另设「本机画面」组件，因为在两个组件之间切换必然出现一帧空白。 */

const FILE_IN = ["file", "sequence"];

export interface LocalPicture {
  // 读取该文件的节点。`view/origin.ts` 据此区分「本输出口自身的画面」与「上游提供的衬底」；
  // 查找沿上游进行，因此下游运算节点也会得到上游读入的文件。
  node: string;
  item: Item;
  space: string; // EXR 解码所用的色彩空间（节点的 colorspace 参数）
}

/** 当前显示节点（或其上游节点）的文件正在上传、或因页面重载而中断上传时，视图显示的提示文字；
 * 仅用于无内容可绘制（文件不在本标签页中）的情况。 */
export function useUploadHint(nodeId: string | null): string | null {
  const tasks = useUploads((s) => s.tasks);
  const edges = useCookInputs((s) => s.edges);
  if (!nodeId) return null;
  const ids = new Set(upstream(nodeId, edges));
  const t = Object.values(tasks).find((x) => ids.has(x.node));
  return t ? uploadLine(t) : null;
}

/** 当前显示节点尚无结果时，视图可从使用者本机文件中绘制的画面：该节点（或其上游节点）读取的图像，
 * 可能仍在上传，或已上传但服务器尚未读取。
 *
 * 同一份文件必须返回同一对象（见 `view/origin.ts pictureKey`）：该查询在每次渲染时都会遍历上游，
 * 结果相同时不得更换对象。 */
export function useLocalPicture(nodeId: string | null): LocalPicture | null {
  const tasks = useUploads((s) => s.tasks);
  const nodes = useCookInputs((s) => s.nodes);
  const edges = useCookInputs((s) => s.edges);
  const last = useRef<LocalPicture | null>(null);
  const next = nodeId ? lookFor(nodeId, tasks, nodes, edges) : null;
  if (!next || !last.current || pictureKey(next) !== pictureKey(last.current)) last.current = next;
  return last.current;
}

type Tasks = ReturnType<typeof useUploads.getState>["tasks"];
type Nodes = ReturnType<typeof useCookInputs.getState>["nodes"];
type Edges = ReturnType<typeof useCookInputs.getState>["edges"];

function lookFor(nodeId: string, tasks: Tasks, nodes: Nodes, edges: Edges): LocalPicture | null {
  const defs = getNodeDefs();
  for (const id of upstream(nodeId, edges)) {
    const node = nodes[id];
    const def = node && defs[node.typeId];
    if (!node || !def) continue;
    const space = String(node.params.colorspace ?? "");
    for (const p of def.params) {
      if (!FILE_IN.includes(p.widget ?? "")) continue;
      const task = taskFor(tasks, id, p.name);
      const picked = task && pickedItem(task.key);
      if (task && picked && picked.files.some((f) => drawable(f.name))) return { node: id, item: picked, space };
      const value = String(node.params[p.name] ?? "");
      const done = uploadedItem(id, p.name, value);  // 参数尚未写入时也返回（见 transfer/local.ts 中的说明）
      if (done && done.files.some((f) => drawable(f.name))) return { node: id, item: done, space };
    }
  }
  return null;
}
