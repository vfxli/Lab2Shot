import type { NodeTypeDef } from "../api";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { originalsFor } from "./originals";
import type { Item } from "../api/sequences";

/** Files the user picked in this tab, which the page can draw itself instead of fetching them: a plate while it is
 * uploading (本机画面), and afterwards every file whose content the server has confirmed (by sha256). A frame the server
 * shows unchanged (a display-referred PNG plate) is therefore drawn from the user's own copy, byte-identical, without
 * downloading it. Only this tab holds them: a reload discards them (the browser grants files to a page only when the
 * user picks them). */

const picked = new Map<string, Item>(); // upload task -> the picked item
const bySha = new Map<string, File>(); // content -> the user's file with that content
const uploaded = new Map<string, { ref: string; item: Item }>(); // node|param -> the pick that became `ref`

/** Stores what a task picked (`item`), or one of its files that the server now holds (`sha`, `file`). */
export function rememberLocal(key: string, item?: Item, sha?: string, file?: File): void {
  if (item) picked.set(key, item);
  if (sha && file) bySha.set(sha, file);
}

/** A task's pick became the upload `ref` of a node's parameter: its frames remain drawable from here until the server's
 * reading is shown. */
export function uploadedLocal(key: string, node: string, param: string, ref: string): void {
  const item = picked.get(key);
  picked.delete(key);
  if (item) uploaded.set(`${node}|${param}`, { ref, item });
}

export const pickedItem = (key: string) => picked.get(key);

/** 该节点的该文件参数所对应的本机文件。
 *
 * `ref` 为空时也返回结果（显示不得出现空白帧）：上传刚完成时，「正在上传的那一份」已被清除，而节点参数尚未写入
 * 新地址，两者之间隔一次渲染；若此时两边都取不到本机文件，画布会空白一帧。
 * 参数写入后按地址比对：不一致说明用户又选择了其他文件，此时才不返回。 */
export const uploadedItem = (node: string, param: string, ref: string) => {
  const u = uploaded.get(`${node}|${param}`);
  return u && (!ref || u.ref === ref) ? u.item : undefined;
};

/** The user's own file with this content, if this tab holds it. */
export const localFile = (sha: string | undefined) => (sha ? bySha.get(sha) : undefined);

const PICTURES = [".png", ".jpg", ".jpeg", ".exr"];

/** 判断浏览器能否解码该文件（全项目唯一的判据，`view/origin.ts` 的整套逻辑均依赖它）：
 * 图像可以解码；USD / FBX / Alembic / BVH / PLY 无法解码，仍交由服务器计算。 */
export const drawable = (name: string) => PICTURES.some((s) => name.toLowerCase().endsWith(s));

/** 序列的文件名模式（文件参数中存储为 `upload:<id>/plate.####.exr`）。 */
const patternOf = (ref: string) => ref.slice(ref.indexOf("/") + 1);

const fileParams = (def: NodeTypeDef) => def.params.filter((p) => p.widget === "file" || p.widget === "sequence");

/** 判断浏览器是否已持有该节点待显示画面的原件（`graph/actions.ts` 的「显示时自行计算」据此决定是否计算）。
 *
 * 这是「来源」判定的第二个入口：完整定义位于 `view/origin.ts` 开头，那里回答的是
 * 「当前画面该端口所绘数据的来源」（有 `ViewItem` 可查）；此处在尚无 ViewItem 时回答同一问题，
 * 即「该节点是否需要服务器计算」。判据相同（`drawable`），不得维护两份。
 * 它位于本层是由于网页分层：`graph` 位于 `view` 之下，不得 import `view/`（`webui/tests/layers.test.ts`）。
 *
 * 读取类节点和输出节点的文件位于用户本机，双击查看时应显示无损原图，而非先在服务器上计算一遍。
 *
 * 判据是「确实持有」，而非「是否为读取节点」：无法获取时照常计算，不会留下空白画面。
 * 两个来源，均可同步得知：
 * 1. 本标签页中选择并已与地址对应的文件（`transfer/local.ts uploadedItem`，刷新后失效）；
 * 2. 已登记的本机原件（`transfer/originals.ts`，来自已授权的目录）。
 *
 * 浏览器能否解码只有一个判据：`drawable`（按文件名）。不能改为「输出端口全属 `image` 系列」：
 * 那是在推测端口类型，而有文件参数的节点中约一半是三维格式（USD / BVH / PLY / Alembic / FBX），
 * 浏览器无法解码，跳过计算后视图中只会显示「还没有结果」。
 *
 * 正在上传的情况无需在此处理：参数须待上传完成后才写入地址，无地址则无指纹，
 * 无指纹的节点本来就不会进入「显示时自行计算」的列表（`graph/actions.ts` 的 `pending`）。
 *
 * 有意不计入正在查找的情况（`transfer/originals.ts` 的 `stillLooking`）：查找原件是异步的，
 * 若将「查找中」视为「已持有」，查找结束发现不存在时将不再有人发起计算，屏幕会一直空白且无任何提示。 */
export function hasLocalFile(nodeId: string, fp: string | null | undefined): boolean {
  const node = useCookInputs.getState().nodes[nodeId];
  const def = node && getNodeDefs()[node.typeId];
  if (!node || !def) return false;
  const reads = fileParams(def);
  if (!reads.length) return false; // 不读取文件的节点不适用此规则：照常计算
  for (const p of reads) {
    const ref = String(node.params[p.name] ?? "");
    // 本标签页中选择的文件：文件名已知，据此判断浏览器能否解码
    const item = uploadedItem(nodeId, p.name, ref);
    if (item?.files.some((f) => drawable(f.name))) return true;
  }
  // 已登记的原件：没有文件对象，按参数中文件名模式的后缀判断（`upload:<id>/plate.####.exr`）
  return !!fp && reads.some((p) => drawable(patternOf(String(node.params[p.name] ?? "")))) && !!originalsFor(fp, null);
}
