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

/** 判断浏览器能否解码该文件（全项目唯一的判据，`view/localPick.ts` 据此决定本机文件能否绘制）：
 * 图像可以解码；USD / FBX / Alembic / BVH / PLY 无法解码，仍交由服务器计算。 */
export const drawable = (name: string) => PICTURES.some((s) => name.toLowerCase().endsWith(s));
