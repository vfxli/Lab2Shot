import type { Item } from "../api/sequences";

/** 使用者在本标签页里选的文件：页面可以自己画它们，不必去取——上传期间的素材（本机画面），以及之后服务器已按 sha256
 * 确认过内容的每一个文件。服务器原样显示的帧（显示空间的 PNG 素材）因此直接从使用者自己的那份画，逐字节相同，不必下载。
 * 只有本标签页持有它们：刷新即丢（浏览器只在使用者亲自选择时把文件交给页面）。 */

const picked = new Map<string, Item>(); // 上传任务 -> 所选的素材
const bySha = new Map<string, File>(); // 内容 -> 使用者手上这份内容的文件（有上限：BY_SHA_MAX）
const BY_SHA_MAX = 20_000;
// graphId|节点|参数 -> 成为 `ref` 的那次选择。`fresh`：参数还没显示过 `ref`（见 uploadedItem）
const uploaded = new Map<string, { ref: string; item: Item; fresh: boolean }>();
const slot = (graph: string, node: string, param: string) => `${graph}|${node}|${param}`;

/** 记下一个任务所选的素材（`item`），或其中服务器已持有的一个文件（`sha`、`file`）。 */
export function rememberLocal(key: string, item?: Item, sha?: string, file?: File): void {
  if (item) picked.set(key, item);
  if (sha && file) {
    bySha.delete(sha);
    bySha.set(sha, file);
    // 有上限（最早的先丢）：它持有的是 File 句柄，随使用时长只增不减；丢掉的只是「本机也有这份」的捷径，照常从服务器取
    if (bySha.size > BY_SHA_MAX) bySha.delete(bySha.keys().next().value as string);
  }
}

/** 一个任务所选的素材成了文档 `graph` 里某节点参数的上传引用 `ref`：在服务器的读取结果显示出来之前，它的帧仍从这里画。 */
export function uploadedLocal(key: string, graph: string, node: string, param: string, ref: string): void {
  const item = picked.get(key);
  picked.delete(key);
  if (item) uploaded.set(slot(graph, node, param), { ref, item, fresh: true });
}

/** 打开另一份文档（`transfer/uploads.ts leaveGraph`）：留作显示的本机文件只保留属于新文档 `graph` 的。 */
export function forgetLocalOf(graph: string): void {
  for (const k of [...uploaded.keys()]) if (!k.startsWith(`${graph}|`)) uploaded.delete(k);
}

/** 「另存为」换了 graphId（`transfer/uploads.ts rehomeUploads`）：本机文件跟着文档走。 */
export function rehomeLocal(from: string, to: string): void {
  for (const [k, u] of [...uploaded.entries()])
    if (k.startsWith(`${from}|`)) (uploaded.delete(k), uploaded.set(`${to}${k.slice(from.length)}`, u));
}

export const pickedItem = (key: string) => picked.get(key);

/** 文档 `graph` 中该节点的该文件参数所对应的本机文件。只认本文档（graphId）：另一份文档里同 id 的节点
 * （模板的读取节点都叫 `read`）拿不到它。
 *
 * `ref` 为空时，只对本文档刚完成、参数尚未写入的那一份返回（显示不得出现空白帧）：上传刚完成时，「正在上传的那一份」
 * 已被清除，而节点参数尚未写入新地址，两者之间隔一次渲染；若此时两边都取不到本机文件，画布会空白一帧。
 * 参数显示过这个地址之后（`fresh` 为假）再变空，是用户清除了参数，不再返回。
 * 参数写入后按地址比对：不一致说明用户又选择了其他文件，此时才不返回。 */
export const uploadedItem = (graph: string, node: string, param: string, ref: string) => {
  const u = uploaded.get(slot(graph, node, param));
  if (!u) return undefined;
  if (!ref) return u.fresh ? u.item : undefined;
  return u.ref === ref ? u.item : undefined;
};

/** 参数已经写上了这个地址（graph/apply.ts 上传完成、写参数之后）：此后参数再变空是使用者清除了它，不再画这份本机文件。
 * 在写参数的事件里标，不在渲染时改状态。 */
export function uploadedShown(graph: string, node: string, param: string): void {
  const u = uploaded.get(slot(graph, node, param));
  if (u) u.fresh = false;
}

/** 一个上传任务不要了（取消、完成、被丢弃）：它选的文件不再留着。 */
export function forgetPicked(key: string): void {
  picked.delete(key);
}

/** 使用者手上这份内容的文件（本标签页持有时）。 */
export const localFile = (sha: string | undefined) => (sha ? bySha.get(sha) : undefined);

const PICTURES = [".png", ".jpg", ".jpeg", ".exr"];

/** 判断浏览器能否解码该文件（全项目唯一的判据，`view/localPick.ts` 据此决定本机文件能否绘制）：
 * 图像可以解码；USD / FBX / Alembic / BVH / PLY 无法解码，仍交由服务器计算。 */
export const drawable = (name: string) => PICTURES.some((s) => name.toLowerCase().endsWith(s));
