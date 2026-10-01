/** 三维视图数据块在使用者硬盘上的副本：整段缓存（view/scene.ts）取回的每一块都写到这里，刷新页面、重开浏览器后
 * 再看同一份结果只从这里读、重新解码，不再下载。
 *
 * 不另建第二套磁盘缓存：直接存进本机代理的同一个存储（transfer/localProxy/store.ts，浏览器私有文件系统 OPFS），
 * 共用它的索引、「本机缓存上限」（管理员设置 view.local_cache_gb）与「最久没看的整份先淘汰」的规则。一份三维视图
 * （同一个数据包、同一代次；点云预览另加相机）是一「份」：`s_<视图的短散列>/view/<块地址的短散列>.bin`。
 *
 * 只存不可变的地址：地址里带内容散列（基础块 `h=`）或数据包代次（分块 `g=`）的，一个地址永远是同一串字节
 * （服务器 view_data.py respond 的 immutable）。边算边看的块（地址里没有代次）不存。
 * 浏览器不支持私有文件系统时（`proxyStoreAvailable` 为假）一切照旧：只在内存里、按需从服务器取。 */

import { shortHash } from "../platform/digest";
import { partBytes, type Lane } from "../transfer/frameStore";
import { serverNow } from "../state/server";
import { proxiesMade, proxyStoreAvailable, readProxy, writeProxy } from "../transfer/localProxy/store";

const TIER = "view";
const DEFAULT_CAP_GB = 10; // 与 transfer/localProxy/index.ts 相同：服务器第一次回答之前的默认值
const capBytes = (): number => (serverNow()?.view?.local_cache_gb ?? DEFAULT_CAP_GB) * (1 << 30);

/** 该地址的字节是否永远不变（可以存盘）：带非空的 `h=`（内容散列）或 `g=`（数据包代次）。 */
export function durable(url: string): boolean {
  return proxyStoreAvailable() && /[?&](h|g)=[^&]+/.test(url);
}

/** 一份视图的键：地址去掉最后一段（部分名）后的路径 + 代次与相机（同一视图的各块同属一份，整份一起淘汰）。 */
function viewKey(url: string): string {
  const [path, query = ""] = url.split("?");
  const q = new URLSearchParams(query);
  return `s_${shortHash(`${path.slice(0, path.lastIndexOf("/"))}|${q.get("g") ?? q.get("h") ?? ""}|${q.get("camera") ?? ""}`)}`;
}

const nameOf = (url: string): string => `${shortHash(url)}.bin`;

/** 已存盘的字节；没有时 null。 */
export async function readStored(url: string): Promise<Uint8Array | null> {
  if (!durable(url)) return null;
  const blob = await readProxy(viewKey(url), TIER, nameOf(url));
  return blob ? new Uint8Array(await blob.arrayBuffer()) : null;
}

/** 存盘（已在盘上的不重写）；成功或本来就在时 true。`bytes` 在返回前被读完，调用方之后可以转交它。 */
export async function store(url: string, bytes: Uint8Array): Promise<boolean> {
  if (!durable(url)) return false;
  const key = viewKey(url);
  if ((await proxiesMade(key, TIER)).includes(nameOf(url))) return true;
  return writeProxy(key, TIER, nameOf(url), bytes, capBytes());
}

/** 这些地址里已存盘的（页面打开一份视图时问一次：已在盘上的块不再下载，时间线直接标为已缓存）。 */
export async function storedOf(urls: string[]): Promise<Set<string>> {
  const out = new Set<string>();
  const byView = new Map<string, string[]>();
  for (const u of urls) if (durable(u)) byView.set(viewKey(u), [...(byView.get(viewKey(u)) ?? []), u]);
  for (const [key, us] of byView) {
    const made = new Set(await proxiesMade(key, TIER));
    for (const u of us) if (made.has(nameOf(u))) out.add(u);
  }
  return out;
}

/** 一个三维块：经取字节的唯一入口（transfer/frameStore.ts partBytes），先读本机硬盘上的整段缓存。 */
export const fetchPart = (url: string, lane: Lane): Promise<Uint8Array> => partBytes(url, { durable, read: readStored }, lane);
