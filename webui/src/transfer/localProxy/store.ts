/** 本机代理的存储位置与淘汰策略：存放在浏览器的私有文件系统（OPFS，位于使用者本地磁盘），刷新页面或重启浏览器后仍保留。
 *
 * 布局：`lab2shot-proxies/<文件键>/<档位-显示变换>/<层或通道>.webp|.l2c1.gz`，另有一份 `index.json`
 * 记录每个 `<文件键>/<档位-显示变换>` 占用的字节数与最近使用时间。淘汰按最久未使用优先，
 * 上限为管理员设定的「本机缓存上限」。
 * 该存储位于磁盘，与内存的两项预算（`transfer/cache.ts`）无关。 */

const ROOT = "lab2shot-proxies";
const INDEX = "index.json";

interface Entry { bytes: number; used: number; names: string[] }
type Index = Record<string, Entry>; // "<fileKey>/<tierKey>" -> entry

const own: { root: FileSystemDirectoryHandle | null; index: Index | null; dirty: boolean; flushing: Promise<void> | null;
             unsupported: boolean } = { root: null, index: null, dirty: false, flushing: null, unsupported: false };

/** 判断浏览器是否支持私有文件系统（Chromium 系支持；不支持时不生成本机代理，每帧实时解码）。 */
export const proxyStoreAvailable = (): boolean =>
  !own.unsupported && typeof navigator !== "undefined" && !!navigator.storage?.getDirectory;

async function root(): Promise<FileSystemDirectoryHandle | null> {
  if (own.root) return own.root;
  if (!proxyStoreAvailable()) return null;
  try {
    const top = await navigator.storage.getDirectory();
    own.root = await top.getDirectoryHandle(ROOT, { create: true });
    return own.root;
  } catch {
    own.unsupported = true;
    return null;
  }
}

async function index(): Promise<Index> {
  if (own.index) return own.index;
  const r = await root();
  let got: Index = {};
  if (r) {
    try {
      const f = await (await r.getFileHandle(INDEX)).getFile();
      got = JSON.parse(await f.text()) as Index;
    } catch { /* 首次使用或索引损坏：从空索引开始，磁盘上的多余目录在下次淘汰时清理 */ }
  }
  own.index = got;
  return got;
}

function flushLater(): void {
  own.dirty = true;
  if (own.flushing) return;
  own.flushing = new Promise((done) => setTimeout(done, 1500)).then(async () => {
    own.flushing = null;
    if (!own.dirty) return;
    own.dirty = false;
    const r = await root();
    if (!r || !own.index) return;
    try {
      const w = await (await r.getFileHandle(INDEX, { create: true })).createWritable();
      await w.write(JSON.stringify(own.index));
      await w.close();
    } catch { /* 写入失败时留待下次重试：索引丢失的代价仅是重新生成代理 */ }
  });
}

const dirOf = async (fileKey: string, tierKey: string, create: boolean): Promise<FileSystemDirectoryHandle | null> => {
  const r = await root();
  if (!r) return null;
  try {
    const a = await r.getDirectoryHandle(fileKey, { create });
    return await a.getDirectoryHandle(tierKey, { create });
  } catch {
    return null;
  }
};

/** 返回该份（文件 × 档位）中已生成的条目（层的显示图、通道的平面），尚未生成时返回 []。 */
export async function proxiesMade(fileKey: string, tierKey: string): Promise<string[]> {
  const e = (await index())[`${fileKey}/${tierKey}`];
  return e ? e.names : [];
}

/** 读取一个已生成的条目（`name` 含后缀，如 `rgba.webp`、`Z.l2c1.gz`）；不存在时返回 null。读取成功即更新使用时间。 */
export async function readProxy(fileKey: string, tierKey: string, name: string): Promise<Blob | null> {
  const d = await dirOf(fileKey, tierKey, false);
  if (!d) return null;
  try {
    const f = await (await d.getFileHandle(name)).getFile();
    const e = (await index())[`${fileKey}/${tierKey}`];
    if (e) { e.used = Date.now(); flushLater(); }
    return f;
  } catch {
    return null;
  }
}

/** 写入一个已生成的条目并登记到索引；超出上限时淘汰最久未使用的份。 */
export async function writeProxy(fileKey: string, tierKey: string, name: string, bytes: Uint8Array, capBytes: number): Promise<boolean> {
  const d = await dirOf(fileKey, tierKey, true);
  if (!d) return false;
  try {
    const w = await (await d.getFileHandle(name, { create: true })).createWritable();
    await w.write(bytes as Uint8Array<ArrayBuffer>);
    await w.close();
  } catch {
    return false;
  }
  const idx = await index();
  const k = `${fileKey}/${tierKey}`;
  const e = idx[k] ?? (idx[k] = { bytes: 0, used: 0, names: [] });
  // 同名条目重写时字节只计一次，否则总量虚高，导致其他份被提前淘汰
  if (!e.names.includes(name)) { e.names.push(name); e.bytes += bytes.byteLength; }
  e.used = Date.now();
  flushLater();
  await trim(capBytes, k);
  return true;
}

/** 淘汰了一份（文件 × 档位）时告诉谁：登记「已就绪」的一方（localProxy/index.ts）据此撤销标记，
 * 否则时间线仍把磁盘上已不存在的代理算作本机可播放的帧。 */
let dropped: (fileKey: string, tierKey: string) => void = () => undefined;
export const onDropped = (f: typeof dropped): void => void (dropped = f);

/** 总量超出上限时，按最久未使用优先整份（文件 × 档位）淘汰，正在写入的份除外。 */
export async function trim(capBytes: number, keep = ""): Promise<void> {
  const idx = await index();
  let total = Object.values(idx).reduce((n, e) => n + e.bytes, 0);
  if (total <= capBytes) return;
  const r = await root();
  if (!r) return;
  const order = Object.entries(idx).filter(([k]) => k !== keep).sort((a, b) => a[1].used - b[1].used);
  for (const [k, e] of order) {
    if (total <= capBytes) break;
    const [fileKey, tierKey] = k.split("/");
    try {
      const a = await r.getDirectoryHandle(fileKey);
      await a.removeEntry(tierKey, { recursive: true });
    } catch { /* 目录已不存在：仍删除索引条目 */ }
    delete idx[k];
    total -= e.bytes;
    dropped(fileKey, tierKey);
  }
  flushLater();
}

/** 磁盘上已生成图片代理的全部 `文件键/档位-显示变换` 条目（页面启动时查询一次，调用方按当前档位筛选，据此视为已在本地）。 */
export async function madeKeys(): Promise<string[]> {
  return Object.entries(await index()).filter(([, e]) => e.names.some((n) => n.endsWith(".webp"))).map(([k]) => k);
}
