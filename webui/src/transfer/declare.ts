import { api } from "../api";
import { prepareLocalProxies } from "./localProxy";
import { useCookInputs } from "../state/cookInputs";
import type { Item } from "../api/sequences";
import { clientInfo } from "../platform/client";
import { useUploads, type UploadTask } from "../state/uploads";
import { rememberLocal } from "./local";
import { cancelUpload, keepTask, patchTask, sendWith, sentHere } from "./uploads";
import { randomId } from "../platform/randomId";

/** 选择文件后的处理流程：此时不传输任何字节。选择图片时不上传、不显示计算中，
 * 直到下游确实需要且用户点击「计算」时才上传。
 *
 * 共三步：
 * 1. 在本机计算每个文件的内容指纹（sha256）（`reading`）。耗时主要来自读取本机磁盘，
 *    不经过网络传输任何字节；9 GB 的序列需读取数十秒，因此须显示进度。
 * 2. 申报（`POST /api/uploads/declare`）：发送清单及第一个文件的前数十 KB。
 *    服务器使用其 `describe_file` 读取图层，并返回最终的引用；
 *    参数随即写入，节点上的端口随之出现，用户即可连线。若头部数据不足，服务器会返回所需的字节数，此处据此补发。
 * 3. 停留在 `picked`：字节仍在用户机器上，点击「计算」时才上传（`graph/apply.ts sendPicked`）。
 *
 * 本模块不自行计算该 id：服务器有同一公式（`lab2shot/transfer/uploads.py set_id`），
 * 且以服务器结果为准。网页预先推算没有收益，推算错误还需修正参数；一次申报往返仅数百字节。
 *
 * 仅适用于服务器只凭文件头即可给出结果的节点（服务器端声明
 * `lab2shot/nodes/base.py ReadsFile.head_is_enough`，在网页端对应文件参数上的 `head_enough`）。
 * 其余节点一律在选择后立即上传：
 * 「导入 USD / FBX / Alembic / BVH / PLY」需要完整文件才能列出层级，「读取多条序列」需要该文件夹
 * 实际存在于磁盘上，「读取视频」需要解码器打开完整文件以统计帧数。场景文件须上传后服务器才能读取（浏览器不解析
 * abc / usd / fbx）。若不作区分，选择 `.usd` 后点击「选择…」将无法挑选层级，且会报出「上传的文件已经不在
 * 服务器上了」这一错误提示，而该文件实际从未上传。
 *
 * 超过 HASH_MAX 的大文件同样在选择后立即上传：`crypto.subtle.digest` 需将整个文件读入内存，数 GB 的视频会导致
 * 浏览器崩溃。此时由服务器边接收边计算 sha，功能不受影响，只是无法省去这次上传。 */
const HASH_MAX = 512 << 20; // 超过此大小的文件不在浏览器中计算指纹（需将整个文件读入内存）
const HEAD_FIRST = 256 << 10; // 申报时首次发送的头部字节数（仅发送一次，取值偏大）
const HASH_AT_ONCE = 4; // 同时读取的文件数

const hex = (b: ArrayBuffer) => Array.from(new Uint8Array(b), (n) => n.toString(16).padStart(2, "0")).join("");

async function sha256(f: File): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", await f.arrayBuffer()));
}

/** The page's action once an upload is declared (graph/apply.ts: writes the reference into the node's parameter). */
let declaredTo: ((t: UploadTask, ref: string) => void) | null = null;
export const onDeclared = (f: typeof declaredTo) => void (declaredTo = f);

export function startUpload(item: Item, folder: string, target: { graph: string; node: string; param: string },
                            headEnough: boolean): void {
  const origin = [folder, item.name].filter(Boolean).join("/");
  // any upload in progress for this parameter is dropped (its parts stay on the server and resume if picked again)
  for (const t of Object.values(useUploads.getState().tasks)) if (t.node === target.node && t.param === target.param) cancelUpload(t.key, false);
  const key = randomId(8);
  // 以下两种情况在选择后立即上传：① 服务器必须获得完整文件才能处理该节点（`headEnough` 为假，
  // 见上方说明）；② 文件过大，浏览器无法计算内容指纹（HASH_MAX）
  const big = !headEnough || item.files.some((f) => f.size > HASH_MAX) || typeof crypto.subtle === "undefined";
  const task: UploadTask = {
    key, ...target, name: item.name, folder, origin, sequence: item.kind === "sequence", frames: item.frames,
    files: item.files.map((f) => ({ name: f.name, size: f.size, modified: f.lastModified })), bytes: item.size,
    state: big ? "sending" : "reading", sent: 0, done: 0, rate: 0, error: "", retryAt: 0,
  };
  rememberLocal(key, item); // before the task appears: the viewer draws the picked files immediately
  useUploads.setState((s) => ({ tasks: { ...s.tasks, [key]: task } }));
  void keepTask(task);
  if (big) void sendWith(task, item.files); // 文件过大：沿用选择后立即上传的流程（原因见上方 HASH_MAX 的说明）
  else void declare(task, item.files);
}

/** 第 1、2 步：本机计算指纹 → 申报 → 参数写入引用 → 停留在 `picked`。 */
async function declare(task: UploadTask, files: File[]): Promise<void> {
  const key = task.key;
  sentHere.set(key, { files, stop: () => undefined }); // 点击「计算」时仍需使用这些文件（`sendUpload`）
  try {
    const shas: Record<string, string> = {};
    let at = 0;
    const worker = async () => {
      for (let i = at++; i < files.length && useUploads.getState().tasks[key]; i = at++) {
        shas[files[i].name] = await sha256(files[i]);
        patchTask(key, { done: Object.keys(shas).length });
      }
    };
    await Promise.all(Array.from({ length: Math.min(HASH_AT_ONCE, files.length) }, worker));
    if (!useUploads.getState().tasks[key]) return; // 用户中途清除了该参数
    // 指纹保留在任务上：点击「计算」时，通道级上传据此向服务器查询每帧缺少的通道（`transfer/planes.ts`）
    patchTask(key, { files: task.files.map((f) => ({ ...f, sha: shas[f.name] })) });
    const sizes = Object.fromEntries(files.map((f) => [f.name, f.size]));
    const body = { name: task.name, files: shas, sizes, origin: { path: task.origin, client: clientInfo() } };
    // 头部数据不足时服务器会返回所需字节数，此处据此补发；不得静默返回空图层
    let head = HEAD_FIRST;
    let said = await api.uploads.declare({ ...body, head_of: files[0].name, head: await b64(files[0].slice(0, head)), head_whole: files[0].size });
    while (said.need && said.need > head) {
      head = said.need;
      said = await api.uploads.declare({ ...body, head_of: files[0].name, head: await b64(files[0].slice(0, head)), head_whole: files[0].size });
    }
    if (!useUploads.getState().tasks[key]) return;
    // 本机代理自此在后台生成，覆盖所有通道：
    // 图层取服务器刚从文件头读出的结果；色彩空间取节点自身的参数
    const layers = Object.entries(((said.layers as { layers?: Record<string, { channels?: string[] }> } | undefined)?.layers) ?? {})
      .map(([name, l]) => ({ name, channels: l.channels ?? [] }));
    const space = String(useCookInputs.getState().nodes[task.node]?.params.colorspace ?? "");
    prepareLocalProxies(`${task.node}|${task.param}`, files, space, layers);
    patchTask(key, { state: "picked" });
    const now = useUploads.getState().tasks[key];
    if (now) {
      void keepTask(now);
      declaredTo?.(now, said.ref); // 参数写入引用：节点上的端口此时才出现
    }
  } catch (e) {
    patchTask(key, { state: "failed", error: (e as Error).message });
  }
}

const b64 = async (blob: Blob): Promise<string> => {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let s = "";
  for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode(...bytes.subarray(i, i + 32768));
  return btoa(s);
};

/** 第 3 步：点击「计算」时上传字节（`graph/apply.ts sendPicked`）。上传完成后同样组装为一份输入
 * （`POST /api/uploads`，与申报得到的引用相同），因此下游无需任何修改。
 * 返回 false 表示上传失败（网络故障、服务器拒绝，或刷新后已无法访问文件）。 */
export async function sendUpload(key: string): Promise<boolean> {
  const t = useUploads.getState().tasks[key];
  const here = sentHere.get(key);
  if (!t || !here) return false;
  patchTask(key, { state: "sending", error: "" });
  await sendWith(useUploads.getState().tasks[key], here.files);
  return !useUploads.getState().tasks[key]; // 上传完成时任务即被删除：任务仍存在表示未成功
}

