import { api } from "../api";
import { request } from "../platform/http";
import { download } from "../platform/util";
import { fileIn } from "./handles";
import { addressOf, type Delivery } from "../api/deliveries";

/** File transfer between the user's machine and the server. Picked files are uploaded (transfer/uploads.ts: chunked,
 * each file once, resumable after a dropped connection, with files already sent by the account skipped); 「输出」
 * deliveries are written into the folder chosen by the user or downloaded. Uploads and deliveries are scoped to the
 * owning account (server/access.py). */


// A delivery is accessible only to the account that submitted the cook and to the administrator.
const deliveryUrl = (d: Delivery) => `/api/deliveries/${d.run}/${addressOf(d)}`;

/** Writes a delivery to the location chosen by the user: the archive is streamed into the file (tar, tar.gz), or its
 * files are written into the folder with sub-folders preserved (文件夹). An interrupted write leaves the target
 * unchanged (the browser writes to a temporary file and swaps it in on close), so a partial write is never treated as
 * saved. `onBytes` receives the number of bytes written so far. `into` is a sub-folder of `to`: an 「输出」 inside a
 * 逐项处理 block delivers one package per item, all into the same chosen folder, each under its own package name. */
export async function saveDelivery(d: Delivery, to: FileSystemHandle, onBytes?: (n: number) => void, into = ""): Promise<void> {
  let done = 0;
  const counting = () =>
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, c) {
        done += chunk.byteLength;
        onBytes?.(done);
        c.enqueue(chunk);
      },
    });
  if (d.mode !== "folder") {
    const r = await request(`${deliveryUrl(d)}/archive`);
    await r.body!.pipeThrough(counting()).pipeTo(await (to as FileSystemFileHandle).createWritable());
    return;
  }
  // Folder delivery: the whole archive is fetched in one request and unpacked client-side rather than requested file
  // by file, since a 150-frame EXR sequence would otherwise cost 150 round trips. The unpacker is loaded on demand
  // (`import(...)`) and is only needed when writing a delivery into a folder (initial bundle budget:
  // tests/test_transfer.py).
  const { untarInto } = await import("./untar");
  const r = await request(`${deliveryUrl(d)}/archive`);
  const folder = to as FileSystemDirectoryHandle;
  await untarInto(r.body!, {
    open: async (name) => fileIn(folder, into ? `${into}/${name}` : name),
  }, onBytes);  // untarInto reports bytes written, the same quantity as `counting` above
}

/** Downloads a delivery through the browser as its archive (a folder delivery as a tar). The server marks it as
 * downloaded once the last byte has been sent. */
export function downloadDelivery(d: Delivery): void {
  download(`${deliveryUrl(d)}/archive?download=1`, d.mode === "folder" ? `${d.name}.tar` : d.name);
}

/** An 「输出」 inside a 逐项处理 block delivers N packages. To avoid N browser download prompts, the server combines
 * them into one archive with one sub-folder per item. Used by 「下载」 in the queue. */
export function downloadBatch(items: Delivery[]): void {
  const first = items[0];
  if (!first) return;
  // The combined archive name is set by the server (transfer/deliveries.py delivered_name). The download attribute is
  // left empty so the Content-Disposition name applies and the name is defined in one place.
  download(api.deliveries.batch(first.run, first.node), "");
}
