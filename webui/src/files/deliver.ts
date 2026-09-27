import { api, type NodeTypeDef } from "../api";
import { run } from "../platform/db";
import { msg, reasonOf, say } from "../state/say";
import { useResults, type NodeDelivery } from "../state/results";
import type { GNode } from "../state/graph";
import { allowWrite, kept, writable, type FileType } from "./handles";
import { addressOf, type Delivery, type DeliveryState } from "../api/deliveries";
import { saveDelivery } from "./transfer";
import { saveProblem } from "./saveProblem";

/** Handling of 「输出」 deliveries. When a cook is submitted, the destination of each 「输出」 node (the chosen file or
 * folder) is stored in this browser under the job (IndexedDB "deliveries"), so the delivery can be saved there
 * regardless of which graph is open, after a reload, from another tab, or after the browser was restarted. A delivery
 * is saved automatically when the browser grants write access without a prompt; otherwise it is added to a single
 * pending list (the page notice) until the user saves it (one click saves all of them through the browser prompt),
 * downloads it, or dismisses it. Saving is serialized across tabs with a lock and proceeds only while the server
 * reports the delivery as pending. The server is notified with "saved" only after the file is written, so a reload
 * or crash mid-way leaves the delivery pending: it is never lost and never saved twice.
 *
 * The pending list and each node's last known delivery are held in state/results.ts (`deliveries`, keyed by
 * graphId:node), so a delivery is never shown on a node of a different graph. `noteDelivery` always records the
 * authoritative server record (which carries `graph`) rather than the raw SSE event (which does not), so a delivery
 * that arrives after the open graph changed is filed under its own graph. */

export const DELIVERY_DIALOG = "lab2shot-delivery"; // save dialogs for deliveries reopen in the last used location

export const DELIVERY_TYPES: Record<Delivery["mode"], FileType[]> = {
  tar: [{ description: "tar 包", accept: { "application/x-tar": [".tar"] } }],
  "tar.gz": [{ description: "压缩的 tar 包", accept: { "application/gzip": [".gz"] } }],
  folder: [],
};

/** Destination of a submitted 「输出」 delivery: the file or folder stored in this browser (its key in handles.ts). */
interface Destination {
  key: string; // run/node
  handle: string | null;
  name: string;
  mode: Delivery["mode"];
}

// Each delivery is one package keyed by its address, so the N packages delivered by an 「输出」 inside a 逐项处理
// block (one per item) are N separate entries and never overwrite each other. The destination belongs to the
// 「输出」 node, so it is looked up by node (`destKeyOf`) and is shared by all items.
const keyOf = (d: Pick<Delivery, "run" | "node" | "address">) => `${d.run}/${addressOf(d)}`;
const destKeyOf = (d: Pick<Delivery, "run" | "node">) => `${d.run}/${d.node}`;

/** Current destination of an 「输出」 node. */
export function destinationOf(node: GNode): Omit<Destination, "key"> {
  const mode = (node.data.params.mode as Delivery["mode"]) ?? "tar";
  return { handle: node.data.saveTo?.handle ?? null, name: String(node.data.params.path ?? ""), mode };
}

/** Stores the destination of each 「输出」 node under the job; called right after a cook is submitted. */
export async function rememberDestinations(job: string, nodes: GNode[]): Promise<void> {
  for (const n of nodes) await run("deliveries", "readwrite", (s) => s.put({ key: `${job}/${n.id}`, ...destinationOf(n) }));
}

/** Whether the chosen file or folder matches a delivery of `mode`: a folder for 文件夹, a .tar / .gz file for the
 * archive modes. A choice made before the mode changed does not match. */
export function fits(h: FileSystemHandle, mode: string): boolean {
  if (mode === "folder") return h.kind === "directory";
  return h.kind === "file" && h.name.toLowerCase().endsWith(mode === "tar.gz" ? ".gz" : ".tar");
}

/** The file or folder for a delivery, if this browser holds it and it matches the delivery. A package of a 逐项处理
 * block (`item`) is saved only into a folder, each item under its own name, because a single file target would be
 * overwritten by the next item; such packages wait for the user and are offered as one download (`downloadBatch`). */
async function target(d: Delivery): Promise<FileSystemHandle | undefined> {
  const dest = await run<Destination | undefined>("deliveries", "readonly", (s) => s.get(destKeyOf(d))).catch(() => undefined);
  const h = await kept(dest?.handle ?? undefined);
  if (!h || !fits(h, d.mode)) return undefined;
  return d.item && h.kind !== "directory" ? undefined : h;
}


/** Records the last delivery of an 「输出」 by (graph, node), used by the node footer and status dot so the node shows
 * the delivery outcome instead of 未计算 (「输出」 is never cached). The state is page-local: it comes from an "output"
 * event seen by this browser or from a pending delivery found on reopening (checkPending), and is updated as saveTo /
 * downloadOne / dismiss settle it. `d.graph` ("" for a submitter without a graph, such as a DCC or the command line)
 * is always taken from the authoritative record, never from the currently open graph. */
function noteDelivery(d: Pick<Delivery, "run" | "node" | "address" | "name" | "mode" | "bytes" | "graph">, state: DeliveryState): void {
  const entry: NodeDelivery = { run: d.run, state, name: d.name, mode: d.mode, bytes: d.bytes, at: Date.now() / 1000 };
  useResults.getState().setDelivery(d.graph ?? "", d.node, entry);
}

/** Reports a delivery as saved, retrying with backoff. The file is already on the user's disk when this runs, so a
 * transient failure of this request must not leave a saved delivery at 待取回 in the queue. */
async function reportSaved(d: Pick<Delivery, "run" | "node" | "address">, tries = 5): Promise<void> {
  for (let i = 0; ; i++) {
    try {
      await api.deliveries.state(d, "saved");
      return;
    } catch (e) {
      if (i >= tries - 1) throw e;
      await new Promise((r) => setTimeout(r, 400 * (i + 1)));
    }
  }
}

/** Saves one delivery to `to` under the lock if the server still reports it as pending, and notifies the server once
 * written. Returns whether the delivery is saved (by this tab or earlier). */
async function saveTo(d: Delivery, to: FileSystemHandle, where: string): Promise<boolean> {
  const k = keyOf(d);
  const work = async () => {
    const now = await api.deliveries.get(d).catch(() => null);
    if (!now || now.state !== "pending") {
      if (now) noteDelivery(now, now.state ?? "pending");
      return now?.state === "saved";
    }
    noteDelivery(now, "pending");
    try {
      // An item of a block is written to its own sub-folder of the folder chosen for the node.
      await saveDelivery(now, to, () => undefined, now.item ? now.name : "");
      try {
        await reportSaved(d);
      } catch (e) {
        // The file is on disk but the server was not notified. It must not be treated as unsaved: a pending entry
        // would be downloaded again in full by the 5-second check. A notice is shown instead; the queue row keeps
        // showing 「待取回」 and can be downloaded from there, with no automatic retry.
        say(msg("W-DELIVER-UNREPORTED", { node: now.label, where, reason: reasonOf(e) }));
        return false;
      }
      noteDelivery(now, "saved");
      say(msg(now.mode === "folder" ? "I-DELIVER-FOLDER" : "I-DELIVER-SAVED", { node: now.label, where }));
      return true;
    } catch (e) {
      // Known failures map to a specific message with a remedy; unknown ones fall back to the browser's own text
      // (files/saveProblem.ts).
      const p = saveProblem(e);
      say(msg(p.code, { node: now.label, where, reason: p.reason }));
      return false;
    }
  };
  return navigator.locks ? navigator.locks.request(`lab2shot-delivery:${k}`, work) : work();
}

/** Handles an arrived delivery (an output event of a followed job): saves it to its destination when the browser
 * grants write access, otherwise lists it for the user. The authoritative record is always fetched first, because
 * the raw SSE event carries no `graph` field and the server record is the only source for the owning graph. */
export async function deliver(d: Pick<Delivery, "run" | "node" | "address">): Promise<void> {
  const now = await api.deliveries.get(d).catch(() => null);
  if (!now) return;
  noteDelivery(now, "pending");
  const to = await target(now);
  if (to && (await writable(to)) && (await saveTo(now, to, to.name))) return;
  // Not writable (no destination chosen, no folder permission, or the originating window is closed): no dialog is
  // shown. The result stays on the server and the queue row offers 「下载」.
  if (now.state === "pending") say(msg("N-DELIVER-INQUEUE", { node: now.label }));
  noteDelivery(now, now.state ?? "pending");
}

/** Called from the click that starts a cook: requests write permission for destinations that need it again (after a
 * reload) through the browser prompt, so deliveries can be saved automatically when they arrive. */
export async function readyToSave(nodes: GNode[], defs: Record<string, NodeTypeDef>): Promise<void> {
  for (const n of nodes) {
    if (!defs[n.data.typeId]?.params.some((p) => p.widget === "deliver")) continue;
    const h = await kept(n.data.saveTo?.handle);
    if (h) await allowWrite(h);
  }
}

/** Jobs of this browser followed in the background (other than the open graph's job), so their deliveries still
 * arrive. */
const watched = new Set<string>();

export function watchJob(id: string): void {
  if (watched.has(id)) return;
  watched.add(id);
  const es = api.cookEvents(id); // self-reconnecting stream (platform/events.ts)
  es.onmessage = (data) => {
    const e = JSON.parse(data) as { type: string; run?: string; node?: string; address?: string };
    if (e.type === "output" && e.run && e.node) void deliver({ run: e.run, node: e.node, address: e.address });
    if (e.type === "finished") {
      es.close();
      watched.delete(id);
    }
  };
  es.ongone = () => void watched.delete(id);
  es.onlogin = () => void watched.delete(id); // session expired; watched again after login (resumeJobs)
}
