/** Files between the user's machine and the server, as the server describes them: an upload, a delivery. */

export interface Upload {
  ref: string; // what the node parameter holds: upload:<id>/<name>
  name: string;
  stem: string; // the plate's own name, as the server's sequence rule reads it (render.####.exr: render; lab2shot/io/sequence.py)
  files: number;
  bytes: number;
  first: number | null; // a sequence's first and last frame
  last: number | null;
}

/** What became of a delivery (lab2shot/transfer/deliveries.py): 待取回 / 已保存 / 已下载 / 不要了 / 已过期. */
export type DeliveryState = "pending" | "saved" | "downloaded" | "dismissed" | "expired";

/** What one 「输出」 delivered in one run (the server's output event, a job's outputs, or the server's record). */
export interface Delivery {
  node: string;
  // An 「输出」 inside a 逐项处理 block delivers one package per item: `item` is that instance's item path
  // ("" outside every block) and `address` is where the package is requested: the node's id, or <node>.<hash> per item
  // (transfer/deliveries.py address). Every /api/deliveries/{run}/{node} path is the address.
  item?: string;
  address?: string;
  label: string;
  run: string; // the job
  name: string; // the archive's file name (sh010.tar), or the folder's name
  mode: "tar" | "tar.gz" | "folder";
  files: string[]; // every file of it, relative: lab2shot.json, then <名字>/... per output
  bytes: number;
  state?: DeliveryState;
  expires?: number | null; // seconds since the epoch: gone from the server after that
  title?: string; // the graph's name, as it was when this delivery was made
  graph?: string; // the graph file's own meta.id: "" for a submitter that
  // sent none (a DCC or the command line); the page draws a delivery on a node only when this matches the open graph
}

/** Where a delivery is asked for: its address (the 「输出」 node, or node.<hash> for one item of a 逐项处理 block).
 * A record made before the block work carries none, and is then its node. */
export const addressOf = (d: Pick<Delivery, "node" | "address">): string => d.address ?? d.node;
