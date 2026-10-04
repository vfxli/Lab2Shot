import { create } from "zustand";

/** The user's files going up. The store lives here with the page's other stores; the
 * transport logic (chunking, retrying, the BroadcastChannel between tabs) stays in transfer/uploads.ts, which imports
 * the type and the store from here. Tagged 计算结果: an upload's progress is the same kind of thing as a job's
 * progress (state/results.ts) — a background operation under way, never saved, never part of the undo history. */

export type UploadState =
  // after picking, a task rests in these two states with no byte sent: bytes go up only when a node downstream is to
  // be cooked, and only the channels its wires use
  | "reading" // reading the files on the user's machine and computing their content fingerprints (sha256), so the upload can be
  // declared and the node's ports appear
  | "picked" // declared: the node's ports are there, the bytes are still on the user's machine; they go on 计算 (`graph/apply.ts
  // sendPicked`)
  | "sending" // going up
  | "waiting" // the line dropped (or the server is away): trying again by itself
  | "finishing" // every file is in: the server puts the set together
  | "paused" // after a reload: the browser no longer has the files; picking them again carries on
  | "elsewhere" // another tab of this browser is sending it
  | "failed"; // the server refused it (too big, the disk full ...): 重试 or pick again

export interface UploadTask {
  key: string;
  // the document the task belongs to (`state/cookInputs.ts` graphId): nodes of different documents may share an id (every
  // template's read node is called `read`), so finding a task, writing the parameter back and drawing the local files
  // all compare it first; another document's task is neither shown nor written
  graphId: string;
  graph: string; // the graph's name (for the messages), and the node and parameter it is for
  node: string;
  param: string;
  name: string; // what the node reads: the file, or the sequence's pattern
  folder: string; // the folder the browser named ("" none)
  origin: string; // folder/name: what it is called on the user's machine, as far as the browser says
  sequence: boolean;
  frames: number[];
  // `sha`: the content fingerprint computed locally in the declare step (`transfer/declare.ts`); the channel-level upload
  // asks the server by it which channels are still missing
  files: { name: string; size: number; modified: number; sha?: string }[];
  bytes: number;
  // how far it got (this tab's own, or what the sending tab says)
  state: UploadState;
  sent: number; // bytes the server has, and the part of a request in flight
  done: number; // files the server has whole
  rate: number; // bytes per second, smoothed (0: not known yet)
  error: string;
  retryAt: number; // waiting: when it asks again (ms since the epoch)
  tries?: number; // waiting: how many times in a row it has not got through (a submission gives up past a limit: graph/submitLine.ts)
}

interface Uploads {
  tasks: Record<string, UploadTask>;
}

export const useUploads = create<Uploads>(() => ({ tasks: {} }));
