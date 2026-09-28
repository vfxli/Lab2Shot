import { create } from "zustand";

/** The user's files going up. The store lives here with the page's other stores; the
 * transport logic (chunking, retrying, the BroadcastChannel between tabs) stays in transfer/uploads.ts, which imports
 * the type and the store from here. Tagged 计算结果: an upload's progress is the same kind of thing as a job's
 * progress (state/results.ts) — a background operation under way, never saved, never part of the undo history. */

export type UploadState =
  // 选完文件之后先停在这两档，一个字节都不传：字节只在下游有节点要算、按连线用到的通道才上传
  | "reading" // 正在读他机器上那些文件、算内容指纹（sha256）：为的是把这份上传申报上去，好让节点长出口来
  | "picked" // 申报完了，节点上的口都在，字节还在他机器上：点「计算」才传（`graph/apply.ts sendPicked`）
  | "sending" // going up
  | "waiting" // the line dropped (or the server is away): trying again by itself
  | "finishing" // every file is in: the server puts the set together
  | "paused" // after a reload: the browser no longer has the files; picking them again carries on
  | "elsewhere" // another tab of this browser is sending it
  | "failed"; // the server refused it (too big, the disk full ...): 重试 or pick again

export interface UploadTask {
  key: string;
  graph: string; // the graph's name, and the node and parameter it is for
  node: string;
  param: string;
  name: string; // what the node reads: the file, or the sequence's pattern
  folder: string; // the folder the browser named ("" none)
  origin: string; // folder/name: what it is called on the user's machine, as far as the browser says
  sequence: boolean;
  frames: number[];
  // `sha`: 内容指纹，申报那一步在本机算出来的（`transfer/declare.ts`）；「通道级上传」按它问服务器还缺哪几条通道
  files: { name: string; size: number; modified: number; sha?: string }[];
  bytes: number;
  // how far it got (this tab's own, or what the sending tab says)
  state: UploadState;
  sent: number; // bytes the server has, and the part of a request in flight
  done: number; // files the server has whole
  rate: number; // bytes per second, smoothed (0: not known yet)
  error: string;
  retryAt: number; // waiting: when it asks again (ms since the epoch)
}

interface Uploads {
  tasks: Record<string, UploadTask>;
}

export const useUploads = create<Uploads>(() => ({ tasks: {} }));
