import type { MessageJson } from "./applies";

/** A background task on the server (lab2shot/farm/tasks.py Task.json): an extension's install, a card's check. One
 * format for all of them; `subject` is what it works on (an extension, a card), `kind` which kind of task. It may wait
 * in line (`queued`, its `waiting` words say why) before it runs. */
export interface BackgroundTask {
  id: string;
  kind: string;
  subject: string;
  title: MessageJson;
  by: string;
  state: "queued" | "running" | "done" | "failed" | "cancelled";
  done: number;
  total: number;
  label: string;
  said: MessageJson[];
  result: MessageJson | null;
  waiting: MessageJson | null;
  submitted: number;
  started: number | null;
  finished: number | null;
  lines: string[]; // its output since the `since` asked for
  next: number; // ask with since=next for what comes after
  last: string; // its latest output line
}

/** Still to finish: waiting in line or running. */
export const taskLive = (t: Pick<BackgroundTask, "state"> | null | undefined): boolean => t?.state === "queued" || t?.state === "running";
