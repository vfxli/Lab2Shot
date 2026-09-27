/** What the install control (install/Install.tsx) shows for one extension, decided from the server's answer alone (its
 * `actions`, resolved by lab2shot/server/available.py extension, and its latest job): no role is looked at here.
 *
 * Pure functions over the answer (the one `shown` rule from api/applies.ts); Node's test runner reads it through
 * tests/support/tsExtResolve.mjs (webui/tests/installState.test.ts). */
import { shown, type Availability } from "../api/applies";
import type { InstallTask } from "../api";
import { taskLive } from "../api/tasks";

export interface InstallFacts {
  installed: boolean;
  ready: boolean;
  actions: Availability;
  job: InstallTask | null;
}

export { taskLive };  // the one "still to finish" rule, every background task's (api/tasks.ts)

/** "none": nothing at all (a login that does not install, or a ready card with no job running); "progress": the job
 * runs or waits; "button": what can be started. `full`: the project page, which also offers reinstall on a ready one. */
export function controlKind(p: InstallFacts, full: boolean): "none" | "progress" | "button" {
  if (taskLive(p.job) && shown(p.actions, "progress")) return "progress";  // an install running now: follow it
  if (!shown(p.actions, "install")) return "none";
  return p.ready && !full ? "none" : "button";
}

/** The step a job is on, or the one it failed at. */
export const currentStep = (job: InstallTask | null | undefined) => job?.steps.find((s) => s.state === "running") ?? job?.steps.find((s) => s.state === "failed");

/** The button's word and whether it rebuilds everything: 重新安装 (ready: every step again, beside the live one), 重试
 * (the last job failed: from its step), 补全安装 (installed but something is missing), 安装. */
export function installButton(p: InstallFacts): { label: string; force: boolean } {
  if (p.ready) return { label: "重新安装", force: true };
  if (p.job?.state === "failed") return { label: "重试", force: false };
  return { label: p.installed ? "补全安装" : "安装", force: false };
}
