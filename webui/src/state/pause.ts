import type { CookKind } from "../api";
import { msg, type Message } from "../messages/message";

/** 计算任务 off (B-QUEUE-PAUSED, lab2shot/messages/web.toml): the only switch the page decides on. The reason a queued
 * job waits (no machine, memory, the cards) is reported by the server in the job's `waiting` text (farm/queue.py). */
export const pausedMessage = (): Message => msg("B-QUEUE-PAUSED");

/** 计算任务: what a click on 计算 or 提交 of this `kind` would encounter ("" when nothing blocks it), shown in the top
 * bar's and the context menu's tooltip before the click (readyForPause enforces it). */
export const pauseNote = (sw: { compute: boolean }, kind: Pick<CookKind, "lane" | "delivers">): string =>
  !sw.compute && (kind.lane !== "light" || kind.delivers) ? `\n\n${pausedMessage().text}` : "";

/** Whether 计算任务 off refuses this `kind` outright (nothing is submitted): disables 提交 and the menu's 计算. */
export const blockedByCompute = (sw: { compute: boolean }, kind: Pick<CookKind, "lane" | "delivers">): boolean =>
  !sw.compute && (kind.lane !== "light" || kind.delivers);

/** Checked before submitting a click-triggered cook (never a display-triggered one, which is always light and delivers
 * nothing): with 计算任务 off, everything except pure viewing is refused outright (a message is shown and nothing is
 * submitted). Everything else goes to the queue; a job that has to wait reports why in the server's text. */
export async function readyForPause(sw: { compute: boolean }, kind: Pick<CookKind, "lane" | "delivers">, blocked: (m: Message) => void): Promise<boolean> {
  if (blockedByCompute(sw, kind)) {
    blocked(pausedMessage());
    return false;
  }
  return true;
}

/** The lanes the page reports as paused, based on the switches received: 计算任务 off; 显卡任务 off only when this
 * session received the cards' switch at all (farm.cards). A switch that was not received is never reported as off. */
export const pausedLanes = (sw: { gpu?: boolean; compute: boolean }): ("compute" | "gpu")[] =>
  !sw.compute ? ["compute"] : sw.gpu === false ? ["gpu"] : [];
