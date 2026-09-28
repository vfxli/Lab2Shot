import type { StorageGate } from "../api/library";
import { msg, type Message } from "../messages/message";
import { blockedByQuota, quotaNote } from "./quota";

/** 计算任务 off (B-QUEUE-PAUSED, lab2shot/messages/web.toml): the only switch the page decides on. The reason a queued
 * job waits (no machine, memory, the cards) is reported by the server in the job's `waiting` text (farm/queue.py). */
const pausedMessage = (): Message => msg("B-QUEUE-PAUSED");

/** 计算任务: what a click on 计算 or 提交 would encounter ("" when nothing blocks it), shown in the top bar's and the
 * context menu's tooltip before the click (readyForPause enforces it). */
const pauseNote = (sw: { compute: boolean }): string => (blockedByCompute(sw) ? `\n\n${pausedMessage().text}` : "");

/** Whether 计算任务 off refuses a cook outright (nothing is submitted). */
const blockedByCompute = (sw: { compute: boolean }): boolean => !sw.compute;

/** The one rule of the top bar's 提交 and the node menu's 计算 before a click: greyed while 计算任务 is off or the
 * storage quota is full (both refuse a cook outright), with the note that says which. */
export const cookBlocked = (sw: { compute: boolean }, storage: StorageGate | null): boolean => blockedByCompute(sw) || blockedByQuota(storage);
export const cookNote = (sw: { compute: boolean }, storage: StorageGate | null): string => pauseNote(sw) + quotaNote(storage);

/** Checked before submitting a cook: with 计算任务 off every cook is refused outright (a message is shown and nothing is
 * submitted); otherwise it goes on, and a task that has to wait reports why in the server's text. */
export async function readyForPause(sw: { compute: boolean }, blocked: (m: Message) => void): Promise<boolean> {
  if (blockedByCompute(sw)) {
    blocked(pausedMessage());
    return false;
  }
  return true;
}

/** The lanes the page reports as paused, based on the switches received: 计算任务 off; 显卡任务 off only when this
 * session received the cards' switch at all (farm.cards). A switch that was not received is never reported as off. */
export const pausedLanes = (sw: { gpu?: boolean; compute: boolean }): ("compute" | "gpu")[] =>
  !sw.compute ? ["compute"] : sw.gpu === false ? ["gpu"] : [];
