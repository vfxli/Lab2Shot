import type { GpuFitItem, GpuFitState } from ".";
import { json } from "../platform/http";
import type { MessageJson } from "./applies";

/** 显卡 (lab2shot/farm/cards.py view): every card of this machine, every parameter tier and the cards that run it, every
 * GPU node's measured VRAM, every GPU node waiting for a card and whether an authorised card could ever run it. */
export interface CardRow {
  uuid: string;
  index: number;
  name: string;
  model: string;
  memory_gb: number;
  arch: string;
  authorized: boolean;
  load: { utilization: number; used_gb: number; temperature: number };
  running: { job: string; title: string; who: string; node: string } | null; // the node on it now, and its job
  extensions: Record<GpuFitState, GpuFitItem[]>;
  tiers: string[];
}

export interface CardTier {
  id: string;
  node: string;
  label: string;
  param: string; // "" the node's default
  option: string;
  vram_gb: number;
  runtime: string; // the extension whose environment runs it (the page groups tiers by it)
  runtime_title: string;
  message: MessageJson;
  available: boolean; // an authorised card runs it
  cards: string[]; // the cards that could (authorised or not)
}

export interface CardNode {
  node: string;
  label: string;
  runtime: string;
  runtime_title: string;
  vram_gb: number;
  vram_measured: boolean;
  measured_on: string | null;
  note: string | MessageJson | null;
}

/** One GPU node waiting for a card (lab2shot/farm/cards.py _waiting_row): the job it is of, and the node. */
export interface CardWaiting {
  job: string;
  title: string;
  who: string;
  node: string; // the node's label
  vram_gb: number;
  runtimes: string[];
  reason: MessageJson | null;
  runnable_ever: boolean;
}

/** 一小时内的平均显卡使用率（lab2shot/farm/scheduler/inventory.py hourly）：`hour` 为 Unix 秒 ÷ 3600，
 * `average` 为每张卡 0–100 的数值；`running` 表示该小时尚未结束，当前值按已采集的样本计算。 */
export interface CardHour {
  hour: number;
  average: Record<string, number>;
  running?: boolean;
}

interface CardsView {
  cards: CardRow[];
  tiers: CardTier[];
  nodes: CardNode[];
  waiting: CardWaiting[];
  hourly: CardHour[];
}

/** What authorising exactly these cards would change, before anything changes (farm/cards.py consequences). */
export interface CardsConsequences {
  authorized: string[];
  lost_tiers: string[];
  gained_tiers: string[];
  stuck_jobs: { job: string; title: string; who: string }[];
  unrunnable_extensions: string[];
  messages: MessageJson[];
}

/** The administrator's cards (server/farm.py /api/admin/cards, the farm.cards capability). */
export const cardsApi = {
  cards: () => json<CardsView>("GET", "/api/admin/cards"),
  cardConsequences: (authorized: string[]) => json<CardsConsequences>("POST", "/api/admin/cards/consequences", { authorized }),
};
