import { useMemo } from "react";
import { create } from "zustand";
import type { NodeStatus as Status, Plan, StatusReply } from "../api";
import type { Output } from "../api/files";
import type { StorageGate } from "../api/library";
import type { JobProgress } from "../api/progress";
import type { Message } from "../messages/message";
import type { ExposedParam } from "../api/catalog";
import { isBlocked, switchedOffNodes, type Note } from "../model/nodeOutcome";
import { shownTarget, splitTarget, targetsOf } from "../model/targets";
import { useCookInputs, type Wire } from "./cookInputs";
import { pendingNodes, type Answered, type PendingInputs } from "../model/pending";
import { useLook } from "./look";
import { rememberGood } from "./stale";
import { shared } from "../model/graphPatch";
import { setGens } from "../transfer/gens";


/** 计算结果: the server's latest report about this graph, kept only as a cache of a server response; it is never
 * undone and is never the source of truth. `reply` is the latest status reply and `forCookInputs` the
 * state/cookInputs.ts version it answers.
 * `useTrustedResults` (and `resultsAreTrusted` / `trustedReplyNow` for code outside a component) is how a reader of a
 * result checks that the cook inputs have not changed since the request.
 *
 * The reply is kept whole even after it no longer answers the current version: what the editor draws before the next
 * reply arrives (a node's ports, a wire's colour, which parameters are greyed: `results[id].applies`, read the same way
 * by the parameter panel, the node's body and the buttons) stays unchanged instead of flickering (graph/rules.ts
 * pendingPorts); what a cook decision depends on (cached, errors, policy) is read only while trusted.
 *
 * What the page SHOWS from a reply (values, sources, overrides, a wire's value, greyed rows, a port's value, the data
 * card) follows one rule, defined here only: the last reply, never emptied while an edit waits for the next one, with
 * the nodes it may no longer be right about marked `pending` (shownOf / useShownResults: the node edited, or one with
 * a wire in or out changed, and everything downstream of it). The node's state cell says 「待更新」 for them and the
 * values are drawn faint; nothing disappears and comes back.
 *
 * Opening another graph clears results, per-node cook status and outputs: `outputs` is keyed by
 * `graphId:nodeId` and `reset()` is called from graph/document.ts's loadGraph. */

export interface NodeCookStatus {
  status: import("./graph").NodeStatus;
  note: Note; // stage / progress / error text shown on the node, said when shown (model/nodeOutcome.ts noteText)
  blocked?: string; // why it cannot be cooked yet (determined before anything is sent to the server)
}

export interface CookJob {
  id: string;
  target: string;
  position: number | null;
  stopping: boolean;
}

/** A message (`messages/message.ts`). No list of shown messages is kept here: every message goes into the log
 * (`state/log.ts`); the marks that stay on a node read `results[node].messages` (the server keeps them with the result). */
export type { Message };

/** Where an 「输出」's packed download is kept (outputs): graph and node. The one spelling of that key. */
export const outputKey = (graphId: string, node: string) => `${graphId}:${node}`;

interface State {
  reply: StatusReply | null; // the latest accepted status reply (kept after it stops answering the current version)
  // its nodes (a cook's messages are added as they arrive): what is SHOWN (Shown.results is this very record, so a narrow
  // selector of one node's field reads the shown answer); a decision reads useTrustedResults
  results: Record<string, Status>;
  forCookInputs: number; // the cookInputs.version that `reply` answers (older versions may no longer hold)
  // the cook inputs the last trusted reply answered (their nodes, wires and frame range): what `pending` compares the
  // current ones with (null: none yet)
  answered: Answered | null;
  plan: Plan | null; // the shown node's plan, from `reply` (plan.node: which node it is for); read only through planOf
  planPort: string | null; // which of that node's outputs the plan was asked for (the request's show; null its main output)
  // Why the server refused this graph, in its own text (its code and detail): the latest status request, or the last
  // submission (计算, 提交). null when nothing was refused, the server never answered with text (the connection dropped),
  // or a later status reply arrived. Shown next to 提交 while it stands (editor/Chrome.tsx SubmitRefusal): a graph the
  // server cannot read says so the moment it is opened. A click on 计算 says it again instead of guessing
  // 「刚改过，或者网络断了」.
  refused: Message | null;
  // cook progress (api/progress.ts), one entry per node, stored exactly as the server sent it. Several nodes of one job
  // can cook at once, and each reads only its own; an entry is removed when its node finishes, fails or is skipped, or
  // the job ends
  running: Record<string, JobProgress>;
  job: CookJob | null;
  // between a click on 计算 and the job actually entering the queue (uploading footage, checking status, then
  // submitting): which node is being submitted. There is no job during this time; without this, repeated clicks on 计算
  // would each upload again and submit several jobs (the latch of graph/actions.ts cook)
  submitting: string | null;
  // the job that just finished (target node, end time, outcome): button parameters briefly show 「✓ 完成」 from it — a job
  // answered from the cache ends in under a second and would otherwise look like nothing happened
  justDone: { target: string; at: number; state: string } | null;
  queueSwitches: { gpu?: boolean; compute: boolean }; // gpu: only for users permitted to see the cards
  maxFrames: number; // the most frames one submission may cook (returned with the queue response; 0: not asked yet, nothing blocked)
  // this account's disk usage (lab2shot/server/quota.py usage), from the queue response. Used only to stop a clicked cook
  // when the quota is full (state/quota.ts). null: the queue has not been asked yet
  storage: StorageGate | null;  // only the three numbers needed to tell whether it is full (the 「我的占用」 panel asks for the details itself)
  byNode: Record<string, NodeCookStatus>; // idle/cooked/cooking/error + note/blocked, by node id (not saved, not undone)
  // the cookInputs.version a blocked click was judged against (-1: none). The reason a click was stopped belongs to that
  // click and to those cook inputs: it persists until they change, not until the next status reply, because a reply
  // arrives on every edit and also on every view change (showing another node requests again), and showing another node does not change the cook inputs
  blockedAt: number;
  // a click on a node's 「计算」 the page or the server refused (graph/actions.ts sayBlocked, submitRefused: a B- message):
  // why, for those cook inputs. Its button is greyed with this under it (graph/actions.ts planError) until they change,
  // so a card's 「计算并打包」 never just does nothing (the reason is not only in the log)
  stopped: { node: string; version: number; message: Message } | null;
  // `${graphId}:${node}` -> what that 「输出」 last packed (its own cook finished: its zip is there to download). From the
  // output event of the job this page follows, or the server's list when the graph is opened (graph/outputs.ts)
  outputs: Record<string, Output>;

  setReply: (reply: StatusReply, port: string | null) => void; // `port`: the displayed output the request asked about
  clearResults: () => void;
  /** A `node_done` event whose graph and cook-inputs version still match this page: writes the node's new output
   * fingerprints directly into `results` without a status round-trip. */
  applyNodeDone: (node: string, outputs: Record<string, string>) => void;
  setRefused: (why: Message | null) => void;
  setProgress: (p: JobProgress) => void;
  endProgress: (node: string) => void;
  setJob: (job: CookJob | null) => void;
  setSubmitting: (target: string | null) => void;
  setJustDone: (done: { target: string; at: number; state: string } | null) => void;
  patchJob: (patch: Partial<CookJob>) => void;
  setQueueSwitches: (sw: { gpu?: boolean; compute: boolean }) => void;
  setMaxFrames: (frames: number) => void;
  setStorage: (storage: StorageGate | null) => void;
  setNodeStatus: (id: string, patch: Partial<NodeCookStatus>) => void;
  removeNodeStatus: (ids: string[]) => void;
  setOutput: (graphId: string, node: string, o: Output) => void;
  outputFor: (graphId: string, node: string) => Output | undefined;
  reset: () => void; // another graph was loaded: no results, status or outputs of the previous one carry over
}

export const useResults = create<State>((set, get) => ({
  reply: null,
  results: {},
  forCookInputs: -1,
  answered: null,
  plan: null,
  planPort: null,
  running: {},
  job: null,
  submitting: null,
  justDone: null,
  queueSwitches: { gpu: true, compute: true },
  maxFrames: 0,
  storage: null,
  byNode: {},
  blockedAt: -1,
  stopped: null,
  refused: null,
  outputs: {},

  setReply: (sent, port) => {
    // handle data comes with a reply only when it changed (a matching key comes back alone): keep the copy already held
    const had = get().reply?.handle_data;
    const hd = sent.handle_data;
    const reply = hd && !hd.handles && had?.handles && had.key === hd.key && had.node === hd.node ? { ...sent, handle_data: had } : sent;
    // each node's answer that is the same as before stays the very object: its node does not draw again for it
    // (model/graphPatch.ts shared), so a reply redraws the nodes whose answer changed, not every node of the graph
    const results = shared(get().results, reply.nodes);
    const ci = useCookInputs.getState();
    const trusted = reply.cook_inputs === ci.version; // the reply answers the current version: the node's parameters now are those of its result
    set({ reply, results, forCookInputs: reply.cook_inputs, plan: reply.plan, planPort: port, refused: null,
          ...(trusted ? { answered: { nodes: ci.nodes, edges: ci.edges, cookRange: ci.cookRange } } : {}) });
    // every packet's generation is part of the cache keys: a recomputed fingerprint's description is no longer valid
    const gens: Record<string, string> = {};
    for (const st of Object.values(reply.nodes))
      for (const [port, fp] of Object.entries(st.outputs ?? {})) if (st.gens?.[port]) gens[fp] = st.gens[port];
    setGens(gens);
    // remember the nodes that have packets (state/stale.ts: the 「过期」 state draws this record after parameters change)
    for (const [id, st] of Object.entries(reply.nodes)) rememberGood(ci.graphId, (n) => ci.nodes[n], ci.edges, id, st.fingerprint, st.outputs, st.present, trusted);
  },
  // the server could not answer: nothing is trusted; the latest reply's nodes remain for drawing in the meantime (shownOf)
  clearResults: () => set({ forCookInputs: -1, plan: null, planPort: null, refused: null }),
  setRefused: (why) => set({ refused: why }),
  applyNodeDone: (node, outputs) =>
    set((s) => {
      const cur = s.results[node];
      if (!cur) return {};
      // node_done lists only the outputs that actually received a packet (engine/cook.py _present): they are present now
      const present = Array.from(new Set([...(cur.present ?? []), ...Object.keys(outputs)]));
      const next: Status = { ...cur, outputs: { ...(cur.outputs ?? {}), ...outputs }, present, cached: true, outcome: undefined };
      const ci = useCookInputs.getState();
      // called only when the event answers the page's current version (graph/streamDone.ts onNodeDone): the node's
      // parameters now are those its result was cooked with
      rememberGood(ci.graphId, (n) => ci.nodes[n], ci.edges, node, next.fingerprint, next.outputs, present, true);
      return { results: { ...s.results, [node]: next } };
    }),
  setProgress: (p) => set((s) => ({ running: { ...s.running, [p.node]: p } })),
  endProgress: (node) =>
    set((s) => {
      if (!(node in s.running)) return {};
      const { [node]: _, ...rest } = s.running;
      return { running: rest };
    }),
  setJob: (job) => set({ job }),
  setSubmitting: (target) => set({ submitting: target }),
  setJustDone: (done) => set({ justDone: done }),
  patchJob: (patch) => set((s) => (s.job ? { job: { ...s.job, ...patch } } : {})),
  setQueueSwitches: (sw) => set({ queueSwitches: sw }),
  setMaxFrames: (frames) => set({ maxFrames: frames }),
  setStorage: (storage) => set({ storage }),
  setNodeStatus: (id, patch) =>
    set((s) => {
      const cur = s.byNode[id];
      // a patch that changes nothing (every reply clears every node's 「拦下」 mark) is no change of the store: no node
      // redraws for it, and no subscriber is told
      if (cur && (Object.keys(patch) as (keyof NodeCookStatus)[]).every((k) => cur[k] === patch[k])) return s;
      return { byNode: { ...s.byNode, [id]: { ...(cur ?? { status: "idle", note: "" }), ...patch } } };
    }),
  removeNodeStatus: (ids) =>
    set((s) => {
      const byNode = { ...s.byNode };
      for (const id of ids) delete byNode[id];
      return { byNode };
    }),
  setOutput: (graphId, node, o) => set((s) => ({ outputs: { ...s.outputs, [outputKey(graphId, node)]: o } })),
  outputFor: (graphId, node) => get().outputs[outputKey(graphId, node)],
  // justDone is kept by node id: left in place across documents, a node of the same id in the new document (nearly every
  // template has read and deliver) would flash 「✓ 完成」 on its button. submitting is not cleared: a 计算 clicked in the
  // previous document may still be uploading / submitting, and the latch must hold until it ends (graph/actions.ts cook)
  reset: () => set({ reply: null, results: {}, forCookInputs: -1, answered: null, plan: null, planPort: null, refused: null, running: {}, byNode: {}, blockedAt: -1, stopped: null, outputs: {}, justDone: null }),
}));

const NO_RESULTS: Record<string, Status> = {};

// ------------------------------------------------------------------ what is shown (the one rule)

/** What the page shows from the server's answer: `results`, always the last reply's (kept while an edit waits for the
 * next one), and `pending`, the nodes that answer may no longer be right about. Every display reads this; a decision
 * (may it cook, is it cached) reads useTrustedResults instead. */
export interface Shown {
  results: Record<string, Status>;
  pending: ReadonlySet<string>;
}

/** The nodes whose shown answer may be out of date: model/pending.ts pendingOf (the one rule), the last result kept
 * (one entry): every node's selector asks the same question after one edit. */
let lastPending: { args: unknown[]; out: ReadonlySet<string> } | null = null;
export function pendingOf(ci: PendingInputs, forCookInputs: number, answered: Answered | null): ReadonlySet<string> {
  const args = [ci.version, ci.nodes, ci.order, ci.edges, ci.cookRange, forCookInputs, answered];
  if (lastPending && lastPending.args.every((a, i) => a === args[i])) return lastPending.out;
  const out = pendingNodes(ci, forCookInputs, answered);
  lastPending = { args, out };
  return out;
}

/** shownOf for code outside a component. */
export function shownNow(): Shown {
  const r = useResults.getState();
  return { results: r.results, pending: pendingOf(useCookInputs.getState(), r.forCookInputs, r.answered) };
}

/** What is shown (see Shown), for a component: recomputed when the results or the cook inputs change. */
export function useShownResults(): Shown {
  const results = useResults((s) => s.results);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const answered = useResults((s) => s.answered);
  const version = useCookInputs((s) => s.version);
  const nodes = useCookInputs((s) => s.nodes);
  const order = useCookInputs((s) => s.order);
  const edges = useCookInputs((s) => s.edges);
  const cookRange = useCookInputs((s) => s.cookRange);
  const pending = useMemo(() => pendingOf({ version, nodes, order, edges, cookRange }, forCookInputs, answered),
    [version, nodes, order, edges, cookRange, forCookInputs, answered]);
  return useMemo(() => ({ results, pending }), [results, pending]);
}

/** Whether one node's shown answer waits for the next reply (its own selectors: a node redraws only when its answer
 * changes, not on every edit elsewhere). */
export function usePendingNode(id: string): boolean {
  const now = () => {
    const r = useResults.getState();
    return pendingOf(useCookInputs.getState(), r.forCookInputs, r.answered).has(id);
  };
  const a = useResults(now);
  const b = useCookInputs(now);
  return a || b;
}

/** The node parameter an exposed entry speaks for now (model/targets.ts shownTarget: its first target on now), by the
 * nodes the last answer found switched off (model/nodeOutcome.ts switchedOffNodes). One target: that one. */
function shownTargetKey(x: Pick<ExposedParam, "target">, byNode: State["byNode"], edges: readonly Wire[]): string {
  const all = targetsOf(x);
  if (all.length < 2) return all[0] ?? "";
  return shownTarget(x, switchedOffNodes(edges, (id) => isBlocked(byNode[id]))).join(".");
}

/** shownTargetKey for a component: redrawn only when the node it speaks for changes. */
export function useShownTarget(x: Pick<ExposedParam, "target">): [string, string] {
  const edges = useCookInputs((s) => s.edges);
  return splitTarget(useResults((s) => shownTargetKey(x, s.byNode, edges)));
}

/** shownTargetKey for code outside a component. */
export const shownTargetNow = (x: Pick<ExposedParam, "target">): [string, string] =>
  splitTarget(shownTargetKey(x, useResults.getState().byNode, useCookInputs.getState().edges));

/** The results while they answer the current cook inputs: an empty record once state/cookInputs.ts's version has moved
 * past the version these results answer ("拿不准就当要算"). Renaming the graph or laying out its parameter interface does not
 * move that version (only its `edits`), so results stay trusted through it. Use this rather than reading `useResults((s) => s.results)` directly. */
export function useTrustedResults(): Record<string, Status> {
  const version = useCookInputs((s) => s.version);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const results = useResults((s) => s.results);
  return forCookInputs === version ? results : NO_RESULTS;
}

export const resultsAreTrusted = (): boolean => useResults.getState().forCookInputs === useCookInputs.getState().version;

/** The latest reply while it answers the current cook inputs (null: none yet, or edited since). */
export const trustedReplyNow = (): StatusReply | null => (resultsAreTrusted() ? useResults.getState().reply : null);

/** What a plan must answer to be acted on: the node shown, the output of it shown (null: its main output) and the cook
 * inputs' version. */
export interface PlanAt {
  node: string | null;
  port: string | null;
  version: number;
}

/** The packet an output of a node actually has (its fingerprint), or null: `outputs` names every output's fingerprint,
 * but only the outputs the status marks `present` were written (outputs are made on demand). The one test of "this port
 * has a result": the view, the data panel, 复制到 Nuke and the reference menu all read it. */
export const packetOf = (st: { present?: string[]; outputs?: Record<string, string> } | undefined, port: string): string | null =>
  st?.present?.includes(port) ? st.outputs?.[port] ?? null : null;

/** The plan for `at` (null: none for it, or edited / switched since): the one test of whether a plan can be acted on.
 * A selector passes what it subscribed to (graph/actions.ts cookHold, editor/Chrome.tsx); code outside one uses
 * planHere. */
export const planOf = (s: Pick<ReturnType<typeof useResults.getState>, "plan" | "planPort" | "forCookInputs">, at: PlanAt): Plan | null =>
  s.forCookInputs === at.version && at.node && s.plan?.node === at.node && s.planPort === at.port ? s.plan : null;

/** Where the plan must be now: the displayed node and output, the current cook inputs. */
export const planAtNow = (): PlanAt => ({ node: useLook.getState().displayId, port: useLook.getState().displayPort, version: useCookInputs.getState().version });

/** The plan of what is shown now (null: none for it yet). */
export const planHere = (): Plan | null => planOf(useResults.getState(), planAtNow());
