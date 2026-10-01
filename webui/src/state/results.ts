import { create } from "zustand";
import type { NodeStatus as Status, Plan, StatusReply } from "../api";
import type { Output } from "../api/files";
import type { StorageGate } from "../api/library";
import type { JobProgress } from "../api/progress";
import type { Message } from "../messages/message";
import { useCookInputs } from "./cookInputs";
import { useLook } from "./look";
import { rememberGood } from "./stale";
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
 * Opening another graph clears results, per-node cook status and outputs: `outputs` is keyed by
 * `graphId:nodeId` and `reset()` is called from graph/document.ts's loadGraph. */

export interface NodeCookStatus {
  status: import("./graph").NodeStatus;
  note: string; // stage / progress / error text shown on the node
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
  results: Record<string, Status>; // its nodes (a cook's messages are added as they arrive)
  forCookInputs: number; // the cookInputs.version that `reply` answers (older versions may no longer hold)
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
  refused: null,
  outputs: {},

  setReply: (sent, port) => {
    // handle data comes with a reply only when it changed (a matching key comes back alone): keep the copy already held
    const had = get().reply?.handle_data;
    const hd = sent.handle_data;
    const reply = hd && !hd.handles && had?.handles && had.key === hd.key && had.node === hd.node ? { ...sent, handle_data: had } : sent;
    set({ reply, results: reply.nodes, forCookInputs: reply.cook_inputs, plan: reply.plan, planPort: port, refused: null });
    // every packet's generation is part of the cache keys: a recomputed fingerprint's description is no longer valid
    const gens: Record<string, string> = {};
    for (const st of Object.values(reply.nodes))
      for (const [port, fp] of Object.entries(st.outputs ?? {})) if (st.gens?.[port]) gens[fp] = st.gens[port];
    setGens(gens);
    // remember the nodes that have packets (state/stale.ts: the 「过期」 state draws this record after parameters change)
    const ci = useCookInputs.getState();
    const trusted = reply.cook_inputs === ci.version; // the reply answers the current version: the node's parameters now are those of its result
    for (const [id, st] of Object.entries(reply.nodes)) rememberGood(ci.graphId, (n) => ci.nodes[n], ci.edges, id, st.fingerprint, st.outputs, st.present, trusted);
  },
  // the server could not answer: nothing is trusted; the latest reply remains for drawing in the meantime
  clearResults: () => set({ results: {}, forCookInputs: -1, plan: null, planPort: null, refused: null }),
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
  setNodeStatus: (id, patch) => set((s) => ({ byNode: { ...s.byNode, [id]: { ...(s.byNode[id] ?? { status: "idle", note: "" }), ...patch } } })),
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
  reset: () => set({ reply: null, results: {}, forCookInputs: -1, plan: null, planPort: null, refused: null, running: {}, byNode: {}, blockedAt: -1, outputs: {}, justDone: null }),
}));

const NO_RESULTS: Record<string, Status> = {};

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
