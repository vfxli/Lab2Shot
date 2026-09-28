import { create } from "zustand";
import type { NodeStatus as Status, Plan, StatusReply } from "../api";
import type { Output } from "../api/files";
import type { StorageGate } from "../api/library";
import type { JobProgress } from "../api/progress";
import type { Message } from "../messages/message";
import { useCookInputs } from "./cookInputs";
import { rememberGood } from "./stale";
import { setGens } from "../transfer/gens";
import { forgetManifest } from "../transfer/frames";
import { onGenChanged } from "../transfer/gens";

onGenChanged((changed) => changed.forEach(forgetManifest));

/** 计算结果: the server's latest report about this graph, kept only as a cache of a server response; it is never
 * undone and is never the source of truth. `reply` is the latest status reply and `forCookInputs` the
 * state/cookInputs.ts version it answers.
 * `useTrustedResults` (and `resultsAreTrusted` / `trustedReplyNow` for code outside a component) is how a reader of a
 * result checks that the cook inputs have not changed since the request.
 *
 * The reply is kept whole even after it no longer answers the current version: what the editor draws before the next
 * reply arrives (a node's ports, a wire's colour) stays unchanged instead of flickering (graph/rules.ts pendingPorts);
 * what a cook decision depends on (cached, errors, policy) is read only while trusted.
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

/** 一条消息（`messages/message.ts`）。此处不保存已显示消息的清单：所有消息统一写入日志
 * （`state/log.ts`）；节点上常驻的标记读取 `results[节点].messages`（服务器将其与结果一并保存）。 */
export type { Message };

const outputKey = (graphId: string, node: string) => `${graphId}:${node}`;

interface State {
  reply: StatusReply | null; // the latest accepted status reply (kept after it stops answering the current version)
  results: Record<string, Status>; // its nodes (a cook's messages are added as they arrive)
  forCookInputs: number; // the cookInputs.version that `reply` answers (older versions may no longer hold)
  plan: Plan | null; // the shown node's plan, from `reply` (plan.node: which node it is for)
  // Why the server refused this graph, in its own text (its code and detail): the latest status request, or the last
  // submission (计算, 提交). null when nothing was refused, the server never answered with text (the connection dropped),
  // or a later status reply arrived. Shown next to 提交 while it stands (editor/Chrome.tsx SubmitRefusal): a graph the
  // server cannot read says so the moment it is opened. A click on 计算 says it again instead of guessing
  // 「刚改过，或者网络断了」.
  refused: Message | null;
  // 计算进度（api/progress.ts），按节点各存一份：服务器发来的数据原样存放于此。一个任务的几个节点可以同时在算，
  // 每个节点只读自己的那一份；节点算完、出错、跳过或任务结束时移除
  running: Record<string, JobProgress>;
  job: CookJob | null;
  queueSwitches: { gpu?: boolean; compute: boolean }; // gpu: only for users permitted to see the cards
  maxFrames: number; // 单次提交允许计算的最大帧数（随队列响应返回；0 表示尚未查询，不拦截）
  // 当前账号的磁盘占用（lab2shot/server/quota.py usage），取自队列响应。仅用于
  // 配额已满时拦截点击触发的计算（state/quota.ts）。null 表示尚未查询队列
  storage: StorageGate | null;  // 仅含判断「是否已满」所需的三个数（明细由「我的占用」面板单独查询）
  byNode: Record<string, NodeCookStatus>; // idle/cooked/cooking/error + note/blocked, by node id (not saved, not undone)
  // the cookInputs.version a blocked click was judged against (-1: none). The reason a click was stopped belongs to that
  // click and to those cook inputs: it persists until they change, not until the next status reply, because a reply
  // arrives on every edit and also on every view change (showing another node requests again), and showing another node does not change the cook inputs
  blockedAt: number;
  // `${graphId}:${node}` -> what that 「输出」 last packed (its own cook finished: its zip is there to download). From the
  // output event of the job this page follows, or the server's list when the graph is opened (graph/outputs.ts)
  outputs: Record<string, Output>;

  setReply: (reply: StatusReply) => void;
  clearResults: () => void;
  /** A `node_done` event whose graph and cook-inputs version still match this page: writes the node's new output
   * fingerprints directly into `results` without a status round-trip. */
  applyNodeDone: (node: string, outputs: Record<string, string>) => void;
  setRefused: (why: Message | null) => void;
  setProgress: (p: JobProgress) => void;
  endProgress: (node: string) => void;
  setJob: (job: CookJob | null) => void;
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
  running: {},
  job: null,
  queueSwitches: { gpu: true, compute: true },
  maxFrames: 0,
  storage: null,
  byNode: {},
  blockedAt: -1,
  refused: null,
  outputs: {},

  setReply: (reply) => {
    set({ reply, results: reply.nodes, forCookInputs: reply.cook_inputs, plan: reply.plan, refused: null });
    // 每个包的生成号计入缓存键：重算后的指纹对应的包说明随之失效
    const gens: Record<string, string> = {};
    for (const st of Object.values(reply.nodes))
      for (const [port, fp] of Object.entries(st.outputs ?? {})) if (st.gens?.[port]) gens[fp] = st.gens[port];
    setGens(gens);
    // 记录有包的节点（state/stale.ts：参数修改后「过期」档绘制的即是此记录）
    const ci = useCookInputs.getState();
    const trusted = reply.cook_inputs === ci.version; // 回复的就是当前版本：节点现在的参数即其结果的参数
    for (const [id, st] of Object.entries(reply.nodes)) rememberGood(ci.graphId, (n) => ci.nodes[n], ci.edges, id, st.fingerprint, st.outputs, st.present, trusted);
  },
  // the server could not answer: nothing is trusted; the latest reply remains for drawing in the meantime
  clearResults: () => set({ results: {}, forCookInputs: -1, plan: null, refused: null }),
  setRefused: (why) => set({ refused: why }),
  applyNodeDone: (node, outputs) =>
    set((s) => {
      const cur = s.results[node];
      if (!cur) return {};
      // node_done lists only the outputs that actually received a packet (engine/cook.py _present): they are present now
      const present = Array.from(new Set([...(cur.present ?? []), ...Object.keys(outputs)]));
      const next: Status = { ...cur, outputs: { ...(cur.outputs ?? {}), ...outputs }, present, cached: true, outcome: undefined };
      const ci = useCookInputs.getState();
      // 只在事件回答页面当前版本时才调用（graph/streamDone.ts onNodeDone）：此时节点现在的参数即算出结果的参数
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
  reset: () => set({ reply: null, results: {}, forCookInputs: -1, plan: null, refused: null, running: {}, byNode: {}, blockedAt: -1, outputs: {} }),
}));

const NO_RESULTS: Record<string, Status> = {};

/** The results while they answer the current cook inputs: an empty record once state/cookInputs.ts's version has moved
 * past the version these results answer ("拿不准就当要算"). Use this rather than reading `useResults((s) => s.results)` directly. */
export function useTrustedResults(): Record<string, Status> {
  const version = useCookInputs((s) => s.version);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const results = useResults((s) => s.results);
  return forCookInputs === version ? results : NO_RESULTS;
}

export const resultsAreTrusted = (): boolean => useResults.getState().forCookInputs === useCookInputs.getState().version;

/** The latest reply while it answers the current cook inputs (null: none yet, or edited since). */
export const trustedReplyNow = (): StatusReply | null => (resultsAreTrusted() ? useResults.getState().reply : null);

/** The plan of the currently shown node, while it answers the current cook inputs (null: none for it yet). */
export function planFor(displayId: string | null): Plan | null {
  const r = useResults.getState();
  return resultsAreTrusted() && displayId && r.plan?.node === displayId ? r.plan : null;
}
