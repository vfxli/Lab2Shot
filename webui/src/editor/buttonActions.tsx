import { useEffect, useState } from "react";
/** 按钮参数（服务端 lab2shot/nodes/params.py Button，widget「button」）：参数面板里和别的参数一样一行（左边三个标记，
 * 右边一个占满参数列的按钮），也可以显示在节点上。按钮没有值，只有一个动作 id；这里是动作表（动作 id → 这个节点上
 * 该做的事、现在能不能点、按钮上写什么）。按钮之间不串联：一个按钮就做一件事。
 *
 *   cook      「计算」（每个节点都有）：和右键「计算」这个节点是同一个函数（graph/actions.ts cookNode）；在「输出」上就是
 *             整理打包它。
 *   download  「下载」（「输出」的）：下载它打好的包（files/outputs.ts downloadOutput，与 ui/OutputDownload.tsx 同一件事）；
 *             还没有包时置灰，这个「输出」正在算时写进度。
 *   pick_in_view 「在视图里点选」（框架给每个 picks / canvas 参数自动生成一个，服务端 nodes/params.py pick_button）：让视图
 *             显示这个节点、手柄可用；亮不亮由事实推导（state/viewPicking.ts）。
 *
 * 新增一种按钮：服务端节点声明 `buttons = (Button(名字, 显示名, 动作 id),)`，这里加一行动作。 */

import type { ParamDef } from "../api";
import { cookHold, cookNode, planError, type CookHold } from "../graph/actions";
import { downloadOutput } from "../files/outputs";
import { sizeText } from "../platform/format";
import { useCookInputs } from "../state/cookInputs";
import { outputKey, useResults } from "../state/results";
import { Button } from "../ui/Button";
import { canLeave, enterPicking, leavePicking, pickingOn } from "../state/viewPicking";
import { why } from "../api/applies";
import { useLook } from "../state/look";
import { useGraphSnapshot } from "../graph/snapshot";
import { handlesOf } from "../graph/rules";

interface ButtonNow {
  text: string; // 按钮上的字
  off: boolean; // 现在点不了
  why: string; // 点不了的原因 / 点了会做什么：只作悬停提示（节点上）；面板里不另起一行：那一行一闪而过会让整个面板跳动。
  // 点不了的时候原因压缩进按钮文字本身（一行、高度不变），细节走顶栏「日志」和通知
  run: () => void;
  on?: boolean; // 一个开关状态的按钮现在开着（「在视图里点选」正在点选）：按钮高亮
}

type ActionHook = (nodeId: string, p: ParamDef) => ButtonNow;

/** 「计算」点不了时压进按钮文字的原因（一行、高度不变）。 */
const HOLD_WORDS: Record<Exclude<CookHold, "unplannable" | null>, string> = { submitting: "提交中…", busy: "有任务在算", paused: "现在不能提交" };

/** 这个节点正在算吗、算到哪：不在算为 undefined；在算为整个任务的完成量（0–1），还不知道为 null。 */
function useCooking(nodeId: string): number | null | undefined {
  const job = useResults((s) => s.job);
  const at = useResults((s) => s.running[nodeId]?.at);
  if (!job || job.target !== nodeId) return undefined;
  return at ?? null;
}

const percent = (at: number) => `${Math.round(at * 100)}%`;

/** 这个节点的任务刚结束（1.5 秒内）：按钮短暂写「✓ 完成」/「✗ 出错」，让秒算完的任务也看得出点过了。 */
function useJustDone(nodeId: string): string | null {
  const done = useResults((s) => s.justDone);
  const [, tick] = useState(0);
  const hit = !!done && done.target === nodeId && Date.now() - done.at < 1500;
  useEffect(() => {
    if (!hit) return;
    const t = setTimeout(() => tick((n) => n + 1), 1500 - (Date.now() - done!.at) + 20);
    return () => clearTimeout(t);
  }, [hit, done]);
  return hit ? (done!.state === "done" ? "✓ 完成" : done!.state === "cancelled" ? "已取消" : "✗ 出错，看日志") : null;
}

const useCook: ActionHook = (nodeId, p) => {
  // 点不点得了、为什么：与 cook() 的闩同一处（graph/actions.ts cookHold）
  const version = useCookInputs((s) => s.version); // 预估只在对上当前编辑时算数（cookHold）
  const port = useLook((s) => s.displayPort); // 预估也要对上看的口
  const hold = useResults((s) => cookHold(s, { node: nodeId, version, port }));
  const unplannable = useResults((s) => planError(s, { node: nodeId, version, port })?.text ?? ""); // 服务器说算不了的原因：原句压进按钮文字
  const mine = useResults((s) => s.submitting === nodeId); // 正在提交的就是这个节点
  const justDone = useJustDone(nodeId);
  const cooking = useCooking(nodeId);
  return {
    text: cooking !== undefined ? (cooking === null ? "计算中…" : `计算中 ${percent(cooking)}`) : (mine ? "上传素材、提交中…" : justDone ? `${p.label} · ${justDone}` : hold ? `${p.label}（${hold === "unplannable" ? unplannable : HOLD_WORDS[hold]}）` : p.label),
    off: hold !== null,
    why: hold === "busy" ? "这张节点图已经有一个任务在算：等它算完，或在顶栏「队列」旁取消"
      : hold === "submitting" ? "正在上传素材、提交任务：等它进队列（顶栏「队列」会显示），不用再点"
      : hold === "paused" ? "现在不能提交计算（计算任务关着，或存储配额满了）：看顶栏的提示"
      : hold === "unplannable" ? unplannable : "",
    run: () => cookNode(nodeId), // 视图随之显示这个节点：「在视图里点选」看的是视图显示谁，自然就不亮了（state/viewPicking.ts）
  };
};

const useDownload: ActionHook = (nodeId, p) => {
  const graphId = useCookInputs((s) => s.graphId);
  const output = useResults((s) => s.outputs[outputKey(graphId, nodeId)]);
  const cooking = useCooking(nodeId);
  const ready = !!output && !output.gone;
  return {
    text: cooking !== undefined ? `打包中 · ${cooking === null ? "…" : percent(cooking)}` : ready ? `${p.label} · ${sizeText(output.bytes)}` : output?.gone ? `${p.label}（包已清理，再计算）` : `${p.label}（先计算）`,
    off: !ready,
    why: output?.gone ? "这个任务已经过了保留天数，服务器上删掉了：再「计算」一次这个「输出」" : ready ? "" : "还没有打好的包：先「计算」这个「输出」（收集接进来的结果、打成 zip）",
    run: () => output && downloadOutput(output),
  };
};

/** 「在视图里点选」（服务端 nodes/params.py pick_button，每个 picks / canvas 参数自动一个，`target` 是那个参数）。亮不亮
 * 只看事实（state/viewPicking.ts pickingOn）：视图正显示这个节点、且它这个参数的手柄可用。没亮时点 = 让视图显示它
 * （同双击，并记下之前显示的）；亮着点 = 回到进入前显示的（没有记录就不动）。目标参数现在不适用（节点规则置灰，例如
 * 「选人」不是点选方式）时不能点。 */
const usePickInView: ActionHook = (nodeId, p) => {
  const param = p.target ?? "";
  const displayId = useLook((s) => s.displayId);
  const snap = useGraphSnapshot();
  const handleActive = handlesOf(snap, nodeId).some((h) => Object.values(h.params).includes(param));
  const on = pickingOn(displayId, nodeId, handleActive);
  const back = on && canLeave(displayId, nodeId); // 双击节点进来的没有「进入前」可回：只说正在点选，不写「点这里结束」
  const inactive = useResults((s) => why(s.results[nodeId]?.applies, param));
  return {
    text: on ? (back ? "正在视图里点选…（点这里结束）" : "正在视图里点选…") : inactive ? `${p.label}（现在用不上）` : p.label,
    off: !on && !!inactive,
    why: inactive ?? "",
    on,
    run: () => (on ? leavePicking(nodeId) : enterPicking(nodeId)),
  };
};

const ACTIONS: Record<string, ActionHook> = { cook: useCook, download: useDownload, pick_in_view: usePickInView };

const useNothing: ActionHook = (_nodeId, p) => ({ text: p.label, off: true, why: "这个页面还不认识这个按钮：刷新页面试试", run: () => {} });

/** 一个按钮参数的控件：参数面板里占满参数列（`mini`：节点上的小号）。 */
export function ButtonParam({ nodeId, p, mini }: { nodeId: string; p: ParamDef; mini?: boolean }) {
  const now = (ACTIONS[p.action ?? ""] ?? useNothing)(nodeId, p);
  return (
    <span className={`pbutton${mini ? " mini" : ""}`}>
      <Button size={mini ? "xs" : "sm"} tone={now.on ? "primary" : mini ? "ghost" : "default"} on={now.on} layout="pbutton-btn" disabled={now.off} tip={now.why || undefined}
        onClick={(e) => (e.stopPropagation(), now.run())}>
        {now.text}
      </Button>
    </span>
  );
}
