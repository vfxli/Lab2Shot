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
import { enterPicking, leavePicking, pickingOn } from "../state/viewPicking";
import { useHandleView } from "../state/handleView";
import { why } from "../api/applies";
import { useLook } from "../state/look";
import { useGraphDoc } from "../graph/snapshot";
import { handlesOf } from "../graph/rules";
import { hasPicture } from "../view/plan";
import { useLocalPicture } from "../view/localPick";
import { t, useT } from "../i18n/t";
import { textOf } from "../messages/message";
import { tipOf } from "../platform/tips";

interface ButtonNow {
  text: string; // 按钮上的字
  off: boolean; // 现在点不了：按钮上只带一个短词（一行、不比按钮本身长），完整原因在 `why`
  why?: string; // 点不了时的完整原因：参数面板里是按钮的悬停提示（节点上不出悬停提示，同一句在参数面板的警告里）
  // 服务器说现在算不了的原句：参数面板和应用模式里直接写在按钮下面（按钮上就只写名字）；节点上的小号按钮照旧只带短词
  below?: string;
  run: () => void;
  on?: boolean; // 一个开关状态的按钮现在开着（「在视图里点选」正在点选）：按钮高亮
}

type ActionHook = (nodeId: string, p: ParamDef) => ButtonNow;

/** 「计算」点不了时压进按钮文字的原因（一行、高度不变）。 */
const HOLD_WORDS: Record<Exclude<CookHold, "unplannable" | null>, string> = { submitting: "ui.params.button.submitting", busy: "ui.params.button.busy", paused: "ui.params.button.paused" };

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
  return hit ? t(done!.state === "done" ? "ui.params.button.done" : done!.state === "cancelled" ? "ui.params.button.cancelled" : "ui.params.button.failed") : null;
}

const useCook: ActionHook = (nodeId, p) => {
  // 点不点得了、为什么：与 cook() 的闩同一处（graph/actions.ts cookHold）
  const version = useCookInputs((s) => s.version); // 预估只在对上当前编辑时算数（cookHold）
  const port = useLook((s) => s.displayPort); // 预估也要对上看的口
  const hold = useResults((s) => cookHold(s, { node: nodeId, version, port }));
  const unplannable = useResults((s) => textOf(planError(s, { node: nodeId, version, port }))); // 服务器说算不了的原因（应用模式按 .app 说）
  const mine = useResults((s) => s.submitting === nodeId); // 正在提交的就是这个节点
  const justDone = useJustDone(nodeId);
  const cooking = useCooking(nodeId);
  return {
    // 服务器说算不了的原句可能很长（「……还没有选蒙皮角色：选一个蒙皮角色后，接到……的线就接上了」）：节点上的按钮只说
    // 「还不能算」；参数面板、应用模式里原句直接写在按钮下面（below）
    text: cooking !== undefined ? (cooking === null ? t("ui.params.button.cooking") : t("ui.params.button.cooking_at", { percent: percent(cooking) })) : (mine ? t("ui.params.button.uploading") : justDone ? `${p.label} · ${justDone}` : hold ? t("ui.params.button.held", { label: p.label, why: t(hold === "unplannable" ? "ui.params.button.unplannable" : HOLD_WORDS[hold]) }) : p.label),
    off: hold !== null,
    why: hold === "unplannable" ? unplannable : undefined,
    below: hold === "unplannable" && unplannable ? unplannable : undefined,
    run: () => cookNode(nodeId), // 视图随之显示这个节点：「在视图里点选」看的是视图显示谁，自然就不亮了（state/viewPicking.ts）
  };
};

const useDownload: ActionHook = (nodeId, p) => {
  const graphId = useCookInputs((s) => s.graphId);
  const output = useResults((s) => s.outputs[outputKey(graphId, nodeId)]);
  const cooking = useCooking(nodeId);
  const ready = !!output && !output.gone;
  return {
    text: cooking !== undefined ? t("ui.params.button.packing", { percent: cooking === null ? "…" : percent(cooking) }) : ready ? `${p.label} · ${sizeText(output.bytes)}` : t(output?.gone ? "ui.params.button.package_gone" : "ui.params.button.cook_first", { label: p.label }),
    off: !ready,
    run: () => output && downloadOutput(output),
  };
};

/** 「在视图里点选」（服务端 nodes/params.py pick_button，2D 手柄所改的每个参数——点选、轮廓、火柴人，nodes/handles.py PICKED_IN_VIEW——自动一个，`target` 是那个参数）。亮不亮
 * 只看事实（state/viewPicking.ts pickingOn）：视图正显示这个节点、且它这个参数的手柄可用。没亮时点 = 让视图显示它
 * （同双击，并记下之前显示的）；亮着点 = 回到进入前显示的（没有记录就不动）。目标参数现在不适用（节点规则置灰，例如
 * 「选人」不是点选方式）时不能点。 */
const usePickInView: ActionHook = (nodeId, p) => {
  const param = p.target ?? "";
  const displayId = useLook((s) => s.displayId);
  const snap = useGraphDoc();
  const mine = handlesOf(snap, nodeId).filter((h) => Object.values(h.params).includes(param));
  const handleActive = mine.length > 0;
  const picking = useHandleView((s) => s.picking);
  const on = pickingOn(displayId, nodeId, handleActive, picking);
  const inactive = useResults((s) => why(s.results[nodeId]?.applies, param));
  // 还没有画面（没选素材）：点了也没东西可点，置灰并在按钮下写原因。画面是视图会画的那一份：本机选的文件（还没传也算，
  // view/localPick.ts），或服务器上已有的上游画面
  const local = useLocalPicture(nodeId);
  const blank = !on && !inactive && !local && !hasPicture(snap, nodeId, mine.flatMap((h) => (h.source ? [h.source] : [])));
  return {
    // 开着：蓝色、写「退出编辑」，再点就退出（同视图工具栏的「退出编辑」，state/viewPicking.ts exitViewOperation）
    text: on ? t("ui.params.edit_exit") : inactive ? t("ui.params.button.inactive", { label: p.label }) : p.label,
    off: !on && (!!inactive || blank),
    why: blank ? t("ui.params.button.no_picture") : undefined,
    below: blank ? t("ui.params.button.no_picture") : undefined,
    on,
    run: () => (on ? leavePicking(nodeId) : enterPicking(nodeId)),
  };
};

const ACTIONS: Record<string, ActionHook> = { cook: useCook, download: useDownload, pick_in_view: usePickInView };

const useNothing: ActionHook = (_nodeId, p) => ({ text: p.label, off: true, run: () => {} });

/** 一个按钮参数的控件：参数面板里占满参数列（`mini`：节点上的小号）。 */
export function ButtonParam({ nodeId, p, mini }: { nodeId: string; p: ParamDef; mini?: boolean }) {
  useT(); // also drawn inside a memoised graph node: its words follow the language by themselves
  const now = (ACTIONS[p.action ?? ""] ?? useNothing)(nodeId, p);
  const said = !mini && now.below; // the reason is written under the button: the button says its name only, no tip
  return (
    <span className={`pbutton${mini ? " mini" : ""}`}>
      <Button size={mini ? "xs" : "sm"} tone={now.on ? "primary" : mini ? "ghost" : "default"} on={now.on} layout="pbutton-btn" disabled={now.off}
        tip={now.off && now.why && !said ? tipOf("disabled", now.why) : undefined}
        onClick={(e) => (e.stopPropagation(), now.run())}>
        {said ? p.label : now.text}
      </Button>
      {said && <span className="pwhy pbutton-why">{now.below}</span>}
    </span>
  );
}
