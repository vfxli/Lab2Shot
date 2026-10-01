import "./gpus.css";
import { useState } from "react";
import type { GpuFitItem, GpuFitState } from "../api";
import { Chip } from "./Button";

// whether an extension fits is judged only from the GPU architectures its package declares and the scan made at
// install time; there is no "actually tried it" grade
const FIT_GROUPS: { state: GpuFitState; label: string; tip: string }[] = [
  { state: "assumed", label: "能跑", tip: "扩展包声明支持这一代显卡，环境里编译的显卡程序也对得上：这张卡接它的任务" },
  { state: "refused", label: "不能跑", tip: "扩展包没声明这一代显卡，或环境里没有这一代显卡的程序：这张卡不接它的任务" },
  { state: "unknown", label: "未知", tip: "扩展包环境没能探测出支持哪些显卡：重装这个扩展包" },
];

/** One card's GPU extensions by how their fit is known (assumed / refused / unknown): a count per group, opened into
 * the extensions with each one's reason (message code and text) on hover. */
export function GpuFits({ fits }: { fits: Record<GpuFitState, GpuFitItem[]> }) {
  const [open, setOpen] = useState<GpuFitState | null>(null);
  const groups = FIT_GROUPS.filter((g) => fits[g.state].length > 0);
  if (!groups.length) return null;
  return (
    <>
      <div className="q-gpu-fits">
        {groups.map((g) => (
          <Chip key={g.state} layout={`q-fit-${g.state}`} tip={g.tip} on={open === g.state} onClick={() => setOpen(open === g.state ? null : g.state)}>
            {g.label} {fits[g.state].length}
          </Chip>
        ))}
      </div>
      {open && (
        <ul className="q-gpu-fit-list">
          {fits[open].map((item) => (
            <li key={item.extension} data-tip={`${item.message.code}：${item.message.text}`}>
              <span>{item.title}</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
