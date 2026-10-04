import "./gpus.css";
import { useState } from "react";
import type { GpuFitItem, GpuFitState } from "../api";
import { Chip } from "./Button";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

// whether an extension fits is judged only from the GPU architectures its package declares and the scan made at
// install time; there is no "actually tried it" grade
// (label, tip: keys, read with t() as they are shown)
const FIT_GROUPS: { state: GpuFitState; label: string; tip?: string }[] = [
  { state: "assumed", label: "ui.misc.fit_assumed" },
  { state: "refused", label: "ui.misc.fit_refused", tip: "ui.misc.fit_refused_tip" },
  { state: "unknown", label: "ui.misc.fit_unknown", tip: "ui.misc.fit_unknown_tip" },
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
          <Chip key={g.state} layout={`q-fit-${g.state}`} tip={g.tip ? tipOf("disabled", t(g.tip)) : undefined} on={open === g.state} onClick={() => setOpen(open === g.state ? null : g.state)}>
            {t(g.label)} {fits[g.state].length}
          </Chip>
        ))}
      </div>
      {open && (
        <ul className="q-gpu-fit-list">
          {fits[open].map((item) => (
            <li key={item.extension} {...tipAttrs(tipOf("error", t("ui.misc.named", { label: item.message.code, text: item.message.text })))}>
              <span>{item.title}</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
