import type { Output } from "../api/files";
import { downloadOutput } from "../files/outputs";
import { agoText, sizeText } from "../platform/format";
import { Button } from "./Button";
import { t } from "../i18n/t";
import { tipOf, type Tip } from "../platform/tips";

/** 下载 in the queue (ui/Queue.tsx): the browser's own download of the zip an 「输出」's cook packed (files/outputs.ts).
 * On the graph (the node's body, the parameter panel) it is the 「输出」's button parameter 「下载」 instead
 * (editor/buttonActions.tsx download), one control for it there. It can be clicked only once there is a zip:
 * computing the nodes before it only puts their results on the server; computing the 「输出」 itself collects and
 * packs them, then this is enabled. An output gone with its task (任务保留天数) is greyed and says so. */

export function outputTip(o: Output): Tip {
  const facts = o.finished
    ? t("ui.misc.output_facts_when", { size: sizeText(o.bytes), count: o.count, when: agoText(o.finished) })
    : t("ui.misc.output_facts", { size: sizeText(o.bytes), count: o.count });
  return { why: "value", text: `${o.name}\n${facts}` };
}

export function OutputDownload({ output, size = "xs" }: { output: Output | undefined | null; size?: "xs" | "sm" }) {
  const ready = !!output && !output.gone;
  return (
    <Button tip={!output ? tipOf("disabled", t("ui.misc.output_none")) : output.gone ? tipOf("disabled", t("ui.misc.output_gone")) : outputTip(output)} tone="ghost" size={size} disabled={!ready}
            onClick={(e) => (e.stopPropagation(), ready && downloadOutput(output))}>
      {ready ? t("ui.misc.download_size", { size: sizeText(output.bytes) }) : t("ui.common.download")}
    </Button>
  );
}
