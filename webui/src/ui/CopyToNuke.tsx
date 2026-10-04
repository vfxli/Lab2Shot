import { useState } from "react";
import { api, type NodeTypeDef } from "../api";
import { packetOf, useResults } from "../state/results";
import { greyed } from "../api/applies";
import { fromServer, msg, reasonOf, say } from "../state/say";
import { Button } from "./Button";
import { t } from "../i18n/t";

/** 复制到 Nuke: an output-settings node that writes text another application can paste declares it
 * (nodes/output.py OutputSettings.clipboard), the server returns the text it wrote (GET /api/packet/{fp}/clipboard),
 * and this component puts it on the clipboard. The core has no knowledge of Nuke and the page writes none of the text:
 * the application's name and the file come from the result itself.
 *
 * It is shown only for a result whose meta declares a clipboard, so a node that writes nothing pasteable never shows a
 * button that does nothing. It appears in 数据信息 and on the node: one component in both places. */
export function CopyToNuke({ fp, app, what, size = "sm", off = false }: {
  fp: string;
  app: string; // the application that reads it, from the result's meta ("nuke")
  what: string; // what is being copied, for the notice ("Camera3", the node's name)
  size?: "sm" | "xs" | "xxs";
  off?: boolean; // it cannot be pressed now: the button stays in place, greyed
}) {
  const [busy, setBusy] = useState(false);
  const copy = async () => {
    setBusy(true);
    try {
      const got = await api.clipboard(fp);
      await navigator.clipboard.writeText(got.text);
      say(msg("N-CLIPBOARD-COPIED", { app: APPS[got.app] ?? got.app, what }));
      // what that application cannot represent of this result, as stated by the result itself (a lens on an OpenCV
      // model: 「畸变系数没有跟着走」). Shown only here, at the moment of copying, never on every cook
      if (got.said) say(fromServer(got.said));
    } catch (e) {
      say(msg("E-CLIPBOARD-COPYFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Button
      size={size}
      tone="ghost"
      disabled={busy || !!off}
      onClick={() => void copy()}
    >
      {t("ui.misc.copy_to", { app: APPS[app] ?? app })}
    </Button>
  );
}

/** How each application's name is written on a button. A result names its own application; this only formats it. */
const APPS: Record<string, string> = { nuke: "Nuke" };

/** 复制到 Nuke on the node's own body, where artists expect it (数据信息 alone is one click too far).
 * Which result holds the text is declared by the node type (nodes/clipboard.py Pasteable): `clipboard_port`, else its
 * main result, e.g. an output-settings node's 文件, 「LensDistortion」's 去畸变 ST-map, 「AnyCalib 镜头标定」's
 * 镜头模型. It is resolved here rather than in the node's layout: the page never guesses a port. */
export function NodeCopyToNuke({ node, def, what }: { node: string; def: NodeTypeDef; what: string }) {
  const port = def.clipboard ? (def.clipboard_port || def.main) : "";
  const fp = useResults((s) => (port ? packetOf(s.results[node], port) ?? undefined : undefined));
  const answer = useResults((s) => s.results[node]?.applies);
  if (!def.clipboard) return null;
  // The button never comes and goes; it is only usable or not. It is unusable in two cases: under the current settings
  // the node writes no Nuke data (the server computes it from Pasteable.clipboard_when, subject id "clipboard"), or no
  // result has been cooked yet. It stays in place, greyed
  const off = greyed(answer, "clipboard") || !fp;
  // xxs: the node's bottom row is 16 px high and cannot hold a 20 px xs
  return <CopyToNuke fp={fp ?? ""} app={def.clipboard} what={what} size="xxs" off={off} />;
}
