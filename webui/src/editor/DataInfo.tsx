import { useMemo } from "react";
import { packetOf } from "../state/results";
import type { Manifest } from "../api";
import { useGraphDoc } from "../graph/snapshot";
import { nodeMessages } from "../graph/nodes";
import { fromServer, msg } from "../messages/message";
import { MessageRow } from "../ui/MessageRow";
import { blocksOf, chainOf } from "../state/items";
import type { GNode } from "../state/graph";
import { useDescribed } from "../transfer/described";
import { CopyToNuke } from "../ui/CopyToNuke";
import { ItemsList } from "../ui/ItemsList";
import { MessageText } from "../ui/MessageText";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** Owns the contents of 数据信息 (the card the mark at a node's bottom right opens): what each of a node's ports holds, i.e. the one summary
 * the server makes from the data type's own declaration (lab2shot/data/summary.py), the same answer the port's
 * tooltip and 「取信息」 read. The page writes none of it. */

/** What every port of this node holds now, by fingerprint (missing: nothing cooked there yet). The description is the
 * one the 2D stage and the timeline read (transfer/described.ts), for the packet's current generation. */
function useManifests(fps: string[]): Record<string, Manifest> {
  const got = useDescribed<Manifest>("manifest", fps);
  return useMemo(() => Object.fromEntries(fps.flatMap((fp, i) => (got[i] ? [[fp, got[i]!]] : []))), [got]); // eslint-disable-line react-hooks/exhaustive-deps -- got pairs one-to-one with fps
}

/** Why a port holds nothing: 「not wired onward, so not cooked」 and 「not cooked yet」 are two different cases (the same
 * rule as the controls under the view, editor/ValuesView.tsx). A port with no wire leading out is never written
 * (outputs on demand, engine/scopes.py); saying 「还没有算」 there would have the user wait for a value that never comes.
 * The test is the same as there: the node is cooked (`cached`) and the port is not in `present`, i.e. nobody needs it. */
const whyNothing = (status: { cached?: boolean; present?: string[]; outputs?: Record<string, string> } | undefined, r: { side: string; name: string }) =>
  // the message object, not its text: its code goes onto the element as `data-code`, so a screenshot tells which
  // message it is (a message's `.text` is for a tooltip; anywhere else the message itself goes)
  r.side === "out" && status?.cached && !packetOf(status, r.name)
    ? msg("I-VALUE-NOTASKED")
    : msg("I-VALUE-NOTCOOKED");

/** The sides' names (keys of the words). */
const SIDE_KEY = { in: "ui.view.inputs", out: "ui.view.outputs" } as const;

/** The ports of a node as 数据信息 lists them: which side, what the port is called, the result it holds now and what
 * the server says that result is. Where a wire feeds an input, its own result is the upstream output's. */
function portRows(snap: ReturnType<typeof useGraphDoc>, id: string): { side: string; name: string; label: string; tip: string; fp: string }[] {
  const status = snap.shown[id];
  const ports = status?.ports;
  const rows: { side: string; name: string; label: string; tip: string; fp: string }[] = [];
  // a fingerprint is the address a result will have; `present` says which of them a result is actually at
  // (server/packets.py status), so a port that has not been cooked is never asked for and never says 读取…
  const at = (s: typeof status, port: string) => packetOf(s, port) ?? "";
  for (const p of ports?.inputs ?? []) {
    const e = snap.edges.find((x) => x.target === id && x.targetHandle === p.name);
    rows.push({ side: "in", name: p.name, label: p.label, tip: p.tip ?? "", fp: e ? at(snap.shown[e.source], e.sourceHandle ?? "") : "" });
  }
  for (const p of ports?.outputs ?? []) rows.push({ side: "out", name: p.name, label: p.label, tip: p.tip ?? "", fp: at(status, p.name) });
  return rows;
}

/** 数据信息 (the card beside the node, editor/NodeInfoCard.tsx): every input and every output of this node, one row each, with what the
 * data actually is: the one summary the server makes from the data type's declaration (data/summary.py), the same
 * answer the port's own tooltip and 「取信息」 read. Nothing here is written by the page.
 *
 * A node inside a 逐项处理 block lists every item under its ports as well: that list is ui/ItemsList.tsx; the page
 * never requests the items itself.
 *
 * 提醒 is the last part: everything this node has to say right now, in full. The node's own footer has room for a few
 * words only, so the sentences live here, in the site's one message row (ui/MessageRow.tsx) with
 * its level square and its code. The information card beside the node renders this component from the status
 * reply: one place, one list, nothing requested twice. */
export function DataGroup({ node }: { node: GNode }) {
  const snap = useGraphDoc();
  const rows = portRows(snap, node.id);
  const by = useManifests([...new Set(rows.map((r) => r.fp).filter(Boolean))]);
  // inside a 逐项处理 block, the status reply's fields are the item the view is on and `summary` is how all of them
  // stand: that is what says to list them (engine/scopes.py), never a guess here
  const status = snap.shown[node.id];
  const scope = chainOf(blocksOf(snap.reply), node.id).at(-1); // the innermost block it is in (state/items.ts)
  const notices = nodeMessages(snap.shown, node.id);
  if (!rows.length && !notices.length) return null;
  return (
    <div className="group">
      <div className="group-title">{t("ui.view.data")}</div>
      {(["in", "out"] as const).map((side) => {
        const mine = rows.filter((r) => r.side === side);
        if (!mine.length) return null;
        return (
          <div className="data-side" key={side}>
            <span className="data-side-name">{t(SIDE_KEY[side])}</span>
            <div className="data-ports">
              {mine.map((r) => {
                const got = by[r.fp];
                const lines = got?.summary?.items ?? [];
                // only a result whose own meta declares one offers 复制到 Nuke
                const clip = got?.meta?.clipboard as { app: string; file: string } | undefined;
                return (
                  <div className="data-port" key={`${side}.${r.name}`}>
                    <span className="data-port-name">{r.label}</span>
                    {/* one line per entry, name first: the server gives name and value apart (`label` and `text` of
                        `lab2shot/data/summary.py _line`); values strung into one long line with 「·」 would leave the
                        reader unable to tell which number means what. An entry without a name (a whole sentence)
                        takes a line of its own. */}
                    <span className="data-port-said tnum" {...tipAttrs(tipOf("value", [r.tip, ...lines.map((l) => l.said)].filter(Boolean).join("\n")))}>
                      {lines.length ? (
                        lines.map((l, k) => (
                          <span className="said-item" key={`${l.id}.${k}`}>
                            {l.label && <span className="said-name">{l.label}</span>}
                            <span className="said-value">{l.text}</span>
                          </span>
                        ))
                      ) : got?.summary?.known === "empty" ? t("ui.view.empty_this_time") : r.fp ? t("ui.view.reading") : <MessageText message={whyNothing(status, r)} />}
                    </span>
                    {clip && <CopyToNuke fp={r.fp} app={clip.app} what={node.id} size="xs" />}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      {status?.summary && snap.reply && (
        <div className="data-side">
          <span className="data-side-name">{t("ui.view.items")}</span>
          <ItemsList graph={snap.reply.graph} node={node.id} total={status.summary.total} begin={scope?.begin} />
        </div>
      )}
      {notices.length > 0 && (
        <div className="data-side">
          <span className="data-side-name">{t("ui.view.notices")}</span>
          <div className="data-notes">
            {notices.map((m) => (
              <MessageRow key={m.code + m.text + (m.port ?? "")} message={fromServer(m)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


