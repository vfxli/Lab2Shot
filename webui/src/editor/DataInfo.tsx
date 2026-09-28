import { useEffect, useState } from "react";
import type { Manifest } from "../api";
import { useGraphSnapshot } from "../graph/snapshot";
import { nodeMessages } from "../graph/nodes";
import { fromServer, msg } from "../messages/message";
import { MessageRow } from "../ui/MessageRow";
import { blocksOf, chainOf } from "../state/items";
import type { GNode } from "../state/graph";
import { manifestOf } from "../transfer/frames";
import { CopyToNuke } from "../ui/CopyToNuke";
import { ItemsList } from "../ui/ItemsList";
import { MessageText } from "../ui/MessageText";

/** 数据信息 (the card the mark at a node's bottom right opens): what each of a node's ports holds, i.e. the one summary
 * the server makes from the data type's own declaration (lab2shot/data/summary.py), the same answer the port's
 * tooltip and 「取信息」 read. The page writes none of it. */

// A result's description is requested once per fingerprint and kept by the page's one cache (transfer/frames.ts
// manifestOf, the same one the 2D stage and the timeline read): a result never changes under its address.

/** What every port of this node holds now, by fingerprint (missing: nothing cooked there yet). */
function useManifests(fps: string[]): Record<string, Manifest> {
  const key = fps.join("|");
  const [got, set] = useState<{ key: string; by: Record<string, Manifest> }>({ key: "", by: {} });
  useEffect(() => {
    let live = true;
    const want = key ? key.split("|") : [];
    void Promise.all(want.map((fp) => manifestOf(fp).catch(() => null))).then(
      (all) => live && set({ key, by: Object.fromEntries(want.map((fp, i) => [fp, all[i]]).filter(([, m]) => m)) as Record<string, Manifest> }),
    );
    return () => {
      live = false;
    };
  }, [key]);
  return got.key === key ? got.by : {};
}

/** 说明端口为何没有内容：「未被连出因此未计算」与「尚未计算」是两种情况（与视图下方控件遵循同一规则，
 * editor/ValuesView.tsx）。没有连线引出的端口不会被写入（按需输出，engine/scopes.py），显示「还没有算」会使使用者一直等待
 * 一个不会出现的数值。判据与该处相同：节点已计算（`cached`）而该端口不在 `present` 中，即表示无人需要。 */
const whyNothing = (status: { cached?: boolean; present?: string[] } | undefined, r: { side: string; name: string }) =>
  // 传递消息对象而非其文字：编号通过 `data-code` 写到元素上，截图即可确定是哪一条消息
  // （a message's `.text` is for a tooltip; anywhere else the message itself goes）
  r.side === "输出" && status?.cached && !(status.present ?? []).includes(r.name)
    ? msg("I-VALUE-NOTASKED")
    : msg("I-VALUE-NOTCOOKED");

/** The ports of a node as 数据信息 lists them: which side, what the port is called, the result it holds now and what
 * the server says that result is. Where a wire feeds an input, its own result is the upstream output's. */
function portRows(snap: ReturnType<typeof useGraphSnapshot>, id: string): { side: string; name: string; label: string; tip: string; fp: string }[] {
  const status = snap.results[id];
  const ports = status?.ports;
  const rows: { side: string; name: string; label: string; tip: string; fp: string }[] = [];
  // a fingerprint is the address a result will have; `present` says which of them a result is actually at
  // (server/packets.py status), so a port that has not been cooked is never asked for and never says 读取…
  const at = (s: typeof status, port: string) => ((s?.present ?? []).includes(port) ? s?.outputs?.[port] ?? "" : "");
  for (const p of ports?.inputs ?? []) {
    const e = snap.edges.find((x) => x.target === id && x.targetHandle === p.name);
    rows.push({ side: "输入", name: p.name, label: p.label, tip: p.tip ?? "", fp: e ? at(snap.results[e.source], e.sourceHandle ?? "") : "" });
  }
  for (const p of ports?.outputs ?? []) rows.push({ side: "输出", name: p.name, label: p.label, tip: p.tip ?? "", fp: at(status, p.name) });
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
  const snap = useGraphSnapshot();
  const rows = portRows(snap, node.id);
  const by = useManifests([...new Set(rows.map((r) => r.fp).filter(Boolean))]);
  // inside a 逐项处理 block, the status reply's fields are the item the view is on and `summary` is how all of them
  // stand: that is what says to list them (engine/scopes.py), never a guess here
  const status = snap.results[node.id];
  const scope = chainOf(blocksOf(snap.reply), node.id).at(-1); // the innermost block it is in (state/items.ts)
  const notices = nodeMessages(snap.results, node.id);
  if (!rows.length && !notices.length) return null;
  return (
    <div className="group">
      <div className="group-title">数据</div>
      {["输入", "输出"].map((side) => {
        const mine = rows.filter((r) => r.side === side);
        if (!mine.length) return null;
        return (
          <div className="data-side" key={side}>
            <span className="data-side-name">{side}</span>
            <div className="data-ports">
              {mine.map((r) => {
                const got = by[r.fp];
                const lines = got?.summary?.items ?? [];
                // only a result whose own meta declares one offers 复制到 Nuke
                const clip = got?.meta?.clipboard as { app: string; file: string } | undefined;
                return (
                  <div className="data-port" key={`${side}.${r.name}`}>
                    <span className="data-port-name" data-tip={r.tip}>{r.label}</span>
                    {/* 每项一行，名称在前：服务器将名称与值分开提供（`lab2shot/data/summary.py _line` 的 `label` 与
                        `text`），若将值以「·」串成一长条，读者无法分辨各数值的含义。
                        没有名称的条目（完整句子）整句占一行。 */}
                    <span className="data-port-said tnum" data-tip={[r.tip, ...lines.map((l) => l.said)].filter(Boolean).join("\n")}>
                      {lines.length ? (
                        lines.map((l, k) => (
                          <span className="said-item" key={`${l.id}.${k}`}>
                            {l.label && <span className="said-name">{l.label}</span>}
                            <span className="said-value">{l.text}</span>
                          </span>
                        ))
                      ) : got?.summary?.known === "empty" ? "这次没有内容" : r.fp ? "读取…" : <MessageText message={whyNothing(status, r)} />}
                    </span>
                    {clip && <CopyToNuke fp={r.fp} app={clip.app} what={node.data.label} size="xs" />}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
      {status?.summary && snap.reply && (
        <div className="data-side">
          <span className="data-side-name">条目</span>
          <ItemsList graph={snap.reply.graph} node={node.id} total={status.summary.total} begin={scope?.begin} />
        </div>
      )}
      {notices.length > 0 && (
        <div className="data-side">
          <span className="data-side-name">提醒</span>
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


