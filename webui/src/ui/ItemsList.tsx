import { useEffect, useState } from "react";
import { api, type ItemStatus, type NodeStatus } from "../api";
import { render } from "../messages/format";
import { Button } from "./Button";
import { Empty } from "./Empty";
import { Table, type Column } from "./Table";
import { useItems } from "../state/items";
import { showItem } from "../graph/actions";
import "./items.css";
import { tipAttrs, tipOf } from "../platform/tips";

/** Item by item for one node inside a 逐项处理 block: the status reply
 * answers per node — a summary and the item the view is on — and the rest is this one call,
 * `GET /api/status/{graph}/node/{node}/items?offset&limit`, a page at a time (50, at most 200).
 *
 * The part the 数据信息 panel puts under a node inside a block: give it the graph key the status reply came
 * with (`StatusReply.graph`), the node's id and how many items there are (`NodeStatus.summary.total`); a click on a row
 * puts the view on that item, the same 当前条目 the item bar changes (state/items.ts). */

const ITEMS_PAGE = 50; // server/packets.py ITEMS_PAGE / ITEMS_MOST
const ITEMS_MOST = 200;

/** Each state's word, by message code: said when a row renders (in the page's language then), never worked out once
 * when the module loads. */
const STATE_CODE: Record<string, string> = {
  cached: "I-ITEMS-CACHED",
  todo: "I-ITEMS-TODO",
  pending: "I-ITEMS-PENDING",
  failed: "I-ITEMS-FAILED",
  skipped: "I-ITEMS-SKIPPED",
  unused: "I-ITEMS-UNUSED",
  error: "I-ITEMS-ERROR",
};

const why = (row: ItemStatus): string => row.error?.text ?? row.skipped?.text ?? row.messages?.[0]?.text ?? "";

export function ItemsList({ graph, node, total, begin }: {
  graph: string; // the version of the graph the status reply answered for (StatusReply.graph)
  node: string;
  total: number; // how many items there are (NodeStatus.summary.total)
  begin?: string; // the block's begin: a click on a row makes that item the current one
}) {
  const [rows, setRows] = useState<ItemStatus[]>([]);
  const [asked, setAsked] = useState(0);
  const view = useItems((s) => (begin ? s.view[begin] : undefined));
  useEffect(() => {
    let alive = true;
    void api.nodeItems(graph, node, asked, ITEMS_PAGE).then(
      (got) => alive && setRows((had) => (asked ? [...had, ...got.items] : got.items)),
      () => alive && setRows([]),
    );
    return () => {
      alive = false;
    };
  }, [graph, node, asked]);
  useEffect(() => setAsked(0), [graph, node]);

  const columns: Column<ItemStatus>[] = [
    { id: "name", label: render("I-ITEMS-NAME"), className: "items-row-name",
      cell: (r) => <span data-user-data {...tipAttrs(tipOf("truncated", r.item.names.join(" / ")))}>{r.item.names.join(" / ")}</span> },
    { id: "state", label: render("I-ITEMS-STATE"),
      cell: (r) => <span className={`items-state ${r.state ?? ""}`}>{STATE_CODE[r.state ?? ""] ? render(STATE_CODE[r.state ?? ""]) : ""}</span> },
    { id: "why", label: render("I-ITEMS-WHY"),
      cell: (r) => <span data-user-data {...tipAttrs(tipOf("truncated", why(r)))}>{why(r)}</span> },
  ];
  if (!total) return <Empty title={render("I-ITEMS-NONE")} />;
  return (
    <div className="items-list">
      <div className="items-list-head">{render("I-ITEMS-TITLE", { total })}</div>
      <Table
        rows={rows}
        columns={columns}
        rowKey={(r) => r.item.path.join("/")}
        picked={view && rows.length ? rows.find((r) => r.item.path.at(-1) === view)?.item.path.join("/") ?? null : null}
        onPick={begin ? (r) => showItem(begin, r.item.path.at(-1) ?? "") : undefined}
      />
      {rows.length < total && rows.length < ITEMS_MOST && (
        <Button onClick={() => setAsked(rows.length)}>
          {render("I-ITEMS-MORE", { count: Math.min(ITEMS_PAGE, total - rows.length) })}
        </Button>
      )}
    </div>
  );
}

export type { ItemStatus, NodeStatus };
