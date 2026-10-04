import { cookNode, mergeToExr, setComment } from "../graph/actions";
import { useLook, type NodeComment } from "../state/look";
import { useViewer } from "../state/viewer";
import { t } from "../i18n/t";
import { tipOf, type Tip } from "../platform/tips";

/** The node shown in the parameter panel, then a field of its head focused there (its name, its comment: editor/NodeNaming.tsx). */
function focusInPanel(id: string, selector: string): void {
  useViewer.setState({ selectedId: id });
  requestAnimationFrame(() => requestAnimationFrame(() => document.querySelector<HTMLElement>(selector)?.focus()));
}

/** What a node's right-click menu offers, as one table. A row says when it is there, what it reads, and what a click does; the menu
 * component (editor/FlowParts.tsx NodeMenu) only draws the rows the table gives it, so a new action is a row here. */

export interface NodeMenuFacts {
  id: string; // the node right-clicked
  typeId: string;
  delivers: boolean; // cooking it collects and packs files for download (the server's policy): an 「输出」
  busy: boolean; // this graph already has a job in the queue
  blocked: boolean; // 计算任务 switched off (state/pause.ts), the storage quota full (state/quota.ts), or the server says this node cannot be cooked now (graph/actions.ts cookHold "unplannable"): each greys 「计算」 out, the reason is in cookTip
  cookTip: Tip | undefined; // why it cannot be cooked now, or that it computes nothing (graph/rules.ts cookWords)
  // a node inside a 逐项处理 block (engine/scopes.py, the status reply's `summary`): how many items it has. 0 outside
  // every block — then the cook rows say nothing about items
  items: number;
  mergeIds: string[]; // 序列图输出设置 nodes this click would merge (none: the row is not there)
  comment?: NodeComment; // its comment (state/look.ts), if it has one
  editable: boolean; // the document may be changed here (not a read-only tab)
}

interface NodeMenuItem {
  key: string;
  when: (f: NodeMenuFacts) => boolean;
  label: (f: NodeMenuFacts) => string;
  desc: (f: NodeMenuFacts) => string; // the grey word on the right: a shortcut, a gesture, a count
  tip?: (f: NodeMenuFacts) => Tip | undefined; // only what the row does not show (platform/tips.ts): why it is off
  off?: (f: NodeMenuFacts) => boolean; // there, but not clickable now (the tip says why)
  run: (f: NodeMenuFacts) => void;
}

/** The keys that cook the node shown, as the one registry has them (editor/App.tsx useShortcut mod+enter). */
const COOK_KEYS = "Ctrl+Enter";


export const NODE_MENU: NodeMenuItem[] = [
  {
    // Inside a 逐项处理 block one click cooks every item — there is no cook of one item alone (the
    // queue takes a node, not an instance), so the row says so rather than letting anyone expect otherwise
    key: "cook",
    when: () => true,
    label: (f) => (f.items ? t("ui.node.menu_cook_items") : t("ui.node.menu_cook")),
    desc: (f) => (f.items ? t("ui.node.menu_items", { count: f.items }) : COOK_KEYS),
    off: (f) => f.busy || f.blocked,
    tip: (f) => (f.busy ? tipOf("disabled", t("ui.node.menu_busy")) : f.cookTip),
    run: (f) => cookNode(f.id),
  },
  {
    key: "show",
    when: () => true,
    label: () => t("ui.node.menu_display"),
    desc: () => t("ui.node.menu_double_click"),
    run: (f) => useLook.getState().setDisplay(f.id),
  },
  {
    key: "rename",
    when: (f) => f.editable,
    label: () => t("ui.common.rename"),
    desc: () => t("ui.node.menu_double_click_name"),
    run: (f) => focusInPanel(f.id, ".insp-title .node-name-edit input"),
  },
  {
    key: "comment",
    when: (f) => f.editable,
    label: (f) => (f.comment ? t("ui.node.menu_comment_edit") : t("ui.node.menu_comment_add")),
    desc: () => "",
    run: (f) => focusInPanel(f.id, ".node-comment-field textarea"),
  },
  {
    key: "showComment",
    when: (f) => f.editable && !!f.comment,
    label: (f) => (f.comment?.show ? t("ui.node.menu_comment_hide") : t("ui.node.menu_comment_show")),
    desc: () => "",
    run: (f) => f.comment && setComment(f.id, f.comment.text, !f.comment.show),
  },
  {
    key: "merge",
    when: (f) => f.mergeIds.length > 0,
    label: () => t("ui.node.menu_merge_exr"),
    desc: (f) => (f.mergeIds.length > 1 ? t("ui.node.menu_nodes", { count: f.mergeIds.length }) : ""),
    run: (f) => mergeToExr(f.mergeIds),
  },
];
