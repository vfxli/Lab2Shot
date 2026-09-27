import { cookNode, mergeToExr } from "../graph/actions";
import { useLook } from "../state/look";

/** What a node's right-click menu offers, as one table. A row says when it is there, what it reads, and what a click does; the menu
 * component (editor/FlowParts.tsx NodeMenu) only draws the rows the table gives it, so a new action is a row here. */

export interface NodeMenuFacts {
  id: string; // the node right-clicked
  typeId: string;
  delivers: boolean; // cooking it hands files over (the server's policy): an 「输出」, or an output-settings node
  busy: boolean; // this graph already has a job in the queue
  blocked: boolean; // 计算任务 off (state/pause.ts) 或配额满了 (state/quota.ts)：都让「计算」变灰，原因写在 cookTip 里
  cookTip: string; // what the cook is, in words (graph/rules.ts cookWords) with the switches' note
  cookShort: string;
  // a node inside a 逐项处理 block (engine/scopes.py, the status reply's `summary`): how many items it has and what the
  // view is on. 0 outside every block — then the cook rows say nothing about items
  items: number;
  itemName: string; // the item the view is on ("" none chosen yet)
  mergeIds: string[]; // 序列图输出设置 nodes this click would merge (none: the row is not there)
}

export interface NodeMenuItem {
  key: string;
  when: (f: NodeMenuFacts) => boolean;
  label: (f: NodeMenuFacts) => string;
  desc: (f: NodeMenuFacts) => string; // the grey word on the right: a shortcut, a gesture, a count
  tip: (f: NodeMenuFacts) => string;
  off?: (f: NodeMenuFacts) => boolean; // there, but not clickable now (the tip says why)
  run: (f: NodeMenuFacts) => void;
}

/** The keys that cook the node shown, as the one registry has them (editor/App.tsx useShortcut mod+enter). */
export const COOK_KEYS = "Ctrl+Enter";

const BUSY_TIP = "这个节点图已经有一个任务在算，等它算完或先取消";

export const NODE_MENU: NodeMenuItem[] = [
  {
    // Inside a 逐项处理 block one click cooks every item — there is no cook of one item alone (the
    // queue takes a node, not an instance), so the row says so rather than letting anyone expect otherwise
    key: "cook",
    when: () => true,
    label: (f) => (f.delivers ? "提交" : f.items ? "计算（全部条目）" : "计算"),
    desc: (f) => (f.delivers ? "" : f.items ? `${f.items} 条` : COOK_KEYS),
    off: (f) => f.busy || f.blocked,
    tip: (f) =>
      f.busy
        ? BUSY_TIP
        : [f.cookTip,
           f.delivers
             ? "只交付这一份「输出」，节点图里其他的「输出」不算"
             : f.items
               ? `这个节点在「逐项处理」块里：一次算完全部 ${f.items} 条，算好以后在视图底部换条目就能逐条看${f.itemName ? `（现在看的是 ${f.itemName}）` : ""}`
               : `算到这个节点为止，并在视图里显示它（${COOK_KEYS}：计算视图里显示的节点）`,
           f.cookShort].join("\n\n"),
    run: (f) => cookNode(f.id),
  },
  {
    key: "show",
    when: () => true,
    label: () => "显示",
    desc: () => "双击",
    tip: () => "在视图里显示这个节点（双击节点也一样）；只用到轻量节点时它自己算",
    run: (f) => useLook.getState().setDisplay(f.id),
  },
  {
    key: "merge",
    when: (f) => f.mergeIds.length > 0,
    label: () => "合并成多层 EXR",
    desc: (f) => (f.mergeIds.length > 1 ? `${f.mergeIds.length} 个节点` : ""),
    tip: (f) =>
      f.mergeIds.length > 1
        ? `把选中的 ${f.mergeIds.length} 个「序列图输出设置」换成一个「多层 EXR 输出设置」：一行一个图层（名字取自各自的「名字」），接到同一个「输出」`
        : "换成「多层 EXR 输出设置」：先框选或 Shift 加选几个「序列图输出设置」，可以一次合并",
    run: (f) => mergeToExr(f.mergeIds),
  },
];
