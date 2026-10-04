import { useEffect, useMemo, useRef, useState } from "react";
import type { Choice } from "../api";
import { getNodeDefs, getTypes } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { portColor } from "../graph/nodes";
import { IconBone, IconCamera, IconChevron, IconClose } from "../ui/icons";
import { Button, IconButton } from "../ui/Button";
import { t } from "../i18n/t";
import { SheetFoot, useDraft, Writes, type SheetEditor, type SheetEditorProps, type SheetSummaryProps } from "./ParamSheet";
import { tipAttrs, tipOf } from "../platform/tips";

// An import node's selection of one kind of 3D data (widget "hierarchy", nodes/formats.py selection_param): in the
// panel, what is chosen (a count and the first paths); 「选择…」 opens the file's hierarchy as a tree to pick in, as
// Houdini's scene-graph tree does. Production files hold thousands of prims: the tree is built once per answer of
// NodeDef.choices (every kind's entries by their paths, what tells them apart) and drawn a screenful at a time.
//
// It follows the generic "edit a parameter in a sheet" contract (editor/ParamSheet.tsx): the summary, 「选择…」, 「清掉」,
// when they are enabled and the tree in the window are registered in `Hierarchy` below; opening, 确定 writing back,
// cancel not writing back, Esc and a click on the backdrop are the framework's, and so is fetching the data (NodeDef.choices).

const ROW = 24; // px: every row of the tree the same height, so only the rows in view are drawn
const LIST = 440; // px: the tree's height
const SHOWN = 3; // chips in the panel before 「+N」
const OPEN_ALL = 300; // a kind with at most this many entries opens with every branch holding one expanded

/** A chosen entry on its chip: the last two parts of its path (a production file repeats names: geo_1 in every prop),
 * the whole path on hover. */
const short = (path: string) => path.split("/").filter(Boolean).slice(-2).join("/") || path;

/** The selections of the node the picker's kind sits among: each hierarchy parameter, its output port's type and
 * label (Port.when names it), so every kind shows its icon and color in the one tree. */
function useKinds(nodeId: string) {
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const def = getNodeDefs()[typeId];
  const types = getTypes();
  return useMemo(() => {
    const out: Record<string, { label: string; type: string; color: string }> = {};
    for (const q of def?.params ?? []) {
      if (q.widget !== "hierarchy") continue;
      const port = def!.outputs.find((o) => o.when === q.name);
      out[q.name] = { label: q.label, type: port?.type ?? "", color: portColor(types, port?.type ?? "") };
    }
    return out;
  }, [def, types]);
}

/** The option set of this kind among the file's answer (null while not known). */
const choiceOf = ({ p, choices }: SheetSummaryProps) => (choices ? (choices[p.name] ?? choices[""] ?? null) : null);
/** What is chosen, as a list whether the parameter takes one path or many. */
const chosenOf = ({ p, value }: SheetSummaryProps): string[] =>
  p.type === "array" ? ((value as string[] | null) ?? []) : value ? [value as string] : [];

/** What the file's selection of this kind is, in the panel: 「没有选」 or the count and the first entries as chips
 * (an entry missing from the file is marked). */
function HierarchySummary(x: SheetSummaryProps) {
  const { nodeId, p, choices: all } = x;
  const kinds = useKinds(nodeId);
  const choice = choiceOf(x);
  const many = p.type === "array";
  const chosen = chosenOf(x);
  const options = choice?.options ?? [];
  const known = useMemo(() => new Set(options), [options]);
  const none = !all ? t("ui.params.hier.file_first") : options.length ? t(many ? "ui.params.hier.none_many" : "ui.params.hier.none_one", { count: options.length.toLocaleString() }) : (choice?.empty ?? t("ui.params.hier.file_first"));
  return (
    <>
      {chosen.length === 0 && <span className="hier-none">{none}</span>}
      {many && chosen.length > 0 && <span className="hier-count">{t("ui.params.hier.count", { count: chosen.length.toLocaleString() })}</span>}
      {chosen.slice(0, SHOWN).map((path) => (
        <span key={path} className={`chip hier-chip${known.has(path) || !all ? "" : " gone"}`} data-user-data>
          <i style={{ background: kinds[p.name]?.color }} />
          {choice?.labels?.[path] ?? short(path)}
        </span>
      ))}
      {chosen.length > SHOWN && (
        <span className="hier-more">
          +{(chosen.length - SHOWN).toLocaleString()}
        </span>
      )}
    </>
  );
}

/** The button that clears the selection, right of 「选择…」 (only when something is chosen). */
function HierarchyClear(x: SheetSummaryProps) {
  if (!chosenOf(x).length) return null;
  return (
    <IconButton tone="ghost" onClick={() => x.set(x.p.type === "array" ? [] : "")} aria-label={t("ui.params.clear")}>
      <IconClose size={10} />
    </IconButton>
  );
}

/** The window's content: the file's hierarchy as a tree (TreePicker); 确定 writes the picked paths back. */
function HierarchyEditor({ nodeId, p, value, choices, set, cancel }: SheetEditorProps) {
  const kinds = useKinds(nodeId);
  const many = p.type === "array";
  const initial = chosenOf({ nodeId, p, value, choices, set });
  return (
    <TreePicker all={choices ?? {}} port={p.name} kinds={kinds} many={many} label={p.label} initial={initial}
      onCancel={cancel} onOk={(paths) => set(many ? paths : (paths[0] ?? ""))} />
  );
}

/** widget "hierarchy": registered in editor/ParamControls.tsx SHEET_EDITORS. */
export const Hierarchy: SheetEditor = {
  editor: HierarchyEditor,
  title: (p) => t("ui.params.hier.title", { label: p.label }),
  button: () => t("ui.params.hier.button"),
  width: 780,
  summary: HierarchySummary,
  actions: HierarchyClear,
  ready: (x) => !!x.choices && !!choiceOf(x)?.options.length,
};

interface TNode {
  path: string;
  name: string;
  depth: number;
  parent: TNode | null;
  children: TNode[];
  ports: string[]; // the kinds it is an entry of (a structure node: none)
  below: number; // entries of the kind picked in its branch, itself included
}

/** The file's hierarchy from every kind's entries: each path's parents are its structure (groups, rigs), in the
 * file's order. */
function buildTree(all: Record<string, Choice>, ports: string[], pick: string) {
  const root: TNode = { path: "", name: "", depth: -1, parent: null, children: [], ports: [], below: 0 };
  const at = new Map<string, TNode>([["", root]]);
  const node = (path: string): TNode => {
    const had = at.get(path);
    if (had) return had;
    const cut = path.lastIndexOf("/");
    const parent = node(cut <= 0 ? "" : path.slice(0, cut));
    const made: TNode = { path, name: path.slice(cut + 1) || path, depth: parent.depth + 1, parent, children: [], ports: [], below: 0 };
    parent.children.push(made);
    at.set(path, made);
    return made;
  };
  for (const port of ports) for (const path of all[port]?.options ?? []) node(path).ports.push(port);
  // an entry the file names better than its path's last part (a 蒙皮角色 by its mesh, not its root joint: Choice.labels)
  for (const port of ports) for (const [path, label] of Object.entries(all[port]?.labels ?? {})) if (at.has(path)) at.get(path)!.name = label;
  for (const path of all[pick]?.options ?? []) for (let n: TNode | null = at.get(path)!; n; n = n.parent) n.below += 1;
  return { root, at };
}

/** The tree as rows: the expanded branches (only those holding the kind picked, or all of them), or, searching, the
 * entries of the kind whose path has the words and the branches down to them. */
function visibleRows(root: TNode, expanded: Set<string>, onlyKind: boolean, pick: string, query: string): TNode[] {
  const out: TNode[] = [];
  const q = query.trim().toLowerCase();
  if (q) {
    const keep = new Set<TNode>();
    const walk = (n: TNode) => {
      for (const c of n.children) {
        if (c.ports.includes(pick) && (c.path.toLowerCase().includes(q) || c.name.toLowerCase().includes(q))) for (let k: TNode | null = c; k && !keep.has(k); k = k.parent) keep.add(k);
        if (c.below) walk(c);
      }
    };
    walk(root);
    const show = (n: TNode) => n.children.forEach((c) => keep.has(c) && (out.push(c), show(c)));
    show(root);
    return out;
  }
  const show = (n: TNode) => {
    for (const c of n.children) {
      if (onlyKind && !c.below) continue;
      out.push(c);
      if (expanded.has(c.path)) show(c);
    }
  };
  show(root);
  return out;
}

/** A small glyph per kind of 3D data, in its type's color; a structure node (group, rig) a hollow square. */
function KindIcon({ ports, kinds, dim }: { ports: string[]; kinds: Record<string, { type: string; color: string }>; dim: boolean }) {
  const k = kinds[ports[0] ?? ""];
  const color = dim ? "var(--text-3)" : (k?.color ?? "var(--text-3)");
  const kind = k?.type.split(".")[1] ?? "";
  if (kind === "camera") return <IconCamera size={13} color={color} />;
  if (kind === "skeleton" || kind === "character") return <IconBone size={13} color={color} />;
  return (
    <svg width={13} height={13} viewBox="0 0 16 16" fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      {kind === "model" && <path d="M8 2.2 13.5 5v6L8 13.8 2.5 11V5L8 2.2Zm0 0v5.8m0 0 5.5-3M8 8 2.5 5" />}
      {kind === "points" && [3.5, 8, 12.5].flatMap((x) => [4, 8, 12].map((y) => <circle key={`${x},${y}`} cx={x + (y === 8 ? 1 : 0)} cy={y} r="0.9" fill={color} stroke="none" />))}
      {/* 3D curves: three lines fanning out from one end (strands, hair guides), to tell them from point clouds and meshes */}
      {kind === "curves" && <path d="M3 13.5C5 9 5.5 5 8 2.5M6.5 13.5C7.5 9.5 8.5 6 10.5 3.6M10 13.5C10.2 9.8 11.4 6.8 13 4.6" />}
      {!kind && <rect x="3" y="3" width="10" height="10" rx="2" />}
    </svg>
  );
}

/** The file's hierarchy to pick entries of one kind in (Houdini's scene-graph tree): collapsible, DCC paths, an icon
 * per kind; the other kinds and the groups greyed, as structure. Click selects, Ctrl+click adds or removes, Shift+click
 * selects the range; a group offers the entries of this kind below it; search by path; expand or collapse all;
 * arrow keys walk the tree (→ ← open and close, space selects, Enter confirms). The camera picker takes one. The window
 * around it (title, Esc, the backdrop) is the framework's (ParamSheet.tsx). */
function TreePicker({ all, port, kinds, many, label, initial, onOk, onCancel }: {
  all: Record<string, Choice>;
  port: string;
  kinds: Record<string, { label: string; type: string; color: string }>;
  many: boolean;
  label: string;
  initial: string[];
  onOk: (paths: string[]) => void;
  onCancel: () => void;
}) {
  const ports = Object.keys(kinds);
  const { root, at } = useMemo(() => buildTree(all, ports, port), [all, port]); // eslint-disable-line react-hooks/exhaustive-deps
  const entries = all[port]?.options ?? [];
  const details = all[port]?.details ?? {};
  const [picked, setPicked] = useDraft(() => new Set(initial)); // 草稿：只读时改不动（ParamSheet.tsx useDraft）
  const [anchor, setAnchor] = useState<string | null>(initial[0] ?? null);
  const [query, setQuery] = useState("");
  const [onlyKind, setOnlyKind] = useState(true);
  const [expanded, setExpanded] = useState(() => {
    const open = new Set<string>();
    const up = (path: string) => {
      for (let n = at.get(path)?.parent; n && n.path; n = n.parent) open.add(n.path);
    };
    if (entries.length <= OPEN_ALL) entries.forEach(up);
    else root.children.forEach((c) => open.add(c.path));
    initial.forEach(up);
    return open;
  });
  const rows = useMemo(() => visibleRows(root, expanded, onlyKind, port, query), [root, expanded, onlyKind, port, query]);
  const [focus, setFocus] = useState(() => Math.max(0, rows.findIndex((r) => picked.has(r.path))));
  const [top, setTop] = useState(0);
  const list = useRef<HTMLDivElement>(null);
  const gone = [...picked].filter((p) => !at.get(p)?.ports.includes(port));

  const isEntry = (n: TNode) => n.ports.includes(port);
  const below = (n: TNode): string[] => {
    const out: string[] = [];
    const walk = (k: TNode) => {
      if (isEntry(k)) out.push(k.path);
      if (k.below) k.children.forEach(walk);
    };
    walk(n);
    return out;
  };
  const choose = (paths: string[], add: boolean) => setPicked((was) => (many ? new Set(add ? [...was, ...paths] : paths) : new Set(paths.slice(0, 1))));
  const range = (to: number) => {
    const from = rows.findIndex((r) => r.path === anchor);
    const [a, b] = from < 0 ? [to, to] : [Math.min(from, to), Math.max(from, to)];
    return rows.slice(a, b + 1).filter(isEntry).map((r) => r.path);
  };
  const toggleOpen = (n: TNode) => setExpanded((was) => {
    const next = new Set(was);
    if (next.has(n.path)) next.delete(n.path);
    else next.add(n.path);
    return next;
  });
  const click = (i: number, e: React.MouseEvent | React.KeyboardEvent, n: TNode) => {
    setFocus(i);
    if (!isEntry(n)) return toggleOpen(n);
    if (many && e.shiftKey) choose(range(i), e.ctrlKey || e.metaKey);
    else if (many && (e.ctrlKey || e.metaKey)) setPicked((was) => {
      const next = new Set(was);
      if (next.has(n.path)) next.delete(n.path);
      else next.add(n.path);
      return next;
    });
    else choose([n.path], false);
    if (!e.shiftKey) setAnchor(n.path);
  };
  const ordered = () => [...entries.filter((p) => picked.has(p)), ...gone];
  const ok = () => onOk(ordered());

  // keep the focused row in view
  useEffect(() => {
    const el = list.current;
    if (!el) return;
    if (focus * ROW < el.scrollTop) el.scrollTop = focus * ROW;
    else if ((focus + 1) * ROW > el.scrollTop + LIST) el.scrollTop = (focus + 1) * ROW - LIST;
  }, [focus]);
  useEffect(() => list.current?.focus(), []);

  const key = (e: React.KeyboardEvent) => {
    const n = rows[focus];
    const move = (to: number) => {
      const i = Math.max(0, Math.min(rows.length - 1, to));
      setFocus(i);
      if (e.shiftKey && many && rows[i] && isEntry(rows[i])) choose(range(i), e.ctrlKey || e.metaKey);
    };
    if (e.key === "ArrowDown") move(focus + 1);
    else if (e.key === "ArrowUp") move(focus - 1);
    else if (e.key === "Home") move(0);
    else if (e.key === "End") move(rows.length - 1);
    else if (e.key === "PageDown") move(focus + Math.floor(LIST / ROW));
    else if (e.key === "PageUp") move(focus - Math.floor(LIST / ROW));
    else if (e.key === "ArrowRight" && n) {
      if (n.children.length && !expanded.has(n.path) && !query) toggleOpen(n);
      else if (n.children.length) move(focus + 1);
    } else if (e.key === "ArrowLeft" && n) {
      if (expanded.has(n.path) && !query) toggleOpen(n);
      else if (n.parent?.path) move(rows.indexOf(n.parent));
    } else if (e.key === " " && n) {
      if (isEntry(n)) click(focus, { ...e, ctrlKey: many, metaKey: false, shiftKey: false } as React.KeyboardEvent, n);
      else if (many) choose(below(n), true);
    } else if (e.key === "Enter") {
      if (!many && n && isEntry(n)) onOk([n.path]);
      else ok();
    } else if (e.key.toLowerCase() === "a" && (e.ctrlKey || e.metaKey) && many) choose(rows.filter(isEntry).map((r) => r.path), true);
    else return;
    e.preventDefault();
  };

  const first = Math.max(0, Math.floor(top / ROW) - 8);
  const last = Math.min(rows.length, Math.ceil((top + LIST) / ROW) + 8);
  const count = many ? t("ui.params.hier.picked_many", { count: picked.size.toLocaleString(), label }) : picked.size ? t("ui.params.hier.picked_one", { path: [...picked][0] }) : t("ui.params.hier.picked_none");
  return (
    <>
      <div className="htree-bar">
        <input className="field htree-search" placeholder={t("ui.params.hier.search")} value={query} onChange={(e) => setQuery(e.target.value)} />
        <label className="htree-only">
          <input type="checkbox" checked={onlyKind} onChange={(e) => setOnlyKind(e.target.checked)} />
          {t("ui.params.hier.only_kind", { label })}
        </label>
        <Button tone="ghost" disabled={!!query} onClick={() => setExpanded(new Set([...at.values()].filter((n) => n.children.length && (!onlyKind || n.below)).map((n) => n.path)))}>
          {t("ui.params.hier.expand_all")}
        </Button>
        <Button tone="ghost" disabled={!!query} onClick={() => setExpanded(new Set())}>
          {t("ui.params.hier.collapse_all")}
        </Button>
      </div>
      <div className="htree" ref={list} tabIndex={0} onKeyDown={key} onScroll={(e) => setTop(e.currentTarget.scrollTop)} style={{ height: LIST }}>
        <div style={{ height: rows.length * ROW, position: "relative" }}>
          {rows.slice(first, last).map((n, j) => {
            const i = first + j;
            const entry = isEntry(n);
            const open = expanded.has(n.path) || !!query;
            return (
              <div key={n.path} className={`hrow${entry ? " entry" : " struct"}${picked.has(n.path) ? " on" : ""}${i === focus ? " focus" : ""}`}
                style={{ top: i * ROW, paddingLeft: 6 + n.depth * 16 }} data-path={n.path}
                onMouseDown={(e) => e.preventDefault()} onClick={(e) => click(i, e, n)} onDoubleClick={() => entry && !many && onOk([n.path])}>
                <span className="htwist" onClick={(e) => (e.stopPropagation(), n.children.length && !query && toggleOpen(n))}>
                  {n.children.length > 0 && <span style={{ display: "inline-flex", transform: open ? "rotate(180deg)" : "rotate(90deg)" }}><IconChevron size={10} up /></span>}
                </span>
                <KindIcon ports={entry ? [port] : n.ports} kinds={kinds} dim={!entry} />
                <span className="hname" data-user-data {...tipAttrs(tipOf("value", n.path))}>{n.name}</span>
                {entry && <span className="hdetail">{details[n.path]}</span>}
                {!entry && n.below > 0 && many && (
                  <Writes><button className="hgroup-pick"
                    onClick={(e) => (e.stopPropagation(), setFocus(i), choose(below(n), e.ctrlKey || e.metaKey))}>
                    {t("ui.params.hier.pick_below", { count: n.below.toLocaleString() })}
                  </button></Writes>
                )}
                {!entry && n.below > 0 && !many && <span className="hdetail">{t("ui.params.hier.below_one", { count: n.below })}</span>}
              </div>
            );
          })}
        </div>
        {rows.length === 0 && <div className="htree-empty">{query ? t("ui.params.hier.no_match", { query, label }) : t("ui.params.hier.none_in_file", { label })}</div>}
      </div>
      <SheetFoot layout="htree-foot" ok={ok} cancel={onCancel}>
        <span className="htree-count" {...tipAttrs(many ? tipOf("value", [...picked].slice(0, 30).join("\n")) : undefined)}>{count}</span>
        {gone.length > 0 && (
          <Button tip={tipOf("value", gone.join("\n"))} tone="ghost" layout="htree-gone" onClick={() => setPicked((was) => new Set([...was].filter((p) => !gone.includes(p))))}>
            {t("ui.params.hier.drop_gone", { count: gone.length })}
          </Button>
        )}
        {many && picked.size > 0 && (
          <Button tone="ghost" onClick={() => setPicked(new Set())}>
            {t("ui.params.hier.pick_none")}
          </Button>
        )}
      </SheetFoot>
    </>
  );
}
