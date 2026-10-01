import { useEffect, useMemo, useRef, useState } from "react";
import type { RigSide } from "../../api";
import { IconChevron } from "../../ui/icons";
import type { Tree } from "./rules";

// 「对应关系」编辑器两侧的骨架树（Maya Outliner / HumanIK 的 Skeleton 列表）：按层级缩进，可以收起；每行右边写这个
// 关节现在属于哪个部位（按所属槽的颜色）。MHR 127 个关节、带全套手指和面部的 FBX 几百个关节，只有树能精确点到，
// 所以默认是树。行高固定、只画看得见的行（照 editor/HierarchyPicker.tsx 的 TreePicker）。

const ROW = 22; // px

interface Props {
  side: RigSide;
  tree: Tree;
  height: number;
  /** 关节名 → 部位的中文名和槽的状态（ok / half / bad …），没有部位的关节不在里面 */
  tagOf: (name: string) => { label: string; state: string } | undefined;
  /** 当前部位（蓝框的槽）在这一侧的关节：高亮，并在换部位时滚到第一个 */
  active: string[];
  /** 这一侧点选的关节（右键槽「指定选中的关节」用） */
  picked: string | null;
  onPick: (name: string, e: React.MouseEvent) => void;
}

export function SkeletonTree({ side, tree, height, tagOf, active, picked, onPick }: Props) {
  const parents = side.parents ?? [];
  const children = useMemo(() => {
    const kids: number[][] = side.names.map(() => []);
    const roots: number[] = [];
    parents.forEach((p, i) => (p >= 0 ? kids[p]?.push(i) : roots.push(i)));
    if (!parents.length) side.names.forEach((_, i) => roots.push(i));
    return { kids, roots };
  }, [side]); // eslint-disable-line react-hooks/exhaustive-deps
  const [closed, setClosed] = useState<Set<number>>(() => new Set());
  const [query, setQuery] = useState("");
  const rows = useMemo(() => {
    const out: number[] = [];
    const q = query.trim().toLowerCase();
    if (q) {
      const keep = new Set<number>();
      side.names.forEach((n, i) => {
        if (!n.toLowerCase().includes(q)) return;
        for (let k = i; k >= 0 && !keep.has(k); k = parents[k] ?? -1) keep.add(k);
      });
      const walk = (i: number) => {
        if (!keep.has(i)) return;
        out.push(i);
        children.kids[i].forEach(walk);
      };
      children.roots.forEach(walk);
      return out;
    }
    const walk = (i: number) => {
      out.push(i);
      if (!closed.has(i)) children.kids[i].forEach(walk);
    };
    children.roots.forEach(walk);
    return out;
  }, [side, children, closed, query]); // eslint-disable-line react-hooks/exhaustive-deps

  const list = useRef<HTMLDivElement>(null);
  const rowsRef = useRef(rows);
  rowsRef.current = rows;
  const [top, setTop] = useState(0);
  const activeSet = useMemo(() => new Set(active), [active]);
  // 换了部位：把它的关节展开出来、滚到第一个（HumanIK 点格子时场景里选中那根骨头）
  const first = active[0];
  useEffect(() => {
    if (!first) return;
    const i = tree.index.get(first);
    if (i === undefined) return;
    setClosed((was) => {
      let next = was;
      for (let k = parents[i] ?? -1; k >= 0; k = parents[k] ?? -1)
        if (next.has(k)) {
          if (next === was) next = new Set(was);
          next.delete(k);
        }
      return next;
    });
    requestAnimationFrame(() => {
      const el = list.current;
      const at = rowsRef.current.indexOf(i);
      if (!el || at < 0) return;
      if (at * ROW < el.scrollTop || (at + 1) * ROW > el.scrollTop + height) el.scrollTop = Math.max(0, at * ROW - height / 3);
    });
  }, [first]); // eslint-disable-line react-hooks/exhaustive-deps

  const from = Math.max(0, Math.floor(top / ROW) - 8);
  const to = Math.min(rows.length, Math.ceil((top + height) / ROW) + 8);
  return (
    <>
      <input className="field rmap-search" placeholder="搜关节名" value={query} onChange={(e) => setQuery(e.target.value)}
        data-tip="只列出名字里有这些字的关节，和到它们的层级" />
      <div className="rmap-tree" ref={list} style={{ height }} onScroll={(e) => setTop(e.currentTarget.scrollTop)}
        data-tip="点一个关节：当前部位（蓝框的槽）就配成它；链状部位 Shift+点最后一节选中间一整串，Ctrl+点加减一节">
        <div style={{ height: rows.length * ROW, position: "relative" }}>
          {rows.slice(from, to).map((i, j) => {
            const name = side.names[i];
            const tag = tagOf(name);
            const kids = children.kids[i].length;
            const open = !closed.has(i) || !!query;
            return (
              <div key={i} className={`rmap-row${activeSet.has(name) ? " on" : ""}${picked === name ? " picked" : ""}`}
                style={{ top: (from + j) * ROW, paddingLeft: 4 + tree.depth[i] * 12 }}
                onMouseDown={(e) => e.preventDefault()} onClick={(e) => onPick(name, e)}>
                <span className="rmap-twist" onClick={(e) => {
                  e.stopPropagation();
                  if (!kids || query) return;
                  setClosed((was) => {
                    const next = new Set(was);
                    if (next.has(i)) next.delete(i);
                    else next.add(i);
                    return next;
                  });
                }}>
                  {kids > 0 && <span style={{ display: "inline-flex", transform: open ? "rotate(180deg)" : "rotate(90deg)" }}><IconChevron size={9} up /></span>}
                </span>
                <span className="rmap-name" data-user-data>{name}</span>
                {tag && <span className={`rmap-tag ${tag.state}`}>{tag.label}</span>}
              </div>
            );
          })}
        </div>
        {rows.length === 0 && <div className="rmap-empty">{query ? `没有名字里有「${query}」的关节` : "没有关节"}</div>}
      </div>
    </>
  );
}
