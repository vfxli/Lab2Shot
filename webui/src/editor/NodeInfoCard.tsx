import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ServerMessage } from "../api";
import { LEVELS_BY_SEVERITY, LEVEL_TABLE } from "../messages/levels";
import { IconInfo } from "../ui/icons";
import { createPortal } from "react-dom";
import { useGraphSnapshot } from "../graph/snapshot";
import { useKeyLayer, useShortcut } from "../platform/keys";
import { useRefSize } from "../platform/size";
import { DataGroup } from "./DataInfo";
import "./styles/28-node-info-card.css";

/** 数据信息: the mark at a node's bottom right and the card it opens (this file holds both — the mark is nothing but
 * the way in). The card itself, beside the node: a glass card floating over the canvas, listing what every input and every output of
 * this node holds now, and — for a node inside a 逐项处理 block — every item.
 *
 * It writes none of that: the rows are editor/DataInfo.tsx's DataGroup, the one place the server's summary is laid
 * out. This file is only where it floats and how it shuts.
 *
 * It hangs in a portal on <body>, like the site's menu: the node graph, the panels and the top bar all clip what is
 * inside them, and a card drawn inside a node would be cut by the first of them it reached. */
export function NodeInfoCard({ node, label, at, onClose }: {
  node: string;
  label: string;
  at: DOMRect; // where the mark is, in the window's pixels
  onClose: () => void;
}) {
  const snap = useGraphSnapshot();
  const g = snap.nodes.find((n) => n.id === node);
  const card = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ left: number; top: number } | null>(null);
  // the card's rows are filled only once the server's summary arrives (DataInfo.tsx useManifests): the first frame holds
  // only 「读取…」 and the card is narrow. When the content grows the card would run off the window's right edge and its
  // text be cut, so its position is recomputed from the card's own size, through the page's one size observer
  // (platform/size.ts; no observer of its own here)
  const size = useRefSize(card);

  // it closes the way every floating layer of this page does: a click outside, a scroll, Esc
  useEffect(() => {
    const close = (e: Event) => {
      if (!card.current?.contains(e.target as Node)) onClose();
    };
    const later = setTimeout(() => {
      window.addEventListener("pointerdown", close);
      window.addEventListener("wheel", close);
    });
    return () => {
      clearTimeout(later);
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("wheel", close);
    };
  }, [onClose]);
  useShortcut({ keys: ["escape"], inText: true, run: () => onClose() }, { layer: useKeyLayer(true) });

  // beside the mark, kept inside the window: to its right if it fits there, otherwise to its left; never off the
  // bottom edge (a node low in the graph would push a tall card out of sight)
  useLayoutEffect(() => {
    const el = card.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const gap = 8;
    const left = at.right + gap + r.width <= window.innerWidth - gap ? at.right + gap : Math.max(gap, at.left - gap - r.width);
    const top = Math.max(gap, Math.min(at.top - 8, window.innerHeight - gap - r.height));
    setBox({ left, top });
  }, [at, size.w, size.h]);

  if (!g) return null;
  return createPortal(
    <div
      ref={card}
      className="node-info-card glass strong"
      role="dialog"
      aria-label={`数据信息：${label}`}
      style={box ? { left: box.left, top: box.top } : { left: at.right + 8, top: at.top - 8, visibility: "hidden" }}
      onPointerDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="node-info-head">
        <span className="node-info-name" data-user-data data-tip={label}>{label}</span>
        <span className="node-info-what">数据信息</span>
      </div>
      <div className="node-info-body">
        <DataGroup node={g} />
      </div>
    </div>,
    document.body,
  );
}

/** The loudest level among a node's messages ("": nothing worth reading): what colours its 数据信息 mark. The order is
 * the page's one level table (messages/levels.ts), never a second list here. */
export function worstLevel(messages: readonly ServerMessage[]): string {
  return LEVELS_BY_SEVERITY.find((l) => LEVEL_TABLE[l].listed && messages.some((m) => m.level === l)) ?? "";
}

export function NodeInfoButton({ node, label, level }: { node: string; label: string; level: string }) {
  const [at, setAt] = useState<DOMRect | null>(null);
  return (
    <>
      <span
        className={`gnode-info nodrag${at ? " on" : ""}`}
        data-level={level || undefined}
        role="button"
        tabIndex={0}
        aria-label="数据信息"
        aria-expanded={!!at}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          setAt(at ? null : e.currentTarget.getBoundingClientRect());
        }}
      >
        <IconInfo size={12} />
      </span>
      {at && <NodeInfoCard node={node} label={label} at={at} onClose={() => setAt(null)} />}
    </>
  );
}
