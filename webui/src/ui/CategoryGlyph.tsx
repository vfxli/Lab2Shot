import React from "react";

/** The 16x16 mark of one node-menu category (the tools band in menu/categories.json, the algorithms band = the templates tree
 * in templates/_categories.json; the administrator's data). Keyed by
 * category id — a drawing per id, not a list of nodes or projects: what belongs to a category is the server's
 * answer. An id with no drawing yet shows the plain dot, never an empty box. */
// a data-tree id that draws as another's mark: the tools band's 「遮罩」 is tools_mask in menu/categories.json (the
// algorithms band's 抠像与遮罩 has a subcategory with the id mask), and the tools band's 「动作」 is tools_motion (its
// algorithms counterpart 动作处理 is animation); the icon table below is keyed by the drawn word
const ALIAS: Record<string, string> = { tools_mask: "mask", tools_motion: "animation" };

export function CategoryGlyph({ category, color }: { category: string; color: string }) {
  const g: Record<string, React.ReactNode> = {
    // ---- the upper band: our own tools
    read: <path d="M3 8h7M7.5 5 10.5 8 7.5 11M12.5 3.5v9" />,
    image: (
      <>
        <rect x="2.8" y="3.5" width="10.4" height="9" rx="1.6" />
        <path d="m3.5 11 3-3 2.5 2.5 1.5-1.5 2 2" />
      </>
    ),
    mask: (
      <>
        <circle cx="8" cy="8" r="5.2" />
        <path d="M8 2.8a5.2 5.2 0 0 1 0 10.4Z" fill={color} stroke="none" />
      </>
    ),
    camera: (
      <>
        <rect x="2" y="4.8" width="8.6" height="6.6" rx="1.4" />
        <path d="m10.6 7.2 3.4-1.8v5.4l-3.4-1.8" />
      </>
    ),
    geometry: (
      <>
        <path d="M2.5 11.5 8 3.5l5.5 8Z" />
        <path d="M5.2 7.6h5.6" />
      </>
    ),
    scene: (
      <>
        <path d="M8 2.8 13 5.5v5L8 13.2 3 10.5v-5Z" />
        <path d="M3 5.5 8 8.2l5-2.7M8 8.2v5" />
      </>
    ),
    blocks: (
      <>
        <rect x="2.4" y="2.4" width="5" height="5" rx="1" />
        <rect x="8.6" y="8.6" width="5" height="5" rx="1" />
        <path d="M7.4 4.9h3.7v3.7" />
      </>
    ),
    output: <path d="M3.5 8h7M7.5 5l3 3-3 3M2.8 3.5v9" transform="translate(16 0) scale(-1 1)" />,
    // ---- the lower band: the delivery categories, in the tree's order (formats has no drawing: the plain dot)
    camera_track: (
      <>
        <rect x="6.6" y="6.2" width="6.6" height="5.2" rx="1.2" />
        <path d="M2.2 12.4c1.4-3.6 3-5.8 4.4-6.8" strokeDasharray="1.5 1.7" />
        <circle cx="2.4" cy="12.6" r="1.1" fill={color} stroke="none" />
      </>
    ),
    reconstruct: (
      <>
        <path d="M8 2.6 13.4 5.6v4.8L8 13.4 2.6 10.4V5.6Z" />
        <circle cx="8" cy="8" r="0.9" fill={color} stroke="none" />
        <circle cx="5.3" cy="6.4" r="0.7" fill={color} stroke="none" />
        <circle cx="10.7" cy="9.6" r="0.7" fill={color} stroke="none" />
      </>
    ),
    depth: (
      <>
        <path d="M2.6 4.4h10.8M3.6 7.4h8.8M4.8 10.4h6.4M6.2 13h3.6" />
      </>
    ),
    matte: (
      <>
        <rect x="2.4" y="2.4" width="11.2" height="11.2" rx="1.6" />
        <path d="M8 4.6c2 0 3.4 1.6 3.4 3.4S10 11.4 8 11.4Z" fill={color} stroke="none" />
      </>
    ),
    tracks: (
      <>
        <path d="M2.5 12.5c2-1 3-5 5.5-5.5s2.6-1.2 3.4-2.4" strokeDasharray="1.6 1.8" />
        <circle cx="12.2" cy="3.9" r="1.7" fill={color} stroke="none" />
      </>
    ),
    warp: (
      <>
        <path d="M2.6 4.6c2.6 0 3.4 2 6 2M2.6 8c2.6 0 3.4 2 6 2M2.6 11.4c2.6 0 3.4 2 6 2" />
        <path d="m11.2 4.4 2 2.2-2 2.2" />
      </>
    ),
    body: (
      <>
        <circle cx="8" cy="4.6" r="2" />
        <path d="M4.2 13.2c.4-2.8 1.8-4.4 3.8-4.4s3.4 1.6 3.8 4.4" />
      </>
    ),
    animation: (
      <>
        <path d="M2.6 12c2.6 0 3.4-8 6-8s3.4 4 4.8 4" />
        <circle cx="2.6" cy="12" r="1.3" fill={color} stroke="none" />
        <circle cx="13.4" cy="8" r="1.3" fill={color} stroke="none" />
      </>
    ),
    face: (
      <>
        <ellipse cx="8" cy="8.2" rx="4.4" ry="5.2" />
        <path d="M6.3 7.2v.4M9.7 7.2v.4M6.4 10.3c1 .8 2.2.8 3.2 0" />
      </>
    ),
    light: (
      <>
        <circle cx="8" cy="8" r="2.6" />
        <path d="M8 2v1.4M8 12.6V14M2 8h1.4M12.6 8H14M3.8 3.8l1 1M11.2 11.2l1 1M3.8 12.2l1-1M11.2 4.8l1-1" />
      </>
    ),
  };
  return (
    <span className="cat-icon" style={{ background: `${color}29` }}>
      <svg width={12} height={12} viewBox="0 0 16 16" fill="none" stroke={color} strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
        {g[category] ?? g[ALIAS[category] ?? ""] ?? <circle cx="8" cy="8" r="2.6" fill={color} stroke="none" />}
      </svg>
    </span>
  );
}
