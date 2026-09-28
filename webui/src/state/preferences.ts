import { create } from "zustand";
import { readPref, writePref } from "../platform/storage";
import type { LoopMode } from "../model/timelineMath";

/** 用户首选项: how this browser likes its tools set up, the same across every node graph, never undone, never sent
 * to the server. The remembered tool settings live here (panel widths, the minimap, recent nodes, the display
 * options' last page, the timeline strip's height and open state, the playback loop mode); the display options
 * themselves are kept by state/viewTools.ts's useViewOptions (model/viewOptions.ts).
 *
 * There is no picture-quality setting: the 2D view always goes through the view proxy, and point clouds are always
 * thinned to the admin page's 「点云上限」, with no switch. */

export type { LoopMode };

interface Preferences {
  split: number; // % height of the viewer vs. the node graph in the left column (editor/App.tsx's horizontal splitter)
  inspectorWidth: number | null; // px, set by dragging the divider; null: fit the parameters
  minimap: boolean;
  recentNodes: string[]; // node type ids, most recent first (NodeMenu.tsx, 6 kept)
  displayOptionsTab: string; // the 显示 options popup's last page (ui/DisplayOptions.tsx TABS; "": its first)
  browseGroup: string; // the templates panel's current 大类 (分类排版); "": the first
  timelineOpen: boolean; // the curve strip under the stage
  timelineHeight: number;
  playbackMode: LoopMode;

  setSplit: (v: number) => void;
  setInspectorWidth: (v: number | null) => void;
  setMinimap: (v: boolean) => void;
  toggleMinimap: () => void;
  pushRecentNode: (id: string) => void;
  setDisplayOptionsTab: (v: string) => void;
  setBrowseGroup: (v: string) => void;
  setTimelineStrip: (patch: Partial<{ open: boolean; height: number }>) => void;
  setPlayback: (patch: Partial<{ mode: LoopMode }>) => void;
}

export const TIMELINE_STRIP = { min: 120, initial: 240 };
const RECENT_MAX = 6;

function readTimelineStrip(): { open: boolean; height: number } {
  const v = readPref<{ open?: unknown; height?: unknown }>("curveStrip", {});
  return { open: v.open === true, height: typeof v.height === "number" && v.height >= TIMELINE_STRIP.min ? v.height : TIMELINE_STRIP.initial };
}

function readPlayback(): { mode: LoopMode } {
  const v = readPref<{ mode?: unknown }>("timeline", {});
  const mode = v.mode === "loop" || v.mode === "once" || v.mode === "bounce" ? v.mode : "loop";
  return { mode };
}

export const usePreferences = create<Preferences>((set, get) => {
  const strip = readTimelineStrip();
  const playback = readPlayback();
  return {
    split: Number(readPref("split", 52)) || 52,
    inspectorWidth: Number(readPref("inspectorWidth", "")) || null,
    minimap: readPref<string>("minimap", "0") === "1",
    recentNodes: readPref<string[]>("recentNodes", []),
    displayOptionsTab: readPref<string>("displayOptions.tab", ""),
    browseGroup: readPref<string>("browse.group", ""),
    timelineOpen: strip.open,
    timelineHeight: strip.height,
    playbackMode: playback.mode,

    setSplit: (v) => (writePref("split", String(Math.round(v))), set({ split: v })),
    setInspectorWidth: (v) => (writePref("inspectorWidth", v ? String(v) : ""), set({ inspectorWidth: v })),
    setMinimap: (v) => (writePref("minimap", v ? "1" : "0"), set({ minimap: v })),
    toggleMinimap: () => get().setMinimap(!get().minimap),
    pushRecentNode: (id) => {
      const next = [id, ...get().recentNodes.filter((x) => x !== id)].slice(0, RECENT_MAX);
      writePref("recentNodes", next);
      set({ recentNodes: next });
    },
    setDisplayOptionsTab: (v) => (writePref("displayOptions.tab", v), set({ displayOptionsTab: v })),
    setBrowseGroup: (v) => (writePref("browse.group", v), set({ browseGroup: v })),
    setTimelineStrip: (patch) => {
      const open = patch.open ?? get().timelineOpen;
      const height = Math.max(TIMELINE_STRIP.min, Math.round(patch.height ?? get().timelineHeight));
      writePref("curveStrip", { open, height });
      set({ timelineOpen: open, timelineHeight: height });
    },
    setPlayback: (patch) => {
      const mode = patch.mode ?? get().playbackMode;
      writePref("timeline", { mode });
      set({ playbackMode: mode });
    },
  };
});
