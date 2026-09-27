import type { GraphJSON } from "../api";
import { run } from "../platform/db";
import { fileJSON } from "../graph/actions";
import type { GraphFile } from "../graph/graphFile";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { readLocalJSON, writeLocal } from "../platform/util";
import { same as sameJson, type Json } from "../model/graphPatch";

/** The editor's working state is kept on the user's own machine, in the browser's storage (their user profile on
 * Windows, Linux or macOS): the current state in localStorage, written synchronously shortly after every edit and
 * as the page closes (a refresh or a crashed browser never loses work), and a minute-by-minute history of the last
 * HISTORY states in IndexedDB. Nothing is stored on the server; the user's graph file changes only when they save.
 *
 * It subscribes to state/cookInputs.ts's and state/look.ts's `version`, look's `playback` and state/viewer.ts's
 * `file`/`dirty`: a change to any of them is "the document changed". Comparing these counters instead of two whole
 * graphs avoids a JSON.stringify of the graph on every check. */

export interface Working {
  graph: GraphJSON;
  file: GraphFile | null;
  dirty: boolean;
  time: number;
  id: string; // the working copy's identity (state/viewer.ts's docId): two tabs with the same one hold the same graph
  rev: number; // bumped on every write; a stored rev higher than the last one this tab wrote is another tab's write
}

const CURRENT = "lab2shot.working";
const DELAY_MS = 400; // after a change: a drag or typing writes a few times a second, not per step
const HISTORY = 30;
const HISTORY_EVERY_MS = 60_000;

function working(rev = 0): Working {
  const viewer = useViewer.getState();
  return { graph: fileJSON(), file: viewer.file, dirty: viewer.dirty, time: Date.now(), id: viewer.docId, rev };
}

/** The last working state on this machine, if any. */
export const lastWorking = (): Working | null => readLocalJSON<Working | null>(CURRENT, null);

async function addHistory(w: Working): Promise<void> {
  await run("history", "readwrite", (s) => s.add(w));
  const keys = await run<IDBValidKey[]>("history", "readonly", (s) => s.getAllKeys());
  if (keys.length > HISTORY) await run("history", "readwrite", (s) => s.delete(IDBKeyRange.upperBound(keys[keys.length - HISTORY - 1])));
}

/** Unsaved work the user puts aside (opening another graph without saving): into the history now. */
export async function keepAside(): Promise<void> {
  await addHistory(working()).catch(() => undefined);
}

/** Where the document stands now, by the counters every edit bumps (state/cookInputs.ts, state/look.ts) and the file and
 * dirty flag state/viewer.ts holds: equal marks, nothing new to write. */
const mark = () => {
  const v = useViewer.getState();
  return { ci: useCookInputs.getState().version, look: useLook.getState().version, playback: useLook.getState().playback, file: v.file, dirty: v.dirty };
};
type Mark = ReturnType<typeof mark>;
const sameMark = (a: Mark | null, b: Mark) =>
  !!a && a.ci === b.ci && a.look === b.look && a.playback === b.playback && a.file === b.file && a.dirty === b.dirty;

export function startAutosave(): () => void {
  // what the browser holds now: a graph corrected on loading (parts that no longer exist removed) differs from it and
  // is written at once, so the correction is not made (and reported) again on every opening
  const stored = lastWorking();
  const loaded = working();
  // the one structural comparison: what the browser holds against what was loaded, once; after it, marks
  let written: Mark | null =
    stored && stored.dirty === loaded.dirty && (stored.file?.name ?? null) === (loaded.file?.name ?? null) && sameJson(stored.graph as unknown as Json, loaded.graph as unknown as Json)
      ? mark()
      : null;
  let rev = stored?.id === useViewer.getState().docId ? (stored?.rev ?? 0) : 0; // the last revision this tab wrote or accepted as its own
  let lastHistory = 0;
  let timer = 0;
  const write = () => {
    window.clearTimeout(timer);
    timer = 0;
    if (useViewer.getState().role !== "editor") return; // a viewer never writes over the editor's copy
    // A version counter, checked against what is actually stored right now (not just what this tab wrote last): if
    // another tab holding the same working copy has since written a later revision (tabs.ts missed it, or this is a
    // browser without BroadcastChannel), this tab backs off instead of silently overwriting newer work.
    const now = lastWorking();
    if (now && now.id === useViewer.getState().docId && now.rev > rev) {
      rev = now.rev;
      useViewer.setState({ role: "viewer", peerBanner: "same-open" });
      return;
    }
    const now2 = mark();
    if (sameMark(written, now2)) return;
    const w = working(++rev);
    written = now2;
    writeLocal(CURRENT, JSON.stringify(w)); // synchronous: done before the page closes (the one serialization: a write, not a comparison)
    if (w.time - lastHistory >= HISTORY_EVERY_MS) {
      lastHistory = w.time;
      addHistory(w).catch(() => undefined);
    }
  };
  // version: the document changed; the playback range is the view's (not a step to undo, state/look.ts's
  // setPlayback does not bump `version`) but set by hand, so kept as its own comparison
  let lastVersions = { ci: useCookInputs.getState().version, look: useLook.getState().version, playback: useLook.getState().playback };
  const check = () => {
    const now = { ci: useCookInputs.getState().version, look: useLook.getState().version, playback: useLook.getState().playback };
    const edited = now.ci !== lastVersions.ci || now.look !== lastVersions.look || now.playback !== lastVersions.playback;
    lastVersions = now;
    if (edited && !timer) timer = window.setTimeout(write, DELAY_MS);
  };
  const unsubCookInputs = useCookInputs.subscribe(check);
  const unsubLook = useLook.subscribe(check);
  // the file/dirty fields live in state/viewer.ts; a save (markSaved/setFile) also counts as a change worth writing
  const unsubViewer = useViewer.subscribe((s, p) => {
    if ((s.file !== p.file || s.dirty !== p.dirty) && !timer) timer = window.setTimeout(write, DELAY_MS);
  });
  if (!written) timer = window.setTimeout(write, DELAY_MS); // a graph corrected on loading: written at once
  // leaving, reloading or switching away: write now, even within DELAY_MS
  const flush = () => document.visibilityState === "hidden" && write();
  document.addEventListener("visibilitychange", flush);
  window.addEventListener("pagehide", write);
  return () => {
    unsubCookInputs();
    unsubLook();
    unsubViewer();
    window.clearTimeout(timer);
    document.removeEventListener("visibilitychange", flush);
    window.removeEventListener("pagehide", write);
  };
}
