import type { GraphJSON } from "../api";
import { fileJSON, loadGraph } from "../graph/actions";
import type { GraphFile } from "../graph/graphFile";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { msg, say } from "../state/say";
import { forgetOtherGraphs } from "../state/stale";
import { useViewer } from "../state/viewer";
import { readLocalJSON, removeLocal, writeLocal } from "../platform/util";
import { same as sameJson, type Json } from "../model/graphPatch";

/** The editor's working state is kept on the user's own machine, in the browser's localStorage (their user profile on
 * Windows, Linux or macOS), written synchronously shortly after every edit and as the page closes (a refresh or a
 * crashed browser never loses work). Nothing is stored on the server; the user's graph file changes only when they
 * save.
 *
 * One working copy per document (state/viewer.ts's docId) and per account, so two tabs with two different graphs, or
 * two people using the same browser one after the other, never write over each other's work. The account keeps the
 * KEEP documents it edited last on this browser. A tab remembers its own document in sessionStorage, so reloading a
 * tab brings back that tab's graph; a new tab opens the document the account wrote last.
 *
 * It subscribes to state/cookInputs.ts's and state/look.ts's `version`, look's `playback` and state/viewer.ts's
 * `file`/`dirty`: a change to any of them is "the document changed". Comparing these counters instead of two whole
 * graphs avoids a JSON.stringify of the graph on every check. */

interface Working {
  graph: GraphJSON;
  file: GraphFile | null;
  dirty: boolean;
  time: number;
  id: string; // the working copy's identity (state/viewer.ts's docId): two tabs with the same one hold the same graph
  rev: number; // bumped on every write; a stored rev higher than the last one this tab wrote is another tab's write
}

const PREFIX = "lab2shot.working";
const TAB = `${PREFIX}.tab`; // sessionStorage: this tab's document, so a reload of the tab brings it back
const KEEP = 8; // working copies kept per account: the documents it edited last on this browser
const DELAY_MS = 400; // after a change: a drag or typing writes a few times a second, not per step

const listKey = (owner: number) => `${PREFIX}.${owner}`; // the account's documents, the one written last first
const copyKey = (owner: number, id: string) => `${PREFIX}.${owner}.${id}`;
const kept = (owner: number): string[] => readLocalJSON<string[]>(listKey(owner), []);
const copyOf = (owner: number, id: string): Working | null => readLocalJSON<Working | null>(copyKey(owner, id), null);

function tabDoc(): string | null {
  try {
    return sessionStorage.getItem(TAB);
  } catch {
    return null;
  }
}

function setTabDoc(id: string): void {
  try {
    sessionStorage.setItem(TAB, id);
  } catch {
    /* storage blocked: a reload opens the account's last document instead */
  }
}

function working(rev: number): Working {
  const viewer = useViewer.getState();
  return { graph: fileJSON(), file: viewer.file, dirty: viewer.dirty, time: Date.now(), id: viewer.docId, rev };
}

/** The working copy this tab opens with: its own (a reload of the tab), else the one the account wrote last here. */
export function lastWorking(owner: number): Working | null {
  const own = tabDoc();
  const last = kept(owner)[0];
  return (own && copyOf(owner, own)) || (last ? copyOf(owner, last) : null);
}

/** Keeps `w` as the account's latest copy and lets the oldest beyond KEEP go. False: the browser's storage is full. */
function store(owner: number, w: Working): boolean {
  if (!writeLocal(copyKey(owner, w.id), JSON.stringify(w))) return false; // the one serialization: a write, not a comparison
  const list = [w.id, ...kept(owner).filter((id) => id !== w.id)];
  for (const id of list.splice(KEEP)) removeLocal(copyKey(owner, id));
  writeLocal(listKey(owner), JSON.stringify(list));
  setTabDoc(w.id);
  return true;
}

/** Room for the open document when the storage is full: the account's other working copies and other graphs' last
 * results (state/stale.ts) go. */
function makeRoom(owner: number, id: string): void {
  for (const other of kept(owner)) if (other !== id) removeLocal(copyKey(owner, other));
  writeLocal(listKey(owner), JSON.stringify([id]));
  forgetOtherGraphs(useCookInputs.getState().graphId);
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

/** Keeps `owner`'s working copy of the open document from now on; returns the function that stops it. */
export function startAutosave(owner: number): () => void {
  // what the browser holds now: a graph corrected on loading (parts that no longer exist removed) differs from it and
  // is written at once, so the correction is not made (and reported) again on every opening
  const stored = copyOf(owner, useViewer.getState().docId);
  const loaded = working(0);
  // the one structural comparison: what the browser holds against what was loaded, once; after it, marks
  let written: Mark | null =
    stored && stored.dirty === loaded.dirty && (stored.file?.name ?? null) === (loaded.file?.name ?? null) && sameJson(stored.graph as unknown as Json, loaded.graph as unknown as Json)
      ? mark()
      : null;
  let rev = stored?.rev ?? 0; // the last revision this tab wrote or accepted as its own
  let seen = Date.now(); // this tab holds everything written up to now (what it opened, or wrote itself)
  let full = false; // the last write did not fit: said once, until one fits again
  let timer = 0;
  const write = () => {
    window.clearTimeout(timer);
    timer = 0;
    const docId = useViewer.getState().docId;
    if (useViewer.getState().role !== "editor") return; // a viewer never writes over the editor's copy
    // A version counter, checked against what is actually stored right now (not just what this tab wrote last): if
    // another tab holding the same working copy has since written a later revision (tabs.ts missed it, or this is a
    // browser without BroadcastChannel), this tab backs off instead of silently overwriting newer work.
    const now = copyOf(owner, docId);
    if (now && now.rev > rev) {
      rev = now.rev;
      useViewer.setState({ role: "viewer", peerBanner: "same-open" });
      return;
    }
    const now2 = mark();
    if (sameMark(written, now2)) return;
    const w = working(rev + 1);
    let fits = store(owner, w);
    if (!fits) (makeRoom(owner, w.id), (fits = store(owner, w)));
    if (!fits) {
      if (!full) say(msg("W-GRAPH-NOTKEPT"));
      full = true;
      return; // not written: tried again on the next change
    }
    full = false;
    rev = w.rev;
    seen = w.time;
    written = now2;
  };
  // Taking over editing (tabs.ts: 在这里编辑, or the editing tab closed): the other tab may have written this document
  // since this one opened it, and that newer copy is what is edited from here on, never the stale one on screen.
  const takeOver = () => {
    const v = useViewer.getState();
    const handle = v.file?.handle;
    let newest: Working | null = null;
    for (const id of kept(owner)) {
      const c = copyOf(owner, id);
      const same = !!c && (c.id === v.docId || (!!handle && c.file?.handle === handle));
      if (c && same && c.time > seen && (!newest || c.time > newest.time)) newest = c;
    }
    if (!newest) return;
    loadGraph(newest.graph, newest.file, newest.dirty, newest.id);
    rev = newest.rev;
    seen = newest.time;
    written = mark();
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
  // the file/dirty fields live in state/viewer.ts; a save (markSaved/setFile) also counts as a change worth writing.
  // Another document loaded in this tab: it is this tab's from now on; its copy, if any, is its own history
  const unsubViewer = useViewer.subscribe((s, p) => {
    if (s.docId !== p.docId) {
      rev = copyOf(owner, s.docId)?.rev ?? 0;
      seen = Date.now();
    } else if (s.role === "editor" && p.role === "viewer") takeOver();
    if ((s.file !== p.file || s.dirty !== p.dirty || s.docId !== p.docId) && !timer) timer = window.setTimeout(write, DELAY_MS);
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
