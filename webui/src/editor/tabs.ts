import { randomId } from "../platform/randomId";
import { useViewer } from "../state/viewer";

/** The same graph open in another tab of this browser: never let two tabs autosave over each other (the user's work
 * must not be lost). Tabs of this browser tell each other apart with BroadcastChannel, keyed by the graph's saved
 * file (state/viewer.ts's file.handle, stable across tabs once a file is opened: files/handles.ts dedupes by
 * isSameEntry) or, for an unsaved graph, its working copy's id (state/viewer.ts's docId — two tabs share it exactly
 * when one is resuming the very autosaved state the other is holding, see graph/document.ts's loadGraph's docId
 * argument).
 *
 * Whichever tab is on this key first is its "editor" (autosave.ts only ever writes from one); a later tab starts as
 * a "viewer" with a banner offering 在这里编辑 (the earlier tab is told to become a viewer, with its own banner
 * saying so) or 只看 (stays a viewer, banner cleared). If the editor's tab closes, a lone remaining viewer quietly
 * becomes the editor again — nothing left to conflict with. autosave.ts additionally checks a version counter on
 * every write as a backstop, for a browser without BroadcastChannel or a message that did not arrive in time. */

interface Msg {
  type: "hello" | "alive" | "claim" | "bye";
  from: string;
  key: string;
  since: number; // when this tab started on the current key: the earliest wins the editor role on a race
}

const CHANNEL = "lab2shot.tabs";
const GATHER_MS = 350; // long enough for every already-open tab to answer a "hello"
const HEARTBEAT_MS = 4000;
const STALE_MS = 11000;

const tabId = randomId();

const docKey = (): string => {
  const s = useViewer.getState();
  return s.file?.handle ? `file:${s.file.handle}` : `doc:${s.docId}`;
};

export function startTabSync(): () => void {
  if (typeof BroadcastChannel === "undefined") return () => undefined; // no cross-tab check possible: stay the editor
  const bc = new BroadcastChannel(CHANNEL);
  let key = docKey();
  let since = Date.now();
  let gathering = true;
  const peers = new Map<string, { since: number; seen: number }>(); // other tabs sharing the current key

  const send = (type: Msg["type"]) => bc.postMessage({ type, from: tabId, key, since } satisfies Msg);

  const prune = () => {
    const now = Date.now();
    for (const [id, p] of peers) if (now - p.seen > STALE_MS) peers.delete(id);
  };

  // after gathering replies to our own "hello", the earliest-started tab on this key is the editor; a tie (the same
  // millisecond — never in practice) falls to the lower id, so exactly one tab ever wins
  const wins = (peerSince: number, peerId: string) => peerSince < since || (peerSince === since && peerId < tabId);
  const settle = () => {
    gathering = false;
    if ([...peers.entries()].some(([id, p]) => wins(p.since, id))) useViewer.setState({ role: "viewer", peerBanner: "same-open" });
  };

  const reset = () => {
    peers.clear();
    key = docKey();
    since = Date.now();
    gathering = true;
    useViewer.getState().freshDoc(); // a fresh graph is nobody else's until proven otherwise
    send("hello");
    window.setTimeout(settle, GATHER_MS);
  };

  bc.onmessage = ({ data: m }: MessageEvent<Msg>) => {
    if (m.from === tabId || m.key !== key) return;
    prune();
    peers.set(m.from, { since: m.since, seen: Date.now() });
    if (m.type === "hello") send(gathering ? "hello" : "alive"); // answer at once: no need to wait for our own heartbeat
    else if (m.type === "claim") {
      if (useViewer.getState().role === "editor") useViewer.setState({ role: "viewer", peerBanner: "demoted" });
    } else if (m.type === "bye") {
      peers.delete(m.from);
      // the only other tab on this graph just left: a lone viewer has nothing left to conflict with
      const s = useViewer.getState();
      if (s.role === "viewer" && s.peerBanner === "same-open" && peers.size === 0) useViewer.setState({ role: "editor", peerBanner: null });
    }
    // a plain "alive" heartbeat only refreshes `peers` (above): who is editor is settled once, at settle(), and only
    // ever changes again by an explicit "claim" — a demoted tab's `since` is still the earliest it ever was, so
    // judging by `since` again here would immediately (and silently) hand editing back to it on its very next
    // heartbeat, undoing every claim
  };

  send("hello");
  const settleTimer = window.setTimeout(settle, GATHER_MS);
  const heartbeat = window.setInterval(() => {
    prune();
    send(useViewer.getState().role === "editor" ? "hello" : "alive");
  }, HEARTBEAT_MS);

  // claiming editing (the user clicked 在这里编辑, state/viewer.ts's claimEditing) tells every other tab on this
  // graph to yield
  const unsubscribe = useViewer.subscribe((s, p) => {
    if (docKey() !== key) reset();
    else if (s.role === "editor" && p.role === "viewer" && !s.peerBanner) send("claim");
  });

  const bye = () => send("bye");
  window.addEventListener("pagehide", bye);

  return () => {
    unsubscribe();
    window.clearTimeout(settleTimer);
    window.clearInterval(heartbeat);
    window.removeEventListener("pagehide", bye);
    bye();
    bc.close();
  };
}
