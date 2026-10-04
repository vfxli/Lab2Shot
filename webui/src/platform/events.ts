import { answer, ApiError, loginEnded } from "./http";
import { noteDown, textBytes } from "./traffic";

/** Follows a Server-Sent Events stream. This is the only place that opens an EventSource. The browser repairs a dropped connection by itself: it reconnects and
 * resumes from the last event received (Last-Event-ID; the server numbers every event permanently and answers a
 * reconnect with the events that followed, lab2shot/server/farm.py events), so a drop is not reported. A stream opened
 * again here (a stalled line, a refused one) is a new EventSource, which sends no Last-Event-ID: the last id received
 * goes as the `last_event_id` query parameter instead, so it too resumes where it was. A stream the browser has given
 * up on (readyState CLOSED: the server refused the reconnect) is indistinguishable by itself, so it is probed once
 * through the shared request layer, which applies the single 401 policy: an expired login is handed to the gate, which
 * asks for it again over the page, and following stops (`onlogin`: the followed resource is unaffected; the caller
 * follows it again after the new login), and only a stream the server reports as no longer present (a 4xx) triggers
 * `ongone`. A stream that still opens was a transient refusal; a
 * probe that never reached the server (the network is down) or a server error (5xx: failing or restarting) says nothing
 * about the stream. In all three cases the stream is followed again with increasing delay. */

interface EventFollowing {
  /** Every event as received: its data and its id ("" when the server supplied none). */
  onmessage: ((data: string, id: string) => void) | null;
  /** The stream has ended permanently (the followed resource no longer exists on the server); the caller reports it in its own words. */
  ongone: ((why: "gone" | "forbidden") => void) | null; // gone: the server has no such stream; forbidden: not visible to this account (a 403 not about the terms)
  /** The login expired: following has stopped and the gate asks for it again; the followed resource is unaffected. */
  onlogin: (() => void) | null;
  /** The stream is open again after it was lost (the browser's own reconnect, or one opened again here after a stall
   * or a refusal). The events that followed are resent, but what the stream follows may have changed while it was
   * down in ways no event says (a server restart): the caller checks it against the server here. */
  onresume: (() => void) | null;
  /** Stops following (normal completion): the stream is closed and nothing further is reported. */
  close(): void;
  /** Opens the stream again now, resuming after the last event received (nothing is resent): what the server says from
   * here on is said in the page's language now (switching the language, state/language.ts). */
  reopen(): void;
}

/** Reconnect delay for a refused stream: the stream's own retry hint (farm.py sends `retry: 2000`), then increasing. */
const AGAIN_MS = [2000, 5000, 15000];

/** A connection silent this long is treated as dropped.
 *
 * An idle server sends a `ping` event every 15 s (lab2shot/server/farm.py KEEPALIVE_S), so a live connection is never
 * silent for long. The browser reconnects only when a connection breaks, but WSL2 forwarding, proxies and sleep/wake
 * can leave a connection that neither breaks nor errors yet carries nothing: `onerror` never fires and the page would
 * wait forever (nothing more can be submitted, with no notice). So this module keeps its own timer: nothing received
 * (pings included) for this long counts as a drop, and the stream is closed and opened again. 45 s is three
 * heartbeats, leaving room for jitter. */
const SILENT_MS = 45_000;

// push streams currently lost (dropped, not yet reconnected): the top bar shows 「已断开」 from this (editor/Chrome.tsx
// TransferRate); a stream leaves the set when it reconnects or is no longer followed
const streamsLost = new Set<EventFollowing>();
export const pushLost = (): boolean => streamsLost.size > 0;

/** `probe`: the address asked why a stream was refused — a light address that opens no stream (the job's status). A
 * GET of the stream address itself would, on success, open a real stream, take one of the stream slots and resend
 * every event from the start; it is used only when no probe is given. */
export function followEvents(url: string, probe: string = url): EventFollowing {
  let stopped = false;
  let tries = 0;
  let es: EventSource | undefined;
  let lastId = ""; // the last event received: where a stream opened again here resumes
  const out: EventFollowing = {
    onmessage: null,
    ongone: null,
    onlogin: null,
    onresume: null,
    close: () => stop(),
    reopen: () => (stopped ? undefined : open()),
  };
  /** Ends following, and is the only place that does. Also clears the silence timer, which would otherwise still fire
   * after the end (returning at once on `stopped`: wasted work only). */
  const stop = () => {
    stopped = true;
    streamsLost.delete(out);
    clearTimeout(silence);
    es?.close();
  };
  let silence: ReturnType<typeof setTimeout> | undefined;
  let lost = false; // the connection dropped (an error, a stall, a refusal): its next open is a resume (onresume)
  const drop = () => ((lost = true), streamsLost.add(out));
  const heard = () => {  // the only liveness test: an event, a ping or a reconnect restarts the timer; nothing within SILENT_MS means dead
    clearTimeout(silence);
    if (stopped) return;
    silence = setTimeout(() => {
      if (stopped) return;
      drop(); // dead connection: close and reopen (open closes the old one first), resuming after the last event received
      open();
    }, SILENT_MS);
  };
  const open = () => {
    es?.close(); // close the old one first: one following has at most one connection at any time
    if (loginEnded()) {
      // the login has ended: the page sends nothing (the server counts 401s toward blocking the address); stop, and the
      // caller follows again after the new login
      stop();
      return queueMicrotask(() => out.onlogin?.()); // on the next tick: a following just created has no onlogin attached yet
    }
    es = new EventSource(lastId ? `${url}${url.includes("?") ? "&" : "?"}last_event_id=${encodeURIComponent(lastId)}` : url);
    heard();
    const now = es;
    now.onopen = () => {
      heard();
      tries = 0; // connected: the next refusal starts again from the shortest delay (also for a stream that only receives pings)
      if (!lost) return;
      lost = false;
      streamsLost.delete(out);
      out.onresume?.();
    };
    // an idle server sends a named `ping` event every 15 s (lab2shot/server/farm.py KEEPALIVE_S): it only keeps the
    // connection alive and is not passed to the caller
    now.addEventListener("ping", heard);
    now.onmessage = (e) => {
      noteDown(textBytes(e.data)); // the top bar's live traffic (platform/traffic.ts)
      heard();
      if (e.lastEventId) lastId = e.lastEventId;
      out.onmessage?.(e.data, e.lastEventId);
    };
    now.onerror = () => {
      if (stopped) return;
      drop();
      // CONNECTING: the connection dropped or the server is restarting; the browser reconnects by itself from the last event's id
      if (now.readyState === EventSource.CONNECTING) return heard();
      clearTimeout(silence); // CLOSED: nothing arrives any more; what happens next is the probe's to decide
      void refused(); // the server refused to open the stream again
    };
  };
  open();

  async function refused(): Promise<void> {
    let status = 0; // 0: the request never reached the server
    let terms = false; // a 403 for terms not agreed (awaits a login like a 401), not for a right
    try {
      const r = await answer(probe); // a 401 here is already reported to the gate (the single 401 policy, http.ts)
      status = r.status;
      if (status === 403) terms = !!((await r.clone().json().catch(() => null)) as { terms?: unknown } | null)?.terms;
      void r.body?.cancel(); // only whether the stream opens is checked, not its content
    } catch (e) {
      // the login has already ended: answer() refuses before sending (the page sends nothing until a new login), which
      // is a 401 as well; without this the stream would be opened again every few seconds, each time a real 401
      if (e instanceof ApiError && e.status === 401) status = 401;
      /* otherwise nothing received: the stream is still retried below */
    }
    if (stopped) return;
    // three kinds of refusal (the same split as transfer/frameStore.ts; "awaits a login" is http.ts awaitsLogin):
    //   401, or a 403 for terms not agreed: answer has handed it to the gate / terms page: stop, and the caller
    //     follows again after the new login (onlogin);
    //   404 / 410, any other 403 (not this account's, or a right it lacks: a new login changes nothing, and reopening
    //     only piles up 403s): cannot be followed, end (ongone);
    //   anything else (429: this account's streams are full, 408, 5xx, never reached the server): says nothing about
    //     the job being gone; back off and reopen.
    if (status === 401 || (status === 403 && terms)) {
      stop();
      return void out.onlogin?.();
    }
    if (status === 404 || status === 410 || status === 403) {
      stop();
      return void out.ongone?.(status === 403 ? "forbidden" : "gone");
    }
    setTimeout(() => !stopped && open(), AGAIN_MS[Math.min(tries, AGAIN_MS.length - 1)]);
    tries++;
  }

  return out;
}
