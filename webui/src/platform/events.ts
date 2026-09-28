import { answer } from "./http";

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
  ongone: (() => void) | null;
  /** The login expired: following has stopped and the gate asks for it again; the followed resource is unaffected. */
  onlogin: (() => void) | null;
  /** Stops following (normal completion): the stream is closed and nothing further is reported. */
  close(): void;
}

/** Reconnect delay for a refused stream: the stream's own retry hint (farm.py sends `retry: 2000`), then increasing. */
const AGAIN_MS = [2000, 5000, 15000];

/** 长时间无数据的连接视为已断开。
 *
 * 服务器每 15 秒发送一次 keep-alive（lab2shot/server/farm.py KEEPALIVE_S），因此连接不应长时间无数据。
 * 浏览器只在连接断开时重连，而 WSL2 转发、代理、睡眠唤醒可能使连接既不断开、不报错，也不再传输数据；
 * 此时 `onerror` 不会触发，页面将无限等待（无法再提交，也没有任何提示）。因此本模块自行计时：
 * 超过该时长未收到任何数据（含注释行）即视为断开，关闭后重新打开。取 45 秒，即三次心跳，为抖动留出余量。 */
const SILENT_MS = 45_000;

export function followEvents(url: string): EventFollowing {
  let stopped = false;
  let tries = 0;
  let es: EventSource;
  let lastId = ""; // the last event received: where a stream opened again here resumes
  const out: EventFollowing = {
    onmessage: null,
    ongone: null,
    onlogin: null,
    close: () => stop(),
  };
  /** 结束对该连接的跟随，仅此一处。同时清除无数据计时器：否则它在连接结束后仍会触发
   * （发现 stopped 即返回，只是空耗）。 */
  const stop = () => {
    stopped = true;
    clearTimeout(silence);
    es.close();
  };
  let silence: ReturnType<typeof setTimeout> | undefined;
  const heard = () => {  // 收到任何数据（事件、心跳注释、重连成功）均表明连接存活
    clearTimeout(silence);
    if (stopped) return;
    silence = setTimeout(() => {
      if (stopped) return;
      es.close();  // 连接已僵死：关闭后重新打开，从收到的最后一个事件之后继续（见 open）
      open();
    }, SILENT_MS);
  };
  const open = () => {
    es = new EventSource(lastId ? `${url}${url.includes("?") ? "&" : "?"}last_event_id=${encodeURIComponent(lastId)}` : url);
    heard();
    es.onopen = heard;
    // 心跳为注释行（`: keep-alive`），浏览器不会作为 message 交付，但其到达会刷新连接状态；
    // 因此除 message 外，每次重连成功（onopen）时也重新计时
    es.onmessage = (e) => {
      tries = 0;
      heard();
      if (e.lastEventId) lastId = e.lastEventId;
      out.onmessage?.(e.data, e.lastEventId);
    };
    es.onerror = () => {
      if (stopped) return;
      // CONNECTING: the connection dropped or the server is restarting; the browser reconnects by itself from the last event's id
      if (es.readyState === EventSource.CONNECTING) return heard();
      clearTimeout(silence); // CLOSED: nothing arrives any more; what happens next is the probe's to decide
      void refused(); // the server refused to open the stream again
    };
  };
  open();

  async function refused(): Promise<void> {
    let status = 0; // 0: the request never reached the server
    try {
      const r = await answer(url); // a 401 here is already reported to the gate (the single 401 policy, http.ts)
      status = r.status;
      void r.body?.cancel(); // only whether the stream opens is checked, not its content
    } catch {
      /* nothing received: the stream is still retried below */
    }
    if (stopped) return;
    if (status === 401) {
      stop(); // the login expired: this does not end the followed resource, which is followed again after a new login
      return void out.onlogin?.();
    }
    if (status >= 400 && status < 500) {
      stop(); // the server answered and has no such stream: ended permanently
      return void out.ongone?.();
    }
    // Retried: a stream that still opens (the server's refusal was transient), a request that never reached the server
    // (status 0: the network is down), or a server that is failing or restarting (5xx). None of these indicates that the
    // followed resource is gone, so none of them ends the stream.
    setTimeout(() => !stopped && open(), AGAIN_MS[Math.min(tries, AGAIN_MS.length - 1)]);
    tries++;
  }

  return out;
}
