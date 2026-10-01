/** The single request layer for the editor and the admin pages. Every request goes through `answer` (no other
 * module calls `fetch`), or through `upload` when its sending progress is needed (XMLHttpRequest: fetch has no upload
 * progress), so an expired login is detected, and stops further requests, the same way wherever it occurs; so is a
 * 429, which quiets every request of the page at once. Imports nothing from the page. */

import { CODE } from "../messages/format";
import { MessageError, fromServer, msg } from "../messages/message";
import { noteDown, noteUp, textBytes } from "./traffic";
import { backoff } from "./backoff";

/** Dispatched when the server requires a new login, with the kicked-out notice as detail when that is the cause (else
 * null); also dispatched by the gate after a successful login over the page (state/session.ts then rereads the login). */
export const NEED_LOGIN = "lab2shot:need-login";

/** How long a request may go without progress before it is given up (a tunnel may hold a dead connection open
 * indefinitely): an upload's part (`upload` stallMs, transfer/uploads.ts) and a fetch of bytes (transfer/frameStore.ts). */
export const STALL_MS = 30_000;
export const LOGGED_IN = "lab2shot:logged-in";

/** Dispatched when the server refuses a request because the account has not agreed to the current 用户协议 and
 * 隐私政策 (lab2shot/server/access.py marks that 403 `terms`): the gate asks for it over the page (gate.tsx). */
export const NEED_TERMS = "lab2shot:need-terms";

/** Dispatched when the server reports that administrator rights are absent or revoked: the admin page asks again. */
export const SIGNED_OUT = "lab2shot:signed-out";

/** A request the server refused: the server's message (`said`: its code and text, lab2shot/messages; a refusal without
 * a code becomes E-REQUEST-REFUSED with the status and the response text), the HTTP status, and the full response body
 * (`body`: a form's field errors, a 409's reason). */
export class ApiError extends MessageError {
  readonly code: string;
  readonly status: number;
  readonly body: Record<string, unknown> | null;
  constructor(message: string, code: string, status: number, body: Record<string, unknown> | null = null) {
    super(CODE.test(code) ? fromServer({ code, text: message }) : msg("E-REQUEST-REFUSED", { status, detail: message }));
    this.message = message; // the server's text, unchanged
    this.code = code;
    this.status = status;
    this.body = body;
  }
}

/** The single 401 policy, based on the server's marking (lab2shot/server/access.py marks every 401 `login` or `admin`):
 * an expired login (the account was disabled or expired, or logged in elsewhere) is reported to the gate, which asks for
 * it over the page (gate.tsx); expired administrator rights are reported to the admin page. Also called by the uploads,
 * which use XMLHttpRequest for progress reporting (transfer/uploads.ts). No route is named here: this file belongs to the
 * gate, which contains none of the page's addresses. */
export function refusedLogin(body: { login?: boolean; admin?: boolean; kicked?: unknown; detail?: unknown; code?: unknown } | null, sentIn = logins): void {
  // a 401 sent before a new login and answered after it (carrying the old cookie) speaks of the previous login: it does
  // not end this one
  if (sentIn !== logins) return;
  if (body?.login) {
    ended = { message: String(body.detail ?? ""), code: String(body.code ?? "") };
    window.dispatchEvent(new CustomEvent(NEED_LOGIN, { detail: body.kicked ?? null }));
  } else if (body?.admin) window.dispatchEvent(new Event(SIGNED_OUT));
}

// Once the server has reported that the login ended, the page sends no further requests until a new login over the page.
// Otherwise a tab left open would keep polling the queue and the load every few seconds, each receiving a 401; the server
// counts these and blocks the address from logging in (an office behind one address would be blocked as a whole).
// Such requests are answered here with the server's last response. The gate's own requests (login status, the login
// itself) are exempt: see fromGate.
let ended: { message: string; code: string } | null = null;
let gate = false;
let logins = 0; // which login this is (loggedIn adds one): noted when a request is sent; a 401 counts only against the same login

/** Runs the gate's own requests, which are the ones that end this state: they reach the server even after the login has
 * ended (the mark holds for the synchronous start of `run`, which is where a request is dispatched). */
export function fromGate<T>(run: () => Promise<T>): Promise<T> {
  gate = true;
  try {
    return run();
  } finally {
    gate = false;
  }
}

/** The login has ended and no new one has happened: the page sends no requests (`answer` / `upload` answer 401 at
 * once). Whoever waits on this waits only for LOGGED_IN. */
export const loginEnded = (): boolean => !!ended && !gate;

// the server says the terms are not agreed (a 403 with `terms`): the terms page has been called up (NEED_TERMS), and
// agreeing calls loggedIn. Until then uploads wait for it rather than retrying on a timer
let termsOwed = false;
const owesTerms = (): void => {
  termsOwed = true;
  window.dispatchEvent(new Event(NEED_TERMS));
};

/** Waiting for a login (the login ended, or the terms are not agreed): whoever waits waits only for LOGGED_IN, never
 * retrying on a timer. */
export const awaitingLogin = (): boolean => loginEnded() || termsOwed;

/** A new login over the page (reported by the gate, here and nowhere else): requests are sent again, and listeners of the
 * login are notified (LOGGED_IN), in that order, so that their next requests are sent. */
export function loggedIn(): void {
  logins++;
  ended = null;
  termsOwed = false;
  window.dispatchEvent(new Event(LOGGED_IN));
}

// Whose page this is. A browser's login is one for all its tabs and windows: when one of them logs out and in as another
// account, every other tab's requests go out as that account from then on, while what those tabs show (the account
// menu, the graph, the working copy, the jobs) is still the first one's. A page therefore never goes on under another
// account: as soon as it learns that the login changed, it opens again (the gate then shows the login, or the page of
// the account logged in now). It learns it from the tab that changed the login (a message on CHANNEL), and, for a login
// changed where no message reaches (another browser window of the same profile that is not open on this site, a
// program sharing the cookie), from the account every answer about the server names (GET /api/server and the
// /api/load that carries it: `account`, lab2shot/server/settings.py server_now).
let owner: number | null | undefined; // undefined: not known yet (the gate has not read the login)
const CHANNEL = "lab2shot:account";
const channel = typeof BroadcastChannel === "undefined" ? null : new BroadcastChannel(CHANNEL);

/** The gate says whose page this is, when it has read the login (null: the login page). */
export function belongsTo(id: number | null): void {
  owner = id;
}

/** An answer names the account this browser is logged in as: another account than the page's opens the page again.
 * No account (the login ended: expired, or ended by a login elsewhere) is not a change of account: the gate asks for
 * the login over the page, and the page underneath stays as it is (gate.tsx). True: the page is opening again. */
export function sawAccount(id: number | null): boolean {
  const other = owner !== undefined && id !== null && id !== owner;
  if (other) window.location.reload();
  return other;
}

/** This tab logged in (the account) or out (null): every other tab of this site in this browser is told, and opens
 * again unless it already is that account's page (a logout, too, is the whole browser's: 退出登录). */
export function changedAccount(id: number | null): void {
  channel?.postMessage(id);
}

channel?.addEventListener("message", (e: MessageEvent<number | null>) => {
  if (owner !== undefined && e.data !== owner) window.location.reload();
});

/** A request the server refused or that never reached it, kept for the feedback report (state/diagnostics.ts reads them). */
export interface FailedRequest {
  t: number;
  method: string;
  url: string;
  status: number; // 0: the request never reached the server
  message: string;
  ms: number;
}

const KEEP_FAILED = 50;
const failed: FailedRequest[] = [];
const cut = (s: string, n: number) => (s.length > n ? `${s.slice(0, n)}…` : s);
const keepFailed = (f: FailedRequest) => (failed.push(f), failed.length > KEEP_FAILED && failed.shift());

/** The failed requests since the page opened, oldest first. */
export const failedRequests = (): FailedRequest[] => [...failed];

/** The bytes of a request body (the top bar's live traffic, platform/traffic.ts): text, a Blob, a byte array; anything
 * else (no body) is 0. */
function bodyBytes(body: BodyInit | null | undefined): number {
  if (!body) return 0;
  if (typeof body === "string") return textBytes(body);
  if (body instanceof Blob) return body.size;
  if (body instanceof ArrayBuffer) return body.byteLength;
  if (ArrayBuffer.isView(body)) return body.byteLength;
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) return textBytes(body.toString());
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    let n = 0;
    body.forEach((v, k) => (n += textBytes(k) + (typeof v === "string" ? textBytes(v) : v.size)));
    return n;
  }
  return 0; // a stream: its size is not known up front
}

/** The response as it is, except that its body's bytes are counted as they stream through (platform/traffic.ts): a
 * large chunk (a block of a 3D whole-range cache) is counted while it downloads, so the speed does not jump. A response
 * without a body (204 / 304) or an opaque cross-origin one is returned untouched. */
function counted(r: Response): Response {
  if (!r.body || r.type === "opaque") return r;
  const meter = new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, c) {
      noteDown(chunk.byteLength);
      c.enqueue(chunk);
    },
  });
  const out = new Response(r.body.pipeThrough(meter), { status: r.status, statusText: r.statusText, headers: r.headers });
  for (const k of ["url", "type", "redirected"] as const) Object.defineProperty(out, k, { value: r[k] });
  return out;
}

// ---- the server says "too many requests" (429): the whole page goes quiet
// A 429 is counted per session (past a number of requests within 10 minutes the server blocks for a while, answering
// every request 429 meanwhile), so it is not about any one address: once any request gets a 429, every request of the
// page (frames, whole-range fetches, prefetch, uploads, status and queue polling — all go through answer / upload)
// waits for the server's Retry-After (without one, backing off by the count in a row: 2 s, doubling, at most 1 min)
// before being sent. The gate's own requests do not wait.
let quietUntil = 0;
let tooFast = 0; // 429s in a row (any other answer resets it)

function heard(status: number, retryAfter: string | null): void {
  if (status !== 429) return void (tooFast = 0);
  tooFast++;
  const said = Number(retryAfter);
  const ms = Number.isFinite(said) && said > 0 ? said * 1000 : backoff(tooFast, 2000, 60_000);
  quietUntil = Math.max(quietUntil, Date.now() + ms);
}

/** How much longer to stay quiet (ms; 0: requests may go). */
export const quietFor = (): number => Math.max(0, quietUntil - Date.now());

/** Waits until requests may go (throws AbortError when the caller gives up). */
async function untilQuiet(signal?: AbortSignal | null): Promise<void> {
  while (quietFor() > 0) {
    await new Promise<void>((done, stop) => {
      const t = setTimeout(finish, quietFor());
      function finish() {
        clearTimeout(t);
        signal?.removeEventListener("abort", quit);
        done();
      }
      function quit() {
        clearTimeout(t);
        stop(new DOMException("aborted", "AbortError"));
      }
      if (signal?.aborted) return quit();
      signal?.addEventListener("abort", quit, { once: true });
    });
  }
}

/** Sends a request and returns the response as received (refusals included), applying the 401 policy; refusals and
 * requests that never arrived are recorded (failedRequests). */
export async function answer(url: string, init?: RequestInit): Promise<Response> {
  const t = Date.now();
  const method = (init?.method ?? "GET").toUpperCase();
  const where = cut(url.startsWith(location.origin) ? url.slice(location.origin.length) : url, 300);
  if (ended && !gate) throw new ApiError(ended.message, ended.code, 401, { login: true });
  const fromTheGate = gate;
  const sentIn = logins;
  let r: Response;
  try {
    if (!fromTheGate && quietFor() > 0) await untilQuiet(init?.signal);
    noteUp(bodyBytes(init?.body)); // counted when it goes out: a request given up while the page is quiet sends nothing
    r = counted(await fetch(url, init));
  } catch (e) {
    if ((e as Error).name !== "AbortError") keepFailed({ t, method, url: where, status: 0, message: cut((e as Error).message ?? String(e), 500), ms: Date.now() - t });
    throw e;
  }
  heard(r.status, r.headers.get("Retry-After"));
  if (!r.ok && r.type !== "opaque") {
    const text = await r.clone().text().catch(() => "");
    let body: { detail?: unknown; login?: boolean; admin?: boolean; kicked?: unknown; terms?: number } | null = null;
    try {
      body = JSON.parse(text);
    } catch {
      /* not JSON: kept as received */
    }
    keepFailed({ t, method, url: where, status: r.status, message: cut(String(body?.detail ?? text), 500), ms: Date.now() - t });
    if (r.status === 401) refusedLogin(body, sentIn);
    else if (r.status === 403 && body?.terms) owesTerms();
  }
  return r;
}

/** The upload path: one request with upload progress (XMLHttpRequest; the part uploads of transfer/uploads.ts), under
 * the same login gate, failure record, 401 handling and traffic count as `answer`. After the login has ended it is not
 * sent (status 401, nothing sent); it goes after the new login. Aborted when nothing progresses for `stallMs` (status
 * 0: a tunnel may hold a dead connection open indefinitely); `started` receives the request itself (the caller aborts
 * it when stopping). `signal`: the caller gave up, perhaps while the page was still waiting out a 429 pause: then nothing
 * is sent at all (status 0, as an aborted request). Returns the server's status and response text; what the status means
 * is the caller's to decide. */
export function upload(method: string, url: string, body: Blob | string | null, o: { progress?: (sent: number) => void; started?: (x: XMLHttpRequest) => void; stallMs: number; signal?: AbortSignal }): Promise<{ status: number; text: string }> {
  if (ended && !gate) return Promise.resolve({ status: 401, text: JSON.stringify({ detail: ended.message, code: ended.code, login: true }) });
  const t = Date.now();
  const where = cut(url.startsWith(location.origin) ? url.slice(location.origin.length) : url, 300);
  const sentIn = logins;
  const gaveUp = { status: 0, text: "" };
  return untilQuiet(o.signal).then(() => new Promise<{ status: number; text: string }>((resolve) => {
    if (o.signal?.aborted) return resolve(gaveUp);
    const x = new XMLHttpRequest();
    o.started?.(x);
    let last = Date.now();
    const watch = window.setInterval(() => Date.now() - last > o.stallMs && x.abort(), 5000);
    x.open(method, url);
    if (typeof body === "string") x.setRequestHeader("Content-Type", "application/json");
    else if (body) x.setRequestHeader("Content-Type", "application/octet-stream");
    let counted = 0;
    x.upload.onprogress = (e) => ((last = Date.now()), noteUp(e.loaded - counted), (counted = e.loaded), o.progress?.(e.loaded));
    x.onprogress = () => (last = Date.now());
    x.onload = () => {
      window.clearInterval(watch);
      const text = x.responseText ?? "";
      noteDown(textBytes(text));
      heard(x.status, x.status === 429 ? x.getResponseHeader("Retry-After") : null);
      if (x.status >= 300) {
        let said: { detail?: unknown; login?: boolean; admin?: boolean; kicked?: unknown; terms?: unknown } | null = null;
        try {
          said = JSON.parse(text);
        } catch {
          /* not JSON: a tunnel's own error page */
        }
        keepFailed({ t, method, url: where, status: x.status, message: cut(String(said?.detail ?? text), 500), ms: Date.now() - t });
        if (x.status === 401) refusedLogin(said, sentIn);
        else if (x.status === 403 && said?.terms) owesTerms(); // as in answer: call up the terms page; the upload continues once agreed
      }
      resolve({ status: x.status, text });
    };
    x.onerror = x.onabort = x.ontimeout = () => {
      window.clearInterval(watch);
      resolve({ status: 0, text: "" });
    };
    x.send(body);
  }), () => gaveUp); // given up during the pause: nothing went out
}

/** Whether only a new login changes this refusal: the login ended (401), or the terms are not agreed (a 403 with
 * `terms`: agreeing calls loggedIn). Any other 403 (a right this login lacks, not this account's) is not about the
 * login: a new login changes nothing, and the caller reports it as a refusal and may try again later. This is the
 * page's only test for "awaits a login" (frameStore, polling and the push streams all ask it). */
export const awaitsLogin = (e: unknown): boolean =>
  e instanceof ApiError && (e.status === 401 || (e.status === 403 && !!e.body?.terms));

/** The response, or the server's reason (`detail`) thrown as an ApiError. */
export async function ok(r: Response): Promise<Response> {
  if (r.ok) return r;
  const body = (await r.json().catch(() => null)) as Record<string, unknown> | null;
  throw new ApiError(String(body?.detail ?? `${r.status} ${r.url}`), String(body?.code ?? ""), r.status, body);
}

/** A request whose refusal is an error. */
export const request = async (url: string, init?: RequestInit): Promise<Response> => ok(await answer(url, init));

/** A JSON request and its JSON answer; `body` goes as JSON when given. */
export async function json<T>(method: "GET" | "POST" | "PUT" | "DELETE", url: string, body?: unknown, init?: RequestInit): Promise<T> {
  const sent = body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  return (await request(url, { ...init, method, ...sent })).json();
}

/** A picture or a file's bytes; a refusal is an ApiError. */
export async function blob(url: string, init?: RequestInit): Promise<Blob> {
  return (await request(url, init)).blob();
}

/** Raw bytes (the per-channel read path: transfer/plane.ts L2C1; gzip is undone by the browser). */
export async function bytes(url: string, init?: RequestInit): Promise<ArrayBuffer> {
  return (await request(url, init)).arrayBuffer();
}
