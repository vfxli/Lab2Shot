/** The single request layer for the editor and the admin pages. Every request goes through `answer`
 * (webui/tests/http.test.ts keeps `fetch` out of all other modules), so an expired login is detected the same way
 * wherever it occurs. Imports nothing from the page. */

import { CODE } from "../messages/format";
import { MessageError, fromServer, msg } from "../messages/message";

/** Dispatched when the server requires a new login, with the kicked-out notice as detail when that is the cause (else
 * null); also dispatched by the gate after a successful login over the page (state/session.ts then rereads the login). */
export const NEED_LOGIN = "lab2shot:need-login";
export const LOGGED_IN = "lab2shot:logged-in";

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
 * gate, which contains none of the page's addresses (tests/test_outsider.py). */
export function refusedLogin(body: { login?: boolean; admin?: boolean; kicked?: unknown; detail?: unknown; code?: unknown } | null): void {
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

/** A new login over the page (reported by the gate, here and nowhere else): requests are sent again, and listeners of the
 * login are notified (LOGGED_IN), in that order, so that their next requests are sent. */
export function loggedIn(): void {
  ended = null;
  window.dispatchEvent(new Event(LOGGED_IN));
}

/** For the tests, and for a page that rereads the login by itself. */
export const loginEnded = (): boolean => ended !== null;

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

/** Sends a request and returns the response as received (refusals included), applying the 401 policy; refusals and
 * requests that never arrived are recorded (failedRequests). */
export async function answer(url: string, init?: RequestInit): Promise<Response> {
  const t = Date.now();
  const method = (init?.method ?? "GET").toUpperCase();
  const where = cut(url.startsWith(location.origin) ? url.slice(location.origin.length) : url, 300);
  if (ended && !gate) throw new ApiError(ended.message, ended.code, 401, { login: true });
  let r: Response;
  try {
    r = await fetch(url, init);
  } catch (e) {
    if ((e as Error).name !== "AbortError") keepFailed({ t, method, url: where, status: 0, message: cut((e as Error).message ?? String(e), 500), ms: Date.now() - t });
    throw e;
  }
  if (!r.ok && r.type !== "opaque") {
    const text = await r.clone().text().catch(() => "");
    let body: { detail?: unknown; login?: boolean; admin?: boolean; kicked?: unknown } | null = null;
    try {
      body = JSON.parse(text);
    } catch {
      /* not JSON: kept as received */
    }
    keepFailed({ t, method, url: where, status: r.status, message: cut(String(body?.detail ?? text), 500), ms: Date.now() - t });
    if (r.status === 401) refusedLogin(body);
  }
  return r;
}

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

/** 一段原始字节（按通道读取的路径：transfer/plane.ts 的 L2C1，gzip 由浏览器解压）。 */
export async function bytes(url: string, init?: RequestInit): Promise<ArrayBuffer> {
  return (await request(url, init)).arrayBuffer();
}

/** Whether anything responds at `url` (another address the server moved to: the browser may not expose the response to
 * the page, only the fact that one arrived). A request that never arrives throws. */
export async function reachable(url: string): Promise<true> {
  await answer(url, { mode: "no-cors", cache: "no-store" });
  return true;
}
