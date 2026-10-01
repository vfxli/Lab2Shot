import { useSyncExternalStore } from "react";
import type { RestartState, ServerInfo } from "../api";
import { json, sawAccount } from "../platform/http";
import { startPolling } from "../platform/poll";

/** 每个页面看到的「服务本身」（GET /api/server，lab2shot/server/settings.py）：哪个服务进程在回答（重启后 `boot`
 * 不同）、是否有待执行的重启、是否还在回答，以及这个浏览器此刻登录的是哪个账号（不是这一页的账号：页面重新打开）。
 * 整个页面只有这一个轮询：有待执行的重启或服务器不回答时每秒一次；否则每 5 秒一次，回复一直不变时（无响应体的 304）
 * 逐渐放慢，最多 60 秒一次。 */


interface ServerWatch {
  info: ServerInfo | null; // 最近一次响应
  down: boolean; // 最近一次请求没有得到响应
  restarted: boolean; // 回答的服务进程已不是页面载入时的那一个
  newPage: boolean; // 并且它提供的是另一版页面
}

let watch: ServerWatch = { info: null, down: false, restarted: false, newPage: false };
const FED_FRESH_MS = 90_000;
const listeners = new Set<() => void>();
/** 本模块的内部记录，集中于一处，不属于页面读取的状态（页面状态为 `watch`）：
 * `first` 为页面首次见到的服务版本；`fedAt` 为其他模块最近一次提供「服务本身」信息的时刻（由 editor 的 /api/load
 * 附带返回）。有外部提供时本轮询降至很低的频率，避免为同一信息发送重复请求；外部停止提供后（切换页面、退出登录）
 * 恢复原有频率。`started` / `polling` 为轮询本身。 */
const own: { first: ServerInfo | null; fedAt: number; started: boolean; polling: { now: () => void } | null } = {
  first: null, fedAt: 0, started: false, polling: null,
};

/** 接收其他模块获得的「服务本身」信息（editor/Chrome.tsx 的 /api/load 附带返回），视为一次响应。
 * 「是否已重启、界面是否已更新」只在一处判断，不在两处重复实现。 */
export function noteServer(info: ServerInfo): void {
  own.fedAt = Date.now();
  answered(info);
}

/** 一次「服务本身」的回复，不论来自本轮询还是 /api/load：是否已重启、界面是否已更新；回复里登录的账号已不是这一页的，
 * 页面重新打开（platform/http.ts sawAccount：别的窗口换了账号）。 */
function answered(info: ServerInfo): void {
  if (sawAccount(info.account)) return;
  own.first ??= info;
  set({ info, down: false, restarted: info.boot !== own.first.boot, newPage: info.ui !== own.first.ui });
}

function set(next: ServerWatch): void {
  watch = next;
  listeners.forEach((f) => f());
}

/** 有待执行的重启或服务器不回答时每秒一次；否则每 5 秒一次，回复一直不变时逐渐放慢，最多 60 秒一次（浏览器带上
 * ETag：没变的回复是 304）。
 *
 * 空闲时每分钟一次：一次往返的字节主要是请求头和 cookie，而非响应本身（304 的响应体为空），
 * 因此节省流量只能依靠减少请求（上行带宽有限）。这不会推迟对重启的发现：服务器一旦停止，本请求立即出错 →
 * `down` → `hurried()` → 每秒一次。 */
function begin(): void {
  const hurried = () => watch.down || !!watch.info?.restart;
  own.polling = startPolling<ServerInfo>({
    read: () => json<ServerInfo>("GET", "/api/server"),
    every: () => (hurried() ? 1000 : Date.now() - own.fedAt < FED_FRESH_MS ? 60_000 : 5000),
    afterError: 1000, // 不回答时每秒一次：服务器一回来页面立即知道
    // 有外部提供时降至每五分钟一次（仅作兜底）；否则每分钟一次
    slowest: () => (hurried() ? 1000 : Date.now() - own.fedAt < FED_FRESH_MS ? 300_000 : 60_000),
    onValue: answered,
    onError: () => set({ ...watch, down: true }),
  });
}

function subscribe(f: () => void): () => void {
  if (!own.started) {
    own.started = true;
    begin();
  }
  listeners.add(f);
  return () => listeners.delete(f);
}

/** 本页最近一次看到的服务器（变化时重渲染）。 */
export function useServer(): ServerWatch {
  return useSyncExternalStore(subscribe, () => watch);
}

/** 服务本身的最近一次响应（不经过 React；本机代理需读取管理员设定的两个数值，`transfer/localProxy`）。 */
export const serverNow = (): ServerInfo | null => watch.info;

/** 服务本身每次响应时通知订阅者（不经过 React，也不启动轮询：只监听已有的请求，页面上始终有 `useServer` /
 * `noteServer` 在提供数据）。本机代理在管理员设定的档位变化后需重新扫描磁盘（`transfer/localProxy`）。 */
export function onServerChange(f: () => void): () => void {
  listeners.add(f);
  return () => void listeners.delete(f);
}

/** 立即再问一次（请服务器重启之后）。 */
export const pollServer = () => own.polling?.now();

/** 重启换了地址（另一个端口、切换 HTTPS）时服务器接下来在哪里回答；null：仍是当前地址。 */
export function nextOrigin(r: RestartState | null | undefined): string | null {
  if (!r) return null;
  const scheme = r.https ? "https:" : "http:";
  const port = String(r.port);
  const here = window.location.port || (window.location.protocol === "https:" ? "443" : "80");
  return scheme === window.location.protocol && port === here ? null : `${scheme}//${window.location.hostname}:${port}`;
}
