/** 页面与服务器之间的全部流量（累计字节）：只在所有网络都经过的几处记——platform/http.ts 的 answer（每个 fetch 的
 * 请求体与应答体，应答边到边记）、platform/events.ts（推送 SSE 的每条消息）、transfer/uploads.ts 的 xhr（上传素材的
 * 分段，按上传进度记）。顶栏（editor/Chrome.tsx TransferRate）按 transfer/rate.ts 的 Rate 算成每秒多少。只是计数，
 * 不改任何请求。计的是页面拿到 / 交出的字节（gzip 已由浏览器解开），不含请求头；浏览器 HTTP 缓存直接给的应答
 * （带版本的帧地址是 immutable）也算在「收到」里——应答流上分不出它是否真的走了网络。 */

let up = 0;
let down = 0;
export const noteUp = (bytes: number): void => void (up += Math.max(0, bytes));
export const noteDown = (bytes: number): void => void (down += Math.max(0, bytes));
export const trafficTotals = (): { up: number; down: number } => ({ up, down });

const encoder = typeof TextEncoder !== "undefined" ? new TextEncoder() : null;
/** 一段文字按 UTF-8 的字节数（JSON 请求体、推送消息）。 */
export const textBytes = (s: string): number => (encoder ? encoder.encode(s).byteLength : s.length);
