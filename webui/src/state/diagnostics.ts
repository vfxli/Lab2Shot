import { clientInfo } from "../platform/client";
import { failedRequests, type FailedRequest } from "../platform/http";
import { pageErrors, type PageError } from "../platform/pageErrors";
import { recentLog } from "./log";
import { t } from "../i18n/t";

/** Diagnostics that the feedback report sends in addition to the user's text, collected in the page: the browser and
 * the machine, the latest page log entries, and (held here in memory since the page opened) its errors with their
 * locations and the requests the server refused or that never reached it. The server adds its own
 * (lab2shot/site/feedback.py) and removes secrets; the dialog shows everything before it is sent. */


const LOG_ENTRIES = 200;
const GRAPH_BYTES = 1 << 20; // a graph larger than this is sent as its node list only



/** The graph as sent with feedback: the whole graph (it refers to files only by name), or, when it is very large, its
 * nodes' ids, types and labels. */
export function graphForFeedback(graph: unknown): unknown {
  const text = JSON.stringify(graph);
  if (text.length <= GRAPH_BYTES) return graph;
  const nodes = ((graph as { nodes?: { id: string; type: string; label?: string }[] }).nodes ?? []).map((n) => ({ id: n.id, type: n.type, label: n.label }));
  return { too_big: text.length, nodes };
}

export interface Diagnostics {
  collected: number;
  page: string;
  browser: Record<string, unknown>;
  client: Record<string, string>;
  log: { t: number; level: string; text: string; count?: number; last?: number }[];
  errors: PageError[];
  requests: FailedRequest[];
  jobs: { id: string; target?: string }[];
  [more: string]: unknown; // page-specific additions: the editor's graph, selection, frame, …
}

/** The page's current diagnostics; `more`: page-specific additions (the editor: graph, selection, frame, its job). */
export function collectDiagnostics(more: Record<string, unknown> = {}): Diagnostics {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { platform?: string; brands?: { brand: string; version: string }[] } };
  const memory = (performance as Performance & { memory?: { usedJSHeapSize: number; jsHeapSizeLimit: number } }).memory;
  return {
    collected: Date.now(),
    page: location.pathname + location.search,
    browser: {
      userAgent: navigator.userAgent,
      brands: nav.userAgentData?.brands?.map((b) => `${b.brand} ${b.version}`).join(", ") ?? "",
      platform: nav.userAgentData?.platform ?? navigator.platform,
      language: navigator.language,
      languages: [...navigator.languages],
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      screen: `${screen.width}×${screen.height}`,
      window: `${innerWidth}×${innerHeight}`,
      pixelRatio: devicePixelRatio,
      cores: navigator.hardwareConcurrency,
      memoryGb: nav.deviceMemory ?? null,
      heapMb: memory ? Math.round(memory.usedJSHeapSize / 2 ** 20) : null,
      webgl: webgl(),
      online: navigator.onLine,
    },
    client: clientInfo(),
    log: recentLog(LOG_ENTRIES),
    errors: pageErrors(),
    requests: failedRequests(),
    jobs: [],
    ...more,
  };
}

let gpu: string | null = null;

/** The browser's graphics card as WebGL names it (the 3D view runs on it). */
function webgl(): string {
  if (gpu !== null) return gpu;
  try {
    const gl = document.createElement("canvas").getContext("webgl");
    const ext = gl?.getExtension("WEBGL_debug_renderer_info");
    gpu = gl ? String(ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) : t("ui.state.no_webgl");
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    gpu = "";
  }
  return gpu;
}
