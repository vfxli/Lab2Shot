/** Whether a message already names the node a log line is about, read from its parameters (never from its words):
 * a template that writes the node itself ({node}, {source} … : W-INPUT-UNUSED, B-COOK-*) gets the node pointed at as a
 * parameter — its word kept by its key ({said, params: {name, type}}: graph/naming.ts nodeWord, the server's
 * engine/naming.py node_ref) or its name — and the log then writes the line as the message says it, without putting
 * the name in front of it a second time. `node`: the node's name (its id).
 *
 * A file of its own that imports nothing. */
export function namesNode(params: Readonly<Record<string, unknown>> | undefined, node: string): boolean {
  if (!node || !params) return false;
  return Object.values(params).some((v) => (typeof v === "string" ? v === node
    : !!v && typeof v === "object" && typeof (v as { said?: unknown }).said === "string" && (v as { params?: { name?: unknown } }).params?.name === node));
}

/** The name (id) of the node a log line is about, from how it points at it (a kept word, or its name). */
export const nodeNameOf = (node: unknown): string =>
  typeof node === "string" ? node : String((node as { params?: { name?: unknown } } | null)?.params?.name ?? "");
