/** A short hash for cache keys (FNV-1a, 32 bits, in base 36): a changed input gives another key. Not for security.
 * Text is hashed by its characters; numbers (a table's values) as they are, every `step`-th one (a large table is
 * sampled). The one such hash of the page. */
export function shortHash(data: string | ArrayLike<number>, step = 1): string {
  let h = 2166136261;
  if (typeof data === "string") for (const ch of data) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  else for (let i = 0; i < data.length; i += step) h = Math.imul(h ^ data[i], 16777619);
  return (h >>> 0).toString(36);
}
