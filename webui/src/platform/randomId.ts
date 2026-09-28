/** A random identifier: `bytes` random bytes as lowercase hex (16 bytes: 128 bits, 32 characters). The one way the
 * page makes an id. crypto.getRandomValues exists in every context; crypto.randomUUID only in a secure one (https or
 * localhost), so a page opened over plain http from another computer has none. */
export function randomId(bytes = 16): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(bytes)), (b) => b.toString(16).padStart(2, "0")).join("");
}
