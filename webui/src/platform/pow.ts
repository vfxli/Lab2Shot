/** The proof of work of the registration page (lab2shot/server/auth.py Challenges): find a number whose
 * sha256("<nonce>:<number>") starts with `bits` zero bits. The server checks an answer with one hash; finding one takes
 * about 2^bits tries, a second or two (server/register.py POW_BITS), which the page spends in a worker (powWorker.ts)
 * while the person types.
 *
 * SHA-256 is written out here for exactly this message: "<nonce>:<number>" is ASCII and always fits in one 64-byte
 * block (a 32-character nonce, a colon and at most 16 digits), so each try is one compression of one block — no
 * general hashing library, no crypto.subtle round trip per try (a promise each: far slower for millions of tries). */

const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
  0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);
const H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
const MAX_TEXT = 55; // what one block holds besides the padding and the length

/** The first two 32-bit words of sha256(one block holding `bytes`): enough for up to 64 zero bits. */
function head(bytes: Uint8Array, length: number, w: Uint32Array): [number, number] {
  for (let i = 0; i < 16; i++) w[i] = 0;
  for (let i = 0; i < length; i++) w[i >> 2] |= bytes[i] << (24 - 8 * (i & 3));
  w[length >> 2] |= 0x80 << (24 - 8 * (length & 3));
  w[15] = length * 8;
  for (let i = 16; i < 64; i++) {
    const a = w[i - 15], b = w[i - 2];
    const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
    const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
    w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
  }
  let [a, b, c, d, e, f, g, h] = H0;
  for (let i = 0; i < 64; i++) {
    const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
    const t1 = (h + S1 + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
    const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
    const t2 = (S0 + ((a & b) ^ (a & c) ^ (b & c))) | 0;
    h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
  }
  return [(H0[0] + a) >>> 0, (H0[1] + b) >>> 0];
}

/** Whether a hash starting with these two words has `bits` leading zero bits (bits ≤ 64). */
function zeros(first: number, second: number, bits: number): boolean {
  if (bits <= 32) return bits === 0 || first >>> (32 - bits) === 0;
  return first === 0 && (bits === 64 || second >>> (64 - bits) === 0);
}

/** The first number answering the challenge (as the server reads it: decimal digits). */
export function solve(nonce: string, bits: number): string {
  const prefix = new TextEncoder().encode(`${nonce}:`);
  const bytes = new Uint8Array(64);
  bytes.set(prefix);
  const w = new Uint32Array(64);
  for (let n = 0; ; n++) {
    const digits = String(n);
    const length = prefix.length + digits.length;
    if (length > MAX_TEXT) throw new Error("no answer within one block");
    for (let i = 0; i < digits.length; i++) bytes[prefix.length + i] = digits.charCodeAt(i);
    const [first, second] = head(bytes, length, w);
    if (zeros(first, second, bits)) return digits;
  }
}
