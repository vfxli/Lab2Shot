import { MessageError } from "../messages/message";

/** Request-and-response plumbing shared by every worker the page uses (the EXR decoder, the 3D scene's arithmetic,
 * the local proxy maker, the registration's proof of work; each is its own script, all addressed the same way). `start` creates the worker; its
 * `new Worker(new URL("…", import.meta.url), { type: "module" })` is written out at the call site, where the bundler
 * can see the script's address (a URL passed around cannot be bundled). It is called on the first request, never at
 * load. Every request carries its message, and every response resolves the promise for the id it returns with. The
 * worker, the serial number and the pending set live in the request function's closure, so none of them is module-level
 * state.
 *
 * Arrays are sent as typed arrays, never as their buffers (`Sendable` has no ArrayBuffer, and one found at run time is
 * refused): a typed array is often a view into a larger buffer (a sample of a chunk, one camera of a whole track), and
 * the buffer alone loses the view's offset and length. `packed` sends exactly the view's own bytes: the buffer itself,
 * transferred, when the page gives the array away and the view spans the whole buffer; otherwise a transferred copy of
 * just its bytes. The receiver therefore always gets a view starting at offset 0 with exactly its length. The worker
 * side responds through `workerAnswers`, which packs its response the same way (everything in it is given away: a
 * worker retains nothing).
 *
 * A request the worker answered with an error (caught by `workerAnswers`) is rejected with E-THREAD-FAILED; a worker
 * that crashes (an uncaught script error, an unreadable message) rejects every pending request with E-THREAD-CRASHED,
 * and the next request starts a new worker. */

/** What may be sent to or from a worker: anything structured cloning supports, with no ArrayBuffer anywhere in it (a
 * typed array is sent as itself). `M & Sendable<M>` is `never` where M contains a buffer, so sending one does not compile. */
type Sendable<T> = T extends ArrayBuffer | SharedArrayBuffer
  ? never
  : T extends ArrayBufferView | ImageBitmap
    ? T
    : T extends object
      ? { [K in keyof T]: Sendable<T[K]> }
      : T;

const isBitmap = (v: unknown): v is ImageBitmap => typeof ImageBitmap !== "undefined" && v instanceof ImageBitmap;

/** A copy of exactly a view's bytes, as the same kind of view from offset 0. */
function exactly(v: ArrayBufferView): ArrayBufferView {
  const bytes = (v.buffer as ArrayBuffer).slice(v.byteOffset, v.byteOffset + v.byteLength);
  if (v instanceof DataView) return new DataView(bytes);
  const Made = v.constructor as new (b: ArrayBuffer) => ArrayBufferView;
  return new Made(bytes);
}

/** The message as posted (every typed array reduced to exactly its own bytes) and its transfer list. `given(v)`: the
 * sender no longer needs `v`, so a view spanning its whole buffer is sent without a copy (and becomes unusable here). */
export function packed(message: unknown, given: (v: object) => boolean): { message: unknown; transfer: Transferable[] } {
  const transfer = new Set<Transferable>();
  const seen = new Map<object, unknown>();
  const pack = (v: unknown): unknown => {
    if (v === null || typeof v !== "object") return v;
    if (seen.has(v)) return seen.get(v);
    let out: unknown;
    if (v instanceof ArrayBuffer || (typeof SharedArrayBuffer !== "undefined" && v instanceof SharedArrayBuffer))
      throw new TypeError("a worker is sent typed arrays, not their buffers: a buffer alone loses a view's offset and length");
    if (ArrayBuffer.isView(v)) {
      const whole = v.byteOffset === 0 && v.byteLength === v.buffer.byteLength && v.buffer instanceof ArrayBuffer;
      const own = whole && given(v) ? v : exactly(v);
      transfer.add(own.buffer as ArrayBuffer);
      out = own;
    } else if (isBitmap(v)) {
      if (given(v)) transfer.add(v);
      out = v;
    } else if (Array.isArray(v)) {
      out = v.map(pack);
    } else {
      out = Object.fromEntries(Object.entries(v).map(([k, x]) => [k, pack(x)]));
    }
    seen.set(v, out);
    return out;
  };
  const message2 = pack(message);
  return { message: message2, transfer: [...transfer] };
}

/** The page side: a request function for the worker created by `start`. `given`: the arrays and bitmaps in the message
 * that the page no longer needs (transferred where possible; everything else is copied). */
type Asker<Answer> = <M extends object>(message: M & Sendable<M>, given?: readonly object[]) => Promise<Answer>;

export function workerAsks<Answer>(start: () => Worker): Asker<Answer> {
  let worker: Worker | null = null;
  let serial = 0;
  const waiting = new Map<number, { resolve: (answer: Answer) => void; reject: (e: Error) => void }>();
  const crashed = (detail: string) => {
    worker?.terminate();
    worker = null; // the next request starts a new worker
    const asks = [...waiting.values()];
    waiting.clear();
    for (const a of asks) a.reject(new MessageError("E-THREAD-CRASHED", { detail }));
  };
  return (message: object, given: readonly object[] = []) => {
    worker ??= (() => {
      const w = start();
      w.onmessage = (e: MessageEvent<Answer & { id: number; error?: string }>) => {
        const ask = waiting.get(e.data.id);
        waiting.delete(e.data.id);
        if (typeof e.data.error === "string") ask?.reject(new MessageError("E-THREAD-FAILED", { detail: e.data.error }));
        else ask?.resolve(e.data);
      };
      w.onerror = (e: ErrorEvent) => {
        e.preventDefault?.();
        crashed(e.message || "script error");
      };
      w.onmessageerror = () => crashed("an answer could not be read");
      return w;
    })();
    const id = ++serial;
    const gone = new Set<object>(given);
    return new Promise((resolve, reject) => {
      let sent: { message: unknown; transfer: Transferable[] };
      try {
        sent = packed(message, (v) => gone.has(v));
      } catch (e) {
        return reject(e as Error);
      }
      waiting.set(id, { resolve, reject });
      worker!.postMessage({ id, ...(sent.message as object) }, sent.transfer);
    });
  };
}

/** The worker side: every request is passed to `answer`, and its result (or its error message) is posted back with the
 * request's id, packed like a request with everything in it given away. */
export function workerAnswers<Ask>(answer: (ask: Ask) => object | Promise<object>): void {
  const post = (message: object, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);
  self.onmessage = async (e: MessageEvent<{ id: number } & Ask>) => {
    const { id, ...ask } = e.data;
    try {
      const sent = packed((await answer(ask as Ask)) ?? {}, () => true);
      post({ id, ...(sent.message as object) }, sent.transfer);
    } catch (err) {
      post({ id, error: (err as Error).message });
    }
  };
}
