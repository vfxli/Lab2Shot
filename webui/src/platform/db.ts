/** The browser's own database on the user's machine (IndexedDB), shared by everything the editor keeps there:
 * history   minute-by-minute autosave of the working graph (autosave.ts)
 * handles   files and folders the user chose in the system's dialogs: the graph's file, output folders (files/handles.ts)
 * uploads   the user's files going up (transfer/uploads.ts): a local file (by where, name, size and time) -> the sha256
 *            the server has it under; "part:" + that key -> the part it is going up in; "task:" + key -> an upload
 *            not finished yet (a reload shows it paused)
 * deliveries where each submitted cook's 「输出」 goes, by job and node, kept from the moment it is submitted
 *            (files/deliver.ts): saved there even when that graph is no longer open */

const NAME = "lab2shot";
const VERSION = 4;

let opened: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  return (opened ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("history")) db.createObjectStore("history", { autoIncrement: true });
      if (!db.objectStoreNames.contains("handles")) db.createObjectStore("handles", { keyPath: "id" });
      if (!db.objectStoreNames.contains("uploads")) db.createObjectStore("uploads");
      if (!db.objectStoreNames.contains("deliveries")) db.createObjectStore("deliveries", { keyPath: "key" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  }));
}

export type Store = "history" | "handles" | "uploads" | "deliveries";

/** One request in its own transaction; resolves when the transaction has committed. */
export async function run<T>(store: Store, mode: IDBTransactionMode, op: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const tx = (await open()).transaction(store, mode);
  const req = op(tx.objectStore(store));
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve(req.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
