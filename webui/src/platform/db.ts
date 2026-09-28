/** The browser's own database on the user's machine (IndexedDB), shared by everything the editor keeps there:
 * handles   files and folders the user chose in the system's dialogs: the graph's file, the folders of footage the view
 *            reads originals from (files/handles.ts, files/localDirs.ts)
 * uploads   the user's files going up (transfer/uploads.ts): a local file (by where, name, size and time) -> the sha256
 *            the server has it under; "part:" + that key -> the part it is going up in; "task:" + key -> an upload
 *            not finished yet (a reload shows it paused) */

const NAME = "lab2shot";
const VERSION = 6;

let opened: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  return (opened ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(NAME, VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("handles")) db.createObjectStore("handles", { keyPath: "id" });
      if (!db.objectStoreNames.contains("uploads")) db.createObjectStore("uploads");
    };
    req.onsuccess = () => {
      // a newer page (another tab, after an update) wants to upgrade: let go at once instead of holding it up; the next
      // request here opens the database again. A tab of an older build that does not let go holds the upgrade until it
      // reloads (ui/Banner.tsx offers that on every new build)
      req.result.onversionchange = () => {
        req.result.close();
        opened = null;
      };
      resolve(req.result);
    };
    req.onerror = () => reject(req.error);
  }));
}

export type Store = "handles" | "uploads";

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
