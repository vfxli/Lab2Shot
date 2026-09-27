// 后台「扩展包」页和许可证同意流程用到的类型（lab2shot/server/installs.py、lab2shot/extensions/manual.py）。
// Re-exported by api/index.ts.

/** An item the extension waits for that the user downloads by hand (see ManualView): missing, or in the inbox
 * waiting for the user to accept its licence. */
export interface ManualNeed {
  key: string;
  title: string;
  status: "manual" | "consent";
}

/** Everything downloaded by hand (lab2shot/extensions/manual.py): the one inbox, every item, what still needs the
 * user (unknown), and what the inbox simply never looks at (unrelated: a folder, an unrecognisable file — shown as
 * one calm line, never a warning). */
export interface ManualView {
  inbox: { path: string; open: string }; // relative to the Lab2Shot folder, e.g. "downloads"
  items: ManualItem[];
  unknown: { name: string; why: string }[];
  /* 服务器把它和「放错了的文件」分开（用户自己的东西不算错）；页面不为它报「还有 N 项不会被使用」 */
  unrelated: { count: number; sample: string[] };
}

export interface ManualItem {
  key: string;
  title: string;
  what: string;
  page: string; // where it is downloaded
  download: string; // which download on that page
  filename: string; // the downloaded file's name (recognised by its contents: it may be renamed)
  alone: string[]; // the name a file needs when it comes on its own, not in its archive
  note: string; // registration, licence
  noncommercial: boolean;
  consent: boolean; // installing it means accepting a licence
  state: "ready" | "consent" | "unrecognised" | "missing";
  files: { name: string; state: "consent" | "unrecognised"; version?: string; why?: string }[]; // its files in the inbox
  needed_by: { name: string; title: string }[];
  installed: string; // where it is installed ("" while it is not)
}

export interface Licence {
  title: string;
  file: string;
  version: string;
  text: string;
  sha256: string; // of the text: the user accepts exactly what was shown
}

export type Commercial = "yes" | "partial" | "no" | "pending";
