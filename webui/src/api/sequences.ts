import { api } from ".";
import { rangeText } from "../platform/format";
import { t } from "../i18n/t.ts";

/** Files the user picked or dropped, the way Nuke reads a folder: numbered frames of one sequence are one item
 * (plate.####.exr 1001-1124, 000000_left.png as ######_left.png), anything else an item of its own. Which names make a
 * sequence is decided in one place, the server's lab2shot/io/sequence.py (only the names are sent): the frame number
 * wherever it sits, left / right eyes and passes as sequences of their own, hidden and system files (._*, .DS_Store,
 * Thumbs.db) left out. */

export interface Item {
  kind: "file" | "sequence";
  name: string; // a sequence: its pattern, #### = the frame number (as many # as digits when zero-padded)
  frames: number[]; // a sequence: its frames, sorted
  files: File[]; // a file: itself; a sequence: its frames in order
  size: number;
}

export interface Choice {
  item: Item;
  folder: string; // which folder it is in ("" when the browser doesn't say)
}

const single = (f: File): Item => ({ kind: "file", name: f.name, frames: [], files: [f], size: f.size });

const byName = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true });

/** Does the name end with one of the accepted suffixes (.exr, .mov ...)? */
const accepted = (name: string, accept: string[]) => !accept.length || accept.some((a) => name.toLowerCase().endsWith(a));

const hidden = (name: string) => name.startsWith(".") || ["thumbs.db", "desktop.ini"].includes(name.toLowerCase());

/** What a pick or a drop offers a parameter taking `accept`. A sequence parameter: every sequence found (longest
 * first; across the folders of a folder pick), else the pictures without a frame number, each on its own; more than
 * one is the user's choice, never picked quietly. A parameter for one file: the first accepted file. `folder`: which
 * folder a file is in. */
export async function choices(files: File[], accept: string[], sequence: boolean, folder: (f: File) => string = () => ""): Promise<Choice[]> {
  const byFolder = new Map<string, File[]>();
  for (const f of files) {
    if (accepted(f.name, accept)) byFolder.set(folder(f), [...(byFolder.get(folder(f)) ?? []), f]);
  }
  if (!sequence) {
    const [dir, list] = [...byFolder].sort((a, b) => byName(a[0], b[0]))[0] ?? [];
    const first = list?.filter((f) => !hidden(f.name)).sort((a, b) => byName(a.name, b.name))[0];
    return first ? [{ item: single(first), folder: dir ?? "" }] : [];
  }
  const folders = [...byFolder];
  if (!folders.length) return [];
  const { folders: grouped } = await api.uploads.sequences(folders.map(([, list]) => list.map((f) => f.name)));
  const seqs: Choice[] = [];
  const singles: Choice[] = [];
  grouped.forEach((g, k) => {
    const [dir, list] = folders[k];
    for (const s of g.sequences) {
      const members = s.files.map((i) => list[i]);
      seqs.push({ folder: dir, item: { kind: "sequence", name: s.name, frames: s.frames, files: members, size: members.reduce((n, f) => n + f.size, 0) } });
    }
    for (const i of g.singles) singles.push({ folder: dir, item: single(list[i]) });
  });
  seqs.sort((a, b) => b.item.frames.length - a.item.frames.length || byName(`${a.folder}/${a.item.name}`, `${b.folder}/${b.item.name}`));
  return seqs.length ? seqs : singles.sort((a, b) => byName(`${a.folder}/${a.item.name}`, `${b.folder}/${b.item.name}`));
}

/** How a choice is listed: its frames and count, 1001-1124 · 124 帧. */
export const framesText = (item: Item) =>
  item.kind === "sequence" ? t("ui.upload.sequence_frames", { range: rangeText(item.frames[0], item.frames[item.frames.length - 1]), count: item.frames.length }) : t("ui.upload.single_image");
