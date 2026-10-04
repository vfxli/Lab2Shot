import type { GraphJSON } from "../api";
import { cookHold, fileJSON, savePoint } from "./actions";
import { useResults } from "../state/results";
import { newGraphId } from "../model/graphId";
import { useCookInputs } from "../state/cookInputs";
import { MessageError, msg } from "../messages/message";
import { say } from "../state/say";
import { useViewer } from "../state/viewer";
import { download } from "../platform/util";
import { allowWrite, canWrite, keep, kept, openFile, saveFile, type FileType } from "../files/handles";
import { pick, t } from "../i18n/t";

/** The node graph's own file, on the user's machine: 打开 / 保存 / 另存为 with the system's file dialogs. 保存 writes
 * the file chosen before, 另存为 when the browser no longer lets the page write it. Browsers without these dialogs
 * open with a file input and save as a download. The server never sees graph files. */

export interface GraphFile {
  name: string;
  handle?: string; // its key in this browser (files/handles.ts); none: opened or saved without the system's dialogs
}

const SCHEMA = "lab2shot.graph/1";
const DIALOG = "lab2shot-graph"; // graph dialogs open where the last one was
const types = (): FileType[] => [{ description: t("ui.graph.file_type"), accept: { "application/json": [".json"] } }];

function parse(text: string, name: string): GraphJSON {
  let g: GraphJSON;
  try {
    g = JSON.parse(text);
  } catch {
    throw new MessageError("E-GRAPHFILE-NOTJSON", { name });
  }
  if (g?.schema !== SCHEMA) throw new MessageError("E-GRAPHFILE-NOTGRAPH", { name });
  return g;
}

const suggestedName = () => `${pick(useCookInputs.getState().meta.name).replace(/[\\/:*?"<>|]+/g, "_") || t("ui.common.unnamed")}.json`;

/** Ask for a graph file; resolves with it (null: cancelled). */
export async function openGraphFile(): Promise<{ graph: GraphJSON; file: GraphFile } | null> {
  if (!canWrite) {
    const f = await pickWithInput();
    return f && { graph: parse(await f.text(), f.name), file: { name: f.name } };
  }
  const h = await openFile(DIALOG, types());
  if (!h) return null;
  const f = await h.getFile();
  return { graph: parse(await f.text(), f.name), file: { name: f.name, handle: await keep(h) } };
}

/** Save into the graph's file, or ask where (the first save, 另存为, or when the browser no longer lets the page write
 * it). Resolves true when saved. */
export async function saveGraphFile(saveAs = false): Promise<boolean> {
  // 另存为 changes the graphId (serialize below), while the job being followed finds its nodes by the graphId it was
  // submitted with (graph/follow.ts) and the packed outputs belong to it too (graph/outputs.ts): after a change the
  // progress, the status and 「下载」 would no longer match. So 另存为 is refused while a job is cooking or being
  // submitted (the same test as opening another graph: cookHold's busy / submitting); a plain 保存 keeps the id and
  // goes ahead
  const hold = saveAs ? cookHold(useResults.getState()) : null;
  if (hold === "busy" || hold === "submitting") {
    say(msg("B-GRAPH-SAVEASCOOKING"));
    return false;
  }
  const viewer = useViewer.getState();
  // 另存为 makes a new document, never sharing the id of the one it was copied from. The id is assigned only once a write
  // is actually about to happen, never eagerly: a cancelled system dialog must leave the still-open original document
  // exactly as it was, id included.
  // what is written is what is marked saved (savePoint), taken together: an edit made while the file is being
  // written is not in it and stays unsaved
  let written: () => void = () => undefined;
  const serialize = () => {
    if (saveAs) useCookInputs.getState().setGraphId(newGraphId());
    written = savePoint();
    return JSON.stringify(fileJSON(), null, 2);
  };
  if (!canWrite) {
    const name = viewer.file?.name ?? suggestedName();
    const url = URL.createObjectURL(new Blob([serialize()], { type: "application/json" }));
    download(url, name);
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
    useViewer.getState().setFile({ name });
    written();
    return true;
  }
  const saved = async (h: FileSystemFileHandle) => {
    const w = await h.createWritable();
    await w.write(serialize());
    await w.close();
    useViewer.getState().setFile({ name: h.name, handle: await keep(h) });
    written();
    say(msg("I-GRAPH-SAVED", { file: h.name }));
    return true;
  };
  const known = saveAs ? undefined : await kept<FileSystemFileHandle>(viewer.file?.handle);
  if (known && (await allowWrite(known))) {
    try {
      return await saved(known);
    } catch {
      // moved or deleted meanwhile: ask where
    }
  }
  const h = await saveFile(DIALOG, types(), viewer.file?.name ?? suggestedName());
  return !!h && saved(h);
}

function pickWithInput(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = () => resolve(input.files?.[0] ?? null);
    input.oncancel = () => resolve(null);
    input.click();
  });
}
