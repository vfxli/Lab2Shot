/** Files between the user's machine and the server, as the server describes them: an upload, an output. */

export interface Upload {
  ref: string; // what the node parameter holds: upload:<id>/<name>
  name: string;
  stem: string; // the plate's own name, as the server's sequence rule reads it (render.####.exr: render; lab2shot/io/sequence.py)
  files: number;
  bytes: number;
  first: number | null; // a sequence's first and last frame
  last: number | null;
}

/** What one 「输出」 collected and packed in one task (lab2shot/transfer/outputs.py): the task's "output" event, an
 * entry of a job's outputs, or of /api/outputs. Only a finished one (its zip written) is ever described. */
export interface Output {
  task: string; // the task (the job's id) it belongs to: it goes with the task
  node: string;
  label: string;
  pkg: string; // the server's name of it (its folder and zip), made of the account's id, the task's id and the node's id
  name: string; // what the browser saves the zip as (<graph>…_u<account>_<task>.zip); the zip's one top folder is named alike
  bytes: number; // the zip's size
  count: number; // files in it
  graph?: string; // the graph file's own meta.id ("" from a DCC or the command line): drawn only on the graph it belongs to
  title?: string; // the graph's name when it was cooked
  finished?: number; // seconds since the epoch: when it was packed
  gone?: boolean; // in a job's record: gone with its task (任务保留天数 after it ended)
  expires?: number | null; // /api/outputs: when its task goes (null while the task still runs)
}

/** The zip of an output, downloaded by the browser itself (resumable: the server answers Range, server/transfer.py). */
export const zipUrl = (o: Pick<Output, "task" | "pkg">): string =>
  `/api/tasks/${encodeURIComponent(o.task)}/outputs/${encodeURIComponent(o.pkg)}/zip`;
