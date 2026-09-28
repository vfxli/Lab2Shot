import { zipUrl, type Output } from "../api/files";
import { download } from "../platform/util";

/** 「输出」's download: the browser's own download of the finished zip (lab2shot/transfer/outputs.py), the one every
 * user knows, with its progress and its resume after a dropped line (the server answers Range / If-Range:
 * server/transfer.py). The file's name is the server's (Content-Disposition), named like the one folder inside it. */
export function downloadOutput(o: Pick<Output, "task" | "pkg">): void {
  download(zipUrl(o), "");
}
