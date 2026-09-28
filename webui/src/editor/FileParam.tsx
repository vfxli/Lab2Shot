import { useEffect, useRef, useState } from "react";
import { api, type ParamDef } from "../api";
import { IconClose, IconFile, IconFolder, IconFrames } from "../ui/icons";
import { pickFile } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { msg, reasonOf, say } from "../state/say";
import { choices, framesText, type Choice } from "../api/sequences";
import type { Upload } from "../api/files";
import { UploadState } from "../ui/UploadState";
import { cancelUpload, startUpload, taskFor, useUploads } from "../transfer/uploads";
import { sizeText } from "../platform/format";
import { useDismiss } from "../platform/dismiss";
import { IconButton } from "../ui/Button";

/** File parameters in the parameter panel. Everything is a button that opens the system's own dialog (or files
 * dropped on the row): an input is uploaded and the parameter holds the upload. The row shows what was chosen as far as
 * a browser tells a page: names, never full paths (a browser tells a page only the file's name and the folder picked). */

/** The name inside an upload reference (upload:<id>/<name>): what the node reads, kept even when the upload is gone. */
const refName = (ref: string) => ref.slice(ref.indexOf("/") + 1);

/** Plates, videos, camera files: picked → uploaded → the node reads it on the server. */
export function InputFileParam({ nodeId, p, value }: { nodeId: string; p: ParamDef; value: string }) {
  const [info, setInfo] = useState<Upload | null>(null);
  const [gone, setGone] = useState<string | null>(null); // why the server can't give the upload (cleaned since)
  const task = useUploads((s) => taskFor(s.tasks, nodeId, p.name)); // going up (transfer/uploads.ts)
  const graph = useCookInputs((s) => s.meta.name);
  const [over, setOver] = useState(false);
  const record = useCookInputs((s) => s.nodes[nodeId]?.picked?.[p.name]);
  const filesInput = useRef<HTMLInputElement>(null);
  const folderInput = useRef<HTMLInputElement>(null);
  const sequence = p.widget === "sequence";
  const kinds = `${p.accept.join(" / ")}${sequence ? " 序列" : ""}`;
  const [offer, setOffer] = useState<{ list: Choice[]; x: number; y: number } | null>(null); // several found: the user picks one
  const row = useRef<HTMLDivElement>(null);

  // the server says whether it still has the upload (and what it is, for a value set elsewhere: a DCC, a template)
  useEffect(() => {
    let current = true;
    setInfo(null);
    setGone(null);
    // 已申报但字节尚未传输的上传同样可以返回描述（服务器 `lab2shot/transfer/uploads.py _describe_declared`）：
    // 此时素材完好地位于使用者本机，只是尚未上传，不得提示「服务器上已经没有这份文件了」
    if (value) api.uploads.describe(value).then((i) => current && setInfo(i), (e: Error) => current && setGone(e.message));
    return () => void (current = false);
  }, [value]);

  // 选定文件后是否立即传输字节由服务器决定（`nodes/base.py ReadsFile.head_is_enough` →
  // 该参数上的 `head_enough`）：真 = 先申报，点击「计算」时才传输；假 = 选定后立即传输，因为服务器需要完整
  // 文件才能回答该节点的问题（USD / FBX / Alembic 的层级、文件夹中的序列数、视频帧数）
  const send = (item: Parameters<typeof startUpload>[0], folder: string) =>
    startUpload(item, folder, { graph, node: nodeId, param: p.name }, p.head_enough === true);

  // What was found in a pick or a drop: one is taken; several (left and right eyes, passes, a second shot) are listed
  // under the row for the user to choose, never one of them taken silently.
  const offered = (found: Choice[], none: string) => {
    if (!found.length) return say(msg(none, { kinds }), nodeId);
    if (found.length === 1) return send(found[0].item, found[0].folder);
    const r = row.current?.getBoundingClientRect();
    setOffer({ list: found, x: r?.left ?? 0, y: (r?.bottom ?? 0) + 4 });
  };
  const failed = (e: Error) => say(msg("E-FILE-NOTRECOGNISED", { reason: reasonOf(e) }), nodeId);

  // the system's file dialog (all frames of a sequence selected) or folder dialog (a file's folder as the browser
  // tells it: the folder picked, sh010, or one below it, sh010/plates)
  const picks = (list: FileList | null, fromFolder: boolean) => {
    if (!list?.length) return;
    const folderOf = (f: File) => f.webkitRelativePath.split("/").slice(0, -1).join("/");
    void choices(Array.from(list), p.accept, sequence, fromFolder ? folderOf : undefined).then((found) => offered(found, fromFolder ? "B-FILE-NONEINFOLDER" : "B-FILE-NONEINFILES"), failed);
  };

  const drop = (e: React.DragEvent) => {
    e.preventDefault();
    setOver(false);
    void dropped(e.dataTransfer, p.accept, sequence).then((found) => offered(found, "B-FILE-NONEDROPPED"), failed);
  };

  useDismiss(!!offer, ".seq-offer", () => setOffer(null));

  // what was picked (kept with the graph), else what the server says
  const shown = record?.ref === value ? record : info;
  const folder = record?.ref === value ? record.folder : "";
  const name = refName(value);
  const frames = shown?.first != null && shown.last != null ? { first: shown.first, last: shown.last, count: shown.files } : null;
  // 仅在序列中间确实跳号时给出说明（如 1001–1300 只有 200 张，中间缺少 100 帧）：下游结果同样缺少这些帧。
  // 该说明不阻止任何操作，所选的帧即构成该序列。会阻止提交的是节点图中保存的计算范围
  // 仍停留在旧素材的帧段上（graph/nodes.ts rangeProblem）。
  const missing = frames ? frames.last - frames.first + 1 - frames.count : 0;
  const gap = frames && missing > 0 ? `中间跳 ${missing} 帧` : "";
  const meta = shown && [frames && `${frames.first}-${frames.last}`, frames && `${frames.count} 帧`, sizeText(shown.bytes)].filter(Boolean).join(" · ");

  return (
    <div
      ref={row}
      className={`fileparam${over ? " drop" : ""}`}
      onDragOver={(e) => (e.preventDefault(), setOver(true))}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <IconButton
        aria-label={sequence ? "选择序列" : `选择${p.label}`}
        layout="fp-pick"
        onClick={() => filesInput.current?.click()}
      >
        {sequence ? <IconFrames size={14} /> : <IconFile size={14} />}
      </IconButton>
      <input ref={filesInput} type="file" hidden multiple={sequence} accept={p.accept.join(",")} onChange={(e) => (picks(e.target.files, false), (e.target.value = ""))} />
      {sequence && (
        <>
          <IconButton
            aria-label="选择文件夹"
            layout="fp-pick"
            onClick={() => folderInput.current?.click()}
          >
            <IconFolder size={14} />
          </IconButton>
          <input
            ref={folderInput}
            type="file"
            hidden
            // @ts-expect-error: webkitdirectory is not in the typings
            webkitdirectory=""
            onChange={(e) => (picks(e.target.files, true), (e.target.value = ""))}
          />
        </>
      )}
      <div className="fp-body">
        {task ? (
          <UploadState task={task} />
        ) : !value ? (
          <span className="fp-line fp-empty">点左边选择，或拖到这里</span>
        ) : (
          <span className="fp-line">
            {/* the name is the user's data: it may be truncated */}
            <span className="fp-name mono" data-user-data>
              {folder && <span className="fp-folder">{folder} / </span>}
              {name}
            </span>
            {gone ? <span className="fp-missing">要重新选择</span> : meta && <span className="fp-meta tnum" data-user-data>{meta}</span>}
            {!gone && gap && (
              <span className="fp-missing">
                {gap}
              </span>
            )}
          </span>
        )}
      </div>
      {task ? (
        <IconButton aria-label="不传了" tone="ghost" layout="fp-clear" onClick={() => cancelUpload(task.key)}>
          <IconClose size={10} />
        </IconButton>
      ) : (
        value && (
          <IconButton aria-label="清除" tone="ghost" layout="fp-clear" onClick={() => pickFile(nodeId, p.name, null)}>
            <IconClose size={10} />
          </IconButton>
        )
      )}
      {offer && (
        // a menu, so it keeps its tips inside the quiet panel (platform/tips.ts)
        <div className="menu ctx-menu seq-offer glass" role="menu" style={{ left: offer.x, top: offer.y }} onContextMenu={(e) => e.preventDefault()}>
          <div className="menu-cat" data-tip="选中的文件里有好几段序列（比如左右眼、不同的 pass）：点一段只用它；按 Esc 或点别处就都不用">
            {offer.list.some((c) => c.item.kind === "sequence") ? "选一段序列" : "选一张图"}
          </div>
          <div className="menu-list">
            {offer.list.map((c, i) => (
              <div
                key={i}
                className="menu-item"
                data-tip={`${[c.folder, c.item.name].filter(Boolean).join("/")}\n${framesText(c.item)} · ${sizeText(c.item.size)}`}
                onClick={() => (setOffer(null), send(c.item, c.folder))}
              >
                <span className="mono seq-offer-name" data-user-data data-tip={[c.folder, c.item.name].filter(Boolean).join("/")}>{[c.folder, c.item.name].filter(Boolean).join("/")}</span>
                <span className="menu-desc tnum">{framesText(c.item)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** What was dropped on a file parameter: a file, several frames, or a folder (its files; Chrome / Edge). */
async function dropped(dt: DataTransfer, accept: string[], sequence: boolean): Promise<Choice[]> {
  // read the drop before the first await: the browser empties it after the event
  const files = Array.from(dt.files);
  const handles = [...dt.items].map((i) => ("getAsFileSystemHandle" in i ? (i as DataTransferItem & { getAsFileSystemHandle(): Promise<FileSystemHandle | null> }).getAsFileSystemHandle() : null));
  const dir = (await Promise.all(handles)).find((h): h is FileSystemDirectoryHandle => h?.kind === "directory");
  if (!dir) return choices(files, accept, sequence);
  const inside: File[] = [];
  for await (const h of dir.values()) if (h.kind === "file") inside.push(await (h as FileSystemFileHandle).getFile());
  return choices(inside, accept, sequence, () => dir.name);
}
