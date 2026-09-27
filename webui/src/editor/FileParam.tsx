import { useEffect, useRef, useState } from "react";
import { api, type ParamDef } from "../api";
import { IconClose, IconFile, IconFolder, IconFrames } from "../ui/icons";
import { pickFile, setSaveTo } from "../graph/actions";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { msg, reasonOf, say } from "../state/say";
import { deliverDefaultName, deliveryLayout } from "../graph/nodes";
import { DELIVERY_DIALOG, DELIVERY_TYPES, fits } from "../files/deliver";
import { canWrite, chooseFolder, keep, kept, saveFile, writable } from "../files/handles";
import { choices, framesText, type Choice } from "../api/sequences";
import type { Upload } from "../api/deliveries";
import { UploadState } from "../ui/UploadState";
import { cancelUpload, startUpload, taskFor, useUploads } from "../transfer/uploads";
import { sizeText } from "../platform/format";
import { useDismiss } from "../platform/dismiss";
import { Button, IconButton } from "../ui/Button";

/** File parameters in the parameter panel. Everything is a button that opens the system's own dialog (or files
 * dropped on the row): an input is uploaded and the parameter holds the upload; 「输出」 gets the file (tar) or the
 * folder of the user's machine its delivery is saved into. The row shows what was chosen as far as a browser tells a
 * page: names, never full paths. */

const NO_PATH = "网页看不到文件在本机上的完整路径：浏览器为了保护本机，只把文件名和选中的文件夹名告诉网页。";

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
    // 已申报但字节尚未传输的上传同样可以返回描述（服务器 `uploads._describe_declared`）：
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

  const details = [
    folder && `文件夹 ${folder}`,
    name,
    frames && `帧 ${frames.first}-${frames.last}，共 ${frames.count} 帧${missing ? `，中间跳过 ${missing} 帧` : ""}`,
    shown && sizeText(shown.bytes),
  ]
    .filter(Boolean)
    .join("\n");
  const tip = !value
      ? `点左边的按钮选择${sequence ? "序列的所有帧，或者选整个文件夹" : kinds}；也可以拖到这里`
      : gone
        ? `${details}\n\n服务器上已经没有这份文件了：${gone}\n点左边的按钮重新选择同一份文件，或者拖进来`
        : `${details}\n已上传到服务器\n\n${NO_PATH}${sequence && !folder ? "用文件夹按钮选整个文件夹时，这里也会显示文件夹的名字。" : ""}`;

  return (
    <div
      ref={row}
      className={`fileparam${over ? " drop" : ""}`}
      onDragOver={(e) => (e.preventDefault(), setOver(true))}
      onDragLeave={() => setOver(false)}
      onDrop={drop}
    >
      <IconButton
        tip={sequence ? "选择序列：在系统对话框里把所有帧一起选中（Ctrl+A）；只选一张图就当作一帧" : `选择${p.label}：${kinds}`}
        layout="fp-pick"
        onClick={() => filesInput.current?.click()}
      >
        {sequence ? <IconFrames size={14} /> : <IconFile size={14} />}
      </IconButton>
      <input ref={filesInput} type="file" hidden multiple={sequence} accept={p.accept.join(",")} onChange={(e) => (picks(e.target.files, false), (e.target.value = ""))} />
      {sequence && (
        <>
          <IconButton
            tip="选择一个文件夹：只有一段序列就用它，有多段（左右眼、不同的 pass）就列出来选。浏览器会先问要不要传给网页，选「上传」"
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
      <div className="fp-body" data-tip={tip}>
        {task ? (
          <UploadState task={task} />
        ) : !value ? (
          <span className="fp-line fp-empty">点左边选择，或拖到这里</span>
        ) : (
          <span className="fp-line">
            {/* The name is the user's data (it may be truncated): its tooltip shows it in full, with the reason only the name is
                shown. It sits over the row's own tooltip, so it must carry that sentence itself (DeliverParam below
                does the same). */}
            <span className="fp-name mono" data-user-data data-tip={`${folder ? `${folder} / ` : ""}${name}\n\n${NO_PATH}`}>
              {folder && <span className="fp-folder">{folder} / </span>}
              {name}
            </span>
            {gone ? <span className="fp-missing">要重新选择</span> : meta && <span className="fp-meta tnum" data-user-data data-tip={meta}>{meta}</span>}
            {!gone && gap && (
              <span className="fp-missing" data-tip={`这段序列中间跳过 ${missing} 帧（帧号不连号）：这几帧没有画面，往下算的结果也不会有。这不影响计算——算的就是你选中的这些帧`}>
                {gap}
              </span>
            )}
          </span>
        )}
      </div>
      {task ? (
        <IconButton tip={value ? "不传了：参数保持原来的文件" : "不传了"} tone="ghost" layout="fp-clear" onClick={() => cancelUpload(task.key)}>
          <IconClose size={10} />
        </IconButton>
      ) : (
        value && (
          <IconButton tip="清除" tone="ghost" layout="fp-clear" onClick={() => pickFile(nodeId, p.name, null)}>
            <IconClose size={10} />
          </IconButton>
        )
      )}
      {offer && (
        <div className="menu ctx-menu seq-offer glass" style={{ left: offer.x, top: offer.y }} onContextMenu={(e) => e.preventDefault()}>
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

type Place = "none" | "ready" | "ask" | "missing" | "kind"; // none chosen; writable; the browser asks again; not in this browser; a file for a folder or the other way

const PACKED = "每个输出设置一个子文件夹（按它的名字），里面是它写的文件，最上面的 lab2shot.json 写着有什么";

/** 「输出」's 保存到: where its delivery goes on the user's machine, as its 保存成 says: one file (the system's save
 * dialog) for tar / tar.gz, a folder (the folder dialog) for 文件夹. The parameter holds the name (the file's or the
 * folder's), shown on the button; what was chosen is kept in this browser (handles.ts) and, when a cook is submitted,
 * remembered for that job (deliver.ts), so it is saved there even if this graph is closed by then. A browser that
 * cannot write into the user's files downloads the archive under the name typed here. */
export function DeliverParam({ nodeId, value, set }: { nodeId: string; p: ParamDef; value: string; set: (v: string) => void }) {
  const saveTo = useCookInputs((s) => s.nodes[nodeId]?.saveTo);
  const mode = useCookInputs((s) => String(s.nodes[nodeId]?.params.mode ?? "tar"));
  const order = useCookInputs((s) => s.order);
  const nodes = useCookInputs((s) => s.nodes);
  const edges = useCookInputs((s) => s.edges);
  const graphName = useCookInputs((s) => s.meta.name);
  const folder = mode === "folder";
  const suffix = mode === "tar.gz" ? ".tar.gz" : ".tar";
  const gnodes = order.map((id) => ({ id, data: { typeId: nodes[id].typeId, params: nodes[id].params, picked: nodes[id].picked } }) as never);
  const defaultName = deliverDefaultName(gnodes, getNodeDefs(), graphName, suffix); // a folder delivery still downloads as a suffix'd tar (transfer.ts)
  const [text, setText] = useState(value || defaultName);
  const [place, setPlace] = useState<Place>("none");
  useEffect(() => setText(value || defaultName), [value, defaultName]);

  // A browser with no folder/file-system access (opened from another computer) only has this typed field: it starts
  // with a real name (the graph's, or the plate's), not empty behind a placeholder that looks filled in but is not.
  // It is written into the param once, so it counts as chosen and is remembered
  // with the graph from then on; the user can still change it.
  const filledDefault = useRef(false);
  useEffect(() => {
    if (canWrite || value || !defaultName || filledDefault.current) return;
    filledDefault.current = true;
    set(defaultName);
  }, [value, defaultName, set]);

  useEffect(() => {
    let current = true;
    void (async () => {
      const h = await kept(saveTo?.handle);
      const state: Place = !saveTo ? "none" : !h ? "missing" : !fits(h, mode) ? "kind" : (await writable(h)) ? "ready" : "ask";
      if (current) setPlace(state);
    })();
    return () => void (current = false);
  }, [saveTo, mode]);

  const pick = async () => {
    try {
      const base = value ? value.replace(/(\.tar\.gz|\.tgz|\.tar)$/i, "").replace(/[\\/:*?"<>|\s]+/g, "_") || "lab2shot" : deliverDefaultName(gnodes, getNodeDefs(), graphName, "");
      const h = folder ? await chooseFolder(DELIVERY_DIALOG) : await saveFile(DELIVERY_DIALOG, DELIVERY_TYPES[mode === "tar.gz" ? "tar.gz" : "tar"], base + suffix);
      if (!h) return;
      setSaveTo(nodeId, { handle: await keep(h), name: h.name });
      set(h.name);
    } catch (e) {
      say(msg("E-FILE-NOPLACE", { reason: reasonOf(e) }), nodeId);
    }
  };

  // 选定后显示交付的完整结构：包（或文件夹）名及其中各层。浏览器不提供本机完整路径（NO_PATH），
  // 可完整显示的只有这一部分，由一处计算（graph/nodes.ts deliveryLayout），此处不识别任何格式
  const layout = deliveryLayout(gnodes, edges, getNodeDefs(), nodeId);
  const chosen = saveTo && place !== "kind" ? saveTo.name : "";
  const full = chosen ? `${chosen}${folder ? "/" : " 里"}${layout.length ? ` ${layout.join("、")}` : ""}` : "";

  const what = folder ? `本机的 ${saveTo?.name} 文件夹` : `本机的 ${saveTo?.name}`;
  const tip = {
    none: folder
      ? `选择本机的一个文件夹，算完每个输出设置一个子文件夹写进去。\n\n${PACKED}\n\n${NO_PATH}`
      : `选择保存成哪个文件（系统的保存对话框），算完存成这一个 ${suffix} 包。\n\n${PACKED}\n\n${NO_PATH}`,
    ready: `算完自动存进${what}。\n\n${PACKED}\n\n${NO_PATH}\n点击换一个`,
    ask: `算完存进${what}。刷新过页面，点「计算」时浏览器会再确认一次能不能写这里。\n\n${NO_PATH}\n点击换一个`,
    missing: `这个保存位置（${saveTo?.name}）是在别的浏览器或电脑上选的，这个浏览器里没有它：点击重新选择。不重选时，算完可以下载。\n\n${NO_PATH}`,
    kind: `「保存成」改过了：${folder ? "要选一个文件夹" : `要选一个 ${suffix} 文件`}，点击重新选择`,
  }[place];

  if (!canWrite) {
    const typed = () => text.trim() !== value && set(text.trim());
    return (
      <div className="fileparam">
        <span className="chip fp-download" data-tip="这个浏览器不能把文件存进本机的文件夹（要 Chrome 或 Edge，地址是 localhost 或 https）：算完下载">
          算完下载
        </span>
        <input className="field mono fp-field" value={text} placeholder={defaultName || `sh010${suffix}`} spellCheck={false} onChange={(e) => setText(e.target.value)}
          onBlur={typed} onKeyDown={(e) => e.key === "Enter" && typed()} data-tip={`下载的文件叫什么：已经填好一个默认名字，可以改。\n\n${PACKED}`} />
      </div>
    );
  }
  return (
    <div className="fp-deliver">
      <div className="fileparam">
        <Button tip={tip} warn={place === "missing" || place === "kind"} layout="fp-where" onClick={() => void pick()}>
          {folder ? <IconFolder size={13} /> : <IconFile size={13} />}
          {/* The name is the user's data (it may be truncated): its tooltip shows it in full, with the reason only the name is shown. */}
          <span data-user-data data-tip={saveTo && place !== "kind" ? `${saveTo.name}\n\n${NO_PATH}` : tip}>
            {saveTo && place !== "kind" ? saveTo.name : folder ? "选择文件夹…" : "选择保存位置…"}
          </span>
        </Button>
      </div>
      {/* 选定后即显示交付的完整结构，无须打开提示即可知道文件的保存位置 */}
      {full && (
        <div className="fp-full" data-user-data data-tip={`${full}\n\n${PACKED}\n\n${NO_PATH}`}>
          {full}
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
