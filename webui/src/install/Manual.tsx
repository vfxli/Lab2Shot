import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Licence, type ManualView } from "../api";
import { Sheet } from "../ui/Sheet";
import { copyText, host, webAddress } from "../platform/util";
import { Button, ButtonLink } from "../ui/Button";

/** 手动下载 (lab2shot/extensions/manual.py): part of installing an extension. Many extensions' weights must be
 * downloaded by the user from the official site, and some also need the user's own acceptance of a licence; while
 * either checklist item (B-INSTALL-MANUAL / B-INSTALL-CONSENT) is unmet the extension cannot be installed. That is why
 * it lives beside the install control and is used by the admin page's 「扩展包」 section.
 *
 * One folder holds everything the user downloads by hand (SMPL, SMPL-X, MANO, FLAME, the Autodesk FBX SDK ...). Each
 * item is a row of the 手动下载 table (admin/Extensions.tsx); this is only what goes in its 操作 cell: where to download it, 重新检查,
 * and 看许可协议 when a dropped file is waiting for the user to accept its licence. `version` bumps when the page
 * should recheck the inbox (an install finished, the user pressed 重新检查). `can`: whether this login may look at the inbox
 * (the availability answer help.manual, read through api/applies.ts). */
export function useManualView(version: number, can: boolean): ManualView | null {
  const [view, setView] = useState<ManualView | null>(null);
  useEffect(() => {
    if (!can) return;
    api.manual.view().then(setView, () => undefined);
  }, [version, can]);
  return can ? view : null;
}

/** The one folder every hand-downloaded file goes into, shown once (not per row, as it is the same folder for all of
 * them), with id="manual" so a "#manual" link lands on it. */
export function InboxNote({ view }: { view: ManualView | null }) {
  const [copied, setCopied] = useState(false);
  const section = useRef<HTMLElement>(null);
  useEffect(() => {
    const show = () => window.location.hash === "#manual" && section.current?.scrollIntoView({ block: "start" });
    show();
    window.addEventListener("popstate", show);
    return () => window.removeEventListener("popstate", show);
  }, []);
  if (!view) return null;
  const folder = view.inbox.path;
  return (
    <section className="manual-inbox" id="manual" ref={section}>
      <span>要手动下载的文件（下面表格里的每一行）放进这个文件夹，不用解压、不用改名：Lab2Shot 按文件内容认出它。</span>
      <code className="manual-path" data-tip={`在服务器上：${view.inbox.open}`}>{folder}</code>
      <Button tip="复制能直接打开的完整路径，粘贴到资源管理器的地址栏" onClick={() => void copyText(view.inbox.open).then(() => setCopied(true))}>
        {copied ? "已复制" : "复制"}
      </Button>
      {view.unknown.length > 0 && (
        <div className="manual-unknown">
          <b>认不出的文件</b>
          {view.unknown.map((u) => (
            <div key={u.name} className="manual-file">
              <code>{u.name}</code>
              <span>{u.why}</span>
            </div>
          ))}
        </div>
      )}
      {/* Content of the inbox unrelated to any extension gets no 「还有 N 项不会被使用」 note; only files that look
          like they belong to an extension but are misplaced are pointed out, and those are the unknown group above. */}
    </section>
  );
}

export function ManualRowControl({ manualKey, downloadPage, view, can, onChange }: { manualKey: string; downloadPage: string; view: ManualView | null; can: { manual: boolean; consent: boolean }; onChange: () => void }) {
  const [licence, setLicence] = useState(false);
  const item = view?.items.find((i) => i.key === manualKey);
  const consentFile = item?.files.find((f) => f.state === "consent");
  return (
    <div className="inst-ctl">
      {webAddress(downloadPage) && (
        <ButtonLink tip={`到 ${host(downloadPage)} 下载，放进这台电脑的手动下载文件夹`} size="sm" tone="ghost" href={downloadPage} target="_blank" rel="noreferrer">
          去下载
        </ButtonLink>
      )}
      {can.consent && consentFile && (
        <Button tip="看许可协议原文，同意后才安装" size="sm" tone="primary" onClick={() => setLicence(true)}>
          看许可协议
        </Button>
      )}
      {can.manual && (
        <Button tip="放好文件后点这里：认出来的自动装好，认不出的写明原因" size="sm" tone="ghost" onClick={onChange}>
          重新检查
        </Button>
      )}
      {licence && consentFile && (
        <LicenceSheet
          file={consentFile.name}
          onClose={() => setLicence(false)}
          onAccepted={() => {
            setLicence(false);
            onChange();
          }}
        />
      )}
    </div>
  );
}

/** The licence of a file in the inbox, as its installer prints it (taken out without accepting). Only 同意并安装
 * installs; the server records who accepted it, when, and which version. */
function LicenceSheet({ file, onClose, onAccepted }: { file: string; onClose: () => void; onAccepted: () => void }) {
  const [licence, setLicence] = useState<Licence | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [installing, setInstalling] = useState(false);

  useEffect(() => {
    api.manual.licence(file).then(setLicence, (e: Error) => setError(e.message));
  }, [file]);

  const accept = useCallback(() => {
    if (!licence) return;
    setInstalling(true);
    setError(null);
    api.manual.accept(file, licence.sha256).then(onAccepted, (e: Error) => (setError(e.message), setInstalling(false)));
  }, [file, licence, onAccepted]);

  return (
    <Sheet title={licence ? `${licence.title} ${licence.version} 许可协议` : "许可协议"} width={900} onClose={installing ? () => undefined : onClose}>
      <p className="inst-note">这是安装程序里的原文。点「同意并安装」表示当前账号本人接受这份协议，Lab2Shot 会记下是谁、什么时候同意的，然后安装；不同意就不安装。</p>
      <div className="licence-text">{licence ? licence.text : error ? "" : "正在从安装程序里取出许可协议…"}</div>
      {error && <p className="inst-note err">{error}</p>}
      <div className="licence-actions">
        {installing && <span className="spinner" aria-hidden />}
        <Button tip="不接受协议，不安装" tone="ghost" disabled={installing} onClick={onClose}>
          不同意
        </Button>
        <Button tip="本人接受这份许可协议，记下是谁、什么时候同意的，然后安装" tone="primary" disabled={!licence || installing} onClick={accept}>
          {installing ? "安装中…" : "同意并安装"}
        </Button>
      </div>
    </Sheet>
  );
}
