import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Licence, type ManualView } from "../api";
import { Sheet } from "../ui/Sheet";
import { copyText, webAddress } from "../platform/util";
import { Button, ButtonLink } from "../ui/Button";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

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
      <span>{t("ui.install.inbox_lede")}</span>
      <code className="manual-path" {...tipAttrs(tipOf("value", t("ui.install.on_server", { path: view.inbox.open })))}>{folder}</code>
      <Button onClick={() => void copyText(view.inbox.open).then(() => setCopied(true))}>
        {copied ? t("ui.install.copied") : t("ui.common.copy")}
      </Button>
      {view.unknown.length > 0 && (
        <div className="manual-unknown">
          <b>{t("ui.install.unknown_files")}</b>
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
        <ButtonLink size="sm" tone="ghost" href={downloadPage} target="_blank" rel="noreferrer">
          {t("ui.install.go_download")}
        </ButtonLink>
      )}
      {can.consent && consentFile && (
        <Button size="sm" tone="primary" onClick={() => setLicence(true)}>
          {t("ui.install.see_licence")}
        </Button>
      )}
      {can.manual && (
        <Button size="sm" tone="ghost" onClick={onChange}>
          {t("ui.install.check_again")}
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
    <Sheet title={licence ? t("ui.install.licence_of", { title: licence.title, version: licence.version }) : t("ui.install.licence")} width={900} onClose={installing ? () => undefined : onClose}>
      <p className="inst-note">{t("ui.install.licence_lede")}</p>
      <div className="licence-text">{licence ? licence.text : error ? "" : t("ui.install.licence_loading")}</div>
      {error && <p className="inst-note err">{error}</p>}
      <div className="licence-actions">
        {installing && <span className="spinner" aria-hidden />}
        <Button tone="ghost" disabled={installing} onClick={onClose}>
          {t("ui.install.decline")}
        </Button>
        <Button tone="primary" disabled={!licence || installing} onClick={accept}>
          {installing ? t("ui.install.installing") : t("ui.install.accept")}
        </Button>
      </div>
    </Sheet>
  );
}
