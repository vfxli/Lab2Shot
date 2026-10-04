import { useCallback, useState } from "react";
import { adminApi, type DatabaseView } from "../api/admin";
import { Section, useAdmin } from "./common";
import { fullTimeText, lastedText, sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 数据库 section: where the records are stored (lab2shot/database) and their safety status — size, version,
 * the last integrity check, retained backups (立即备份), and how to restore one. */

const REASONS: Record<string, () => string> = {
  created: () => t("ui.admin.database.reason_created"),
  daily: () => t("ui.admin.database.reason_daily"),
  manual: () => t("ui.admin.database.reason_manual"),
  merge: () => t("ui.admin.database.reason_merge"),
  "undo-merge": () => t("ui.admin.database.reason_undo_merge"),
};
const reasonText = (r: string) => REASONS[r]?.() ?? (r.startsWith("before-v") ? t("ui.admin.database.reason_upgrade", { version: r.slice(8) }) : r);
const backupReason = (name: string) => name.replace(/^lab2shot-\d{8}-\d{6}-/, "").replace(/(~\d+)?\.db$/, "");

export function DatabaseSection() {
  const { problem } = useAdmin();
  const read = useCallback(() => adminApi.database(), []);
  const { data: loaded, reload: reload } = usePoll(read, null, { onError: (e) => problem(reasonOf(e)) });
  const [view, setView] = useState<DatabaseView | null>(null);
  const [busy, setBusy] = useState("");
  const d = view ?? loaded;

  const act = async (what: string, call: () => Promise<DatabaseView>) => {
    setBusy(what);
    try {
      setView(await call());
      problem(null);
    } catch (e) {
      problem((e as Error).message);
    } finally {
      setBusy("");
    }
  };

  if (!d) return <Section title={t("ui.admin.database.title")}>{<p className="adm-lede">{t("ui.admin.common.reading")}</p>}</Section>;
  const ok = d.checked.ok !== false;
  return (
    <Section
      title={t("ui.admin.database.title")}
      lede={t("ui.admin.database.lede")}
      actions={
        <>
          <Button tone="ghost" disabled={!!busy} onClick={() => void act("check", adminApi.checkNow)}>
            {busy === "check" ? t("ui.admin.database.checking") : t("ui.admin.database.check")}
          </Button>
          <Button disabled={!!busy} onClick={() => void act("backup", adminApi.backupNow)}>
            {busy === "backup" ? t("ui.admin.database.backing_up") : t("ui.admin.database.backup_now")}
          </Button>
          <Button tone="ghost" onClick={() => (setView(null), reload())}>
            {t("ui.admin.common.refresh")}
          </Button>
        </>
      }
    >
      {!ok && <div className="notice">{t("ui.admin.database.check_failed", { detail: d.checked.detail })}</div>}
      <div className="adm-tiles db-tiles">
        <div className="adm-tile" {...tipAttrs(tipOf("value", d.path))}>
          <span className="adm-tile-label">{t("ui.admin.database.size")}</span>
          <b className="tnum">{sizeText(d.bytes)}</b>
          <span className="adm-tile-sub">{t("ui.admin.database.version", { version: d.version })}</span>
        </div>
        <div className="adm-tile" {...tipAttrs(tipOf("value", d.checked.at ? t("ui.admin.database.checked_at", { at: fullTimeText(d.checked.at), detail: d.checked.detail }) : undefined))}>
          <span className="adm-tile-label">{t("ui.admin.database.integrity")}</span>
          <b className={ok ? "db-ok" : "db-bad"}>{ok ? t("ui.admin.database.intact") : t("ui.admin.database.damaged")}</b>
          <span className="adm-tile-sub">{d.checked.at ? t("ui.admin.database.checked_ago", { ago: lastedText(d.checked.at) }) : t("ui.admin.database.checked_at_start")}</span>
        </div>
        <div className="adm-tile" {...tipAttrs(tipOf("value", d.last_backup?.file))}>
          <span className="adm-tile-label">{t("ui.admin.database.last_backup")}</span>
          <b>{d.last_backup ? t("ui.admin.database.ago", { ago: lastedText(d.last_backup.at) }) : t("ui.admin.database.no_backup")}</b>
          <span className="adm-tile-sub">{d.last_backup ? reasonText(d.last_backup.reason) : ""}</span>
        </div>
        <div className="adm-tile">
          <span className="adm-tile-label">{t("ui.admin.database.backups")}</span>
          <b className="tnum">{t("ui.admin.database.copies", { n: d.backups.length })}</b>
          <span className="adm-tile-sub">{t("ui.admin.database.keep", { n: d.keep })}</span>
        </div>
      </div>

      <h3 className="adm-h3">{t("ui.admin.database.backups")}</h3>
      <div className="q-table-wrap">
        <table className="q-table">
          <thead>
            <tr>
              <th>{t("ui.admin.database.col_time")}</th>
              <th>{t("ui.admin.database.col_reason")}</th>
              <th>{t("ui.admin.database.size")}</th>
              <th>{t("ui.admin.database.col_file")}</th>
            </tr>
          </thead>
          <tbody>
            {d.backups.map((b) => (
              <tr key={b.name}>
                <td className="tnum">{fullTimeText(b.at)}</td>
                <td>{reasonText(backupReason(b.name))}</td>
                <td className="tnum">{sizeText(b.bytes)}</td>
                <td className="mono">{b.name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="adm-lede">
        {t("ui.admin.database.restore_before")} <code>{t("ui.admin.database.restore_command")}</code>{t("ui.admin.database.restore_after")}
      </p>
    </Section>
  );
}
