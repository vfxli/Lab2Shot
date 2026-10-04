import { useCallback, useState } from "react";
import { Num } from "../ui/controls";
import { ByUser } from "./Resources";
import { api, type DiskArea } from "../api";
import { Section, useAdmin } from "./common";
import { agoText, sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { msg, type Message } from "../messages/message";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** 硬盘 section: disk usage of the task folders (their outputs inside), every account's cache and uploads on the server, and cleanup
 * of what is older. Everything is kept per task (lab2shot/farm/disk.py): 任务保留天数 is configured in 设置. The server
 * measures in the background (it may take minutes): the section shows the last figures and asks again every
 * MEASURING_POLL_MS while a measurement runs. */
const MEASURING_POLL_MS = 2000;
export function DiskSection() {
  const { problem, go, overview } = useAdmin();
  const read = useCallback(() => api.admin.disk(), []);
  const { data: measured, reload } = usePoll(read, MEASURING_POLL_MS, { until: (d) => !d.measuring, onError: (e) => problem(reasonOf(e)) });
  const areas = measured?.areas ?? null;
  const space = measured?.space ?? null;
  const measureAgain = async () => {
    try {
      await api.admin.disk(true); // starts a new measurement; the poll follows it until it is through
      reload();
    } catch (e) {
      problem(reasonOf(e));
    }
  };
  const [days, setDays] = useState(30);
  const [ask, confirmSheet] = useConfirm();
  const [cleaned, setCleaned] = useState<Message | null>(null);

  const clean = async (a: DiskArea) => {
    if (!(await ask({ title: t("ui.admin.disk.clean_title"), say: msg("N-DISK-CLEAN", { area: a.label, days, note: a.note }), yes: t("ui.admin.common.delete"), danger: true }))) return;
    try {
      const r = await api.admin.clean(a.id, days);
      problem(null);
      setCleaned(msg("I-DISK-CLEANED", { area: a.label, count: r.removed, size: sizeText(r.bytes) }));
      reload();
    } catch (e) {
      problem((e as Error).message);
    }
  };

  const disk = overview?.disk;
  return (
    <Section
      title={t("ui.admin.disk.title")}
      lede={
        <>
          {t("ui.admin.disk.lede")}
          {disk && t("ui.admin.disk.lede_disk", { path: disk.path, free: sizeText(disk.free), total: sizeText(disk.total) })}
        </>
      }
      actions={
        <>
          <Button tone="ghost" onClick={() => go("settings-storage")}>
            {t("ui.admin.disk.settings")}
          </Button>
          <Button tone="ghost" disabled={!!measured?.measuring} onClick={() => void measureAgain()}>
            {t("ui.admin.common.refresh")}
          </Button>
        </>
      }
    >
      <div className="disk-days">
        <label>
          {t("ui.admin.disk.days")}
        </label>
        <Num value={days} min={0} integer label={t("ui.admin.disk.days")} tip={tipOf("value", t("ui.admin.disk.days_tip"))} onChange={setDays} />
        <span>{t("ui.admin.disk.days_unit")}</span>
      </div>
      {/* 数据盘对「暂停新计算的剩余空间」（lab2shot/farm/policy.py space）：低于它时所有账号的新计算暂停，恢复后自动接着 */}
      {space && (
        <div className={`notice${space.low ? " warn" : ""}`} data-field="disk-space">
          {space.low
            ? t("ui.admin.disk.space_low", { path: space.path, pct: space.pct.toFixed(1), free: sizeText(space.free), floor: space.floor_pct })
            : t("ui.admin.disk.space_ok", { path: space.path, pct: space.pct.toFixed(1), free: sizeText(space.free), total: sizeText(space.total), floor: space.floor_pct })}
        </div>
      )}
      {measured && (measured.measuring || measured.at !== null) && (
        <p className="adm-lede">
          {measured.measuring ? t("ui.admin.disk.measuring") : ""}
          {measured.at !== null && t("ui.admin.disk.measured", { ago: agoText(measured.at) })}
        </p>
      )}
      {areas ? (
        <div className="q-table-wrap">
          <table className="q-table">
            <thead>
              <tr>
                <th>{t("ui.admin.disk.col_what")}</th>
                <th>{t("ui.admin.disk.col_used")}</th>
                <th>{t("ui.admin.disk.col_items")}</th>
                <th>{t("ui.admin.disk.col_idle7")}</th>
                <th>{t("ui.admin.disk.col_idle30")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {areas.map((a) => (
                <tr key={a.id}>
                  <td>
                    <span className="q-title">{a.label}</span>
                    <span className="q-targets">{a.note}</span>
                  </td>
                  <td className="tnum">{sizeText(a.bytes)}</td>
                  <td className="tnum">{a.items}</td>
                  <td className="tnum">{sizeText(a.idle_7_bytes)}</td>
                  <td className="tnum">{sizeText(a.idle_30_bytes)}</td>
                  <td className="q-act">
                    <Button onClick={() => void clean(a)}>
                      {t("ui.admin.disk.clean")}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-lede">{t("ui.admin.disk.counting")}</p>
      )}
      {cleaned && <div className="notice" data-code={cleaned.code}>{cleaned.text}</div>}
      {confirmSheet}
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="disk" />
    </Section>
  );
}
