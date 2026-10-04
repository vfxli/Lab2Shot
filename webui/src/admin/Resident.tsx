import { useState } from "react";
import { api, type ResidentProcess, type ResidentView } from "../api";
import { Section, useAdmin } from "./common";
import { sizeText } from "../platform/format";
import { durationText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const STATE: Record<ResidentProcess["state"], [() => string, string]> = {
  busy: [() => t("ui.admin.resident.busy"), "var(--accent)"],
  moving: [() => t("ui.admin.resident.moving"), "var(--orange)"],
  gpu: [() => t("ui.admin.resident.gpu"), "var(--green)"],
  ram: [() => t("ui.admin.resident.ram"), "var(--text-3)"],
};

const mbText = (mb: number) => (mb ? sizeText(mb * 2 ** 20) : "—");

/** 常驻模型 section: worker processes kept alive between jobs with their models loaded — their location and
 * loaded models — and moving those models to RAM or unloading them. Lifetime and count limits are configured in 设置. */
export function ResidentSection() {
  const { problem, go, settings } = useAdmin();
  const { data: view, reload: reload } = usePoll(api.admin.resident, 3000, { onError: (e) => problem(reasonOf(e)) });
  const [acting, setActing] = useState<string | null>(null);
  const current = view;

  const act = async (id: string, what: (id: string) => Promise<ResidentView>) => {
    setActing(id);
    try {
      await what(id);
      problem(null);
    } catch (e) {
      problem((e as Error).message);
    } finally {
      setActing(null);
      reload();
    }
  };

  const value = (key: string) => settings?.settings.find((s) => s.key === key)?.value;
  const policy = !settings
    ? ""
    : value("resident.keep")
      ? value("resident.to_ram")
        ? t("ui.admin.resident.policy_to_ram", { minutes: String(value("resident.idle_minutes")), per_gpu: String(value("resident.per_gpu")) })
        : t("ui.admin.resident.policy_unload", { minutes: String(value("resident.idle_minutes")), per_gpu: String(value("resident.per_gpu")) })
      : t("ui.admin.resident.policy_off");

  return (
    <Section
      title={t("ui.admin.resident.title")}
      lede={
        <>
          {t("ui.admin.resident.lede")}{policy}
          {current && t("ui.admin.resident.available", { gb: current.available_gb.toFixed(0) })}
        </>
      }
      actions={
        <>
          <Button tone="ghost" onClick={() => go("settings-compute")}>
            {t("ui.admin.resident.settings")}
          </Button>
          <Button tone="ghost" onClick={reload}>
            {t("ui.admin.common.refresh")}
          </Button>
        </>
      }
    >
      {current === null ? (
        <p className="adm-lede">{t("ui.admin.common.reading")}</p>
      ) : current.processes.length ? (
        <div className="q-table-wrap">
          <table className="q-table">
            <thead>
              <tr>
                <th>{t("ui.admin.resident.col_models")}</th>
                <th>{t("ui.admin.resident.col_gpu")}</th>
                <th>{t("ui.admin.resident.col_state")}</th>
                <th>{t("ui.admin.resident.col_vram")}</th>
                <th>{t("ui.admin.resident.col_ram")}</th>
                <th>{t("ui.admin.resident.col_idle")}</th>
                <th>{t("ui.admin.resident.col_jobs")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {current.processes.map((p) => {
                const [label, color] = STATE[p.state];
                const working = p.state === "busy" || p.state === "moving";
                const why = working ? t("ui.admin.resident.working") : undefined;
                return (
                  <tr key={p.id}>
                    <td>
                      <span className="q-title">{p.title}</span>
                      <span className="q-targets">
                        {p.models.map((m, i) => (
                          <span key={m.name} {...tipAttrs(tipOf("value", m.on_gpu ? t("ui.admin.resident.on_gpu", { size: mbText(m.gpu_mb) }) : t("ui.admin.resident.on_ram", { size: mbText(m.ram_mb) })))}>
                            {i ? " · " : ""}
                            {m.name}
                          </span>
                        ))}
                      </span>
                    </td>
                    <td>{p.gpu_name || "—"}</td>
                    <td>
                      <span className="chip q-state">
                        <i style={{ background: color }} />
                        {label()}
                      </span>
                    </td>
                    <td className="tnum">
                      {mbText(p.vram_mb)}
                    </td>
                    <td className="tnum">{mbText(p.ram_mb)}</td>
                    <td className="tnum">{working ? "—" : durationText(p.idle_s)}</td>
                    <td className="tnum">{p.jobs}</td>
                    <td className="q-act resident-act">
                      <Button
                        tip={why ? tipOf("disabled", why) : p.state === "ram" ? tipOf("disabled", t("ui.admin.resident.already_ram")) : undefined}
                        disabled={p.state !== "gpu" || acting === p.id}
                        onClick={() => void act(p.id, api.admin.offload)}
                      >
                        {t("ui.admin.resident.offload")}
                      </Button>
                      <Button
                        tip={why ? tipOf("disabled", why) : tipOf("consequence", t("ui.admin.resident.unload_tip"))}
                        disabled={working || acting === p.id}
                        onClick={() => void act(p.id, api.admin.unload)}
                      >
                        {t("ui.admin.resident.unload")}
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-empty">{t("ui.admin.resident.empty")}</p>
      )}
    </Section>
  );
}
