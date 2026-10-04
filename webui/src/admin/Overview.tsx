import { Section, useAdmin } from "./common";
import { gbText, lastedText, sizeText } from "../platform/format";
import { Button } from "../ui/Button";
import { RecentPart } from "./Recent";
import { t } from "../i18n/t";
import { tipAttrs, tipOf, type Tip } from "../platform/tips";

/** 概览 section: a summary of the machine and the farm — GPUs (switched in 显卡), the queue, memory, disk, resident
 * models, the server itself — and items requiring the administrator's attention; below, what happened today and
 * lately (Recent.tsx). Each tile and card links to its section. */
export function OverviewSection() {
  const { queue, overview, go } = useAdmin();
  const active = queue?.jobs.filter((j) => j.state === "running" || j.state === "queued") ?? [];
  const running = active.filter((j) => j.state === "running").length;
  const takers = queue?.gpus?.filter((g) => g.authorized).length ?? 0;
  const mem = overview?.memory;
  const disk = overview?.disk;
  const diskLow = !!disk && disk.total > 0 && (disk.free < 50 * 2 ** 30 || disk.free / disk.total < 0.05);

  const alerts: [string, string, string][] = []; // Entries: text, section, tooltip (what to do; "" when the button says it).
  if (queue && (queue.gpus ?? []).length && !takers) alerts.push([t("ui.admin.overview.no_gpu"), "cards", t("ui.admin.overview.no_gpu_tip")]);
  if (mem && mem.available_gb < mem.keep_free_gb) alerts.push([t("ui.admin.overview.low_memory", { available: gbText(mem.available_gb), keep: gbText(mem.keep_free_gb) }), "settings-compute", t("ui.admin.overview.low_memory_tip")]);
  if (overview?.space?.low) alerts.push([t("ui.admin.overview.space_low", { pct: overview.space.pct.toFixed(1), floor: overview.space.floor_pct }), "disk", t("ui.admin.overview.space_low_tip")]);
  else if (diskLow) alerts.push([t("ui.admin.overview.disk_low", { free: sizeText(disk!.free) }), "disk", t("ui.admin.overview.disk_low_tip")]);
  if (overview?.pending.length) alerts.push([t("ui.admin.overview.pending", { settings: overview.pending.map((p) => p.label).join(t("list.sep")) }), `settings-${overview.pending[0].page}`, t("ui.admin.overview.pending_tip")]);
  const reg = overview?.registering;
  if (reg?.paused)
    alerts.push([
      reg.paused === "hour" ? t("ui.admin.overview.register_paused_hour", { n: reg.hour }) : t("ui.admin.overview.register_paused_day", { n: reg.day }),
      "settings-register",
      t("ui.admin.overview.register_paused_tip"),
    ]);
  if (overview?.feedback_new) alerts.push([t("ui.admin.overview.feedback_new", { n: overview.feedback_new }), "feedback", ""]);

  const tiles: { label: string; value: string; sub: string; section: string; tip?: Tip }[] = [
    { label: t("ui.admin.overview.tile_gpus"), value: queue ? t("ui.admin.overview.gpus_taking", { taking: takers, all: (queue.gpus ?? []).length }) : "…", sub: queue?.gpus?.map((g) => g.short_name).join(t("list.sep")) || t("ui.admin.overview.no_gpus_found"), section: "cards" },
    { label: t("ui.admin.overview.tile_queue"), value: queue ? t("ui.admin.overview.queue_value", { running, queued: active.length - running }) : "…", sub: t("ui.admin.overview.queue_sub"), section: "queue" },
    { label: t("ui.admin.overview.tile_memory"), value: mem ? t("ui.admin.overview.memory_value", { available: gbText(mem.available_gb) }) : "…", sub: mem ? t("ui.admin.overview.memory_sub", { total: gbText(mem.total_gb), keep: gbText(mem.keep_free_gb) }) : "", section: "settings-compute" },
    { label: t("ui.admin.overview.tile_disk"), value: disk ? t("ui.admin.overview.disk_value", { free: sizeText(disk.free) }) : "…", sub: disk ? t("ui.admin.overview.disk_sub", { total: sizeText(disk.total) }) : "", section: "disk", tip: disk ? tipOf("value", t("ui.admin.overview.disk_tip", { path: disk.path })) : undefined },
    { label: t("ui.admin.overview.tile_resident"), value: overview ? t("ui.admin.overview.resident_value", { n: overview.resident.processes }) : "…", sub: overview ? t("ui.admin.overview.resident_sub", { vram: sizeText(overview.resident.vram_mb * 2 ** 20) }) : "", section: "resident" },
    { label: t("ui.admin.overview.tile_server"), value: overview ? t("ui.admin.overview.server_value", { lasted: lastedText(overview.server.started) }) : "…", sub: overview ? t("ui.admin.overview.server_sub", { address: overview.server.address, version: overview.server.version }) : "", section: "settings-network", tip: overview ? tipOf("value", t("ui.admin.overview.server_tip", { pid: overview.server.pid, command: overview.server.command })) : undefined },
  ];

  return (
    <Section title={t("ui.admin.overview.title")} lede={t("ui.admin.overview.lede")}>
      {alerts.length > 0 && (
        <ul className="adm-alerts">
          {alerts.map(([text, section, tip]) => (
            <li key={text} {...tipAttrs(tipOf("error", tip))}>
              <i />
              <span>{text}</span>
              {section !== "overview" && (
                <Button tone="ghost" onClick={() => go(section)}>
                  {t("ui.admin.overview.go")}
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="adm-tiles">
        {tiles.map((tile) => (
          <button key={tile.section} className="adm-tile" {...tipAttrs(tile.tip)} onClick={() => go(tile.section)}>
            <span className="adm-tile-label">{tile.label}</span>
            <b className="tnum">{tile.value}</b>
            <span className="adm-tile-sub">{tile.sub}</span>
          </button>
        ))}
      </div>
      <RecentPart />
    </Section>
  );
}
