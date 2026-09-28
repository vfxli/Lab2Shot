import { Section, useAdmin } from "./common";
import { gbText, lastedText, sizeText } from "../platform/format";
import { Button } from "../ui/Button";
import { RecentPart } from "./Recent";

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

  const alerts: [string, string, string][] = []; // Entries: text, section, tooltip.
  if (queue && (queue.gpus ?? []).length && !takers) alerts.push(["还没有显卡接任务：用到显卡的任务会一直排队", "cards", "到「显卡」里打开显卡的开关，它就开始接队列里的任务"]);
  if (mem && mem.available_gb < mem.keep_free_gb) alerts.push([`可用内存 ${gbText(mem.available_gb)}，少于保留的 ${gbText(mem.keep_free_gb)}：任务会等内存`, "settings-compute", "机器上别的程序占了内存；也可以在「计算与显卡」里调低保留内存"]);
  if (diskLow) alerts.push([`硬盘只剩 ${sizeText(disk!.free)}`, "disk", "在「硬盘」里清理缓存和上传的素材，或者在「存储与视图」里让它们早点自动清理"]);
  if (overview?.pending.length) alerts.push([`${overview.pending.map((p) => p.label).join("、")} 改了，重启服务后生效`, `settings-${overview.pending[0].page}`, "点页面上方的「重启服务」"]);
  const reg = overview?.registering;
  if (reg?.paused)
    alerts.push([
      `注册自动暂停了：最近一${reg.paused === "hour" ? "小时" : "天"}已有 ${reg.paused === "hour" ? reg.hour : reg.day} 个账号自己注册，到了上限`,
      "settings-register",
      "过了这段时间自己恢复；是正常的来人就在「注册设置」里调高上限，是有人刷就在「注册设置」下面的邀请码里按时间段停用这些账号",
    ]);
  if (overview?.feedback_new) alerts.push([`${overview.feedback_new} 条新的用户反馈`, "feedback", "在「用户反馈」里看写了什么、附带的截图和资料"]);

  const tiles: { label: string; value: string; sub: string; section: string; tip: string }[] = [
    { label: "显卡", value: queue ? `${takers} / ${(queue.gpus ?? []).length} 接任务` : "…", sub: queue?.gpus?.map((g) => g.short_name).join("、") || "没有找到显卡", section: "cards", tip: "授权接任务的显卡 / 这台机器的显卡；每张卡接不接任务，点开到「显卡」里开关" },
    { label: "队列", value: queue ? `${running} 计算中 · ${active.length - running} 排队` : "…", sub: "所有人的任务", section: "queue", tip: "打开「队列」看每个任务、谁提交的、取消任务" },
    { label: "内存", value: mem ? `可用 ${gbText(mem.available_gb)}` : "…", sub: mem ? `共 ${gbText(mem.total_gb)} · 保留 ${gbText(mem.keep_free_gb)}` : "", section: "settings-compute", tip: "这台机器现在可用的内存；保留多少在「计算与显卡」里改" },
    { label: "硬盘", value: disk ? `剩 ${sizeText(disk.free)}` : "…", sub: disk ? `共 ${sizeText(disk.total)} · 工作文件夹所在盘` : "", section: "disk", tip: disk ? `工作文件夹 ${disk.path}；各类内容占多少，在「硬盘」里看` : "" },
    { label: "常驻模型", value: overview ? `${overview.resident.processes} 个进程` : "…", sub: overview ? `显存 ${sizeText(overview.resident.vram_mb * 2 ** 20)}` : "", section: "resident", tip: "算完留在进程里的模型；在「常驻模型」里卸载" },
    { label: "服务", value: overview ? `已运行 ${lastedText(overview.server.started)}` : "…", sub: overview ? `${overview.server.address} · 版本 ${overview.server.version}` : "", section: "settings-network", tip: overview ? `进程 ${overview.server.pid}\n启动命令：${overview.server.command}` : "" },
  ];

  return (
    <Section title="概览" lede="这台机器和队列现在的样子，下面是今天和最近的数字。点一块进到对应的页面。">
      {alerts.length > 0 && (
        <ul className="adm-alerts">
          {alerts.map(([text, section, tip]) => (
            <li key={text} data-tip={tip}>
              <i />
              <span>{text}</span>
              {section !== "overview" && (
                <Button tip="打开这一项对应的管理页面" tone="ghost" onClick={() => go(section)}>
                  去处理
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="adm-tiles">
        {tiles.map((t) => (
          <button key={t.label} className="adm-tile" data-tip={t.tip} onClick={() => go(t.section)}>
            <span className="adm-tile-label">{t.label}</span>
            <b className="tnum">{t.value}</b>
            <span className="adm-tile-sub">{t.sub}</span>
          </button>
        ))}
      </div>
      <RecentPart />
    </Section>
  );
}
