import { useState } from "react";
import { api, type ResidentProcess, type ResidentView } from "../api";
import { adminApi } from "../api/admin";
import { Section, useAdmin } from "./common";
import { sizeText } from "../platform/format";
import { durationText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

const STATE: Record<ResidentProcess["state"], [string, string]> = {
  busy: ["计算中", "var(--accent)"],
  moving: ["移动中", "var(--orange)"],
  gpu: ["显存中", "var(--green)"],
  ram: ["内存中", "var(--text-3)"],
};

const mbText = (mb: number) => (mb ? sizeText(mb * 2 ** 20) : "—");

/** 常驻模型 section: worker processes kept alive between jobs with their models loaded — their location and
 * loaded models — and moving those models to RAM or unloading them. Lifetime and count limits are configured in 设置. */
export function ResidentSection() {
  const { problem, go } = useAdmin();
  const { data: view, reload: reload } = usePoll(api.admin.resident, 3000, { onError: (e) => problem(reasonOf(e)) });
  const { data: settings } = usePoll(adminApi.settings, null, { onError: (e) => problem(reasonOf(e)) });
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
      ? `空闲 ${value("resident.idle_minutes")} 分钟后卸载，每张显卡最多常驻 ${value("resident.per_gpu")} 个，给别的项目让路时${value("resident.to_ram") ? "先移到内存" : "立即卸载"}。`
      : "常驻模型已关掉：每个任务算完就卸载。";

  return (
    <Section
      title="常驻模型"
      lede={
        <>
          算完的模型留在进程里，下一个用同样模型的任务不用重新加载。{policy}
          {current && ` 现在可用内存 ${current.available_gb.toFixed(0)} GB。`}
        </>
      }
      actions={
        <>
          <Button tip="常驻模型留多久、留几个、怎么让路，在「计算与显卡」里改" tone="ghost" onClick={() => go("settings-compute")}>
            改设置
          </Button>
          <Button tip="重新读取常驻进程" tone="ghost" onClick={reload}>
            刷新
          </Button>
        </>
      }
    >
      {current === null ? (
        <p className="adm-lede">读取中…</p>
      ) : current.processes.length ? (
        <div className="q-table-wrap">
          <table className="q-table">
            <thead>
              <tr>
                <th>项目 · 模型</th>
                <th>显卡</th>
                <th>状态</th>
                <th>显存</th>
                <th>内存</th>
                <th>空闲</th>
                <th>任务</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {current.processes.map((p) => {
                const [label, color] = STATE[p.state];
                const working = p.state === "busy" || p.state === "moving";
                const why = working ? "正在计算或移动，等它结束" : undefined;
                return (
                  <tr key={p.id}>
                    <td>
                      <span className="q-title">{p.title}</span>
                      <span className="q-targets">
                        {p.models.map((m, i) => (
                          <span key={m.name} data-tip={m.on_gpu ? `显存中 ${mbText(m.gpu_mb)}` : `内存中 ${mbText(m.ram_mb)}`}>
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
                        {label}
                      </span>
                    </td>
                    <td className="tnum" data-tip="PyTorch 在显卡上占的，另有约 0.5 GB 显卡驱动本身">
                      {mbText(p.vram_mb)}
                    </td>
                    <td className="tnum">{mbText(p.ram_mb)}</td>
                    <td className="tnum">{working ? "—" : durationText(p.idle_s)}</td>
                    <td className="tnum">{p.jobs}</td>
                    <td className="q-act resident-act">
                      <Button
                        tip={why ?? (p.state === "ram" ? "已经在内存里" : "从显存移到内存，下次用时很快移回显卡")}
                        disabled={p.state !== "gpu" || acting === p.id}
                        onClick={() => void act(p.id, api.admin.offload)}
                      >
                        卸载到内存
                      </Button>
                      <Button
                        tip={why ?? "显存和内存都释放，下次用时重新加载"}
                        disabled={working || acting === p.id}
                        onClick={() => void act(p.id, api.admin.unload)}
                      >
                        完全卸载
                      </Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-empty">没有常驻的模型：扩展包把模型留在显存里时，这里列出来</p>
      )}
    </Section>
  );
}
