import { useCallback, useState } from "react";
import { ByUser } from "./Resources";
import { api, type DiskArea } from "../api";
import { Section, useAdmin } from "./common";
import { sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { msg, type Message } from "../messages/message";

/** 硬盘 section: disk usage of the task folders (their outputs inside), every account's cache and uploads on the server, and cleanup
 * of what is older. Everything is kept per task (lab2shot/farm/disk.py): 任务保留天数 is configured in 设置. */
export function DiskSection() {
  const { problem, go, overview } = useAdmin();
  const read = useCallback(() => api.admin.disk(), []);
  const { data: areas, reload: reload } = usePoll(read, null, { onError: (e) => problem(reasonOf(e)) });
  const [days, setDays] = useState(30);
  const [ask, confirmSheet] = useConfirm();
  const [cleaned, setCleaned] = useState<Message | null>(null);

  const clean = async (a: DiskArea) => {
    if (!(await ask({ title: "清理硬盘", say: msg("N-DISK-CLEAN", { area: a.label, days, note: a.note }), yes: "删掉", tip: `删掉「${a.label}」里 ${days} 天没用过的内容`, danger: true }))) return;
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
      title="硬盘"
      lede={
        <>
          一切按任务保存：任务结束后过了保留天数整个删除，缓存和素材跟着没有任务再用的时候清掉。这里的「清理」按同样的规则，
          不会动还有任务在用的内容；队列里有任务时不清理缓存和素材。
          {disk && ` 工作文件夹 ${disk.path} 所在的盘还剩 ${sizeText(disk.free)}，共 ${sizeText(disk.total)}。`}
        </>
      }
      actions={
        <>
          <Button tip="任务保留天数、单任务上传上限、数据位置，在「存储与视图」里改" tone="ghost" onClick={() => go("settings-storage")}>
            自动清理设置
          </Button>
          <Button tip="重新读取磁盘占用" tone="ghost" onClick={reload}>
            刷新
          </Button>
        </>
      }
    >
      <div className="disk-days">
        <label htmlFor="disk-days" data-tip="下面的「清理」按钮删掉这么多天没被用到的内容；0 表示全部">
          清理多少天没用过的
        </label>
        <input id="disk-days" className="field" type="number" data-tip="下面的「清理」按钮删掉这么多天没被用到的内容；0 表示全部" min={0} value={days} onChange={(e) => setDays(Math.max(0, Number(e.target.value)))} />
        <span>天</span>
      </div>
      {areas ? (
        <div className="q-table-wrap">
          <table className="q-table">
            <thead>
              <tr>
                <th>内容</th>
                <th>占用</th>
                <th>项数</th>
                <th>7 天没用过</th>
                <th>30 天没用过</th>
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
                    <Button tip={`删掉「${a.label}」里 ${days} 天没用过的，先问一句`} onClick={() => void clean(a)}>
                      清理
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-lede">统计中…</p>
      )}
      {cleaned && <div className="notice" data-code={cleaned.code}>{cleaned.text}</div>}
      {confirmSheet}
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="disk" />
    </Section>
  );
}
