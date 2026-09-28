import { useCallback, useState } from "react";
import { adminApi, type DatabaseView } from "../api/admin";
import { Section, useAdmin } from "./common";
import { fullTimeText, lastedText, sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

/** 数据库 section: where the records are stored (lab2shot/database) and their safety status — size, version,
 * the last integrity check, retained backups (立即备份), and how to restore one. */

const REASONS: Record<string, string> = {
  created: "建库时", daily: "每天", manual: "手动", merge: "合并名字前", "undo-merge": "撤销合并前",
};
const reasonText = (r: string) => REASONS[r] ?? (r.startsWith("before-v") ? `升级到第 ${r.slice(8)} 版前` : r);
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

  if (!d) return <Section title="数据库">{<p className="adm-lede">读取中…</p>}</Section>;
  const ok = d.checked.ok !== false;
  return (
    <Section
      title="数据库"
      lede="账号、任务记录、使用统计、结果的状态、用时记录都在这一个数据库文件里；缓存、上传的素材和交出的文件还是普通文件。每次写入要么完整写进去、要么一点不留，服务被强行结束也不会写坏。每天自动备份一份，升级数据库前也备份一份。"
      actions={
        <>
          <Button tip="从头到尾检查一遍数据库的完整性（几秒钟，不用停服务）" tone="ghost" disabled={!!busy} onClick={() => void act("check", adminApi.checkNow)}>
            {busy === "check" ? "检查中…" : "检查完整性"}
          </Button>
          <Button tip="现在就在线备份一份（不用停服务），和自动备份放在一起" disabled={!!busy} onClick={() => void act("backup", adminApi.backupNow)}>
            {busy === "backup" ? "备份中…" : "立即备份"}
          </Button>
          <Button tip="重新读取数据库状态" tone="ghost" onClick={() => (setView(null), reload())}>
            刷新
          </Button>
        </>
      }
    >
      {!ok && <div className="notice">完整性检查发现问题：{d.checked.detail}。停掉服务后用下面的方法从备份恢复。</div>}
      <div className="adm-tiles db-tiles">
        <div className="adm-tile" data-tip={d.path}>
          <span className="adm-tile-label">大小</span>
          <b className="tnum">{sizeText(d.bytes)}</b>
          <span className="adm-tile-sub">第 {d.version} 版</span>
        </div>
        <div className="adm-tile" data-tip={d.checked.at ? `检查于 ${fullTimeText(d.checked.at)}：${d.checked.detail}` : "服务启动时检查"}>
          <span className="adm-tile-label">完整性</span>
          <b className={ok ? "db-ok" : "db-bad"}>{ok ? "完好" : "有问题"}</b>
          <span className="adm-tile-sub">{d.checked.at ? `${lastedText(d.checked.at)}前检查` : "启动时检查"}</span>
        </div>
        <div className="adm-tile" data-tip={d.last_backup ? `${d.last_backup.file}` : "还没有备份"}>
          <span className="adm-tile-label">上次备份</span>
          <b>{d.last_backup ? `${lastedText(d.last_backup.at)}前` : "没有"}</b>
          <span className="adm-tile-sub">{d.last_backup ? reasonText(d.last_backup.reason) : ""}</span>
        </div>
        <div className="adm-tile" data-tip="自动留几份在「存储与视图」里改">
          <span className="adm-tile-label">备份</span>
          <b className="tnum">{d.backups.length} 份</b>
          <span className="adm-tile-sub">最多留 {d.keep} 份</span>
        </div>
      </div>

      <h3 className="adm-h3">备份</h3>
      <div className="q-table-wrap">
        <table className="q-table">
          <thead>
            <tr>
              <th>时间</th>
              <th>原因</th>
              <th>大小</th>
              <th>文件</th>
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
        从备份恢复：先停掉服务，在 Lab2Shot 文件夹里执行 <code>uv run lab2shot db restore 备份文件名</code>，再启动服务。恢复前会检查备份本身，换下来的数据库留在
        work/db/ 里（lab2shot.db.replaced-时间），不会删。
      </p>
    </Section>
  );
}
