import { useEffect, useState } from "react";
import { api } from "../api";
import type { StoragePlan, StorageUsage } from "../api/library";
import { Empty } from "./Empty";
import { Loading } from "./Loading";
import { Button } from "./Button";
import { Num } from "./controls";
import { Sheet } from "./Sheet";
import { messageOf, msg, reasonOf } from "../messages/message";
import { say } from "../state/say";
import { agoText, gbText, sizeText, whenText } from "../platform/format";
import "./storage.css";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const GB = 2 ** 30;

/** 自己的占用（GET /api/my/storage）：队列窗口打开时、删除任务后（`again` 变化时）各读一次，不随队列的轮询请求——
 * 占用在每次计算时都会变化，放入轮询会使本应返回 304 的响应每次都完整重发（api/library.ts StorageGate）。
 * 「我的占用」一段和表格里每个已结束任务的占用读的是同一份。`again` 为 null 时不读（后台的队列）。 */
export function useStorageUsage(again: number | null): { usage: StorageUsage | null; problem: string } {
  const [usage, setUsage] = useState<StorageUsage | null>(null);
  const [problem, setProblem] = useState("");
  useEffect(() => {
    if (again === null) return; // 后台的队列：不是谁自己的占用
    let live = true;
    api.storage().then(
      (v) => live && (setUsage(v), setProblem("")),
      (e: Error) => live && (setProblem(reasonOf(e)), say(messageOf(e))), // every error goes to the log too
    );
    return () => {
      live = false;
    };
  }, [again]);
  return { usage, problem };
}

/** 想腾出多少的起始值：占到 80% 以上时腾到 80% 以下要的量，否则 1 GB（取到 0.01 GB）。 */
const suggested = (u: StorageUsage): number => {
  const need = u.limit ? u.total - 0.8 * u.limit : 0;
  return Math.max(0.01, Math.ceil((need > 0 ? need : GB) / GB * 100) / 100);
};

/** 我的占用：队列窗口中的一段。一行总量（已用 / 上限、占用条、还剩），一行「删除最旧的已完成任务，腾出约 N GB」，
 * 和每账号保留的已完成任务的上限。
 *
 * 占用 = 自己所有任务的文件夹 + 自己的计算缓存 + 还没有任务用到的上传 + 存在服务器上的模板，同一份字节只算一次
 * （lab2shot/server/quota.py）。不分项列出明细，也不提供按部分「清理」：腾出空间的操作均在任务上——表格里每一行的「删除」、
 * 队列上方的「删除全部」，和这里一次删掉最旧几条的「腾出空间」（先预览删哪些、腾多少，确认后执行；腾出的就是预览的数）。
 *
 * 此处不显示流量：流量只在后台「用户」页显示（admin/traffic.tsx、admin/UserQuota.tsx）。服务器的 `/api/my/storage` 根本不
 * 返回 `traffic` 键（lab2shot/server/quota.py my_storage），网页组件中不写角色判断。 */
export function StoragePanel({ usage, problem, onFreed }: { usage: StorageUsage | null; problem: string; onFreed: () => void }) {
  const [gb, setGb] = useState<number | null>(null);
  const [planning, setPlanning] = useState(false);
  const [plan, setPlan] = useState<StoragePlan | null>(null);
  const [freeing, setFreeing] = useState(false);

  if (problem && !usage) return <Empty title={problem} hint={t("ui.storage.reload_hint")} />;
  if (!usage) return <Loading what={t("ui.storage.loading")} />;

  const pct = usage.limit ? Math.min(100, Math.round((usage.total / usage.limit) * 100)) : 0;
  const want = gb ?? suggested(usage);
  const preview = async () => {
    setPlanning(true);
    try {
      setPlan(await api.storagePlan(want));
    } catch (e) {
      say(msg("E-STORAGE-PLANFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setPlanning(false);
    }
  };
  const free = async () => {
    if (!plan) return;
    setFreeing(true);
    try {
      const got = await api.storageFree(plan.jobs.map((j) => j.id));
      say(msg("I-STORAGE-FREED", { jobs: got.jobs, size: sizeText(got.bytes) }));
      if (got.pending > 0) say(msg("N-STORAGE-PENDING", { size: sizeText(got.pending) }));
      setPlan(null);
      onFreed();
    } catch (e) {
      say(msg("E-JOB-FORGETFAILED", { reason: reasonOf(e as Error) }));
    } finally {
      setFreeing(false);
    }
  };
  const nothing = !usage.finished;
  return (
    <div className="sto">
      <div className="sto-head">
        <span className="tnum sto-total">
          {sizeText(usage.total)} / {usage.limit ? gbText(usage.limit / GB) : t("ui.storage.no_limit")}
        </span>
        <span className={`sto-bar${usage.over ? " over" : usage.stage === 1 ? " warn" : ""}`} {...tipAttrs(tipOf("value", t("ui.storage.used_pct", { pct })))}>
          <i style={{ ["--pct" as string]: `${pct}%` }} />
        </span>
        {usage.over ? (
          <span className="chip sto-over">{t("ui.storage.full")}</span>
        ) : (
          <span className="chip sto-left tnum">
            {t("ui.storage.left", { left: usage.limit ? sizeText(usage.left) : t("ui.storage.no_limit") })}
          </span>
        )}
      </div>
      <div className="sto-free">
        <span>{t("ui.storage.free_about")}</span>
        <Num value={want} min={0.01} max={10000} digits={2} label={t("ui.storage.free_gb")} className="sto-gb" onChange={(v) => setGb(v)} />
        <span>GB</span>
        <Button
          size="sm"
          disabled={nothing || planning}
          tip={nothing ? tipOf("disabled", t("ui.queue.wipe_none")) : undefined}
          onClick={() => void preview()}
        >
          {planning ? t("ui.storage.planning") : t("ui.storage.preview")}
        </Button>
      </div>
      <p className="sto-note tnum" data-code={usage.trimmed ? "N-STORAGE-TRIMMED" : "N-STORAGE-KEEPMOST"}>
        {msg("N-STORAGE-KEEPMOST", { count: usage.finished, most: usage.keep_most }).text}
        {usage.trimmed && t("ui.storage.and", { text: msg("N-STORAGE-TRIMMED", { when: agoText(usage.trimmed.at), count: usage.trimmed.count, most: usage.keep_most }).text })}
      </p>
      {problem && (
        <p className="login-problem" role="alert">
          {problem}
        </p>
      )}
      {plan && (
        <Sheet title={t("ui.storage.free_title")} width={560} onClose={() => !freeing && setPlan(null)}>
          {plan.jobs.length ? (
            <>
              <p className="confirm-say" data-code={plan.enough ? "N-STORAGE-FREE" : "N-STORAGE-FREESHORT"}>
                {(plan.enough
                  ? msg("N-STORAGE-FREE", { count: plan.jobs.length, size: sizeText(plan.bytes) })
                  : msg("N-STORAGE-FREESHORT", { count: plan.jobs.length, size: sizeText(plan.bytes), want: gbText(plan.want / GB) })).text}
              </p>
              <div className="q-table-wrap sto-plan">
                <table className="q-table">
                  <thead>
                    <tr>
                      <th>{t("ui.storage.col_job")}</th>
                      <th>{t("ui.storage.col_submitted")}</th>
                      <th>{t("ui.storage.col_own")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {plan.jobs.map((j) => (
                      <tr key={j.id}>
                        <td className="sto-plan-title" data-user-data {...tipAttrs(tipOf("truncated", j.title))}>{j.title}</td>
                        <td className="tnum">{j.submitted ? whenText(j.submitted) : ""}</td>
                        <td className="tnum">{sizeText(j.bytes)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="sto-note tnum">
                {t("ui.storage.total_freed", { size: sizeText(plan.bytes) })}
              </p>
            </>
          ) : (
            <p className="confirm-say">{t("ui.queue.wipe_none")}</p>
          )}
          <div className="dialog-row confirm-end">
            <Button tone="ghost" disabled={freeing} onClick={() => setPlan(null)}>
              {t("ui.common.cancel")}
            </Button>
            {plan.jobs.length > 0 && (
              <Button tone="primary" danger disabled={freeing} onClick={() => void free()}>
                {freeing ? t("ui.queue.deleting") : t("ui.storage.delete_these", { count: plan.jobs.length })}
              </Button>
            )}
          </div>
        </Sheet>
      )}
    </div>
  );
}
