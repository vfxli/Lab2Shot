import { useCallback, useEffect, useState } from "react";
import { ByUser } from "./Resources";
import { api, type JobRecord } from "../api";
import { fromRecord, JobTable, QueueView } from "../ui/Queue";
import { adminApi } from "../api/admin";
import { useSignedIn } from "../state/session";
import { usable } from "../api/applies";
import { Section, useAdmin } from "./common";
import { Button } from "../ui/Button";

/** 队列 section: all users' jobs with full information about who started them (the same queue component as the
 * editor's 队列 window), cancellation of any job, and the job log. */
export function QueueSection() {
  const { queue, refreshQueue, problem } = useAdmin();
  const state = useSignedIn();
  const [history, setHistory] = useState<JobRecord[] | null>(null);
  const loadHistory = useCallback(() => api.admin.history().then(setHistory, (e: Error) => problem(e.message)), [problem]);
  useEffect(() => void loadHistory(), [loadHistory]);

  const cancel = (id: string) =>
    api.admin.cancel(id).then(
      () => (refreshQueue(), void loadHistory()),
      (e: Error) => problem(e.message),
    );

  // Drag to reorder: the server logs who moved which job from which position to which (admin action log) and returns the updated queue.
  const reorder = (job: { id: string }, position: number) =>
    api.admin.place(job.id, position).then(refreshQueue, (e: Error) => problem(e.message));

  const setSwitch = (key: "gpu" | "compute", on: boolean) =>
    adminApi.queueSwitches({ [key]: on }).then(refreshQueue, (e: Error) => problem(e.message));
  const graphUrl = usable(state?.applies, "queue.graph") ? api.admin.graphUrl : undefined;

  return (
    <>
      <Section title="队列" lede="所有人的任务，按真正要算的顺序排：每张接任务的显卡同时算一个，几个账号轮流来。点一行看提交者的全部记录。">
        {queue ? (
          <>
            <QueueView
              data={queue}
              admin
              onCancel={cancel}
              onSwitch={usable(state?.applies, "queue.switches") ? setSwitch : undefined}
              onReorder={usable(state?.applies, "queue.reorder") ? reorder : undefined}
              graphUrl={graphUrl}
            />
          </>
        ) : (
          <p className="adm-lede">读取中…</p>
        )}
      </Section>
      <Section
        title="任务记录"
        lede="所有任务，最新的在前：谁、在哪台机器、什么时候、算了什么、结果如何。服务重启时正在算、又没排回队列的任务记为「中断」。"
        actions={
          <Button tip="重新读取任务记录" tone="ghost" onClick={() => void loadHistory()}>
            刷新
          </Button>
        }
      >
        {history === null ? (
          <p className="adm-lede">读取中…</p>
        ) : history.length ? (
          <JobTable jobs={history.map((r) => ({ job: fromRecord(r) }))} admin onCancel={cancel} onForgotten={() => (refreshQueue(), void loadHistory())} graphUrl={graphUrl} />
        ) : (
          <p className="adm-empty">还没有任务：有人提交计算以后，这里列出来</p>
        )}
        {/* Filter by user: the same table and listing function as the 用户 detail page. */}
        <ByUser section="queue" />
      </Section>
    </>
  );
}
