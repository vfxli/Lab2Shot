import { useCallback, useEffect, useRef, useState } from "react";
import { UserChips, useAccounts } from "./Resources";
import { api, type JobRecord } from "../api";
import { QueueView } from "../ui/Queue";
import { adminApi } from "../api/admin";
import { useSignedIn } from "../state/session";
import { usable } from "../api/applies";
import { Section, useAdmin } from "./common";
import { Button } from "../ui/Button";
import { Filters } from "../ui/Categories";
import { t } from "../i18n/t";

/** 队列 section: all users' jobs with full information about who started them (the same queue component as the
 * editor's 队列 window), cancellation of any job, and the job log. One table: the jobs waiting and running first, then
 * the finished ones and the job log, each job once; 「按人」 above it narrows the whole table to one account. */
export function QueueSection() {
  const { queue, refreshQueue, problem } = useAdmin();
  const state = useSignedIn();
  const users = useAccounts();
  const [who, setWho] = useState<number | null>(null);
  // the job log: every unexpired task (a group is never cut in half), read by the server for the chosen account; the
  // queue is polled whole and narrowed below. The poll carries the log's version (farm/queue.py listed_version): the
  // log is read again only when it changed, or when another account is chosen
  const [history, setHistory] = useState<JobRecord[] | null>(null);
  const shown = useRef(who); // an answer for an account no longer chosen arrives late and is dropped
  shown.current = who;
  const loadHistory = useCallback(
    () => api.admin.history(who).then((h) => shown.current === who && setHistory(h), (e: Error) => problem(e.message)),
    [problem, who],
  );
  useEffect(() => {
    setHistory(null);
  }, [who]);
  const version = queue?.history_version;
  useEffect(() => {
    void loadHistory();
  }, [loadHistory, version]);
  const refresh = () => (refreshQueue(), void loadHistory());

  const cancel = (id: string) => api.admin.cancel(id).then(refresh, (e: Error) => problem(e.message));

  // 插队: the server logs who moved which job to the front, from which position (admin action log), and returns the
  // updated queue.
  const first = (job: { id: string }) => api.admin.first(job.id).then(refreshQueue, (e: Error) => problem(e.message));

  const setSwitch = (key: "gpu" | "compute", on: boolean) =>
    adminApi.queueSwitches({ [key]: on }).then(refreshQueue, (e: Error) => problem(e.message));
  const graphUrl = usable(state?.applies, "queue.graph") ? api.admin.graphUrl : undefined;

  return (
    <Section
      title={t("ui.admin.queue.title")}
      lede={t("ui.admin.queue.lede")}
      actions={
        <Button tone="ghost" onClick={refresh}>
          {t("ui.admin.common.refresh")}
        </Button>
      }
    >
      {/* Filter by user: narrows the table below to one account's jobs. */}
      {!!users?.length && (
        <Filters>
          <UserChips users={users} chosen={who} onChoose={setWho} />
        </Filters>
      )}
      {queue && history ? (
        <QueueView
          data={who === null ? queue : { ...queue, jobs: queue.jobs.filter((j) => j.client?.user === who) }}
          history={history}
          admin
          applies={state?.applies}
          onCancel={cancel}
          onRefresh={refresh}
          onSwitch={usable(state?.applies, "queue.switches") ? setSwitch : undefined}
          onFirst={usable(state?.applies, "queue.first") ? first : undefined}
          graphUrl={graphUrl}
        />
      ) : (
        <p className="adm-lede">{t("ui.admin.common.reading")}</p>
      )}
    </Section>
  );
}
