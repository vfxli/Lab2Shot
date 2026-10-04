/** Owns the editor's sheets and banner: 队列, 未保存 and the other-tab banner (模板 has its own file,
 * editor/Templates.tsx), plus the OPEN_GRAPH event that asks the editor to open a graph. */


import { api, type QueueView as QueueData } from "../api";
import { useViewer } from "../state/viewer";
import { QueueView } from "../ui/Queue";
import { Sheet } from "../ui/Sheet";
import { msg, reasonOf, say } from "../state/say";
import { Button } from "../ui/Button";
import { Banner } from "../ui/Banner";
import { Loading } from "../ui/Loading";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** Sent to the editor (App.tsx) to open a graph as a new unsaved document: it asks first when the one open has unsaved
 * changes. */
export const OPEN_GRAPH = "lab2shot:open-graph";

export function QueueSheet({ data, onRefresh, onClose }: { data: QueueData | null; onRefresh: () => void; onClose: () => void }) {
  const cancel = (id: string) => api.cancelCook(id).catch((e: Error) => say(msg("E-JOB-CANCELFAILED", { reason: reasonOf(e) })));
  const load = async (id: string) => {
    try {
      const job = await api.job(id);
      onClose();
      window.dispatchEvent(new CustomEvent(OPEN_GRAPH, { detail: job.graph }));
      say(msg(job.cache.mark === "all" ? "N-JOB-LOADED" : job.cache.mark === "some" ? "N-JOB-LOADEDSOME" : "N-JOB-LOADEDNONE", { title: job.title }));
    } catch (e) {
      say(msg("E-JOB-LOADFAILED", { reason: reasonOf(e) }));
    }
  };
  // The default min(90vw, 1800px) is too wide; 900 fits 「提交 · 时间」 split into two columns without a horizontal
  // scrollbar. The graph-name column shrinks first: it is the user's own data and gives up space first (queue.css .q-title)
  return (
    <Sheet title={t("ui.editor.queue")} width={900} solid onClose={onClose}>
      {data ? <QueueView data={data} onCancel={cancel} onLoad={(id) => void load(id)} onRefresh={onRefresh} /> : <Loading what={t("ui.editor.queue")} />}
    </Sheet>
  );
}

/** Replacing a graph that has unsaved changes asks first, like a DCC opening another scene. */
export function UnsavedSheet({ onChoice }: { onChoice: (choice: "save" | "discard" | "cancel") => void }) {
  const file = useViewer((s) => s.file);
  return (
    <Sheet title={t("ui.editor.unsaved_title")} width={520} onClose={() => onChoice("cancel")}>
      <p className="tpl-desc" style={{ fontSize: 13 }}>
        {file ? t("ui.editor.unsaved_file", { name: file.name }) : t("ui.editor.unsaved_nofile")}
        {t("ui.editor.unsaved_lost")}
      </p>
      <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
        <Button tone="ghost" onClick={() => onChoice("cancel")}>
          {t("ui.common.cancel")}
        </Button>
        <Button tip={tipOf("consequence", t("ui.editor.discard_tip"))} onClick={() => onChoice("discard")}>
          {t("ui.editor.discard")}
        </Button>
        <Button tone="primary" onClick={() => onChoice("save")} autoFocus>
          {file ? t("ui.editor.save_open") : t("ui.editor.save_first")}
        </Button>
      </div>
    </Sheet>
  );
}

/** tabs.ts: this graph is open in another tab of this browser too. "same-open": this tab is the later one, offered
 * the choice; "demoted": this tab was editing and just lost that to a tab that claimed it. Either way 在这里编辑
 * claims it back (the other tab is told to become a viewer in turn); 只看 (only on "same-open") just clears the
 * offer and stays a viewer. */
export function TabBanner() {
  const banner = useViewer((s) => s.peerBanner);
  const claim = useViewer((s) => s.claimEditing);
  const stay = useViewer((s) => s.stayViewer);
  if (!banner) return null;
  return (
    <Banner
      tone="warn"
      float
      actions={
        <>
          <Button tip={tipOf("consequence", t("ui.editor.claim_tip"))} tone="primary" size="sm" onClick={claim}>
            {t("ui.editor.claim")}
          </Button>
          {banner === "same-open" && (
            <Button tone="ghost" size="sm" onClick={stay}>
              {t("ui.editor.view_only")}
            </Button>
          )}
        </>
      }
    >
      {banner === "demoted" ? t("ui.editor.tab_demoted") : t("ui.editor.tab_same")}
    </Banner>
  );
}
