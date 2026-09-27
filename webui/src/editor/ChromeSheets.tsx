/** The editor's sheets and banner: 队列, 未保存 and the other-tab banner (模板 has its own file,
 * editor/Templates.tsx). */


import { api, type QueueView as QueueData } from "../api";
import { useViewer } from "../state/viewer";
import { QueueView } from "../ui/Queue";
import { Sheet } from "../ui/Sheet";
import { msg, reasonOf, say } from "../state/say";
import { Button } from "../ui/Button";
import { Banner } from "../ui/Banner";
import { Loading } from "../ui/Loading";

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
  // 默认的 min(90vw, 1800px) 过宽；900 恰好容纳拆成两列的「提交 · 时间」且不出现横向滚动条，
  // 节点图列优先收缩：它是使用者自己的数据，应优先让出空间（queue.css .q-title）
  return (
    <Sheet title="队列" width={900} solid onClose={onClose}>
      {data ? <QueueView data={data} onCancel={cancel} onLoad={(id) => void load(id)} onRefresh={onRefresh} /> : <Loading what="队列" />}
    </Sheet>
  );
}

/** Replacing a graph that has unsaved changes asks first, like a DCC opening another scene. Whatever is chosen,
 * the changes are also kept in the autosave history. */
export function UnsavedSheet({ onChoice }: { onChoice: (choice: "save" | "discard" | "cancel") => void }) {
  const file = useViewer((s) => s.file);
  return (
    <Sheet title="当前节点图有还没保存的修改" width={520} onClose={() => onChoice("cancel")}>
      <p className="tpl-desc" style={{ fontSize: 13 }}>
        {file ? `打开新的节点图之前，要先把修改存进 ${file.name} 吗？` : "这个节点图还没有存成文件。打开新的节点图之前，可以先保存。"}
        不保存的话，这些修改也会留在这个浏览器的自动保存历史里。
      </p>
      <div className="dialog-row" style={{ justifyContent: "flex-end" }}>
        <Button tip="不打开新的节点图，留在当前这个" tone="ghost" onClick={() => onChoice("cancel")}>
          取消
        </Button>
        <Button tip="不保存修改，打开新的节点图（修改留在自动保存历史里）" onClick={() => onChoice("discard")}>
          不保存打开
        </Button>
        <Button tip="先保存当前节点图，再打开新的" tone="primary" onClick={() => onChoice("save")} autoFocus>
          {file ? "保存并打开" : "先保存…"}
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
          <Button tip="在这个标签页编辑；另一个标签页变成只看" tone="primary" size="sm" onClick={claim}>
            在这里编辑
          </Button>
          {banner === "same-open" && (
            <Button tip="这里只看不改，另一个标签页照常编辑" tone="ghost" size="sm" onClick={stay}>
              只看
            </Button>
          )}
        </>
      }
    >
      {banner === "demoted" ? "这张节点图在另一个标签页里正在编辑：这里改不了" : "这张节点图在另一个标签页里也开着：在这里改会覆盖那边"}
    </Banner>
  );
}
