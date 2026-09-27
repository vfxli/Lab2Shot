import "./loading.css";
import { msg } from "../messages/message";
import { MessageText } from "./MessageText";

/** What the page shows while a part of it loads: what is being read, never a blank page. `what`: that part, as the
 * user calls it (编辑器, 管理页面). `fill`: over the whole page. */
export function Loading({ what, fill = false }: { what: string; fill?: boolean }) {
  return (
    <div className={fill ? "loading loading-fill" : "loading"} role="status" aria-busy="true">
      <span className="spinner" />
      <MessageText message={msg("I-PAGE-LOADING", { what })} />
    </div>
  );
}
