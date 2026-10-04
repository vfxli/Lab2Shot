import { COPYRIGHT } from "../platform/brand";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { BrandMark, IconGrid, IconOpen } from "../ui/icons";
import { useMenuAt } from "./NodeEditor";
import { ProjectNotice } from "./ProjectNotice";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** The node graph while it is empty (the first opening in a browser, or every node deleted): what to do first — a
 * template, a node of one's own (the node menu, as Tab opens it), or a graph file. Gone with the first node; the
 * canvas around it stays usable. */
export function Welcome({ onOpen }: { onOpen: () => void }) {
  const noNodes = useCookInputs((s) => s.order.length === 0);
  const noBoxes = useLook((s) => s.boxes.length === 0);
  const empty = noNodes && noBoxes;
  const setTemplatesOpen = useViewer((s) => s.setTemplatesOpen);
  const menuAt = useMenuAt();
  if (!empty) return null;
  return (
    <div className="welcome">
      <div className="welcome-card">
        <div className="welcome-head">
          <BrandMark big />
          <h2 className="welcome-title">{t("ui.app.welcome.title")}</h2>
          <p className="welcome-sub">{t("ui.app.welcome.sub")}</p>
        </div>
        <div className="welcome-acts">
          <button className="welcome-act main" onClick={() => setTemplatesOpen(true)}>
            <span className="welcome-icon">
              <IconGrid size={14} />
            </span>
            <span className="welcome-name">{t("ui.templates.title")}</span>
            <span className="welcome-desc">{t("ui.app.welcome.templates_desc")}</span>
          </button>
          <button
            className="welcome-act"
            onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect(); // also when Enter clicks it: no pointer position
              menuAt(r.left, r.top);
            }}
          >
            <span className="welcome-icon">Tab</span>
            <span className="welcome-name">{t("ui.app.welcome.add")}</span>
            <span className="welcome-desc">{t("ui.app.welcome.add_desc")}</span>
          </button>
          <button className="welcome-act" onClick={onOpen} {...tipAttrs(tipOf("shortcut", "Ctrl+O"))}>
            <span className="welcome-icon">
              <IconOpen size={14} />
            </span>
            <span className="welcome-name">{t("ui.app.welcome.open")}</span>
            <span className="welcome-desc">{t("ui.app.welcome.open_desc")}</span>
          </button>
        </div>
        <ProjectNotice className="welcome-notice" />
        <p className="welcome-legal">{COPYRIGHT}</p>
      </div>
    </div>
  );
}
