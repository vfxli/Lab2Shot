import { COPYRIGHT } from "../platform/brand";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { BrandMark, IconGrid, IconOpen } from "../ui/icons";
import { useMenuAt } from "./NodeEditor";
import { ProjectNotice } from "./ProjectNotice";

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
          <h2 className="welcome-title">开始一张节点图</h2>
          <p className="welcome-sub">从模板开始最快，也可以自己一个个加节点</p>
        </div>
        <div className="welcome-acts">
          <button className="welcome-act main" onClick={() => setTemplatesOpen(true)} data-tip="内置的模板：现成的流程，选好素材、按面板上的按钮一步步算就能出结果">
            <span className="welcome-icon">
              <IconGrid size={14} />
            </span>
            <span className="welcome-name">从模板新建</span>
            <span className="welcome-desc">现成的流程，选好素材就能算</span>
          </button>
          <button
            className="welcome-act"
            onClick={(e) => {
              const r = e.currentTarget.getBoundingClientRect(); // also when Enter clicks it: no pointer position
              menuAt(r.left, r.top);
            }}
            data-tip="打开节点菜单：按分类找节点，或输入名字搜索。在节点图里任何地方按 Tab 或点右键也一样"
          >
            <span className="welcome-icon">Tab</span>
            <span className="welcome-name">添加节点</span>
            <span className="welcome-desc">在节点图里按 Tab 或点右键</span>
          </button>
          <button className="welcome-act" onClick={onOpen} data-tip="打开本机的节点图文件（Ctrl+O）">
            <span className="welcome-icon">
              <IconOpen size={14} />
            </span>
            <span className="welcome-name">打开节点图</span>
            <span className="welcome-desc">本机上存过的节点图文件</span>
          </button>
        </div>
        <ProjectNotice className="welcome-notice" />
        <p className="welcome-legal">{COPYRIGHT}</p>
      </div>
    </div>
  );
}
