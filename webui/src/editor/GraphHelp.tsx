import { useState } from "react";
import { msg } from "../messages/message";
import { MessageText } from "../ui/MessageText";
import { IconButton } from "../ui/Button";

/** 操作说明: a 「?」 in the node graph's corner; the pointer over it opens the short table of what the mouse and the
 * keys do — one line each, the gesture on the left and what it does on the right. The 「?」 itself has no tip: the
 * table is what it says. The sentences are the message
 * catalogue's (lab2shot/messages/web.toml I-EDIT-*); the gestures' own names are UI vocabulary and sit here, beside
 * what they name.
 *
 * The division: 右键 only ever opens a menu where it is clicked, 中键 pans everywhere, 左键 selects and drags, and
 * 数据信息 is the mark at a node's bottom right — no shortcut. editor/graphPointer.ts does it. */
const HELP = [
  { title: "左键", code: "I-EDIT-LEFT" },
  { title: "中键", code: "I-EDIT-MIDDLE" },
  { title: "右键", code: "I-EDIT-RIGHT" },
  { title: "滚轮", code: "I-EDIT-WHEEL" },
  { title: "连线", code: "I-EDIT-WIRING" },
  { title: "改接", code: "I-EDIT-REWIRE" },
  { title: "数据信息", code: "I-EDIT-INFO" },
  { title: "快捷键", code: "I-EDIT-KEYS" },
];

export function GraphHelp() {
  const [open, setOpen] = useState(false);
  return (
    <div className="graph-help" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <IconButton tone="ghost" size="xs" layout="graph-help-mark" aria-label="操作说明" aria-expanded={open}
        onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onClick={() => setOpen((o) => !o)}>
        ?
      </IconButton>
      {open && (
        <div className="popover graph-help-panel glass strong" role="dialog" aria-label="操作说明">
          {HELP.map((h) => (
            <div key={h.code} className="graph-help-row">
              <span className="graph-help-title">{h.title}</span>
              <MessageText className="graph-help-text" message={msg(h.code)} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
