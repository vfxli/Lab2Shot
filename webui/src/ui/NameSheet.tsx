import { useState } from "react";
import { Button } from "./Button";
import { Sheet } from "./Sheet";

/** A single name entered in a small sheet (a new category, a rename): the only text input a manager uses here; the
 * templates panel's and the node menu's categories both use it. */
export function NameSheet({ title, label, initial, save, onClose }: {
  title: string; label: string; initial: string; save: (name: string) => Promise<void>; onClose: () => void;
}) {
  const [name, setName] = useState(initial);
  const [busy, setBusy] = useState(false);
  const ready = !!name.trim() && name.trim() !== initial && !busy;
  const go = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      await save(name.trim());
      onClose();
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={420} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span data-tip={label}>{label}</span>
          <input className="field" value={name} autoFocus maxLength={10} aria-label={label} data-tip="最多 10 个字：界面上一行放得下"
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && void go()} />
        </label>
        <div className="dialog-row">
          <Button tip="不改了" tone="ghost" onClick={onClose}>取消</Button>
          <Button tip={ready ? "保存" : "先填个名字"} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? "保存中…" : "保存"}</Button>
        </div>
      </div>
    </Sheet>
  );
}

/** The sheet's parameters: the title, the field's label, the initial name and the handler for the entered name. */
export interface Naming {
  title: string;
  label: string;
  initial: string;
  save: (name: string) => Promise<void>;
}

/** A name and a paragraph entered in a small sheet (a template card's name and intro, a node's name and description). */
export function TextSheet({ title, nameLabel, textLabel, initialName, initialText, nameMax, textMax, save, onClose }: {
  title: string; nameLabel: string; textLabel: string; initialName: string; initialText: string;
  nameMax: number; textMax: number; // the server's limits for this kind of text (lab2shot/library.py, nodes/text.py): every caller must specify them
  save: (name: string, text: string) => Promise<void>; onClose: () => void;
}) {
  const [name, setName] = useState(initialName);
  const [text, setText] = useState(initialText);
  const [busy, setBusy] = useState(false);
  const ready = !!name.trim() && (name.trim() !== initialName || text.trim() !== initialText) && !busy;
  const go = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      await save(name.trim(), text.trim());
      onClose();
    } catch {
      // the caller has reported the failure (a message); the sheet stays open with the text preserved
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={560} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span data-tip={nameLabel}>{nameLabel}<b className="field-count" data-tip={`最多 ${nameMax} 个字：一行放得下`}>{name.length} / {nameMax}</b></span>
          <input className="field" value={name} autoFocus maxLength={nameMax} aria-label={nameLabel} data-tip={`最多 ${nameMax} 个字`}
            onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="login-field">
          <span data-tip={textLabel}>{textLabel}<b className="field-count" data-tip={`最多 ${textMax} 个字：按卡片的大小定`}>{text.length} / {textMax}</b></span>
          <textarea className="field" value={text} rows={6} maxLength={textMax} aria-label={textLabel} data-tip={`最多 ${textMax} 个字`}
            onChange={(e) => setText(e.target.value)} />
        </label>
        <div className="dialog-row">
          <Button tip="不改了" tone="ghost" onClick={onClose}>取消</Button>
          <Button tip={ready ? "保存" : "先填个名字，或改点什么"} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? "保存中…" : "保存"}</Button>
        </div>
      </div>
    </Sheet>
  );
}
