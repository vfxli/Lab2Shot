/** 节点名与备注的编辑控件（Houdini 的 node name / node comment）：节点上双击名字就地改、参数面板顶部的名字栏与备注栏、
 * 右键菜单的「编辑备注」都用这里的同一套。改名的规则与同步在 graph/edit.ts renameNode（不合规就地说明、不改），
 * 备注在 graph/edit.ts setComment。 */
import { useEffect, useRef, useState } from "react";
import { renameNode, setComment } from "../graph/actions";
import { useLook } from "../state/look";
import { composing } from "../platform/keys";
import { t } from "../i18n/t";

/** 节点名输入框：回车或点到别处时生效，Esc 不改；不合规或重名时保持输入、就地说明原因（`onDone` 只在生效或取消时调用）。 */
export function NodeNameInput({ id, className, autoFocus, disabled, onDone, revertOnBlur }: { id: string; className?: string; autoFocus?: boolean; disabled?: boolean; onDone?: () => void; revertOnBlur?: boolean }) {
  const [text, setText] = useState(id);
  const [problem, setProblem] = useState<string | null>(null);
  const cancelled = useRef(false);
  useEffect(() => {
    setText(id);
    setProblem(null);
  }, [id]);
  const commit = () => {
    if (cancelled.current) {
      cancelled.current = false;
      setText(id);
      setProblem(null);
      onDone?.();
      return;
    }
    const why = renameNode(id, text);
    // 节点上就地改：点到别处时不合规就退回原名（原因已在回车时说过）；面板里保留输入和原因，改对为止
    if (why && revertOnBlur) {
      setText(id);
      setProblem(null);
      onDone?.();
      return;
    }
    setProblem(why);
    if (!why) onDone?.();
  };
  return (
    <span className={`node-name-edit${problem ? " bad" : ""}`}>
      <input
        className={`${className ?? ""} nodrag`}
        value={text}
        disabled={disabled}
        autoFocus={autoFocus}
        spellCheck={false}
        aria-label={t("ui.params.naming.name")}
        aria-invalid={!!problem}
        size={Math.max(6, text.length + 1)}
        onFocus={(e) => autoFocus && e.currentTarget.select()}
        onChange={(e) => (setText(e.target.value), setProblem(null))}
        onDoubleClick={(e) => e.stopPropagation()}
        onBlur={commit}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter" && !composing(e)) {
            const why = renameNode(id, text);
            setProblem(why);
            if (!why) onDone?.();
          }
          if (e.key === "Escape") {
            cancelled.current = true;
            e.currentTarget.blur();
          }
        }}
      />
      {problem && <span className="node-name-problem" role="alert">{problem}</span>}
    </span>
  );
}

/** 备注栏（参数面板顶部）：多行文字，点到别处时写入（一步撤销）；旁边「显示备注」开关决定节点旁边是否显示。 */
export function NodeCommentField({ id, disabled }: { id: string; disabled?: boolean }) {
  const comment = useLook((s) => s.comments[id]);
  const [text, setText] = useState(comment?.text ?? "");
  useEffect(() => setText(comment?.text ?? ""), [comment?.text, id]);
  return (
    <div className="node-comment-field">
      <textarea
        value={text}
        disabled={disabled}
        placeholder={t("ui.params.naming.comment_placeholder")}
        aria-label={t("ui.params.naming.comment")}
        rows={Math.min(6, Math.max(1, text.split("\n").length))}
        onChange={(e) => setText(e.target.value)}
        onBlur={() => text !== (comment?.text ?? "") && setComment(id, text)}
        onKeyDown={(e) => e.stopPropagation()}
      />
      <label className="node-comment-show">
        <input type="checkbox" disabled={disabled || !comment} checked={!!comment?.show}
          onChange={(e) => comment && setComment(id, comment.text, e.target.checked)} />
        {t("ui.params.naming.show_comment")}
      </label>
    </div>
  );
}
