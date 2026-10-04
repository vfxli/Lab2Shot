import { t } from "../i18n/t";

/** The project's word to the research authors, with the address to write to about a licence: always in sight on the
 * node graph — its bottom left corner, or under the welcome card while the graph is empty (editor/Welcome.tsx), where
 * the corner would run under the card. Plain text over everything, taking no pointer: every click, drag and wheel
 * goes through to what is under it. */
export function ProjectNotice({ className }: { className: string }) {
  return (
    <p className={className}>
      {t("ui.editor.project_notice")}
    </p>
  );
}
