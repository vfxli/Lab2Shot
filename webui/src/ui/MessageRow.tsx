import type { Message } from "../messages/message";
import { LEVEL_TABLE } from "../messages/levels";
import { t } from "../i18n/t";
import "./message.css";
import { tipAttrs, tipOf } from "../platform/tips";

/** One message as displayed by the page: the level's square with its letter, the text and the code at the end, a
 * sentence displayed in place (the information card's 提醒). The level's colour comes from the single table
 * (messages/levels.ts); red is used only for production risk. */
export function MessageRow({ message }: { message: Message }) {
  return (
    <div className={`msg-row ${message.level}`} data-code={message.code} data-level={message.level}>
      <i className="msg-dot" {...tipAttrs(tipOf("value", t(LEVEL_TABLE[message.level].tip)))}>
        {message.level}
      </i>
      <span className="msg-text">
        {message.text}
        <span className="msg-code">{message.code}</span>
      </span>
    </div>
  );
}
