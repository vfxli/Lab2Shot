import { textOf, type Message } from "../messages/message";

/** A message's words where the page shows them in its own layout (a loading line, a panel that failed): its code and
 * level ride along (data-code, data-level), so a screenshot and 提交反馈 can tell which message it is. */
export function MessageText({ message, className }: { message: Message; className?: string }) {
  return (
    <span className={className} data-code={message.code} data-level={message.level}>
      {textOf(message)}
    </span>
  );
}
