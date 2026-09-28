import { useCallback, useRef, useState, type ReactNode } from "react";
import type { Message } from "../messages/message";
import { Button } from "./Button";
import { Sheet } from "./Sheet";

/** The page's one confirmation: a sheet that says, from the message catalogue, what an action will do, with
 * 取消 and the action's own word. Never the browser's window.confirm.
 *   const [ask, confirmSheet] = useConfirm();
 *   if (await ask({ title: "清理硬盘", say: msg("N-DISK-CLEAN", {...}), yes: "删掉", tip: "...", danger: true })) ...
 * and render {confirmSheet} where the component renders. Esc, the backdrop and 取消 answer false. */
export interface Ask {
  title: string;
  say: Message;
  yes: string; // the action's own word: 删掉, 清零, 不保存
  tip: string; // what the action button does
  danger?: boolean; // it removes or ends something
}

export function useConfirm(): [(ask: Ask) => Promise<boolean>, ReactNode] {
  const [open, setOpen] = useState<Ask | null>(null);
  const answer = useRef<((yes: boolean) => void) | null>(null);
  const ask = useCallback(
    (a: Ask) =>
      new Promise<boolean>((resolve) => {
        answer.current?.(false); // a question still open is answered no
        answer.current = resolve;
        setOpen(a);
      }),
    [],
  );
  const done = (yes: boolean) => {
    answer.current?.(yes);
    answer.current = null;
    setOpen(null);
  };
  const sheet = open && (
    <Sheet title={open.title} width={480} onClose={() => done(false)}>
      <p className="confirm-say" data-code={open.say.code}>
        {open.say.text}
      </p>
      <div className="dialog-row confirm-end">
        <Button tip="不做，回到刚才" tone="ghost" onClick={() => done(false)}>
          取消
        </Button>
        <Button tip={open.tip} tone="primary" danger={open.danger} data-field="confirm-yes" onClick={() => done(true)}>
          {open.yes}
        </Button>
      </div>
    </Sheet>
  );
  return [ask, sheet];
}
