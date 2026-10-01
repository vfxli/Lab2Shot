/** 「再问一次服务器这张图现在怎样」：做法在 graph/actions.ts refreshStatus（要 toJSON、结果 store、上传……），
 * 要叫它的有 graph/document.ts（编辑之后）与 graph/follow.ts（任务结束、节点算完之后）——而 actions 汇总了它们，
 * 它们不能反过来 import actions（环）。actions 载入时把 refreshStatus 登记在这里；没登记时什么也不做。 */

let asker: () => Promise<void> = async () => undefined;

export const askStatus = (): Promise<void> => asker();

export function answerStatusWith(fn: () => Promise<void>): void {
  asker = fn;
}
