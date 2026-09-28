/** The rules of an account's fields, checked on the page before anything is sent: the same rules as the server's
 * (lab2shot/accounts.py USERNAME, check_name, rule_problem), which checks them again. One copy for every form that
 * makes or changes an account: the admin page's 新建用户 and 改, the password forms, and the login page's 注册 (which
 * is why this is not in the admin page's code: the login page gets nothing of that). */

export const MIN_CHARS = 8; // lab2shot/accounts.py MIN_CHARS / MAX_CHARS
const MAX_CHARS = 128;
const USERNAME = /^[a-z][a-z0-9_.-]{2,31}$/; // lab2shot/accounts.py USERNAME
// the ideographs the server takes (lab2shot/accounts.py _HAN), not all of Unicode's Han script: 〇, 々 and the radicals
// would pass here and be refused there
const HAN = "[\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF\\u{20000}-\\u{3134F}]";
const NAME = new RegExp(`^${HAN}+(·${HAN}+)*$`, "u"); // lab2shot/accounts.py _NAME
const DOTS = /[·•・‧∙]/g;

export const tidyName = (name: string) => name.trim().replace(DOTS, "·");

export function usernameProblem(raw: string): string {
  if (!raw) return "";
  return USERNAME.test(raw.trim().toLowerCase()) ? "" : "用户名要 3 到 32 个字符：小写英文字母开头，只用小写字母、数字和 _ . -";
}

export function nameProblem(raw: string): string {
  const name = tidyName(raw);
  if (!name) return "要填中文名：2 到 6 个字";
  if (/\s/.test(name)) return `名字「${name}」里有空格：少数民族名字用 · 隔开，如 阿依·买买提`;
  if (!NAME.test(name)) return /[A-Za-z]/.test(name) ? `名字「${name}」要写中文名，不是拼音或英文名` : `名字「${name}」只能是中文字，名字之间可以用 · 隔开`;
  const n = [...name.replaceAll("·", "")].length;
  return n < 2 || n > 6 ? `名字「${name}」有 ${n} 个字：中文名要 2 到 6 个字` : "";
}

/** Returns the reason a new password is rejected, or "" if it is acceptable. */
export function passwordProblem(text: string, again: string): string {
  if (!text) return "";
  const chars = [...text].length; // characters as the server counts them (not UTF-16 units: an emoji is one)
  if (chars < MIN_CHARS) return `新密码至少要 ${MIN_CHARS} 个字符`;
  if (chars > MAX_CHARS) return `新密码最多 ${MAX_CHARS} 个字符`;
  if (!text.trim()) return "新密码不能全是空格";
  if (again && again !== text) return "两次输入的新密码不一样";
  return "";
}
