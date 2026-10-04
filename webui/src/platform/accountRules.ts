/** The rules of an account's fields, checked on the page before anything is sent: the same rules as the server's
 * (lab2shot/accounts.py USERNAME, check_name, rule_problem), which checks them again. One copy for every form that
 * makes or changes an account: the admin page's 新建用户 and 改, the password forms, and the login page's 注册 (which
 * is why this is not in the admin page's code: the login page gets nothing of that). */

import { t } from "../i18n/t";

export const MIN_CHARS = 8; // lab2shot/accounts.py MIN_CHARS / MAX_CHARS
const MAX_CHARS = 128;
const USERNAME = /^[a-z][a-z0-9_.-]{2,31}$/; // lab2shot/accounts.py USERNAME
// the ideographs the server takes (lab2shot/accounts.py _HAN), not all of Unicode's Han script: 〇, 々 and the radicals
// would pass here and be refused there
const HAN = "[\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF\\u{20000}-\\u{3134F}]";
const NAME = new RegExp(`^${HAN}+(·${HAN}+)*$`, "u"); // lab2shot/accounts.py _NAME
const LATIN = /^[A-Za-z][A-Za-z0-9]*$/; // lab2shot/accounts.py _LATIN: an English name, letters and digits
const LATIN_MIN = 2; // lab2shot/accounts.py LATIN_MIN / LATIN_MAX
const LATIN_MAX = 20;
const DOTS = /[·•・‧∙]/g;

export const tidyName = (name: string) => name.trim().replace(DOTS, "·");

export function usernameProblem(raw: string): string {
  if (!raw) return "";
  return USERNAME.test(raw.trim().toLowerCase()) ? "" : t("ui.register.rule_username");
}

export function nameProblem(raw: string): string {
  const name = tidyName(raw);
  if (!name) return t("ui.register.rule_name_empty");
  if (/\s/.test(name)) return t("ui.register.rule_name_space", { name });
  if (LATIN.test(name))
    return name.length < LATIN_MIN || name.length > LATIN_MAX ? t("ui.register.rule_name_latin_length", { name, count: name.length }) : "";
  if (!NAME.test(name)) return t("ui.register.rule_name_chars", { name });
  const n = [...name.replaceAll("·", "")].length;
  return n < 2 || n > 6 ? t("ui.register.rule_name_length", { name, count: n }) : "";
}

/** Returns the reason a new password is rejected, or "" if it is acceptable. `plain`: said of 「密码」 (the sign-up
 * form's only password) rather than 「新密码」 (a change of password). */
export function passwordProblem(text: string, again: string, plain = false): string {
  if (!text) return "";
  const chars = [...text].length; // characters as the server counts them (not UTF-16 units: an emoji is one)
  if (chars < MIN_CHARS) return plain ? t("ui.register.rule_password_short_plain", { count: MIN_CHARS }) : t("ui.register.rule_password_short", { count: MIN_CHARS });
  if (chars > MAX_CHARS) return plain ? t("ui.register.rule_password_long_plain", { count: MAX_CHARS }) : t("ui.register.rule_password_long", { count: MAX_CHARS });
  if (!text.trim()) return plain ? t("ui.register.rule_password_blank_plain") : t("ui.register.rule_password_blank");
  if (again && again !== text) return t("ui.register.rule_password_again");
  return "";
}
