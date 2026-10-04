/** 英文复数（i18n/words.ts、messages/format.ts）的测试：Node 直接跑（node --experimental-strip-types src/i18n/plural.test.ts）。 */

import { addWords, asNumber, pluralCount, pluralForm } from "./words.ts";
import { t } from "./t.ts";
import { addCatalogue, render } from "../messages/format.ts";
import { WORDS, CATALOGUE } from "../messages/generatedCatalogue.ts";

addWords(WORDS);
addCatalogue(CATALOGUE);

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) throw new Error(`${what}\n  得到 ${g}\n  应为 ${w}`);
}

// the rule itself
eq(pluralForm("en", 1), "one", "en 1 -> one");
eq(pluralForm("en", 0), "other", "en 0 -> other");
eq(pluralForm("en", 2), "other", "en 2 -> other");
eq(pluralForm("en", "1"), "one", "en a number written out");
eq(pluralForm("zh", 1), "other", "zh always other");
eq(asNumber("1,234"), 1234, "thousands");
eq(asNumber("1001–1050"), undefined, "a range is no number");
eq(pluralCount("{a} of {b} x", { a: "q", b: 3 }), 3, "first placeholder given a number");
eq(pluralCount("{a} x", { a: 5, count: 1 }), 1, "count first");

// words: a key written in plural forms, read by its key without the form
eq(t("ui.history.nodes_add", { count: 1 }, "en"), "Add 1 Node", "en one");
eq(t("ui.history.nodes_add", { count: 3 }, "en"), "Add 3 Nodes", "en other");
eq(t("ui.history.nodes_add", { count: 1 }, "zh"), t("ui.history.nodes_add", { count: 1 }, "zh"), "zh other");
if (t("ui.history.nodes_add", { count: 1 }, "zh").includes("Node")) throw new Error("zh reads Chinese");

// a plain key reads as before
eq(typeof t("ui.account.language", {}, "en"), "string", "plain key");

// the page's own messages in plural forms (web.toml)
const plural = Object.keys(CATALOGUE.en).find((k) => k.endsWith(".one"));
if (plural) {
  const code = plural.slice(0, -4);
  const other = CATALOGUE.en[`${code}.other`];
  const names = [...other.matchAll(/\{(\w+)(?::[^{}]*)?\}/g)].map((m) => m[1]);
  const params = Object.fromEntries(names.map((n) => [n, n === "count" || names.indexOf(n) === 0 ? 1 : "x"]));
  const one = render(code, params, "en"), many = render(code, { ...params, ...Object.fromEntries(Object.entries(params).map(([k, v]) => [k, v === 1 ? 4 : v])) }, "en");
  if (one === many) throw new Error(`${code}: one and other read the same`);
  count++;
  eq(typeof render(code, params, "zh"), "string", `${code} zh`);
}

console.log(`plural: ${count} 项通过`);
