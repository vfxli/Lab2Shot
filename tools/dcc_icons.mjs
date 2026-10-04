#!/usr/bin/env node
/* The DCC panel's icons, rendered from the page's own React icons (webui/src/ui/icons.tsx, ui/CategoryGlyph.tsx): one
 * source for both, the page's code untouched (设计_DCC新面板.md §2.2, plan A). Each exported Icon* component is rendered
 * with color="currentColor" to clients/common/lab2shot_dcc/ui/icons/<name>.svg (IconChevron → chevron.svg); each
 * category id with a drawing of its own (templates/_categories.json and menu/categories.json, first level) to
 * icons/category-<id>.svg — an id that draws as the plain dot gets no file (the panel draws its own dot). The panel
 * colours `currentColor` at run time (lab2shot_dcc/ui/theme.py icon).
 *
 *   node tools/dcc_icons.mjs           # write the files
 *   node tools/dcc_icons.mjs --check   # exit 1 when a file differs from what would be written (or one is extra)
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const webui = join(root, "webui");
const out = join(root, "clients", "common", "lab2shot_dcc", "ui", "icons");
const check = process.argv.includes("--check");

const need = createRequire(join(webui, "package.json"));
const { createServer } = await import(pathToFileURL(need.resolve("vite")).href);
const vite = await createServer({ root: webui, configFile: false, server: { middlewareMode: true, hmr: false }, appType: "custom",
  logLevel: "error", esbuild: { jsx: "automatic" }, oxc: { jsx: { runtime: "automatic" } } });
const React = need("react");
const { renderToStaticMarkup } = need("react-dom/server");
const icons = await vite.ssrLoadModule("/src/ui/icons.tsx");
const { CategoryGlyph } = await vite.ssrLoadModule("/src/ui/CategoryGlyph.tsx");

const files = new Map();
const svgOf = (markup) => {
  const m = markup.match(/<svg[\s\S]*<\/svg>/);
  if (!m) throw new Error("no <svg> in " + markup.slice(0, 80));
  // sized by the panel: the 16×16 viewBox stays, the width / height go (theme.py renders at the size it needs)
  return m[0].replace(/ width="[^"]*"/, "").replace(/ height="[^"]*"/, "").replace(/ aria-hidden="true"/, "")
    .replace(/ style="[^"]*"/, "") + "\n";
};

for (const [name, comp] of Object.entries(icons)) {
  if (!name.startsWith("Icon") || typeof comp !== "function") continue;
  const file = name.slice(4).replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
  files.set(`${file}.svg`, svgOf(renderToStaticMarkup(React.createElement(comp, { color: "currentColor", size: 16 }))));
}

const ids = new Set();
for (const tree of [join(root, "templates", "_categories.json"), join(root, "menu", "categories.json")]) {
  const rows = JSON.parse(readFileSync(tree, "utf8"));
  for (const [id, row] of Object.entries(rows)) if (!row.parent) ids.add(id);
}
const dot = svgOf(renderToStaticMarkup(React.createElement(CategoryGlyph, { category: "\u0000none", color: "currentColor" })));
for (const id of [...ids].sort()) {
  const svg = svgOf(renderToStaticMarkup(React.createElement(CategoryGlyph, { category: id, color: "currentColor" })));
  if (svg !== dot) files.set(`category-${id}.svg`, svg);
}
await vite.close();

let stale = [];
if (check) {
  for (const [name, text] of files) {
    const at = join(out, name);
    if (!existsSync(at) || readFileSync(at, "utf8") !== text) stale.push(name);
  }
  if (existsSync(out)) stale.push(...readdirSync(out).filter((n) => n.endsWith(".svg") && !files.has(n)));
  console.log(stale.length ? `icons not up to date (${stale.join(", ")}): run node tools/dcc_icons.mjs` : "up to date");
  process.exit(stale.length ? 1 : 0);
}
mkdirSync(out, { recursive: true });
for (const n of readdirSync(out)) if (n.endsWith(".svg") && !files.has(n)) stale.push(n);
for (const [name, text] of files) writeFileSync(join(out, name), text);
if (stale.length) console.log("no longer made (left in place, remove them):", stale.join(", "));
console.log(`wrote ${files.size} icons to ${out}`);
