#!/usr/bin/env node
/* 整理节点图（给模板用）：和网页画布右下角「整理节点图」同一份排版（webui/src/graph/layout.ts，经 vite 的 ssrLoadModule
 * 在 node 里载入），原地改写节点图 json 的 nodes[].ui.x / ui.y 与 boxes，
 * 其余字段一字不动（数字按原文保留：JSON.parse 的 source + JSON.rawJSON，缩进同 lab2shot/library.py 写模板的 indent=1）。
 * 对 templates/ 运行会改写预设模板，运行前请保存原文件。
 *
 * 用法（在仓库根目录）：
 *   1) 导出节点定义（节点尺寸按定义估，「输出」类放最右）：
 *      PYTHONPATH=$PWD:$PWD/worker_sdk .venv/bin/python -c "import json, lab2shot.catalog as c; c.install(); \
 *        from lab2shot.nodes import node_types; json.dump({k: t.describe() for k, t in node_types().items()}, \
 *        open('/tmp/lab2shot_defs.json', 'w'), ensure_ascii=False, default=str)"
 *   2) node tools/layout_graph.mjs --defs /tmp/lab2shot_defs.json templates/a.json templates/b.json …
 *      加 --check：不写文件，只报告每张图 整理前 / 后的交叉数、有没有重叠、连线是否都从左到右、再整理一次是否不变。
 *      加 --strict：功能区排版的交叉比不分区的多时用不分区的；并且交叉比整理前多算不通过。
 *      不加时：交叉变多只报告（功能区把支线拉开，跨区长线常多交叉一两处），不通过只看重叠、向左的线、是否稳定。
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const webui = join(root, "webui");
const args = process.argv.slice(2);
const defsAt = args.indexOf("--defs");
if (defsAt < 0 || !args[defsAt + 1]) {
  console.error("要 --defs <节点定义 json>（导出方法见本文件开头）");
  process.exit(2);
}
const defs = JSON.parse(readFileSync(args[defsAt + 1], "utf8"));
const check = args.includes("--check");
const strict = args.includes("--strict");
const files = args.filter((a, i) => a !== "--check" && a !== "--strict" && a !== "--defs" && args[i - 1] !== "--defs");

const vitePath = createRequire(join(webui, "package.json")).resolve("vite");
const { createServer } = await import(pathToFileURL(vitePath).href);
const vite = await createServer({ root: webui, configFile: false, server: { middlewareMode: true, hmr: false }, appType: "custom", logLevel: "error" });
const L = await vite.ssrLoadModule("/src/graph/layout.ts");

const readJson = (text) => JSON.parse(text, (_k, v, ctx) => (typeof v === "number" ? JSON.rawJSON(ctx.source) : v));
const num = (v) => Number(v && typeof v === "object" && "rawJSON" in v ? v.rawJSON : v); // 按原文读进来的数字（JSON.rawJSON）

/** 一张图的排版输入：节点尺寸按定义估，框按整理前「中心在框内」（折叠的按名单）算成员。 */
function inputOf(g) {
  const nodes = (g.nodes ?? []).map((n) => {
    const def = defs[n.type];
    const { w, h } = L.estimateSize(def, n);
    return { id: n.id, x: num(n.ui?.x ?? 0), y: num(n.ui?.y ?? 0), w, h, delivers: !!def?.delivers };
  });
  const edges = (g.edges ?? []).map((e) => ({ from: e.from[0], to: e.to[0], param: String(e.to[1]).startsWith("param:") }));
  const blocks = L.eachBlocks((g.nodes ?? []).map((n) => ({ id: n.id, type: n.type, block: n.params?.block })));
  const boxes = (g.boxes ?? []).map((b) => {
    const [x, y, w, h] = [num(b.x), num(b.y), num(b.w), num(b.h)];
    const members = b.collapsed ? b.members ?? [] : nodes.filter((n) => {
      const cx = n.x + n.w / 2, cy = n.y + n.h / 2;
      return cx > x && cx < x + w && cy > y + L.LAYOUT.boxHead && cy < y + h;
    }).map((n) => n.id);
    return { id: b.id, x, y, w, h, members };
  });
  return { nodes, edges, blocks, boxes };
}

let bad = 0;
for (const file of files) {
  const text = readFileSync(file, "utf8");
  const g = readJson(text);
  if (g?.schema !== "lab2shot.graph/1") {
    console.log(`跳过  ${file.split("/").pop()}：不是节点图（schema 不是 lab2shot.graph/1）`);
    continue;
  }
  const input = inputOf(g);
  const out = L.layoutGraph({ ...input, strict });
  const before = L.countCrossings(Object.fromEntries(input.nodes.map((n) => [n.id, { x: n.x, y: n.y }])), input.nodes, input.edges);
  const after = L.countCrossings(out.positions, input.nodes, input.edges);
  const overlap = L.overlaps(out.positions, input.nodes);
  const size = new Map(input.nodes.map((n) => [n.id, n]));
  const backwards = input.edges.filter((e) => out.positions[e.from].x + size.get(e.from).w > out.positions[e.to].x);
  const again = L.layoutGraph({ ...input, strict, nodes: input.nodes.map((n) => ({ ...n, ...out.positions[n.id] })) });
  const stable = Object.keys(out.positions).every((id) => again.positions[id].x === out.positions[id].x && again.positions[id].y === out.positions[id].y);
  const ok = !overlap.length && !backwards.length && (!strict || after <= before) && stable;
  if (!ok) bad++;
  console.log(`${ok ? "通过" : "不通过"}  ${file.split("/").pop()}：${input.nodes.length} 个节点 ${input.edges.length} 根线，交叉 ${before} → ${after}${after > before ? "（多了）" : ""}，重叠 ${overlap.length}，向左的线 ${backwards.length}，再整理一次${stable ? "不变" : "变了"}`);
  if (check) continue;
  for (const n of g.nodes ?? []) {
    const p = out.positions[n.id];
    n.ui = { ...(n.ui ?? {}), x: p.x, y: p.y };
  }
  for (const b of g.boxes ?? []) Object.assign(b, out.boxes[b.id] ?? {});
  writeFileSync(file, JSON.stringify(g, null, 1) + (text.endsWith("\n") ? "\n" : ""));
}
console.log(`共 ${files.length} 张，不通过 ${bad} 张`);
await vite.close().catch(() => {});
process.exit(bad ? 1 : 0);
