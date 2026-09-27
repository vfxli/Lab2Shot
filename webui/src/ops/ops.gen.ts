// 由 tools/gen_ops_ts.py 从 lab2shot/ops/ops.toml 生成，**不要手改**。
// 算法目录只有一个源（需求 6：算法只写一次）；改算法改那个 toml，然后跑：
//   uv run python tools/gen_ops_ts.py
export const OPS = {
  "image.merge.multiply": {"kind": "pixel", "expr": "a * b", "desc": "留下：只保留遮罩里的那块，其余变黑"},
  "image.merge.stencil": {"kind": "pixel", "expr": "a * (1 - b)", "desc": "挡掉：把遮罩里的那块抹黑（Nuke 的 stencil），比如解相机前挡掉走动的人"},
  "image.merge.plus": {"kind": "pixel", "expr": "a + b", "desc": "相加：逐像素相加，不看 alpha"},
  "image.merge.minus": {"kind": "pixel", "expr": "a - b", "desc": "相减：逐像素相减，不看 alpha"},
  "image.merge.min": {"kind": "pixel", "expr": "min(a, b)", "desc": "取小：逐像素取较小的那个"},
  "image.merge.max": {"kind": "pixel", "expr": "max(a, b)", "desc": "取大：逐像素取较大的那个"},
  "boxes.to_mask": {"kind": "boxes.paint", "axis": "clamp(min(p + 1, hi) - max(p, lo), 0, 1)", "combine": "x * y", "accumulate": "max", "base": 0.0, "desc": "人物框 → 框里是 1 的遮罩（原画面大小），框的边按覆盖率抗锯齿"},
  "items.take_one": {"kind": "items.pick", "rules": ["index", "name"], "count": "one", "desc": "从一条列表里取出一条：按序号（从 1 数）或者按名字"},
  "people.select": {"kind": "items.pick", "rules": ["first", "top", "all", "ids", "at_point"], "count": "any", "desc": "从人物框里挑人：最显眼的一个（第一条）、最显眼的前几个、全部、按编号、在 2D 视图里点到的"},
} as const;

export type OpId = keyof typeof OPS;
