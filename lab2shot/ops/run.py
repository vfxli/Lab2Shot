"""算法目录的服务器端执行器（numpy），按 `ops.toml` 的描述执行。

浏览器端执行器 `webui/src/ops/run.ts` 执行同一份描述。两端结果以 `vocab.DISPLAY_TOLERANCE` 为一致性判据
（像素最大差 ≤ 1/255；视图仅用于预览，不要求逐位一致）。

执行器不产生用户消息：未选中条目、绘制结果为空等情况原样返回给节点，由节点决定是否提示及提示内容
（消息由消息目录按编号统一管理）。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from .vocab import RULE_ARGS, BadOp, check, parse, variables

_TOML = Path(__file__).with_name("ops.toml")


def _load() -> dict[str, dict]:
    catalog = tomllib.loads(_TOML.read_text(encoding="utf-8"))["ops"]
    for op_id, desc in catalog.items():
        check(op_id, desc)  # 目录本身有误时在加载阶段即报错
    return catalog


CATALOG: dict[str, dict] = _load()


def describe(op_id: str) -> dict:
    if op_id not in CATALOG:
        raise BadOp(f"算法 {op_id!r} 不在目录里（{_TOML}）")
    return CATALOG[op_id]


# ------------------------------------------------------------------ 公式（逐像素算术，亦供「按框涂」使用）


def _eval(tree: tuple, values: dict[str, Any]) -> Any:
    kind = tree[0]
    if kind == "num":
        return np.float32(tree[1])
    if kind == "var":
        if tree[1] not in values:
            raise BadOp(f"公式用到变量 {tree[1]!r}，但调用时没给（给了 {sorted(values)}）")
        return values[tree[1]]
    if kind == "neg":
        return -_eval(tree[1], values)
    if kind == "bin":
        a, b = _eval(tree[2], values), _eval(tree[3], values)
        op = tree[1]
        return a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
    name, args = tree[1], [_eval(a, values) for a in tree[2]]
    if name == "min":
        return np.minimum(*args)
    if name == "max":
        return np.maximum(*args)
    return np.clip(args[0], args[1], args[2])


def _pixel(desc: dict, args: dict) -> dict:
    tree = parse(desc["expr"])
    values = {}
    for name in variables(tree):
        if name not in args:
            raise BadOp(f"算法要 {name!r}，调用时没给")
        v = args[name]
        values[name] = float(v) if np.isscalar(v) else np.asarray(v, np.float32)
    # 通道数较少的一方由 numpy 广播扩展到通道数较多的一方（例如单通道遮罩作用于三通道图像）。
    # (h, w, 1) 与 (h, w, 3) 的广播要求最后一维为 1，端口类型保证 b 为单通道。
    return {"value": np.asarray(_eval(tree, values), np.float32)}


# ------------------------------------------------------------------ 按框涂


def _boxes_paint(desc: dict, args: dict) -> dict:
    axis, combine = parse(desc["axis"]), parse(desc["combine"])
    w, h = int(args["width"]), int(args["height"])
    cols, rows = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    out = np.full((h, w), np.float32(desc["base"]), np.float32)
    for box in args["boxes"]:
        x1, y1, x2, y2 = (float(v) for v in box)
        cx = _eval(axis, {"p": cols, "lo": np.float32(x1), "hi": np.float32(x2)})
        cy = _eval(axis, {"p": rows, "lo": np.float32(y1), "hi": np.float32(y2)})
        one = _eval(combine, {"x": np.asarray(cx)[None, :], "y": np.asarray(cy)[:, None]})
        out = np.maximum(out, one)  # desc["accumulate"] == "max"（词汇表目前仅支持此一种）
    return {"mask": np.asarray(out, np.float32)}


# ------------------------------------------------------------------ 挑条目


def _box_at(boxes: dict, frame: int) -> list[float] | None:
    """返回该帧的框；该帧无框时使用最近一帧的框（帧号按数值比较，距离相同时取较小的帧号）。"""
    if not boxes:
        return None
    key = str(frame)
    if key not in boxes:
        best = None
        for k in sorted(boxes, key=lambda s: int(s)):
            d = abs(int(k) - frame)
            if best is None or d < best[0]:
                best = (d, k)
        key = best[1]
    return list(boxes[key])


def _items_pick(desc: dict, args: dict) -> dict:
    rule = args["rule"]
    if rule not in desc["rules"]:
        raise BadOp(f"这条算法只认 {desc['rules']}，给的是 {rule!r}")
    for need in RULE_ARGS[rule]:
        if need not in args:
            raise BadOp(f"规则 {rule!r} 要 {need!r}，调用时没给")
    items: list[dict] = list(args["items"])
    picked: list[int] = []
    missed: list[dict] = []

    if rule == "index":
        n = int(args["index"])
        if 1 <= n <= len(items):
            picked = [n - 1]
        else:
            missed = [{"rule": "index", "index": n, "count": len(items)}]
    elif rule == "name":
        want = str(args["name"]).strip()
        found = next((i for i, it in enumerate(items) if it.get("name") == want), None)
        if found is None:
            missed = [{"rule": "name", "name": want}]
        else:
            picked = [found]
    elif rule == "first":
        picked = [0] if items else []
    elif rule == "top":
        picked = list(range(min(max(int(args["count"]), 0), len(items))))
    elif rule == "all":
        picked = list(range(len(items)))
    elif rule == "ids":
        wanted = {int(v) for v in args["ids"]}
        picked = [i for i, it in enumerate(items) if it.get("id") in wanted]
    else:  # at_point：2D 视图中点选的条目。多个框重叠时取面积最小者（即前景中的对象）
        keep: set[int] = set()
        for pick in args["picks"]:
            frame, x, y = int(pick[0]), float(pick[1]), float(pick[2])
            hits = []
            for i, it in enumerate(items):
                box = _box_at(it.get("boxes") or {}, frame)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    hits.append(((x2 - x1) * (y2 - y1), it.get("id", i), i))
            if hits:
                keep.add(min(hits)[2])
            else:
                missed.append({"rule": "at_point", "frame": frame, "x": x, "y": y})
        picked = sorted(keep)  # 结果按条目表顺序返回，与点选顺序无关

    if desc["count"] == "one":
        picked = picked[:1]
    return {"indices": picked, "missed": missed}


_KIND_RUNNERS = {"pixel": _pixel, "boxes.paint": _boxes_paint, "items.pick": _items_pick}


def run(op_id: str, args: dict) -> dict:
    """执行一条算法。`op_id` 为 `ops.toml` 中的 id，`args` 为该运算所需的输入。"""
    desc = describe(op_id)
    return _KIND_RUNNERS[desc["kind"]](desc, args)
