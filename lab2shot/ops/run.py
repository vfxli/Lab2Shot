"""算法目录的执行器（numpy），按 `ops.toml` 的描述执行。

执行器不产生用户消息：未选中条目、绘制结果为空等情况原样返回给节点，由节点决定是否提示及提示内容
（消息由消息目录按编号统一管理）。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _tree(expr: str) -> tuple:
    """公式的语法树。目录中的公式是固定的几条，每条只解析一次：逐帧调用的节点（「人物框转遮罩」「图像合成」）
    不必每帧、每个框重新解析。语法树由元组组成，不可修改，多个线程共用同一棵树是安全的。"""
    return parse(expr)


@lru_cache(maxsize=None)
def _names(expr: str) -> frozenset[str]:
    return frozenset(variables(_tree(expr)))



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
    tree = _tree(desc["expr"])
    values = {}
    for name in _names(desc["expr"]):
        if name not in args:
            raise BadOp(f"算法要 {name!r}，调用时没给")
        v = args[name]
        values[name] = float(v) if np.isscalar(v) else np.asarray(v, np.float32)
    # 通道数较少的一方由 numpy 广播扩展到通道数较多的一方（例如单通道遮罩作用于三通道图像）。
    # (h, w, 1) 与 (h, w, 3) 的广播要求最后一维为 1，端口类型保证 b 为单通道。
    return {"value": np.asarray(_eval(tree, values), np.float32)}


# ------------------------------------------------------------------ 按框涂


def _boxes_paint(desc: dict, args: dict) -> dict:
    axis, combine = _tree(desc["axis"]), _tree(desc["combine"])
    w, h = int(args["width"]), int(args["height"])
    cols, rows = np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    base = np.float32(desc["base"])
    out = np.full((h, w), base, np.float32)
    boxes = np.asarray([[float(v) for v in box] for box in args["boxes"]], np.float64).reshape(-1, 4)
    if not len(boxes):
        return {"mask": out}
    # 所有框一起算两条轴上的覆盖（每个框一行，逐元素运算，数值与逐框计算相同），不按框逐个调用：
    # 一帧十几个人时，逐框的几十次小运算比运算本身还费时，多帧并行时更甚（每次运算都要交还再取回解释器锁）
    x1, y1, x2, y2 = (boxes[:, [i]].astype(np.float32) for i in range(4))
    cx = np.broadcast_to(_eval(axis, {"p": cols[None, :], "lo": x1, "hi": x2}), (len(boxes), w))
    cy = np.broadcast_to(_eval(axis, {"p": rows[None, :], "lo": y1, "hi": y2}), (len(boxes), h))
    # 只计算每个框所覆盖的那一块：一个框只占画面的一小块，块外各像素沿某条轴的覆盖为 0。只要这些像素的合成值
    # 不超过底色（逐行、逐列核对；公式里任何一处不满足，该框就退回整幅计算），取较大值后它们保持原值，
    # 因此结果与整幅计算逐位相同
    zero = np.float32(0)
    off_x = np.broadcast_to(_eval(combine, {"x": zero, "y": cy}), cy.shape)  # 覆盖为 0 的列上，每一行的合成值
    off_y = np.broadcast_to(_eval(combine, {"x": cx, "y": zero}), cx.shape)  # 覆盖为 0 的行上，每一列的合成值
    blocks = np.all(off_x <= base, axis=1) & np.all(off_y <= base, axis=1)  # NaN 不满足，同样退回整幅计算
    on_x, on_y = cx != 0, cy != 0
    seen = on_x.any(axis=1) & on_y.any(axis=1)  # 框至少覆盖画面内的一个像素
    left, right = on_x.argmax(axis=1), w - on_x[:, ::-1].argmax(axis=1)
    top, bottom = on_y.argmax(axis=1), h - on_y[:, ::-1].argmax(axis=1)
    for i in range(len(boxes)):
        if not blocks[i]:
            one = _eval(combine, {"x": cx[i][None, :], "y": cy[i][:, None]})
            out = np.maximum(out, one)  # desc["accumulate"] == "max"（词汇表目前仅支持此一种）
        elif seen[i]:  # 框完全在画面外（且块外不超过底色）：对结果没有影响
            ys, xs = slice(top[i], bottom[i]), slice(left[i], right[i])
            one = _eval(combine, {"x": cx[i][None, xs], "y": cy[i][ys, None]})
            np.maximum(out[ys, xs], one, out=out[ys, xs])
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
