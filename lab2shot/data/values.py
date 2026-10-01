"""Basic values: 浮点, 整数, 布尔, 向量, 文字. A focal length and a film back are both 浮点 in mm, told apart only by the
name of the port they come out of; the user wires whichever they mean.

A value is one value, or one value per frame with its frame numbers (a per-frame 浮点 is a curve; 曲线 is several named
ones, and 「曲线取值」 picks one out). It lives in its packet's description (meta), nothing on disk:

    one value        {"value": 35.0, "unit": "mm"}
    one per frame    {"frames": [1001, 1002], "values": [35.0, 35.2], "unit": "mm"}

and, when the value belongs to a picture (a focal length measured on a plate), that picture's "width" and "height".
The unit is part of the data: a focal length in mm and one in px never mix silently (a length converts, px to mm does
not). Any value parameter of any node can be driven by a wire of its type (NodeDef.param_port, the engine's
promoted parameters); this module says what a wired value is worth to a parameter, one place for every node.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..errors import Invalid
from ..messages import Msg
from .units import UNIT_KINDS, UNITS

if TYPE_CHECKING:
    from .packet import Packet

FLOAT, INT, BOOL, VECTOR, TEXT = "value.float", "value.int", "value.bool", "value.vector", "value.text"
# 镜头内参：{"model": 镜头模型 id, "params": {系数名: 数}}（data/types.py 那条说明；Focal Length 和 Filmback 不在里面）
LENS = "value.lens"

# how close numbers of a per-frame value must be to count as the same all along (a camera read back from USD)
SAME = 1e-6


def is_value(data_type: str) -> bool:
    """One basic value; a list of them is not one (it holds no value of its own: data/packet.py)."""
    from .types import is_list

    return not is_list(data_type) and (data_type == "value" or data_type.startswith("value."))


def unit_problem(have: str, want: str) -> Msg | None:
    """Why a value in unit `have` can't go where `want` is meant (None it can): the same unit, or one of the same kind
    (converted), or either without a unit (a plain number takes the meaning of where it goes)."""
    if not have or not want or have == want:
        return None
    a, b = UNITS.get(have), UNITS.get(want)
    if a and b and a[0] == b[0]:
        return None
    kind = lambda u: UNIT_KINDS[UNITS[u][0]] if u in UNITS else u  # noqa: E731
    return Msg("B-VALUES-UNIT", have=have, want=want, have_kind=kind(have), want_kind=kind(want))


def factor(have: str, want: str) -> float:
    """What a number in `have` is multiplied by to be in `want` (1 when either has no unit)."""
    if not have or not want or have == want:
        return 1.0
    return UNITS[have][1] / UNITS[want][1]


# ------------------------------------------------------------------ a value


@dataclass(frozen=True)
class Value:
    """A value as a node reads it: its type and unit, and one value or one per frame."""

    type: str
    unit: str = ""
    value: Any = None  # one value
    frames: tuple[int, ...] = ()  # one per frame: their frame numbers ...
    values: tuple = ()  # ... and the values

    @property
    def per_frame(self) -> bool:
        return bool(self.frames)

    @property
    def numeric(self) -> bool:
        return self.type in (FLOAT, INT, VECTOR)

    def constant(self) -> bool:
        """The same value on every frame (one value always is)."""
        if not self.per_frame:
            return True
        if self.numeric:
            a = np.asarray(self.values, np.float64)
            return bool(np.allclose(a, a[:1], rtol=SAME, atol=SAME))
        return all(v == self.values[0] for v in self.values)

    def one(self) -> Any:
        """The value that stands for the whole shot: the one value; of a per-frame one its median (numbers, a vector
        per axis), else its first."""
        if not self.per_frame:
            return self.value
        if self.type in (FLOAT, VECTOR):
            m = np.median(np.asarray(self.values, np.float64), axis=0)
            return tuple(float(x) for x in m) if self.type == VECTOR else float(m)
        if self.type == INT:
            return int(np.median(np.asarray(self.values)))
        return self.values[0]

    def at(self, frames) -> np.ndarray:
        """Numbers at `frames`: one value everywhere; one per frame interpolated between its frames, held beyond them
        (as USD holds a camera). A vector gives [F,3]."""
        frames = np.asarray(list(frames), np.float64)
        if not self.per_frame:
            return np.broadcast_to(np.asarray(self.value, np.float64), (len(frames),) + np.shape(self.value)).copy()
        known = np.asarray(self.frames, np.float64)
        v = np.asarray(self.values, np.float64)
        if v.ndim == 1:
            return np.interp(frames, known, v)
        return np.stack([np.interp(frames, known, v[:, k]) for k in range(v.shape[1])], -1)

    def span(self) -> tuple[Any, Any]:
        """The smallest and largest value (numbers; a vector by its length)."""
        a = np.asarray(self.values if self.per_frame else [self.value], np.float64)
        if a.ndim > 1:
            a = np.linalg.norm(a, axis=-1)
        return float(a.min()), float(a.max())

    def in_unit(self, unit: str) -> Value:
        """The same value in another unit of its kind (unit_problem says whether it can be)."""
        k = factor(self.unit, unit)
        if k == 1.0 or not self.numeric:
            return Value(self.type, unit or self.unit, self.value, self.frames, self.values)
        scale = lambda x: _typed(self.type, np.asarray(x, np.float64) * k)  # noqa: E731
        return Value(self.type, unit, None if self.per_frame else scale(self.value), self.frames,
                     tuple(scale(v) for v in self.values))


def _typed(type_: str, x) -> Any:
    if type_ == VECTOR:
        return tuple(float(c) for c in np.asarray(x).reshape(3))
    if type_ == INT:
        return int(round(float(x)))
    return float(x)


def read(p: Packet) -> Value:
    return read_meta(p.type, p.meta)


def read_meta(type_: str, m: dict) -> Value:
    if "frames" in m and "values" in m:
        return Value(type_, m.get("unit", ""), None, tuple(int(f) for f in m["frames"]), tuple(_plain(type_, v) for v in m["values"]))
    return Value(type_, m.get("unit", ""), _plain(type_, m["value"]))


def _plain(type_: str, v: Any) -> Any:
    return tuple(v) if type_ == VECTOR else v


def value_meta(type_: str, value: Any = None, *, frames=None, values=None, unit: str = "", **meta) -> dict:
    """A value packet's description: one `value`, or `values` at `frames` (a frame-carrying
    packet always has one, the contract checks it)."""
    if frames is not None:
        body = {"frames": [int(f) for f in frames], "values": [_json(type_, v) for v in values]}
    else:
        body = {"value": _json(type_, value)}
    return {**body, "unit": unit, **meta}


def value_packet(directory: Path, type_: str, value: Any = None, **kw) -> Packet:
    """A value packet (value_meta's arguments)."""
    from .packet import Packet

    return Packet(directory, type_, value_meta(type_, value, **kw))


def list_values(p: Packet, load=None) -> list[Value]:
    """The values a list packet holds, in order: its items are ordinary value packets, each cached on its own.
    `load`: how a fingerprint becomes a packet (the engine hands its own manifest cache; the default reads the cache).
    One place, so the engine, the benchmark and any node read a 浮点列表 the same way."""
    from .packet import Packet as _P
    from .packet import items_of, packet_dir

    load = load or (lambda fp: _P.load(packet_dir(fp)))
    return [read(item) for _, fp in items_of(p) if (item := load(fp)) is not None]


def value_list_packet(directory: Path, type_: str, values: list[tuple[str, Any]], node: str, **kw) -> Packet:
    """A list of basic values a node gives out (AnyCalib's 「畸变系数」, its items named k1, k2 … as the model calls
    them): one ordinary value packet per item, each cached on its own, the list naming them in order. What is on the
    wire is exactly what 「拆成列表」 and 「取一条」 already take apart — no type of its own for a handful of numbers."""
    from .packet import Packet, item_fingerprint, items_meta, produce
    from .types import list_of

    parts = []
    for name, v in values:
        fp = item_fingerprint(node, type_, name, _json(type_, v), sorted(kw.items()))
        produce(fp, lambda d, v=v: value_packet(d, type_, v, **kw).commit(node))  # under the entry's lock
        parts.append((name, fp))
    return Packet(directory, list_of(type_), items_meta(parts))


def _json(type_: str, v: Any) -> Any:
    if type_ == VECTOR:
        return [float(c) for c in v]
    if type_ == LENS:  # 组和公式表 id 跟着走（nodes/lens.py packed_lens）
        return {"group": str(v["group"]), "model": str(v["model"]), "table": str(v["table"]),
                "params": {str(k): float(x) for k, x in dict(v["params"]).items()}}
    if type_ == FLOAT:
        return float(v)
    if type_ == INT:
        return int(v)
    if type_ == BOOL:
        return bool(v)
    return str(v)


# ------------------------------------------------------------------ the contract


def _problem(type_: str, v: Any) -> Msg | None:
    """What is wrong with one value of the type (None nothing)."""
    number = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)  # noqa: E731
    if type_ == FLOAT and not number(v):
        return Msg("E-VALUES-NOTFLOAT", value=repr(v))
    if type_ == INT and not (isinstance(v, int) and not isinstance(v, bool)):
        return Msg("E-VALUES-NOTINT", value=repr(v))
    if type_ == BOOL and not isinstance(v, bool):
        return Msg("E-VALUES-NOTBOOL", value=repr(v))
    if type_ == VECTOR and not (isinstance(v, (list, tuple)) and len(v) == 3 and all(number(c) for c in v)):
        return Msg("E-VALUES-NOTVECTOR", value=repr(v))
    if type_ == TEXT and not isinstance(v, str):
        return Msg("E-VALUES-NOTTEXT", value=repr(v))
    if type_ == LENS and not (isinstance(v, dict) and all(isinstance(v.get(k), str) and v[k] for k in ("group", "model", "table"))
                              and isinstance(v.get("params"), dict) and all(isinstance(k, str) and number(x) for k, x in v["params"].items())):
        return Msg("E-VALUES-NOTLENS", value=repr(v)[:80])
    return None


def check(p: Packet) -> list[Msg]:
    """A value packet keeps its contract: one value or one per frame (frame numbers rising, one value each), every
    value of its type and finite, a unit Lab2Shot knows (or none)."""
    m = p.meta
    problems = []
    if m.get("unit", "") and m["unit"] not in UNITS:
        problems.append(Msg("E-VALUES-UNKNOWNUNIT", unit=repr(m["unit"]), units=" / ".join(UNITS)))
    if "frames" in m or "values" in m:
        frames, values = m.get("frames"), m.get("values")
        if not isinstance(frames, list) or not isinstance(values, list):
            return problems + [Msg("E-VALUES-PERFRAMELISTS")]
        if frames != sorted({int(f) for f in frames}) or not frames:
            problems.append(Msg("E-CONTRACT-FRAMEORDER"))
        if len(values) != len(frames):
            problems.append(Msg("E-VALUES-COUNTS", frames=len(frames), values=len(values)))
        bad = next((Msg("E-VALUES-ONFRAME", frame=f, reason=why) for f, v in zip(frames, values) if (why := _problem(p.type, v))), None)
    elif "value" in m:
        bad = _problem(p.type, m["value"])
    else:
        return problems + [Msg("E-VALUES-NOVALUE")]
    return problems + ([bad] if bad else [])


# ------------------------------------------------------------------ as people read it


def _num(x: float) -> str:
    """A number as a person reads it: whole numbers as they are, others to four significant digits."""
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    digits = max(0, 3 - math.floor(math.log10(abs(x))))
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _with_unit(text: str, unit: str) -> str:
    return f"{text}{unit}" if unit == "°" else f"{text} {unit}" if unit else text


def _one(type_: str, v: Any) -> str:
    if type_ == BOOL:
        return "开" if v else "关"
    if type_ == TEXT:
        return f"「{v}」"
    if type_ == LENS:  # 「AnyCalib · simple_kb:4」k1 -0.012 k2 0.003：组 · 模型按它自己的名字，系数按模型自己的顺序
        from ..nodes.lens import group_label

        who = group_label(v["group"])
        return f"「{who} · {v.get('model', '')}」" + "".join(f" {k} {_num(x)}" for k, x in dict(v.get("params") or {}).items() if k not in ("center_x_mm", "center_y_mm", "pixel_aspect"))
    if type_ == VECTOR:
        return "(" + ", ".join(_num(c) for c in v) + ")"
    return _num(v)


def option_label(spec: dict, v: Any) -> str:
    """The word a choice is called by (「OpenCV 鱼眼」), "" when `v` is not one of this parameter's options. Whoever
    shows a value of a choice parameter — typed on the node or brought by a wire — says it with this, never with the
    id the wire carries（同一个东西处处叫同一个词）."""
    if not spec["options"]:
        return ""
    label = (spec.get("option_labels") or {}).get(str(v))
    return f"「{label}」" if label else ""


def say(spec: dict, v: Any) -> str:
    """A parameter's own (typed) value as the node says it: "35 mm", a choice by its own word."""
    if label := option_label(spec, v):
        return label
    kind = {"number": FLOAT, "integer": INT, "boolean": BOOL}.get(spec["type"], VECTOR if spec["widget"] == "vec3" else TEXT)
    return _with_unit(_one(kind, v), spec["unit"])


def describe(p: Packet) -> str:
    """The value as the node, the viewer and a wired parameter show it: "38.6 mm", "34.2–36.9 mm（逐帧）". A value that
    stands for a choice says the word that choice is called by (「OpenCV 鱼眼」, not the id 「opencv_fisheye」 the wire
    carries): whoever made it puts that word in its description as `said`, and everywhere it is shown reads it from
    there — 同一个东西处处叫同一个词，不在每个界面里各翻一遍."""
    return str(p.meta["said"]) if p.meta.get("said") else describe_value(read(p))


def describe_value(v: Value) -> str:
    if not v.per_frame:
        return _with_unit(_one(v.type, v.value), v.unit)
    if v.constant():
        return _with_unit(_one(v.type, v.values[0]), v.unit) + "（逐帧，不变）"
    if v.type in (FLOAT, INT):
        lo, hi = v.span()
        return _with_unit(f"{_num(lo)}–{_num(hi)}", v.unit) + "（逐帧）"
    return f"{len(v.frames)} 帧，每帧不同"


# ------------------------------------------------------------------ what a wired value is worth to a parameter


def param_type(spec: dict) -> str:
    """The value type a parameter takes as a wire ("" when it can't be driven by one): numbers, switches, three-number
    vectors, free text — and a choice, which takes 文字 (so a node such as AnyCalib can hand a model type on by
    wire). The receiving end checks the text is one of its own options, which says more
    than a type ever could（「上游给的是 radial:2，这里可选的是 无畸变 / 3DE Classic / …」）. Files, lists, names and
    pickers are set in the panel only; an output-settings node's 名字 takes 文字 too."""
    from .types import list_of

    if spec["items"] is not None:  # a table: one list per column is meaningless, one column is a plain list of numbers
        fields = [f for f in spec["items"] if f["widget"] != "fixed" and f["panel"]]
        kind = {"number": FLOAT, "integer": INT}.get(fields[0]["type"]) if len(fields) == 1 else None
        return list_of(kind) if kind and not fields[0]["options"] else ""
    if spec["options"]:
        return "" if spec["choices_from"] else TEXT
    # `unique` only numbers the value a node added in the editor gets (nodes/params.py); a wire may still drive it
    # (an output-settings node's 名字 from a 「文字」 node: several nodes of a card named from one place)
    if spec["choices_from"] or spec["derived_from"]:
        return ""
    widget = spec["widget"]
    if widget == "vec3":
        return VECTOR
    if widget not in (None, "slider"):
        return ""
    return {"number": f"{FLOAT}|{INT}", "integer": INT, "boolean": BOOL, "string": TEXT}.get(spec["type"], "")


def rows_for_param(spec: dict, rows: list[dict], values: list[Value], where: str) -> list[dict]:
    """What a table parameter driven by a wire gets: `rows` (the rows it has — for a table whose rows follow another
    parameter, the ones NodeDef.derive() gives for that parameter's current value) with each row's number taken from
    the list, in order. The row names stay the node's own: they are what its own model calls them, and saying
    「上游给了 3 个，这里的 OpenCV Brown 要 5 个：k1 k2 p1 p2 k3」 is worth more than any type check could be."""
    field = next(f for f in spec["items"] if f["widget"] != "fixed" and f["panel"])
    if len(values) != len(rows):
        raise Invalid(Msg("E-VALUES-WIREDCOUNT", name=spec["label"], where=where, got=len(values), want=len(rows),
                          names=" ".join(str(r.get("name", "")) for r in rows) or "（无）"))
    out = []
    for row, v in zip(rows, values, strict=True):
        one, _ = for_param(field, v, where)
        out.append({**row, field["name"]: one})
    return out


def for_param(spec: dict, v: Value, where: str) -> tuple[Any, Value]:
    """What a parameter gets from a wired value: the one value in the parameter's unit, as the parameter's type, and the
    value itself (in that unit) for a parameter that takes one per frame (spec per_frame). A per-frame value that
    changes over the frames, into a parameter that takes one value for the whole shot, is refused (ValueError saying
    what to do), as is a unit that can't be converted. `where`: how the message names the wire."""
    label = spec["label"]
    why = unit_problem(v.unit, spec["unit"])
    if why:
        raise Invalid(Msg("E-VALUES-WIREDUNIT", name=label, where=where, reason=why))
    v = v.in_unit(spec["unit"]) if spec["unit"] else v
    if not v.constant() and not spec["per_frame"]:
        lo, hi = v.span()
        if v.type in (FLOAT, INT):
            raise Invalid(Msg("E-VALUES-CHANGESSPREAD", name=label, where=where, low=_num(lo), high=_with_unit(_num(hi), v.unit)))
        raise Invalid(Msg("E-VALUES-CHANGES", name=label, where=where))
    one = v.one()
    kind = spec["type"]
    if kind == "number":
        one = float(one)
    elif kind == "integer":
        one = int(one)
    elif spec["widget"] == "vec3":
        one = tuple(float(c) for c in one)
    return one, v
