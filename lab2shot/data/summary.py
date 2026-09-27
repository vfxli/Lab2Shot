"""One summary of a piece of data, for the three places that say what it is.

A data type declares which items its summary has (`DataType.summary`: ids); what each item reads and how it is said
lives here, once. The server makes the summary; the port's tooltip, the middle-click information panel and the
「取信息」 node all read the same answer, and no extension writes a word of it.

    describe(packet)              what a cooked result is: every item its type declares that its meta has
    describe_info(type, info)     what is known before it is cooked (Info: frames, size, frame rate)

It reads a packet's meta and nothing else — no file is opened, so a tooltip is instant. What the summary needs is
therefore in the meta: the packet constructors put it there (data/payloads.py, data/camera.py), which is what the
contract checks.

Every item's Chinese comes from the message catalogue (`I-SUMMARY-<ID>`), never from code here: the reply carries the
value, the code and the text, so the page shows the text and 「取信息」 uses the value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..messages import Msg
from .types import DATA_TYPES


@dataclass(frozen=True)
class Item:
    """One line of a summary: how it is read from a packet's meta, how it is said, and what is already known of it
    before the data is cooked (from the node's Info)."""

    id: str
    code: str  # the message that says it (written out, so the catalogue's guards see it)
    read: Callable[[Mapping], Any]  # (meta) -> the value, or None when this data does not have it
    before: Callable[[Any], Any] | None = None  # (Info) -> the value known before cooking

    def said(self, value: Any) -> Msg:
        params = dict(value) if isinstance(value, dict) else {"value": value}
        code = params.pop("said", None) or self.code  # a value with its own wording (a still's frame range)
        return Msg(code, **params)


def _frames(meta: Mapping) -> Any:
    frames = meta.get("frames")
    if not frames:
        return None
    if meta.get("still"):  # one picture for the whole shot: its own wording
        return {"first": frames[0], "last": frames[-1], "said": "I-SUMMARY-STILL"}
    return {"first": frames[0], "last": frames[-1], "count": len(frames)}


def _size(meta: Mapping) -> Any:
    return {"width": meta["width"], "height": meta["height"]} if meta.get("width") else None


def _data_window(meta: Mapping) -> Any:
    from .windows import Window

    if "data_window" not in meta or not meta.get("width"):
        return None
    window = Window.of(meta)
    if not window.has_overscan:
        return None
    left, top, right, bottom = window.overscan
    return {"left": left, "top": top, "right": right, "bottom": bottom,
            "width": window.canvas[0], "height": window.canvas[1]}


def _channels(meta: Mapping) -> Any:
    from .payloads import channel_names
    from .types import channels_of

    count = channels_of(meta.get("_type", ""))
    if not count:
        return None
    return {"names": " ".join(channel_names(count, bool(meta.get("values")) and count < 4))}


SCALE_SAID = {"metric": "I-SUMMARY-SCALEMETRIC", "relative": "I-SUMMARY-SCALERELATIVE", "affine": "I-SUMMARY-SCALEAFFINE"}
SPACE_SAID = {"world": "I-SUMMARY-SPACEWORLD", "camera": "I-SUMMARY-SPACECAMERA",
              "canonical": "I-SUMMARY-SPACECANONICAL"}
DIRECTION_SAID = {"undistort": "I-SUMMARY-DIRECTIONUNDISTORT", "distort": "I-SUMMARY-DIRECTIONDISTORT"}
PROJECTION_SAID = {"cylinder": "I-SUMMARY-PROJECTIONCYLINDER", "sphere": "I-SUMMARY-PROJECTIONSPHERE",
                   "plane": "I-SUMMARY-PROJECTIONPLANE", "unknown": "I-SUMMARY-PROJECTIONUNKNOWN"}
LEVEL_SAID = {"measured": "I-SUMMARY-LEVELMEASURED", "solved": "I-SUMMARY-LEVELSOLVED",
              "estimated": "I-SUMMARY-LEVELESTIMATED"}


def _pixel_aspect(meta: Mapping) -> Any:
    aspect = meta.get("pixel_aspect")
    return {"value": aspect} if aspect not in (None, 1, 1.0) else None  # 1: nothing to say


def _enum(name: str, codes: Mapping[str, str], said: str = "value"):
    """A meta key whose values are a fixed set (a depth's scale, a position's space): each one has its own message, so
    the summary says 「是（估计值）」 and not 「metric」."""
    def read(meta: Mapping, name=name, codes=codes, said=said):
        value = meta.get(name)
        return None if not value or value not in codes else {said: Msg(codes[value]).text}
    return read


def _key(name: str, said: str = "value"):
    def read(meta: Mapping, name=name, said=said):
        value = meta.get(name)
        return None if value is None else {said: value}
    return read


def _count(name: str):
    def read(meta: Mapping, name=name):
        value = meta.get(name)
        return None if value is None else {"count": len(value) if isinstance(value, (list, tuple, dict)) else value}
    return read


def _names(name: str, most: int = 5):
    def read(meta: Mapping, name=name):
        value = meta.get(name)
        if not value:
            return None
        names = [str(v) for v in (value if isinstance(value, (list, tuple)) else [value])]
        return {"count": len(names), "names": "、".join(names[:most]) + ("…" if len(names) > most else "")}
    return read


def _items(meta: Mapping) -> Any:
    """A list's items: how many, and the first of their names (「图像序列列表 · 5 条」)."""
    got = [str(i.get("name", "")) for i in meta.get("items") or ()]
    return {"count": len(got), "names": "、".join(got[:5]) + ("…" if len(got) > 5 else "")}


def _strands(meta: Mapping) -> Any:
    """三维曲线有多少条、一共多少个点（data/payloads.py _scene_contents 做包时数出来的）。"""
    if not meta.get("strands"):
        return None
    return {"count": meta["strands"], "points": meta.get("curve_points", 0)}


def _subsets(meta: Mapping) -> Any:
    """网格的分区有哪些、各多少个面（「分区：3 个（scalp 489 面、face 2400 面…）」，data/subsets.py 数出来的）。"""
    got = meta.get("subsets")
    if not got:
        return None
    named = [f"{name} {faces} 面" for name, faces in got.items()]
    return {"count": len(named), "names": "、".join(named[:5]) + ("…" if len(named) > 5 else "")}


def _attributes(meta: Mapping) -> Any:
    """点和曲线上带的属性（「属性：3 个（confidence 浮点 · 逐点、normal 法线 · 逐点…）」，
    data/payloads.py point_attributes 做包时数出来的）。"""
    got = meta.get("attributes")
    if not got:
        return None
    named = [f"{a['name']} {a['type']} · {a['per']}" for a in got]
    return {"count": len(named), "names": "、".join(named[:5]) + ("…" if len(named) > 5 else "")}


def _body(meta: Mapping) -> Any:
    """哪个人体模型，几个关节（「SMPL-X · 55 个关节」）。"""
    from lab2shot_shared.smpl import BODIES

    got = BODIES.get(meta.get("body") or "")
    return None if got is None else {"value": got.title, "joints": meta.get("joints") or got.joints}


def _contents(meta: Mapping) -> Any:
    """A scene's kinds and how many of each (written when the scene packet is made)."""
    held = meta.get("contents")
    if not held:
        return None
    from .types import SCENE_KINDS

    return {"kinds": "、".join(f"{SCENE_KINDS[k].label} {n}" for k, n in held.items() if n)}


def _range(meta: Mapping) -> Any:
    got = meta.get("range")
    return {"low": f"{float(got[0]):.4g}", "high": f"{float(got[1]):.4g}"} if got else None


def _focal(meta: Mapping) -> Any:
    """整段一个 Focal Length 就写一个数，变焦才写区间。

    两端一样的区间（「18 – 18 mm」）读起来像出了什么事，所以不变焦时用 `said` 换一条消息
    （Item.said 支持，单帧的帧范围走的就是它）。"""
    got = meta.get("focal_mm")
    if got is None:
        return None
    values = got if isinstance(got, (list, tuple)) else [got]
    low, high = min(values), max(values)
    if low == high:
        return {"low": f"{float(low):.4g}", "said": "I-SUMMARY-FOCALONE"}
    return {"low": f"{float(low):.4g}", "high": f"{float(high):.4g}"}


def _filmback(meta: Mapping) -> Any:
    got = meta.get("filmback_mm")
    if got is None:
        return None
    values = got if isinstance(got, (list, tuple)) else [got, None]
    return {"width": f"{float(values[0]):.4g}"} if values[0] is not None else None


def _distortion(meta: Mapping) -> Any:
    """畸变模型说的是界面上那个名字，不是核心表里的 id（`data/lens_models.py MODELS` 的键）。

    同一个东西处处叫同一个词：「LensDistortion」上、AnyCalib / COLMAP 的「拟合模型」里写的都是表里的 label，
    这里也写那个词。表里没有的 id（别处塞进来的）照原样念，不编。"""
    from .lens_models import MODELS

    model = (meta.get("distortion") or {}).get("model") if isinstance(meta.get("distortion"), dict) else meta.get("distortion")
    if not model:
        return None
    said = MODELS[model].label if model in MODELS else str(model)
    return {"model": said}


def _value(meta: Mapping) -> Any:
    """A basic value (data/values.py): one value, or its range over the frames; with its unit."""
    unit = meta.get("unit") or ""
    if "value" in meta:
        value = meta["value"]
        text = " ".join(f"{float(v):.4g}" for v in value) if isinstance(value, (list, tuple)) else \
            (f"{float(value):.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value))
        return {"value": text, "unit": unit}
    values = meta.get("values")
    if not values:
        return None
    flat = [v for row in values for v in (row if isinstance(row, (list, tuple)) else [row])]
    return {"value": f"{float(min(flat)):.4g}–{float(max(flat)):.4g}", "unit": unit}


POINTS_FROM_SAID = {  # 点云的三维坐标从哪来
    "native": "I-SUMMARY-POINTSFROMNATIVE",
    "depth": "I-SUMMARY-POINTSFROMDEPTH",
}

ITEMS: Mapping[str, Item] = MappingProxyType({item.id: item for item in (
    Item("size", "I-SUMMARY-SIZE", _size, before=lambda i: {"width": i.width, "height": i.height} if i.width else None),
    Item("frames", "I-SUMMARY-FRAMES", _frames, before=lambda i: (({"first": i.frames[0], "last": i.frames[-1], "said": "I-SUMMARY-STILL"}
                                               if i.still else
                                               {"first": i.frames[0], "last": i.frames[-1], "count": len(i.frames)})
                                              if i.frames else None)),
    Item("channels", "I-SUMMARY-CHANNELS", _channels),
    Item("colorspace", "I-SUMMARY-COLORSPACE", _key("colorspace")),
    Item("values", "I-SUMMARY-VALUES", lambda meta: {}),  # a data map: its values are data, never colour-managed
    Item("range", "I-SUMMARY-RANGE", _range),
    Item("pixel_aspect", "I-SUMMARY-PIXELASPECT", _pixel_aspect),
    Item("data_window", "I-SUMMARY-DATAWINDOW", _data_window),
    Item("scale", "I-SUMMARY-SCALE", _enum("scale", SCALE_SAID)),
    Item("space", "I-SUMMARY-SPACE", _enum("space", SPACE_SAID)),
    Item("model", "I-SUMMARY-MODEL", _key("model")),
    Item("classes", "I-SUMMARY-CLASSES", _names("classes")),
    Item("direction", "I-SUMMARY-DIRECTION", _enum("direction", DIRECTION_SAID)),
    Item("projection", "I-SUMMARY-PROJECTION", _enum("projection", PROJECTION_SAID)),
    Item("people", "I-SUMMARY-PEOPLE", _count("people")),
    Item("chosen", "I-SUMMARY-CHOSEN", lambda meta: {} if meta.get("chosen") else None),
    Item("points", "I-SUMMARY-POINTS", _count("names")),
    # 这一片点的三维坐标从哪来。一条常在的说明，不是计算时弹一次的
    # 消息：结果一命中缓存，计算时的消息就没了，而这件事在这份数据上永远成立
    Item("points_from", "I-SUMMARY-POINTSFROM", _enum("points_from", POINTS_FROM_SAID)),
    Item("curves", "I-SUMMARY-CURVES", _names("names")),
    # 三维曲线：条数和总点数（顶层的「动画曲线」用的是上面那条 curves，两条各是各的，id 不共用）
    Item("strands", "I-SUMMARY-STRANDS", _strands),
    # 「SMPL 人体」是哪个身体，几个关节（lab2shot_shared.smpl 的 BODIES 说了算，这里只念出来）
    Item("body", "I-SUMMARY-BODY", _body),
    Item("items", "I-SUMMARY-ITEMS", _items),  # a list's own line (no type declares it: describe adds it)
    Item("contents", "I-SUMMARY-CONTENTS", _contents),
    # 网格的分区（USD 的 GeomSubset）：「按分区取出」按它列选项，交付到 DCC 时它就是组
    Item("subsets", "I-SUMMARY-SUBSETS", _subsets),
    # 点和曲线上还带着哪些属性（置信度、法线、类别、跟踪编号……）：名字 + 类型 + 逐点还是逐条，
    # 点节点右下角的感叹号就能看到
    Item("attributes", "I-SUMMARY-ATTRIBUTES", _attributes),
    Item("top", "I-SUMMARY-TOP", _names("top")),
    Item("focal", "I-SUMMARY-FOCAL", _focal),
    Item("filmback", "I-SUMMARY-FILMBACK", _filmback),
    Item("distortion", "I-SUMMARY-DISTORTION", _distortion),
    Item("raster", "I-SUMMARY-RASTER", lambda meta: ({"width": meta["raster"][0], "height": meta["raster"][1]} if meta.get("raster") else None)),
    Item("source", "I-SUMMARY-SOURCE",
         lambda meta: ({"level": Msg(LEVEL_SAID[(meta["source"] or {}).get("level")]).text}
                       if (meta.get("source") or {}).get("level") in LEVEL_SAID else None)),
    Item("value", "I-SUMMARY-VALUE", _value),
    Item("name", "I-SUMMARY-NAME", _key("name")),
    Item("main", "I-SUMMARY-MAIN", _key("main")),
    Item("files", "I-SUMMARY-FILES", _count("files")),
    Item("commercial", "I-SUMMARY-COMMERCIAL", lambda meta: {} if meta.get("commercial") else None),
    Item("learned", "I-SUMMARY-LEARNED", lambda meta: {"by": meta["learned"]} if meta.get("learned") else None),
)})


def items_of(type_id: str) -> tuple[str, ...]:
    """The summary items of a type: its own and every parent's, each once, in the order they are declared."""
    from .contracts import lineage

    out: list[str] = []
    for t in lineage(type_id):
        for item in DATA_TYPES[t].summary if t in DATA_TYPES else ():
            if item not in out:
                out.append(item)
    return tuple(out)


def describe(packet) -> dict:
    """What this result is, as the port tooltip, the information panel and 「取信息」 read it."""
    from .types import is_list, type_label

    if packet is None:
        return {}
    meta = {**packet.meta, "_type": packet.type}
    if is_list(packet.type):  # a list holds no data of its own: how many items it has, and their names
        lines = [] if packet.meta.get("empty") else [line for line in [_line(ITEMS["items"], _items(meta))] if line]
        return {"type": packet.type, "label": type_label(packet.type),
                "known": "empty" if packet.meta.get("empty") else "cooked", "items": lines}
    kind = DATA_TYPES.get(packet.type)
    if packet.meta.get("empty"):
        return {"type": packet.type, "label": kind.label if kind else packet.type, "known": "empty", "items": []}
    return {"type": packet.type, "label": kind.label if kind else packet.type, "known": "cooked",
            "items": [line for item in items_of(packet.type) if (line := _line(ITEMS[item], ITEMS[item].read(meta)))]}


def describe_info(type_id: str, info) -> dict:
    """What is known before it is cooked (the node's Info): the same items, as far as they are known."""
    kind = DATA_TYPES.get(type_id)
    lines = []
    for item in items_of(type_id):
        entry = ITEMS[item]
        if entry.before is not None and (line := _line(entry, entry.before(info))):
            lines.append(line)
    return {"type": type_id, "label": kind.label if kind else type_id, "known": "planned", "items": lines}


def _line(item: Item, value: Any) -> dict | None:
    """One line of the reply: its value (for 「取信息」 and the page's own use) and its text. The message says the name
    and the value in one template, 「名字：值」 (all the Chinese in the catalogue, one entry per item); the name is
    handed over on its own so a panel can put it in its own column."""
    if value is None:
        return None
    said = item.said(value)
    label, _, rest = said.text.partition("：")
    return {"id": item.id, "code": said.code, "label": label if rest else "", "value": value,
            "text": rest or said.text, "said": said.text}
