"""Splitting data that holds several items, and merging the items back together.

Some types carry several items: 人物框 several people, 场景 several groups under /shot, 2D 跟踪点 several groups
of points, 分割图 several objects. 「逐项开始」 processes them one at a time, 「逐项结束」 collects them again, and
「取一条」 extracts one. All of this goes through this module: a type declares the items it holds (`DataType.items`, the
Chinese word for one item) and registers here how to name, split and merge them. Nodes never do this themselves.

    names(packet)                 the items' names, in order (meta only: no file opened)
    split(packet, name, out)      one item as a packet of the same type, holding only it
    merge([(name, packet)], out)  the items back into one packet of the type

Names are a first-class property: two items of one list may not share a name (B-NAME-SAME), and nothing is
ever renamed silently; the user sees the name and can change it («命名»).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ..errors import Invalid
from ..messages import Msg
from .packet import Packet
from .types import DATA_TYPES


@dataclass(frozen=True)
class Item:
    """One item of a list: its name (unique in that list) and the fingerprint of its packet. A list never copies what
    it holds, so the item is that packet (data/packet.py). Its key is the hash of both, so the same data under two
    names forms two items. This is what 「逐项开始」 hands the engine (engine/scopes.py: the scope protocol) and what
    every node that produces a list writes."""

    name: str
    packet: str

    @property
    def key(self) -> str:
        from ..io.digest import key

        return key(["item", self.name, self.packet], 24)


def port_fp(*parts) -> str:
    """The fingerprint of an output of a block's begin node, derived from its dependencies: the item's packet and
    the port for 名字, those plus the index for 序号, the list's packet and the port for 总数."""
    from ..io.digest import key

    return key(["item-output", *parts], 24)


@dataclass(frozen=True)
class ItemKind:
    """How one type's items are named, taken apart and put back together."""

    names: Callable[[Mapping], list[str]]  # (meta) -> the items' names, in order
    split: Callable[[Packet, str, Path], Packet]
    merge: Callable[[list[tuple[str, Packet]], Path], Packet]


# ------------------------------------------------------------------ 人物框: one person each


def _person_name(pid) -> str:
    return f"person_{int(pid):02d}" if str(pid).isdigit() else str(pid)


def _boxes_names(meta: Mapping) -> list[str]:
    return [_person_name(p) for p in meta.get("people") or ()]


def _boxes_split(p: Packet, name: str, out: Path) -> Packet:
    from .payloads import read_boxes, write_boxes

    people = [q for q in read_boxes(p) if _person_name(q["id"]) == name]
    if not people:
        raise Invalid(Msg("E-ITEMS-NONAME", name=name, kind=DATA_TYPES[p.type].label))
    return write_boxes(out, people, p.meta["width"], p.meta["height"], p.meta["frames"], chosen=True)


def _boxes_merge(parts: list[tuple[str, Packet]], out: Path) -> Packet:
    from .payloads import read_boxes, write_boxes

    _refuse_same_names(parts, "boxes")
    people, first = [], parts[0][1]
    for _, packet in parts:
        people += read_boxes(packet)
    _refuse_same_names([(_person_name(q["id"]), first) for q in people], "boxes")
    return write_boxes(out, people, first.meta["width"], first.meta["height"], first.meta["frames"],
                       chosen=all(q.meta.get("chosen") for _, q in parts))


# ------------------------------------------------------------------ 2D 跟踪点: a group of points each


def _track_groups(meta: Mapping) -> list[dict]:
    """The groups of points a packet holds: those it declares (`groups`), otherwise all of its points as one group
    named after the data itself (a tracker produces one grid; a planar track produces one quad)."""
    groups = meta.get("groups")
    if groups:
        return [dict(g) for g in groups]
    return [{"name": "tracks", "first": 0, "count": int(meta.get("count") or len(meta.get("names") or ()))}]


def _tracks_names(meta: Mapping) -> list[str]:
    return [g["name"] for g in _track_groups(meta)]


def _tracks_split(p: Packet, name: str, out: Path) -> Packet:
    import numpy as np

    from .payloads import read_tracks, tracks_packet

    group = next((g for g in _track_groups(p.meta) if g["name"] == name), None)
    if group is None:
        raise Invalid(Msg("E-ITEMS-NONAME", name=name, kind=DATA_TYPES[p.type].label))
    rows = slice(int(group["first"]), int(group["first"]) + int(group["count"]))
    d = read_tracks(p)
    extra = {}
    if "confidence" in d:
        extra["confidence"] = d["confidence"][rows]
    if "homography" in d and int(group["count"]) == int(p.meta["count"]):  # a plane's homography belongs to all its corners
        extra["homography"] = d["homography"]
    return tracks_packet(out, p.meta["frames"], p.meta["width"], p.meta["height"], d["tracks"][rows],
                         d["visible"][rows], d["query_frames"][rows], list(p.meta["names"])[rows], **extra,
                         groups=[{"name": name, "first": 0, "count": int(group["count"])}],
                         **{k: p.meta[k] for k in ("pixel_aspect",) if k in p.meta})


def _tracks_merge(parts: list[tuple[str, Packet]], out: Path) -> Packet:
    import numpy as np

    from .payloads import read_tracks, tracks_packet

    _refuse_same_names(parts, "tracks2d")
    first = parts[0][1]
    tracks, visible, queries, names, groups = [], [], [], [], []
    for name, packet in parts:
        d = read_tracks(packet)
        groups.append({"name": name, "first": sum(g["count"] for g in groups), "count": int(packet.meta["count"])})
        tracks.append(d["tracks"])
        visible.append(d["visible"])
        queries.append(d["query_frames"])
        names += [f"{name}_{n}" for n in packet.meta["names"]]
    return tracks_packet(out, first.meta["frames"], first.meta["width"], first.meta["height"],
                         np.concatenate(tracks), np.concatenate(visible), np.concatenate(queries), names, groups=groups,
                         **{k: first.meta[k] for k in ("pixel_aspect",) if k in first.meta})


# ------------------------------------------------------------------ 分割图: one object each


def _class_names(meta: Mapping) -> list[str]:
    return [str(c.get("name") or c["index"]) for c in meta.get("classes") or () if int(c.get("index", 0)) > 0]


def _segmentation_split(p: Packet, name: str, out: Path) -> Packet:
    import numpy as np

    from .maps import map_at
    from .payloads import ExrWriter, window_of

    entry = next((c for c in p.meta.get("classes") or () if str(c.get("name") or c["index"]) == name), None)
    if entry is None:
        raise Invalid(Msg("E-ITEMS-NONAME", name=name, kind=DATA_TYPES[p.type].label))
    index = int(entry["index"])
    writer = ExrWriter(out, 1, window=window_of(p), value_range=(0.0, float(index)),
                       classes=[dict(entry)], **{k: p.meta[k] for k in ("pixel_aspect",) if k in p.meta})
    for f in p.meta["frames"]:
        labels = map_at(p, f)[0]
        writer.add(f, np.where(np.rint(labels) == index, labels, 0.0))
    return writer.packet()


def _segmentation_merge(parts: list[tuple[str, Packet]], out: Path) -> Packet:
    import numpy as np

    from .maps import map_at
    from .payloads import ExrWriter, window_of

    _refuse_same_names(parts, "image.1")
    first = parts[0][1]
    classes = [c for _, packet in parts for c in packet.meta.get("classes") or ()]
    writer = ExrWriter(out, 1, window=window_of(first),
                       value_range=(0.0, float(max((int(c["index"]) for c in classes), default=1))), classes=classes,
                       **{k: first.meta[k] for k in ("pixel_aspect",) if k in first.meta})
    for f in first.meta["frames"]:
        merged = None
        for _, packet in parts:
            labels = np.rint(map_at(packet, f)[0])
            if merged is None:
                merged = labels
                continue
            if bool(((merged > 0) & (labels > 0)).any()):  # two objects on one pixel: no silent choice is made
                raise Invalid(Msg("E-ITEMS-OVERLAP", frame=f))
            merged = np.where(labels > 0, labels, merged)
        writer.add(f, merged)
    return writer.packet()


# ------------------------------------------------------------------ 场景: one group under /shot each


def _scene_names(meta: Mapping) -> list[str]:
    return [str(n) for n in meta.get("top") or ()]


def _scene_split(p: Packet, name: str, out: Path) -> Packet:
    from .scene import only_group

    got = only_group(p, name, out)
    if got is None:
        raise Invalid(Msg("E-ITEMS-NONAME", name=name, kind=DATA_TYPES[p.type].label))
    return got


def _scene_merge(parts: list[tuple[str, Packet]], out: Path) -> Packet:
    """Merge named scenes into one. The result keeps the type the parts share (three merged cameras are 相机, not the
    generic 场景), so it can be used wherever one of them could (the output of 「列表合并」 carries its items' type).
    Differing types have only 場景 in common, which is the purpose of 场景."""
    from .scene import join_named

    _refuse_same_names(parts, "scene")
    return join_named(parts, out, _one_type([p.type for _, p in parts]))


def _one_type(types: list[str]) -> str:
    """The type shared by all of them: the widest one that accepts every other (data/types.py accepts), or 场景 when
    that is their only common type."""
    from .types import accepts

    return next((t for t in types if all(accepts(t, other) for other in types)), "scene")


def _refuse_same_names(parts, type_id: str) -> None:
    """Two items of one list may never share a name: nothing is renamed silently, and the conflict is reported for
    「命名」 to resolve."""
    names = [name for name, _ in parts]
    same = sorted({n for n in names if names.count(n) > 1})
    if same:
        raise Invalid(Msg("B-NAME-SAME", kind=DATA_TYPES[type_id].label, names=same))


KINDS: Mapping[str, ItemKind] = MappingProxyType({
    "boxes": ItemKind(_boxes_names, _boxes_split, _boxes_merge),
    "tracks2d": ItemKind(_tracks_names, _tracks_split, _tracks_merge),
    # A single-channel mask has items only when it holds class indices (with its own class table `classes`):
    # segmentation is one use of a channel, not a type of its own; a channel's meaning is defined where it is used.
    "image.1": ItemKind(_class_names, _segmentation_split, _segmentation_merge),
    "scene": ItemKind(_scene_names, _scene_split, _scene_merge),
})


def kind_of(type_id: str) -> ItemKind | None:
    """How this type's items are handled (None for a type that holds no items); a subtype inherits its parent's (every
    场景 subtype)."""
    from .contracts import lineage

    for t in reversed(lineage(type_id)):
        if t in KINDS:
            return KINDS[t]
    return None


def names(p: Packet) -> list[str]:
    """The names of the items this packet holds, in order ([] for a type that holds no items, or for an empty one)."""
    kind = kind_of(p.type)
    return kind.names(p.meta) if kind is not None and not p.meta.get("empty") else []


def split(p: Packet, name: str, out: Path) -> Packet:
    """One item as a packet of the same type, holding only it."""
    kind = kind_of(p.type)
    if kind is None:
        raise Invalid(Msg("E-ITEMS-NOITEMS", kind=DATA_TYPES[p.type].label))
    return kind.split(p, name, out)


def as_items(p: Packet, by: str, version: int, only: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """The items of `p`, each as a packet of its own: [(name, its fingerprint)] in `p`'s own order, exactly as a list
    packet names them (data/packet.py items_meta). `only`: restrict to these names (「选人」 passes on the chosen
    people); None: every item.

    Two nodes build lists this way, 「拆成列表」 (everything in the data) and 「选人」 (the chosen people), so the logic
    lives in one place. Both produce lists of the same shape, so any reader of one also reads the other; the viewer's
    item bar handles a single case.

    An item's packet is named after the node type, its version, the source data and the item's own name
    (packet.item_fingerprint), never after the list: an additional person in the detection leaves the other people's
    packets, and everything cooked from them, unchanged, and a second cook finds them already present.
    """
    from .packet import item_fingerprint, produce

    wanted = None if only is None else set(only)
    parts: list[tuple[str, str]] = []
    for name in names(p):
        if wanted is not None and name not in wanted:
            continue
        fp = item_fingerprint(by, version, p.fingerprint, name)
        produce(fp, lambda d, name=name: split(p, name, d).commit(by))  # under the entry's lock (data/packet.py produce)
        parts.append((name, fp))
    return parts


def merge(parts: list[tuple[str, Packet]], out: Path) -> Packet:
    """Named items back into one packet of their type (they must be of one type, and their names must differ)."""
    if not parts:
        raise Invalid(Msg("E-ITEMS-NOPARTS"))
    type_id = parts[0][1].type
    kind = kind_of(type_id)
    if kind is None:
        raise Invalid(Msg("E-ITEMS-NOITEMS", kind=DATA_TYPES[type_id].label))
    return kind.merge(parts, out)


def holds_nothing(type_id: str, meta: dict) -> bool:
    """Data of a type that holds items, with none in it: 人物框 with nobody, no tracked point. An input that does not
    take that (Port.takes_empty) has nothing to cook (engine/cook.py)."""
    kind = kind_of(type_id)
    if kind is None or meta.get("empty"):
        return False
    return kind.names(meta) == [] and any(k in meta for k in ("people", "names", "classes", "top", "count"))
