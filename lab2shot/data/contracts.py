"""The contract of every data type, checked by code: the rules that keep nodes composable are enforced here, not
remembered.

check() runs on every packet a node produces, before it is committed (engine/cook.py): a node that breaks its type's
contract fails with the reason instead of passing bad data downstream, whichever project it wraps. check(deep=True)
also opens the files (the GPU tests run it on every result). A new type gets its contract here, together with its
entry in types.DATA_TYPES; a type without a contract is not a type.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


from ..data.payloads import SCENE_FILE
from ..data.packet import Packet, file_path
from ..messages import Msg
from .types import DATA_TYPES

ANY = None

# ------------------------------------------------------------------ what was photographed
# The shot properties: what the camera and the lens made of these pixels (or of these pixel coordinates), never what a
# node did to their values. A node that does not move pixels cannot change them, so it does not carry them itself: the
# engine fills them in from the node's picture input, by what its output port declares (Shape). A node that writes one
# without declaring it fails (E-CONTRACT-SHOTCHANGED), so no result can quietly lose the lens the plate carries.
# `lens`: the lens these pixels were made with, as a camera's lens description writes one ({}: none said). An
# undistorted picture carries the lens 「LensDistortion」 took out of it, so what is solved on it can be delivered with the
# distortion to put back; a redistorted one carries the lens that is in it again.
# Whether a picture is original or undistorted is not a data state: common DCCs (Nuke included) have no such flag, and
# the user knows whether an image is distorted, so there are only the lens and the pixel aspect here, no distortion
# state key.
SHOT_KEYS: Mapping[str, frozenset | None] = MappingProxyType({"pixel_aspect": ANY, "lens": ANY})
# what a packet says before anything says better: not said (the packet constructors write these, so no packet is ever
# without them; the engine replaces them with the picture's). A node writing anything else says something of its own,
# and its output port must declare that it does (settle)
NOT_SAID: Mapping[str, object] = MappingProxyType({"pixel_aspect": 1.0, "lens": {}})
# the type roots that carry them: pictures and float images, and the pixel coordinates measured on them
SHOT_TYPES = ("video", "image", "boxes", "tracks2d")

# The optional meanings a 2D map (image.N) may carry. What a channel means is defined by its use, so these are not part
# of the type; the writing node may state them, and a stated value must be valid (one of the values in this table).
# own_meta passes them unchanged through nodes that only rearrange pixels (「STMap」, 「运动矢量变形」).
MEANING: Mapping[str, frozenset | None] = MappingProxyType({k: v if v is ANY else frozenset(v) for k, v in {
    "values": {True, False},   # numeric values, not colour (the viewer maps them by range, without OCIO)
    "range": ANY,              # the viewer's display range
    "validity": {True, False}, # an extra valid channel per frame marking the pixels that hold a value
    # metric: the method's claim of real distances (see its help for how close); relative: right up to one unknown
    # factor (consistent with the camera of the same solve); affine: up to an unknown factor and offset
    "scale": {"metric", "relative", "affine", "disparity"},
    "space": {"world", "camera", "canonical"},  # canonical: positions in a template / canonical space (faces, bodies and object models alike; no project terms)
    # whose score it is ("VGGT"; "" read from a file): scores of one model compare, of two models don't
    "model": ANY,
    # how the positions were projected onto UV (data/projection.py WAYS): cylinder / sphere / plane
    "projection": {"cylinder", "sphere", "plane", "unknown"},
    "classes": ANY,            # id-to-name table: its presence marks a segmentation (written as Cryptomatte in multi-layer EXR)
    # the channel is a matte alpha (0 = fully transparent, 1 = fully opaque). Wiring ignores it (a channel may connect
    # to any single-channel port); it is read at two points of use:
    #   · viewer: by default overlaid semi-transparently on the upstream original instead of shown as a grey image
    #     (otherwise only the alpha would be visible after matting, without the original colours);
    #   · delivery: written to the EXR a channel, not R.
    "matte": {True},
    "direction": {"undistort", "distort"},  # whether this ST-map undistorts or distorts
    "aligned": ANY,            # the values computed by depth alignment
}.items()})

# meta keys every packet of a type carries -> the allowed values (ANY: any value). A type also carries its parents'.
# Read-only, like the type table.
META: Mapping[str, Mapping[str, frozenset | None]] = MappingProxyType({
    t: MappingProxyType({k: v if v is ANY else frozenset(v) for k, v in keys.items()}) for t, keys in {
    "video": {"path": ANY, "count": ANY, "frames": ANY, "fps": ANY, "width": ANY, "height": ANY,  # fps: this file's own, see payloads.video_packet  # frames count from 0
              **SHOT_KEYS},
    # alpha: the pictures carry one (RGBA, premultiplied: io/images.py)
    # width, height: the plate frame (display window); data_window [x, y, w, h] from its corner, the union over the
    # frames: every pixel the packet keeps, overscan and all (one window, data/windows.py Window)
    "image": {"frames": ANY, "files": ANY, "width": ANY, "height": ANY, "colorspace": ANY, "alpha": {True, False},
              "data_window": ANY, **SHOT_KEYS},
    # the four 2D pixel types (data/types.py PIXELS) require nothing beyond the channel count. What a channel means is
    # defined by its use, so there are no per-kind keys such as "depth must state scale" or "positions must state space";
    # values / range of numeric maps (data/payloads.py map_packet) and the coordinate system are statements written by
    # the producing node, used by the viewer and outputs when present and treated as unstated otherwise
    **{f"image.{n}": {} for n in (1, 2, 3, 4)},
    # people: the ids; chosen: someone chose them (「选人」), else everyone a detection found
    "boxes": {"people": ANY, "chosen": {True, False}, "frames": ANY, "width": ANY, "height": ANY, **SHOT_KEYS},
    "tracks2d": {"frames": ANY, "width": ANY, "height": ANY, "count": ANY, "names": ANY, **SHOT_KEYS},
    "curves": {"frames": ANY, "names": ANY, "range": ANY},
    "scene": {"frames": ANY},
    "scene.camera": {},
    "scene.character": {},
    "scene.model": {},
    "scene.skeleton": {},
    # like a depth's: real distances, or right up to one factor (a solve without scale: COLMAP)
    "scene.points": {"scale": {"metric", "relative"}},
    # 3D curves require no keys: curve and point counts are counted when the packet is made
    # (data/payloads.py _scene_contents), which also holds for scenes copied unchanged from DCC files
    "scene.curves": {},
    # what an output-settings node wrote: its 名字 (the sub-folder 「输出」 puts it in), the main file (a sequence as ####),
    # every file relative to the packet (the provenance sidecar too), whether all of it may be used commercially
    "files": {"name": ANY, "main": ANY, "files": ANY, "commercial": ANY, "learned": ANY},
    # a basic value (nodes/values.py): one value, or one per frame; its unit ("" none). What else it holds is checked
    # by values.check
    "value": {"unit": ANY},
    "value.float": {},
    "value.int": {},
    "value.bool": {},
    "value.vector": {},
    "value.text": {},
    "value.lens": {},
}.items()})


@dataclass(frozen=True)
class Shape:
    """What an output port says about its own geometry and the shot properties it gives back.

    `window`: "picture" the picture input's window (the default: the node does not move pixels, so its result covers
    the same plate frame; it may leave the overscan out, saying so); "display" the plate frame alone; "input:<port>"
    the window of that input (a warp lands on its map's window); "node" the node decides (「LensDistortion」's canvas, an
    HDRI), and then its `info()` says the size before it cooks, or the node's face says it is known only afterwards.
    `pixel_aspect` / `lens`: "keep" the picture input's (the engine fills it in), "node" the node
    writes it itself (NodeDef.said_shot before it cooks), "unknown" the engine writes 1.0 / 没有镜头 (a warp by
    something that is not a lens). `said`: the message that says
    what the size will be, for a "node" window."""

    window: str = "picture"
    pixel_aspect: str = "keep"
    lens: str = "keep"
    said: str = ""

    def __post_init__(self) -> None:
        if not (self.window in ("picture", "display", "node") or self.window.startswith("input:")):
            raise ValueError(f"window must be picture / display / node / input:<port>, not {self.window!r}")
        for key, how in (("pixel_aspect", self.pixel_aspect), ("lens", self.lens)):
            if how not in ("keep", "node", "unknown"):
                raise ValueError(f"{key} must be keep / node / unknown, not {how!r}")

    @property
    def follows(self) -> str:
        """The input port whose window this output takes ("" it takes the picture's, its own, or the plate frame)."""
        return self.window.removeprefix("input:") if self.window.startswith("input:") else ""


KEEPS = Shape()  # the default: same window, same shot properties
PLATE_FRAME = Shape(window="display")  # it can only work on the plate frame (it says what it leaves out)
NEW_PICTURE = Shape(window="node", pixel_aspect="node", lens="node")  # a picture of its own (an HDRI)


# an ST-map a node hands out (a solve's or an estimate's): it decides its own window (the undistort map
# lies on the canvas its overscan makes) and it is not a picture of the shot, so it carries no lens
STMAP_SHAPE = Shape(window="node", lens="unknown")


def warped_by(port: str) -> Shape:
    """A warp by a map that is not known to be a lens: the result lies on that map's window and no longer shows what
    the lens saw."""
    return Shape(window=f"input:{port}", lens="unknown")


def shot_of(shape: Shape, picture: Packet | Mapping | None, own: Mapping = MappingProxyType({})) -> dict:
    """The shot properties an output with this shape gets ({} for what is not known here). Pure, so the graph works
    it out before anything is cooked as well as the engine after (engine/evaluation.py Evaluation.shot, settle):
    `picture` is the node's picture input (a packet, or just what is known of it), `own` what the node says itself
    (NodeDef.said_shot: a reader's 镜头状态 and 像素比, 「LensDistortion」's direction), which only an output declaring
    "node" for that key takes."""
    meta = picture.meta if isinstance(picture, Packet) else (picture or {})
    out = {}
    for key in SHOT_KEYS:
        how = getattr(shape, key)
        if how == "unknown":  # a warp by something that is not a lens: what was photographed is no longer said
            out[key] = NOT_SAID[key]
        elif how == "node" and key in own:
            out[key] = own[key]
        elif how == "keep" and key in meta:
            out[key] = meta[key]
    return out


def lineage(type_id: str) -> list[str]:
    """The type and its parents, most general first: image.1 -> [image, image.1]."""
    parts = type_id.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def own_meta(p: Packet) -> dict:
    """The meanings a packet states about its data (a depth's scale, a position's coordinate system, a segmentation's
    class table, ...), passed unchanged by nodes that only rearrange pixels (「STMap」, 「运动矢量变形」). Only stated
    meanings are passed."""
    said = {k: p.meta[k] for t in lineage(p.type)[1:] for k in META[t] if k in p.meta}
    said.update({k: p.meta[k] for k in MEANING if k in p.meta and k not in ("values", "range", "validity")})
    return said


# What a consuming node assumes when a 2D map does not state a meaning (the safe side: scale as requiring manual
# scaling rather than real distances; coordinate system as world, which is what renderers write for P and N).
# Using a default is reported on the node (N-MEANING-ASSUMED), so that the source of every value is stated
ASSUMED: Mapping[str, str] = MappingProxyType({"scale": "relative", "space": "world", "projection": "unknown", "model": ""})


def meant(ctx, p: Packet, key: str, source: str = "") -> object:
    """The meaning the packet states for `key`; if unstated, the ASSUMED value, reported on the node."""
    if key in p.meta:
        return p.meta[key]
    if ctx is not None:
        ctx.say("N-MEANING-ASSUMED", source=source or key, key=key, value=ASSUMED[key])
    return ASSUMED[key]


def shot_meta(p: Packet | Mapping) -> dict:
    """What a packet (or just its description, as a manifest gives it) says of the shot it was made from (SHOT_KEYS):
    {} for a type that carries none."""
    meta = p.meta if isinstance(p, Packet) else p
    return {k: meta[k] for k in SHOT_KEYS if k in meta}


def carries_shot(type_id: str) -> bool:
    return type_id.split(".")[0] in SHOT_TYPES


def settle(produced: Mapping[str, Packet], shapes: Mapping[str, "Shape"], picture: Packet | None,
           own: Mapping = MappingProxyType({})) -> list[tuple[str, Msg]]:
    """Fill in every output's shot properties from the node's picture input and what the node says itself (`own`:
    NodeDef.said_shot, for the keys a port declares "node", the same answer the graph foresaw before the cook), and
    check its window against what its port declares, then check each packet against its contract. Returns [(port, what is wrong)];
    it changes the packets' meta, so it runs once, before they are committed (engine/cook.py _settle_outputs).

    A node that wrote a shot property its port did not declare it would write fails here (E-CONTRACT-SHOTCHANGED):
    losing the lens the plate carries is a production accident (the delivered camera could not give the distortion
    back), so it is caught at the node that did it."""
    from .windows import Window

    problems: list[tuple[str, Msg]] = []
    for port, packet in produced.items():
        shape = shapes.get(port) or KEEPS
        if packet.meta.get("empty") or not carries_shot(packet.type):
            problems += [(port, said) for said in check(packet)]
            continue
        for key, value in shot_of(shape, picture, own).items():
            said = packet.meta.get(key, NOT_SAID[key])
            if said != value and said != NOT_SAID[key]:  # it said something of its own without declaring it
                problems.append((port, Msg("E-CONTRACT-SHOTCHANGED", key=key, value=str(said), plate=str(value))))
            packet.meta[key] = value
        problems += [(port, said) for said in _window_problems(packet, shape, picture, Window)]
        problems += [(port, said) for said in check(packet)]
    return problems


def _window_problems(packet: Packet, shape: Shape, picture: Packet | None, Window) -> list[Msg]:
    """The output's size against what its port declares: every 2D output covers the plate frame its picture
    input covered, unless the port says the node decides it. A picture node may leave the overscan out (it says so:
    N-COOK-OVERSCAN), so the data window is either the picture's or the plate frame."""
    if shape.window == "node" or picture is None or "width" not in packet.meta:
        return []
    want = picture if shape.window in ("picture", "display") else None
    if want is None:  # "input:<port>": the engine passes that input as the picture (it is what this output follows)
        return []
    if "width" not in want.meta or want.meta.get("empty"):
        return []
    theirs, mine = (want.meta["width"], want.meta["height"]), (packet.meta["width"], packet.meta["height"])
    if mine != theirs:
        return [Msg("E-COOK-OUTPUTSIZE", width=mine[0], height=mine[1], plate_width=theirs[0], plate_height=theirs[1])]
    if "data_window" not in packet.meta or "data_window" not in want.meta:
        return []
    got, plate = Window.of(packet.meta), Window.of(want.meta)
    allowed = [plate.display_box] if shape.window == "display" else [plate.data, plate.display_box]
    if got.data not in allowed:
        return [Msg("E-COOK-OUTPUTWINDOW", window=str(list(got.data)), want=" / ".join(str(list(w)) for w in allowed))]
    return []


def check(p: Packet, deep: bool = False) -> list[Msg]:
    """What is wrong with this packet for its type (empty: nothing). An empty packet (meta "empty": a node that
    has nothing to give, e.g. 「相机属性」 without a focal length) only needs its type."""
    from .types import channels_of, is_list

    if is_list(p.type):
        return _check_list(p, deep)
    if p.type not in DATA_TYPES:
        return [Msg("E-CONTRACT-UNKNOWNTYPE", type=p.type)]
    if p.meta.get("empty"):
        return []
    problems = []
    for t in lineage(p.type):
        for key, allowed in META[t].items():
            if key not in p.meta:
                problems.append(Msg("E-CONTRACT-MISSINGKEY", key=key))
            elif allowed is not None and p.meta[key] not in allowed:
                problems.append(Msg("E-CONTRACT-KEYVALUE", key=key, value=repr(p.meta[key]), allowed=" / ".join(map(str, sorted(allowed, key=str)))))
    if channels_of(p.type):  # a 2D map: its meanings are optional, but stated ones must be valid
        for key, allowed in MEANING.items():
            if key in p.meta and allowed is not None and p.meta[key] not in allowed:
                problems.append(Msg("E-CONTRACT-KEYVALUE", key=key, value=repr(p.meta[key]), allowed=" / ".join(map(str, sorted(allowed, key=str)))))
    if problems:
        return problems
    root = p.type.split(".")[0]
    if root == "image":
        problems += _check_frames(p, deep)
    elif root == "files":
        missing = [f for f in p.meta["files"] if not p.path(f).is_file()]
        if not p.meta["files"] or missing:
            problems.append(Msg("E-CONTRACT-FILEMISSING", file=missing[0]) if missing else Msg("E-CONTRACT-NOFILES"))
    elif root == "scene":
        if not p.path(SCENE_FILE).is_file():
            problems.append(Msg("E-CONTRACT-NOSCENE", file=SCENE_FILE))
        elif deep:
            problems += _check_stage(p)
    elif root == "value":
        from .values import check as check_value

        problems += check_value(p)
    elif deep and root in _FILES:
        problems += _FILES[root](p)
    return problems


def _check_list(p: Packet, deep: bool) -> list[Msg]:
    """A list packet (data/packet.py): its items in order, each with a name of its own (never two the same)
    and a packet of the list's item type that is really in the cache. The items' own contracts were checked where they
    were made: a list copies nothing, so it can break nothing."""
    from .packet import Packet as P
    from .packet import items_of, packet_dir
    from .types import DATA_TYPES, accepts, element_of

    carried = element_of(p.type)
    if carried not in DATA_TYPES:
        return [Msg("E-CONTRACT-UNKNOWNTYPE", type=p.type)]
    if p.meta.get("empty"):
        return []
    if "items" not in p.meta:
        return [Msg("E-CONTRACT-MISSINGKEY", key="items")]
    names = [n for n, _ in items_of(p)]
    same = sorted({n for n in names if names.count(n) > 1})
    if same:
        return [Msg("B-NAME-SAME", kind=DATA_TYPES[carried].label, names=same)]
    for name, fp in items_of(p):
        d = packet_dir(fp)
        if not P.exists(d):
            return [Msg("E-CONTRACT-NOITEM", name=name)]
        item = P.load(d)
        if not accepts(carried, item.type):
            return [Msg("E-CONTRACT-ITEMTYPE", name=name, got=item.type, want=carried)]
        if deep:
            problems = check(item, deep)
            if problems:
                return problems
    return []


def _check_boxes(p: Packet) -> list[Msg]:
    """boxes.json: the people the description lists, each box [x1, y1, x2, y2] finite with x1 <= x2, y1 <= y2, on
    frames of the packet."""
    import json

    people = json.loads(p.path("boxes.json").read_text(encoding="utf-8"))["people"]
    if [q["id"] for q in people] != p.meta["people"]:
        return [Msg("E-CONTRACT-BOXPEOPLE")]
    frames = set(map(str, p.meta["frames"]))
    for q in people:
        for f, b in q["boxes"].items():
            if f not in frames:
                return [Msg("E-CONTRACT-BOXFRAME", person=q["id"], frame=f)]
            if len(b) != 4 or not all(math.isfinite(v) for v in b) or b[0] > b[2] or b[1] > b[3]:
                return [Msg("E-CONTRACT-BOX", person=q["id"], frame=f, box=str(b))]
    return []


def _check_tracks(p: Packet) -> list[Msg]:
    """tracks.npz: tracks [N,F,2] and visible [N,F] for the packet's frames, N points as named; visible positions
    finite; a confidence [N,F] in 0..1. Points on one plane (a planar track) may carry its homography [F,3,3]: the
    plane from the points' query frame to each frame, which must take the points there on every frame they are visible
    (within a pixel)."""
    import numpy as np

    d = np.load(p.path("tracks.npz"))
    n, frames = p.meta["count"], len(p.meta["frames"])
    if d["tracks"].shape != (n, frames, 2) or d["visible"].shape != (n, frames) or len(p.meta["names"]) != n:
        return [Msg("E-CONTRACT-TRACKSHAPE", shape=str(d["tracks"].shape), count=n, frames=frames)]
    if not np.isfinite(d["tracks"][d["visible"]]).all():
        return [Msg("E-CONTRACT-TRACKNAN")]
    if "confidence" in d and (d["confidence"].shape != (n, frames) or not np.isfinite(d["confidence"]).all()
                              or d["confidence"].min(initial=0) < 0 or d["confidence"].max(initial=0) > 1):
        return [Msg("E-CONTRACT-TRACKCONFIDENCE", count=n, frames=frames, shape=str(d["confidence"].shape))]
    if "homography" not in d:
        return []
    from ..data.payloads import apply_homography

    h, query = d["homography"], d["query_frames"]
    frames = p.meta["frames"]
    if h.shape != (len(frames), 3, 3) or not np.isfinite(h).all():
        return [Msg("E-CONTRACT-HOMOGRAPHYSHAPE", shape=str(h.shape), frames=len(frames))]
    if len(set(query.tolist())) != 1 or int(query[0]) not in frames:
        return [Msg("E-CONTRACT-HOMOGRAPHYQUERY")]
    ref = d["tracks"][:, frames.index(int(query[0]))]
    for j in range(len(frames)):
        seen = d["visible"][:, j]
        err = np.linalg.norm(apply_homography(h[j], ref[seen]) - d["tracks"][seen, j], axis=-1)
        if len(err) and err.max() > 1.0:
            return [Msg("E-CONTRACT-HOMOGRAPHY", frame=frames[j], pixels=float(err.max()))]
    return []


# The arrays in a packet's npz must match the counts the packet states and be finite. 「动画曲线」 and 「SMPL 人体」
# follow the same rule (as _check_point_arrays does for point clouds and 3D curves: an array's length is set by what it
# belongs to), so there is one check and one table rather than one per type. Each entry:
# file name -> (meta -> {array: expected shape}, arrays that must be finite)
ARRAYS: Mapping[str, tuple[str, object, tuple[str, ...]]] = MappingProxyType({
    "curves": ("curves.npz", lambda m: {"values": (len(m["frames"]), len(m["names"]))}, ("values",)),
})


def _check_arrays(p: Packet) -> list[Msg]:
    """Check the packet against the ARRAYS table: every array in its npz matches the stated counts of people, frames,
    curves or joints, and the arrays that must be finite contain no NaN or infinity."""
    import numpy as np

    file, shapes, finite = ARRAYS[p.type]
    d = np.load(p.path(file))
    label = DATA_TYPES[p.type].label
    for key, want in shapes(p.meta).items():
        got = d[key].shape if key in d.files else None
        if got is None or len(got) != len(want) or any(w is not None and g != w for g, w in zip(got, want)):
            return [Msg("E-CONTRACT-ARRAYSHAPE", kind=label, key=key, shape=str(got),
                        want="(" + ", ".join("*" if w is None else str(w) for w in want) + ")")]
    for key in finite:
        if not np.isfinite(d[key]).all():
            return [Msg("E-CONTRACT-ARRAYNAN", kind=label, key=key)]
    return []


_FILES = {"boxes": _check_boxes, "tracks2d": _check_tracks, **{t: _check_arrays for t in ARRAYS}}


def _check_frames(p: Packet, deep: bool) -> list[Msg]:
    frames, files = p.meta["frames"], p.meta["files"]
    problems = []
    if frames != sorted({int(f) for f in frames}):
        problems.append(Msg("E-CONTRACT-FRAMEORDER"))
    if set(map(str, frames)) != set(files):
        problems.append(Msg("E-CONTRACT-FRAMEFILES"))
    missing = [f for f, ref in files.items() if not file_path(p, ref).is_file()]
    if missing:
        problems.append(Msg("E-CONTRACT-FRAMESMISSING", count=len(missing), first=missing[0]))
    if problems or not deep or not frames:
        return problems
    from ..data.payloads import is_data
    from ..io import images

    path = file_path(p, files[str(frames[0])])
    w, h = images.image_size(path)
    if (w, h) != (p.meta["width"], p.meta["height"]):
        problems.append(Msg("E-CONTRACT-SIZE", width=w, height=h, meta_width=p.meta["width"], meta_height=p.meta["height"]))
    if is_data(p):
        problems += _check_map_values(path, p.type)
    return problems


def _check_map_values(path, type_: str) -> list[Msg]:
    """Check a 2D map written by this project: the channel count, and that valid pixels hold finite values.

    Value ranges are not checked per kind (depth, mask, normals): 2D data has only a channel count, and what a float
    channel means is defined by its use. Conventions such as value ranges are the responsibility of the writing node."""
    import numpy as np

    from ..data.payloads import VALIDITY, channel_names, read_map
    from ..data.types import channels_of
    from ..io import images

    count = channels_of(type_)
    named = images.read_named(path)
    want = (set(channel_names(count)), set(channel_names(count, True)))  # with or without the valid channel
    if set(named) not in want:
        return [Msg("E-CONTRACT-CHANNELS", have=" ".join(sorted(named)),
                    want=" ".join(sorted(channel_names(count))) + f" (+{VALIDITY})")]
    data, alpha = read_map(path)
    if not np.isfinite(data[alpha > 0]).all():
        return [Msg("E-CONTRACT-PIXELNAN")]
    return []


def _check_stage(p: Packet) -> list[Msg]:
    from pxr import Usd, UsdGeom, UsdSkel

    from ..io.usd import STAGE_FPS

    stage = Usd.Stage.Open(str(p.path(SCENE_FILE)))
    problems = []
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.y:
        problems.append(Msg("E-CONTRACT-UPAXIS"))
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 0.01):
        problems.append(Msg("E-CONTRACT-CENTIMETRES"))
    if not math.isclose(stage.GetTimeCodesPerSecond(), STAGE_FPS):  # frame numbers are time codes (io/usd.py apply_conventions)
        problems.append(Msg("E-CONTRACT-USDFPS", usd_fps=stage.GetTimeCodesPerSecond(), fps=STAGE_FPS))
    wanted = {"scene.camera": UsdGeom.Camera, "scene.points": UsdGeom.Points, "scene.model": UsdGeom.Mesh,
              "scene.character": UsdSkel.Skeleton, "scene.skeleton": UsdSkel.Skeleton,
              "scene.curves": UsdGeom.BasisCurves}.get(p.type)
    if wanted and not any(prim.IsA(wanted) for prim in stage.Traverse()):
        problems.append(Msg("E-CONTRACT-NOKIND", kind=DATA_TYPES[p.type].label))
    if p.type == "scene.camera" and not problems:
        problems += _check_camera(p)
    return problems or _check_point_arrays(stage, p.meta["frames"])


def _check_point_arrays(stage, frames: list[int]) -> list[Msg]:
    """Per-point and per-curve data matches what it belongs to, on every frame. One check covers both kinds of
    point-based geometry, because it is one rule:

    - a 点云: its ids (unique: a tracked point keeps its own), velocities and vertex primvars have one value per point;
    - a 三维曲线: its curve vertex counts add up to its points; its widths, normals and vertex primvars have one value
      per point, and a uniform primvar one per curve.
    """
    import numpy as np
    from pxr import Usd, UsdGeom

    vertex = (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying)  # a linear curve's varying is per point too
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Points):
            cloud = UsdGeom.Points(prim)
            per_point = [("ids", cloud.GetIdsAttr()), ("velocities", cloud.GetVelocitiesAttr())]
            per_point += [(pv.GetPrimvarName(), pv.GetAttr()) for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars()
                          if pv.GetInterpolation() in vertex]
            per_curve, counts_attr = [], None
        elif prim.IsA(UsdGeom.BasisCurves):
            curves = UsdGeom.BasisCurves(prim)
            counts_attr = curves.GetCurveVertexCountsAttr()
            named = [("widths", curves.GetWidthsAttr(), curves.GetWidthsInterpolation()),
                     ("normals", curves.GetNormalsAttr(), curves.GetNormalsInterpolation())]
            named += [(pv.GetPrimvarName(), pv.GetAttr(), pv.GetInterpolation()) for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]
            per_point = [(n, a) for n, a, i in named if i in vertex]
            per_curve = [(n, a) for n, a, i in named if i == UsdGeom.Tokens.uniform]
        else:
            continue
        points_attr = UsdGeom.PointBased(prim).GetPointsAttr()
        for f in frames or [Usd.TimeCode.Default()]:
            n = len(points_attr.Get(f) or [])
            counts = np.asarray(counts_attr.Get(f) or [], np.int64) if counts_attr is not None else None
            if counts is not None and int(counts.sum()) != n:
                return [Msg("E-CONTRACT-CURVEPOINTS", curves=prim.GetName(), frame=str(f), count=n, counted=int(counts.sum()))]
            for name, attr in per_point:
                values = attr.Get(f) if attr and attr.HasAuthoredValue() else None
                if values is not None and len(values) != n:
                    said = {"curves": prim.GetName()} if counts is not None else {"cloud": prim.GetName()}
                    code = "E-CONTRACT-CURVECOUNT" if counts is not None else "E-CONTRACT-CLOUDCOUNT"
                    return [Msg(code, **said, frame=str(f), count=n, attribute=name, values=len(values))]
                if name == "ids" and values is not None and len(np.unique(np.asarray(values))) != n:
                    return [Msg("E-CONTRACT-CLOUDIDS", cloud=prim.GetName(), frame=str(f))]
            for name, attr in per_curve:
                values = attr.Get(f) if attr and attr.HasAuthoredValue() else None
                if values is not None and len(values) != len(counts):
                    return [Msg("E-CONTRACT-STRANDCOUNT", curves=prim.GetName(), frame=str(f), count=len(counts), attribute=name, values=len(values))]
    return []


def _check_camera(p: Packet) -> list[Msg]:
    """A real camera on every frame of the packet: a finite place, a focal length and a film back."""
    import numpy as np

    from ..data.camera import CameraSamples

    cam = CameraSamples.from_packet(p, p.meta["frames"] or [0])
    checked = (cam.cam_to_world, cam.focal_mm, cam.h_aperture_mm, cam.v_aperture_mm)
    if not all(np.isfinite(v).all() for v in checked):
        return [Msg("E-CONTRACT-CAMERANAN")]
    if (cam.focal_mm <= 0).any() or (cam.h_aperture_mm <= 0).any():
        return [Msg("E-CONTRACT-CAMERALENS")]
    return []
