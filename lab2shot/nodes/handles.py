"""Viewer handles: how a node is worked on in the viewer.

A handle is bound to parameters, as in Houdini: when the node is the active one the viewer shows its handles in their
stage, and using a handle edits those parameters (undoable). A handle only displays: while it is dragged, clicked or
drawn the viewer shows the results already there with the handle's current parameters (the picks, outlines and
figures drawn on the picture, the chosen people lit, the scene placed by the current matrix), and releasing it only
stores the parameters. Nothing is cooked; computing is right-click 「计算」 (webui/src/view/plan.ts underHandles says
what each kind shows). Nodes only declare handles; the viewer has one tool per kind, so a new paper that needs clicks
or a gizmo reuses a kind, and a new kind is written once for every node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .. import i18n
from ..availability import Cond


@dataclass(frozen=True)
class Kind:
    """Everything a kind of handle asks of its declaration, in one place (Handle.check reads only this): its stage, the
    roles it must bind to parameters and those it may, what each role's parameter is (its widget, or "float"), whether
    it works on an input of the node (`source`), and the class it is declared with when it has one of its own (Places,
    Poses: Handle itself refuses the kind, so the class's own rules cannot be stepped round)."""

    stage: str
    must: tuple[str, ...]
    may: tuple[str, ...]
    holds: str  # what the bound parameters hold
    widgets: dict[str, str]  # role -> the widget of its parameter ("float": a plain number parameter)
    # how many numbers an entry "frame:n1,n2,…" of a bound parameter holds (entries): the counts it may have, or with
    # `pairs` any even count of at least that many; () for a kind whose parameters are no such entries
    counts: tuple[int, ...] = ()
    pairs: int = 0
    # the place in an entry's numbers of a label (an index into the handle's labels, the first without labels), -1 none
    label_at: int = -1
    source: bool = False  # the handle works on one of the node's inputs, which it must name
    declared_by: str = ""  # the Handle subclass of this module the kind is declared with ("" Handle itself)
    # the stage needs data the node reads from its inputs (NodeDef.handle_data, carried in the status reply for the
    # node the page names): the kind says so, not the class a declaration happens to use
    wants_data: bool = False


HANDLE_KINDS: dict[str, Kind] = {
    # clicks on the picture: points to track, or prompts ("this", "not this") for a method that segments by clicks
    "points": Kind("2d", ("points",), (), "frame:x,y or frame:x,y,label (image pixels, top-left 0,0; label an integer index into the handle's labels)", {"points": "picks"}, counts=(2, 3), label_at=2),
    # boxes dragged on the picture (a subject to segment, a region to exclude)
    "box": Kind("2d", ("boxes",), (), "frame:x1,y1,x2,y2 (image pixels)", {"boxes": "picks"}, counts=(4,)),
    # a plane's four corners on one frame (a screen, a sign, a wall to track): dragged out as a box, each corner then
    # moved onto the plane's; one quad, a new one replaces it
    "corners": Kind("2d", ("corners",), (), "frame:x1,y1,x2,y2,x3,y3,x4,y4 (image pixels, corners in order around the quad)",
                    {"corners": "picks"}, counts=(8,)),
    # a click inside one of the people boxes of the node's input picks that person
    "person": Kind("2d", ("picks",), (), "frame:x,y (image pixels)", {"picks": "picks"}, source=True, counts=(2,)),
    # outlines drawn on the picture by hand (a garbage matte, a rough matte to guide a matting model): each entry is
    # one closed outline, as many as the node needs. Same string grammar as "corners", only with as many points as
    # were drawn; the viewer draws, hit-tests and removes them with the same polygon code
    "canvas": Kind("2d", ("shapes",), (), "frame:x1,y1,x2,y2,… (image pixels, a closed outline, three points or more)",
                   {"shapes": "canvas"}, pairs=3),
    # a stick figure drawn on the picture: one body pose per entry, FIGURE_JOINTS in order. One figure per frame, and
    # the frame is what is added: the parameter panel's 「添加帧」 puts a standing figure (T-pose, real human
    # proportions) on the current frame, or copies the previous frame's pose onto it; the picture is only for dragging
    # the joints. Figures are deliberately not dragged out as a box: two on one frame would replace each other, and a
    # dragged box gives whatever body proportions the hand drew. The four joints a body has but nobody draws (spine2,
    # spine3 and the two shoulders) are not in it: they sit on the lines between the drawn ones
    "figure": Kind("2d", ("figure",), (), "frame:x1,y1,…,x18,y18 (image pixels, FIGURE_JOINTS in order)", {"figure": "figure"},
                   counts=(36,)),
    # a move / rotate (/ scale) gizmo in the scene
    "transform": Kind("3d", ("translate", "rotate"), ("scale",), "vector parameters: cm, degrees XYZ, a factor",
                      {"translate": "vec3", "rotate": "vec3", "scale": "float"}, declared_by="Places"),
    # a skeleton in the scene, its pose corrected joint by joint when the handle edits a parameter: pick a joint, a
    # move / rotate / scale gizmo on it, its children follow (FK), left and right can be mirrored; without one only
    # shown. Declared as Poses (which input's skeleton); the status reply carries the skeleton (NodeDef.handle_data)
    "skeleton_pose": Kind("3d", (), ("pose",), "rows {joint, translate cm, rotate ° XYZ, scale}: local @ T·R·S in the joint's own axes",
                          {"pose": "skeleton_pose"}, source=True, declared_by="Poses", wants_data=True),
    # two skeletons side by side in the scene, edited in the view: the mapping of their body parts (click a joint on
    # one, then on the other), each side's reference pose corrected joint by joint (as "skeleton_pose": rows in the
    # joint's own axes) and which joints are left out of a solver (click a joint: it and what hangs below it). One side
    # may be a model's fixed skeleton, which has names, a hierarchy and body parts but no positions (drawn as a tree
    # only). Declared as RigPair; the status reply carries both skeletons (NodeDef.handle_data, the shape RigPair says)
    "rig_pair": Kind("3d", ("mapping",), ("src_pose", "dst_pose", "src_ignore", "dst_ignore", "ignore_rule", "auto_record",
                                         "src_scale", "dst_scale"),
                     "mapping: rows {part, src, dst} (joint names); poses: rows {joint, translate cm, rotate ° XYZ, scale}; "
                     "ignore: joint names; ignore_rule: a rule id; auto_record: JSON of the last automatic fills; "
                     "scale: a side's size factor (empty = automatic)",
                     {"mapping": "rig_map", "src_pose": "skeleton_pose", "dst_pose": "skeleton_pose",
                      "src_ignore": "joint_list", "dst_ignore": "joint_list", "ignore_rule": "choice",
                      "auto_record": "hidden_json", "src_scale": "float", "dst_scale": "float"},
                     source=True, declared_by="RigPair", wants_data=True),
}

# The widgets of the parameters a 2D handle works on in the view (picks, outlines, a stick figure): each such parameter
# gets its own 「在视图里点选」 (nodes/params.py pick_button), and the page offers it the pick-in-view row. Derived from
# the kinds, the one list (the catalogue carries it to the page: server/app.py `picked_in_view`)
PICKED_IN_VIEW: tuple[str, ...] = tuple(sorted({w for k in HANDLE_KINDS.values() if k.stage == "2d" for w in k.widgets.values()}))

# The 18 joints drawn on the stick figure, as (SMPL joint name, label id: its words handle.label.<id>). The order is the order of the coordinate
# pairs in a "figure" handle entry; the bone connections on the web side (webui/src/view/figure2d.ts FIGURE_BONES)
# use the same order. Labels name where the point is on the body (points are drawn, not bones), so 「左胯」 is not
# called 「左大腿」:
# data/joints.py part_label names a bone and this table names a point; the two are not interchangeable.
# The body has 22 joints (the first 22 SMPL joints of lab2shot_shared.smpl); the 4 omitted ones (spine2, spine3 and
# both shoulders) lie on the lines between drawn points and are filled in by the algorithm rather than drawn.
FIGURE_JOINTS: tuple[tuple[str, str], ...] = tuple((j, j) for j in (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
))


def figure_handle(param: str) -> "Handle":
    """The stick-figure handle bound to a node's parameter: every node that has one declares it the same way, with the
    same joints and the same labels."""
    return Handle("figure", {"figure": param}, labels=tuple(label for _, label in FIGURE_JOINTS))


@dataclass(frozen=True)
class Handle:
    kind: str  # one of HANDLE_KINDS
    params: dict[str, str]  # role -> the node's parameter it edits, e.g. {"points": "picks"}
    # the input the handle works on (e.g. "boxes" for "person"): the viewer shows its upstream result with the handle,
    # so the handle has something to work on before the node is computed
    source: str | None = None
    # only while it holds (nodes/applies.py: Param("mode").one_of("picked")); the status reply names the handles that
    # apply now (Graph.handles), the page never checks it itself
    when: Cond | None = None
    # "points": what a click means, the first by default: ids, their words handle.label.<id> (e.g. subject, exclude)
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kind = HANDLE_KINDS.get(self.kind)
        if kind is not None and kind.declared_by and not isinstance(self, globals()[kind.declared_by]):
            raise TypeError(f"a {self.kind!r} handle is declared as {kind.declared_by}(...), which holds its own rules")

    @property
    def ports(self) -> tuple[str, ...]:
        """The inputs whose data the handle works on: the stand-ins the status reply reads for it (engine/routing.py
        stand_ins) before the node has cooked."""
        return (self.source,) if self.source else ()

    @property
    def wants_data(self) -> bool:
        """Whether the stage needs data the node gives for this handle (Kind.wants_data)."""
        kind = HANDLE_KINDS.get(self.kind)
        return bool(kind and kind.wants_data)

    def check(self, node) -> None:
        """The declaration against its node type (NodeDef.__init_subclass__): a known kind, every role it must bind
        bound and none it does not know, each bound parameter one of the node's, its input one of the node's, and a
        node that says it gives the data the kind wants (NodeDef.handle_data). Raises TypeError naming what is wrong:
        a misspelt name would otherwise leave a handle that silently edits nothing."""
        from .base import NodeDef

        where = f"{node.__name__}: handle {self.kind!r}"
        if self.kind not in HANDLE_KINDS:
            raise TypeError(f"{where} is no kind of handle ({', '.join(HANDLE_KINDS)})")
        kind = HANDLE_KINDS[self.kind]
        if missing := [r for r in kind.must if r not in self.params]:
            raise TypeError(f"{where} does not bind the roles {missing}")
        if unknown := [r for r in self.params if r not in kind.must + kind.may]:
            raise TypeError(f"{where} binds roles it does not have: {unknown}")
        fields = node.Params.model_fields
        if absent := [p for p in self.params.values() if p not in fields]:
            raise TypeError(f"{where} edits {absent}, which are not parameters of the node")
        for role, param in self.params.items():
            extra = fields[param].json_schema_extra if isinstance(fields[param].json_schema_extra, dict) else {}
            want = kind.widgets.get(role)
            got = "float" if fields[param].annotation in (float, float | None) and not extra.get("widget") else extra.get("widget")
            if want and got != want:
                raise TypeError(f"{where} binds {role!r} to {param!r}, a {got or 'plain'} parameter, not a {want} one")
        if kind.source and not self.source:
            raise TypeError(f"{where} works on one of the node's inputs: name it (source=)")
        if self.source and self.source not in {p.name for p in node.inputs}:
            raise TypeError(f"{where} works on input {self.source!r}, which is not one of its inputs")
        if self.wants_data and node.handle_data.__func__ is NodeDef.handle_data.__func__:
            raise TypeError(f"{where} needs the node to give its data (NodeDef.handle_data)")

    def describe(self) -> dict:
        return {"kind": self.kind, "stage": HANDLE_KINDS[self.kind].stage, "params": self.params, "source": self.source,
                "labels": [i18n.lookup(f"handle.label.{x}") or x for x in self.labels]}


@dataclass(frozen=True, kw_only=True)
class Places(Handle):
    """The transform handle, and where the node places what it gives: one declaration
    of the parameters that move (cm, Y up), turn (degrees, X then Y then Z) and scale it (a factor; "" none). The
    handle edits them; the viewer shows what is there (the node's own result, or before the node has a current one its
    `source` input) placed by `placement()` of the current parameters, and the cook applies `matrix` of the same
    parameters, so the result lands where the display was. The two are the same matrix written twice (Python here,
    webui/src/model/places.ts placeMatrix for the viewer); `lab2shot check places` runs both on a set of parameters and
    compares them. A node declares it among its handles; NodeDef.places is that handle."""

    kind: str = field(default="transform", init=False)
    params: dict[str, str] = field(default_factory=dict, init=False)
    translate: str
    rotate: str
    scale: str = ""

    def __post_init__(self) -> None:
        roles = {"translate": self.translate, "rotate": self.rotate, **({"scale": self.scale} if self.scale else {})}
        object.__setattr__(self, "params", roles)

    def matrix(self, params: dict) -> "np.ndarray":
        """4x4, column vectors: scale, then rotate (XYZ), then translate: the placement the cook applies."""
        from lab2shot_shared.poses import trs_matrix

        return trs_matrix(params[self.translate], params[self.rotate], params[self.scale] if self.scale else 1.0)

    def placement(self) -> dict:
        """What the viewer previews by: the parameters, the order and rotation convention, the units."""
        return {"translate": self.translate, "rotate": self.rotate, "scale": self.scale or None,
                "order": "scale-rotate-translate", "rotation": "XYZ", "units": {"translate": "cm", "rotate": "°"}}


@dataclass(frozen=True, kw_only=True)
class Poses(Handle):
    """The skeleton pose handle: `source` is the input whose skeleton it shows, `skeleton` the parameter naming that
    skeleton's prim (None: the first) and `pose` the parameter it edits (kit/rig_map.py pose_param, rows of JointPose),
    or None for a skeleton only shown. What the stage draws (the joints, their hierarchy, the pose before and after the
    correction, mirror pairs) the node gives from its inputs (NodeDef.handle_data, kit/rig_map.py skeleton_handle),
    carried in the status reply for the displayed node (server/packets.py status)."""

    kind: str = field(default="skeleton_pose", init=False)
    params: dict[str, str] = field(default_factory=dict, init=False)
    pose: str | None
    skeleton: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise TypeError("a skeleton pose handle names the input whose skeleton it shows (source=)")
        object.__setattr__(self, "params", {"pose": self.pose} if self.pose else {})

    def check(self, node) -> None:
        super().check(node)
        fields = node.Params.model_fields
        if self.skeleton and self.skeleton not in fields:
            raise TypeError(f"{node.__name__}: skeleton pose handle reads skeleton {self.skeleton!r}, not a parameter of the node")

    def describe(self) -> dict:
        return {**super().describe(), "skeleton": self.skeleton, "readonly": not self.pose}


@dataclass(frozen=True, kw_only=True)
class RigPair(Handle):
    """The two-skeleton handle (kind "rig_pair"): `src` is the input that drives (its skeleton chosen by the parameter
    `src_skeleton`, None the first), `dst` the input driven, or None when the other side is the node's own model
    skeleton (fixed: joint names, hierarchy and body parts only). The parameters it edits are named by role:
    `mapping` always (kit/rig_map.py rig_map_param, rows of PartMap); each side's pose (JointPose rows), ignored joints
    (joint names), the ignore rule (a choice), the record of the automatic fills (JSON, not in the panel) and each
    side's size factor (a number, empty = automatic: the stage draws that side at that size) where the node has them. What the stage draws for both sides the node gives from its inputs (NodeDef.handle_data: one entry,
    {"src", "dst", "parts_table", "auto_mapping"[, "rules", "default_rule"]}, kit/rig_map.py rig_pair_data)."""

    kind: str = field(default="rig_pair", init=False)
    params: dict[str, str] = field(default_factory=dict, init=False)
    source: str | None = field(default=None, init=False)
    src: str
    dst: str | None
    mapping: str
    src_skeleton: str | None = None
    dst_skeleton: str | None = None
    src_pose: str | None = None
    dst_pose: str | None = None
    src_ignore: str | None = None
    dst_ignore: str | None = None
    ignore_rule: str | None = None
    auto_record: str | None = None
    src_scale: str | None = None
    dst_scale: str | None = None

    def __post_init__(self) -> None:
        roles = {"mapping": self.mapping, "src_pose": self.src_pose, "dst_pose": self.dst_pose,
                 "src_ignore": self.src_ignore, "dst_ignore": self.dst_ignore, "ignore_rule": self.ignore_rule,
                 "auto_record": self.auto_record, "src_scale": self.src_scale, "dst_scale": self.dst_scale}
        object.__setattr__(self, "params", {r: p for r, p in roles.items() if p})
        object.__setattr__(self, "source", self.src)

    @property
    def ports(self) -> tuple[str, ...]:
        return (self.src, self.dst) if self.dst else (self.src,)

    def check(self, node) -> None:
        super().check(node)
        where = f"{node.__name__}: rig pair handle"
        if self.dst and self.dst not in {p.name for p in node.inputs}:
            raise TypeError(f"{where} drives input {self.dst!r}, which is not one of its inputs")
        fields = node.Params.model_fields
        for name in (self.src_skeleton, self.dst_skeleton):
            if name and name not in fields:
                raise TypeError(f"{where} reads skeleton {name!r}, not a parameter of the node")
        if not self.dst and (self.dst_pose or self.dst_ignore or self.dst_skeleton or self.dst_scale):
            raise TypeError(f"{where}: a model's fixed skeleton (dst=None) has no pose, ignored joints, size or skeleton choice")
        if bool(self.src_scale) != bool(self.dst_scale) or (self.src_scale and not self.auto_record):
            raise TypeError(f"{where}: the two sizes go together, with the record of the automatic fills")
        if bool(self.src_ignore or self.dst_ignore) != bool(self.ignore_rule):
            raise TypeError(f"{where}: ignored joints and the ignore rule go together")

    def describe(self) -> dict:
        return {**super().describe(), "src": self.src, "dst": self.dst, "src_skeleton": self.src_skeleton,
                "dst_skeleton": self.dst_skeleton}


# ---------------------------------------------------------------- what a handle saved, parsed


def _bound(node_type, param: str) -> Handle:
    handle = next((h for h in getattr(node_type, "handles", ()) if param in h.params.values()), None)
    if handle is None:
        raise TypeError(f"{getattr(node_type, '__name__', node_type)} has no handle bound to {param!r}")
    return handle


def kind_of(node_type, param: str) -> str:
    """The kind of the handle bound to `param` on `node_type` (its handles): the one thing that says how the entries
    saved in that parameter read."""
    return _bound(node_type, param).kind


def entries(node_type, param: str, saved: list[str]) -> tuple[list[tuple[int, list[float]]], list[str]]:
    """The one reading of a handle's saved entries "frame:n1,n2,…" (every 2D kind), by the handle bound to `param`
    (kind_of): (the entries that read, as (frame, numbers), in their order; those that do not, as saved). An entry
    reads when its frame is a whole number, its numbers are numbers and their count is one that kind declares
    (Kind.counts / Kind.pairs), and a label (Kind.label_at) is a whole number that indexes the handle's labels. An
    entry that does not read is left out, and a cook says so with say_bad_entries — a parameter edited or pasted by
    hand never fails a cook with a ValueError, whichever kind of handle wrote it."""
    handle = _bound(node_type, param)
    k = HANDLE_KINDS[handle.kind]
    good, bad = [], []
    for entry in saved or []:
        try:
            frame, rest = str(entry).split(":")
            values = [float(v) for v in rest.split(",")]
            n = len(values)
            if not (n in k.counts or (k.pairs and n % 2 == 0 and n >= 2 * k.pairs)) or not all(map(np.isfinite, values)):
                raise ValueError(entry)
            if 0 <= k.label_at < n and not (values[k.label_at].is_integer()
                                            and 0 <= values[k.label_at] < max(1, len(handle.labels))):
                raise ValueError(entry)
            good.append((int(frame), values))
        except ValueError:
            bad.append(str(entry))
    return good, bad


def say_bad_entries(ctx, param: str) -> None:
    """W-HANDLE-BADPICK for the entries of `param` (as the node's handle bound to it reads them) that `entries`
    leaves out."""
    saved = ctx.params[param]
    bad = entries(ctx.node_type, param, saved)[1]
    if bad:
        ctx.say("W-HANDLE-BADPICK", param=param, count=len(bad), grammar=HANDLE_KINDS[kind_of(ctx.node_type, param)].holds,
                entries=i18n.Both.of(lambda: i18n.separator().join(bad[:4]) + (i18n.t("list.etc") if len(bad) > 4 else "")))


class Pick(NamedTuple):
    """One viewer click of a "points" / "person" handle."""

    frame: int
    x: float  # image pixels, top-left corner = 0,0
    y: float
    label: int = 0  # which of the handle's labels the click is (subject / exclude), 0 without labels


def parse_picks(node_type, param: str, picks: list[str]) -> list[Pick]:
    """Viewer clicks saved in `param` by a "points" or "person" handle, by that handle's grammar (entries): "frame:x,y",
    and for points "frame:x,y,label" too (a label one of the handle's)."""
    return [Pick(f, v[0], v[1], int(v[2]) if len(v) == 3 else 0) for f, v in entries(node_type, param, picks)[0]]


def parse_corners(node_type, param: str, saved: list[str]) -> tuple[int, list[tuple[float, float]]] | None:
    """A plane's corners saved by a "corners" handle: "frame:x1,y1,...,x4,y4" -> (frame, four (x, y) in order around
    the quad, image pixels from the top-left corner); None when there is none that reads (entries). The first entry
    that reads counts (a quad drawn again replaces it)."""
    got = entries(node_type, param, saved)[0]
    if not got:
        return None
    frame, v = got[0]
    return frame, [(v[2 * k], v[2 * k + 1]) for k in range(4)]


def parse_shapes(node_type, param: str, saved: list[str]) -> list[list[tuple[float, float]]]:
    """Outlines drawn by a "canvas" handle: "frame:x1,y1,x2,y2,…" -> each shape's points (image pixels from the
    top-left corner, a closed outline; three points or more, entries). The frame says which frame it was drawn on; what
    a shape counts for is the node's own rule (「手画遮罩」: the whole shot)."""
    return [[(v[2 * k], v[2 * k + 1]) for k in range(len(v) // 2)] for _, v in entries(node_type, param, saved)[0]]


def parse_figures(node_type, param: str, saved: list[str]) -> list[tuple[int, list[tuple[float, float]]]]:
    """Stick figures drawn by a "figure" handle: "frame:x1,y1,…" -> (frame, the joints in FIGURE_JOINTS order,
    image pixels from the top-left corner), in frame order; a figure holds every joint of the table (entries)."""
    got = [(f, [(v[2 * k], v[2 * k + 1]) for k in range(len(FIGURE_JOINTS))])
           for f, v in entries(node_type, param, saved)[0]]
    return sorted(got, key=lambda row: row[0])
