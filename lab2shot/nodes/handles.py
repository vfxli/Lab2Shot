"""Viewer handles: how a node is worked on in the viewer.

A handle is bound to parameters, as in Houdini: when the node is the active one the viewer shows its handles in their
stage, and using a handle edits those parameters (undoable, cooked again like any edit). Nodes only declare
handles; the viewer has one tool per kind, so a new paper that needs clicks or a gizmo reuses a kind, and a new
kind is written once for every node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..availability import Cond
from ..errors import Invalid
from ..messages import Msg

if TYPE_CHECKING:
    import numpy as np

# kind -> (stage, the roles it must bind to parameters, roles it may bind, what the bound parameters hold)
HANDLE_KINDS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str]] = {
    # clicks on the picture: points to track, or prompts ("this", "not this") for a method that segments by clicks
    "points": ("2d", ("points",), (), "frame:x,y or frame:x,y,label (image pixels, top-left 0,0)"),
    # boxes dragged on the picture (a subject to segment, a region to exclude)
    "box": ("2d", ("boxes",), (), "frame:x1,y1,x2,y2 (image pixels)"),
    # a plane's four corners on one frame (a screen, a sign, a wall to track): dragged out as a box, each corner then
    # moved onto the plane's; one quad, a new one replaces it
    "corners": ("2d", ("corners",), (), "frame:x1,y1,x2,y2,x3,y3,x4,y4 (image pixels, corners in order around the quad)"),
    # a click inside one of the people boxes of the node's input picks that person
    "person": ("2d", ("picks",), (), "frame:x,y (image pixels)"),
    # outlines drawn on the picture by hand (a garbage matte, a rough matte to guide a matting model): each entry is
    # one closed outline, as many as the node needs. Same string grammar as "corners", only with as many points as
    # were drawn; the viewer draws, hit-tests and removes them with the same polygon code
    "canvas": ("2d", ("shapes",), (), "frame:x1,y1,x2,y2,… (image pixels, a closed outline, three points or more)"),
    # a stick figure drawn on the picture: one body pose per entry, FIGURE_JOINTS in order. One figure per frame, and
    # the frame is what is added: the parameter panel's 「添加帧」 puts a standing figure (T-pose, real human
    # proportions) on the current frame, or copies the previous frame's pose onto it; the picture is only for dragging
    # the joints. Figures are deliberately not dragged out as a box: two on one frame would replace each other, and a
    # dragged box gives whatever body proportions the hand drew. The four joints a body has but nobody draws (spine2,
    # spine3 and the two shoulders) are not in it: they sit on the lines between the drawn ones
    "figure": ("2d", ("figure",), (), "frame:x1,y1,…,x18,y18 (image pixels, FIGURE_JOINTS in order)"),
    # a move / rotate (/ scale) gizmo in the scene
    "transform": ("3d", ("translate", "rotate"), ("scale",), "vector parameters: cm, degrees XYZ, a factor"),
}

# The 18 joints drawn on the stick figure, as (SMPL joint name, UI label). The order is the order of the coordinate
# pairs in a "figure" handle entry; the bone connections on the web side (view/handles2d.ts FIGURE_BONES) use the same
# order. Labels name where the point is on the body (points are drawn, not bones), so 「左胯」 is not called 「左大腿」:
# data/joints.py part_label names a bone and this table names a point; the two are not interchangeable.
# The body has 22 joints (the first 22 SMPL joints of lab2shot_shared.smpl); the 4 omitted ones (spine2, spine3 and
# both shoulders) lie on the lines between drawn points and are filled in by the algorithm rather than drawn.
FIGURE_JOINTS: tuple[tuple[str, str], ...] = (
    ("pelvis", "髋"), ("left_hip", "左胯"), ("right_hip", "右胯"), ("spine1", "腰"),
    ("left_knee", "左膝"), ("right_knee", "右膝"), ("left_ankle", "左踝"), ("right_ankle", "右踝"),
    ("left_foot", "左脚"), ("right_foot", "右脚"), ("neck", "颈"),
    ("left_collar", "左肩"), ("right_collar", "右肩"), ("head", "头"),
    ("left_elbow", "左肘"), ("right_elbow", "右肘"), ("left_wrist", "左腕"), ("right_wrist", "右腕"),
)


def figure_handle(param: str) -> "Handle":
    """The stick-figure handle bound to a node's parameter: every node that has one declares it the same way, with the
    same joints and the same labels."""
    return Handle("figure", {"figure": param}, labels=tuple(label for _, label in FIGURE_JOINTS))


@dataclass(frozen=True)
class Handle:
    kind: str  # one of HANDLE_KINDS
    params: dict[str, str]  # role -> the node's parameter it edits, e.g. {"points": "picks"}
    source: str | None = None  # the input the handle works on (e.g. "boxes" for "person")
    # only while it holds (nodes/applies.py: Param("mode").one_of("picked")); the status reply names the handles that
    # apply now (Graph.handles), the page never checks it itself
    when: Cond | None = None
    labels: tuple[str, ...] = ()  # "points": what a click means, the first by default, e.g. ("主体", "排除")

    def __post_init__(self) -> None:
        if self.kind == "transform" and not isinstance(self, Places):
            raise TypeError("a transform handle is the node's placement: declare it as Places(translate=, rotate=, scale=)")

    def describe(self) -> dict:
        return {"kind": self.kind, "stage": HANDLE_KINDS[self.kind][0], "params": self.params, "source": self.source,
                "labels": list(self.labels)}


@dataclass(frozen=True, kw_only=True)
class Places(Handle):
    """The transform handle, and where the node places what it gives: one declaration
    of the parameters that move (cm, Y up), turn (degrees, X then Y then Z) and scale it (a factor; "" none). The
    handle edits them, the viewer applies `placement()` to the data it already shows while the handle or a parameter
    is dragged, and the cook applies `matrix` of the same parameters, so the result lands where the preview was. A node
    declares it among its handles; NodeDef.places is that handle."""

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
        from ..data.scene import trs_matrix

        return trs_matrix(params[self.translate], params[self.rotate], params[self.scale] if self.scale else 1.0)

    def placement(self) -> dict:
        """What the viewer previews by: the parameters, the order and rotation convention, the units."""
        return {"translate": self.translate, "rotate": self.rotate, "scale": self.scale or None,
                "order": "scale-rotate-translate", "rotation": "XYZ", "units": {"translate": "cm", "rotate": "°"}}


# ---------------------------------------------------------------- what a handle saved, parsed


def parse_picks(picks: list[str]) -> list[tuple[int, float, float]]:
    """Viewer clicks saved by a "picks" parameter: "frame:x,y" (image pixels, top-left corner = 0,0)."""
    out = []
    for pick in picks:
        frame, xy = pick.split(":")
        x, y = (float(v) for v in xy.split(","))
        out.append((int(frame), x, y))
    return out


def parse_corners(entries: list[str]) -> tuple[int, list[tuple[float, float]]] | None:
    """A plane's corners saved by a "corners" handle: "frame:x1,y1,...,x4,y4" -> (frame, four (x, y) in order around
    the quad, image pixels from the top-left corner); None when there is none. The first entry counts (a quad drawn
    again replaces it)."""
    if not entries:
        return None
    frame, xy = entries[0].split(":")
    v = [float(a) for a in xy.split(",")]
    if len(v) != 8:
        raise Invalid(Msg("E-TRACKS-CORNERS", count=len(v), entry=entries[0]))
    return int(frame), [(v[2 * k], v[2 * k + 1]) for k in range(4)]


def parse_shapes(entries: list[str]) -> list[list[tuple[float, float]]]:
    """Outlines drawn by a "canvas" handle: "frame:x1,y1,x2,y2,…" -> each shape's points (image pixels from the
    top-left corner, a closed outline). The frame says which frame it was drawn on; what a shape counts for is the
    node's own rule (「手画遮罩」: the whole shot). Shapes of fewer than three points cover no pixel and are left out."""
    out = []
    for entry in entries:
        _, xy = entry.split(":")
        v = [float(a) for a in xy.split(",")]
        if len(v) % 2:
            raise Invalid(Msg("E-ROTO-SHAPE", count=len(v), entry=entry))
        if len(v) >= 6:
            out.append([(v[2 * k], v[2 * k + 1]) for k in range(len(v) // 2)])
    return out


def parse_figures(entries: list[str]) -> list[tuple[int, list[tuple[float, float]]]]:
    """Stick figures drawn by a "figure" handle: "frame:x1,y1,…" -> (frame, the joints in FIGURE_JOINTS order,
    image pixels from the top-left corner), in frame order. A figure holds every joint of the table, so an entry
    with another number of points is refused rather than quietly padded."""
    out = []
    for entry in entries:
        frame, xy = entry.split(":")
        v = [float(a) for a in xy.split(",")]
        if len(v) != 2 * len(FIGURE_JOINTS):
            raise Invalid(Msg("E-FIGURE-JOINTS", count=len(v) // 2, want=len(FIGURE_JOINTS), entry=entry))
        out.append((int(frame), [(v[2 * k], v[2 * k + 1]) for k in range(len(FIGURE_JOINTS))]))
    return sorted(out, key=lambda row: row[0])
