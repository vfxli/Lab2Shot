"""Usage checks (质检): what a node expects of an input beyond its type.

A type determines what can connect. Some connections cook without error yet are probably not what the artist
intended, or silently rely on an assumption: a detection of five people wired directly into a per-person solver, a
camera from another shot, a relative depth interpreted in centimetres. A node declares its expectations of an input on
the port (Port(expects=...)); a single engine function (engine/lint.py) checks every wire against them using what is
known of the data. Before the upstream node is cooked, that is the graph's own information (Info: frames, whether
it is a still); afterwards, it is the packet's description (meta: the people in boxes, a depth's scale, the picture size).

A failed expectation is a warning, never an error: the node is marked yellow, the page lists the warning when a cook is
submitted, and the cook reports it when the node starts. When inserting one node in front of the input resolves it
(`fix`), the parameter panel offers the insertion as a single action, and the node menu lists that node first for a
wire drawn out of the input.

The exceptions are preconditions known before cooking that no cook can satisfy (FrameCount: a temporal model given
fewer frames than it requires, or a method given more than it takes; DistinctNames: two items of one name; OneValue: a
condition that varies between frames). Their messages use B- codes, and the node is refused before cooking, like a
node whose file is missing (engine/evaluation.py plan); the message states what was found and the corrective action.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from ..messages import Msg
from .base import typed_list
from ..data.windows import same_framing



@dataclass(frozen=True)
class Seen:
    """What is known of the data that one wire brings into an input."""

    node: str  # the upstream node
    source: str  # the label used to name the data in warnings
    type: str  # the data type carried by the wire
    frames: tuple[int, ...] = ()  # frames covered; () when not yet known or timeless (a lens, a camera with no frames)
    still: bool = False  # a single picture for the whole shot (a photo, an HDRI, an ST-map)
    meta: dict | None = None  # the packet's description once the upstream node is cooked; None before that
    # shot properties (data/contracts.py SHOT_KEYS): from the packet once cooked, otherwise what the graph derives from
    # the nodes' parameters (engine/evaluation.py Evaluation.shot); keys not yet known are absent
    shot: Mapping = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int] | None:
        """The picture size taken from the packet. A node's Info holds a single size for all outputs, which does not
        necessarily apply to each output (a light probe's HDRI differs from the plate's size)."""
        m = self.meta or {}
        return (int(m["width"]), int(m["height"])) if m.get("width") and m.get("height") else None


@dataclass(frozen=True)
class Checked:
    """The node as seen by the usage checks: its label, the parameters it cooks with, and the data on each input."""

    label: str
    params: dict
    inputs: dict[str, tuple[Seen, ...]]

    def first(self, port: str) -> Seen | None:
        got = self.inputs.get(port) or ()
        return got[0] if got else None


@dataclass(frozen=True)
class Expect:
    """One expectation of an input. check() returns a message describing what is probably wrong with the data `got`
    (a W- code of the catalogue, a B- code for a precondition), or None when nothing is wrong or the data is not yet
    known. The message is shown on the node itself, so it names the data rather than the node."""

    fix: ClassVar[str] = ""  # node type that resolves the issue when inserted between the input and its source
    per_wire: ClassVar[bool] = True  # True: judges each wire independently, so `fix` is also offered by the node menu
    # for a wire drawn out of the input (Port.describe inserts). False: judges the input's wires together
    # (DistinctNames); the fix then only answers a reported check, never an empty input

    def check(self, got: Seen, node: Checked) -> Msg | None:
        raise NotImplementedError


@dataclass(frozen=True)
class EachPerson(Expect):
    """The node runs once for every person in the boxes (one character, pair of hands or matte object each).
    Unselected boxes holding several people are probably unintended: a detection finds everyone, including passers-by
    and tracks broken by a cut. A single person is accepted, as are people selected in 「选人」, even if all are selected.

    「选人」 outputs one 人物框 (the selected people, with `chosen` set), so inserting it between a detection and this
    port resolves the warning; it is therefore the `fix`. The check judges each wire on its own, so the node menu also
    lists 「选人」 first for a wire drawn out of such an input (Port.describe inserts)."""

    fix: ClassVar[str] = "core.select_people"

    def check(self, got: Seen, node: Checked) -> Msg | None:
        people = (got.meta or {}).get("people") or ()
        if len(people) <= 1 or (got.meta or {}).get("chosen"):
            return None
        return Msg("W-EXPECT-SEVERALPEOPLE", source=got.source, count=len(people))


@dataclass(frozen=True)
class SameShot(Expect):
    """The data belongs to the shot in input `of` (the plate; the first wire when `of` is this multi-wire input): it
    covers the plate's frames and has the plate's picture size. For a camera, a lens or a value measured on a picture
    (such as a focal length), only the aspect ratio is compared, since focal lengths are converted using the width. A
    still (one picture for the whole shot), a camera without frames and a single value for the whole shot apply to
    every frame; per-frame values apply between their frames, as a camera does."""

    of: str = "image"
    frames: bool = True
    size: bool = True

    def check(self, got: Seen, node: Checked) -> Msg | None:
        plate = node.first(self.of)
        if plate is None or plate.node == got.node:
            return None
        camera = got.type.startswith(("scene", "value"))
        return ((self.frames and _frames(got, plate, camera))
                or (self.size and (_proportions(got, plate) if camera else _size(got, plate))) or None)


def _span(frames: tuple[int, ...]) -> str:
    return f"{frames[0]}–{frames[-1]}" if len(frames) > 1 else f"{frames[0]}"


def _frames(got: Seen, plate: Seen, camera: bool) -> Msg | None:
    """A camera applies between its keys (USD interpolates), so it covers its first to its last key. A 2D result
    covers only the frames it contains."""
    if got.still or plate.still or len(got.frames) < (2 if camera else 1) or not plate.frames:
        return None
    have = set(got.frames)
    lacking = [f for f in plate.frames if not (got.frames[0] <= f <= got.frames[-1] if camera else f in have)]
    if not lacking:
        return None
    return Msg("W-EXPECT-CAMERAFRAMES" if camera else "W-EXPECT-FRAMES", source=got.source, span=_span(got.frames),
               plate=plate.source, count=len(lacking), first=lacking[0])


def _size(got: Seen, plate: Seen) -> Msg | None:
    a, b = got.size, plate.size
    if a is None or b is None or a == b:
        return None
    return Msg("W-EXPECT-SIZE", source=got.source, width=a[0], height=a[1], plate=plate.source, plate_width=b[0], plate_height=b[1])


def _proportions(got: Seen, plate: Seen) -> Msg | None:
    a, b = got.size, plate.size
    if a is None or b is None or same_framing(a, b):
        return None
    return Msg("W-EXPECT-ASPECT", source=got.source, width=a[0], height=a[1], aspect=a[0] / a[1], plate=plate.source,
               plate_width=b[0], plate_height=b[1], plate_aspect=b[0] / b[1])


@dataclass(frozen=True)
class OwnCamera(Expect):
    """A relative depth (correct up to one unknown scale factor) matches only the camera from the same solve; with a
    camera from elsewhere, all positions are uniformly mis-scaled."""

    camera: str = "camera"

    def check(self, got: Seen, node: Checked) -> Msg | None:
        cam = node.first(self.camera)
        if cam is None or (got.meta or {}).get("scale") != "relative" or cam.node == got.node:
            return None
        return Msg("W-EXPECT-OWNCAMERA", source=got.source, camera=cam.source)


SCALE_LABELS = {"relative": "相对尺度", "affine": "只知远近"}


@dataclass(frozen=True)
class Metric(Expect):
    """The node interprets the depth values as centimetres (consistent with the distances in its parameters)."""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        scale = (got.meta or {}).get("scale")
        if scale in (None, "metric"):
            return None
        return Msg("W-EXPECT-NOTMETRIC", source=got.source, scale=SCALE_LABELS.get(scale, scale))


@dataclass(frozen=True)
class NotDisparity(Expect):
    """The incoming image declares itself as disparity (「尺度」 is disparity in the packet meta), whereas this node
    requires depth.

    The check is not based on the type, since the meaning of a float channel is defined by its consumer; it relies on
    the data's own declaration. Unprojecting disparity as depth yields incorrect geometry that is not evident from the
    values, so a warning is issued."""

    fix: ClassVar[str] = "core.depth_align"

    def check(self, got: Seen, node: Checked) -> Msg | None:
        if (got.meta or {}).get("scale") != "disparity":
            return None
        return Msg("W-DEPTH-DISPARITY", source=got.source)


@dataclass(frozen=True)
class NotAlready(Expect):
    """The node converts the data to the value of its parameter `param`, but the data already has that value (its
    meta holds the same value under the same name), so it passes through unchanged."""

    param: str
    labels: tuple[tuple[str, str], ...] = ()  # value -> display label

    def check(self, got: Seen, node: Checked) -> Msg | None:
        value = (got.meta or {}).get(self.param)
        if value is None or value != node.params.get(self.param):
            return None
        return Msg("W-EXPECT-ALREADY", source=got.source, value=dict(self.labels).get(value, value))


@dataclass(frozen=True)
class KnownPeople(Expect):
    """The person numbers entered in parameter `param` (such as 1,3) must exist in the boxes."""

    param: str = "ids"

    def check(self, got: Seen, node: Checked) -> Msg | None:
        people = (got.meta or {}).get("people")
        typed = typed_list(node.params.get(self.param) or "")
        if people is None or not typed:
            return None
        unknown = [t for t in typed if not t.isdigit() or int(t) not in people]
        if not unknown:
            return None
        return Msg("W-EXPECT-UNKNOWNPEOPLE", source=got.source, unknown=unknown, people=list(people))


HDR_FILES = (".exr", ".hdr")


@dataclass(frozen=True)
class HighDynamicRange(Expect):
    """The node interprets the file's values as light (an HDRI); an 8-bit picture's 0–255 values would be misread as
    radiance."""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        files = (got.meta or {}).get("files") or {}
        suffix = Path(str(next(iter(files.values())))).suffix.lower() if files else ""
        if not suffix or suffix in HDR_FILES:
            return None
        return Msg("W-EXPECT-NOTHDR", source=got.source, format=suffix[1:].upper())


@dataclass(frozen=True)
class FrameCount(Expect):
    """The number of input frames the method works with. `least`: the minimum it requires (a temporal network's
    window, two frames to track between, COLMAP's minimum number of views). `most`: the maximum it accepts (an
    environment probe: one frame). Excess frames are not silently reduced to the first one; the node is refused before
    submission, and a one-click fix inserts 「FrameHold」 so that the user selects the frame (frame selection and solving
    are kept in separate nodes). Both limits are preconditions of the method that no cook can satisfy, so both use B-
    codes and are refused before cooking. `step`: the name of a parameter that takes every n-th frame (隔帧); the count
    is taken after applying it. A still counts as one frame."""

    least: int = 0
    most: int | None = None
    step: str = ""

    @property
    def fix(self) -> str:  # type: ignore[override]
        """Only an excess of frames has a one-click fix; no inserted node can supply missing frames."""
        return "core.frame_hold" if self.most is not None else ""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        if self.most is not None and not got.still and len(got.frames) > self.most:
            return Msg("B-EXPECT-MANYFRAMES", node=node.label, most=self.most, source=got.source,
                       count=len(got.frames))
        if not self.least:
            return None
        if got.still:
            return Msg("B-EXPECT-FEWFRAMES", node=node.label, need=self.least, source=got.source, count=1)
        if not got.frames:  # frames not yet known
            return None
        step = int(node.params.get(self.step) or 1) if self.step else 1
        used = len(got.frames[::step])
        if used >= self.least:
            return None
        if step > 1:
            return Msg("B-EXPECT-FEWFRAMESSTEP", node=node.label, need=self.least, source=got.source, count=len(got.frames),
                       step=step, used=used)
        return Msg("B-EXPECT-FEWFRAMES", node=node.label, need=self.least, source=got.source, count=len(got.frames))


@dataclass(frozen=True)
class DistinctNames(Expect):
    """Every item wired into this input must have a distinct name. Two cameras both named /shot/camera wired into one
    output-settings node would produce two objects of the same name in one file, and nothing is renamed implicitly; the
    artist names them with 「命名」, the one-click fix of this check. The check judges the input's wires together, so
    the fix is not offered by the node menu for a wire drawn out of an empty input (per_wire=False: such a wire is a
    scene to write). The check runs once the data exists, since names come from the data itself (a scene's groups, a
    list's items); before that there is nothing to compare."""

    fix: ClassVar[str] = "core.name_item"
    per_wire: ClassVar[bool] = False

    def check(self, got: Seen, node: Checked) -> Msg | None:
        from ..data.types import type_label

        mine = _named(got)
        if not mine:
            return None
        seen = next((list(s) for s in node.inputs.values() if any(x is got for x in s)), [])
        before = seen[: next((k for k, s in enumerate(seen) if s is got), 0)]  # reported once, on the later wire
        before = [s for s in before if s.node != got.node]  # outputs of one node form a single item (an import's camera
        # and models share its folder), so they never count as duplicate names
        same = sorted({n for s in before for n in _named(s) if n in mine})
        if not same:
            return None
        return Msg("B-NAME-SAME", kind=type_label(got.type), names=same)


def _named(got: Seen) -> list[str]:
    """The names of the items the data carries: a list's items, or the elements of a multi-item type
    (data/items.py: a scene's groups under /shot, the people of 人物框)."""
    from ..data.items import kind_of
    from ..data.types import element_of, is_list

    m = got.meta or {}
    if is_list(got.type):
        return [str(i.get("name", "")) for i in m.get("items") or ()]
    kind = kind_of(element_of(got.type))
    return [str(n) for n in kind.names(m)] if kind is not None and m else []


@dataclass(frozen=True)
class OneValue(Expect):
    """A single value for the whole shot rather than one per frame: 「切换」 selects one branch for the whole shot, so a
    condition that varies between frames cannot decide. Refused before cooking (a B- code)."""

    def check(self, got: Seen, node: Checked) -> Msg | None:
        values = (got.meta or {}).get("values")
        if not values or all(v == values[0] for v in values):
            return None
        return Msg("B-SWITCH-PERFRAME", node=node.label, source=got.source)
