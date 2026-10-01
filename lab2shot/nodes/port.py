"""A port: what a node takes on one input or gives on one output.

`Port` is the most frequently written item in node declarations and does not depend on NodeDef: type, unit,
applicability and shape contract are all declared on this one dataclass. Node authors import it from `nodes.base`
(the declaration facade).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..availability import Cond
from ..data.types import DATA_TYPES, channels_of, element_of, type_label

if TYPE_CHECKING:
    from ..data.contracts import Shape
    from .expects import Expect


EITHER = "either"  # Port.data on an input that takes a picture or values alike (it passes the kind on, or writes it as it is)


@dataclass(frozen=True)
class Port:
    name: str
    type: str
    label: str
    optional: bool = False
    multi: bool = False  # accepts several wires (e.g. USD pack)
    # an output that carries the type of what is wired into an input ("input:src": 「STMap」 gives back the kind
    # of data it warps); until that input is wired it stays open between the alternatives of `type`
    type_from: str = ""
    # an image output whose pixels are values, not a picture (normals, motion vectors, positions, canonical
    # coordinates): what the node says it gives, the one answer to "picture or values" (engine/graph.py output_data).
    # The conditions on a picture (WiredPicture: 「输出色彩空间」, 「EXR 位深」) read it before anything is cooked, the
    # engine checks what the node wrote against it and marks the packet with it (payloads.is_data reads the mark).
    # None: by its type — one or two channels are values, three or four a picture, and a port that follows an input
    # (type_from: crop, warp, merge) gives what comes in. True / False only where the type cannot tell (three or four
    # channels of values; a reader whose file says, 读取序列's layers).
    # On an input the same field says what it takes: False a picture only (a model's 「RGB」, kit/ports.py rgb_port),
    # True values only (normals, motion vectors: kit/ports.py values_port), EITHER both (crop, merge, an output
    # setting). A wire bringing the other kind is refused (engine/graph.py wire_problem, B-WIRE-DATAKIND) and so is a
    # packet of the other kind at the cook (Engine, the same check): a normal map fed to a model as a photograph would be
    # quantized to 8-bit sRGB on the quiet, a photograph fed to a warp as motion vectors moves pixels by colours. An
    # input of three or four channels, or of any image, must say one of the three (lab2shot check nodes); one or two
    # channels are values by their type.
    data: bool | str | None = None

    # an input: what the node means by it beyond its type (nodes/expects.py), checked by engine/lint.py
    expects: tuple[Expect, ...] = ()
    # a value's unit (nodes/values.py): what a value output gives ("param:<name>": the unit its node's parameter says,
    # as the constant nodes do) or a parameter's input takes; "" none or not a value
    unit: str = ""
    # an input that takes a plain number only, never one with a unit (「切换」's 「走哪一路」: a way's number, not 2 mm):
    # a wire bringing a unit is refused (engine/graph.py wire_problem, B-WIRE-UNIT)
    plain: bool = False
    # an input: the node type the node menu offers first for a wire drawn out of it (a parameter's input: the constant
    # node of its type), besides the one its expectations insert (`fix`)
    recommend: str = ""
    # an optional input: what connecting it does, in the port's tooltip (数据信息)
    help: str = ""
    # When the port applies (e.g. once 「图像」 is wired the rgba ports do not, and vice versa). A port is a control and
    # follows the parameter mechanism: when the condition does not hold it is greyed with the reason and keeps its
    # position; the server resolves it for the web page, and wiring is refused by the same condition. Both inputs and
    # outputs may use it: engine/graph.py ports() reads `Resolved.out_inactive`, the page greys the row and no wire can
    # be drawn from it, and engine/graph.py wire_problem returns `B-WIRE-OUTINACTIVE`. It is available / unavailable,
    # never hidden / shown. Not for a value that is merely trivial: a lens without distortion is still a lens, so the
    # lens calibrations' 「镜头内参」 stays usable with a pinhole (its type never changes with a choice).
    # This differs from `when`: `applies` means the port is currently unusable but stays in place; `when` (below)
    # means the port is currently absent, and an existing wire from it switches to waiting for a selection.
    applies: Cond | None = None
    # an output that is there only while this node's own parameters say so (Param("camera").set(): 「导入 USD」 gives 相机
    # once a camera is chosen; Param("matte").one_of(True): Sapiens2 gives 前景 while 精细抠像 runs the matting model;
    # nodes/applies.py output_ports). A wire from it while it is not there waits (applies.waiting_port), with `waits`:
    # what brings it ("选一台相机", "打开「精细抠像」")
    when: Cond | None = None
    waits: str = ""
    # an output's 3D data beyond its type's kind, always (「烘焙成模型」: every 模型 it gives deforms, types.DEFORMING)
    kinds: tuple[str, ...] = ()
    # an output whose 3D data the node knows from one of its facts (NodeDef.facts: an import node, whether the 模型
    # selected deform): the fact's name
    kinds_from: str = ""
    # an output the node makes out of another of its own outputs (点云 is unprojected from 深度图 + 相机): those
    # names. Wanting this one wants them too (Evaluation.demand), so a node that only writes what is wanted
    # still has what it needs (without it, a node whose 点云 is wanted but whose 深度图 is not gets an empty depth
    # packet and fails while unprojecting)
    made_from: tuple[str, ...] = ()
    # an output that carries only its own type's kind of what an input carries ("input:scene": 「按种类取出」's 相机 is the
    # scene's cameras, nothing when the scene holds none); kinds not known upstream: nothing known
    narrows: str = ""
    # an input that takes data with no items in it (人物框 with nobody): False, the node has nothing to compute and
    # gives nothing, saying so (engine/cook.py); True, the node takes it and decides itself
    takes_empty: bool = True
    # an output that may give nothing (an empty packet) where its source has no such value (「读取视频」's 帧率: a
    # video that records none). A parameter it drives stays the node's to set: the source's value when it has one,
    # else the value typed here (the engine already keeps the typed value under an empty wire: evaluation.py
    # _wired_packet). The status reply carries it with the port; the page keeps that parameter editable and says so
    # (webui graph/rules.ts wiredFrom `fallback`), and apply_values lets a client set it (engine/templates.py)
    may_be_empty: bool = False
    # an image input whose alpha the node uses (warped with the picture, written, composited): the others take the RGB
    # of a picture with an alpha as it is stored, premultiplied (over black), and say so in the input's tooltip
    alpha: bool = False
    # an output: which aspects of the data's meaning this port guarantees to state (keys of data/contracts.py MEANING:
    # a depth port states scale, position and normal ports state space, a UV port states projection). The type states
    # only the channel count; the meaning is guaranteed by the writing port. The contract checks it and fails at once
    # when one is missing, so downstream never has to guess.
    means: tuple[str, ...] = ()
    # an output: what it says of its own geometry and of what was photographed (data/contracts.py Shape). The default
    # (KEEPS) is the picture input's window and its shot properties, which the engine fills in; only a node that moves
    # pixels or makes a picture of its own declares another
    shape: "Shape" = None  # type: ignore[assignment]  (KEEPS: set in __post_init__, which is where contracts is imported)

    def __post_init__(self) -> None:
        from ..data.contracts import KEEPS, carries_shot

        for t in self.type.split("|"):
            if element_of(t) not in DATA_TYPES:  # a list of a type is that type with "[]" (data/types.py)
                raise ValueError(f"Unknown data type {t!r} on port {self.name!r}")
        if self.shape is None:
            object.__setattr__(self, "shape", KEEPS)
        # `is not KEEPS`: rejects only shapes written by the author, not the default filled in by the line above.
        # Without it, `dataclasses.replace(port, ...)` would pass the filled KEEPS back unchanged, and ports without a
        # picture such as 「值」 and 「文字」 would raise "carries no picture" on copy; the family applies `applies` to
        # value ports through replace, and the extension loader swallows the exception, so the whole extension would
        # silently fail to load.
        elif self.shape is not KEEPS and not any(carries_shot(t) for t in self.type.split("|")):
            raise ValueError(f"port {self.name!r} of type {self.type!r} carries no picture: it declares no shape")
        if self.type_from and not self.type_from.startswith("input:"):
            raise ValueError(f"type_from of port {self.name!r} must be 'input:<port>', not {self.type_from!r}")

    @property
    def param(self) -> str:
        """The parameter this input drives ("" a declared input): a promoted parameter's port is "param:<name>"."""
        return self.name.removeprefix(PARAM) if self.name.startswith(PARAM) else ""

    def describe(self) -> dict:  # noqa: D401
        """As front ends read it: `inserts`, the node type the node menu offers first for a wire drawn out of this
        input: the one it recommends, else the one a per-wire check puts in front of it (「LensDistortion」, 「选人」); a
        check's own one click travels with the check (engine/lint.py, errors.Refused), not with the port. `when`, the
        name of the parameter whose value brings the output (a label's link: the hierarchy picker shows that kind's
        colour). Whether the output is there is never worked out from it: the status reply lists the outputs as they
        are (Graph.ports). `type_label`, the port's type in the words the artist knows (data/types.py type_label:
        「场景或场景列表」): a front end never spells a type's name itself, so a type added or a list form never leaves
        a raw id like "scene[]" in a tooltip."""
        from ..data.types import is_list

        # `help` travels inside the port's tip (Port.tip, the status reply), never on its own
        # `plain` is the server's wire check alone (the reply says the wire is wrong), never drawn
        out = {k: v for k, v in self.__dict__.items() if k not in ("expects", "recommend", "when", "shape", "help", "plain")}
        when = getattr(self.when, "name", "") if self.when is not None else ""
        # a list port: the editor draws it as a list of its items' type, never as another colour
        return {**out, "when": when, "list": all(is_list(t) for t in self.type.split("|")),
                "type_label": type_label(self.type),
                "inserts": self.recommend or next((e.fix for e in self.expects if e.fix and e.per_wire), "")}

    def tip(self, type_: str = "") -> str:
        """What the pointer says over this port, made here and nowhere else. It travels with the graph's ports
        (engine/graph.py ports), not with the catalogue: the catalogue describes types, and repeating every type's
        description on every port of every node type would add several KB to every first load. A node just added
        says its own description until its status arrives, a moment later.

        The type's name (the same `type_label` the port carries, with the unit), what that type is, what this port
        means beyond its type (`help`), and what an image input does with a picture's alpha. The summary of what it actually holds
        is added by the page from the result's own summary (data/summary.py), which is the only part that is not known
        before cooking; the page never writes a sentence of its own. `type_`: the type the port carries in a graph,
        when that is narrower than what it declares (engine/graph.py ports)."""
        kind = type_ or self.type  # a port that carries what an input gives it (Port.type_from): the type it has here
        head = type_label(kind) + (f" · {self.unit}" if self.unit else "")
        # a port that takes a type or a list of it ("scene|scene[]") says what that type is once, not twice
        said = list(dict.fromkeys(DATA_TYPES[element_of(t)].description for t in kind.split("|") if element_of(t) in DATA_TYPES))
        return "\n".join(line for line in [head, *said, self.help, self.alpha_note()] if line)

    def alpha_note(self) -> str:
        """What an image input does with a picture's alpha, for its tooltip ("" a port that never carries one).
        Ports with one or two channels (depth, mask, ST-map, ...) have no alpha to describe; only ports with three or
        four channels, and ports that accept any 2D data, describe it."""
        roots = {element_of(t).split(".")[0] for t in self.type.split("|")}
        if roots != {"image"} or channels_of(self.type) in (1, 2):
            return ""
        return "图像的 alpha 跟着一起处理" if self.alpha else "只用图像的 RGB：带 alpha 的图像按压在黑底上的颜色用，alpha 不看"


# the name of the input a promoted parameter gets: "param:focal_mm" (a wire into it sets the parameter)
PARAM = "param:"


def kind_declaration(port: Port) -> str:
    """Whether an input says what kind of pixels it takes as it must (Port.data; lab2shot check nodes): "" fine,
    "undeclared" (it may take an image of three or four channels, or any image — any member of its type's union, so
    image.3 second in "tracks2d|image.3" counts — and says neither picture, values nor both), "contrary" (it takes one or
    two channels only, values by their type, yet says a picture only)."""
    from ..data.types import channels_of, element_of

    images = [k for k in (element_of(x) for x in port.type.split("|")) if k.startswith("image")]
    if not images:
        return ""
    if all(channels_of(k) in (1, 2) for k in images):
        return "contrary" if port.data is False else ""
    return "" if port.data in (True, False, EITHER) else "undeclared"
