"""Reading a Nuke script (.nk) for the format module.

LD_3DE4 nodes and Camera nodes are read by the single parser, lab2shot/formats/nuke/parse.py; the functions below
delegate to it and contain no parsing of their own. This module adds the node wiring: a Nuke script is a stack (a node
with `inputs n` takes the top n, input 0 deepest; `set X [stack 0]` records the top, `push $X` pushes it again,
`push 0` pushes an empty input), so walking it determines which Read supplies an STMap's ST-map. The ST-map's direction
is taken from its file name by data/layers.py stmap_direction_of_name and is never inferred from the image.

All other nodes are ignored and reported (`ignored`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ...data.layers import stmap_direction_of_name
from . import parse as parser
from ...errors import Invalid
from ...messages import Msg

# A statement at the start of a line: a node block, a push, or a set that records the top of the stack.
STATEMENT = re.compile(r"(?m)^[ \t]*(?:([A-Za-z_][\w.]*)[ \t]*\{|push[ \t]+(\S+)|set[ \t]+(\w+)[ \t]+\[stack 0\])")
NO_INPUT = ("Read", "Root", "Constant", "ColorBars", "CheckerBoard", "ColorWheel", "Camera")  # 0 inputs unless an `inputs` knob says otherwise
KNOWN = ("Root", "Read", "STMap", "Merge2", "Camera")  # classes this module reads; LD_3DE nodes are recognised by class


@dataclass(frozen=True)
class Node:
    """One node of a script with the nodes wired to its inputs (None for an unconnected input)."""

    cls: str
    name: str
    knobs: dict[str, str]
    inputs: tuple = field(default_factory=tuple)

    @property
    def file(self) -> str:
        """A Read's file path with surrounding braces removed."""
        return self.knobs.get("file", "").strip().strip("{}").strip()


def nodes(text: str):
    """Every top-level node as (class, {knob: raw value})."""
    return parser.nodes(text)


def lens_nodes(text: str) -> list[dict]:
    """The LD_3DE nodes: [{name, class, model, label, animated}]."""
    return parser.lens_nodes(text)


def lens(text: str, name: str, width: int, height: int) -> dict:
    """The LD_3DE node `name` as a lens description: model, parameters, direction, raster and the additional values
    3DE records (focal length, film back, lens centre, pixel aspect, focus), converted from centimetres to
    millimetres."""
    return parser.lens(text, name, width, height)


def cameras(text: str) -> list[dict]:
    """The Camera nodes as [{name, frames: the keyed frames}], using the parser's own listing (`_cameras`, which
    assigns the fallback name for a node without a `name` knob) so that logic is not duplicated."""
    return [{"name": n, "frames": parser.keyed_frames(k)} for n, k in parser._cameras(text)]


def camera(text: str, name: str, frames: list[int], unit: str = "cm") -> dict:
    """The Camera node `name` at `frames`: {c2w [F,4,4] in centimetres Y up looking down -Z, focal_mm, h_aperture_mm,
    keyed, source}. `unit` is the length unit of the script's scene."""
    return parser.camera(text, name, frames, unit)


# ------------------------------------------------------------------ the wiring


def _inputs_of(cls: str, knobs: dict[str, str]) -> int:
    said = knobs.get("inputs", "").strip().strip("{}").strip()
    if said.lstrip("-").isdigit():
        return max(0, int(said))
    return 0 if cls.startswith(NO_INPUT) else 1


def statements(text: str):
    """The script's statements in order: ("node", class, knobs), ("push", what) or ("set", variable). A node's knobs
    are produced by the shared parser, applied to that node's block."""
    i, n = 0, len(text)
    while i < n:
        m = STATEMENT.search(text, i)
        if not m:
            return
        cls, pushed, named = m.groups()
        if cls is None:
            yield ("push", pushed) if pushed is not None else ("set", named)
            i = m.end()
            continue
        depth, j = 1, m.end()
        while j < n and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        found = parser.nodes(text[m.start():j])
        yield ("node", cls, found[0][1] if found else {})
        i = j


def graph(text: str) -> list[Node]:
    """Every node of the script with its inputs resolved, in script order."""
    stack: list[Node | None] = []
    remembered: dict[str, Node | None] = {}
    out: list[Node] = []
    for statement in statements(text):
        if statement[0] == "push":
            what = statement[1]
            stack.append(None if what in ("0", "$0") else remembered.get(what.lstrip("$")))
            continue
        if statement[0] == "set":
            remembered[statement[1]] = stack[-1] if stack else None
            continue
        _, cls, knobs = statement
        count = _inputs_of(cls, knobs)
        taken = tuple(stack[len(stack) - count:]) if count else ()
        del stack[len(stack) - len(taken):]
        node = Node(cls, knobs.get("name", f"{cls}{len(out) + 1}"), knobs, taken)
        out.append(node)
        if cls != "Root":
            stack.append(node)
    return out


def stmaps(text: str) -> list[dict]:
    """Every STMap node and the ST-map it applies: [{node, source, file, direction}]. The ST-map arrives on the
    second input; without a Read there the file is unknown (E-NUKE-NOSTMAPINPUT), and a file name that indicates
    neither direction is rejected (E-NUKE-STMAPDIR). The direction is never inferred from the image."""
    out = []
    for node in graph(text):
        if node.cls != "STMap":
            continue
        wired = [n for n in node.inputs if n is not None]
        read = next((n for n in reversed(wired) if n.cls == "Read"), None)
        if read is None or not read.file:
            raise Invalid(Msg("E-NUKE-NOSTMAPINPUT", node=node.name))
        direction = stmap_direction_of_name(Path(read.file).name)
        if not direction:
            raise Invalid(Msg("E-NUKE-STMAPDIR", node=node.name, file=Path(read.file).name[:60]))
        source = next((n.name for n in wired if n is not read), "")
        out.append({"node": node.name, "source": source, "file": read.file, "direction": direction})
    return out


def ignored(text: str) -> list[str]:
    """The distinct classes of nodes this module does not read, reported to the user handing over the script as
    ignored."""
    return sorted({cls for cls, _ in parser.nodes(text)
                   if not cls.startswith(KNOWN) and not parser.ld_model(cls) and not cls.upper().startswith("LD_3DE")})
