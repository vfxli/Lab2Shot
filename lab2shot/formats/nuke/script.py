"""The textual form of a Nuke script (.nk): node block layout, per-frame knob values, and valid node names.

A .nk file is Tcl: `NodeClass { knob value ... }`, one block per node in creation order. A pasted snippet begins by
taking the current selection in the comp as its input (`set cut_paste_input [stack 0]` / `push
$cut_paste_input`), which is what Nuke writes when nodes are copied. A knob that varies over time holds a curve:
`{curve x1001 1 2 3}`, i.e. the frame of the first key followed by one value per subsequent frame (a key on every
frame, with `x<frame>` repeated wherever the sequence skips frames). Frame numbers are written as the plate's own and
are never renumbered.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ...errors import Invalid
from ...messages import Msg

HEAD = ("set cut_paste_input [stack 0]", "push $cut_paste_input")  # a pasted snippet takes the selected node as input
MARK = "Lab2Shot"  # written into every block's label so the result is distinguishable from hand-tracked work


def node_name(name: str, fallback: str) -> str:
    """`name` as a Nuke node name: letters, digits and underscores, starting with a letter (Nuke silently renames any
    other name on paste). `fallback` is used when nothing remains after cleaning."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    if not cleaned:
        return fallback
    return cleaned if cleaned[0].isalpha() else f"{fallback}_{cleaned}"


def number(value: float, digits: int = 4) -> str:
    """A number in Nuke's notation: no exponent, no trailing zeros (0.5, 1, -12.25). Lens coefficients require more
    digits than positions (`digits`). Non-finite values are rejected (E-NUKE-VALUE): Nuke cannot read nan or inf and
    would silently leave the knob at its default."""
    v = float(value)
    if not math.isfinite(v):
        raise Invalid(Msg("E-NUKE-VALUE", knob="", value=str(value)[:20]))
    text = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-", "-0") else "0"


def curve(frames: Sequence[int], values: Sequence[float], digits: int = 4) -> str:
    """One knob's per-frame values as an animation curve. Constant values are written as a plain number, so a static
    camera has no keys, consistent with a hand-built one."""
    kept = [number(v, digits) for v in values]
    if not kept:
        return "0"
    if len(set(kept)) == 1:
        return kept[0]
    keys, last = [], None
    for f, v in zip(frames, kept):
        keys.append(v if last is not None and f == last + 1 else f"x{f} {v}")
        last = f
    return "{curve " + " ".join(keys) + "}"


def channel(frames: Sequence[int], values: Sequence[float], digits: int = 4) -> str:
    """One channel's value in knob form: a plain number when constant, otherwise its curve wrapped in the braces a knob
    value requires, e.g. `focal {{curve x1001 35 36}}`. A multi-channel knob instead places each channel's curve()
    inside a single pair of braces (see channels())."""
    got = curve(frames, values, digits)
    return "{" + got + "}" if got.startswith("{") else got


def channels(frames: Sequence[int], columns: Sequence[Sequence[float]], digits: int = 4) -> str:
    """A multi-channel knob (translate, rotate, win_translate): each channel's number or curve within one pair of
    braces."""
    return "{%s}" % " ".join(curve(frames, column, digits) for column in columns)


def label(text: str) -> str:
    """A node's label stating that the result comes from Lab2Shot and by which method, so it is not mistaken for
    hand-tracked work (a machine-learning result must be marked where it is visible, not only in metadata)."""
    return quoted(f"{MARK}: {text}" if not text.startswith(MARK) else text)


def quoted(text: str) -> str:
    """A string knob's value: Tcl quoting (no line breaks, backslashes and quotes escaped)."""
    cleaned = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{cleaned}"'


def block(kind: str, knobs: Sequence[tuple[str, str]]) -> str:
    """One node block, `Camera3 { ... }`, with one knob per line."""
    return "\n".join([kind + " {"] + [f" {name} {value}" for name, value in knobs] + ["}"])


def script(blocks: Sequence[str]) -> str:
    """The complete snippet: the paste header followed by the node blocks."""
    return "\n".join([*HEAD, *blocks]) + "\n"
