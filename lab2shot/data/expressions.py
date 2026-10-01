"""Facial expression names (「表情重定向（ARKit52）」, core.expression_retarget_arkit52): which of a source's expression curves drives which
of a character's blend shapes, guessed by name. The project's one table of expression names and their aliases.

The common vocabulary is Apple ARKit's 52 blend shapes (ARFaceAnchor.BlendShapeLocation): MediaPipe's face landmarker
gives 51 of them under the same names (it does not predict tongueOut), and most character pipelines (Character
Creator / AccuRIG's ARKit set, MetaHuman's ARKit remap, Faceware, Live Link Face) name their shapes after them. A name
is compared by its words, not its spelling: case, separators, a namespace ("blendShape1.", "Head:"), MediaPipe's
per-face prefix ("face2_") and the side written as L / R / Left / Right before or after the rest all read the same
(eyeBlinkLeft = EyeBlink_L = eye_blink_left = Eye_Blink_L). ALIASES adds the few names that say the same thing in
other words (FLAME's eyelid_left is ARKit's eyeBlinkLeft).

FLAME's own expression curves (SMIRK expression_00…49, Pixel3DMM expression_000…099) are components of a PCA basis,
not named movements: they reach ARKit shapes through no name, only through a FLAME-based character that carries the
same components as blend shapes (matched by their own names, as any name is matched when both sides use it).

A mapping row is the 「对应关系」 row of data/joints.py: {"part": slot, "src": [curve], "dst": [blend shape]}; the slots
are the source's curves (ARKit ones by their ARKit name, any other by its own name), so the same editor
(webui/src/editor/RigMap.tsx) shows them as a grid of expression slots instead of a body.
"""

from __future__ import annotations

import re

SIDES = {"left": "Left", "right": "Right"}
# ARKit's 52, by region (the editor groups its slots by these; the notes of the node too)
ARKIT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("眉毛", ("browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight")),
    ("眼睛", ("eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
              "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
              "eyeWideLeft", "eyeWideRight")),
    ("脸颊、鼻子", ("cheekPuff", "cheekSquintLeft", "cheekSquintRight", "noseSneerLeft", "noseSneerRight")),
    ("下巴", ("jawForward", "jawLeft", "jawRight", "jawOpen")),
    ("嘴", ("mouthClose", "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
           "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft",
           "mouthStretchRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
           "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
           "mouthUpperUpRight")),
    ("舌头", ("tongueOut",)),
)
OTHER = "其他"  # the region of a curve that is not an ARKit shape
# names that say an ARKit shape in other words (compared by key(), like every name)
ALIASES = {"eyelid_left": "eyeBlinkLeft", "eyelid_right": "eyeBlinkRight",  # FLAME / SMIRK / Pixel3DMM
           "jaw_drop": "jawOpen", "mouth_open": "jawOpen", "brow_raise_inner": "browInnerUp"}

_NAMESPACE = re.compile(r"^.*[:.|]")  # Maya's "blendShape1.", a namespace "Head:"
_FACE_PREFIX = re.compile(r"^face\d+_", re.IGNORECASE)  # MediaPipe's per-face prefix when several faces are tracked


def key(name: str) -> str:
    """A name as its words, side last: "EyeBlink_L", "eye_blink_left", "blendShape1.eyeBlinkLeft", "face2_eyeBlinkLeft"
    -> "eye blink left"."""
    name = _FACE_PREFIX.sub("", _NAMESPACE.sub("", str(name)))
    words = [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)]
    side = [w for w in words if w in ("l", "left", "r", "right")]
    rest = [w for w in words if w not in ("l", "left", "r", "right")]
    if len(side) == 1:
        rest.append("left" if side[0] in ("l", "left") else "right")
    else:
        rest += side
    return " ".join(rest)


_CANON = {key(n): n for _, names in ARKIT for n in names} | {key(a): n for a, n in ALIASES.items()}
REGION_OF = {n: region for region, names in ARKIT for n in names}


def arkit(name: str) -> str | None:
    """The ARKit shape a name is (by its words or an alias), or None."""
    return _CANON.get(key(name))


def slot(curve: str) -> str:
    """The slot a source curve fills: its ARKit name when it is one, else the curve's own name."""
    return arkit(curve) or curve


def slot_rows(curves: list[str]) -> list[dict]:
    """The 「对应关系」 editor's slots, one per source curve (the rows data/joints.py part_rows gives for a body): id,
    label, region (ARKit's regions, 其他 for the rest), never a chain, none required. ARKit order first."""
    ids = list(dict.fromkeys(slot(c) for c in curves))
    order = {n: k for k, n in enumerate(n for _, names in ARKIT for n in names)}
    ids.sort(key=lambda s: order.get(s, len(order)))
    return [{"id": s, "label": s, "region": REGION_OF.get(s, OTHER), "chain": False, "required": False} for s in ids]


def auto_rows(curves: list[str], shapes: list[str]) -> list[dict]:
    """The guessed mapping as rows: every source curve, and the target shape that says the same (its ARKit shape,
    or the same words), none when the target has no such shape."""
    by_slot: dict[str, str] = {}
    for s in shapes:
        by_slot.setdefault(slot(s), s)
        by_slot.setdefault(key(s), s)
    rows: dict[str, dict] = {}
    for c in curves:
        if slot(c) in rows:  # one row per slot: the first curve that fills it
            continue
        target = by_slot.get(slot(c)) or by_slot.get(key(c))
        rows[slot(c)] = {"part": slot(c), "src": [c], "dst": [target] if target else []}
    return list(rows.values())


def merged_rows(mapping: list[dict] | None, auto: list[dict]) -> tuple[list[dict], set[str]]:
    """The rows a cook uses: the parameter's rows, and the guess for every slot the parameter does not list.
    Returns (rows, the slots that were guessed)."""
    given = {r["part"]: r for r in mapping or []}
    rows = [given.get(r["part"], r) for r in auto] + [r for p, r in given.items() if p not in {a["part"] for a in auto}]
    return rows, {r["part"] for r in auto if r["part"] not in given}


def check_mapping(rows: list[dict], curves: list[str], shapes: list[str]) -> None:
    """Refuses a row naming a curve the source lacks or a shape the target lacks (E-EXPRMAP-NONAME), more than one of
    either on a slot (E-EXPRMAP-ONENAME), and one curve or one shape on two slots (E-EXPRMAP-SHARED) — the rules the
    editor shows in the error colour (webui/src/editor/rigMap/rules.ts problems); this is the authority."""
    from ..errors import Invalid
    from ..messages import Msg

    for col, have, said in (("src", curves, "表情"), ("dst", shapes, "目标")):
        taken: dict[str, str] = {}
        for row in rows:
            names = list(row[col])
            for n in names:
                if n not in have:
                    raise Invalid(Msg("E-EXPRMAP-NONAME", slot=row["part"], name=n, side=said))
            if len(names) > 1:
                raise Invalid(Msg("E-EXPRMAP-ONENAME", slot=row["part"], count=len(names), side=said))
            for n in names:
                if n in taken and taken[n] != row["part"]:
                    raise Invalid(Msg("E-EXPRMAP-SHARED", first=taken[n], second=row["part"], name=n, side=said))
                taken[n] = row["part"]
