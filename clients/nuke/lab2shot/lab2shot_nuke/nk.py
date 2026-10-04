"""The node blocks of a .nk text the server delivers (lab2shot/formats/nuke: a camera, 2D tracks), read without Nuke.

A delivered .nk is never pasted (nuke.nodePaste wires the pasted nodes to whatever the user has selected and changes
the selection): each block becomes a node made with nuke.nodes.<Class>() and its knob lines are handed to readKnobs.
Pure Python, no Nuke: tests run it anywhere.
"""

from __future__ import annotations

import re

# knobs a delivered block carries that belong to the node's place in a script, not to its data: they are left out
# (the node is made where the host puts it, named by the host, wired to nothing)
PLACE_KNOBS = frozenset({"inputs", "name", "xpos", "ypos", "selected"})
# a delivered class -> the class made in Nuke 17's 3D system (the classic cameras share every knob name with Camera4)
NEW_3D = {"Camera": "Camera4", "Camera2": "Camera4", "Camera3": "Camera4"}
_CLASS = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\{")


def _statements(body: str) -> list[str]:
    """A block's body split into its top-level knob statements (a value may run over lines inside braces or quotes)."""
    out, cur, depth, quoted, escaped = [], [], 0, False, False
    for ch in body:
        if escaped:
            cur.append(ch)
            escaped = False
            continue
        if ch == "\\":
            cur.append(ch)
            escaped = True
            continue
        if ch == '"':
            quoted = not quoted
        elif not quoted and ch == "{":
            depth += 1
        elif not quoted and ch == "}":
            depth -= 1
        if ch == "\n" and depth == 0 and not quoted:
            line = "".join(cur).strip()
            if line:
                out.append(line)
            cur = []
            continue
        cur.append(ch)
    line = "".join(cur).strip()
    if line:
        out.append(line)
    return out


def blocks(text: str) -> list[dict]:
    """[{"class": the class Nuke 17 makes, "delivered": the class in the file, "knobs": readKnobs text, "name": the
    block's own name}] for every node block of the text, in order. Script lines outside blocks (set / push) are
    skipped."""
    out, i = [], 0
    while True:
        m = _CLASS.search(text, i)
        if m is None:
            return out
        if m.group(1) in ("set", "push"):
            i = m.end()
            continue
        start, depth, j, quoted = m.end(), 1, m.end(), False
        while j < len(text) and depth:
            ch = text[j]
            if ch == "\\":
                j += 2
                continue
            if ch == '"':
                quoted = not quoted
            elif not quoted and ch == "{":
                depth += 1
            elif not quoted and ch == "}":
                depth -= 1
            j += 1
        body = text[start:j - 1]
        kept, name = [], ""
        for s in _statements(body):
            key = s.split(None, 1)[0]
            if key == "name":
                name = s.split(None, 1)[1].strip().strip('"') if len(s.split(None, 1)) > 1 else ""
            if key not in PLACE_KNOBS:
                kept.append(s)
        delivered = m.group(1)
        out.append({"class": NEW_3D.get(delivered, delivered), "delivered": delivered, "knobs": "\n".join(kept),
                    "name": name})
        i = j


_CURVE = re.compile(r"\{curve\b[^{}]*\}")
_KEY_FRAME = re.compile(r"(?<![A-Za-z0-9_.])x(-?\d+(?:\.\d+)?)")


def shift_frames(knobs: str, by: int) -> str:
    """Every animation key of the knob text moved by `by` frames (the delivered curves are keyed at the picture's own
    frame numbers; Nuke's frame = the picture's frame - plate offset). Only the `x<frame>` marks inside curves move:
    a curve's first key always carries one, the keys after it follow frame by frame."""
    if not by:
        return knobs

    def keys(m):
        return _KEY_FRAME.sub(lambda k: "x" + _number(float(k.group(1)) + by), m.group(0))

    return _CURVE.sub(keys, knobs)


def _number(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(v)
