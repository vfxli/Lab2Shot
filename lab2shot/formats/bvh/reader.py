"""Reading a BVH file into scene arrays: its hierarchy as one skeleton named by its root joint (an End Site is a joint
<parent>_end, as Maya and MotionBuilder make it), its motion as each joint's world transform per frame, numbered from 1,
at 1 / Frame Time frames per second. A joint's local transform is its position channels (else its offset) then its
rotation channels in the order the file lists them (Zrotation Xrotation Yrotation: Rz · Rx · Ry). Depends only on
numpy."""

from __future__ import annotations

from ... import i18n

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from lab2shot_shared.motion import axis_angle
from lab2shot_shared.names import decode, unique
from lab2shot_shared.scene_arrays import SceneArrays
from lab2shot_shared.units import fps_of_period

from ...errors import Invalid
from ...messages import Msg

FIRST_FRAME = 1


def _refuse(path: Path, detail: str) -> Invalid:
    return Invalid(Msg("E-BVH-UNREADABLE", file=path.name, detail=detail))


def _finite(path: Path, rows, where: str, first: int = 0) -> np.ndarray:
    """Numbers of the file as [rows, columns] float64, every one finite: the one check every number goes through
    (OFFSET, Frame Time, MOTION). A NaN or an infinity is E-BVH-UNREADABLE naming where (`where`, "{row}" the row
    counted from `first`), never carried on into the bind pose or the 「帧率」."""
    values = np.array(rows, np.float64)
    if not np.isfinite(values).all():
        row = int(np.argwhere(~np.isfinite(values))[0][0]) + first
        raise _refuse(path, i18n.t("reader.bvh.not_finite", where=where.format(row=row)))
    return values


def _sections(path: Path, text: str) -> tuple[list[str], list[str]]:
    """The hierarchy's tokens and the motion's lines, split at the line that is MOTION alone (a joint named ROOT_MOTION
    is a name). In the hierarchy a joint's name is the rest of its ROOT / JOINT line (names with spaces, as Blender and
    MotionBuilder read them; "{" on the same line stays a token of its own), everything else split on white space."""
    lines = text.splitlines()
    at = next((n for n, line in enumerate(lines) if line.strip() == "MOTION"), len(lines))
    words: list[str] = []
    for n, line in enumerate(lines[:at], 1):
        found = re.match(r"(ROOT|JOINT)(\s+(.*))?$", line.strip())  # any white space after the keyword (a Tab too)
        if found:
            name = (found.group(3) or "").strip()
            brace = name.endswith("{")
            name = name[:-1].strip() if brace else name
            if not name:
                raise _refuse(path, i18n.t("reader.bvh.no_name", line=n, what=found.group(1)))
            words += [found.group(1), name] + (["{"] if brace else [])
        else:
            words += line.split()
    return words, lines[at:]


@lru_cache(maxsize=8)
def parsed(path: str) -> tuple[dict, list[str], int]:
    """The file's scene arrays (SceneArrays.arrays(): one character item without meshes, 骨架动画), the names whose
    bytes were not UTF-8 (replaced by U+FFFD, lab2shot_shared/names.py decode) and how many lines of motion there are
    past the frames its Frames: counts (left out, W-BVH-EXTRAFRAMES). Uploads never change: parsed once per
    file, however many times a cook and its listing ask. A file that does not parse (a number that is not one, a line
    cut short, a NaN) is E-BVH-UNREADABLE saying where, never a bare error."""
    text, replaced = decode(Path(path).read_bytes())
    try:
        out, extra = _arrays(Path(path), *_sections(Path(path), text))
    except Invalid:
        raise
    except (ValueError, IndexError) as exc:
        raise _refuse(Path(path), i18n.t("reader.bvh.malformed", why=exc)) from None
    names = [str(n) for n in out["character0_joints"]] if replaced else []
    return out, [n for n in names if "\ufffd" in n], extra


def arrays(path: Path) -> dict:
    return parsed(str(path))[0]


def _arrays(path: Path, words: list[str], motion_lines: list[str]) -> tuple[dict, int]:
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    i = 0

    def need(n: int) -> None:
        if i + n > len(words):
            raise _refuse(path, i18n.t("reader.bvh.hierarchy_cut"))

    need(1)
    if words[0] != "HIERARCHY":
        raise _refuse(path, i18n.t("reader.bvh.no_hierarchy"))
    i = 1
    pending: str | None = None
    while True:
        need(1)
        word = words[i]
        if word in ("ROOT", "JOINT"):
            need(2)
            pending = words[i + 1]
            i += 2
        elif word == "End":
            need(2)
            pending = f"{names[stack[-1]]}_end" if stack else "end"
            i += 2
        elif word == "{":
            if pending is None:
                raise _refuse(path, i18n.t("reader.bvh.extra", what="{"))
            # siblings of one name told apart as every format does (names.unique among the parent's children); the
            # same name under two parents (both hands' Index1) stays, the joints' keys are their paths (joints.joint_keys)
            parent = stack[-1] if stack else -1
            names.append(unique(pending, {n for n, p in zip(names, parents) if p == parent}))
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            stack.append(len(names) - 1)
            pending = None
            i += 1
        elif word == "OFFSET":
            need(4)
            offsets[stack[-1]] = _finite(path, [words[i + 1:i + 4]], i18n.t("reader.bvh.offset_of", joint=names[stack[-1]]))[0]
            i += 4
        elif word == "CHANNELS":
            need(2)
            n = int(words[i + 1])
            need(2 + n)
            channels[stack[-1]] = words[i + 2:i + 2 + n]
            i += 2 + n
        elif word == "}":
            if not stack:
                raise _refuse(path, i18n.t("reader.bvh.extra", what="}"))
            stack.pop()
            i += 1
            if not stack:
                break
        else:
            raise _refuse(path, i18n.t("reader.bvh.unknown", what=word))
    if i != len(words):
        raise _refuse(path, i18n.t("reader.bvh.after_hierarchy", what=words[i]))
    if not motion_lines:
        raise _refuse(path, i18n.t("reader.bvh.no_motion"))
    filled = [line.split() for line in motion_lines[1:] if line.strip()]  # blank lines anywhere after MOTION are no line
    head = filled[:2]
    if len(head) < 2 or head[0][:1] != ["Frames:"] or len(head[0]) != 2:
        raise _refuse(path, i18n.t("reader.bvh.no_frames"))
    if head[1][:2] != ["Frame", "Time:"] or len(head[1]) != 3:
        raise _refuse(path, i18n.t("reader.bvh.no_frame_time"))
    count = int(head[0][1])
    frame_time = float(_finite(path, [[head[1][2]]], "Frame Time")[0, 0])
    written_time = head[1][2]
    if frame_time <= 0:
        raise _refuse(path, i18n.t("reader.bvh.bad_frame_time", value=frame_time))
    width = sum(len(c) for c in channels)
    rows = filled[2:]  # one frame a line, as every writer writes it
    if count <= 0 or len(rows) < count:
        raise _refuse(path, i18n.t("reader.bvh.frames_cut", count=count, rows=len(rows)))
    for n, row in enumerate(rows[:count]):
        if len(row) != width:
            raise _refuse(path, i18n.t("reader.bvh.row_width", frame=n + FIRST_FRAME, have=len(row), want=width))
    motion = _finite(path, rows[:count], i18n.lookup("reader.bvh.frame_n"), FIRST_FRAME)

    local = np.repeat(np.eye(4)[None, None], count, 0).repeat(len(names), 1)  # [F,J,4,4]
    column = 0
    for j, own in enumerate(channels):
        position = offsets[j].copy()[None].repeat(count, 0)
        turn = np.repeat(np.eye(3)[None], count, 0)
        for name in own:
            v = motion[:, column]
            column += 1
            axis, what = name[0].lower(), name[1:].lower()
            if what == "position":
                position[:, "xyz".index(axis)] = v
            elif what == "rotation":
                turn = turn @ axis_angle(np.eye(3)["xyz".index(axis)], np.radians(v))
            else:
                raise _refuse(path, i18n.t("reader.bvh.unknown_channel", name=name))
        local[:, j, :3, :3] = turn
        local[:, j, :3, 3] = position
    anim = np.empty_like(local)
    rest = np.empty((len(names), 4, 4))
    for j, parent in enumerate(parents):  # parents come first
        still = np.eye(4)
        still[:3, 3] = offsets[j]
        rest[j] = still if parent < 0 else rest[parent] @ still
        anim[:, j] = local[:, j] if parent < 0 else anim[:, parent] @ local[:, j]
    out = SceneArrays(fps_of_period(written_time))  # Frame Time is written to a few places (0.016667: 59.9988 fps)
    frames = list(range(FIRST_FRAME, FIRST_FRAME + count))
    out.add("character", names[0], f"/{names[0]}", frames, joints=np.array(names), parents=np.array(parents, np.int64),
            bind=rest, anim=anim)
    return out.arrays(), len(rows) - count


# the longest chain of bones from where the body branches (the hips) to an end (a fingertip), measured along the bones
# so that any rest pose gives the same (LAFAN1's is folded): an adult's is 120–140 cm (LAFAN1 122, Mixamo 138)
HUMAN_REACH_CM = 125.0
UNIT_OFF = 5.0  # a chosen unit this many times off the guess is surely wrong (a child or a giant is not)


def reach(items: dict) -> float:
    """The skeleton's longest chain from its first joint with more than one child (the hips: legs and spine part there),
    summed bone by bone, in file units; 0 for a single joint. The joints above it only place the body (a Reference
    root on the ground, 95 cm below the hips, once made a person read 58% of their size); a skeleton that never
    branches is measured from its root."""
    character = items["character"][0]
    bind = np.asarray(character["bind"], np.float64).reshape(-1, 4, 4)
    parents = np.asarray(character["parents"]).reshape(-1)
    children = [np.flatnonzero(parents == j) for j in range(len(parents))]
    start = int(np.flatnonzero(parents < 0)[0]) if len(parents) else 0
    while len(children[start]) == 1:
        start = int(children[start][0])
    if not len(children[start]):  # one chain: nothing places it, measured whole
        start = int(np.flatnonzero(parents < 0)[0])
    along = np.full(len(bind), -np.inf)
    along[start] = 0.0
    for j, p in enumerate(parents):  # parents come first
        if p >= 0 and np.isfinite(along[p]):
            along[j] = along[p] + np.linalg.norm(bind[j, :3, 3] - bind[p, :3, 3])
    return float(along.max(initial=0.0))


def unit_guess(items: dict) -> float | None:
    """Centimetres per file unit that would make the skeleton an adult's (HUMAN_REACH_CM); None when it has no bones.
    CMU's files come out about 6 cm a unit: neither 厘米 nor 米."""
    length = reach(items)
    return HUMAN_REACH_CM / length if length > 0 else None


# the units a file is written in, as they are called (centimetres per unit); a guess within SNAP of one is that one
STANDARD_UNITS = {"cm": 1.0, "inch": 2.54, "dm": 10.0, "m": 100.0}  # their words: length.<id>
SNAP = 0.25


def unit_of(items: dict) -> tuple[float, str] | None:
    """The file's unit as 「自动」 reads it: the standard unit nearest the guess (unit_guess) when within ±25% of it —
    people differ in size, units do not, so a centimetre file (Mixamo's guess 0.91) is read as centimetres —, else the
    guess itself and "" (CMU's about 6 cm, which W-BVH-UNIT then says). None: no bones to measure."""
    guess = unit_guess(items)
    if guess is None:
        return None
    name, cm = min(STANDARD_UNITS.items(), key=lambda u: abs(np.log(guess / u[1])))
    return (cm, name) if abs(guess / cm - 1) <= SNAP else (guess, "")
