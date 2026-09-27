"""Reading a BVH file into scene arrays: its hierarchy as one skeleton named by its root joint (an End Site is a joint
<parent>_end, as Maya and MotionBuilder make it), its motion as each joint's world transform per frame, numbered from 1,
at 1 / Frame Time frames per second. A joint's local transform is its position channels (else its offset) then its
rotation channels in the order the file lists them (Zrotation Xrotation Yrotation: Rz · Rx · Ry). Depends only on
numpy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lab2shot_shared.scene_arrays import SceneArrays

from ...errors import Invalid
from ...messages import Msg

FIRST_FRAME = 1


def _refuse(path: Path, detail: str) -> Invalid:
    return Invalid(Msg("E-BVH-UNREADABLE", file=path.name, detail=detail))


def _rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    """[F,4,4] rotations about x, y or z (column vectors)."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    m = np.zeros((len(a), 4, 4))
    m[:, 3, 3] = 1.0
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    k = 3 - i - j
    m[:, k, k] = 1.0
    m[:, i, i], m[:, i, j], m[:, j, i], m[:, j, j] = c, -s, s, c
    return m


def arrays(path: Path) -> dict:
    """The file's scene arrays (SceneArrays.arrays()): one character item without meshes (骨架动画)."""
    words = path.read_text(encoding="utf-8", errors="replace").split()
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    stack: list[int] = []
    i = 0

    def need(n: int) -> None:
        if i + n > len(words):
            raise _refuse(path, "层级没写完，文件不完整")

    need(1)
    if words[0] != "HIERARCHY":
        raise _refuse(path, "没有 HIERARCHY")
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
                raise _refuse(path, "多出的 {")
            names.append(pending)
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            stack.append(len(names) - 1)
            pending = None
            i += 1
        elif word == "OFFSET":
            need(4)
            offsets[stack[-1]] = np.array([float(w) for w in words[i + 1:i + 4]])
            i += 4
        elif word == "CHANNELS":
            need(2)
            n = int(words[i + 1])
            need(2 + n)
            channels[stack[-1]] = words[i + 2:i + 2 + n]
            i += 2 + n
        elif word == "}":
            if not stack:
                raise _refuse(path, "多出的 }")
            stack.pop()
            i += 1
            if not stack:
                break
        else:
            raise _refuse(path, f"不认识的 {word}")
    need(5)
    if words[i] != "MOTION" or words[i + 1] != "Frames:":
        raise _refuse(path, "没有 MOTION")
    count = int(words[i + 2])
    if words[i + 3] != "Frame" or words[i + 4] != "Time:":
        raise _refuse(path, "没有 Frame Time")
    need(6)
    frame_time = float(words[i + 5])
    width = sum(len(c) for c in channels)
    values = words[i + 6:i + 6 + count * width]
    if count <= 0 or frame_time <= 0 or len(values) < count * width:
        raise _refuse(path, f"应该有 {count} 帧、每帧 {width} 个数，文件不完整")
    motion = np.array(values, np.float64).reshape(count, width)

    local = np.repeat(np.eye(4)[None, None], count, 0).repeat(len(names), 1)  # [F,J,4,4]
    column = 0
    for j, own in enumerate(channels):
        position = offsets[j].copy()[None].repeat(count, 0)
        turn = np.repeat(np.eye(4)[None], count, 0)
        for name in own:
            v = motion[:, column]
            column += 1
            axis, what = name[0].lower(), name[1:].lower()
            if what == "position":
                position[:, "xyz".index(axis)] = v
            elif what == "rotation":
                turn = turn @ _rotation(axis, v)
            else:
                raise _refuse(path, f"不认识的通道 {name}")
        local[:, j] = turn
        local[:, j, :3, 3] = position
    anim = np.empty_like(local)
    rest = np.empty((len(names), 4, 4))
    for j, parent in enumerate(parents):  # parents come first
        still = np.eye(4)
        still[:3, 3] = offsets[j]
        rest[j] = still if parent < 0 else rest[parent] @ still
        anim[:, j] = local[:, j] if parent < 0 else anim[:, parent] @ local[:, j]
    out = SceneArrays(round(1.0 / frame_time, 3))
    frames = list(range(FIRST_FRAME, FIRST_FRAME + count))
    out.add("character", names[0], f"/{names[0]}", frames, joints=np.array(names), parents=np.array(parents, np.int64),
            bind=rest, anim=anim)
    return out.arrays()
