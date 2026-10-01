"""Retiming a scene: the only place where a frame number may be mapped to a different frame.

Elsewhere, frame numbers identify the shot's frames and are never converted implicitly (data/scene.py
`_layer_under`). 「重定时」 is the exception, and the node reports exactly what it changed.

Two independent quantities can change:

  * 偏移: all frames shift by a whole number; nothing is interpolated.
  * 速度 (equivalent to the knob on Nuke's Retime): each output frame advances this many source frames, so
    intermediate frames are interpolated. For example, 30 fps mocap in a 24 fps shot uses 速度 1.25 (30 / 24).

This module has no notion of frame rate: a scene carries only frame numbers, so relabelling the rate is not possible
here. The frame rate is the 「帧率」 parameter of the output-settings node and applies only to the file it writes.

Implementation: the source file is added as a sublayer of a new root layer with a time offset, the standard USD
mechanism for aligning shots. Nothing is copied, and everything in the scene (camera, mesh, point cloud, skeleton,
light) moves together. USD's own sampling performs the interpolation, spherically for quaternions, so UsdSkel joints
rotate along the shorter arc. Two cases where USD's result differs from what a DCC expects are overridden on top of
that layer:

  1. A time-varying transform is a matrix, and USD blends matrices element-wise, which shrinks rotations.
     `_turn_smoothly` re-authors those prims with linear translation and scale and slerped rotation.
  2. An array whose length varies between frames (a per-frame point cloud, a point cache that gains points) cannot
     be blended. USD holds the preceding sample; a DCC retime takes the nearest frame, and `_nearest_frames` does
     the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

ON_FRAME = 1e-6  # a source time this close to a whole frame was not interpolated


@dataclass(frozen=True)
class Timing:
    """Which source time each output frame shows. `offset` moves the result this many frames later; `speed` is how many
    source frames one output frame advances (1: none of them interpolated); `anchor` is the frame that stays put while
    the speed changes."""

    speed: float
    offset: int
    anchor: int
    source: tuple[int, ...]  # the source's own frames, sorted: a gap among them (a solve's missing frames) stays a gap

    @property
    def first(self) -> int:
        return self.source[0]

    @property
    def last(self) -> int:
        return self.source[-1]

    @property
    def changes_speed(self) -> bool:
        return abs(self.speed - 1.0) > 1e-9

    @property
    def ratio(self) -> float:
        """Source frames per output frame: the speed itself."""
        return float(self.speed)

    def source_time(self, frame) -> np.ndarray:
        f = np.asarray(frame, np.float64)
        return self.anchor + (f - self.offset - self.anchor) * self.ratio

    def output_time(self, source) -> np.ndarray:
        t = np.asarray(source, np.float64)
        return self.offset + self.anchor + (t - self.anchor) / self.ratio

    def _span(self) -> list[int]:
        lo, hi = self.output_time(self.first), self.output_time(self.last)
        return [int(f) for f in range(int(np.ceil(lo - ON_FRAME)), int(np.floor(hi + ON_FRAME)) + 1)]

    def _covered(self, t: float, have=None) -> bool:
        """A source time on one of its frames, or between two that follow on one another: not in a gap. `have`: the
        frames that count (an object's own visible ones, _hidden_between), the source's by default."""
        have = set(self.source) if have is None else have
        near = round(t)
        if abs(t - near) <= ON_FRAME:
            return near in have
        return int(np.floor(t)) in have and int(np.floor(t)) + 1 in have

    def frames(self) -> list[int]:
        """The frames the result has: every whole frame whose source time is on a source frame or between two that
        follow on one another (a pure 偏移 gives the source's frames moved); one whose source time falls in a gap of
        the source is missing too (in_gaps), never made up from the frames either side."""
        return [f for f in self._span() if self._covered(float(self.source_time(f)))]

    def in_gaps(self) -> list[int]:
        """The frames within the result's range left out because their source time falls in a gap of the source."""
        return [f for f in self._span() if not self._covered(float(self.source_time(f)))]

    def interpolated(self) -> list[int]:
        """The result's frames that fall between two source frames."""
        out = self.frames()
        t = self.source_time(out)
        return [f for f, x in zip(out, t) if abs(x - round(x)) > ON_FRAME]

    def layer_offset(self) -> Sdf.LayerOffset:
        """The sublayer offset that makes USD read `source_time(g)` at stage time g: a sublayer's time t shows at
        `offset + scale * t`. Every Lab2Shot layer shares the same time base (io/usd.py STAGE_FPS), so USD applies no
        additional scaling and this offset expresses the entire mapping."""
        r = self.ratio
        return Sdf.LayerOffset(float(self.offset) + self.anchor * (1.0 - 1.0 / r), 1.0 / r)


def _compose(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    from lab2shot_shared.motion import quat_to_matrix

    m = np.repeat(np.eye(4)[None], len(t), 0)
    m[:, :3, :3] = quat_to_matrix(q / np.linalg.norm(q, axis=1, keepdims=True)) * s[:, None, :]
    m[:, :3, 3] = t
    return m


def _sample_local(prim, times: list[float]) -> np.ndarray:
    return np.stack([np.array(UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode(t))).T for t in times])


def _turn_smoothly(stage: Usd.Stage, out_frames: list[int]) -> int:
    """Re-author every prim whose own transform varies over time so that its rotation follows the shorter arc
    instead of being blended as a matrix. Returns the number of prims written.

    The stage already carries the time offset, so key times and output frames are both in stage time here; the
    mapping is applied once, by the layer, and must not be applied again."""
    from lab2shot_shared.motion import continuous, decompose, slerp

    written = 0
    for prim in list(stage.Traverse()):
        x = UsdGeom.Xformable(prim)
        if not prim.IsA(UsdGeom.Xformable) or not x.TransformMightBeTimeVarying():
            continue
        keys = sorted({float(t) for op in x.GetOrderedXformOps() for t in (op.GetTimeSamples() or [])})
        if len(keys) < 2:
            continue
        source = np.clip(np.asarray(out_frames, np.float64), keys[0], keys[-1])
        m = _sample_local(prim, keys)
        t, q, s = decompose(m)  # motion.decompose: a mirrored joint keeps its mirror
        q = continuous(q)
        key_times = np.asarray(keys, np.float64)
        right = np.clip(np.searchsorted(key_times, source, side="left"), 1, len(keys) - 1)
        left = right - 1
        span = key_times[right] - key_times[left]
        u = np.where(span > 0, (source - key_times[left]) / np.where(span > 0, span, 1.0), 0.0)
        blended = _compose(t[left] + (t[right] - t[left]) * u[:, None], slerp(q[left], q[right], u),
                           s[left] + (s[right] - s[left]) * u[:, None])
        over = stage.OverridePrim(prim.GetPath())
        xo = UsdGeom.Xformable(over)
        xo.ClearXformOpOrder()
        op = xo.AddTransformOp()
        for f, one in zip(out_frames, blended):
            op.Set(Gf.Matrix4d(one.T.tolist()), Usd.TimeCode(f))
        written += 1
    return written


def _nearest_frames(stage: Usd.Stage, out_frames: list[int]) -> int:
    """Re-author the point data of every cloud or cache whose point count varies over the shot: where USD would hold
    the preceding sample, a retime takes the nearest source frame. Returns the number of frames that took data from a
    neighbouring frame."""
    taken = 0
    for prim in list(stage.Traverse()):
        based = UsdGeom.PointBased(prim)
        if not prim.IsA(UsdGeom.PointBased):
            continue
        keys = sorted({float(t) for t in (based.GetPointsAttr().GetTimeSamples() or [])})
        if len(keys) < 2:
            continue
        counts = [len(based.GetPointsAttr().Get(Usd.TimeCode(t)) or []) for t in keys]
        if len(set(counts)) == 1:  # constant point count: USD's own blending is correct

            continue
        key_times = np.asarray(keys, np.float64)
        source = np.clip(np.asarray(out_frames, np.float64), keys[0], keys[-1])
        near = np.abs(key_times[None, :] - source[:, None]).argmin(axis=1)
        over = stage.OverridePrim(prim.GetPath())
        for attr in prim.GetAuthoredAttributes():
            if not attr.GetTimeSamples():
                continue
            values = {t: attr.Get(Usd.TimeCode(t)) for t in keys}
            if not any(hasattr(v, "__len__") for v in values.values()):
                continue
            write = over.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=False)
            for f, k in zip(out_frames, near):
                write.Set(values[keys[k]], Usd.TimeCode(f))
        taken += int(sum(abs(source - key_times[near]) > ON_FRAME))
    return taken


def _hidden_between(stage: Usd.Stage, out_frames: list[int], timing: Timing) -> int:
    """Hide every object with a visibility of its own on the frames whose source time falls between one of its shown
    frames and one of its hidden ones: its own gap, the same rule as the source's (Timing._covered over the frames it
    is shown on). USD holds the shown side's token and blends its pose towards the far side of the gap — a pose made
    up from the frames either side. Returns the number of frames hidden this way that USD would show (of all objects)."""
    hidden = 0
    times = timing.source_time(out_frames)
    for prim in list(stage.Traverse()):
        attr = UsdGeom.Imageable(prim).GetVisibilityAttr() if prim.IsA(UsdGeom.Imageable) else None
        if not attr or attr.GetNumTimeSamples() < 2:
            continue
        shown = {k for k in timing.source
                 if attr.Get(Usd.TimeCode(float(timing.output_time(k)))) != UsdGeom.Tokens.invisible}
        cut = [abs(t - round(t)) > ON_FRAME and timing._covered(t) and not timing._covered(t, shown) for t in times]
        held = [attr.Get(Usd.TimeCode(f)) for f in out_frames]  # read before the override below is written
        changed = sum(gone and was != UsdGeom.Tokens.invisible for gone, was in zip(cut, held))
        if not changed:
            continue
        # a stronger layer's samples replace the weaker one's whole: every output frame is written
        write = UsdGeom.Imageable(stage.OverridePrim(prim.GetPath())).CreateVisibilityAttr()
        for f, gone, was in zip(out_frames, cut, held):
            write.Set(UsdGeom.Tokens.invisible if gone else was, Usd.TimeCode(f))
        hidden += int(changed)
    return hidden


def retime(src, out: Path, timing: Timing):
    """The scene shown at new frame numbers. Returns (packet, report)."""
    from ...data.payloads import SCENE_FILE, scene_packet
    from ...io import usd

    frames = timing.frames()
    stage = Usd.Stage.CreateNew(str(out / SCENE_FILE))
    usd.apply_conventions(stage, frames)
    root = stage.GetRootLayer()
    root.subLayerPaths = [str(src.path(SCENE_FILE))]
    root.subLayerOffsets[0] = timing.layer_offset()
    turned = _turn_smoothly(stage, frames) if frames else 0
    nearest = _nearest_frames(stage, frames) if frames else 0
    hidden = _hidden_between(stage, frames, timing) if frames and timing.changes_speed else 0
    root.Save()
    report = {"frames": frames, "interpolated": len(timing.interpolated()), "gaps": len(timing.in_gaps()),
              "nearest": nearest, "turned": turned, "hidden": hidden,
              "first": frames[0] if frames else 0, "last": frames[-1] if frames else 0}
    keep = {k: v for k, v in src.meta.items() if k not in ("frames", "contents", "top")}
    return scene_packet(out, frames, src.type, **keep), report
