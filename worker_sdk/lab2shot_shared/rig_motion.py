"""The rig-and-model contract's files, numpy only: the node writes the job (write_job), the worker reads it
(read_job) and the node reads the answer (read_result); the worker's writing of it is
lab2shot_worker.rig_motion.write_result. The layout is in lab2shot_worker/rig_motion.py.

One contract for both kinds of work of the rig-motion family (lab2shot/nodes/families/rig_motion.py, the shared
node side in lab2shot/nodes/kit/rig.py): 动作补帧 sends the animator's keys and gets the motion between them, 动作清理
sends every frame and gets the repaired motion back. `frames` is what the node sent — the key
frames for one, all of the rig's frames for the other — so both sides are the same code."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .motion import Retarget, Skeleton
from .protocol import Failure

MOTION = "motion.npz"


def write_job(path: Path, rig: Skeleton, frames, poses: np.ndarray, fps: float, pairs: dict[str, int],
              aims: dict[str, str], legs: tuple[str, str, str, str], landmarks: tuple[str, ...]) -> Path:
    """The node's side: the rig, the frames it sends and which model joint follows which rig joint (the module doc)."""
    model = list(pairs)
    np.savez(path, names=np.array(rig.names), parents=rig.parents, rest=rig.rest, keys=np.asarray(frames, np.int64),
             poses=np.asarray(poses, np.float64), fps=float(fps), model=np.array(model),
             rig=np.array([pairs[m] for m in model], np.int64), aim=np.array([aims.get(m) or "" for m in model]),
             legs=np.array(legs), landmarks=np.array(landmarks, dtype=str))
    return path


@dataclass
class MotionJob:
    """What a worker reads (read_job)."""

    rig: Skeleton
    keys: np.ndarray
    poses: np.ndarray
    fps: float
    pairs: dict[str, int]
    aims: dict[str, str]
    legs: list[str]
    landmarks: list[str]  # the model's hands and head (those it has): where its trunk runs (motion.trunk_of)

    def retarget(self, model: Skeleton) -> Retarget:
        """The rest-pose alignment between the rig and the model's skeleton (motion.Retarget.align); a rig whose rest
        pose is no standing pose (a BVH zero pose) aligns in the most model-like of its own sent poses."""
        index = {n: i for i, n in enumerate(model.names)}
        missing = [m for m in [*self.pairs, *self.legs] if m not in index]
        if missing:
            raise Failure("E-MOTION-NOJOINTS", joints=missing)
        pairs = sorted((index[m], p) for m, p in self.pairs.items())
        aims: list = [None] * len(model.names)
        for m, a in self.aims.items():
            aims[index[m]] = "up" if a == "up" else index.get(a)
        return Retarget.align(model, self.rig, pairs, aims, tuple(index[n] for n in self.legs), self.poses,
                              tuple(index[n] for n in self.landmarks if n in index))

    def model_times(self, model_fps: float) -> np.ndarray:
        """Where each sent frame sits on the model's timeline (the first at 0), in model frames, not rounded: the
        model runs at its own rate, and a frame of the rig generally falls between two of the model's."""
        return (self.keys - self.keys[0]) / self.fps * model_fps

    def model_frames(self, model_fps: float) -> np.ndarray:
        """Each key's frame on the model's timeline (the first key at 0): the model frame nearest to its time. The rig's
        frames between two keys map evenly onto the model frames between them (read on the node's side), so a key sits
        exactly on a model frame and the timing between keys moves by less than half a model frame.

        Only 动作补帧 uses this: keys are few and far apart, so each lands on its own model frame. A cleanup node sends
        every frame — several of them would round to the same model frame — and uses `model_times` instead."""
        return np.round(self.model_times(model_fps)).astype(np.int64)


def read_job(path: Path) -> MotionJob:
    d = np.load(path)
    rig = Skeleton([str(n) for n in d["names"]], d["parents"], d["rest"])
    model = [str(m) for m in d["model"]]
    return MotionJob(rig, d["keys"], d["poses"], float(d["fps"]), dict(zip(model, (int(j) for j in d["rig"]))),
                     {m: str(a) for m, a in zip(model, d["aim"]) if str(a)}, [str(n) for n in d["legs"]],
                     [str(n) for n in d["landmarks"]])


def read_result(raw: Path) -> dict:
    d = np.load(raw / MOTION)
    return {**{k: d[k] for k in d.files}, "info": json.loads((raw / "result.json").read_text(encoding="utf-8"))}
