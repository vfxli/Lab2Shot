"""The rig-and-model contract: the node sends a production rig's world poses, the worker answers with the model's
motion on that rig. Both sides use this module (numpy only).

Both tasks of the 骨骼动作 family send through it (lab2shot/nodes/families/rig_motion.py, lab2shot/nodes/kit/rig.py).
Generation (Kimodo, Two-stage Transformer) sends the animator's key frames and gets the motion between them; cleanup
(StableMotion, UnderPressure) sends every frame and gets the repaired motion back, plus `labels`: what the model
thought was wrong with each frame.

Node -> worker, the job input "motion" (write_job):
    motion.npz   names [J], parents [J], rest [J,4,4]   the rig: its bind pose in the world (joint-to-world, cm, Y up)
                 keys [K], poses [K,J,4,4]              the frames sent (frame numbers) and the rig's world pose at each
                 fps                                     the rig's frame rate
                 model [M], rig [M]                      the model joints the rig has, and their rig joint
                 aim [M]                                 each one's aim joint: a model joint, "up", or "" (its parent's)
                 legs [4]                                the model's left thigh, left shin, right thigh, right shin
                 landmarks [n]                           the model's hands and head (those it has): where the trunk runs
Worker -> node (write_result / read_result):
    raw/motion.npz  keys [K]             where each sent frame sits on the model's timeline, in model frames
                                         (model_frames for generation, model_times for cleanup: it need not be whole);
                                         the model's own frames are 0 ... T-1
                    joints [P]           rig joints the model moved
                    rotations [T,P,3,3]  their world rotations
                    root [T,3]           the rig's hip point in the world: the middle of its two thighs (cm)
                    contacts [T,4]       0..1: left heel, left toe, right heel, right toe on the ground
                    labels [T,L]         0..1 per model frame: how wrong the model found that frame, one column per
                                         thing it judges (a cleanup model that judges nothing writes none)
    raw/result.json root_joint (the rig joint moved to place it), thighs [2] (the two rig joints), model_fps, scale,
                    label_names [L], and what the model was given

The path without a production rig (a generating node given text only, no animation) goes through write_model_result: it
returns the model's own skeleton, and the file layout is described in that function's docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lab2shot_shared.rig_motion import MOTION, MotionJob, read_job, read_result, write_job  # noqa: F401
from lab2shot_shared.motion import Retarget

from .run import Run


def write_result(run: Run, retarget: Retarget, model_fps: float, keys: np.ndarray, rotations: np.ndarray,
                 root: np.ndarray, contacts: np.ndarray, labels: np.ndarray | None = None,
                 label_names: tuple[str, ...] = (), **info) -> Path:
    """The worker's side: the model's motion (world rotations [T,M,3,3] of all its joints and its root position
    [T,3], on its own timeline from the first frame sent, in the rig's world at the model's scale) back on the rig;
    `keys`: where each sent frame sits on the model's timeline (MotionJob.model_frames / model_times).
    `labels` [T] or [T,L] with `label_names`: what a cleanup model judged wrong on each of its own frames.
    The Run's standard timing and memory fields go into result.json with the family's (`frames` is the model's frame
    count, not the standard [first, last])."""
    joints, rot, root = retarget.to_production(np.asarray(rotations, np.float64), np.asarray(root, np.float64))
    raw = run.job.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    arrays = {} if labels is None else {"labels": np.asarray(labels, np.float32).reshape(len(rot), -1)}
    np.savez(raw / MOTION, keys=np.asarray(keys, np.float64), joints=np.asarray(joints, np.int64),
             rotations=rot, root=root, contacts=np.asarray(contacts, np.float32), **arrays)
    return run.finish(list(range(len(rot))), model_fps=model_fps, frames=len(rot), scale=retarget.scale,
                      root_joint=dict(retarget.pairs)[0], thighs=list(retarget.thighs), label_names=list(label_names),
                      **info)


def write_model_result(run: Run, skeleton, frames: int, rotations: np.ndarray, positions: np.ndarray,
                       contacts: np.ndarray, **info) -> Path:
    """The result without a production rig: an animation of the model's own skeleton, already resampled to the frame
    count the node asked for.

    The second path of the generating side of the 「骨骼动作」 family (lab2shot/nodes/families/rig_motion.py): the user
    connected no animation and gave only a text prompt and a frame count, so there is no rig, no joint pairing and no
    Retarget; the model's skeleton itself is returned, and the node converts it to CG bone names and joint axes.
    `rotations` [F,J,3,3] are each joint's world rotations, `positions` [F,J,3] its world positions (cm).

        raw/motion.npz  names [J], parents [J], rest [J,4,4]  the model skeleton and its rest pose (joint-to-world, cm, Y up)
                        anim [F,J,4,4]                        joint-to-world per frame (cm); F is the frame count requested
                        contacts [F,4]                        left heel, left toe, right heel, right toe on the ground

    result.json holds, besides the worker's own fields, Run's standard timing and VRAM fields (`frames` is the frame
    count requested, not the standard [first, last]).
    """
    anim = np.zeros((len(rotations), len(skeleton.names), 4, 4), np.float64)
    anim[..., :3, :3], anim[..., :3, 3], anim[..., 3, 3] = rotations, positions, 1.0
    raw = run.job.raw_dir
    raw.mkdir(parents=True, exist_ok=True)
    np.savez(raw / MOTION, names=np.array(skeleton.names), parents=np.asarray(skeleton.parents, np.int64),
             rest=np.asarray(skeleton.rest, np.float64), anim=anim,
             contacts=np.asarray(contacts, np.float32).reshape(len(anim), -1))
    return run.finish(list(range(int(frames))), frames=frames, joints=len(skeleton.names), **info)
