"""Character animation in scene packets (UsdSkel): reading a skeleton's bind pose, its per-frame animation and the
frames an animator keyed; writing new animation over a scene while keeping the rest of it; and transferring a model's
motion back onto the rig. 动作补帧 (`on_rig`) and 动作清理 (`repaired`) share the same transfer (`_from_model`:
resample the model's timeline onto the rig's frames and set the world rotations of the affected joints) and differ
only in the final step."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lab2shot_shared import motion as mo
from pxr import Usd, UsdGeom, UsdSkel

from ..errors import Invalid
from ..messages import Msg
from .payloads import SCENE_FILE, scene_packet
from .packet import Packet


def _matrices(values) -> np.ndarray:
    """USD matrices (row vectors) -> [N,4,4] column-vector matrices."""
    return np.transpose(np.array([np.array(m) for m in values], np.float64).reshape(-1, 4, 4), (0, 2, 1))


def skeletons(packet: Packet) -> list[dict]:
    """The skeletons in a scene packet, in scene order: prim path, joint names and parents."""
    from .scene import open_scene

    stage = open_scene([packet])  # kept alive while its prims are read
    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            joints = skel.GetJointsAttr().Get() or []
            out.append({"path": str(prim.GetPath()), "joints": _joint_names(skel),
                        "parents": list(UsdSkel.Topology(joints).GetParentIndices())})
    return out


def _joint_names(skel: UsdSkel.Skeleton) -> list[str]:
    names = skel.GetJointNamesAttr().Get()
    return [str(n) for n in names] if names else [str(j).rsplit("/", 1)[-1] for j in skel.GetJointsAttr().Get() or []]


@dataclass
class Rig:
    """A skeleton of a scene packet and its animation over the packet's frames."""

    path: str  # the Skeleton prim
    names: list[str]
    parents: np.ndarray
    bind: np.ndarray  # [J,4,4] joint-to-skeleton in the bind pose
    frames: list[int]
    local: np.ndarray  # [F,J,4,4] joint-to-parent per frame (a root joint: to the skeleton prim)
    placement: np.ndarray  # [F,4,4] skeleton prim to world per frame
    keys: list[int]  # the frames its animation has authored samples at
    anim: str  # its UsdSkelAnimation prim ("" when it has none)

    def world(self, local: np.ndarray | None = None) -> np.ndarray:
        """Joint-to-world [F,J,4,4] (the rig's own animation, or `local`)."""
        return self.placement[:, None] @ mo.world_from_local(self.local if local is None else local, self.parents)

    def skeleton(self) -> mo.Skeleton:
        """The rig as used by retargeting: its bind pose placed in the world at the first frame."""
        return mo.Skeleton(self.names, self.parents, self.placement[0] @ self.bind)

    def index(self, frames) -> np.ndarray:
        return np.searchsorted(self.frames, frames)


def read_rig(packet: Packet, path: str | None = None, *, least_frames: int = 2) -> Rig:
    """The skeleton at `path` (default: the first) of a scene packet, sampled at every one of the packet's frames.

    `least_frames`: the minimum number of frames the caller requires. The default of 2 applies to callers that read
    the rig's animation, since a single frame does not constitute motion. Callers that only need the rig itself (its
    joints and rest pose) pass 1; this covers the target of 线性蒙皮变形, whose motion arrives through another input,
    and a freshly auto-rigged character, which consists of a single static frame."""
    from .scene import open_scene

    stage = open_scene([packet])
    prims = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if not prims:
        raise Invalid(Msg("E-ANIM-NOSKELETON"))
    prim = next((p for p in prims if str(p.GetPath()) == path), None) if path else prims[0]
    if prim is None:
        raise Invalid(Msg("E-ANIM-NOSUCHSKELETON", path=path, found=[str(p.GetPath()) for p in prims]))
    frames = [int(f) for f in packet.meta.get("frames") or []]
    if len(frames) < least_frames:
        raise Invalid(Msg("E-ANIM-ONEFRAME"))
    skel = UsdSkel.Skeleton(prim)
    query = UsdSkel.Cache().GetSkelQuery(skel)
    parents = np.asarray(query.GetTopology().GetParentIndices(), np.int64)
    order = [str(j) for j in query.GetJointOrder()]
    names = _joint_names(skel) if len(_joint_names(skel)) == len(order) else [j.rsplit("/", 1)[-1] for j in order]
    bind_attr = skel.GetBindTransformsAttr().Get()
    if bind_attr is not None and len(bind_attr) == len(order):
        bind = _matrices(bind_attr)
    else:
        bind = mo.world_from_local(_matrices(skel.GetRestTransformsAttr().Get()), parents)
    local = np.stack([_matrices(query.ComputeJointLocalTransforms(Usd.TimeCode(f))) for f in frames])
    xf = UsdGeom.Xformable(prim)
    placement = np.stack([np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode(f))).T for f in frames])
    anim = query.GetAnimQuery()
    samples = anim.GetJointTransformTimeSamples() if anim else []
    keys = sorted({int(round(t)) for t in samples} & set(frames))
    return Rig(str(prim.GetPath()), names, parents, bind, frames, local, placement, keys,
               str(anim.GetPrim().GetPath()) if anim else "")


def parse_frames(text: str) -> list[int]:
    """Parse frame numbers in Nuke frame-range syntax: "1001, 1012, 1030", "1001-1100x12" (every 12th frame),
    separated by spaces or commas."""
    out: set[int] = set()
    for part in re.split(r"[,，\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(-?\d+)(?:-(-?\d+)(?:x(\d+))?)?", part)
        if not m:
            raise Invalid(Msg("E-ANIM-BADFRAMES", text=part))
        a, b, step = int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3) or 1)
        if b < a or step < 1:
            raise Invalid(Msg("E-ANIM-BACKWARDS", text=part))
        out.update(range(a, b + 1, step))
        out.add(b)
    return sorted(out)


def write_animation(src: Packet, out: Path, rig: Rig, local: np.ndarray, info: dict) -> Packet:
    """The scene with the rig moved by `local` [F,J,4,4] at its frames: a layer over the input (meshes, other
    characters, cameras stay as they were) that gives the skeleton's animation new samples on every frame. A skeleton
    without an animation gets one next to it.

    The frames are the motion's own (`rig.frames`). The frame rate is not handled here: a scene carries frame
    numbers only, and the rate is set by the output-settings node's 「帧率」 parameter (data/units.py DEFAULT_FPS)."""
    from ..io import usd
    from .scene import _file, _new_layer_stage

    stage = _new_layer_stage(out / SCENE_FILE, [_file(src)], rig.frames)
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(rig.path))
    joints = skel.GetJointsAttr().Get()
    if rig.anim:
        anim = UsdSkel.Animation(stage.GetPrimAtPath(rig.anim))
    else:
        anim = UsdSkel.Animation.Define(stage, skel.GetPath().GetParentPath().AppendChild("lab2shot_anim"))
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    usd.write_joint_samples(anim, joints, local, rig.frames)  # the skeleton's own joint order: every joint animated
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData or {})
    data["lab2shot"] = usd.layer_data({**dict(data.get("lab2shot", {})), **info})
    layer.customLayerData = data
    layer.Save()
    # `still` is not carried over: the output carries motion (rig.frames, at least two frames as read_rig requires)
    # and is therefore not a still. A 「标准人」 packet is a still until motion is applied; if the flag were kept,
    # downstream frame-range computation would treat the result as a still unrelated to the shot (data/frames.py union).
    keep = {k: v for k, v in src.meta.items() if k not in ("frames", "keys", "still")}
    return scene_packet(out, rig.frames, src.type, keys=list(info.get("keys", [])), **keep)


# --------------------------------------------------------------------------- the model's motion back on the rig


def _from_model(rig: Rig, span: np.ndarray, at: np.ndarray, result: dict) -> tuple[np.ndarray, list[int]]:
    """The model's motion (lab2shot_worker.rig_motion.read_result: world rotations of the rig joints it moved and the
    root joint's world position, on its own timeline) as the rig's locals [n,J,4,4] over the frames `span` picks,
    `at` saying where each of those frames sits on the model's timeline. Joints the model does not have keep their own
    animation, so fingers, face and twist joints are untouched. Returns (locals, the rig joints the model moved)."""
    placement = rig.placement[span]
    inv = np.linalg.inv(placement)
    timeline = np.arange(len(result["rotations"]))
    rot = mo.resample_rotations(timeline, result["rotations"], at)
    rot = mo.orthonormal(inv[:, None, :3, :3]) @ rot  # into the skeleton's space
    root_world = mo.resample_values(timeline, result["root"], at)
    root = (inv[:, :3, :3] @ root_world[..., None])[..., 0] + inv[:, :3, 3]
    joints = [int(j) for j in result["joints"]]
    root_joint = int(result["info"]["root_joint"])
    part = mo.set_world(rig.local[span], rig.parents, {j: rot[:, i] for i, j in enumerate(joints)}, {root_joint: root})
    return part, joints


def model_at(rig: Rig, sent: list[int], result: dict) -> np.ndarray:
    """Where each of the rig's frames inside the sent span sits on the model's timeline."""
    frames = np.asarray(rig.frames)
    span = (frames >= sent[0]) & (frames <= sent[-1])
    return np.interp(frames[span], sent, result["keys"])


def on_rig(rig: Rig, keys: list[int], result: dict, exact: bool,
           feet: list[tuple[int, int, int]] | None) -> tuple[np.ndarray, dict]:
    """动作补帧: a model's motion between an animator's keys as the rig's locals [F,J,4,4] at its frames.

    Between the first and the last key the model's result is resampled to the rig's frame rate (`_from_model`): a key
    is the model frame it was put on, the rig's frames between two keys spread evenly over the model's.
    `exact`: at every key the difference to the animator's pose is taken out, blended smoothly towards the
    neighbouring keys (key_residuals), so every key is the animator's pose exactly. `feet` [(thigh, shin, foot)]:
    during the model's foot contacts the ankle is pinned with two-bone IK (keys themselves never move). Before the
    first and after the last key the rig keeps its own animation. Returns (locals, measurements: the distance to the
    keys of the model's result and of the final one, model_at_keys_cm and result_at_keys_cm)."""
    frames = np.asarray(rig.frames)
    span = (frames >= keys[0]) & (frames <= keys[-1])
    at = model_at(rig, keys, result)
    placement = rig.placement[span]
    timeline = np.arange(len(result["rotations"]))
    local = rig.local.copy()
    part, joints = _from_model(rig, span, at, result)
    k = rig.index(keys)
    raw = part[np.searchsorted(frames[span], keys)]
    stats = {"keys": [int(f) for f in keys], "model_at_keys_cm": _pose_error(rig, raw, rig.local[k], joints)}
    if exact:
        rotation, offset = mo.key_residuals(rig.local[k], raw)
        part = mo.apply_residuals(frames[span], np.asarray(keys), rotation, offset, part)
    if feet:
        contacts = mo.resample_values(timeline, result["contacts"], at) > 0.5
        part = _pin_feet(rig, part, placement, feet, contacts, np.isin(frames[span], keys))
    local[span] = part
    stats["result_at_keys_cm"] = _pose_error(rig, local[k], rig.local[k], joints)
    return local, stats


def blend_locals(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Joint-to-parent transforms [F,J,4,4] a -> b by `weight` [F] (0 = a, 1 = b): rotations along the short way round
    (slerp), positions straight. Blending the matrices themselves would shrink and shear the bones."""
    w = np.asarray(weight, np.float64)[:, None]
    out = np.asarray(a, np.float64).copy()
    qa, qb = mo.matrix_to_quat(mo.orthonormal(a[..., :3, :3])), mo.matrix_to_quat(mo.orthonormal(b[..., :3, :3]))
    out[..., :3, :3] = mo.quat_to_matrix(mo.slerp(qa, qb, np.broadcast_to(w, qa.shape[:2])))
    out[..., :3, 3] = a[..., :3, 3] + w[..., None] * (b[..., :3, 3] - a[..., :3, 3])
    return out


def repaired(rig: Rig, sent: list[int], result: dict, keep: np.ndarray | None, ramp: int = 3) -> tuple[np.ndarray, dict]:
    """动作清理: a model's repaired motion as the rig's locals [F,J,4,4] at its frames.

    Every frame the node sent is transferred as in 动作补帧 (`_from_model`), resampled from the model's frame rate
    to the rig's. `keep` [F] (over the frames sent, True = the frame needs no repair): those frames keep the incoming
    animation, and the repair is eased in over `ramp` frames on each side so that a repaired stretch joins its
    untouched neighbours without a discontinuity. Only faulty frames are modified, consistent with StableMotion's own
    pipeline, which keeps the good frames and inpaints the rest; a model that does not classify frames passes None and
    every frame is replaced.

    Returns (locals, measurements: how many frames were touched, and how far the touched ones moved in cm)."""
    frames = np.asarray(rig.frames)
    span = (frames >= sent[0]) & (frames <= sent[-1])
    part, joints = _from_model(rig, span, model_at(rig, sent, result), result)
    if keep is not None:
        weight = (~np.asarray(keep, bool)).astype(np.float64)
        if ramp > 0:  # ease the repair in and out of the frames it leaves alone
            eased = weight.copy()
            for f in np.flatnonzero(weight > 0):
                for d in range(1, ramp + 1):
                    for n in (f - d, f + d):
                        if 0 <= n < len(weight) and weight[n] == 0.0:
                            eased[n] = max(eased[n], float(mo.smoothstep(1 - d / (ramp + 1))))
            weight = eased
        part = blend_locals(rig.local[span], part, weight)
    else:
        weight = np.ones(int(span.sum()))
    local = rig.local.copy()
    local[span] = part
    touched = weight > 0
    gap = np.linalg.norm(mo.world_from_local(local[span], rig.parents)[..., :3, 3]
                         - mo.world_from_local(rig.local[span], rig.parents)[..., :3, 3], axis=-1)
    return local, {"frames_sent": len(sent), "frames_changed": int(touched.sum()),
                   "moved_cm": {"mean": float(gap[touched].mean()) if touched.any() else 0.0,
                                "max": float(gap.max()) if gap.size else 0.0}}


def _pose_error(rig: Rig, a: np.ndarray, b: np.ndarray, joints: list[int]) -> dict:
    """How far two sets of poses are apart (cm, skeleton space): the mean over `joints` and the largest of any joint."""
    gap = np.linalg.norm(mo.world_from_local(a, rig.parents)[..., :3, 3] - mo.world_from_local(b, rig.parents)[..., :3, 3], axis=-1)
    return {"mean": float(gap[:, joints].mean()), "max": float(gap.max())}


def _pin_feet(rig: Rig, local: np.ndarray, placement: np.ndarray, feet: list[tuple[int, int, int]],
              contacts: np.ndarray, is_key: np.ndarray, ramp: int = 3) -> np.ndarray:
    """Foot contacts [F,4] (left heel, left toe, right heel, right toe): while a heel is down its ankle stays where
    it is at a key inside that contact (else where it is on average), eased in and out over `ramp` frames."""
    world = placement[:, None] @ mo.world_from_local(local, rig.parents)
    for (thigh, shin, foot), heel in zip(feet, (contacts[:, 0], contacts[:, 2])):
        target = world[:, foot, :3, 3].copy()
        weight = np.zeros(len(local))
        for a, b in mo.contact_segments(heel):
            inside = np.arange(a, b + 1)
            keyed = inside[is_key[inside]]
            spot = world[keyed[0] if len(keyed) else inside, foot, :3, 3].reshape(-1, 3).mean(0)
            lo, hi = max(0, a - ramp), min(len(local) - 1, b + ramp)
            for f in range(lo, hi + 1):
                w = 1.0 if a <= f <= b else float(mo.smoothstep(1 - (a - f if f < a else f - b) / (ramp + 1)))
                if w > weight[f]:
                    weight[f], target[f] = w, spot
        weight[is_key] = 0.0
        on = weight > 0
        if not on.any():
            continue
        goal = world[on, foot, :3, 3] + weight[on, None] * (target[on] - world[on, foot, :3, 3])
        hip, knee, ankle = (world[on, j, :3, 3] for j in (thigh, shin, foot))
        turn_thigh, turn_shin = mo.leg_ik(hip, knee, ankle, goal)
        rotations = {thigh: turn_thigh @ mo.orthonormal(world[on, thigh, :3, :3]),
                     shin: turn_shin @ mo.orthonormal(world[on, shin, :3, :3]),
                     foot: mo.orthonormal(world[on, foot, :3, :3])}  # the foot keeps its orientation
        inv = np.linalg.inv(placement[on])
        local_on = mo.set_world(local[on], rig.parents, {j: mo.orthonormal(inv[:, :3, :3]) @ r for j, r in rotations.items()})
        local[on] = local_on
        world[on] = placement[on][:, None] @ mo.world_from_local(local_on, rig.parents)
    return local
