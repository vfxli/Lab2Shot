"""Character animation in scene packets (UsdSkel): reading a skeleton's bind pose, its per-frame animation and the
frames an animator keyed (and, for a display of one pose, just its bind pose: bind_skeleton); writing new animation
over a scene while keeping the rest of it (or one skeleton alone: skeleton_animation); and transferring a model's
motion back onto the rig. 动作补帧 (`on_rig`) and 动作清理 (`repaired`) share the same transfer (`_from_model`:
resample the model's timeline onto the rig's frames and set the world rotations of the affected joints) and differ
only in the final step."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lab2shot_shared import motion as mo
from pxr import Usd, UsdGeom, UsdSkel, Vt

from ..errors import Invalid
from ..messages import Msg
from .payloads import SCENE_FILE, scene_packet
from .packet import Packet


def _matrices(values) -> np.ndarray:
    """USD matrices (row vectors) -> [N,4,4] column-vector matrices."""
    return np.transpose(np.array([np.array(m) for m in values], np.float64).reshape(-1, 4, 4), (0, 2, 1))


def skeletons(packet: Packet) -> list[dict]:
    """The skeletons in a scene packet, in scene order: prim path, joint names and parents."""
    from ..io import usd
    from .scene import open_scene

    stage = open_scene([packet])  # kept alive while its prims are read
    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            joints = skel.GetJointsAttr().Get() or []
            out.append({"path": str(prim.GetPath()), "joints": usd.joint_names(skel),
                        "parents": list(UsdSkel.Topology(joints).GetParentIndices())})
    return out


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
    joints and rest pose) pass 1; this covers the target of 「动作重定向」 and the character of 「线性蒙皮变形」, whose
    motion arrives through another input, and a freshly auto-rigged character, which consists of a single static
    frame."""
    from ..io import usd
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
    from ..io import usd

    names = usd.joint_names(skel)
    bind = _bind_world(skel, parents)
    local = np.stack([_matrices(query.ComputeJointLocalTransforms(Usd.TimeCode(f))) for f in frames])
    xf = UsdGeom.Xformable(prim)
    placement = np.stack([np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode(f))).T for f in frames])
    anim = query.GetAnimQuery()
    samples = anim.GetJointTransformTimeSamples() if anim else []
    keys = sorted({int(round(t)) for t in samples} & set(frames))
    return Rig(str(prim.GetPath()), names, parents, bind, frames, local, placement, keys,
               str(anim.GetPrim().GetPath()) if anim else "")


def sole_height(packet: Packet, rig: Rig, feet: list[int], up=(0.0, 1.0, 0.0)) -> float | None:
    """How far the lowest of `feet` (joint indices; the ankles) sits above the character's sole in the bind pose, in
    cm along `up` — the body's own up in that pose (motion.Body.axes; a character imported Z-up or lying on its side
    has its soles below its ankles along its legs, not along the world's Y): the lowest point of the meshes skinned to this skeleton (UsdSkel's own bindings, data/evaluate.py
    skin_bindings: a mesh's own skel:skeleton or one it inherits; a mesh bound to no skeleton is no sole), mesh and
    joints both in the bind space (geomBindTransform, bindTransforms) placed in the world as on the rig's first frame,
    as Rig.skeleton places the bind pose — the Skeleton prim may carry an import's unit scale or up axis, or an
    upstream 「3D 变换」. None for a skeleton without meshes (a BVH, a solve's joints). Retargeting stands two
    characters sole to sole, not ankle to ankle: AccuRIG's ankle is higher over its sole than SMPL's."""
    from .evaluate import skin_bindings
    from .scene import open_scene

    if not feet:
        return None
    stage = open_scene([packet])
    place = rig.placement[0]
    up = np.asarray(up, np.float64) / np.linalg.norm(up)
    lows = []
    for binding, _ in skin_bindings(stage):
        if str(binding.GetSkeleton().GetPrim().GetPath()) != rig.path:
            continue
        for target in binding.GetSkinningTargets():
            prim = target.GetPrim()
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get(Usd.TimeCode.Default()) if prim.IsA(UsdGeom.Mesh) else None
            if not points:
                continue
            m = place @ np.asarray(target.GetGeomBindTransform(), np.float64).T
            lows.append(float(((np.asarray(points, np.float64) @ m[:3, :3].T + m[:3, 3]) @ up).min()))
    if not lows:
        return None
    return float(min((place @ rig.bind[j])[:3, 3] @ up for j in feet) - min(lows))


def _bind_world(skel: UsdSkel.Skeleton, parents: np.ndarray) -> np.ndarray:
    """The skeleton's bind pose, joint-to-world [J,4,4]: its bindTransforms, else its restTransforms taken to the
    world (a skeleton written without a bind pose). Neither, or not one per joint: E-ANIM-NOBIND naming the prim —
    nothing a skeleton could be posed from, said as such rather than a TypeError from deep in the maths."""
    for attr, to_world in ((skel.GetBindTransformsAttr(), False), (skel.GetRestTransformsAttr(), True)):
        got = attr.Get()
        if got is not None and len(got) == len(parents):
            return mo.world_from_local(_matrices(got), parents) if to_world else _matrices(got)
    raise Invalid(Msg("E-ANIM-NOBIND", path=str(skel.GetPrim().GetPath())))


def bind_skeleton(packet: Packet, path: str | None = None) -> dict:
    """A skeleton's bind pose read without sampling its animation (read_rig computes every frame's joint transforms,
    which a display of one pose does not need): the Skeleton prim's path, its joint names and parents, and every
    joint's bind pose joint-to-world [J,4,4] (cm), placed by the prim's transform on the packet's first frame. {} when
    the packet has no skeleton (or none at `path`)."""
    from ..io import usd
    from .scene import open_scene

    stage = open_scene([packet])
    prims = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    prim = next((p for p in prims if str(p.GetPath()) == path), None) if path else (prims[0] if prims else None)
    if prim is None:
        return {}
    skel = UsdSkel.Skeleton(prim)
    topology = UsdSkel.Topology(skel.GetJointsAttr().Get() or [])
    parents = np.asarray(topology.GetParentIndices(), np.int64)
    frames = packet.meta.get("frames") or [0]
    placement = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode(int(frames[0])))).T
    return {"path": str(prim.GetPath()), "names": usd.joint_names(skel), "parents": [int(p) for p in parents],
            "world": placement @ _bind_world(skel, parents)}


def placement_at(packet: Packet, path: str, frames) -> np.ndarray:
    """The Skeleton prim's transform to the world [F,4,4] at `frames` (which need not be the packet's own: a target
    of 「动作重定向」 standing on one frame gets animation on all of the motion's; a prim moved by an animated
    「3D 变换」 is where it is on each of them)."""
    from .scene import open_scene

    stage = open_scene([packet])
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    return np.stack([np.array(xf.ComputeLocalToWorldTransform(Usd.TimeCode(int(f)))).T for f in frames])


def same_hierarchy(anim: Rig, character: Rig) -> list[int]:
    """「线性蒙皮变形」's check that a skeleton animation belongs to a character's skeleton: the same joint names (each
    once) and every joint under the same parent, by name. Only checked, never guessed — no matching by position or
    body part, no prefixes dropped, no case ignored. Returns, per joint of the character, the animation's joint of
    that name (the two may list their joints in different orders). Raises E-SKIN-HIERARCHY with what differs."""
    a, b = list(anim.names), list(character.names)

    def parent_names(rig: Rig) -> dict[str, str | None]:
        return {n: (rig.names[int(p)] if int(p) >= 0 else None) for n, p in zip(rig.names, rig.parents)}

    only_anim = [n for n in a if n not in set(b)]
    only_char = [n for n in b if n not in set(a)]
    twice = sorted({n for n in a if a.count(n) > 1} | {n for n in b if b.count(n) > 1})
    pa, pb = parent_names(anim), parent_names(character)
    moved = [f"{n}（{pa[n] or '根'} ≠ {pb[n] or '根'}）" for n in b if n in pa and pa[n] != pb[n]]
    if only_anim or only_char or twice or moved:
        def few(names: list[str]) -> str:
            return "、".join(names[:5]) + (f" 等 {len(names)} 个" if len(names) > 5 else "") if names else "无"

        raise Invalid(Msg("E-SKIN-HIERARCHY", only_anim=few(only_anim), only_char=few(only_char),
                          parents=few(moved), twice=few(twice)))
    return [a.index(n) for n in b]


def skeleton_animation(src: Packet, out: Path, rig: Rig, local: np.ndarray, info: dict) -> Packet:
    """A 骨架动画 of one skeleton of `src` (「动作重定向」's result): that Skeleton prim with its ancestors (they carry
    its placement) and a new animation of `local` [F,J,4,4] on `rig.frames`; the meshes, the other skeletons and
    everything else of the scene are left out, as 「提取骨架」 leaves the meshes out (data/scene.py keep_only).
    A new animation prim, not the old one rewritten: the target's own animation may have samples on other frames."""
    from ..io import usd
    from .scene import keep_only, open_scene

    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, rig.frames)
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(rig.path))
    binding = UsdSkel.BindingAPI.Apply(skel.GetPrim())
    name, n = "lab2shot_anim", 1
    while stage.GetPrimAtPath(skel.GetPath().GetParentPath().AppendChild(name)):
        n += 1
        name = f"lab2shot_anim{n}"
    anim = UsdSkel.Animation.Define(stage, skel.GetPath().GetParentPath().AppendChild(name))
    binding.CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    keep_only(stage, [skel.GetPath(), anim.GetPath()])
    usd.write_joint_samples(anim, skel.GetJointsAttr().Get(), local, rig.frames)
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData or {})
    data["lab2shot"] = usd.layer_data({**dict(data.get("lab2shot", {})), **info})
    layer.customLayerData = data
    out.mkdir(parents=True, exist_ok=True)
    layer.Export(str(out / SCENE_FILE))
    return scene_packet(out, rig.frames, "scene.skeleton",
                        **{k: v for k, v in src.meta.items() if k in ("width", "height")})


def blend_shapes(packet: Packet, path: str) -> list[str]:
    """The blend shapes of the meshes skinned to the skeleton at `path` (UsdSkel bindings, data/evaluate.py
    skin_bindings), each name once, in the meshes' order: what 「表情重定向（ARKit52）」 can drive on a character."""
    from .evaluate import skin_bindings
    from .scene import open_scene

    stage = open_scene([packet])
    out: list[str] = []
    for binding, _ in skin_bindings(stage):
        if str(binding.GetSkeleton().GetPrim().GetPath()) != path:
            continue
        for target in binding.GetSkinningTargets():
            out += [str(n) for n in UsdSkel.BindingAPI(target.GetPrim()).GetBlendShapesAttr().Get() or []]
    return list(dict.fromkeys(out))


def blend_weights(packet: Packet, path: str | None = None) -> tuple[str, list[str], np.ndarray]:
    """A character's blend-shape animation as curves (「表情重定向（ARKit52）」 with a character for its expressions): (the
    skeleton's path, the animation's shape names, weights [F,K] on the packet's frames). The skeleton at `path`, or
    the first whose animation has blend shapes; ("", [], empty) when none has."""
    from .scene import open_scene

    stage = open_scene([packet])
    frames = [int(f) for f in packet.meta.get("frames") or []]
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Skeleton) or (path and str(prim.GetPath()) != path):
            continue
        anim = UsdSkel.Cache().GetSkelQuery(UsdSkel.Skeleton(prim)).GetAnimQuery()
        names = [str(n) for n in anim.GetBlendShapeOrder()] if anim else []
        if names:
            weights = np.stack([np.asarray(anim.ComputeBlendShapeWeights(Usd.TimeCode(f)), np.float32) for f in frames])
            return str(prim.GetPath()), names, weights.reshape(len(frames), len(names))
    return "", [], np.zeros((len(frames), 0), np.float32)


def write_blend_weights(src: Packet, out: Path, path: str, frames: list[int], names: list[str], weights: np.ndarray,
                        info: dict) -> Packet:
    """The character with its blend shapes `names` moved by `weights` [F,K] at `frames` (「表情重定向（ARKit52）」): a layer over
    the input, as write_animation for the joints, whose weights replace the animation's own on every frame (shapes it
    does not name stay at 0, as UsdSkel reads a shape the animation leaves out). The joints keep their animation; a
    skeleton without one gets an animation with shapes only (its joints then hold the rest pose)."""
    from ..io import usd
    from .scene import _file, _new_layer_stage

    stage = _new_layer_stage(out / SCENE_FILE, [_file(src)], frames)
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(path))
    binding = UsdSkel.BindingAPI(skel.GetPrim())
    targets = binding.GetAnimationSourceRel().GetTargets()
    if targets:
        anim = UsdSkel.Animation(stage.GetPrimAtPath(targets[0]))
    else:
        anim = UsdSkel.Animation.Define(stage, skel.GetPath().GetParentPath().AppendChild("lab2shot_anim"))
        UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    anim.CreateBlendShapesAttr(Vt.TokenArray(list(names)))
    attr = anim.CreateBlendShapeWeightsAttr()
    for i, f in enumerate(frames):
        attr.Set(Vt.FloatArray.FromNumpy(np.asarray(weights[i], np.float32)), Usd.TimeCode(f))
    layer = stage.GetRootLayer()
    data = dict(layer.customLayerData or {})
    data["lab2shot"] = usd.layer_data({**dict(data.get("lab2shot", {})), **info})
    layer.customLayerData = data
    layer.Save()
    keep = {k: v for k, v in src.meta.items() if k not in ("frames", "keys", "still")}  # as write_animation
    return scene_packet(out, frames, src.type, **keep)


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
    rig's hip point, the middle of its thighs, on its own timeline) as the rig's locals [n,J,4,4] over the frames
    `span` picks, `at` saying where each of those frames sits on the model's timeline. Joints the model does not have
    keep their own animation, so fingers, face and twist joints are untouched. The rig is placed by its hip point once
    every rotation is set (motion.set_hips). Returns (locals, the rig joints the model moved)."""
    placement = rig.placement[span]
    inv = np.linalg.inv(placement)
    timeline = np.arange(len(result["rotations"]))
    rot = mo.resample_rotations(timeline, result["rotations"], at)
    rot = mo.orthonormal(inv[:, None, :3, :3]) @ rot  # into the skeleton's space
    root_world = mo.resample_values(timeline, result["root"], at)
    root = (inv[:, :3, :3] @ root_world[..., None])[..., 0] + inv[:, :3, 3]
    joints = [int(j) for j in result["joints"]]
    root_joint = int(result["info"]["root_joint"])
    thighs = tuple(int(t) for t in result["info"]["thighs"])  # the hip point (result "root") is their middle
    part = mo.set_hips(rig.local[span], rig.parents, {j: rot[:, i] for i, j in enumerate(joints)}, root_joint, thighs,
                       root)
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
