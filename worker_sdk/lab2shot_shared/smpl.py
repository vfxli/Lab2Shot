"""The SMPL family's skeleton, and the conversion between its parameters and a skeletal animation, both directions.
Numpy only, one implementation for the core and for every worker.

SMPL, SMPL-H and SMPL-X are one body model in three sizes: the same 22 body joints, then hands (SMPL-H, SMPL-X) and
a jaw and two eyes (SMPL-X). A method that speaks any of them hands over the same three things:

    pose      [F,J,3] axis-angle, one rotation per joint, relative to its parent (the root's is the body's
              orientation in the world). The parts a method computes separately (global_orient, body_pose, the two
              hands, the jaw ...) are one array here: `pack` puts them together in the model's order, `split` takes
              them apart again. A part a method does not solve stays at rest (zeros).
    transl    [F,3] where the body is, as the model defines it: the root joint sits at transl + rest_joints[0]
    rest      [J,3] the joints of this body's shape (betas) standing in the rest pose. It comes from the body model
              file the user downloaded (worker side) or from a solved person's npz; nothing here needs that file,
              so the core can convert without it.

`to_motion` turns those into joint-to-world transforms [F,J,4,4] (a skeletal animation, the form every DCC works in),
and `from_motion` turns joint-to-world transforms back into parameters. They are exact inverses.

A rig that is not an SMPL body (an animator's production rig, other joints, other bone lengths, another rest
pose) reaches the same place through the retargeting in motion.py, and this module meets it there:

    rig -> SMPL   Retarget.align(skeleton(body, rest), rig) -> to_model(rig poses) -> to_params  -> pose, transl
    SMPL -> rig   pose, transl -> from_params -> Retarget.to_production            -> the rig's joints

Thus 骨架动画 ↔ SMPL 参数 is motion.Retarget for the rig mismatch plus this module for the parameters: two halves,
one implementation each, and nothing else implements either.

The rest orientations of every SMPL joint are the identity (the model rotates each joint about its rest position),
which is why a joint's local rotation is its pose parameter. What Lab2Shot delivers to an animator carries CG bone
names and CG joint axes instead: that is lab2shot/data/joints.py cg_names and data/skeleton.py cg_orientations,
applied after this conversion, never inside it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .motion import Skeleton, local_from_world, matrix_to_rotvec, orthonormal, rotvec_to_matrix, world_from_local
from .protocol import Failure

# SMPL-X's 55 joints, in the model's own order, checked against SMPLX_NEUTRAL.npz's kintree_table and the smplx
# package's joint_names.JOINT_NAMES. SMPL and SMPL-H are cut from it below: the model is one family, one table.
# Left and right are the person's own side. Note the offset to CG naming: "left_shoulder" turns the upper arm,
# "left_hip" the thigh, "left_foot" the toes; the CG names are in lab2shot/data/joints.py.
_SMPLX_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", "right_ankle",
    "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "jaw", "left_eye_smplhf", "right_eye_smplhf",
    *(f"{side}_{finger}{k}" for side in ("left", "right") for finger in ("index", "middle", "pinky", "ring", "thumb")
      for k in (1, 2, 3)),
)
_SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
    15, 15, 15,  # the jaw and the two eyes hang off the head
    # each finger's first joint hangs off its wrist (left 20, right 21), the next two off the one before
    *(p for wrist, first in ((20, 25), (21, 40)) for f in range(5)
      for p in (wrist, first + 3 * f, first + 3 * f + 1)),
)

_FACE = tuple(_SMPLX_NAMES.index(n) for n in ("jaw", "left_eye_smplhf", "right_eye_smplhf"))  # SMPL-H has none of these


def _without(names, parents, drop: tuple[int, ...]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """The same skeleton with `drop`'s joints (and nothing below them) taken out, the remaining joints renumbered."""
    keep = [j for j in range(len(names)) if j not in drop]
    at = {j: i for i, j in enumerate(keep)}
    return tuple(names[j] for j in keep), tuple(at[parents[j]] if parents[j] >= 0 else -1 for j in keep)


_SMPLH_NAMES, _SMPLH_PARENTS = _without(_SMPLX_NAMES, _SMPLX_PARENTS, _FACE)


@dataclass(frozen=True)
class Body:
    """One member of the family: what its joints are called, who each one's parent is, and how a method's separate
    pose parts line up into one [F,J,3] array (`layout`: each part's name and how many joints it holds, the root
    first, adding up to every joint)."""

    id: str
    title: str
    names: tuple[str, ...]
    parents: tuple[int, ...]
    layout: tuple[tuple[str, int], ...]

    @property
    def joints(self) -> int:
        return len(self.names)


BODIES: Mapping[str, Body] = MappingProxyType({
    b.id: b for b in (
        # SMPL's last two joints are a hand joint each (no fingers), where SMPL-X starts its thumb chain
        Body("smpl", "SMPL", (*_SMPLX_NAMES[:22], "left_hand", "right_hand"), (*_SMPLX_PARENTS[:22], 20, 21),
             (("global_orient", 1), ("body_pose", 23))),
        Body("smplh", "SMPL-H", _SMPLH_NAMES, _SMPLH_PARENTS,
             (("global_orient", 1), ("body_pose", 21), ("left_hand_pose", 15), ("right_hand_pose", 15))),
        Body("smplx", "SMPL-X", _SMPLX_NAMES, _SMPLX_PARENTS,
             (("global_orient", 1), ("body_pose", 21), ("jaw_pose", 1), ("leye_pose", 1), ("reye_pose", 1),
              ("left_hand_pose", 15), ("right_hand_pose", 15))),
    )
})


def body(which: str | Body) -> Body:
    """The body model by name ("smpl", "SMPL-X", "smplx" ...); a Body passes through."""
    if isinstance(which, Body):
        return which
    key = str(which).strip().lower().replace("-", "").replace("_", "")
    if key not in BODIES:
        raise Failure("E-SMPL-UNKNOWNBODY", model=str(which), models=[b.title for b in BODIES.values()])
    return BODIES[key]


# --------------------------------------------------------------------------- the parameter vector


def pack(which: str | Body, **parts: np.ndarray) -> np.ndarray:
    """A method's separate pose parts -> one [F,J,3] axis-angle array in the model's joint order. A part the method
    does not solve may be left out and stays at rest (SMPL-X's jaw and eyes for a body-only method). Each part is
    [F,n,3] or flat [F,n*3]."""
    b = body(which)
    frames = next((len(np.asarray(v)) for v in parts.values() if v is not None), 0)
    known = {name for name, _ in b.layout}
    extra = sorted(set(parts) - known)
    if extra:
        raise Failure("E-SMPL-UNKNOWNPART", model=b.title, parts=extra, known=sorted(known))
    out = []
    for name, count in b.layout:
        v = parts.get(name)
        if v is None:
            out.append(np.zeros((frames, count, 3)))
            continue
        v = np.asarray(v, np.float64).reshape(len(np.asarray(v)), -1, 3)
        if v.shape[1] != count:
            raise Failure("E-SMPL-PARTSIZE", model=b.title, part=name, want=count, have=int(v.shape[1]))
        out.append(v)
    return np.concatenate(out, axis=1)


def split(which: str | Body, pose: np.ndarray) -> dict[str, np.ndarray]:
    """One [F,J,3] (or flat [F,J*3]) axis-angle array -> the model's named parts, each [F,n,3]."""
    b = body(which)
    p = _pose(b, pose)
    out, at = {}, 0
    for name, count in b.layout:
        out[name] = p[:, at:at + count]
        at += count
    return out


def _pose(b: Body, pose: np.ndarray) -> np.ndarray:
    p = np.asarray(pose, np.float64)
    p = p.reshape(len(p), -1, 3)
    if p.shape[1] != b.joints:
        raise Failure("E-SMPL-POSESIZE", model=b.title, want=b.joints, have=int(p.shape[1]))
    return p


# --------------------------------------------------------------------------- the skeleton, both directions


def world_of(parents, rest_joints: np.ndarray, rotations: np.ndarray, root_position: np.ndarray) -> np.ndarray:
    """Joint-to-world [F,J,4,4] of the SMPL skeleton: every joint turns about its rest position by `rotations`
    [F,J,3,3] (its rotation relative to its parent) and keeps its rest offset from its parent; the root joint sits at
    `root_position` [F,3]. The units are the caller's; rest_joints and root_position only have to agree."""
    rest = np.asarray(rest_joints, np.float64)
    parents = np.asarray(parents, np.int64)
    rot = np.asarray(rotations, np.float64)
    local = np.zeros((len(rot), len(parents), 4, 4))
    local[..., 3, 3] = 1.0
    local[..., :3, :3] = rot
    local[:, 1:, :3, 3] = rest[1:] - rest[parents[1:]]  # every joint keeps its rest offset from its parent
    local[:, 0, :3, 3] = np.asarray(root_position, np.float64)
    return world_from_local(local, parents)


def local_rotations_of(parents, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The exact inverse of `world_of`: joint-to-world [F,J,4,4] -> (local rotations [F,J,3,3], root_position
    [F,3])."""
    w = np.asarray(world, np.float64)
    return orthonormal(local_from_world(w, np.asarray(parents, np.int64))[..., :3, :3]), w[:, 0, :3, 3]


def to_motion(which: str | Body, pose: np.ndarray, transl: np.ndarray, rest_joints: np.ndarray) -> np.ndarray:
    """SMPL parameters -> a skeletal animation: joint-to-world [F,J,4,4] in the units of `rest_joints` / `transl`."""
    b = body(which)
    rest = np.asarray(rest_joints, np.float64)
    return world_of(b.parents, rest, rotvec_to_matrix(_pose(b, pose)), np.asarray(transl, np.float64) + rest[0])


def from_motion(which: str | Body, world: np.ndarray, rest_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A skeletal animation -> SMPL parameters: joint-to-world [F,J,4,4] of this body's skeleton (its rest
    joints, its joint order) -> (pose [F,J,3] axis-angle, transl [F,3]). A rig with other joints or other bone
    lengths goes through motion.Retarget first (`to_params` below)."""
    b = body(which)
    w = np.asarray(world, np.float64)
    if w.shape[1] != b.joints:
        raise Failure("E-SMPL-JOINTCOUNT", model=b.title, want=b.joints, have=int(w.shape[1]))
    local, root = local_rotations_of(b.parents, w)
    return matrix_to_rotvec(local), root - np.asarray(rest_joints, np.float64)[0]


# --------------------------------------------------------------------------- the same two ways, for the retargeting
# motion.Retarget speaks in a model's world rotations plus its root's position (that is what a production rig's
# motion arrives as, and what it takes back). These two turn that into parameters and back, so a rig that is not an
# SMPL body needs no conversion of its own: Retarget.to_model -> to_params, from_params -> Retarget.to_production.


def to_params(which: str | Body, rotations: np.ndarray, root_position: np.ndarray,
              rest_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World rotations [F,J,3,3] of this body's joints and its root joint's world position [F,3] (what
    motion.Retarget.to_model hands over) -> (pose [F,J,3] axis-angle, transl [F,3])."""
    b = body(which)
    r = np.asarray(rotations, np.float64)
    if r.shape[1] != b.joints:
        raise Failure("E-SMPL-JOINTCOUNT", model=b.title, want=b.joints, have=int(r.shape[1]))
    local = local_from_world(orthonormal(r), np.asarray(b.parents, np.int64))
    return matrix_to_rotvec(local), np.asarray(root_position, np.float64) - np.asarray(rest_joints, np.float64)[0]


def from_params(which: str | Body, pose: np.ndarray, transl: np.ndarray,
                rest_joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of `to_params`: SMPL parameters -> (world rotations [F,J,3,3], root position [F,3]), what
    motion.Retarget.to_production takes to put the motion back on a production rig."""
    b = body(which)
    world = world_from_local(rotvec_to_matrix(_pose(b, pose)), np.asarray(b.parents, np.int64))
    return world, np.asarray(transl, np.float64) + np.asarray(rest_joints, np.float64)[0]


def skeleton(which: str | Body, rest_joints: np.ndarray) -> Skeleton:
    """This body standing in its rest pose, as the retargeting sees it (motion.Skeleton): every joint at its rest
    position with the model's own axes (the identity), so Retarget.align can match it to a production rig."""
    b = body(which)
    rest = np.repeat(np.eye(4)[None], b.joints, 0)
    rest[:, :3, 3] = np.asarray(rest_joints, np.float64)
    return Skeleton(list(b.names), np.asarray(b.parents, np.int64), rest)
