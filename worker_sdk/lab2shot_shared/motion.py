"""Skeletal motion math, numpy only, shared by the core and the motion workers (one implementation for both
sides). Rotations are unit quaternions (w, x, y, z) or 3x3 matrices (column vectors, p' = R @ p); a hierarchy lists
parents before children (-1 for a root).

    rotations     matrix_to_quat, quat_to_matrix, quat_mul, quat_inv, slerp, continuous, orthonormal, axis_angle,
                  rotvec_to_matrix / matrix_to_rotvec (the axis-angle the SMPL family's parameters are written in:
                  the one pair, for the workers and the core alike)
    hierarchies   world_from_local, local_from_world (transforms or rotations), set_world
    retargeting   Skeleton, Retarget: a production rig's motion on a model's skeleton and back (rest-pose alignment,
                  length scale), exact both ways for the joints they share; BodyRetarget: one rig's motion on another
                  (「动作重定向」: chains of different lengths bend by length, chain_bones / arc_middles / bracket),
                  aims_toward (where each joint's bone points, for both), chain_lengths
    time          resample_rotations, resample_values, windows (a long shot in parts that share their boundary keys)
    keys          key_residuals, apply_residuals: a generated motion moved onto the animator's keys, C1 between keys
    feet          contact_segments, leg_ik (two-bone IK that pins the ankle and keeps the foot's orientation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protocol import Failure

UP = np.array([0.0, 1.0, 0.0])
ZERO_BONE = 1e-9  # a bone shorter than this has no direction (two joints on the same point); see Retarget.align

# --------------------------------------------------------------------------- rotations


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """[...,3,3] rotation matrices -> [...,4] unit quaternions (w, x, y, z) with w >= 0."""
    r = np.asarray(r, np.float64)
    shape = r.shape[:-2]
    r = r.reshape(-1, 3, 3)
    tr = np.trace(r, axis1=1, axis2=2)
    cand = np.stack([1 + tr, 1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1 - r[:, 0, 0] + r[:, 1, 1] - r[:, 2, 2],
                     1 - r[:, 0, 0] - r[:, 1, 1] + r[:, 2, 2]], axis=1)
    best = np.argmax(cand, axis=1)
    q = np.empty((len(r), 4))
    for k in range(4):
        idx = best == k
        if not idx.any():
            continue
        m = r[idx]
        s = np.sqrt(np.maximum(cand[idx, k], 1e-12)) * 2  # = 4 * component k
        if k == 0:
            q[idx] = np.stack([s / 4, (m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 1, 0] - m[:, 0, 1]) / s], 1)
        elif k == 1:
            q[idx] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, s / 4, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s], 1)
        elif k == 2:
            q[idx] = np.stack([(m[:, 0, 2] - m[:, 2, 0]) / s, (m[:, 0, 1] + m[:, 1, 0]) / s, s / 4, (m[:, 1, 2] + m[:, 2, 1]) / s], 1)
        else:
            q[idx] = np.stack([(m[:, 1, 0] - m[:, 0, 1]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, s / 4], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.reshape(*shape, 4)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """[...,4] quaternions (w, x, y, z), normalised here -> [...,3,3] rotation matrices."""
    q = np.asarray(q, np.float64)
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b (apply b, then a), broadcast over leading dimensions."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_inv(q: np.ndarray) -> np.ndarray:
    """The inverse of unit quaternions (their conjugate)."""
    return np.asarray(q, np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def slerp(a: np.ndarray, b: np.ndarray, t) -> np.ndarray:
    """Spherical interpolation a -> b along the shorter arc; `t` broadcasts against the leading dimensions (t < 0 or
    > 1 extrapolates along the same arc)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    t = np.asarray(t, np.float64)[..., None]
    dot = np.sum(a * b, -1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin = np.sin(theta)
    small = sin < 1e-6
    safe = np.where(small, 1.0, sin)
    wa = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    wb = np.where(small, t, np.sin(t * theta) / safe)
    out = wa * a + wb * b
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def continuous(q: np.ndarray) -> np.ndarray:
    """[T,...,4] quaternion tracks with their signs flipped so that no frame jumps to the other hemisphere."""
    q = np.array(q, np.float64)
    if len(q) < 2:
        return q
    sign = np.sign(np.sum(q[1:] * q[:-1], -1))
    sign[sign == 0] = 1.0
    q[1:] *= np.cumprod(sign, axis=0)[..., None]
    return q


def orthogonal(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> the nearest orthogonal matrices (scale and shear removed), keeping a mirror: det -1 where `m`'s is
    negative (a joint scaled by -1 on one axis)."""
    u, _, vt = np.linalg.svd(np.asarray(m, np.float64))
    return u @ vt


def handedness(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> +1, or -1 where `m` mirrors (a negative determinant)."""
    return np.where(np.linalg.det(np.asarray(m, np.float64)) < 0, -1.0, 1.0)


def orthonormal(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> the rotations of `m` (scale and shear removed). A mirroring `m` is its rotation times a scale of -1
    along x — the one convention (signed_scales, usd.decompose, set_world): the rotation is its orthogonal part with the
    first column turned back, so that rotation @ diag(signed_scales) gives `m` again and the mirror is kept, not
    turned into a rotation."""
    q = orthogonal(m)
    return q * np.stack([handedness(q), np.ones(q.shape[:-2]), np.ones(q.shape[:-2])], -1)[..., None, :]


def decompose(m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[...,4,4] -> translations [...,3], rotations as quaternions [...,4] (w,x,y,z), signed scales [...,3]: the one
    TRS split (USD animation samples, retiming), a mirror kept as a negative x scale."""
    basis = np.asarray(m, np.float64)[..., :3, :3]
    return np.asarray(m, np.float64)[..., :3, 3], matrix_to_quat(orthonormal(basis)), signed_scales(basis)


def signed_scales(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> its scale along each axis [...,3] (the column lengths), negative along x where `m` mirrors:
    m = orthonormal(m) @ diag(signed_scales(m)) up to shear."""
    s = np.linalg.norm(np.asarray(m, np.float64), axis=-2)
    return s * np.stack([handedness(m), np.ones(s.shape[:-1]), np.ones(s.shape[:-1])], -1)


def axis_angle(axis: np.ndarray, angle) -> np.ndarray:
    """Rotation matrices [...,3,3] about unit `axis` [...,3] by `angle` (radians)."""
    axis = np.asarray(axis, np.float64)
    angle = np.asarray(angle, np.float64)[..., None, None]
    x, y, z = np.moveaxis(axis, -1, 0)
    zero = np.zeros_like(x)
    k = np.stack([np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)], -2)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """[...,3] rotation vectors (axis times angle in radians, what SMPL calls axis-angle) -> [...,3,3] rotations."""
    v = np.asarray(v, np.float64)
    angle = np.linalg.norm(v, axis=-1)
    safe = np.where(angle < 1e-12, 1.0, angle)
    return axis_angle(v / safe[..., None], np.where(angle < 1e-12, 0.0, angle))


def matrix_to_rotvec(r: np.ndarray) -> np.ndarray:
    """[...,3,3] rotations -> [...,3] rotation vectors, the shorter way round (angle <= pi): the exact inverse of
    rotvec_to_matrix, through the quaternion so that half a turn stays well behaved."""
    q = matrix_to_quat(r)  # w >= 0, so the angle is already in [0, pi]
    axis = q[..., 1:]
    sin_half = np.linalg.norm(axis, axis=-1)
    angle = 2 * np.arctan2(sin_half, q[..., 0])
    safe = np.where(sin_half < 1e-12, 1.0, sin_half)
    return np.where((sin_half < 1e-12)[..., None], 0.0, axis / safe[..., None] * angle[..., None])


def between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotations [...,3,3] taking directions a to b (both [...,3], any length)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis, axis=-1)
    c = np.sum(a * b, -1)
    fallback = np.cross(a, np.where(np.abs(a[..., :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))  # for opposite a, b
    axis = np.where((s < 1e-9)[..., None], fallback, axis)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis_angle(axis, np.arctan2(s, c))


# --------------------------------------------------------------------------- hierarchies


def world_from_local(local: np.ndarray, parents) -> np.ndarray:
    """[...,J,4,4] joint-to-parent -> joint-to-world (a root's local is its world); [...,J,3,3] rotations the same."""
    world = np.array(local, np.float64)
    for j, p in enumerate(parents):
        if p >= 0:
            world[..., j, :, :] = world[..., p, :, :] @ world[..., j, :, :]
    return world


def local_from_world(world: np.ndarray, parents) -> np.ndarray:
    """[...,J,4,4] joint-to-world -> joint-to-parent; [...,J,3,3] rotations the same."""
    world = np.asarray(world, np.float64)
    local = world.copy()
    for j, p in enumerate(parents):
        if p >= 0:
            local[..., j, :, :] = np.linalg.inv(world[..., p, :, :]) @ world[..., j, :, :]
    return local


def set_world(local: np.ndarray, parents, rotations: dict[int, np.ndarray],
              positions: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Locals [F,J,4,4] with some joints turned to given world rotations [F,3,3] (their scale kept) and some moved to
    given world positions [F,3]; every other joint keeps its local transform (so it follows its parent). `local`'s
    roots are relative to the world of the given rotations and positions."""
    local = np.array(local, np.float64)
    world = np.empty_like(local)
    positions = positions or {}
    for j, p in enumerate(parents):
        parent = world[:, p] if p >= 0 else np.broadcast_to(np.eye(4), world[:, j].shape)
        if j in rotations:
            # the joint's own scale kept, a mirror included: world = parent @ local, so with the parent's orthogonal
            # part P (a mirror in it too) the local is P^T @ rotation @ diag(the joint's signed scales in the world)
            scale = np.linalg.norm(local[:, j, :3, :3], axis=-2)  # column lengths
            mirror = handedness(parent[:, :3, :3]) * handedness(local[:, j, :3, :3])
            world_axes = np.asarray(rotations[j], np.float64) * \
                np.stack([mirror, np.ones_like(mirror), np.ones_like(mirror)], -1)[:, None, :]
            rot = np.swapaxes(orthogonal(parent[:, :3, :3]), -1, -2) @ world_axes
            local[:, j, :3, :3] = rot * scale[:, None, :]
        if j in positions:
            inv = np.linalg.inv(parent)
            local[:, j, :3, 3] = (inv[:, :3, :3] @ np.asarray(positions[j], np.float64)[..., None])[..., 0] + inv[:, :3, 3]
        world[:, j] = parent @ local[:, j]
    return local


def set_hips(local: np.ndarray, parents, rotations: dict[int, np.ndarray], move: int, thighs: tuple[int, int],
             at: np.ndarray) -> np.ndarray:
    """set_world with the body placed by its hip point: the middle of the two `thighs` goes to `at` [F,3] on every
    frame, by moving joint `move` (the hips, above both thighs). Where the thighs hang comes from the hierarchy after
    every rotation is set, not from an offset fixed on the hips: 3ds Max Biped hangs them under Spine, which the spine's
    bend turns, so a hip point taken as rigid on the hips slides the feet as the character bends over."""
    first = set_world(local, parents, rotations)
    world = world_from_local(first, parents)
    mid = (world[:, thighs[0], :3, 3] + world[:, thighs[1], :3, 3]) / 2
    return set_world(first, parents, {}, {move: world[:, move, :3, 3] + (np.asarray(at, np.float64) - mid)})


# --------------------------------------------------------------------------- retargeting


@dataclass
class Skeleton:
    """Joint names, parents (before children, the root first) and a rest pose, joint-to-world [J,4,4]: the character
    standing (its own up: Body.axes), the pose two skeletons are aligned in (a rig's bind pose, T or A; for a model a standing
    pose in its own joint frames, e.g. from its training data)."""

    names: list[str]
    parents: np.ndarray
    rest: np.ndarray

    def __post_init__(self) -> None:
        self.parents = np.asarray(self.parents, np.int64)
        self.rest = np.asarray(self.rest, np.float64)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def rest_positions(self) -> np.ndarray:
        return self.rest[:, :3, 3]

    @property
    def rest_rotations(self) -> np.ndarray:
        return orthonormal(self.rest[:, :3, :3])


def aim_frame(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """An orthonormal frame [3,3] (columns): x along `direction`, z the part of `reference` across it.

    The one place a bone's axes are built from a direction: the retargeting rest-pose alignment here and the CG joint
    orientations a delivered skeleton carries (lab2shot/data/skeleton.py cg_orientations) both use it."""
    x = direction / np.linalg.norm(direction)
    z = reference - np.dot(reference, x) * x
    z /= np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], axis=1)


TWIST_LEAST = 0.2  # a twist reference closer than this (the sine of its angle) to the bone gives it no roll: next one


def twist_refs(dm: np.ndarray, dp: np.ndarray, refs) -> tuple[np.ndarray, np.ndarray]:
    """The pair of directions (one per skeleton) a bone's roll is taken from when two rest poses are aligned bone by
    bone (aim_frame(dp, rp) @ aim_frame(dm, rm).T): `refs` = ((first_m, first_p), (second_m, second_p)), a body's
    forward then up (body_refs) or a hand's palm normal then fingers (hand_refs). The first, or the second for a bone
    that points more along the first in either skeleton; and when the pick lies along the bone in either skeleton
    (a rest pose that is no standing pose: a BVH's zero pose has the legs along the spine, so a leg is along "up" in
    one skeleton and along "forward" in the other) the other one, then the third axis across both — a reference
    along the bone gives aim_frame no roll at all. Retarget.align and BodyRetarget.align both take a bone's roll here,
    the one rule."""
    (fm, fp), (um, up) = refs
    along_first = max(abs(dm @ fm), abs(dp @ fp)) > max(abs(dm @ um), abs(dp @ up))
    first, second = ((um, up), (fm, fp)) if along_first else ((fm, fp), (um, up))
    for rm, rp in (first, second, (np.cross(fm, um), np.cross(fp, up))):
        if min(np.linalg.norm(np.cross(dm, rm)), np.linalg.norm(np.cross(dp, rp))) > TWIST_LEAST:
            return rm, rp
    return first


def upper_joints(parents, joints, thighs: tuple[int, int]) -> list[int]:
    """Of `joints`, the upper body's: every one but the root and those in the two thighs' subtrees."""
    def in_leg(j: int) -> bool:
        while j >= 0:
            if j in thighs:
                return True
            j = int(parents[j])
        return False

    return [j for j in joints if int(parents[j]) >= 0 and not in_leg(j)]


def _up(parents, j: int) -> list[int]:
    """`j` and its ancestors, `j` first."""
    out = []
    while j >= 0:
        out.append(j)
        j = int(parents[j])
    return out


def common_above(parents, joints) -> int:
    """The deepest joint every one of `joints` is itself or hangs under; -1 when they share none (or none are given)."""
    joints = [int(j) for j in joints]
    if not joints:
        return -1
    others = [set(_up(parents, j)) for j in joints[1:]]
    return next((a for a in _up(parents, joints[0]) if all(a in o for o in others)), -1)


def hips_joint(parents, named: int, legs, top: int) -> int:
    """The hips of a skeleton: the joint the body is moved by (set_hips), so both legs and the trunk hang below it.
    `named` (the joint a name or a mapping says) when it does; otherwise the nearest joint above both legs and the
    trunk's top `top` — a COG rig forks the legs from Pelvis and hangs the spine from COG above it, and moving Pelvis
    would leave the upper body behind. `legs`: joints of the two legs (thighs or feet); `top`: -1 when unknown. The one
    rule the part guess (data/joints.py guess), the mapping check (E-MAP-HIPS) and the bodies (trunk_of) read."""
    fork = common_above(parents, [*legs, *([top] if top >= 0 else [])])
    return named if named >= 0 and fork >= 0 and named in _up(parents, fork) else fork


def trunk_top(parents, thighs, landmarks=(), pos: np.ndarray | None = None) -> int:
    """Where a skeleton's trunk ends (its chest): the one definition the part guess, the mapping check (E-MAP-HIPS)
    and the bodies (trunk_of) read. With two or more `landmarks` (its hands and head, whichever are known), the deepest
    joint they all hang under; one alone says nothing of where the trunk ends (a hand alone would be its own top, the
    trunk running into its arm), so it is passed over. Otherwise from the hierarchy: going down a branch that hangs
    from the leg fork or a joint above it (never into a leg), along the child that holds the most, the first joint
    where a pair of limbs parts beside another branch (the neck): a shape hung by exactly two children of three
    joints or more (fingers come four or five alike, a skirt's strands as many) beside a child of another shape, the
    two lying mirrored about the joint (their branches' centroids equally far, spread sideways: a left and a right
    arm) when the rest positions
    `pos` are known. A tail, a skirt, or a pair of arms made unlike by a prop on one hand has no such joint,
    whatever its length. -1: none (the caller has no trunk and says W-BODY-NOTRUNK)."""
    if len(landmarks) >= 2:
        return common_above(parents, landmarks)
    parents = [int(p) for p in parents]
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(p, []).append(j)
    shapes: dict[int, tuple] = {}

    def shape(j: int) -> tuple:
        if j not in shapes:
            shapes[j] = tuple(sorted(shape(c) for c in kids.get(j, [])))
        return shapes[j]

    def size(j: int) -> int:
        return 1 + sum(size(c) for c in kids.get(j, []))

    if pos is not None:
        pos = np.asarray(pos, np.float64)

    def centroid(j: int) -> np.ndarray:
        todo, got = [j], []
        while todo:
            k = todo.pop()
            got.append(k)
            todo += kids.get(k, [])
        return pos[got].mean(axis=0)

    def mirrored(j: int, a: int, b: int) -> bool:
        # the two whole branches about the joint, by their centroids: their first joints may sit on the joint itself
        # (CMU's shoulders at OFFSET 0 0 0), where comparing the roots compares rounding noise
        u, v = centroid(a) - pos[j], centroid(b) - pos[j]
        la, lb = np.linalg.norm(u), np.linalg.norm(v)
        if max(la, lb) <= ZERO_BONE:
            return False
        # equally far (mirror images about the plane between them) and not lying on each other: arms folded up a BVH
        # zero pose's spine still part sideways by the shoulders' width
        return abs(la - lb) <= 0.25 * max(la, lb) and np.linalg.norm(u - v) >= 0.2 * max(la, lb)

    def parts_at(j: int) -> bool:
        # a shape hung by exactly two children of three joints or more (the arms: fingers come four or five alike, a
        # skirt's strands four by four), with the rest positions `pos` also mirrored about the joint, beside a child
        # of another shape (the neck)
        others = kids.get(j, [])
        twice = [c for c in others if size(c) >= 3 and sum(shape(d) == shape(c) for d in others) == 2]
        if pos is not None:
            twice = [c for c in twice if any(d != c and shape(d) == shape(c) and mirrored(j, c, d) for d in twice)]
        return bool(twice) and any(shape(c) != shape(twice[0]) for c in others)

    def chest(j: int) -> int:
        while kids.get(j):
            if parts_at(j):
                return j
            j = max(kids[j], key=size)
        return -1

    legs = {j for j in range(len(parents)) if any(t in _up(parents, j) for t in thighs)}
    fork = common_above(parents, thighs)
    above = _up(parents, fork) if fork >= 0 else []
    found = [(size(c), top) for a in above for c in kids.get(a, [])
             if c not in legs and c not in above and (top := chest(c)) >= 0]
    return max(found)[1] if found else -1


def trunk_of(parents, thighs, landmarks=(), pos: np.ndarray | None = None) -> tuple[int, tuple[int, ...]]:
    """(the hips, the trunk) of a skeleton from its hierarchy: the trunk runs from the hips up to trunk_top (`pos`:
    its rest positions, when known), the hips are hips_joint's. No top: the leg fork and no trunk."""
    parents = [int(p) for p in parents]
    top = trunk_top(parents, thighs, landmarks, pos)
    if top < 0:
        return common_above(parents, thighs), ()
    hips = hips_joint(parents, -1, thighs, top)
    chain = _up(parents, top)
    return hips, tuple(reversed(chain[:chain.index(hips)])) if hips in chain else ()


@dataclass(frozen=True)
class Body:
    """The joints a skeleton's own frame is taken from: its two thighs and two shins (left first) and its trunk (the
    chain from the hips up to where the arms and the head part: trunk_of, the one definition). The one definition
    of a body's up, forward and left: the hips' alignment, the twist references, a T pose, the angle differences of two
    base poses, the sole alignment and the mirror plane all read `axes`. No world direction is assumed up: a rig
    imported lying down, or a base frame of a person lying, has its own up. Only the trunk speaks for the upper body,
    never every joint above the hips: fingers outnumber the spine there, and arms hanging down put such a centroid
    below the hips — a standing character taken for a folded one, its up turned over."""

    thighs: tuple[int, int]
    shins: tuple[int, int]
    trunk: tuple[int, ...] = ()

    @classmethod
    def of_hierarchy(cls, parents, thighs: tuple[int, int], shins: tuple[int, int], landmarks=(),
                     pos: np.ndarray | None = None) -> Body:
        """A skeleton's body: its legs as given, its trunk trunk_of's (`landmarks`: its hands and head, where known;
        `pos`: its rest positions)."""
        return cls(tuple(thighs), tuple(shins), trunk_of(parents, thighs, landmarks, pos)[1])

    def stands(self, pos: np.ndarray) -> bool:
        """Whether a pose [J,3] has its legs hanging away from its trunk (the knees on the other side of the hips from
        the chest): a bind pose, T or A, arms up or down; not a BVH zero pose whose legs lie along the spine. Without
        a trunk there is nothing to fold the legs against: it stands."""
        if not self.trunk:
            return True
        hip = (pos[self.thighs[0]] + pos[self.thighs[1]]) / 2
        up = pos[list(self.trunk)].mean(axis=0) - hip
        down = (pos[self.shins[0]] + pos[self.shins[1]]) / 2 - hip
        return float(np.dot(up, down)) < 0.0

    def axes(self, pos: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
        """The body's axes in a pose [J,3] (columns, aim_frame: up, forward, left). Left across the thighs; up from the
        hip point to the trunk's centroid — the same trunk `stands` reads, in a folded pose as in a standing one. Not
        along the legs: a first frame or an idle often bends the knees forward (Mixamo's by 18°), which would tilt
        every alignment taken from it. Without a trunk, up along the legs (knees to hips). `up`: given instead
        (hips_turn level)."""
        up = self._up(pos) if up is None else np.asarray(up, np.float64)
        if np.linalg.norm(up) < ZERO_BONE:
            up = UP
        return aim_frame(up, pos[self.thighs[0]] - pos[self.thighs[1]])

    def _up(self, pos: np.ndarray) -> np.ndarray:
        """The unnormalised up of poses [...,J,3] (axes)."""
        hip = (pos[..., self.thighs[0], :] + pos[..., self.thighs[1], :]) / 2
        if self.trunk:
            return pos[..., list(self.trunk), :].mean(axis=-2) - hip
        return hip - (pos[..., self.shins[0], :] + pos[..., self.shins[1], :]) / 2

    def up_over(self, positions: np.ndarray) -> np.ndarray:
        """The body's up over a shot [F,J,3]: the median of its (axes) over the frames it stands on its legs, unit.
        The direction a motion's heights are taken along (hips_path): a bind pose need not stand (a BVH zero pose lies
        along its spine), and a solve's world need not have Y up; a frame's lean (the trunk ahead of the hips while
        walking, 3–10°) mostly evens out over a shot. Only the frames whose thighs hang under the trunk (the knees'
        middle within STANDING_AXIS_DEG of straight below the hips, along the trunk) say where up is: a take that sits
        on the floor leaning back for most of its length (jump, land, sit down) put the median trunk 46° off the
        vertical, every height was taken along that slant, and the retargeted feet sank 6–9 cm where the person stood.
        No frame stands (a take lying or sitting throughout): every frame, as before. Without a trunk up runs along the
        thighs and every frame stands."""
        pos = np.asarray(positions, np.float64)
        ups = self._up(pos)
        hip = (pos[..., self.thighs[0], :] + pos[..., self.thighs[1], :]) / 2
        hang = hip - (pos[..., self.shins[0], :] + pos[..., self.shins[1], :]) / 2
        n, m = np.linalg.norm(ups, axis=-1), np.linalg.norm(hang, axis=-1)
        ok = n > ZERO_BONE
        stands = ok & (m > ZERO_BONE) & (np.einsum("...i,...i->...", ups, hang)
                                         >= np.cos(np.radians(STANDING_AXIS_DEG)) * n * m)
        pick = stands if stands.any() else ok
        ups = ups[pick] / n[pick, None]
        if not len(ups):
            return UP.copy()
        up = np.median(ups, axis=0)
        return up / np.linalg.norm(up) if np.linalg.norm(up) > ZERO_BONE else UP.copy()


def hips_turn(target_pos: np.ndarray, source_pos: np.ndarray, target: Body, source: Body,
              level: bool = False) -> np.ndarray:
    """The alignment of the hips (the fork of the legs and the spine: no one bone to aim by) between two base poses
    [J,3]: the one body's axes onto the other's (Body.axes). The one rule Retarget.align and 「动作重定向」 turn the
    hips by. `level`: two standing base poses — each stands on its own ground, so up is that ground's normal, the
    world axis nearest the body's up (standing_up), and the turn only changes the way the body faces. The body's own
    up runs from between the thighs to the trunk's centroid, and rigs differ there: the thigh joints set ahead of the
    hips (SOMA 2.6 cm), a spine curved forward (AccuRIG's woman): 8–11° between two upright bodies, which a turn
    carrying every joint (BodyRetarget keep) turns into the whole target leaning back."""
    if not level:
        return source.axes(source_pos) @ target.axes(target_pos).T
    return source.axes(source_pos, standing_up(source._up(source_pos))) @ target.axes(
        target_pos, standing_up(target._up(target_pos))).T


STANDING_AXIS_DEG = 30.0  # a standing body's up this close to a world axis is taken as that axis (hips_turn level)


def standing_up(up: np.ndarray) -> np.ndarray | None:
    """The world axis (±X, ±Y, ±Z) nearest a standing body's up, within STANDING_AXIS_DEG; None: none that near (a
    rig set in the world at a slant), the body's own up stays."""
    n = float(np.linalg.norm(up))
    if n < ZERO_BONE:
        return None
    k = int(np.argmax(np.abs(up)))
    if abs(float(up[k])) / n < np.cos(np.radians(STANDING_AXIS_DEG)):
        return None
    axis = np.zeros(3)
    axis[k] = np.sign(up[k])
    return axis


def kept_joints(parents, body: Body, landmarks, pos: np.ndarray, axes: np.ndarray, mapped=()) -> set[int]:
    """The joints of a standing skeleton that Retarget.align turns by the root's alignment alone: the root, the trunk
    (Body.trunk: the spine up to the chest), the joints from the chest up to the head — the one of `landmarks` (its
    hands and head) that lies on the body's middle, less than half the thighs' spread to either side of it (`axes`:
    Body.axes, left in the third column; hands stand out by an arm's length in a T- or A-pose) — and each foot: the
    joint under the shin that `mapped` holds (the ankle), with everything under it. No head among the landmarks: the
    neck stays bone by bone."""
    out = {0, *body.trunk}
    hip = (pos[body.thighs[0]] + pos[body.thighs[1]]) / 2
    spread = float(np.linalg.norm(pos[body.thighs[0]] - pos[body.thighs[1]]))
    central = [j for j in landmarks if abs(float((pos[j] - hip) @ axes[:, 2])) < 0.5 * spread]
    if body.trunk and len(central) == 1:
        top = body.trunk[-1]
        path = _up(parents, central[0])
        if top in path:
            out.update(path[:path.index(top)])
    ankles = {c for c in range(len(parents)) if int(parents[c]) in body.shins and c in mapped}
    out.update(j for j in range(len(parents)) if ankles & set(_up(parents, j)))
    return out


def closest_pose(model: Skeleton, rig: Skeleton, poses: np.ndarray, pairs, aims, bodies: tuple[Body, Body]) -> tuple[Skeleton, int]:
    """Of a rig's own poses (joint-to-world [K,J,4,4]), as a Skeleton, the one whose bones point most like the model's
    rest pose, each side's directions taken in its own body axes (Body.axes, `bodies` the model's and the rig's: the way it faces
    does not count), and its index in `poses`: the pose a rig whose rest pose is no standing pose aligns in
    (Retarget.align). (-1: none compares; the rig's own rest.)"""
    mapped = dict(pairs)
    pm = model.rest_positions
    fm = bodies[0].axes(pm)
    other = bodies[1]
    best, score = -1, np.inf
    poses = np.asarray(poses, np.float64)
    for k, pose in enumerate(poses):
        pp = pose[:, :3, 3]
        fp = other.axes(pp)
        turns = []
        for m, _ in pairs:
            a = pair_aim(aims, mapped, m, pm, pp)
            d = pair_dirs(pm, pp, mapped, m, a) if isinstance(a, (int, np.integer)) else None
            if d is not None:
                turns.append(np.arccos(np.clip((fm.T @ d[0]) @ (fp.T @ d[1]), -1.0, 1.0)))
        if turns and float(np.mean(turns)) < score:
            best, score = k, float(np.mean(turns))
    return (rig, -1) if best < 0 else (Skeleton(rig.names, rig.parents, poses[best]), best)


def same_axes(model: Skeleton, rig: Skeleton, pairs, aims, tolerance_deg: float = 5.0) -> bool:
    """Whether the two skeletons put every shared bone along the same axis of its joint (the direction to its aim, in
    the joint's own rest frame, within `tolerance_deg`): the same joint convention on both sides, the rig being the
    model's own kind of skeleton (LaFAN1's BVH into a LaFAN1 model, SMPL into SMPL). A bone's own axis does not change
    with the pose it stands in, so this holds between a zero pose and a standing one alike. At least three bones must
    say so."""
    mapped = dict(pairs)
    pm, pp, rm0, rp0 = model.rest_positions, rig.rest_positions, model.rest_rotations, rig.rest_rotations
    said = 0
    for m, p in pairs:
        a = pair_aim(aims, mapped, m, pm, pp)
        d = pair_dirs(pm, pp, mapped, m, a) if isinstance(a, (int, np.integer)) else None
        if d is None:
            continue
        if np.degrees(np.arccos(np.clip((rm0[m].T @ d[0]) @ (rp0[p].T @ d[1]), -1.0, 1.0))) > tolerance_deg:
            return False
        said += 1
    return said >= 3


@dataclass
class Retarget:
    """A production rig's motion on a model's skeleton, and the model's result back on the rig.

    `pairs` (model joint, production joint) are the joints the two share; for each a constant `offsets` rotation
    with model world rotation = production world rotation @ offset, so both directions are exact inverses. Lengths
    scale by `scale` (model length per production length, units included). The two skeletons are placed by their hip
    joints, where the legs start (the point between the two thighs), not by their root joints, which sit at different
    heights above the legs (SMPL's pelvis ~9 cm above its hip joints, LaFAN1's Hips ~2 cm). On the model's side that
    point is fixed in its root frame (`hips[0]`: a model's thighs hang from its pelvis); on the rig's side it is taken
    from where its `thighs` are on every frame, as a rig may hang them under a joint that turns (Biped's Spine).
    `hips[1]`: the rig's point in its root frame at rest. The model's other joints take their own bone lengths."""

    model: Skeleton
    production: Skeleton
    pairs: list[tuple[int, int]]
    offsets: np.ndarray
    scale: float
    hips: tuple[np.ndarray, np.ndarray]
    thighs: tuple[int, int] = (-1, -1)  # the production rig's two thighs: its hip point is their middle

    @classmethod
    def align(cls, model: Skeleton, production: Skeleton, pairs: list[tuple[int, int]], aims: list,
              legs: tuple[int, int, int, int], poses: np.ndarray | None = None, landmarks=()) -> Retarget:
        """Rest-pose alignment, in three cases (each joint's `offsets`):

        - The two skeletons put every bone along the same axis of its joint (same_axes: the rig is the model's own
          skeleton, LaFAN1 data into a LaFAN1 model, SMPL into SMPL): the joints' own frames correspond, whatever pose
          each stands in, so the offset is the difference of the two frames — only turned onto the bone's direction if
          it strays. Guessing each bone's roll from the rest poses instead cannot know a pose's twist: a BVH whose
          zero pose puts every bone along +x comes out 5–170° off its own rotations.
        - Otherwise every shared joint's bone (towards its `aims` joint: a model joint index, "up" for a head that
          stands straight, None to take its parent's alignment) is turned from the model's rest direction to the
          rig's, its roll from the character's forward or up (twist_refs); the root, where the legs and the spine
          fork, turns by the body's own axes (Body.axes, hips_turn), as a
          fork has no one bone to aim by. The rest poses need not match (T-pose, A-pose, any bind pose). An aim joint
          the rig lacks passes to that joint's aim.
        - Two standing rest poses (Body.stands on both sides): the trunk — the root, the spine up to the chest and on
          up the neck to the head — and the feet (kept_joints) turn by the root's alignment alone, every joint of them,
          as hips_turn level does for the root: both stand upright on flat feet there, so the rig at rest is the model
          at rest. Their bones are not aimed one by one: rigs place these joints differently (SMPL's lower spine runs
          25° forward, its neck-to-head 38°, its ankle-to-ball 40° down; SAM 3D Body's MHR runs the spine straight up
          and its foot bone from mid-foot, 21° down), and aiming the model's bones onto the rig's bends the model's own
          shape away — SMPL from MHR stood 18° leaning back on feet tipped up, which StableMotion judged broken on
          86–100% of the frames and redrew stooped. The limbs, which T- and A-poses hold differently, are aimed bone by
          bone as above.
        - A rig whose rest pose is no standing pose (Body.stands: its legs do not hang below its upper body — a BVH
          zero pose) aligns in the one of `poses` (the rig's own sent poses, joint-to-world [K,J,4,4]) that is most like
          the model's rest pose instead: aligning a leg that points up the spine onto one that hangs down is a half
          turn whose roll nothing decides.

        `legs`: the model's (left thigh, left shin, right thigh, right shin): they give the forward direction and the
        length scale (leg lengths). `landmarks`: the model's hands and head (those it has), which say where its trunk
        runs (trunk_of), and through the pairs the rig's."""
        mapped = dict(pairs)
        if model.parents[0] >= 0 or 0 not in mapped:
            raise Failure("E-MOTION-NOROOT")
        for j in legs:
            if j not in mapped:
                raise Failure("E-MOTION-NOLEG", joint=model.names[j])
        lt, ls, rt, rs = legs
        both = [j for j in landmarks if j in mapped]  # the same landmarks on both sides
        body = Body.of_hierarchy(model.parents, (lt, rt), (ls, rs), both, model.rest_positions)
        body_p = Body.of_hierarchy(production.parents, (mapped[lt], mapped[rt]), (mapped[ls], mapped[rs]),
                                   [mapped[j] for j in both], production.rest_positions)
        # two standing rest poses: the root turns by each one's ground (hips_turn level), not by the bodies' own ups,
        # which rigs put 1–22° apart (the thigh joints' place, the spine's curve). A pose picked from the sent ones
        # holds the take's own lean, which stays
        level = body_p.stands(production.rest_positions) and body.stands(model.rest_positions)
        if poses is not None and len(poses) and not body_p.stands(production.rest_positions):
            production = closest_pose(model, production, poses, pairs, aims, (body, body_p))[0]
        pm, pp = model.rest_positions, production.rest_positions
        axes_m, axes_p = body.axes(pm), body_p.axes(pp)

        ratios = []
        for thigh, shin in ((lt, ls), (rt, rs)):  # thigh + shin (+ the foot's bone to the ankle when shared)
            foot = next((c for c in range(len(model.parents)) if model.parents[c] == shin and c in mapped), None)
            chain = [thigh, shin] + ([foot] if foot is not None else [])
            ratios.append(float(chain_lengths(pm[None], [chain])[0] / chain_lengths(pp[None], [[mapped[j] for j in chain]])[0]))
        scale = float(np.mean(ratios))

        def bone(m: int, a):
            return pair_dirs(pm, pp, mapped, m, a, (axes_m[:, 0], axes_p[:, 0]))

        def aim_of(m: int):
            return pair_aim(aims, mapped, m, pm, pp)

        rm0, rp0 = model.rest_rotations, production.rest_rotations
        same = same_axes(model, production, pairs, aims)
        root_turn = hips_turn(pm, pp, body, body_p, level=level)
        kept = kept_joints(model.parents, body, both, pm, axes_m, mapped) if level else {0}
        align: dict[int, np.ndarray] = {}
        for m in range(len(model.parents)):  # parents first: an inherited alignment is known when needed
            if m not in mapped:
                continue
            a = aim_of(m)
            if same:  # the joints' own frames correspond; onto the bone's own direction if that strays
                frames = rp0[mapped[m]] @ rm0[m].T
                align[m] = between(frames @ bone(m, a)[0], bone(m, a)[1]) @ frames if isinstance(a, (int, np.integer)) \
                    else frames
                continue
            if m in kept:  # the root, and with two standing rest poses the trunk and the feet (the docstring above)
                align[m] = root_turn
                continue
            if a is None:
                p = model.parents[m]
                while p >= 0 and p not in align:
                    p = model.parents[p]
                align[m] = align[p] if p >= 0 else np.eye(3)
                continue
            dm, dp = bone(m, a)
            rm, rp = twist_refs(dm, dp, ((axes_m[:, 1], axes_p[:, 1]), (axes_m[:, 0], axes_p[:, 0])))
            align[m] = aim_frame(dp, rp) @ aim_frame(dm, rm).T
        offsets = np.stack([rp0[p].T @ align[m] @ rm0[m] for m, p in pairs])
        hips = (rm0[0].T @ ((pm[lt] + pm[rt]) / 2 - pm[0]),
                rp0[mapped[0]].T @ ((pp[mapped[lt]] + pp[mapped[rt]]) / 2 - pp[mapped[0]]))
        return cls(model, production, list(pairs), offsets, scale, hips, (mapped[lt], mapped[rt]))

    def to_model(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rig poses (joint-to-world [K,J,4,4]) -> the model's world rotations [K,M,3,3] (joints the rig lacks keep
        their rest pose relative to their parent) and its root position [K,3]."""
        world = np.asarray(world, np.float64)
        mapped = dict(self.pairs)
        offset = {m: o for (m, _), o in zip(self.pairs, self.offsets)}
        rest_local = local_from_world(self.model.rest_rotations, self.model.parents)
        out = np.empty((len(world), len(self.model.parents), 3, 3))
        for m, p in enumerate(self.model.parents):
            if m in mapped:
                out[:, m] = orthonormal(world[:, mapped[m], :3, :3]) @ offset[m]
            else:
                out[:, m] = out[:, p] @ rest_local[m]
        hips = (world[:, self.thighs[0], :3, 3] + world[:, self.thighs[1], :3, 3]) / 2  # the rig's hip point, as it hangs
        return out, hips * self.scale - out[:, 0] @ self.hips[0]

    def to_production(self, rotations: np.ndarray, root: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
        """The model's world rotations [T,M,3,3] and root [T,3] -> (the shared production joints, their world
        rotations [T,P,3,3], the production rig's hip point [T,3]: the middle of its `thighs`). The rig is placed by
        that point once its rotations are set (set_hips), not by its root joint: its thighs may hang under a joint the
        motion turns (a Biped's Spine)."""
        rot = np.stack([rotations[:, m] @ o.T for (m, _), o in zip(self.pairs, self.offsets)], axis=1)
        hips = (np.asarray(root, np.float64) + np.asarray(rotations, np.float64)[:, 0] @ self.hips[0]) / self.scale
        return [p for _, p in self.pairs], rot, hips


# --------------------------------------------------------------------------- retargeting between two rigs
# 「动作重定向」（lab2shot/nodes/core/scene.py Retarget）：一副骨架的动作换到另一副骨架上。和上面的 Retarget 不同：
# 那是给模型类节点用的「rig → 模型骨架 → rig」往返，必须精确可逆，链状部位一节对一节（joints.spread）；这里是单向的，
# 链状部位（脊柱、颈、手指）两边节数不同时按骨长把源链的弯曲分摊到目标链的每一节（Unreal IK Retargeter 的
# Interpolated FK 链），缩放按每帧实际的腿长量（SAM 3D Body 的 MHR 逐关节带缩放，绑定姿势是模板）。


def chain_bones(pos: np.ndarray, chain: list[int], after: int | None, tip: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """A chain's bones in the positions `pos` [J,3]: (lengths [n], directions [n,3], zero where a bone has no length).
    Bone i runs from chain[i] to chain[i+1]; the last one to `after` (the joint the chain ends in: the chest after the
    spine, the head after the neck), else to `tip` (its only child), else it borrows the one before it (a finger's
    last joint)."""
    ends = list(chain[1:]) + ([after] if after is not None else [tip] if tip is not None else [])
    vec = [pos[b] - pos[a] for a, b in zip(chain, ends)]
    if len(vec) < len(chain):
        vec.append(vec[-1] if vec else np.zeros(3))
    vec = np.asarray(vec, np.float64).reshape(-1, 3)
    length = np.linalg.norm(vec, axis=-1)
    unit = np.where((length > ZERO_BONE)[:, None], vec / np.where(length > ZERO_BONE, length, 1.0)[:, None], 0.0)
    return length, unit


def arc_middles(lengths: np.ndarray) -> np.ndarray:
    """Where the middle of each bone sits along its chain, 0 at the chain's start and 1 at its end, by length
    (every bone the same share when the chain has no length at all)."""
    lengths = np.asarray(lengths, np.float64)
    if lengths.sum() <= ZERO_BONE:
        lengths = np.ones(len(lengths))
    u = np.concatenate([[0.0], np.cumsum(lengths)]) / lengths.sum()
    return (u[:-1] + u[1:]) / 2


SNAP = 1e-9  # a target bone's middle this close to a source key is that key: the joint is turned one to one


def bracket(at: list[float], x: float) -> tuple[int, int, float]:
    """The two keys around `x` in the sorted key places `at` and the fraction between them; beyond either end (a
    finger has no key after its last bone) the end key; on a key (within SNAP) that key alone."""
    for i, a in enumerate(at):
        if abs(x - a) <= SNAP:
            return i, i, 0.0
    if x <= at[0]:
        return 0, 0, 0.0
    if x >= at[-1]:
        return len(at) - 1, len(at) - 1, 0.0
    b = next(i for i, a in enumerate(at) if a > x)
    return b - 1, b, float((x - at[b - 1]) / (at[b] - at[b - 1]))


@dataclass
class Chain:
    """A chain part both skeletons have (the spine, the neck, a finger): its joints in each, from the root towards
    the tip, and the one-to-one joints it hangs from and ends in when both skeletons have them (the spine: the hips
    and the chest; the neck: the chest and the head; a finger: the hand and none)."""

    target: list[int]
    source: list[int]
    before: tuple[int, int] | None = None  # (target joint, source joint)
    after: tuple[int, int] | None = None


def _children(parents) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for j, up in enumerate(parents):
        kids.setdefault(int(up), []).append(j)
    return kids


def aims_toward(parents, driven=None) -> list[int | None]:
    """Which joint each joint's bone points at: the `aims` of Retarget.align and BodyRetarget.align. The child whose
    subtree holds a driven joint, when exactly one child's does; None when several do (the hips, the chest, a hand)
    or none does (an end): align then takes the parent's alignment. `driven` None = every joint, i.e. the only child
    when a joint has exactly one (a model's own skeleton: SMPL, SMPL-X). By the hierarchy alone, no names: a rig's
    thigh, upper arm and forearm often carry twist and helper joints beside the next limb joint (AccuRIG, Character
    Creator, Daz, most game rigs), so counting children would call them forks and copy only the turn relative to the
    bind pose (an A-pose motion on a T-pose rig would hold its arms at a fixed angle); counting the children that lead
    to a driven joint points them along the limb. Joints are ordered parents before children."""
    n = len(parents)
    has = [driven is None or j in driven for j in range(n)]
    for j in range(n - 1, -1, -1):  # children first: a subtree is known before its parent needs it
        if has[j] and int(parents[j]) >= 0:
            has[int(parents[j])] = True
    kids = _children(parents)
    out: list[int | None] = []
    for j in range(n):
        branches = [c for c in kids.get(j, ()) if has[c]]
        out.append(branches[0] if len(branches) == 1 else None)
    return out


def chain_lengths(positions: np.ndarray, chains: list[list[int]]) -> np.ndarray:
    """A length on every frame: the mean over `chains` (each a list of joints, root to tip) of the sum of their bones,
    from world positions [F,J,3] — the legs ([thigh, shin, foot] each), the arms, the trunk (「动作重定向」's 髋高依据,
    nodes/kit/retarget.py measure). A bone's length does not depend on how it turns, so this is the person's own
    length on that frame, scale included (MHR keeps the proportions in the joint scales, its bind pose is a
    template)."""
    total = 0.0
    for chain in chains:
        total = total + sum(np.linalg.norm(positions[:, b] - positions[:, a], axis=-1) for a, b in zip(chain, chain[1:]))
    return np.asarray(total, np.float64) / len(chains)


@dataclass
class BodyRetarget:
    """A rig's motion on another rig（「动作重定向」）.

    One-to-one joints (`pairs`, the hips, the chest, the head, the limbs, and a chain joint whose place along its
    chain is exactly a source joint's) turn as in Retarget.to_model: the target's world rotation is the source's
    world rotation times a constant `offsets` (rest-pose alignment by bone direction and the character's forward).
    A chain joint in between two source keys (`blends`: target joint, key a, key b, fraction, alignment) takes the
    spherical blend of the two keys' turns relative to their rest pose, so a chain of any number of joints bends as
    much in total as the source's and the bend spreads over its joints by length. Every other target joint follows
    its parent. Two chains of the same number of joints in the same proportions put every joint on a key: the result
    is exactly the one-to-one one (no switch between the two ways is needed).

    A one-to-one joint whose bone, in the source, reaches the joint it aims at through joints of its own that nothing
    is paired with (`swings`: SAM 3D Body's MHR ankle turns at l_talocrural / l_subtalar / l_transversetarsal between
    LeftFoot and LeftToeBase, not at LeftFoot) would miss their turns: the source joint's rotation alone does not
    carry them. Such a bone is swung on every frame onto the source's actual direction, from the paired joint to the
    one it aims at (the smallest turn, so its twist stays the paired joint's).

    Two ways to align the rest poses (`align`): bone by bone (each bone's rest direction onto the source's, its roll by
    twist_refs: the target copies the source's bone directions, whatever its own rest pose holds), or `keep` — one
    turn for every joint (the two bodies' axes, hips_turn): with the source in its rest pose the target is in its own,
    and every joint turns from there as the source's turns from its rest pose (the turns passed on, in the body's
    frame). The second keeps what the target's rest pose has of its own (a foot's slope, a spine's curve, the
    「初始姿势」 corrections); the caller matches the poses first where the two rests differ (nodes/kit/retarget.py)."""

    target: Skeleton
    source: Skeleton
    pairs: list[tuple[int, int]]
    offsets: np.ndarray
    blends: list[tuple[int, int, int, float, np.ndarray]]
    keys: list[int]  # the source joints the blends read (a, b above index this list)
    # (index into pairs, source joint, the source joint it aims at, the target bone in its joint's own axes, and with
    # `keep` the source bone's rest direction and the target's in the source's frame: the turn between the source's
    # rest and actual direction carries the target's own; None bone by bone, where the two rest directions are one)
    swings: list[tuple] = field(default_factory=list)

    @classmethod
    def align(cls, target: Skeleton, source: Skeleton, singles: dict[int, int], chains: list[Chain], aims: list,
              refs: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]], target_ref: np.ndarray,
              source_ref: np.ndarray, fixed: dict[int, np.ndarray] | None = None,
              keep: np.ndarray | None = None) -> BodyRetarget:
        """`singles`: target joint -> source joint, turned one to one; `chains`: the chain parts; `aims`: which joint
        each target joint's bone points at (aims_toward: towards the one branch that holds driven joints, None for a
        fork or an end) — the same alignment as Retarget.align, so a one-to-one joint is aligned exactly as there.
        `refs`: the two twist references, each a (target, source) pair of directions in the rest poses: a bone twists
        by the first, or by the second when it points more along the first (body_refs: the character's forward, then
        up; hand_refs: the palm's normal, then along the fingers). `fixed`: target joint -> its alignment given outright
        (hand_refs: a hand alone, whose wrist has no one bone to aim by). `target_ref` / `source_ref` [J,3]: the
        joints' actual positions on a reference frame, for the chains' bone lengths (a bone's length does not depend
        on how it turns; a scaled skeleton has its lengths there, not in its bind pose). `keep` [3,3]: every joint
        aligned by this one turn, the target's rest pose kept (class doc); `aims`, `refs` and `fixed` then serve only
        the chains' and the swings' bookkeeping."""
        mapped = dict(singles)
        blends_at: dict[int, tuple[int, int, float, np.ndarray, np.ndarray]] = {}
        keys: list[int] = []

        def key(j: int) -> int:
            if j not in keys:
                keys.append(j)
            return keys.index(j)

        pm, pp = target.rest_positions, source.rest_positions
        kids_t, kids_s = _children(target.parents), _children(source.parents)

        def only_child(kids, j):
            c = kids.get(j, [])
            return c[0] if len(c) == 1 else None

        for ch in chains:
            if not ch.target:
                continue
            src = list(ch.source)
            if not src and not (ch.before and ch.after):
                continue  # nothing to take the chain's bend from: it follows its parent
            after_t = ch.after[0] if ch.after else None
            after_s = ch.after[1] if ch.after else None
            lengths_t, _ = chain_bones(target_ref, ch.target, after_t, only_child(kids_t, ch.target[-1]))
            _, dirs_t = chain_bones(pm, ch.target, after_t, only_child(kids_t, ch.target[-1]))
            nu = arc_middles(lengths_t)
            if src:
                lengths_s, _ = chain_bones(source_ref, src, after_s, only_child(kids_s, src[-1]))
                _, dirs_s = chain_bones(pp, src, after_s, only_child(kids_s, src[-1]))
                at = [*([0.0] if ch.before else []), *arc_middles(lengths_s), *([1.0] if ch.after else [])]
                joints = [*([ch.before[1]] if ch.before else []), *src, *([after_s] if ch.after else [])]
                dirs = [*([dirs_s[0]] if ch.before else []), *dirs_s, *([dirs_s[-1]] if ch.after else [])]
            else:  # the source has no joint between the two ends: one bone from one to the other
                d = pp[after_s] - pp[ch.before[1]]
                d = d / np.linalg.norm(d) if np.linalg.norm(d) > ZERO_BONE else np.zeros(3)
                at, joints, dirs = [0.0, 1.0], [ch.before[1], after_s], [d, d]
            for k, t in enumerate(ch.target):
                a, b, alpha = bracket(at, float(nu[k]))
                if a == b and joints[a] in src:  # exactly on a source joint of the chain: one to one
                    mapped[t] = joints[a]
                    continue
                d = (1 - alpha) * dirs[a] + alpha * dirs[b]
                blends_at[t] = (key(joints[a]), key(joints[b]), alpha, d, dirs_t[k])

        def twist(dm: np.ndarray, dp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return twist_refs(dm, dp, refs)

        # one-to-one joints: the same two functions as Retarget.align (pair_dirs, pair_aim)
        def bone(m: int, a):
            return pair_dirs(pm, pp, mapped, m, a)

        def aim_of(m: int):
            return pair_aim(aims, mapped, m, pm, pp)

        align: dict[int, np.ndarray] = {}
        blend_align: dict[int, np.ndarray] = {}
        aimed: dict[int, int] = {}  # one-to-one target joint -> the target joint its bone aims at
        for m in range(len(target.parents)):  # parents first: an inherited alignment is known when needed
            if keep is not None:
                if m in mapped or m in blends_at:
                    a = aim_of(m) if m in mapped else None
                    if isinstance(a, (int, np.integer)):
                        aimed[m] = int(a)
                    align[m] = keep
                    if m in blends_at:
                        blend_align[m] = keep
                continue
            if fixed and m in fixed and m in mapped:
                align[m] = fixed[m]
                continue
            if m in blends_at:
                _, _, _, dp, dm = blends_at[m]
                if np.linalg.norm(dp) > ZERO_BONE and np.linalg.norm(dm) > ZERO_BONE:
                    dp = dp / np.linalg.norm(dp)
                    rm, rp = twist(dm, dp)
                    align[m] = blend_align[m] = aim_frame(dp, rp) @ aim_frame(dm, rm).T
                    continue
            if m not in mapped and m not in blends_at:
                continue
            a = aim_of(m) if m in mapped else None
            if isinstance(a, (int, np.integer)):
                aimed[m] = int(a)
            if a is None:
                p = target.parents[m]
                while p >= 0 and p not in align:
                    p = target.parents[p]
                align[m] = align[p] if p >= 0 else np.eye(3)
                if m in blends_at:
                    blend_align[m] = align[m]
                continue
            dm, dp = bone(m, a)
            rm, rp = twist(dm, dp)
            align[m] = aim_frame(dp, rp) @ aim_frame(dm, rm).T
        rm0, rp0 = target.rest_rotations, source.rest_rotations
        pairs = sorted(mapped.items())
        offsets = np.stack([rp0[p].T @ align[m] @ rm0[m] for m, p in pairs]) if pairs else np.zeros((0, 3, 3))
        blends = [(t, a, b, alpha, blend_align[t] @ rm0[t]) for t, (a, b, alpha, _, _) in sorted(blends_at.items())]
        swings = []
        for k, (m, p) in enumerate(pairs):  # the source reaches the aimed joint through joints of its own (class doc)
            a = aimed.get(m)
            if a is None or a not in mapped or int(source.parents[mapped[a]]) == p or keep is None and fixed and m in fixed:
                continue
            dm = pm[a] - pm[m]
            if np.linalg.norm(dm) > ZERO_BONE:
                dm = dm / np.linalg.norm(dm)
                rest = None
                if keep is not None:
                    dp = pp[mapped[a]] - pp[p]
                    if np.linalg.norm(dp) <= ZERO_BONE:
                        continue
                    rest = (dp / np.linalg.norm(dp), keep @ dm)
                swings.append((k, p, mapped[a], rm0[m].T @ dm, rest))
        return cls(target, source, pairs, offsets, blends, keys, swings)

    def rotations(self, world: np.ndarray) -> dict[int, np.ndarray]:
        """Source poses (joint-to-world [F,J,4,4]) -> the world rotation [F,3,3] of every target joint it turns."""
        world = np.asarray(world, np.float64)
        out = {m: orthonormal(world[:, p, :3, :3]) @ o for (m, p), o in zip(self.pairs, self.offsets)}
        for k, p, s, bone, rest in self.swings:  # onto the source bone's actual direction (class doc)
            m = self.pairs[k][0]
            want = world[:, s, :3, 3] - world[:, p, :3, 3]
            ok = np.linalg.norm(want, axis=-1) > ZERO_BONE
            if ok.any() and rest is not None:  # keep: the source bone's turn from its rest direction, on the target's
                want = want.copy()
                want[ok] = between(np.broadcast_to(rest[0], want[ok].shape), want[ok]) @ rest[1]
            if ok.any():
                turn = between(out[m][ok] @ bone, want[ok])
                out[m] = out[m].copy()
                out[m][ok] = turn @ out[m][ok]
        if self.blends:
            rest = self.source.rest_rotations
            turn = np.stack([orthonormal(world[:, j, :3, :3]) @ rest[j].T for j in self.keys], axis=1)  # [F,K,3,3]
            q = matrix_to_quat(turn)
            for t, a, b, alpha, fixed in self.blends:
                out[t] = quat_to_matrix(slerp(q[:, a], q[:, b], np.full(len(world), alpha))) @ fixed
        return out


def pair_dirs(pm: np.ndarray, pp: np.ndarray, mapped: dict, m: int, a, ups=(UP, UP)):
    """The unit direction from joint `m` to its aim `a` in both skeletons (rest positions `pm` of the one whose joints
    are indexed, `pp` of the other, `mapped` the first's joints -> the second's), or the two bodies' `ups` (Body.axes)
    for an aim "up";
    None when either has no direction there: the two joints sit on the same point (a zero-length bone).
    Retarget.align and BodyRetarget.align both align one-to-one joints through this and pair_aim."""
    if a == "up":
        return ups[0] / np.linalg.norm(ups[0]), ups[1] / np.linalg.norm(ups[1])
    dm, dp = pm[a] - pm[m], pp[mapped[a]] - pp[mapped[m]]
    lm, lp = float(np.linalg.norm(dm)), float(np.linalg.norm(dp))
    return None if min(lm, lp) < ZERO_BONE else (dm / lm, dp / lp)


def pair_aim(aims, mapped: dict, m: int, pm: np.ndarray, pp: np.ndarray):
    """Which joint `m` points at (`aims`: each joint's aim). Two kinds of aim give no direction and pass on to that
    joint's own aim: one the other skeleton does not have, and one sitting exactly on top of `m` (a zero-length bone).
    Zero-length bones are ordinary in production rigs (CMU's BVH puts LowerBack on Hips; HumanIK and Mixamo rigs carry
    helpers the same way); dividing by that length would spread NaN through every child's alignment. None: a cycle in
    the aim chain, nothing to aim with."""
    a, seen = aims[m], set()
    while isinstance(a, (int, np.integer)) and (a not in mapped or pair_dirs(pm, pp, mapped, m, a) is None):
        if a in seen:
            return None
        seen.add(a)
        a = aims[a]
    return a


def bone_turn(target: Skeleton, source: Skeleton, joint: tuple[int, int], aim: tuple[int, int], refs) -> np.ndarray | None:
    """A joint's rest-pose alignment (target -> source) along one bone given outright: from `joint` to `aim`, each a
    (target joint, source joint) pair, its roll by twist_refs — BodyRetarget.align's rule for a one-to-one joint, for a
    fork whose bone the caller chooses (the chest along the trunk). None when either bone has no length."""
    d = pair_dirs(target.rest_positions, source.rest_positions, {joint[0]: joint[1], aim[0]: aim[1]}, joint[0], aim[0])
    if d is None:
        return None
    rm, rp = twist_refs(d[0], d[1], refs)
    return aim_frame(d[1], rp) @ aim_frame(d[0], rm).T


def body_refs(target: Skeleton, source: Skeleton, bodies: tuple[Body, Body]):
    """BodyRetarget.align's twist references for a body (`bodies`: the target's, the source's): its forward, then up
    (Body.axes)."""
    at, as_ = bodies[0].axes(target.rest_positions), bodies[1].axes(source.rest_positions)
    return ((at[:, 1], as_[:, 1]), (at[:, 0], as_[:, 0]))


def hand_frame(pos: np.ndarray, wrist: int, middle: int, index: int, pinky: int) -> np.ndarray:
    """A hand's own axes in a pose [J,3] (columns): along the fingers (wrist to the middle finger's root), the palm's
    normal, and across. The same for any rest pose of the hand: flat, palm down, hanging at the side."""
    return aim_frame(pos[middle] - pos[wrist], np.cross(pos[middle] - pos[wrist], pos[index] - pos[pinky]))


def hand_refs(target: Skeleton, source: Skeleton, wrist: tuple[int, int], middle: tuple[int, int],
              index: tuple[int, int], pinky: tuple[int, int]):
    """For a hand alone (「动作重定向」 of two skeletons that are hands only: HaMeR's MANO, a hand rig), each argument a
    (target joint, source joint) pair: BodyRetarget.align's twist references (the palm's normal, then along the
    fingers — a body's forward and up mean nothing to a hand whose rest pose lies any way), and the wrist's own
    alignment (the rest hand's axes onto the other's: the wrist forks into five fingers, no one bone aims it)."""
    ft = hand_frame(target.rest_positions, wrist[0], middle[0], index[0], pinky[0])
    fs = hand_frame(source.rest_positions, wrist[1], middle[1], index[1], pinky[1])
    return ((ft[:, 2], fs[:, 2]), (ft[:, 0], fs[:, 0])), {wrist[0]: fs @ ft.T}


def hips_path(positions: np.ndarray, thighs: tuple[int, int], feet: list[int], scale: float,
              lift: bool = True, extra: float = 0.0, up: np.ndarray = UP) -> np.ndarray:
    """Where the target's hip point goes on every frame [F,3], from the source's world positions [F,J,3]: the source's
    hip point (the middle of its two thighs, where the legs start), raised for the target's legs.

    Heights are along `up`, the source body's up over the shot (Body.up_over), not along a world axis: a solve's world
    need not have gravity along Y (Z up, a pitched camera), and a height taken along the wrong axis moves the hips
    forward instead of up — scaled on every frame, it drifts the whole walk.

    Across `up` the path is the source's, 1 : 1: the character is retargeted into a plate and must stand where the
    person stands on every frame (scaling the whole path about a ground point amplified every step: a 14% longer-legged
    character ended half a body away after a few metres). The height, so that the feet reach the ground: `lift` raises
    every frame by (scale - 1) times that frame's height of the hip over its lowest of `feet` (the ankles: the same
    joints the sole alignment measures from) — legs `scale` times as long then put the target's lowest ankle where the
    source's is, whatever holds the body up: on stairs, ladders, in a jump, a deep crouch, sitting on the floor (the
    hip barely over the ankles, so barely raised). One constant for the shot (the median of that height) was right
    only while the person stood most of the take: a take sitting on the floor most of its length got the sitting
    height's raise, and the standing frames sank by the rest. `lift=False` scales the height over the ground (the
    lowest any of `feet` comes over the shot) by `scale` on every frame — right on flat ground, wrong as soon as the
    ground rises. `extra` is added along `up` on every frame in both modes: the ankles' different heights over the
    soles, and the user's own offset."""
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    hips = (positions[:, thighs[0]] + positions[:, thighs[1]]) / 2
    height = hips @ up
    lowest = (positions[:, feet] @ up).min(axis=1) if feet else height  # each frame's lowest ankle
    if lift:
        # the hip's height over the feet on each frame, not over a global ground: a person walking metres along a
        # tilted street puts the global lowest foot far below any one frame's feet
        raised = (scale - 1.0) * (height - lowest)
    else:
        ground = float(lowest.min())
        raised = (scale - 1.0) * (height - ground)
    return hips + (raised + extra)[:, None] * up


# --------------------------------------------------------------------------- time


def _brackets(times: np.ndarray, at: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each time in `at`: the samples before and after it in sorted `times` and the fraction between them
    (clamped at both ends)."""
    times = np.asarray(times, np.float64)
    at = np.clip(np.asarray(at, np.float64), times[0], times[-1])
    hi = np.clip(np.searchsorted(times, at, side="right"), 1, len(times) - 1)
    lo = hi - 1
    span = times[hi] - times[lo]
    u = np.where(span > 0, (at - times[lo]) / np.where(span > 0, span, 1.0), 0.0)
    return lo, hi, np.clip(u, 0.0, 1.0)


def resample_rotations(times: np.ndarray, rotations: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Rotations [T,...,3,3] sampled at `times` -> at the times `at` (slerp; held beyond the ends)."""
    if len(times) == 1:
        return np.repeat(np.asarray(rotations, np.float64)[:1], len(at), 0)
    lo, hi, u = _brackets(times, at)
    q = continuous(matrix_to_quat(rotations))
    return quat_to_matrix(slerp(q[lo], q[hi], u.reshape(-1, *([1] * (q.ndim - 2)))))


def resample_values(times: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Values [T,...] sampled at `times` -> at `at` (linear; held beyond the ends)."""
    values = np.asarray(values, np.float64)
    if len(times) == 1:
        return np.repeat(values[:1], len(at), 0)
    lo, hi, u = _brackets(times, at)
    u = u.reshape(-1, *([1] * (values.ndim - 1)))
    return values[lo] * (1 - u) + values[hi] * u


def windows(keys: list[int], max_frames: int, max_keys: int) -> list[tuple[int, int]]:
    """Split sorted key frames into parts a model generates one at a time: (first, last) key indices, consecutive
    parts sharing their boundary key; each part spans at most `max_frames` frames (first and last key included) and
    holds at most `max_keys` keys."""
    if len(keys) < 2:
        return []
    out, start = [], 0
    while start < len(keys) - 1:
        end = start + 1
        if keys[end] - keys[start] + 1 > max_frames:
            raise Failure("E-MOTION-KEYGAP", first=keys[start], last=keys[end], gap=keys[end] - keys[start], most=max_frames - 1)
        while end + 1 < len(keys) and keys[end + 1] - keys[start] + 1 <= max_frames and end + 1 - start + 1 <= max_keys:
            end += 1
        out.append((start, end))
        start = end
    return out


# --------------------------------------------------------------------------- keys


def smoothstep(u):
    """3u² − 2u³: 0 -> 1 with zero slope at both ends (a C1 blend)."""
    u = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def key_residuals(authored: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What moves generated local transforms [K,J,4,4] onto the authored ones at the keys: per joint a rotation r
    (quaternion, r * generated = authored) and a translation offset (authored − generated)."""
    qa = matrix_to_quat(orthonormal(authored[..., :3, :3]))
    qg = matrix_to_quat(orthonormal(generated[..., :3, :3]))
    return quat_mul(qa, quat_inv(qg)), authored[..., :3, 3] - generated[..., :3, 3]


def apply_residuals(frames: np.ndarray, keys: np.ndarray, rotation: np.ndarray, offset: np.ndarray,
                    local: np.ndarray) -> np.ndarray:
    """Locals [F,J,4,4] at `frames` moved by the key residuals (key_residuals at `keys`): between two keys the two
    residuals blend with smoothstep weights, so every key is hit exactly and the motion stays C1 there (the residual's
    slope is zero at the keys); before the first and after the last key that key's residual holds."""
    frames, keys = np.asarray(frames, np.float64), np.asarray(keys, np.float64)
    lo, hi, u = _brackets(keys, frames) if len(keys) > 1 else (np.zeros(len(frames), int),) * 2 + (np.zeros(len(frames)),)
    w = smoothstep(u)[:, None]
    q = slerp(rotation[lo], rotation[hi], w)
    t = offset[lo] * (1 - w[..., None]) + offset[hi] * w[..., None]
    out = np.array(local, np.float64)
    scale = signed_scales(out[..., :3, :3])  # a mirrored joint stays mirrored (orthonormal's convention)
    rot = quat_to_matrix(quat_mul(q, matrix_to_quat(orthonormal(out[..., :3, :3]))))
    out[..., :3, :3] = rot * scale[..., None, :]
    out[..., :3, 3] += t
    return out


# --------------------------------------------------------------------------- feet


def contact_segments(flags: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a per-frame flag array: [(first, last)] indices, inclusive."""
    flags = np.asarray(flags, bool)
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def leg_ik(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, target: np.ndarray,
           bend: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two-bone IK, per frame: world positions [F,3] of the hip, knee and ankle joints and the ankle's target ->
    world rotations [F,3,3] to apply to the thigh (about the hip) and to the shin (after the thigh's). The knee
    keeps bending in its own plane; a straight leg bends about `bend` [F,3] (or the world X axis). Bone lengths are
    kept, so an unreachable target is approached as far as the leg reaches."""
    u, s = knee - hip, ankle - knee
    l1, l2 = np.linalg.norm(u, axis=-1), np.linalg.norm(s, axis=-1)
    n = np.cross(u, s)
    fallback = np.cross(u, np.broadcast_to([1.0, 0.0, 0.0] if bend is None else bend, u.shape))
    n = np.where((np.linalg.norm(n, axis=-1) < 1e-6 * l1 * l2)[:, None], fallback, n)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    reach = np.clip(np.linalg.norm(target - hip, axis=-1), np.abs(l1 - l2) + 1e-6, l1 + l2 - 1e-6)
    angle_now = np.arccos(np.clip(np.sum(-u * s, -1) / (l1 * l2), -1.0, 1.0))  # the knee's interior angle
    angle_new = np.arccos(np.clip((l1 ** 2 + l2 ** 2 - reach ** 2) / (2 * l1 * l2), -1.0, 1.0))
    shin = axis_angle(n, angle_now - angle_new)  # opens (or closes) the knee about its bend axis
    moved = knee + (shin @ s[..., None])[..., 0]
    thigh = between(moved - hip, target - hip)
    return thigh, thigh @ shin
