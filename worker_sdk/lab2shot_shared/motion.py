"""Skeletal motion math, numpy only, shared by the core and the motion workers (one implementation for both
sides). Rotations are unit quaternions (w, x, y, z) or 3x3 matrices (column vectors, p' = R @ p); a hierarchy lists
parents before children (-1 for a root).

    rotations     matrix_to_quat, quat_to_matrix, quat_mul, quat_inv, slerp, continuous, orthonormal, axis_angle,
                  rotvec_to_matrix / matrix_to_rotvec (the axis-angle the SMPL family's parameters are written in:
                  the one pair, for the workers and the core alike)
    hierarchies   world_from_local, local_from_world (transforms or rotations), set_world
    retargeting   Skeleton, Retarget: a production rig's motion on a model's skeleton and back (rest-pose alignment,
                  length scale), exact both ways for the joints they share
    time          resample_rotations, resample_values, windows (a long shot in parts that share their boundary keys)
    keys          key_residuals, apply_residuals: a generated motion moved onto the animator's keys, C1 between keys
    feet          contact_segments, leg_ik (two-bone IK that pins the ankle and keeps the foot's orientation)
"""

from __future__ import annotations

from dataclasses import dataclass

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


def orthonormal(m: np.ndarray) -> np.ndarray:
    """[...,3,3] -> the nearest rotations (scale and shear removed)."""
    u, _, vt = np.linalg.svd(np.asarray(m, np.float64))
    d = np.sign(np.linalg.det(u @ vt))
    d[d == 0] = 1.0
    return u @ (np.eye(3) * np.stack([np.ones_like(d), np.ones_like(d), d], -1)[..., None, :]) @ vt


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
            scale = np.linalg.norm(local[:, j, :3, :3], axis=-2)  # column lengths
            parent_rot = orthonormal(parent[:, :3, :3])
            rot = np.swapaxes(parent_rot, -1, -2) @ np.asarray(rotations[j], np.float64)
            local[:, j, :3, :3] = rot * scale[:, None, :]
        if j in positions:
            inv = np.linalg.inv(parent)
            local[:, j, :3, 3] = (inv[:, :3, :3] @ np.asarray(positions[j], np.float64)[..., None])[..., 0] + inv[:, :3, 3]
        world[:, j] = parent @ local[:, j]
    return local


# --------------------------------------------------------------------------- retargeting


@dataclass
class Skeleton:
    """Joint names, parents (before children, the root first) and a rest pose, joint-to-world [J,4,4]: the character
    standing upright (Y up), the pose two skeletons are aligned in (a rig's bind pose, T or A; for a model a standing
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


@dataclass
class Retarget:
    """A production rig's motion on a model's skeleton, and the model's result back on the rig.

    `pairs` (model joint, production joint) are the joints the two share; for each a constant `offsets` rotation
    with model world rotation = production world rotation @ offset, so both directions are exact inverses. Lengths
    scale by `scale` (model length per production length, units included). The two skeletons are placed by their hip
    joints, where the legs start (the point between the two thighs), not by their root joints, which sit at different
    heights above the legs (SMPL's pelvis ~9 cm above its hip joints, LaFAN1's Hips ~2 cm): `hips` holds that point in
    each skeleton's root frame (model, production); the model's other joints take their own bone lengths."""

    model: Skeleton
    production: Skeleton
    pairs: list[tuple[int, int]]
    offsets: np.ndarray
    scale: float
    hips: tuple[np.ndarray, np.ndarray]

    @classmethod
    def align(cls, model: Skeleton, production: Skeleton, pairs: list[tuple[int, int]], aims: list,
              legs: tuple[int, int, int, int]) -> Retarget:
        """Rest-pose alignment. Every shared joint's bone (towards its `aims` joint: a model joint index, "up" for a
        head that stands straight, None to take its parent's alignment) is turned from the model's rest direction to
        the rig's, twisted so that the character's forward (or up, for bones along the forward axis) agrees: the rest
        poses need not match (T-pose, A-pose, any bind pose). An aim joint the rig lacks passes to that joint's aim.
        `legs`: the model's (left thigh, left shin, right thigh, right shin): they give the forward direction and the
        length scale (leg lengths)."""
        mapped = dict(pairs)
        if model.parents[0] >= 0 or 0 not in mapped:
            raise Failure("E-MOTION-NOROOT")
        for j in legs:
            if j not in mapped:
                raise Failure("E-MOTION-NOLEG", joint=model.names[j])
        pm, pp = model.rest_positions, production.rest_positions

        def forward(pos: np.ndarray, left: int, right: int) -> np.ndarray:
            side = pos[left] - pos[right]
            side[1] = 0.0
            f = np.cross(side, UP)
            return f / np.linalg.norm(f)

        lt, ls, rt, rs = legs
        fwd_m = forward(pm, lt, rt)
        fwd_p = forward(pp, mapped[lt], mapped[rt])

        def length(pos, chain):
            return sum(np.linalg.norm(pos[b] - pos[a]) for a, b in zip(chain, chain[1:]))

        ratios = []
        for thigh, shin in ((lt, ls), (rt, rs)):  # thigh + shin (+ the foot's bone to the ankle when shared)
            foot = next((c for c in range(len(model.parents)) if model.parents[c] == shin and c in mapped), None)
            chain = [thigh, shin] + ([foot] if foot is not None else [])
            ratios.append(length(pm, chain) / length(pp, [mapped[j] for j in chain]))
        scale = float(np.mean(ratios))

        def bone(m: int, a):
            """The unit direction from joint `m` to its aim `a`, in both skeletons — None when either skeleton has
            no direction there, i.e. the two joints sit on the same point (a zero-length bone)."""
            if a == "up":
                return UP, UP
            dm, dp = pm[a] - pm[m], pp[mapped[a]] - pp[mapped[m]]
            lm, lp = float(np.linalg.norm(dm)), float(np.linalg.norm(dp))
            return None if min(lm, lp) < ZERO_BONE else (dm / lm, dp / lp)

        def aim_of(m: int):
            """Which joint `m` points at. Two kinds of aim give no direction and pass on to that joint's own aim:
            one the rig does not have, and one **sitting exactly on top of `m`** (a zero-length bone).

            Zero-length bones are ordinary in production rigs: CMU's BVH put `LowerBack` on `Hips` and `Neck` on
            `Spine1` with a (0,0,0) offset, Maya HumanIK and Mixamo rigs carry helper joints the same way. Dividing
            by that length gives 0/0 = NaN, which spreads through every child's alignment and only surfaces far
            away (an SVD that does not converge, a NaN in a message's JSON)."""
            a, seen = aims[m], set()
            while isinstance(a, (int, np.integer)) and (a not in mapped or bone(m, a) is None):
                if a in seen:  # a cycle in the aim chain: nothing to aim with
                    return None
                seen.add(a)
                a = aims[a]
            return a

        align: dict[int, np.ndarray] = {}
        for m in range(len(model.parents)):  # parents first: an inherited alignment is known when needed
            if m not in mapped:
                continue
            a = aim_of(m)
            if a is None:
                p = model.parents[m]
                while p >= 0 and p not in align:
                    p = model.parents[p]
                align[m] = align[p] if p >= 0 else np.eye(3)
                continue
            dm, dp = bone(m, a)
            # the twist follows the character's forward, or its up for a bone that points forward in either skeleton
            use_up = max(abs(dm @ fwd_m), abs(dp @ fwd_p)) > max(abs(dm @ UP), abs(dp @ UP))
            rm, rp = (UP, UP) if use_up else (fwd_m, fwd_p)
            align[m] = aim_frame(dp, rp) @ aim_frame(dm, rm).T
        rm0, rp0 = model.rest_rotations, production.rest_rotations
        offsets = np.stack([rp0[p].T @ align[m] @ rm0[m] for m, p in pairs])
        hips = (rm0[0].T @ ((pm[lt] + pm[rt]) / 2 - pm[0]),
                rp0[mapped[0]].T @ ((pp[mapped[lt]] + pp[mapped[rt]]) / 2 - pp[mapped[0]]))
        return cls(model, production, list(pairs), offsets, scale, hips)

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
        root = orthonormal(world[:, mapped[0], :3, :3])
        hips = world[:, mapped[0], :3, 3] + root @ self.hips[1]  # the rig's hip point
        return out, hips * self.scale - out[:, 0] @ self.hips[0]

    def to_production(self, rotations: np.ndarray, root: np.ndarray) -> tuple[list[int], np.ndarray, np.ndarray]:
        """The model's world rotations [T,M,3,3] and root [T,3] -> (the shared production joints, their world
        rotations [T,P,3,3], the production root joint's world position [T,3])."""
        rot = np.stack([rotations[:, m] @ o.T for (m, _), o in zip(self.pairs, self.offsets)], axis=1)
        hips = (np.asarray(root, np.float64) + np.asarray(rotations, np.float64)[:, 0] @ self.hips[0]) / self.scale
        root_rot = rot[:, [p for m, p in self.pairs].index(dict(self.pairs)[0])]
        return [p for _, p in self.pairs], rot, hips - root_rot @ self.hips[1]


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
    scale = np.linalg.norm(out[..., :3, :3], axis=-2)
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
