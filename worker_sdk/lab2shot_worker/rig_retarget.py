"""Official learned-retargeting data bridge (numpy only until fairmotion is used).

retarget.npz: src/dst names, parents, rest [J,4,4], source world [F,J,4,4],
parts JSON, ignored joint indices, optional mesh points/faces/dense skin, frames
and fps. All coordinates are centimetres, Y up. raw/retarget.npz contains world
[F,target J,4,4].

Which joints a model receives is decided before the job is written (the
「重定向预处理」 node's ignored joints, checked against the node's declared
needs): an adapter never drops joints of its own accord, it leaves out exactly
the ignored ones (graph_inputs, fixed_body) and reports what it actually sent,
virtual markers included (sent, in result.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lab2shot_shared import motion as mo

SCHEMA = 'lab2shot.retarget-input/1'


def write_job(path, source, target, base, mapping, characters, *, fps, src_ignored=(), dst_ignored=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema": np.asarray(SCHEMA), "unit_cm": np.asarray(1.), "up_axis": np.asarray('Y'),
            "frames": np.asarray(source.frames, np.int64), "fps": np.asarray(fps, np.float64),
            "src_ignored": np.asarray(sorted(src_ignored), np.int64),
            "dst_ignored": np.asarray(sorted(dst_ignored), np.int64)}
    for side, rig, skeleton, parts in (("src", source, base.source, mapping.src),
                                        ("dst", target, base.target, mapping.dst)):
        rest = np.asarray(skeleton.rest, np.float64).copy()
        mesh_rest = rest.copy()
        rest[..., :3, :3] = mo.orthonormal(rest[..., :3, :3])
        data.update({f"{side}_names": np.asarray(rig.names), f"{side}_parents": np.asarray(rig.parents, np.int64),
                     f"{side}_rest": rest, f"{side}_parts": np.asarray(json.dumps(parts))})
        if side == "src":
            world = np.asarray(source.world(), np.float64).copy()
            world[..., :3, :3] = mo.orthonormal(world[..., :3, :3])
            data["src_world"] = world
        item = characters.get(side)
        if item is None or not item.get("meshes"):
            continue
        vertices, faces, weights = [], [], []
        offset = 0
        bind = np.asarray(item["bind"], np.float64)
        # The original mesh is posed onto exactly the corrected rest the model
        # receives, using all original influences, never a four-influence proxy.
        deformation = mesh_rest @ np.linalg.inv(bind)
        for mesh in item["meshes"]:
            points = np.asarray(mesh["points"], np.float64)
            idx = np.asarray(mesh["joint_indices"], np.int64)
            w = np.asarray(mesh["joint_weights"], np.float64)
            dense = np.zeros((len(points), len(rig.names)), np.float64)
            np.add.at(dense, (np.arange(len(points))[:, None], idx), w)
            homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
            posed = np.einsum("vj,jab,vb->va", dense, deformation, homogeneous)[:, :3]
            counts, indices = np.asarray(mesh["counts"]), np.asarray(mesh["indices"])
            triangles, at = [], 0
            for count in counts:
                ring = indices[at:at + int(count)]
                triangles.extend((int(ring[0]), int(ring[k]), int(ring[k + 1])) for k in range(1, len(ring) - 1))
                at += int(count)
            vertices.append(posed)
            faces.append(np.asarray(triangles, np.int64).reshape(-1, 3) + offset)
            weights.append(dense)
            offset += len(points)
        data.update({f"{side}_points": np.concatenate(vertices), f"{side}_faces": np.concatenate(faces),
                     f"{side}_weights": np.concatenate(weights)})
    np.savez(path, **data)
    return path


def read_job(path):
    with np.load(path, allow_pickle=False) as d:
        out = {k: np.asarray(d[k]) for k in d.files}
    if str(out.get('schema', '')) != SCHEMA or float(out.get('unit_cm', 0)) != 1. or str(out.get('up_axis', '')) != 'Y':
        raise ValueError('Expected explicit Lab2Shot retarget input: centimetres, Y up, world-space prepared references')
    if out['frames'].ndim != 1 or len(out['frames']) < 2 or np.any(np.diff(out['frames']) <= 0) or float(out['fps']) <= 0:
        raise ValueError('Retarget input must have increasing actual frame numbers and a positive frame rate')
    for side in ("src", "dst"):
        out[f"{side}_ignored"] = np.asarray(out.get(f"{side}_ignored", ()), np.int64)
        out[f"{side}_parts"] = json.loads(str(out[f"{side}_parts"]))
        out[f"{side}_names"] = [str(n) for n in out[f"{side}_names"]]
        parents = out[f"{side}_parents"]
        if parents[0] != -1 or np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, len(parents))):
            raise ValueError("Learned retargeting requires one skeleton root and parent-before-child joint order")
    return out


def write_result(run, world, **info):
    world = np.asarray(world, np.float64)
    if world.ndim != 4 or world.shape[-2:] != (4, 4) or not np.isfinite(world).all():
        raise ValueError("Official retargeting returned invalid joint transforms")
    run.job.raw_dir.mkdir(parents=True, exist_ok=True)
    # Separate files are committed only after the whole network sequence finishes.
    np.savez(run.job.raw_dir / "retarget.npz", world=world)
    run.finish(list(range(len(world))), **info)


def read_result(raw):
    return {"world": raw.arrays("retarget.npz")["world"], "info": raw.result()}


def sent(src, dst, *, src_names=None, dst_names=None):
    """What actually went into the model, for result.json (write_result(run,
    world, **sent(...))): per side the number of the job's joints the
    projection carries and the names of the markers the adapter made up
    (index -1 in `idx`). `src` / `dst` are projections with an "idx" (fixed_body,
    graph_inputs); `*_names` name the projection's slots (a marker is said by
    its slot number without them)."""
    out = {"sent": {}, "markers": {}}
    for side, projection, names in (("src", src, src_names), ("dst", dst, dst_names)):
        idx = np.asarray(projection["idx"], np.int64)
        names = list(names) if names is not None else [f"{side}:{k}" for k in range(len(idx))]
        out["sent"][side] = int(np.count_nonzero(idx >= 0))
        out["markers"][side] = [str(names[k]) for k, j in enumerate(idx) if j < 0]
    return out


def graph_driven(projection):
    """Official qb marks End Sites as zero-DOF: they are input markers only.

    The decoder emits rows for them, but upstream BVH export discards those
    rows' rotations. An actual FBX terminal bone must likewise retain its rest
    local transform instead of receiving the decoder's unconstrained rotation.
    Keep its geometry and weights in the input projection.
    """
    driven = {**projection, 'idx': projection['idx'].copy()}
    rotating = set(int(p) for p in projection['parents'] if p >= 0)
    for k in range(len(driven['idx'])):
        if k not in rotating:
            driven['idx'][k] = -1
    return driven


def fairmotion_input(data, side, fps=20):
    """Build a fairmotion Motion that motion_normalize_h2s consumes the same
    way as the official loaders (verified against bvh.load + preprocess_data.py
    on the SATA demo assets).

    fairmotion's real decomposition (what bvh.load produces and
    motion_normalize reads): the skeleton's per-joint local carries only the
    offset translation with identity rotation, and every pose row carries the
    channel rotation (parent-frame relative) with the absolute world position
    only on the root. This function therefore writes the rest pose as channel
    form and the source animation as per-frame channel rotations. The rest
    pose is prepended as the tpose frame because motion_normalize_h2s consumes
    the first frame as the skeleton pose and drops it. The official chain
    resamples to the worker's training rate (SATA 20, kinref 30 fps).
    """
    from fairmotion.core.motion import Joint, Skeleton, Motion

    names, parents = data[f"{side}_names"], data[f"{side}_parents"]
    rest = np.asarray(data[f"{side}_rest"], np.float64)
    rest_rot, rest_pos = rest[:, :3, :3], rest[:, :3, 3]
    # channel decomposition of the rest: offsets in the parent's rest frame and
    # the frame-0 channel rotations (R_p^-1 @ R_j, identity for a child that is
    # not yet rotated away from its parent).
    offsets = np.zeros((len(names), 3), np.float64)
    chan0 = np.zeros((len(names), 3, 3), np.float64)
    for j, p in enumerate(parents):
        if p < 0:
            offsets[j] = 0.0
            chan0[j] = rest_rot[j]
        else:
            offsets[j] = rest_rot[p].T @ (rest_pos[j] - rest_pos[p])
            chan0[j] = rest_rot[p].T @ rest_rot[j]
    skeleton = Skeleton()
    for j, name in enumerate(names):
        parent = skeleton.joints[int(parents[j])] if parents[j] >= 0 else None
        local = np.eye(4)
        local[:3, 3] = offsets[j]
        skeleton.add_joint(Joint(name, xform_from_parent_joint=local), parent)
    motion = Motion(skel=skeleton, fps=fps)
    tpose = np.repeat(np.eye(4)[None], len(names), axis=0)
    tpose[:, :3, :3] = chan0
    tpose[0, :3, 3] = rest_pos[0]
    motion.add_one_frame(tpose)
    if side == "src":
        world = np.asarray(data["src_world"], np.float64)
        times = (data["frames"] - data["frames"][0]) / float(data["fps"])
        samples = np.arange(int(np.ceil(times[-1] * fps)) + 1) / fps
        rotations = mo.resample_rotations(times, world[..., :3, :3], samples)
        positions = mo.resample_values(times, world[..., :3, 3], samples)
        poses = np.repeat(np.eye(4)[None, None], len(samples), axis=0).repeat(len(names), axis=1)
        poses[:, 0, :3, :3], poses[:, 0, :3, 3] = rotations[:, 0], positions[:, 0]
        for j, p in enumerate(parents):
            if p >= 0:
                poses[:, j, :3, :3] = np.swapaxes(rotations[:, p], -1, -2) @ rotations[:, j]
        # motion_2_states drops the velocity-less first frame. Supply a copy,
        # rather than losing the first real sample and shifting the whole shot.
        motion.add_one_frame(poses[0].copy())
        for frame in poses:
            motion.add_one_frame(frame)
    else:
        # the target only contributes its skeleton: one extra rest frame so the
        # normalizer keeps a frame and hands back skel for skel_2_state
        motion.add_one_frame(tpose.copy())
    return motion


def fairmotion_output(motion, data, fps=30, normalized=False, facing=None):
    """Model output already denormalized onto dst's fairmotion skeleton.
    Resample the neural output back to the source packet's original frame numbers.
    """
    world = motion.to_matrix(local=False)
    if facing is not None:
        world = np.asarray(facing)[None, None] @ world
    if normalized:
        # The official normalized skeleton has identity rest rotations. Restore
        # the real target joint axes after prediction, with rotations only.
        world[..., :3, :3] = world[..., :3, :3] @ data["dst_rest"][..., :3, :3]
    frames = data["frames"]
    times = np.arange(len(world)) / fps
    at = (frames - frames[0]) / float(data["fps"])
    out = np.repeat(np.eye(4)[None, None], len(frames), axis=0).repeat(len(data["dst_names"]), axis=1)
    out[..., :3, :3] = mo.resample_rotations(times, world[..., :3, :3], at)
    out[..., :3, 3] = mo.resample_values(times, world[..., :3, 3], at)
    return out


# STaR's exact 22-joint order (datasets/inference_set.py); these are anatomical
# slots, not guesses based on a custom character's labels.
STAR_PARTS = (("hips", 0), ("spine", 0), ("spine", 1), ("chest", 0), ("neck", 0), ("head", 0),
              ("l.thigh", 0), ("l.shin", 0), ("l.foot", 0), ("l.toe", 0),
              ("r.thigh", 0), ("r.shin", 0), ("r.foot", 0), ("r.toe", 0),
              ("l.clavicle", 0), ("l.upperarm", 0), ("l.forearm", 0), ("l.hand", 0),
              ("r.clavicle", 0), ("r.upperarm", 0), ("r.forearm", 0), ("r.hand", 0))
STAR_PARENTS = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15, 16, 3, 18, 19, 20])

# MeshRet's 65-joint extended Mixamo armature (run/preprocess_fbx.py EXTENDED_JOINT_NAMES)
# and the 25-joint body subset its released checkpoint predicts (only_body: true;
# data_loaders/mret.py body_bone_names). Slots our mapping cannot fill from real
# joints (Mixamo end markers, fingers a rig lacks) are synthesised geometrically.
MESHRET_FULL_PARTS = (("hips", 0), ("spine", 0), ("spine", 1), ("chest", 0), ("neck", 0), ("head", 0),
                      ("head", -1),
                      ("l.clavicle", 0), ("l.upperarm", 0), ("l.forearm", 0), ("l.hand", 0),
                      *(("l." + f, k) for f in ("thumb", "index", "middle", "ring", "pinky") for k in range(4)),
                      ("r.clavicle", 0), ("r.upperarm", 0), ("r.forearm", 0), ("r.hand", 0),
                      *(("r." + f, k) for f in ("thumb", "index", "middle", "ring", "pinky") for k in range(4)),
                      ("l.thigh", 0), ("l.shin", 0), ("l.foot", 0), ("l.toe", 0), ("l.toe", -1),
                      ("r.thigh", 0), ("r.shin", 0), ("r.foot", 0), ("r.toe", 0), ("r.toe", -1))
MESHRET_FULL_PARENTS = np.array([-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 10, 11, 12, 13, 10, 15, 16, 17, 10, 19, 20, 21, 10,
                                 23, 24, 25, 10, 27, 28, 29, 3, 31, 32, 33, 34, 35, 36, 37, 34, 39, 40, 41, 34,
                                 43, 44, 45, 34, 47, 48, 49, 34, 51, 52, 53, 0, 55, 56, 57, 58, 0, 60, 61, 62, 63])
MESHRET_BODY = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 31, 32, 33, 34, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64)
MESHRET_HEAD_TOP, MESHRET_L_TOE_END, MESHRET_R_TOE_END = 6, 59, 64

# MeshRet's 65 extended Mixamo joint names without the mixamorig: prefix, in the
# same order as MESHRET_FULL_PARTS (run/preprocess_fbx.py EXTENDED_JOINT_NAMES).
# build_armature reads them as the character's joint names and needs the exact
# Mixamo names to form its body-part groups.
MESHRET_NAMES = ("Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End",
                 "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
                 "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
                 "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4",
                 "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4",
                 "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4",
                 "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4",
                 "RightShoulder", "RightArm", "RightForeArm", "RightHand",
                 "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
                 "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4",
                 "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4",
                 "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4",
                 "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4",
                 "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End",
                 "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End")


def surface_tip(data, side, joint, direction):
    """Estimate a native terminal marker from the prepared, skinned surface.

    Only the model projection receives this marker. Artist bones, their rest
    references and the character's skin are unchanged. Return None when the
    surface does not constrain a positive extent along the requested axis.
    """
    points, weights = data.get(f'{side}_points'), data.get(f'{side}_weights')
    length = np.linalg.norm(direction)
    if points is None or weights is None or length < 1e-8 or joint < 0:
        return None
    descendants = {int(joint)}
    for j, parent in enumerate(data[f'{side}_parents']):
        if int(parent) in descendants:
            descendants.add(j)
    support = weights[:, sorted(descendants)].sum(axis=1) > .5
    axis = np.asarray(direction, np.float64) / length
    origin = data[f'{side}_rest'][joint, :3, 3]
    projected = (points - origin) @ axis
    valid = support & (projected > 0)
    if np.count_nonzero(valid) < 3:
        return None
    extent = float(np.quantile(projected[valid], .995))
    return origin + axis * extent


def fixed_body(data, side, slots=STAR_PARTS, parents=STAR_PARENTS, *, end_names=None):
    """Project the animator's mapping onto an official fixed skeleton.

    The joints the job lists as ignored are none of its slots: a part's slots
    take its remaining joints in order (the 「重定向预处理」 left exactly as many
    as the model has, its first and last among them).

    Pure index/mesh bookkeeping, never the inference solver and never a
    rotation convention: the returned rest and world keep the rig's own joint
    axes. All skin influences are retained by accumulating omitted joints onto
    their nearest supported ancestor. A slot with ordinal -1 is a Mixamo end
    marker: retain a matching native terminal joint when it exists. For a rig
    without that marker, the fallback is an explicit geometric estimate
    (weights stay zero). Each model's worker then turns the projected world
    into its own parametrisation (STaR raw channels, MeshRet flattened pose).
    """
    skip = {int(j) for j in data.get(f"{side}_ignored", ())}
    parts = {part: [j for j in joints if j not in skip] for part, joints in data[f"{side}_parts"].items()}
    rest_all = np.asarray(data[f"{side}_rest"], np.float64)
    mapped, virtual = [], []
    names = data[f'{side}_names']
    for k, (part, ordinal) in enumerate(slots):
        if ordinal < 0:
            candidates = []
            if end_names is not None:
                wanted = end_names[k].lower()
                parent_joint = mapped[int(parents[k])]
                original_parents = data[f'{side}_parents']
                children = np.flatnonzero(original_parents == parent_joint)
                assigned = {j for values in parts.values() for j in values}
                if len(children) == 1 and children[0] not in original_parents and int(children[0]) not in assigned:
                    candidates.append(int(children[0]))
                if not candidates:
                    candidates = [j for j, name in enumerate(names)
                                  if name.rsplit('/', 1)[-1].rsplit(':', 1)[-1].lower() == wanted
                                  and int(original_parents[j]) == parent_joint]
            mapped.append(candidates[0] if len(candidates) == 1 else -1)
            if mapped[-1] < 0:
                virtual.append(k)
            continue
        values = parts.get(part, [])
        if len(values) <= ordinal:
            if part.split(".")[-1] in ("thumb", "index", "middle", "ring", "pinky"):
                mapped.append(-1)
                virtual.append(k)
                continue
            raise ValueError(f"Official fixed skeleton needs mapped {part} joint {ordinal + 1}")
        mapped.append(int(values[ordinal]))
    real = [j for j in mapped if j >= 0]
    if len(set(real)) != len(real):
        raise ValueError("Official fixed skeleton slots must map to distinct joints")
    rest = np.zeros((len(slots), 4, 4), np.float64)
    for k, j in enumerate(mapped):
        if j >= 0:
            rest[k] = rest_all[j]
    for k in virtual:
        p = int(parents[k])
        rest[k] = rest[p].copy()
        rest[k, :3, 3] = 2.0 * rest[p, :3, 3] - rest[int(parents[p]), :3, 3] if parents[p] >= 0 else rest[p, :3, 3]
        # MeshRet uses the wrist -> middle knuckle segment for palm sensors,
        # even in its body-only checkpoint. Extending a missing knuckle by one
        # forearm length places those sensors outside the hand. Estimate this
        # model-only marker from the actual hand surface, without altering the
        # artist's skeleton or giving the marker any skin weight.
        part, ordinal = slots[k]
        if ordinal < 0:
            direction = rest[p, :3, 3] - rest[int(parents[p]), :3, 3]
            if part.endswith('.toe'):
                direction = direction.copy()
                direction[1] = 0.
            tip = surface_tip(data, side, mapped[p], direction)
            if tip is not None:
                rest[k, :3, 3] = tip
        if ordinal == 0 and part.split('.')[-1] in ('thumb', 'index', 'middle', 'ring', 'pinky'):
            hand = mapped[p]
            points = data.get(f'{side}_points')
            skin = data.get(f'{side}_weights')
            direction = rest[p, :3, 3] - rest[int(parents[p]), :3, 3]
            length = np.linalg.norm(direction)
            if hand >= 0 and points is not None and skin is not None and length > 1e-8:
                direction = direction / length
                distance = (points - rest[p, :3, 3]) @ direction
                supported = (skin[:, hand] > .5) & (distance > 0)
                if np.count_nonzero(supported) >= 3:
                    palm = float(np.median(distance[supported]))
                    rest[k, :3, 3] = rest[p, :3, 3] + direction * palm
    inverse_map = {j: k for k, j in enumerate(mapped) if j >= 0}
    weights = np.zeros((len(data[f"{side}_points"]), len(slots)), np.float64)
    for j in range(len(data[f"{side}_names"])):
        ancestor = j
        while ancestor not in inverse_map and data[f"{side}_parents"][ancestor] >= 0:
            ancestor = int(data[f"{side}_parents"][ancestor])
        if ancestor in inverse_map:
            weights[:, inverse_map[ancestor]] += data[f"{side}_weights"][:, j]
    out = {"idx": np.asarray(mapped, np.int64), "rest": rest, "parents": np.asarray(parents),
           "points": np.asarray(data[f"{side}_points"], np.float64),
           "faces": np.asarray(data[f"{side}_faces"], np.int64), "weights": weights}
    if side == "src":
        world = np.asarray(data["src_world"], np.float64)
        out["world"] = np.empty((len(world), len(slots), 4, 4), np.float64)
        for k, j in enumerate(mapped):
            if j >= 0:
                out["world"][:, k] = world[:, j]
            else:
                p = int(parents[k])
                out["world"][:, k] = out["world"][:, p] @ np.linalg.inv(rest[p]) @ rest[k]
    return out


def star_offsets(fixed):
    """STaR's _skel: each joint's offset in its parent's rest frame,
    R_rest_p^-1 @ (t_rest_j - t_rest_p), so the upstream FK with raw channel
    rotations reproduces the character's rest shape with its own joint axes.
    """
    rest, parents = fixed["rest"], fixed["parents"]
    out = np.zeros((len(parents), 3), np.float64)
    for j, p in enumerate(parents):
        if p >= 0:
            out[j] = rest[p, :3, :3].T @ (rest[j, :3, 3] - rest[p, :3, 3])
        else:
            out[j] = rest[j, :3, 3]
    return out


def star_channels(world, fixed):
    """STaR's _quat: raw BVH channel rotations R_p(t)^-1 @ R_j(t) (verified
    against the upstream BVH loader on Mixamo assets, |dot| == 1). The root
    keeps its absolute rotation.
    """
    parents = fixed["parents"]
    rot = np.asarray(world, np.float64)[..., :3, :3]
    out = np.zeros_like(rot)
    out[:, 0] = rot[:, 0]
    for j, p in enumerate(parents):
        if p >= 0:
            out[:, j] = np.swapaxes(rot[:, p], -1, -2) @ rot[:, j]
    return out


def star_world(channels, offsets, fixed):
    """Inverse of star_channels + star_offsets: forward kinematics that turns
    predicted channel rotations back into world matrices in the target's own
    axes (offsets are parent-local, so the rest shape and its joint axes come
    back exactly).
    """
    parents = fixed["parents"]
    ch = np.asarray(channels, np.float64)[..., :3, :3]
    off = np.asarray(offsets, np.float64)
    world = np.repeat(np.eye(4)[None, None], len(ch), axis=0).repeat(len(parents), axis=1)
    for j, p in enumerate(parents):
        local = np.zeros(ch.shape[:-3] + (4, 4), np.float64)
        local[..., :3, :3] = ch[..., j, :, :]
        local[..., :3, 3] = off[j]
        local[..., 3, 3] = 1.0
        if p < 0:
            world[..., j, :, :] = local
        else:
            world[..., j, :, :] = world[..., p, :, :] @ local
    return world


def flattened(world, fixed):
    """MeshRet's canonical basis: world rotations with the rest pose as zero,
    S(t) = R(t) @ R_rest^-1, so the rest pose is identity and the rest offsets
    are plain parent-relative positions."""
    rot = np.asarray(world, np.float64).copy()
    rot[..., :3, :3] = rot[..., :3, :3] @ np.swapaxes(fixed["rest"][..., :3, :3], -1, -2)
    return rot


def relative_locals(flat_world, fixed):
    """MeshRet's rotation parametrisation (run/preprocess_fbx.py pose formula
    with the rest pose as reference): pose_j = S_p(t)^-1 @ S_j(t)."""
    parents = fixed["parents"]
    rot = np.asarray(flat_world, np.float64)[..., :3, :3]
    out = np.zeros_like(rot)
    out[:, 0] = rot[:, 0]
    for j, p in enumerate(parents):
        if p >= 0:
            out[:, j] = np.swapaxes(rot[:, p], -1, -2) @ rot[:, j]
    return out


def relative_world(locals_, fixed):
    """Inverse of relative_locals: flattened world rotations from MeshRet's
    predicted poses."""
    parents = fixed["parents"]
    out = np.zeros_like(locals_)
    out[:, 0] = locals_[:, 0]
    for j, p in enumerate(parents):
        if p >= 0:
            out[:, j] = out[:, p] @ locals_[:, j]
    return out


def fixed_output(world, target, data, fps=30):
    """Retain unsupported target joints at their rest local transforms and
    resample the predictions (already in the target's own axes) back to the
    source packet's frame numbers. Neural predictions alone drive mapped joints.
    """
    at = (data["frames"] - data["frames"][0]) / float(data["fps"])
    times = np.arange(len(world)) / fps if fps is not None else at
    predicted_rot = mo.resample_rotations(times, world[..., :3, :3], at)
    predicted_pos = mo.resample_values(times, world[..., :3, 3], at)
    parents, rest = data["dst_parents"], data["dst_rest"]
    local = mo.local_from_world(rest, parents)
    out = np.repeat(rest[None], len(at), axis=0)
    lookup = {int(j): i for i, j in enumerate(target["idx"]) if j >= 0}
    for j, parent in enumerate(parents):
        if parent >= 0:
            out[:, j] = out[:, parent] @ local[j]
        if j in lookup:
            k = lookup[j]
            out[:, j, :3, :3] = predicted_rot[:, k]
            if j == int(data["dst_parts"]["hips"][0]):
                out[:, j, :3, 3] = predicted_pos[:, k]
    return out


def resampled_source(data, fps):
    """Uniform model time grid, retaining both endpoints of the source shot."""
    times = (data["frames"] - data["frames"][0]) / float(data["fps"])
    at = np.arange(int(np.ceil(times[-1] * fps)) + 1) / fps
    world = np.repeat(np.eye(4)[None, None], len(at), axis=0).repeat(len(data["src_names"]), axis=1)
    world[..., :3, :3] = mo.resample_rotations(times, data["src_world"][..., :3, :3], at)
    world[..., :3, 3] = mo.resample_values(times, data["src_world"][..., :3, 3], at)
    return {**data, "src_world": world}


def canonical_body(fixed):
    """Identity rest axes, world-space rest offsets: the Mixamo BVH basis.

    Joint axes are a rig's coordinate convention, not an anatomical feature.
    Strip them before the network and restore the target's axes on output.
    """
    rest = fixed["rest"].copy()
    rest[:, :3, :3] = np.eye(3)
    out = {**fixed, "rest": rest}
    if "world" in fixed:
        out["world"] = flattened(fixed["world"], fixed)
    return out


def restore_axes(world, fixed):
    out = world.copy()
    out[..., :3, :3] = out[..., :3, :3] @ fixed["rest"][:, :3, :3]
    return out


def graph_inputs(data):
    """Anatomical graph for human rigs: the job's mapped joints that are not
    ignored (the 「重定向预处理」 left out helpers, fingers, face … as the
    node's rule says; nothing is dropped here by name or part).

    Preserve each rig's spine/neck chain. A trunk or limb part is a node of
    the graph on whichever side has it; any other part (fingers, face) only
    when both sides send it, since there is nothing to pair it with otherwise.
    Terminal markers keep hands/head/toes as rotating joints (upstream qb).
    Return the projection needed to restore the full target hierarchy.
    """
    body = {"hips", "spine", "chest", "neck", "head"}
    body |= {f"{s}.{b}" for s in ("l", "r") for b in
             ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe")}
    given = {}
    for side in ("src", "dst"):
        skip = {int(j) for j in data.get(f"{side}_ignored", ())}
        given[side] = {part: [j for j in joints if j not in skip] for part, joints in data[f"{side}_parts"].items()}
    parts = body | {p for p in given["src"] if p not in body and given["src"][p] and given["dst"].get(p)}
    canonical = {"hips": "Hips", "spine": "Spine", "chest": "Spine2", "neck": "Neck", "head": "Head",
                 "clavicle": "Shoulder", "upperarm": "Arm", "forearm": "ForeArm", "hand": "Hand",
                 "thigh": "UpLeg", "shin": "Leg", "foot": "Foot", "toe": "ToeBase", "jaw": "Jaw", "eye": "Eye"}
    fingers = ("thumb", "index", "middle", "ring", "pinky")
    projected = dict(data)
    projections = {}
    for side in ("src", "dst"):
        pp = given[side]
        labels = {}
        for part in sorted(parts):
            for k, j in enumerate(pp.get(part, [])):
                s, _, bone = part.partition(".")
                prefix = "Left" if s == "l" else "Right" if s == "r" else ""
                key = bone or s
                stem = ("Spine" + str(len(pp.get("spine", []))) if part == "chest" else canonical[key] if key in canonical
                        else "Hand" + key.capitalize() if key in fingers else key.capitalize())
                name = prefix + stem
                labels[int(j)] = name + (str(k) if k else "")
        idx = sorted(labels)
        lookup = {j: k for k, j in enumerate(idx)}
        parents = []
        for j in idx:
            p = int(data[f"{side}_parents"][j])
            while p >= 0 and p not in lookup:
                p = int(data[f"{side}_parents"][p])
            parents.append(lookup[p] if p >= 0 else -1)
        rest = list(data[f"{side}_rest"][idx].copy())
        names = [labels[j] for j in idx]
        world = list(np.moveaxis(data["src_world"][:, idx], 1, 0)) if side == "src" else None
        original_parents = data[f'{side}_parents']
        for k in range(len(idx)):
            if k in parents:
                continue
            # The graph checkpoints were trained with actual BVH End Sites.
            # Preserve a unique terminal child; replacing a real head/toe tip
            # with an arbitrary fraction of the preceding bone changes go/lo.
            children = np.flatnonzero(original_parents == idx[k])
            terminal = int(children[0]) if len(children) == 1 and children[0] not in original_parents else -1
            if terminal >= 0 and terminal not in lookup:
                lookup[terminal] = len(idx)
                rest.append(data[f'{side}_rest'][terminal].copy())
                names.append(names[k] + '_End')
                parents.append(k)
                idx.append(terminal)
                if world is not None:
                    world.append(data['src_world'][:, terminal])
                continue
            p = parents[k]
            tip = rest[k].copy()
            if p >= 0:
                tip[:3, 3] += 0.3 * (rest[k][:3, 3] - rest[p][:3, 3])
                part = next((part for part, joints in pp.items() if idx[k] in joints), None)
                if part in ('head', 'l.hand', 'r.hand', 'l.toe', 'r.toe'):
                    direction = rest[k][:3, 3] - rest[p][:3, 3]
                    if part.endswith('.toe'):
                        direction = direction.copy()
                        direction[1] = 0.
                    estimated = surface_tip(data, side, idx[k], direction)
                    if estimated is not None:
                        tip[:3, 3] = estimated
            rest.append(tip)
            names.append(names[k] + "_End")
            parents.append(k)
            idx.append(-1)
            if world is not None:
                world.append(world[k] @ np.linalg.inv(rest[k]) @ tip)
        # Collapse skin weights of unmodelled bones into their nearest ancestor.
        if f"{side}_weights" in data:
            w = np.zeros((len(data[f"{side}_weights"]), len(idx)))
            for j in range(len(data[f"{side}_names"])):
                a = j
                while a >= 0 and a not in lookup:
                    a = int(data[f"{side}_parents"][a])
                w[:, lookup[a] if a >= 0 else 0] += data[f"{side}_weights"][:, j]
            projected[f"{side}_weights"] = w
        projected.update({f"{side}_rest": np.asarray(rest), f"{side}_parents": np.asarray(parents),
                          f"{side}_names": names,
                          f"{side}_parts": {p: [lookup[j] for j in pp.get(p, [])] for p in parts if pp.get(p)}})
        if world is not None:
            projected["src_world"] = np.moveaxis(np.asarray(world), 0, 1)
        projections[side] = {"idx": np.asarray(idx), "rest": np.asarray(rest), "parents": np.asarray(parents)}
    return projected, projections
