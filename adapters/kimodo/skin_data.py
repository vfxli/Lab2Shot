"""Standard-human body data for the SOMA and G1 options (core reads it, the extension provides it).

The kimodo repository ships the skin meshes the core's 「标准人」 needs for its SOMA and G1 options, but the core
must not read the research repository (architecture: core has zero third-party dependency; body data lives in
third_party/_body_models/, like SMPL-X's neutral body). This module copies / builds that data into the core's body
model area. Run by the extension's post_install with the extension's own Python (torch + the pinned kimodo package).

- SOMA: `somaskel77/skin_standard.npz` is copied as `_body_models/soma/skin_standard.npz` (77-joint LBS skin; the
  core reduces it to the 30-joint drive skeleton).
- G1: the rigid link meshes (`g1skel34/meshes/g1/*.STL`) are merged at the 34-joint skeleton's rest pose into
  `_body_models/g1/g1_body.npz` (bind transforms + one rigid mesh, metres, Y up).

Only the pinned repository's assets are read. Writing is confined to third_party/_body_models/.
"""

from __future__ import annotations

import os
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# MuJoCo (z up, x forward) -> Lab2Shot / kimodo (y up, z forward): [0,1,0; 0,0,1; 1,0,0]
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], np.float64)


def _read_stl(path: Path) -> np.ndarray:
    """Binary STL -> [V,3] vertices (each triangle's three corners, duplicates kept)."""
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        out = np.empty((n, 3, 3), np.float64)
        for i in range(n):
            f.read(12)  # normal (ignored)
            out[i] = np.frombuffer(f.read(36), np.float32).reshape(3, 3).astype(np.float64)
            f.read(2)  # attribute byte count
    return out.reshape(-1, 3)


def _quat_matrix(values: str) -> np.ndarray:
    """A MuJoCo quaternion (scalar-first w x y z) -> rotation matrix."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = [float(v) for v in values.split()]
    return Rotation.from_quat([x, y, z, w]).as_matrix().astype(np.float64)


def build_g1(repo: Path, out: Path) -> None:
    """Merge the G1 link meshes into `out` (g1_body.npz)."""
    import sys

    sys.path.insert(0, str(repo))
    from kimodo.skeleton.registry import build_skeleton

    skeleton = build_skeleton(34)
    names = skeleton.bone_order_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    # Neutral pose = zero pose (identity rotations over neutral_joints): this is exactly what the kimodo worker's
    # `model_skeleton` writes as `rest`, so the character's bind pose matches the animation it will receive.
    neutral = skeleton.neutral_joints.detach().cpu().numpy().astype(np.float64)

    xml = repo / "kimodo" / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml"
    meshdir = repo / "kimodo" / "assets" / "skeletons" / "g1skel34" / "meshes" / "g1"
    tree = ET.parse(xml)
    root = tree.getroot()
    parent_map = {c: p for p in root.iter() for c in p}

    # body name -> skeleton joint (a body's own joint; the root pelvis's free joint is the pelvis)
    body_to_skel: dict[str, str] = {}
    body_quat: dict[str, np.ndarray] = {}
    for body in root.find("worldbody").iter("body"):
        quat = np.eye(3)
        if body.get("quat") and body.get("quat") != "1 0 0 0":
            quat = _quat_matrix(body.get("quat"))
        body_quat[body.get("name")] = quat
        joint = body.find("joint")
        if joint is not None:
            body_to_skel[body.get("name")] = joint.get("name").replace("_joint", "_skel")
        elif body.find("freejoint") is not None:
            body_to_skel[body.get("name")] = "pelvis_skel"

    verts, faces, joints = [], [], []
    for body in root.find("worldbody").iter("body"):
        # the skeleton joint this body (and its decorative geoms) belongs to: its own, or the nearest ancestor's
        skel, node = None, body
        while node is not None and node.tag == "body":
            skel = body_to_skel.get(node.get("name"))
            if skel:
                break
            node = parent_map.get(node)
        if skel is None or skel not in name_to_idx:
            continue
        j = name_to_idx[skel]
        for geom in body.findall("geom"):
            mesh = geom.get("mesh")
            if not mesh:
                continue
            stl = meshdir / f"{mesh}.STL"
            if not stl.is_file():
                continue
            v = _read_stl(stl)
            # the geom's rigid offset inside its body frame (MuJoCo), then the body's own orientation offset
            if geom.get("quat") and geom.get("quat") != "1 0 0 0":
                v = (_quat_matrix(geom.get("quat")) @ v.T).T
            if geom.get("pos"):
                v = v + np.array([float(x) for x in geom.get("pos").split()])
            v = (body_quat[body.get("name")] @ v.T).T
            # body frame (MuJoCo, z up) -> kimodo frame (y up), then the joint's neutral position
            v = (MUJOCO_TO_KIMODO @ v.T).T
            w = v + neutral[j]
            faces.append(np.arange(len(verts), len(verts) + len(w), dtype=np.int64).reshape(-1, 3))
            verts.append(w)
            joints.append(np.full(len(w), j, np.int32))
    verts = np.concatenate(verts).astype(np.float64)
    faces = np.concatenate(faces).astype(np.int64)
    joints = np.concatenate(joints).astype(np.int32)
    bind = np.repeat(np.eye(4)[None], len(names), 0)
    bind[:, :3, 3] = neutral
    parents = np.asarray(skeleton.joint_parents.tolist(), np.int64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, joint_names=np.array(names), parents=parents, bind_world=bind,
                        vertices=verts, faces=faces,
                        joint_indices=joints[:, None], joint_weights=np.ones((len(joints), 1), np.float32))
    print(f"g1_body.npz: {len(names)} joints, {len(verts)} vertices, {len(faces)} faces")


def provide_skin_data(repo: Path, body_root: Path) -> None:
    """Copy the SOMA skin and build the G1 body into body_root (third_party/_body_models/)."""
    src = repo / "kimodo" / "assets" / "skeletons"
    soma = body_root / "soma"
    soma.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "somaskel77" / "skin_standard.npz", soma / "skin_standard.npz")
    build_g1(repo, body_root / "g1" / "g1_body.npz")


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:]]
    repo = Path(args[0] if args else os.environ.get("LAB2SHOT_EXT_REPO", "third_party/kimodo/repo")).resolve()
    body_root = Path(args[1] if len(args) > 1 else os.environ.get("LAB2SHOT_BODY_ROOT",
                                                                  "third_party/_body_models")).resolve()
    provide_skin_data(repo, body_root)
