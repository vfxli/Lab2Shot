"""The two bodies Kimodo brings to the core's 「标准人」 (its 「骨架」 options SOMA 30 and G1 34), declared on the
extension (extension.py standard_bodies) and registered by the core when the extension loads
(lab2shot/data/standard_bodies.py). They match Kimodo's own skeletons one to one: picking the same skeleton on both
nodes lets 「线性蒙皮变形」 put Kimodo's animation back on the body joint for joint.

Their data comes with the pinned repository and is copied (SOMA) or built (G1) into the core's body-model area by
post_install (skin_data.py): third_party/_body_models/<soma|g1>/. These functions read only that area, never the
research repository, and run in the main environment (numpy and lab2shot.sdk only).
"""

from __future__ import annotations

import numpy as np

from lab2shot.sdk import StandardBody, character_of_model, merge_weights, scaled_to_ground

SOMA_SKIN = "skin_standard.npz"  # somaskel77's 77-joint LBS skin (metres)
G1_SKIN = "g1_body.npz"  # 34-joint rigid link mesh + bind pose built from Kimodo's assets (metres)

# SOMA's 30-joint drive skeleton (kimodo.skeleton.SOMASkeleton30's order). It is a view of the 77-joint SOMA skin:
# the joints of the same name match one to one, the other 47 (whole finger chains, HeadEnd, ToeEnd) are end points
# following their parents (their skin weights merged into the nearest kept ancestor). The bone names are SOMA's own
# (as in the Kimodo model); character_of_model's rig_of_model turns them into CG names, as for Kimodo's unconstrained
# output.
SOMA_30_NAMES = (
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase",
)
SOMA_30_PARENTS = (
    -1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28,
)


def _said(model: str, names, verts_cm: np.ndarray, faces: np.ndarray) -> dict:
    return {"model": model, "joints": len(names), "vertices": len(verts_cm), "faces": len(faces),
            "height_cm": round(float(verts_cm[:, 1].max() - verts_cm[:, 1].min()), 1)}


def soma_body(height_cm: float | None = None):
    """SOMA 30 joints with skinning (centimetres, Y up): the 77-joint SOMA skin (bind_vertices, faces, lbs_indices /
    lbs_weights, rig_joint_names, bind_rig_transform) reduced to the 30 drive joints, the other 47 joints' weights
    merged into their nearest kept ancestor."""
    with np.load(SOMA.folder() / SOMA_SKIN, allow_pickle=True) as f:
        verts = np.asarray(f["bind_vertices"], np.float64)  # metres
        faces = np.asarray(f["faces"], np.int64)
        lbs_indices = np.asarray(f["lbs_indices"], np.int64)
        lbs_weights = np.asarray(f["lbs_weights"], np.float64)
        names = [str(n) for n in f["rig_joint_names"]]
        bind77 = np.asarray(f["bind_rig_transform"], np.float64)  # [77,4,4] metres
        connections = np.asarray(f["rig_joint_connections"], np.int64)
    parents77 = np.full(len(names), -1, np.int64)
    for a, b in connections:
        parents77[b] = a
    kept = tuple(names.index(n) for n in SOMA_30_NAMES)
    dense = np.zeros((len(verts), len(names)), np.float64)
    for k in range(lbs_indices.shape[1]):
        dense[np.arange(len(verts)), lbs_indices[:, k]] += lbs_weights[:, k]
    merged = merge_weights(dense, parents77, kept)
    order = np.argsort(-merged, axis=1)[:, : lbs_indices.shape[1]]
    joint_indices = order.astype(np.int64)
    joint_weights = np.take_along_axis(merged, order, 1)
    joint_weights /= np.maximum(joint_weights.sum(1, keepdims=True), 1e-9)
    verts_cm, bind_cm = scaled_to_ground(verts, bind77[list(kept)], height_cm)
    character = character_of_model(
        list(SOMA_30_NAMES), np.asarray(SOMA_30_PARENTS, np.int64),
        bind_world=bind_cm, anim_world=bind_cm[None],
        rest_points=verts_cm, faces=faces, joint_indices=joint_indices, joint_weights=joint_weights,
        custom_data={"body_model": SOMA.id},
    )
    return character, _said(SOMA.model, SOMA_30_NAMES, verts_cm, faces)


def g1_body(height_cm: float | None = None):
    """The Unitree G1 robot, 34 joints (centimetres, Y up). G1 is rigid links, not an LBS skin: each link follows its
    joint rigidly, written as a skinned character with one influence of weight 1 per vertex (UsdSkel takes it), so it
    goes into 「线性蒙皮变形」 like any body. The mesh and bind pose are built by skin_data.py (rest = zero pose, as
    Kimodo's worker writes `rest`)."""
    with np.load(G1.folder() / G1_SKIN, allow_pickle=True) as f:
        names = [str(n) for n in f["joint_names"]]
        parents = np.asarray(f["parents"], np.int64)
        verts = np.asarray(f["vertices"], np.float64)  # metres
        faces = np.asarray(f["faces"], np.int64)
        bind = np.asarray(f["bind_world"], np.float64)  # [34,4,4] metres
        joint_indices = np.asarray(f["joint_indices"], np.int64)
        joint_weights = np.asarray(f["joint_weights"], np.float64)
    verts_cm, bind_cm = scaled_to_ground(verts, bind, height_cm)
    character = character_of_model(
        names, parents,
        bind_world=bind_cm, anim_world=bind_cm[None],
        rest_points=verts_cm, faces=faces, joint_indices=joint_indices, joint_weights=joint_weights,
        custom_data={"body_model": G1.id},
    )
    return character, _said(G1.model, names, verts_cm, faces)


# The skins come from the Kimodo repository (Apache-2.0): commercial use, no licence switch. Their names in the
# 「骨架」 list: extension.kimodo.body.<id> (i18n/<lang>.toml).
SOMA = StandardBody("soma", "SOMA", soma_body, word="extension.kimodo.body.soma", files=(SOMA_SKIN,))
G1 = StandardBody("g1", "G1", g1_body, word="extension.kimodo.body.g1", files=(G1_SKIN,))
BODIES = (SOMA, G1)
