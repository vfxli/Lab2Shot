"""Official kinref transformer and artifact-driven geometry refinement.

Both stages use the official graph features. Geometry inference follows
geo_test.py retarget_frame_chunk, with features computed on the whole motion
before batching. The geometry forward needs Jacobian autograd.
"""
from __future__ import annotations

import numpy as np
import torch
import trimesh
from lab2shot_worker import load_job, serve
from lab2shot_worker.run import Run
from lab2shot_worker import rig_retarget as bridge

# kinref's eight part classes (preprocess/preprocess_mesh.py PartIndex):
# 0 body, 1 head, 2 l.arm, 3 r.arm, 4 l.leg, 5 r.leg, 6 l.hand, 7 r.hand.
# Official groups come from Mixamo joint names; ours come through the same
# animator-edited mapping slots, which name the same anatomy.
_CLASS_OF = {"neck": 1, "head": 1, "eye": 1, "jaw": 1, "hips": 0, "spine": 0, "chest": 0}
_ARM, _HAND = {"clavicle", "upperarm", "forearm"}, {"hand", "thumb", "index", "middle", "ring", "pinky"}
_LEG = {"thigh", "shin", "foot", "toe"}


def classes_of(data, side):
    classes = np.zeros(len(data[f"{side}_names"]), np.int64)
    assigned = set()
    for part, joints in data[f"{side}_parts"].items():
        side_dot, _, bone = part.partition(".")
        bone = bone or side_dot
        if bone in _CLASS_OF:
            category = _CLASS_OF[bone]
        elif bone in _LEG:
            category = 4 if side_dot == "l" else 5
        elif bone in _HAND:
            category = 6 if side_dot == "l" else 7
        else:
            category = 2 if side_dot == "l" else 3
        classes[list(joints)] = category
        assigned.update(joints)
    # End Sites may carry real skin weights. Their part is the parent's part,
    # rather than the default torso category.
    for j, parent in enumerate(data[f"{side}_parents"]):
        if j not in assigned and parent >= 0:
            classes[j] = classes[parent]
    return classes


def mesh(data, side, normalized):
    from kinref.geo_dataset import npz_2_mesh_data
    points = data[f"{side}_points"].copy()
    shift = data[f"{side}_rest"][0, :3, 3] * np.array([1, 0, 1])
    points -= shift
    faces = data[f"{side}_faces"]
    normals = trimesh.Trimesh(points, faces, process=False).vertex_normals
    weights = data[f"{side}_weights"]
    classes = classes_of(data, side)
    bind = normalized.skel.joints
    bind_inv = np.stack([np.linalg.inv(j.xform_global) for j in bind])
    return npz_2_mesh_data(points, normals, weights, bind_inv, classes[weights.argmax(axis=1)], faces, side)


def no_mesh(joints):
    """A mesh of no vertices for a source without one: the batch pads it away (SkelPoseMeshBatch.from_data_list)."""
    from kinref.geo_dataset import MeshData
    return MeshData(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, joints)),
                    torch.eye(4).repeat(joints, 1, 1), torch.zeros((0,), dtype=torch.long),
                    torch.zeros((0, 3), dtype=torch.long), name="src")


def main(job_path):
    run = Run(load_job(job_path), "Kinematic Refinement")
    data, p = bridge.read_job(run.job.inputs["retarget"]), run.params
    original = data
    # The 「重定向预处理」 node decided which joints go in (its ignored list,
    # checked against this node's rules); the graph carries the rest.
    data, projection = bridge.graph_inputs(data)
    from utils.motion_utils import motion_normalize_h2s
    from conversions.motion_to_graph import motion_2_states, skel_2_state
    from conversions.graph_to_motion import hatD_recon_motion

    stage = p["stage"]
    if stage == "geo":
        from kinref.geo_test import prepare_model_test
        from kinref.geo_dataset import npz_2_data, SkelData
        from kinref.skel_pose_mesh_graph import SkelPoseMeshGraph as Graph, SkelPoseMeshBatch as Batch
    else:
        from kinref.kin_test import prepare_model_test
        from kinref.kin_dataset import npz_2_data, SkelData
        from kinref.skel_pose_graph import SkelPoseGraph as Graph, SkelPoseBatch as Batch
    model, cfg, stats = run.model("load_model", prepare_model_test, stage, "cuda:0",
                                  stage_params={"model": "Kinematic Refinement"})
    source, _ = motion_normalize_h2s(bridge.fairmotion_input(data, "src", fps=30), False)
    target, _ = motion_normalize_h2s(bridge.fairmotion_input(data, "dst", fps=30), False)
    (lo, go, qb, edges), (q, pos, root, pv, qv, previous, contact) = motion_2_states(source)
    skeleton, poses = npz_2_data(lo, go, qb, edges, q, pos, qv, pv, previous, contact, root)
    tlo, tgo, tqb, ted = skel_2_state(target.skel)
    target_skeleton = SkelData(torch.tensor(tlo), torch.tensor(tgo), torch.tensor(tqb, dtype=torch.bool),
                               torch.tensor(ted[:, :2], dtype=torch.long).T, torch.tensor(ted[:, 2:], dtype=torch.long))
    # the geo model reads the target's mesh only (geo_model.GeoModel.forward: the source goes through the kinematic
    # encoder); the graph batch still wants mesh fields on every graph, so a source without one carries an empty mesh
    src_mesh = (mesh(data, "src", source) if "src_points" in data else no_mesh(len(source.skel.joints))) \
        if stage == "geo" else None
    dst_mesh = mesh(data, "dst", target) if stage == "geo" else None
    graphs = [Graph(skeleton, pose, src_mesh) if stage == "geo" else Graph(skeleton, pose) for pose in poses]
    targets = [Graph(target_skeleton, None, dst_mesh) if stage == "geo" else Graph(target_skeleton, None)] * len(graphs)
    run.stage("geo_refine" if stage == "geo" else "kin_retarget")
    if stage == "geo":
        pieces = []
        batch_frames = int(p.get('geo_batch_frames', 16))
        for start in range(0, len(graphs), batch_frames):
            end = min(start + batch_frames, len(graphs))
            src = Batch.from_data_list(graphs[start:end]).to('cuda:0')
            dst = Batch.from_data_list(targets[start:end]).to('cuda:0')
            prediction = model(src, dst, stats, cfg['representation']['out'], end - start)
            pieces.append(prediction[2].detach())  # delta_z, delta_joint, hatD
            del src, dst, prediction
        output = torch.cat(pieces)
    else:
        src, dst = Batch.from_data_list(graphs).to("cuda:0"), Batch.from_data_list(targets).to("cuda:0")
        with torch.inference_mode():
            _, output = model(src, dst)
    dst = Batch.from_data_list(targets).to("cuda:0")
    out, _ = hatD_recon_motion(output.detach(), dst, cfg["representation"]["out"], stats, len(graphs))
    facing = source.poses[1].get_root_facing_transform_byRoot(use_height=False)
    facing[:3, 3] = 0
    predicted = bridge.fairmotion_output(out[0], data, fps=30, normalized=True, facing=facing)
    world = bridge.fixed_output(predicted, bridge.graph_driven(projection["dst"]), original, fps=None)
    bridge.write_result(run, world, **bridge.sent(projection["src"], projection["dst"], src_names=data["src_names"],
                                                  dst_names=data["dst_names"]),
                        method="Kinematic Refinement", stage=stage, model_fps=30)


if __name__ == "__main__":
    serve(main)
