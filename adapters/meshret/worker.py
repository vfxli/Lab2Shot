"""MeshRet official inference: RetNet (only_body) with the upstream sensor /
collate preprocessing.

The character data (T-posed mesh, joint locations, parents, skin weights) is
built by fixed_body's 65-joint extended Mixamo projection; the canonical
rotation pose (relative_locals of the flattened world) and the surface sensors
(utils/body_armatures + run/motion2points.get_sensor_data) follow the upstream
fbx2motion pipeline exactly. The model is called in the released only_body
configuration (25 body joints, 30-frame clips with the upstream ±3-frame
boundary blend).
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from lab2shot_worker import load_job, resident, serve
from lab2shot_worker.run import Run
from lab2shot_worker import rig_retarget as bridge

WINDOW = 30  # meshret_config seq_len
CHECKPOINT = "epoch=36-step=182743.ckpt"


def _repo_on_path(repo_dir: Path) -> None:
    """The upstream repo and its submodules importable, submodules first (a resident process runs main again: no duplicates)."""
    import sys

    for path in (str(repo_dir), str(repo_dir / "submodules")):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


@resident
def load_model(checkpoint: Path, repo_dir: Path):
    """The released RetNet (only_body), frozen in eval mode."""
    _repo_on_path(repo_dir)
    from model.retnet import RetNet

    model = RetNet.load_from_checkpoint(str(checkpoint), map_location="cuda:0")
    model.freeze()
    model.eval()
    return model


def main(job_path):
    run = Run(load_job(job_path), "MeshRet")
    data = bridge.read_job(run.job.inputs["retarget"])
    p = run.params

    _repo_on_path(run.job.repo_dir)
    from data_loaders.mret import mret_collate, all_pose_to_body_pose, all_static_to_body_static, body_pose_to_all_pose
    from run.motion2points import get_sensor_data
    from utils.rotation_conversions import rotation_6d_to_matrix

    # RetNet stays loaded between jobs (@resident): forward reads nothing a job leaves on it
    model = run.model("load_model", load_model, run.job.weights_dir / CHECKPOINT, run.job.repo_dir,
                      stage_params={"model": "MeshRet"})

    sampled = bridge.resampled_source(data, 30)
    src_axes = bridge.fixed_body(sampled, "src", slots=bridge.MESHRET_FULL_PARTS, parents=bridge.MESHRET_FULL_PARENTS, end_names=bridge.MESHRET_NAMES)
    dst_axes = bridge.fixed_body(data, "dst", slots=bridge.MESHRET_FULL_PARTS, parents=bridge.MESHRET_FULL_PARENTS, end_names=bridge.MESHRET_NAMES)
    src, dst = bridge.canonical_body(src_axes), bridge.canonical_body(dst_axes)

    def motion_data(side, is_src):
        # canonical pose: flattened world rotations -> relative locals (rotmat)
        world = side["world"] if is_src else None
        if is_src:
            pose = bridge.relative_locals(world, side)         # [F,65,3,3]
        else:
            pose = np.repeat(np.eye(3)[None, None], 2, axis=0).repeat(65, axis=1)
        root = world[:, 0, :3, 3] if is_src else side["rest"][0:1, :3, 3]
        return {"verts": side["points"].astype(np.float32)[None],
                "faces": side["faces"].astype(np.int32),
                "vgrp_label": np.asarray(bridge.MESHRET_NAMES),
                "vgrp_cors": side["rest"][:, :3, 3].astype(np.float32)[None],
                "vgrp_parents": side["parents"].copy(),
                "lbs_weights": side["weights"].astype(np.float32),
                "motion_poses": {0: pose.astype(np.float32)},
                "motion_translations": {0: root.astype(np.float32)[:, None, :]}}

    run.stage("prepare_input")
    src_md = motion_data(src, True)
    dst_md = motion_data(dst, False)
    src_md["sensor_data"] = {(4, 4): get_sensor_data(src_md, 4, 4)}
    dst_md["sensor_data"] = {(4, 4): get_sensor_data(dst_md, 4, 4)}

    # one batch entry in the MRet.__getitem__ format, then the official collate
    item = {"meta": ("src", 0, "dst", 0), "inp": src_md["motion_poses"][0], "root_translation": src_md["motion_translations"][0],
            "mask": np.ones((len(src_md["motion_poses"][0]),), bool), "is_intra": torch.zeros(1, dtype=torch.bool),
            "gt": None,
            "src_static": {"joint_locations": src_md["vgrp_cors"], "parents": src_md["vgrp_parents"],
                           **{k: v for k, v in src_md["sensor_data"][(4, 4)].items()},
                           "verts": src_md["verts"], "faces": src_md["faces"], "lbs_weights": src_md["lbs_weights"]},
            "tgt_static": {"joint_locations": dst_md["vgrp_cors"], "parents": dst_md["vgrp_parents"],
                           **{k: v for k, v in dst_md["sensor_data"][(4, 4)].items()},
                           "verts": dst_md["verts"], "faces": dst_md["faces"], "lbs_weights": dst_md["lbs_weights"]}}
    batch = mret_collate([item])

    def _cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda()
        if isinstance(obj, dict):
            return {k: _cuda(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_cuda(v) for v in obj]
        return obj

    batch = _cuda(batch)
    x, y = batch["x"], batch["y"]
    x_body = all_pose_to_body_pose(x)
    y_body = dict(y)
    y_body["src_static"] = all_static_to_body_static(y["src_static"])
    y_body["tgt_static"] = all_static_to_body_static(y["tgt_static"])

    run.stage("retarget")
    with torch.inference_mode():
        x_hat = []
        T = x_body.shape[1]
        num_clips = (T + WINDOW - 1) // WINDOW
        for i in range(num_clips):
            s = i * WINDOW
            clip = x_body[:, s:s + WINDOW]
            n = clip.shape[1]
            if n < WINDOW:
                clip = torch.cat((clip, clip[:, -1:].expand(-1, WINDOW - n, -1, -1)), dim=1)
            clip_y = dict(y_body)
            clip_y["mask"] = torch.ones_like(y_body["mask"][:, :1]).expand(-1, WINDOW, -1, -1)
            x_hat.append(model(clip, clip_y)[:, :n])
        x_hat = torch.cat(x_hat, dim=1)
        for i in range(1, num_clips):
            for k in range(3):
                m1 = x_hat[:, i * WINDOW - k:i * WINDOW - k + 2].mean(dim=1)
                m2 = x_hat[:, i * WINDOW + k:i * WINDOW + k + 2].mean(dim=1)
                x_hat[:, i * WINDOW - k] = m1
                if i * WINDOW + k + 1 < T:
                    x_hat[:, i * WINDOW + k + 1] = m2
        x_hat = body_pose_to_all_pose(x, x_hat)

    # The official collate removes the source's initial root rotation and
    # translation. Undo it for every joint, after FK in the canonical basis.
    rotmat = rotation_6d_to_matrix(x_hat[0]).detach().cpu().numpy()
    world = bridge.star_world(rotmat, bridge.star_offsets(dst), dst)
    root_delta = batch["root_translation"][0, :, 0].detach().cpu().numpy()
    heading = src["world"][0, 0, :3, :3]
    world[..., :3, :3] = heading @ world[..., :3, :3]
    root = root_delta @ heading.T
    root += dst["rest"][0, :3, 3]
    world[..., :3, 3] = np.einsum("ab,fjb->fja", heading, world[..., :3, 3] - world[:, :1, :3, 3]) + root[:, None]
    world = bridge.restore_axes(world, dst_axes)
    # The released checkpoint predicts the body only. Target fingers retain
    # their own local rest transforms instead of receiving source fingers.
    driven = dict(dst_axes)
    driven["idx"] = dst_axes["idx"].copy()
    driven["idx"][[k for k in range(65) if k not in bridge.MESHRET_BODY]] = -1
    bridge.write_result(run, bridge.fixed_output(world, driven, data, fps=30),
                        **bridge.sent(src_axes, dst_axes, src_names=bridge.MESHRET_NAMES, dst_names=bridge.MESHRET_NAMES),
                        method="MeshRet", model_fps=30)


if __name__ == "__main__":
    serve(main)
