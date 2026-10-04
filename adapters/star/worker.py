"""STaR official inference: Pct shape encoder + Retarget_Model.

Project explicit anatomical references into the released 22-joint BVH basis,
sample at 30 fps and apply the upstream position/facing preprocessor to both
positions and root channels. Use the released 120-frame model, retain padded
tail frames, reconstruct the whole body's motion, restore target joint axes
and resample onto the original shot. shapeA/shapeB are unused by forward.
"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path

from lab2shot_worker import load_job, resident, serve
from lab2shot_worker.run import Run
from lab2shot_worker import rig_retarget as bridge

WINDOW = 120          # official inference_config min/max steps
SCALE_FACTOR = 0.007  # inference.py scale_factor


def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
    return -softmax(-x, **kw)


def process(positions, Quaternions, Pivots):
    """Copied verbatim from preprocess/3_preprocess_q.py process()."""
    import scipy.ndimage.filters as filters

    fid_l, fid_r = np.array([8, 9]), np.array([12, 13])
    foot_heights = np.minimum(positions[:, fid_l, 1], positions[:, fid_r, 1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    positions[:, :, 1] -= floor_height
    trajectory_filterwidth = 3
    reference = positions[:, 0]
    positions = np.concatenate([reference[:, np.newaxis], positions], axis=1)
    velfactor, heightfactor = np.array([0.15, 0.15]), np.array([9.0, 6.0])
    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    feet_l_h = positions[:-1, fid_l, 1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    feet_r_h = positions[:-1, fid_r, 1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
    velocity = (positions[1:, 0:1] - positions[:-1, 0:1]).copy()
    positions[:, :, 0] = positions[:, :, 0] - positions[:, :1, 0]
    positions[1:, 1:, 1] = positions[1:, 1:, 1] - (positions[1:, :1, 1] - positions[:1, :1, 1])
    positions[:, :, 2] = positions[:, :, 2] - positions[:, :1, 2]
    sdr_l, sdr_r, hip_l, hip_r = 15, 19, 7, 11
    across1 = positions[:, hip_l] - positions[:, hip_r]
    across0 = positions[:, sdr_l] - positions[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode="nearest")
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    positions = rotation * positions
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:, :, 0]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 1]], axis=-1)
    positions = np.concatenate([positions, velocity[:, :, 2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    return positions, rotation


def centroid_scale(points):
    low, high = points.min(axis=0), points.max(axis=0)
    return float(np.max(high - low)), (low + high) / 2.0


def get_height(joints):
    d = lambda a, b: np.sqrt(((joints[a] - joints[b]) ** 2).sum(axis=-1))  # noqa: E731
    return d(5, 4) + d(4, 3) + d(3, 2) + d(2, 1) + d(1, 0) + d(6, 7) + d(7, 8) + d(8, 9)


@resident
def load_networks(repo_dir: Path, weights_dir: Path):
    """The Pct shape encoder and Retarget_Model with the released weights (official inference.py), kept between jobs.
    network.py eagerly JIT-compiles a Chamfer loss at import (not used by Pct/Retarget_Model.forward): that
    training-only import is deferred in the code run here; every network class and forward method is byte-for-byte
    upstream."""
    import sys
    import types

    for path in (str(repo_dir), str(repo_dir / "submodules")):
        if path not in sys.path:
            sys.path.insert(0, path)
    network_path = repo_dir / "method" / "network.py"
    code = network_path.read_text()
    begin = code.index('sys.path.append("./submodules")')
    end = code.index('\n\nclass Attention', begin)
    code = code[:begin] + '''def ChamDist(*args, **kwargs):
    from ChamferDistancePytorch.chamfer_python import distChamfer_a2b
    return distChamfer_a2b(*args, **kwargs)
''' + code[end:]
    network = types.ModuleType("method.network")
    network.__file__ = str(network_path)
    exec(compile(code, str(network_path), "exec"), network.__dict__)

    encoder = network.Pct(dropout=0.5, output_channels=40).cuda()
    encoder = torch.nn.DataParallel(encoder, device_ids=[0])  # 官方 inference.py 同款：checkpoint 带 module. 前缀
    encoder.load_state_dict(torch.load(weights_dir / "model.t7", map_location="cuda:0"))
    encoder.eval()
    retarget = network.Retarget_Model(num_joint=22, num_frame=WINDOW, token_channels=64, hidden_channels=64,
                                      embed_channels=32, use_rot6d=False).cuda()
    retarget = torch.nn.DataParallel(retarget, device_ids=[0])
    retarget.load_state_dict(torch.load(repo_dir / "work_dir" / "paper_version" / "retarget_model.pt",
                                        map_location="cuda:0"))
    retarget.eval()
    return encoder, retarget


def main(job_path):
    run = Run(load_job(job_path), "STaR")
    data = bridge.read_job(run.job.inputs["retarget"])
    p = run.params

    import sys
    for path in (str(run.job.repo_dir), str(run.job.repo_dir / "submodules")):  # submodules first, as before
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    from Quaternions import Quaternions
    from Pivots import Pivots
    from lab2shot_shared import motion as mo

    stats_dir = run.job.weights_dir
    local_mean = np.load(stats_dir / "mixamo_quat_local_motion_mean.npy")   # (1,22,3)
    local_std = np.load(stats_dir / "mixamo_quat_local_motion_std.npy")
    quat_mean = np.load(stats_dir / "mixamo_quat_mean.npy")       # (1,22,4)
    quat_std = np.load(stats_dir / "mixamo_quat_std.npy")
    local_std[local_std == 0] = 1

    # the two networks stay loaded between jobs (@resident): forward reads nothing a job leaves on them
    encoder, retarget = run.model("load_model", load_networks, run.job.repo_dir, run.job.weights_dir,
                                  stage_params={"model": "STaR"})

    sampled = bridge.resampled_source(data, 30)
    src_axes = bridge.fixed_body(sampled, "src")
    dst_axes = bridge.fixed_body(data, "dst")
    src, dst = bridge.canonical_body(src_axes), bridge.canonical_body(dst_axes)
    src_quat = mo.matrix_to_quat(bridge.star_channels(src["world"], src))  # [F,22,4] (w,x,y,z)
    src_offsets = bridge.star_offsets(src)
    dst_offsets = bridge.star_offsets(dst)
    src_positions = src["world"][..., :3, 3].copy()

    run.stage("prepare_input")
    # Upstream repeats the last position to retain all frames and applies
    # process()'s facing rotation to the root quaternion too.
    seqA, facing = process(np.concatenate((src_positions, src_positions[-1:])), Quaternions, Pivots)
    seqA = seqA[:, 3:]                                                     # drop the reference joint (3_preprocess_q.py)
    T = min(len(seqA), len(src_quat))
    seqA = seqA[:T]
    quatA = src_quat[:T].copy()
    quatA[:, 0] = (facing[:T, 0] * Quaternions(quatA[:, 0])).qs           # the facing rotation on the root channel too
    seqA_pos = seqA[:, :-8].reshape(T, 22, 3)                              # positions, (T,22,3)
    seqA_n = (seqA_pos - local_mean) / local_std                           # normalised per joint
    seqA_off = seqA[:, -8:-4]                                              # vel + rvel
    seqA_n = np.concatenate([seqA_n.reshape(T, -1), seqA_off], axis=-1)    # [T, 70]
    skelA = np.repeat(src_offsets[None], T, axis=0)
    skelB = np.repeat(dst_offsets[None], T, axis=0)
    skelA[:, 0] = seqA_pos[:, 0]
    skelB[:, 0] = seqA_pos[:, 0]
    skelB[:, 0, 1] += dst_offsets[0, 1] - src_offsets[0, 1]
    skelA_n = ((skelA - local_mean) / local_std).reshape(T, -1)
    skelB_n = ((skelB - local_mean) / local_std).reshape(T, -1)
    quatA_n = (quatA - quat_mean) / quat_std
    heightA = np.array([[get_height(src["rest"][:, :3, 3]) / 100.0]], np.float32)
    heightB = np.array([[get_height(dst["rest"][:, :3, 3]) / 100.0]], np.float32)

    def pct_encoding(side):
        ext, cent = centroid_scale(side["points"])
        pts = torch.from_numpy((side["points"] - cent).astype(np.float32)).cuda()
        # Official random_point_select_torch uses NumPy, not torch RNG.
        indices = np.random.choice(len(pts), 1024, replace=len(pts) < 1024)
        pts = pts[torch.as_tensor(indices, device=pts.device)].unsqueeze(0) * SCALE_FACTOR
        return encoder(pts.permute(0, 2, 1)).detach()

    torch.manual_seed(int(p["seed"]))
    np.random.seed(int(p["seed"]))
    run.stage("encode_shape")
    enc_src, enc_dst = pct_encoding(src), pct_encoding(dst)
    shape = torch.zeros((1, 66)).cuda()  # shapeA/shapeB are not read by forward
    info = {"local_mean": torch.from_numpy(local_mean.astype(np.float32)).cuda(),
            "local_std": torch.from_numpy(local_std.astype(np.float32)).cuda(),
            "quat_mean": torch.from_numpy(quat_mean.astype(np.float32)).cuda(),
            "quat_std": torch.from_numpy(quat_std.astype(np.float32)).cuda(),
            "parents": torch.from_numpy(src["parents"]).cuda(), "all_names": []}
    to_t = lambda a: torch.from_numpy(a.astype(np.float32)).unsqueeze(0).cuda()  # noqa: E731

    run.stage("retarget")
    pieces, velocities = [], []
    with torch.inference_mode():
        for start in range(0, T, WINDOW):
            end = min(start + WINDOW, T)
            def clip(a):
                values = a[start:end]
                if len(values) < WINDOW:
                    values = np.concatenate((values, np.repeat(values[-1:], WINDOW - len(values), axis=0)))
                return to_t(values)
            quatB, _, globalB, _, _, _, _ = retarget(
                ["src"], ["dst"], enc_src, enc_dst, clip(seqA_n), clip(skelA_n), clip(skelB_n),
                shape, shape, clip(quatA_n), torch.from_numpy(heightA).cuda(),
                torch.from_numpy(heightB).cuda(), info, None, SCALE_FACTOR)
            pieces.append(quatB[0, :end-start].cpu().numpy())
            velocities.append(globalB[0, :end-start].cpu().numpy())
    quatB = np.concatenate(pieces)
    rot = mo.quat_to_matrix(quatB)
    # Undo facing for the root before FK, so the entire body turns together.
    undo = mo.quat_to_matrix((-facing[:, 0]).qs)
    rot[:, 0] = undo[:-1] @ rot[:, 0]
    offsets = dst_offsets.copy()
    offsets[0, (0, 2)] = src["world"][0, 0, (0, 2), 3]
    world = bridge.star_world(rot, offsets, dst)
    vel = np.concatenate(velocities)[:, :3]
    delta = np.einsum("fab,fb->fa", undo[1:], vel)
    path = np.concatenate((np.zeros((1, 3)), np.cumsum(delta[:-1], axis=0)))
    world[..., :3, 3] += path[:, None]
    world = bridge.restore_axes(world, dst_axes)
    slots = [f"{part}:{k}" for part, k in bridge.STAR_PARTS]
    bridge.write_result(run, bridge.fixed_output(world, dst_axes, data, fps=30),
                        **bridge.sent(src_axes, dst_axes, src_names=slots, dst_names=slots),
                        method="STaR", model_fps=30)


if __name__ == "__main__":
    serve(main)
