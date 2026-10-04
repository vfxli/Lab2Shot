"""Run official WildPose algorithms on a Lab2Shot frame sequence."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from lab2shot_worker import fail, save_npz, serve
from lab2shot_worker.recon import save_cameras
from lab2shot_worker.run import Run


def main(job_path):
    import cv2
    import torch.multiprocessing as mp
    from moge.model.v2 import MoGeModel
    from src.utils.datasets import BaseDataset, Sintel
    from src.modules.mast3r_utils import load_mast3r
    import src.modules.models as models
    import src.utils.metric_depth_estimators as estimators
    from src.slam import SLAM

    mp.set_start_method("spawn", force=True)
    run = Run.start(job_path, "wildpose.reconstruct", "WildPose")
    job, p = run.job, run.params
    shot = run.frames(step=p["step"], least=8)
    run.weights(*(job.weights_dir / filename for filename in ("wildpose_v0.pth", "mast3r.pth", "moge2.pt")))
    cfg = yaml.safe_load((job.repo_dir / "configs/wildgs_slam.yaml").read_text())
    cfg.update(scene="shot", device="cuda:0", verbose=False)
    cfg["data"] = {"input_folder": str(job.raw_dir), "output": str(job.raw_dir / "_wildpose")}
    # 「处理分辨率」 is the long side, as every upstream config sets MASt3R-512's input (configs/wildgs_slam.yaml:64-65,
    # the long side 512 it was trained at): taken as the width, a portrait plate's long side came out near 912.
    # The model cuts the picture into 16-pixel patches (it asserts both sides are multiples of 16): a wide frame's
    # height rounded to 8 failed on KITTI (152)
    long_side = max(16, round(int(p["resolution"]) / 16) * 16)
    short = lambda a, b: max(16, round(a * long_side / b / 16) * 16)  # noqa: E731
    width, height = ((long_side, short(shot.height, shot.width)) if shot.width >= shot.height
                     else (short(shot.width, shot.height), long_side))
    fx = float(p["focal_px"])
    cx, cy = (float(v) for v in p["principal_px"])  # the picture's centre in the pixels sent (nodes.py prepare)
    cfg["cam"].update(H=shot.height, W=shot.width, fx=fx, fy=fx, cx=cx, cy=cy,
                       H_out=height, W_out=width, H_edge=0, W_edge=0)
    cfg["tracking"].update(pretrained=str(job.weights_dir / "wildpose_v0.pth"), buffer=p["buffer"], full_ba=False)
    cfg["tracking"]["backend"]["final_ba"] = p["final_ba"]

    class Sequence(Sintel):
        # Sintel marks a sequence without ground-truth pose in upstream's
        # terminate() path. BaseDataset retains official resize/calibration.
        def __init__(self):
            BaseDataset.__init__(self, cfg)
            self.n_img = len(shot)
            self.color_paths = [str(path) for path in shot.paths]
            self.image_timestamps = np.arange(len(shot)) / job.fps

    def local_mast3r(*args, **kwargs):
        return load_mast3r(str(job.weights_dir / "mast3r.pth"), device="cuda")

    def local_metric(config):
        return MoGeModel.from_pretrained(str(job.weights_dir / "moge2.pt")).cuda().eval()

    # upstream run.py: setup_seed(cfg["setup_seed"]) before anything is built (43 in every config)
    import random
    torch.manual_seed(cfg["setup_seed"])
    torch.cuda.manual_seed_all(cfg["setup_seed"])
    np.random.seed(cfg["setup_seed"])
    random.seed(cfg["setup_seed"])
    torch.backends.cudnn.deterministic = True
    with patch.object(models, "load_mast3r", local_mast3r), patch.object(estimators, "get_metric_depth_estimator", local_metric):
        # motion_filter and trajectory_filler imported the function by name
        # (from src.utils.metric_depth_estimators import ...), so their own
        # namespace bindings must be patched too, not just the module's.
        import src.motion_filter as filtering
        import src.trajectory_filler as filler
        with patch.object(filtering, "get_metric_depth_estimator", local_metric), \
             patch.object(filler, "get_metric_depth_estimator", local_metric):
            slam = run.model("load_model", SLAM, cfg, Sequence(), stage_params={"model": "WildPose"})
            run.stage("track_and_ba")
            try:
                slam.run()
            finally:
                slam.printer.terminate()

    folder = job.raw_dir / "_wildpose/shot"
    if cfg["delete_mono_priors"]:  # upstream run.py: the per-frame priors are only the SLAM's working files
        import shutil
        shutil.rmtree(folder / "mono_priors", ignore_errors=True)
    tum = np.loadtxt(folder / "traj/est_poses_full.txt", ndmin=2)
    from scipy.spatial.transform import Rotation
    poses = np.repeat(np.eye(4)[None], len(tum), axis=0)
    poses[:, :3, :3] = Rotation.from_quat(tum[:, 4:8]).as_matrix()
    poses[:, :3, 3] = tum[:, 1:4]
    if len(poses) != len(shot):
        fail("E-WILDPOSE-FRAMES", got=len(poses), frames=len(shot))
    poses = np.linalg.inv(poses[0]) @ poses
    K = np.repeat(np.eye(3)[None], len(shot), axis=0)
    K[:, 0, 0] = K[:, 1, 1] = fx
    K[:, 0, 2], K[:, 1, 2] = cx, cy
    save_cameras(job.raw_dir, shot.numbers, K, poses, shot.width, shot.height)
    video = np.load(folder / "video.npz")
    for stamp, depth, mask in zip(video["timestamps"], video["depths"], video["valid_depth_masks"], strict=True):
        index = int(stamp)
        # up to the plate: bilinear over the valid pixels only (weights renormalised), so the edge of a hole is not
        # blended with the zeros inside it
        keep = np.isfinite(depth) & (depth > 0)
        size = (shot.width, shot.height)
        weight = cv2.resize(keep.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        z = cv2.resize(np.where(keep, depth, 0).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        z = np.where(weight > 1e-6, z / np.maximum(weight, 1e-6), 0).astype(np.float32)
        usable = cv2.resize((keep & (mask > 0)).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        save_npz(job.raw_dir / f"frame_{shot.numbers[index]}.npz", depth=z, mask=usable & np.isfinite(z) & (z > 0))
    run.finish(shot.numbers, kind="reconstruction", metric=True, depth_frames="keyframes", models=["WildPose", "MASt3R", "MoGe-2"])


if __name__ == "__main__":
    serve(main)
