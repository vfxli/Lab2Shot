"""Official VidMap frontend/mapping, fixed local models and native outputs."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lab2shot_worker import fail, local_hub, save_npz, serve


def main(job_path):
    # Import Ceres/COLMAP before torch, as required by the official run.py:
    # otherwise libtorch's BLAS symbols can make CHOLMOD factorization fail.
    from vidmap.mapper.runtime import load_mapping_runtime
    load_mapping_runtime()
    import torch
    import huggingface_hub
    import huggingface_hub.hub_mixin as mixin
    from lab2shot_worker.recon import save_cameras
    from lab2shot_worker.run import Run

    run = Run.start(job_path, "vidmap.camera_solve", "VidMap")
    job, params = run.job, run.params
    shot = run.frames(step=params["step"], least=3)
    local_hub({"dinov3": job.repo_dir.parent / "dinov3"}, "VidMap")
    # the saved (compiled) RoMa graph imports DINOv3 from torch hub's own checkout folder, named after the pinned
    # commit (frontend/models/romav2_compile_cache.py:50-56); torch.hub.load never runs on that path, so the
    # installer's checkout (the same commit, extension.py extra_sources) is linked there
    pinned = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa"
    if not pinned.is_dir():
        pinned.parent.mkdir(parents=True, exist_ok=True)
        try:
            pinned.symlink_to(job.repo_dir.parent / "dinov3", target_is_directory=True)
        except FileExistsError:  # another worker linked it first
            pass
    hf_models = {
        ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "b2359bdf726fb44ef62acca04d629dcf158053e7"): "da3",
        ("gberton/MegaLoc", "7cb9f7970d366fdf059963d04d372e503e8e9df9"): "megaloc",
    }

    def local_hf(repo_id, filename, revision=None, **kw):
        folder = hf_models.get((repo_id, revision))
        if folder is None:
            fail("E-VIDMAP-UNDECLARED", model=f"{repo_id}@{revision}/{filename}")
        path = job.weights_dir / folder / filename
        run.weights(path)
        return str(path)

    inputs = job.raw_dir / "_ordered"
    inputs.mkdir(exist_ok=True)
    frame_by_name = {}
    for index, (number, path) in enumerate(shot):
        target = inputs / f"{index:09d}.png"
        if not target.exists():
            target.symlink_to(path.resolve())
        frame_by_name[target.name] = number
    output = job.raw_dir / "_vidmap"
    output.mkdir(exist_ok=True)
    focal = params.get("focal_px")
    intrinsics_path = None
    if focal is not None:
        intrinsics_path = output / "intrinsics.json"
        intrinsics_path.write_text(json.dumps({"1": {"params": [focal, focal, shot.width / 2, shot.height / 2], "images": "all"}}))
    mode = "calib" if focal is not None else "uncalib"

    with ExitStack() as stack:
        stack.enter_context(patch.object(huggingface_hub, "hf_hub_download", local_hf))
        stack.enter_context(patch.object(mixin, "hf_hub_download", local_hf))
        from vidmap.configuration.build import build_frontend_config, build_mapping_config
        from vidmap.configuration.names import FRONTEND_CONFIG_DIR, MAPPING_CONFIG_DIR, resolve_config_path
        from vidmap.frontend.runner import run_local_frontend
        from vidmap.reconstruction import reconstruct
        from vidmap.run_options import RunOptions

        # upstream defaults: RoMaV2 and DA3 run as torch.compile graphs (frontend/options/matching.py, depth.py), which
        # the environment's PyTorch 2.14 (the README's tested version) supports; saved under VIDMAP_*_CACHE_DIR
        front = build_frontend_config(resolve_config_path(mode + "/base", FRONTEND_CONFIG_DIR), source_name=mode + "/base")
        mapping_name = "uncalib/smooth_trajectory" if params["smooth"] and mode == "uncalib" else mode + "/base"
        mapping = build_mapping_config(resolve_config_path(mapping_name, MAPPING_CONFIG_DIR), source_name=mapping_name)
        run.stage("frontend")
        frontend = run_local_frontend(front, inputs, workspace=output, intrinsics_path=intrinsics_path, device="cuda")
        run.stage("global_mapping")
        rec = reconstruct(mapping, front, inputs, workspace=output, mapper_inputs_dir=frontend.mapper_inputs,
                          intrinsics_path=intrinsics_path, run_options=RunOptions(device="cuda", terminate_on_error=True),
                          overwrite_outputs=True, output_dir=output)
    images = sorted((image for image in rec.images.values() if image.name in frame_by_name and image.has_pose),
                    key=lambda image: frame_by_name[image.name])
    if len(images) < 2:
        fail("E-VIDMAP-FEWCAMERAS", registered=len(images), frames=len(shot))
    poses, K, numbers = [], [], []
    for image in images:
        transform = np.eye(4)
        transform[:3] = image.cam_from_world().inverse().matrix()
        poses.append(transform)
        K.append(rec.cameras[image.camera_id].calibration_matrix())
        numbers.append(frame_by_name[image.name])
    origin = np.linalg.inv(poses[0])
    poses = origin @ np.asarray(poses)
    save_cameras(job.raw_dir, numbers, np.asarray(K), poses, shot.width, shot.height)
    points = list(rec.points3D.values())
    xyz = np.asarray([point.xyz for point in points], np.float64).reshape(-1, 3)
    rgb = np.asarray([point.color for point in points], np.float32).reshape(-1, 3)
    save_npz(job.raw_dir / "points.npz", xyz=xyz @ origin[:3, :3].T + origin[:3, 3], rgb=rgb)
    run.finish(numbers, kind="camera", metric=False, registered=len(images), input_frames=len(shot), sparse_points=len(points))


if __name__ == "__main__":
    serve(main)
