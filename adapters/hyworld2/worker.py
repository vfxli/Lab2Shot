from pathlib import Path
import numpy as np
import torch
from lab2shot_worker import fail, resident, save_npz, say, serve
from lab2shot_worker.run import Run


@resident
def load_model(path: Path):
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    return WorldMirrorPipeline.from_pretrained(str(path))


def _official_splats(run, paths, images, predictions):
    """The gaussians the official pipeline writes to gaussians.ply: the raw per-pixel splats,
    sky+edge masked, voxel merged, capped and large-scale filtered — every step an upstream
    function (worldrecon/pipeline.py __call__ + inference_utils.save_results + save_utils.save_gs_ply)."""
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import (
        compute_filter_mask, compute_sky_mask, _voxel_prune_gaussians)

    _, _, _, height, width = images.shape
    steps = len(paths)
    # 官方默认：apply_sky_mask=True（source="auto"、阈值 0.45）、apply_edge_mask=True、apply_confidence_mask=False
    sky = compute_sky_mask(paths, height, width, steps, predictions=predictions, source="auto",
                           model_threshold=0.45, processed_aspect_ratio=width / height)
    _, gs_mask = compute_filter_mask(predictions, images, paths, height, width, steps,
                                     apply_confidence_mask=False, apply_edge_mask=True, apply_sky_mask=True,
                                     confidence_percentile=10.0, edge_normal_threshold=1.0,
                                     edge_depth_threshold=0.03, sky_mask=sky, use_gs_depth=True)
    sp = predictions["splats"]
    # 官方 save_results 在体素合并前一律 .detach().cpu()（_voxel_prune_gaussians 只吃 CPU 张量）
    means = sp["means"][0].reshape(-1, 3).detach().cpu().float()
    scales = sp["scales"][0].reshape(-1, 3).detach().cpu().float()
    quats = sp["quats"][0].reshape(-1, 4).detach().cpu().float()
    colors = (sp["sh"][0] if "sh" in sp else sp["colors"][0]).reshape(-1, 3).detach().cpu().float()
    opacities = sp["opacities"][0].reshape(-1).detach().cpu().float()
    weights = sp["weights"][0].reshape(-1).detach().cpu().float() if "weights" in sp else torch.ones_like(opacities)
    keep = torch.from_numpy((gs_mask if gs_mask is not None else sky).reshape(-1)).bool()
    means, scales, quats, colors, opacities, weights = (v[keep] for v in (means, scales, quats, colors, opacities, weights))
    run.stage("merge_gaussians")
    means, scales, quats, colors, opacities = _voxel_prune_gaussians(means, scales, quats, colors, opacities, weights)
    if len(means) > 5_000_000:  # compress_gs_max_points（官方默认）
        pick = torch.from_numpy(np.random.default_rng(42).choice(len(means), 5_000_000, replace=False)).long()
        means, scales, quats, colors, opacities = (v[pick] for v in (means, scales, quats, colors, opacities))
    large = torch.quantile(scales.max(dim=-1)[0], 0.98)  # save_gs_ply 的 0.98 分位过滤
    small = scales.max(dim=-1)[0] <= large
    # sh 度数为 0（gs_renderer sh_degree=0）：合并时颜色是 [N,3]，输出契约是 [N,1,3]
    return {k: v[small] for k, v in zip(("points", "scales", "rotations", "sh", "opacity"),
                                        (means, scales, quats, colors.reshape(-1, 1, 3), opacities))}


# Peak VRAM (reserved) measured on an RTX 5090, 32 views of a 1280x720 shot: 18.1 GB at 518; the model holds 4.8 GB,
# the rest grows with the input's pixels and the view count (the gaussian renderer's convolutions over every view).
# At upstream's 952 that is ~50 GB: more than either card here, so 952 runs only where the card holds it.
MODEL_GB = 4.8
ACTIVATION_GB_518_32 = 13.3
RESOLUTIONS = (952, 518, 280)  # the node's settings, largest first


def resolution_vram_gb(size: int, views: int) -> float:
    return MODEL_GB + ACTIVATION_GB_518_32 * (size / 518) ** 2 * views / 32


def fitting_resolution(paths: list[str], effective: int, views: int, adaptive, budget: float | None = None) -> int:
    """The asked processing size, or the largest smaller one that fits (said on the node) instead of running out of
    memory. `budget`: the VRAM the scheduler gave this run's tier (the job's vram_budget_gb): the size is picked by it
    alone, so the result is the tier the cache key says; a card that has less free than the size picked needs is an
    error, never a further silent step down. None: by the card's free memory."""
    free, _total = torch.cuda.mem_get_info()
    free = (free + torch.cuda.memory_reserved()) / 2**30
    card = float(budget) if budget else free
    picked = None
    if resolution_vram_gb(effective, views) <= card:
        picked = effective
    else:
        for size in RESOLUTIONS:
            if size < effective and resolution_vram_gb(size, views) <= card:
                picked = size
                break
    if picked is None:
        # 最低一档也放不下：说清楚，不硬跑到显存溢出
        low = min(RESOLUTIONS[-1], effective)
        fail("E-HYWORLD2-VRAM", size=low, views=views, card=card, need=resolution_vram_gb(low, views))
    if budget and resolution_vram_gb(picked, views) > free:
        fail("E-WORKER-VRAMSHORT", free=free, need=resolution_vram_gb(picked, views))
    if picked == effective:
        return effective
    fitted = adaptive(paths, picked)
    say("W-HYWORLD2-VRAMSTEP", asked=effective, size=fitted, views=views, card=card,
        need=resolution_vram_gb(effective, views))
    return fitted


def main(job_path):
    run = Run.start(job_path, "hyworld2.reconstruct", "HY-World 2.0")
    job, p = run.job, run.params
    root = job.weights_dir
    run.weights(root / "HY-WorldMirror-2.0/model.safetensors", root / "HY-WorldMirror-2.0/config.json",
                root / "skyseg.onnx")
    frames = run.frames(step=p["step"], least=2)
    pick = np.unique(np.linspace(0, len(frames)-1, min(len(frames), p["max_views"])).round().astype(int))
    paths = [str(frames.paths[i]) for i in pick]
    numbers = [frames.numbers[i] for i in pick]
    pipe = run.model("load_model", load_model, root, stage_params={"model": "WorldMirror 2.0"})
    # 官方 compute_sky_mask 在工作目录找 skyseg.onnx（工作目录是上游检出）；worker 无网，
    # 不能依赖上游的联网回退，故把校验过的权重链接进工作目录
    link = Path("skyseg.onnx")
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to((root / "skyseg.onnx").resolve())
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import compute_adaptive_target_size
    effective = compute_adaptive_target_size(paths, p["resolution"])  # 官方：不放大小图
    effective = fitting_resolution(paths, effective, len(paths), compute_adaptive_target_size, p.get("vram_budget_gb"))
    run.stage("reconstruct_gaussians")
    with torch.inference_mode(), run.frame():
        predictions, images, _ = pipe._run_inference(paths, effective, None, None)
    sp = _official_splats(run, paths, images, predictions)
    arr = lambda x: x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    save_npz(job.raw_dir / f"gaussian_{numbers[0]}.npz", **{k: arr(v) for k, v in sp.items()})
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import compute_preprocessing_transform
    transform = compute_preprocessing_transform(paths, effective)
    K = arr(predictions["camera_intrs"][0])
    K[:, 0, 2] += transform["crop_x"]; K[:, 1, 2] += transform["crop_y"]
    K[:, 0, :] /= transform["scale_x"]; K[:, 1, :] /= transform["scale_y"]
    cams = arr(predictions["camera_poses"][0])
    if cams.shape[-2:] == (3, 4):
        cams = np.concatenate((cams, np.tile(np.array([0, 0, 0, 1], np.float32), (len(cams), 1, 1))), axis=1)
    save_npz(job.raw_dir / "cameras.npz", frames=np.asarray(numbers), K=K, cam_to_world=cams,
             width=frames.width, height=frames.height)
    run.finish(numbers, dynamic=False, kind="gaussian", model="WorldMirror 2.0", resolution=effective,
               max_views=len(paths))


if __name__ == "__main__":
    serve(main)
