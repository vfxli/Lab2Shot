from pathlib import Path
import sys
import types
import numpy as np
import torch
from PIL import Image
from lab2shot_shared.gaussians import transform
from lab2shot_worker import fail, resident, save_npz, serve
from lab2shot_worker.run import Run
from lab2shot_worker.recon import plan_chunks


def model_class(repo):
    # Import the official reconstructor alone. Upstream diffsynth.__init__ imports all
    # generation pipelines eagerly; namespace packages retain relative imports while
    # avoiding loading unrelated Wan / text models into the reconstruction worker.
    for name in ("diffsynth", "diffsynth.models", "diffsynth.auxiliary_models"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(repo / name.replace(".", "/"))]
            sys.modules[name] = module
    from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
    return WorldMirror


@resident
def load_model(repo: Path, checkpoint: Path):
    cls = model_class(repo)
    # 官方 diffsynth 加载路径：state_dict hash = 1a1d001a... -> WorldMirror(enable_norm=False)，
    # strict=True + assign=True，然后 cast 到 bf16 上 GPU（inference.py 的 torch_dtype=bfloat16）
    model = cls(enable_norm=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(torch.bfloat16).cuda().eval()


def _depths(points: np.ndarray, pose: np.ndarray) -> np.ndarray | None:
    """Quartiles of the splats' depth in front of one camera (cam_to_world, OpenCV: +Z forward): the scene's own
    measure of a segment's scale at that frame."""
    z = ((points - pose[:3, 3]) @ pose[:3, :3])[:, 2]
    z = z[z > 0]
    return np.quantile(z, [.25, .5, .75]) if len(z) else None


def _similarity(solved: dict, poses: np.ndarray, indices: list[int], depths: dict, raw: dict) -> np.ndarray:
    """The 4x4 similarity that carries this segment's world onto the frames already solved (the overlap with the
    previous segment); identity for the first segment. Rotation from the overlap's camera rotations (determined even
    when the camera centres do not move); scale from the scene depth seen by the same overlap frames in both segments
    (`depths`: solved world, `raw`: this segment) -- the camera centres of a few neighbouring frames lie millimetres
    apart and give a noisy scale, the scene depth does not; translation from the camera centres."""
    shared = [(j, i) for j, i in enumerate(indices) if i in solved]
    fit = np.eye(4)
    if not shared:
        return fit
    rel = np.stack([solved[i][:3, :3] @ poses[j, :3, :3].T for j, i in shared])
    u, _, vt = np.linalg.svd(rel.mean(0))
    rot = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    src = np.stack([poses[j, :3, 3] for j, _ in shared]) @ rot.T
    dst = np.stack([solved[i][:3, 3] for _, i in shared])
    ratios = [depths[i] / raw[i] for _, i in shared
              if depths.get(i) is not None and raw.get(i) is not None and (raw[i] > 0).all()]
    if ratios:
        scale = float(np.median(np.concatenate(ratios)))
    else:
        scale, spread = 1.0, ((src - src.mean(0)) ** 2).sum()
        if spread > 1e-10:
            scale = float(((src - src.mean(0)) * (dst - dst.mean(0))).sum() / spread)
    if not np.isfinite(scale) or scale <= 0:
        fail("E-NEOVERSE-SEGMENTS")
    fit[:3, :3] = scale * rot
    fit[:3, 3] = dst.mean(0) - scale * src.mean(0)
    return fit


def main(job_path):
    run = Run.start(job_path, "neoverse.reconstruct", "NeoVerse")
    job, p = run.job, run.params
    checkpoint = job.weights_dir / "reconstructor.ckpt"
    run.weights(checkpoint)
    frames = run.frames(step=p["step"], least=2)
    model = run.model("load_model", load_model, job.repo_dir, checkpoint, stage_params={"model": "NeoVerse"})
    # 官方预处理（diffsynth.utils.auxiliary.center_crop）：LANCZOS 放大到覆盖目标再中心裁到模型原生比例 560x336
    # （节点「长边分辨率」是裁后的宽）。`resize` 是画面的缩放比，内参按它和裁切量还原回原画面像素。
    target_w = p["resolution"]
    target_h = round(target_w * 336 / 560)
    resize = max(target_w / frames.width, target_h / frames.height)
    sw, sh = int(frames.width * resize), int(frames.height * resize)
    crop_x, crop_y = (sw - target_w) // 2, (sh - target_h) // 2
    chunks = plan_chunks(len(frames), p["max_frames"], min(4, p["max_frames"] - 1))
    solved, intrinsics, depths = {}, {}, {}
    for first, end in chunks:
        indices = list(range(first, end))
        images = []
        for i in indices:
            with Image.open(frames.paths[i]) as image:
                image = image.convert("RGB").resize((sw, sh), Image.Resampling.LANCZOS)
                image = image.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
                images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
        views = {"img": torch.stack(images)[None].cuda(),
                 "is_target": torch.zeros((1, len(indices)), dtype=torch.bool, device="cuda"),
                 "is_static": torch.zeros((1, len(indices)), dtype=torch.bool, device="cuda"),
                 "timestamp": torch.arange(len(indices), device="cuda")[None]}
        run.stage("reconstruct")
        with torch.inference_mode(), run.frame(), torch.autocast("cuda", dtype=torch.bfloat16):
            # use_motion=False as upstream runs it (inference.py:33): the motion heads only move a gaussian to other
            # timestamps (rasterization.py transition), and every frame here takes its own timestamp's gaussians
            pred = model(views, is_inference=True, use_motion=False)
        poses = pred["rendered_extrinsics"][0].detach().float().cpu().numpy()
        K = pred["rendered_intrinsics"][0].detach().float().cpu().numpy()
        K[:, 0, 2] += crop_x
        K[:, 1, 2] += crop_y
        K[:, :2, :] /= resize
        def splats_at(j, i):
            timestamp = int(pred["rendered_timestamps"][0, j].item())
            splats = [g.transition(timestamp) for g in pred["splats"][0]
                      if g.timestamp == -1 or g.timestamp == timestamp]
            if not splats:
                fail("E-NEOVERSE-NOSPLATS", frame=frames.numbers[i])
            return lambda name: torch.cat([getattr(g, name) for g in splats]).detach().float().cpu().numpy()

        # Same world across segments: the overlap's similarity moves this segment's cameras and gaussians alike
        # (positions, covariance and SH through the one bake, lab2shot_shared.gaussians.transform).
        raw = {i: _depths(splats_at(j, i)("means"), poses[j]) for j, i in enumerate(indices) if i in solved}
        fit = _similarity(solved, poses, indices, depths, raw)
        scale = float(np.cbrt(np.linalg.det(fit[:3, :3])))
        # Only the frames not solved yet are committed: a moving gaussian is never blended with an unrelated point
        # identity at the overlap.
        for j, i in enumerate(indices):
            if i in solved:
                continue
            arr = splats_at(j, i)
            points = arr("means")
            values = transform({"points": points, "scales": arr("scales"), "rotations": arr("rotations"),
                                "opacity": arr("opacities").reshape(-1), "sh": arr("harmonics").reshape(len(points), -1, 3)},
                               fit)
            save_npz(job.raw_dir / f"gaussian_{frames.numbers[i]}.npz", **values)
            pose = fit @ poses[j]
            pose[:3, :3] /= scale  # a camera keeps a pure rotation; the scale lives in its position
            solved[i], intrinsics[i] = pose, K[j]
            depths[i] = _depths(np.asarray(values["points"], np.float64), pose)
        del pred, views
        torch.cuda.empty_cache()
    save_npz(job.raw_dir / "cameras.npz", frames=np.asarray(frames.numbers),
             K=np.stack([intrinsics[i] for i in range(len(frames))]),
             cam_to_world=np.stack([solved[i] for i in range(len(frames))]), width=frames.width, height=frames.height)
    # run.finish 默认只写 [首, 尾] 两帧；动态家族要完整帧表逐帧读回 gaussian_<frame>.npz
    run.finish(frames.numbers, dynamic=True, kind="gaussian", resolution=p["resolution"], segments=len(chunks),
               frames=frames.numbers)


if __name__ == "__main__":
    serve(main)
