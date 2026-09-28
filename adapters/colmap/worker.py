"""COLMAP worker: classic structure-from-motion camera solve. Runs inside
third_party/colmap/.venv (official pycolmap-cuda12 wheel); never imports Lab2Shot core.

    python worker.py <job.json>

Frames (every `step`-th) -> SIFT features (CPU / VLFeat by default, GPU / SiftGPU
on request; moving objects masked out) ->
sequential / exhaustive matching -> global (GLOMAP) or incremental mapper ->
raw/cameras.json (lens + camera-to-world per registered frame), raw/points.npz
(sparse point cloud), raw/sparse/ (the COLMAP binary model, same world).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from lab2shot_worker import cpu_budget, fail, nothing, progress, reason, recon, save_npz, say, serve
from lab2shot_worker.run import Run

NODE = "colmap.camera_solve"
CONVENTION = (
    "OpenCV camera: +X right, +Y down, +Z forward; poses are camera-to-world, row-major 4x4; "
    "world = first registered frame's camera, arbitrary scale (COLMAP)"
)
MIN_REGISTERED = 3
# Below this the camera barely moved relative to the scene depth: no parallax.
MIN_BASELINE_RATIO = 0.02
MIN_TRI_ANGLE_DEG = 1.0
NARROW_FOV_DEG = 25.0  # long-side field of view below which a solved focal is unreliable (~80 mm full frame)


# ---------------------------------------------------------------------- COLMAP


def link_images(frames: list[tuple[int, Path]], image_dir: Path) -> dict[str, int]:
    """COLMAP names = zero-padded order index: the sequential matcher pairs images by
    name order, so the job's frame order is kept whatever the frame numbers are."""
    image_dir.mkdir(parents=True)
    names: dict[str, int] = {}
    for i, (frame, path) in enumerate(frames):
        name = f"{i:06d}{path.suffix.lower()}"
        os.symlink(path.resolve(), image_dir / name)
        names[name] = frame
    return names


def camera_from_model(model: str, width: int, height: int, focal_px: float) -> str:
    """Known lens as COLMAP's comma string (principal point at the center, no distortion)."""
    import pycolmap

    camera = pycolmap.Camera.create_from_model_name(1, model, focal_px, width, height)
    return ",".join(repr(float(v)) for v in camera.params)


def run_colmap(run: Run, work: Path, names: dict[str, int], masked: dict[str, np.ndarray], params: dict,
               focal_px: float | None) -> tuple[dict, dict[str, float], bool]:
    """COLMAP's models, the seconds of each step (features, matching, mapping) and whether SIFT ran on the GPU."""
    import pycolmap
    from PIL import Image

    image_dir = work / "images"
    database = work / "database.db"
    width, height = run.job.width, run.job.height
    # GPU SIFT = SiftGPU (non-commercial licence); CPU SIFT = VLFeat (BSD), slower.
    use_gpu = bool(params["sift_gpu"]) and pycolmap.has_cuda and pycolmap.get_num_cuda_devices() > 0
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    if params["sift_gpu"] and not use_gpu:
        say("N-COLMAP-NOGPU")

    reader = pycolmap.ImageReaderOptions()
    # Unknown focal with a distorting model runs in two stages. Geometric verification undistorts with the camera, and
    # for a distorting model COLMAP only has a guessed default focal (1.2 x the long side); fisheye projection under
    # that focal is wrong, so every pair fails verification ("no matching points between frames").
    # Features are therefore extracted, matched and verified as SIMPLE_PINHOLE (the fundamental matrix does not depend
    # on the focal), then the database camera is replaced by the target model (estimated focal, zero coefficients)
    # and the mapper refines focal and coefficients. Not used when the focal is known: the camera is correct from the start.
    two_stage = focal_px is None and params["fit_model"] not in ("SIMPLE_PINHOLE", "PINHOLE")
    reader.camera_model = "SIMPLE_PINHOLE" if two_stage else params["fit_model"]
    if two_stage:
        say("I-COLMAP-TWOSTAGE", model=params["fit_model"])
    if focal_px:
        reader.camera_params = camera_from_model(params["fit_model"], width, height, focal_px)
    if masked:
        # COLMAP masks: <mask_path>/<image name>.png, black = ignore features there.
        mask_dir = work / "masks"
        mask_dir.mkdir()
        for name in names:
            keep = ~masked[name] if name in masked else np.ones((height, width), dtype=bool)
            Image.fromarray(keep.astype(np.uint8) * 255).save(mask_dir / f"{name}.png")
        reader.mask_path = str(mask_dir)

    # COLMAP spawns its own threads (default -1 = machine core count). The core pins this process to the cores outside
    # 「保留核心数」; without passing the count down, it would run 32 threads on 24 cores and saturate the machine.
    threads = cpu_budget()

    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = params["resolution"]
    extraction.use_gpu = use_gpu
    extraction.num_threads = threads

    run.stage("提取特征 (SIFT)")
    t0 = time.time()
    # One camera for the whole shot (a zoom lens would need one per frame: not supported).
    pycolmap.extract_features(
        database, image_dir, image_names=sorted(names), camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader, extraction_options=extraction, device=device,
    )
    # Known focal: a prior (kept fixed below). Unknown: COLMAP's placeholder guess
    # (1.2 x the long side) must not count as a prior, or the global mapper trusts it.
    db = pycolmap.Database.open(database)
    try:
        for camera in db.read_all_cameras():
            camera.has_prior_focal_length = focal_px is not None
            db.update_camera(camera)
    finally:
        db.close()

    seconds = {"features": time.time() - t0}
    run.stage("匹配特征")
    t0 = time.time()
    matching = pycolmap.FeatureMatchingOptions()
    matching.use_gpu = use_gpu
    matching.num_threads = threads
    if params["matcher"] == "exhaustive":
        pycolmap.match_exhaustive(database, matching_options=matching, device=device)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.loop_detection = False  # needs a vocabulary tree download: off (offline worker)
        pycolmap.match_sequential(database, matching_options=matching, pairing_options=pairing, device=device)

    seconds["matching"] = time.time() - t0
    if two_stage:  # after verification: estimate a focal from the matches on the pinhole camera (view graph
        # calibration; the fundamental matrix is independent of distortion), then switch the camera to the target model
        # initialised with that focal, zero coefficients and no prior; the mapper refines the coefficients too
        # (TUM-VI fisheye: 45/45 frames registered, focal 191 vs. ground truth 191)
        if params["mapper"] != "incremental" and not pycolmap.calibrate_view_graph(database):
            say("W-COLMAP-VIEWGRAPH")
        db = pycolmap.Database.open(database)
        try:
            for camera in db.read_all_cameras():
                swapped = pycolmap.Camera.create_from_model_name(camera.camera_id, params["fit_model"], float(camera.params[0]),
                                                                 camera.width, camera.height)
                swapped.has_prior_focal_length = False
                db.update_camera(swapped)
        finally:
            db.close()
    t0 = time.time()
    sparse_dir = work / "sparse"
    sparse_dir.mkdir()
    refine_focal = focal_px is None
    total = len(names)
    if params["mapper"] == "incremental":
        run.stage("增量解算相机")
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = threads  # 「保留核心数」: bundle adjustment is the most CPU-heavy step
        options.ba_refine_focal_length = refine_focal
        options.mapper.abs_pose_refine_focal_length = refine_focal
        registered = [0]

        def next_image():
            registered[0] += 1
            if registered[0] % 5 == 0:
                progress(min(registered[0], total), total, "注册画面")

        models = pycolmap.incremental_mapping(
            database, image_dir, sparse_dir, options=options,
            initial_image_pair_callback=lambda: progress(2, total, "初始画面对"),
            next_image_callback=next_image,
        )
    else:
        if focal_px is None and not two_stage:  # the two-stage path already calibrated on the pinhole camera
            # GLOMAP needs a focal close to the truth to start from: estimate it from
            # the fundamental matrices of all matched pairs first (COLMAP's view_graph_calibrator).
            run.stage("估计 Focal Length (视图图标定)")
            if not pycolmap.calibrate_view_graph(database):
                say("W-COLMAP-VIEWGRAPH")
        run.stage("全局解算相机 (GLOMAP)")
        options = pycolmap.GlobalPipelineOptions()
        options.num_threads = threads  # 「保留核心数」 (reserved cores)
        options.mapper.bundle_adjustment.refine_focal_length = refine_focal
        models = pycolmap.global_mapping(database, image_dir, sparse_dir, options=options)
    seconds["mapping"] = time.time() - t0
    return models, seconds, use_gpu


def pair_stats(database: Path) -> dict[str, int]:
    """What the matcher found between frame pairs: parallax (general), rotation-only /
    planar (no depth information), or too few consistent matches (degenerate)."""
    import pycolmap

    kind = pycolmap.TwoViewGeometryConfiguration
    groups = {
        "parallax": (kind.CALIBRATED, kind.UNCALIBRATED, kind.CALIBRATED_RIG),
        "rotation_or_planar": (kind.PANORAMIC, kind.PLANAR, kind.PLANAR_OR_PANORAMIC),
    }
    db = pycolmap.Database.open(database)
    try:
        geometries = db.read_two_view_geometries()[1]
    finally:
        db.close()
    stats = {"pairs": len(geometries), "parallax": 0, "rotation_or_planar": 0, "degenerate": 0}
    for geometry in geometries:
        group = next((g for g, members in groups.items() if geometry.config in [int(m) for m in members]), "degenerate")
        stats[group] += 1
    return stats


def unsolved(pairs: dict[str, int], mapper: str, registered: int = 0) -> None:
    """End a solve that registered too few frames, saying why from the verified pair statistics. No frame pair matched at
    all: matching found nothing, an empty result (nothing(); an empty result is not an error); pairs that matched
    without enough parallax: the shot does not meet COLMAP's premise, an error."""
    if pairs["parallax"] + pairs["rotation_or_planar"] == 0:
        nothing("N-COLMAP-NOMATCHES", registered=registered, pairs=pairs["pairs"])
    if pairs["rotation_or_planar"] >= 0.8 * (pairs["parallax"] + pairs["rotation_or_planar"]):
        fail("E-COLMAP-NOPARALLAX", registered=registered, rotation=pairs["rotation_or_planar"], pairs=pairs["pairs"])
    fail("E-COLMAP-WEAKPARALLAX", registered=registered, parallax=pairs["parallax"], pairs=pairs["pairs"],
         other_mapper="全局" if mapper == "incremental" else "增量")


# ---------------------------------------------------------------------- results


def to_first_camera(rec, first_image_id: int) -> None:
    """Move the world so the first registered frame's camera is the origin (like ViPE)."""
    import pycolmap

    cam_from_world = rec.image(first_image_id).cam_from_world()
    rec.transform(pycolmap.Sim3d(1.0, cam_from_world.rotation, cam_from_world.translation))


def camera_to_world(image) -> np.ndarray:
    m = np.eye(4)
    m[:3, :4] = image.cam_from_world().inverse().matrix()
    return m


def path_stats(centers: np.ndarray, frames: list[int]) -> dict:
    """Frame-to-frame camera jumps (per solved step): a smooth path has max ~ a few x median."""
    if len(centers) < 3:
        return {}
    steps = np.diff(frames)
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / steps
    median = float(np.median(jumps))
    return {
        "path_length": float(jumps.sum()),
        "median_jump": median,
        "max_jump": float(jumps.max()),
        "max_jump_frame": int(frames[int(np.argmax(jumps)) + 1]),
        "max_over_median": float(jumps.max() / median) if median > 0 else None,
    }


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "COLMAP", gpu=False)  # pycolmap, not torch: no GPU bookkeeping (SiftGPU is its own)
    job, params = run.job, run.params  # sift_gpu: GPU SIFT is SiftGPU (non-commercial); off = CPU SIFT (VLFeat, BSD)
    step = params["step"]
    width, height = job.width, job.height
    focal_px = params["focal_px"]  # a known lens: focal in pixels at the plate width, else None

    import pycolmap

    all_frames = job.frames
    frame_numbers = [f for f, _ in all_frames]
    used = all_frames[::step]
    if len(used) < MIN_REGISTERED:
        fail("E-WORKER-TOOFEWFRAMES", least=MIN_REGISTERED, have=len(used),
             why=reason("I-WORKER-WHYSTEP", step=step))

    raw = job.raw_dir
    work = job.scratch("colmap")

    run.stage("准备画面")
    names = link_images(used, work / "images")
    moving = recon.MovingMasks(job, width, height)  # the node's moving-object masks: no features there
    masked = {name: moving.get(frame) for name, frame in names.items() if frame in moving}
    if masked:
        share = float(np.mean([m.mean() for m in masked.values()]))
        print(f"masks: {len(masked)} frames, {share:.1%} of their pixels ignored", flush=True)

    models, step_seconds, sift_on_gpu = run_colmap(run, work, names, masked, params, focal_px)
    pairs = pair_stats(work / "database.db")
    print(f"frame pairs: {pairs}", flush=True)
    if not models:
        unsolved(pairs, params["mapper"])
    ranked = sorted(models.values(), key=lambda r: r.num_reg_images(), reverse=True)
    rec = ranked[0]
    if len(ranked) > 1:
        say("W-COLMAP-SPLIT", count=len(ranked), sizes=[r.num_reg_images() for r in ranked])

    by_frame = {names[img.name]: img for img in rec.images.values() if img.has_pose}
    registered = sorted(by_frame, key=frame_numbers.index)
    if len(registered) < MIN_REGISTERED:
        unsolved(pairs, params["mapper"], len(registered))
    to_first_camera(rec, by_frame[registered[0]].image_id)

    camera = next(iter(rec.cameras.values()))
    poses = {str(f): camera_to_world(by_frame[f]) for f in registered}
    centers = np.array([poses[str(f)][:3, 3] for f in registered])
    stats = path_stats(centers, registered)

    points = rec.points3D
    xyz = np.array([p.xyz for p in points.values()], dtype=np.float32).reshape(-1, 3)
    rgb = np.array([p.color for p in points.values()], dtype=np.uint8).reshape(-1, 3)
    error = np.array([p.error for p in points.values()], dtype=np.float32)
    track_len = np.array([p.track.length() for p in points.values()], dtype=np.int32)
    save_npz(raw / "points.npz", xyz=xyz, rgb=rgb, error=error, track_len=track_len)

    # Parallax check: camera travel vs. how far away the scene is, and triangulation angles.
    depth = []
    for f in registered[:: max(1, len(registered) // 20)]:
        img = by_frame[f]
        w2c = img.cam_from_world().matrix()
        pts = np.array([rec.point3D(p.point3D_id).xyz for p in img.points2D if p.has_point3D()])
        if len(pts):
            depth.append(np.median(pts @ w2c[:, :3].T + w2c[:, 3], axis=0)[2])
    scene_depth = float(np.median(depth)) if depth else float("nan")
    baseline = float(np.max(np.linalg.norm(centers - centers[0], axis=1)))
    baseline_ratio = baseline / scene_depth if scene_depth > 0 else float("nan")
    tri_angle = _median_triangulation_angle(rec, points, centers_by_image={
        img.image_id: img.projection_center() for img in by_frame.values()})
    if baseline_ratio < MIN_BASELINE_RATIO or tri_angle < MIN_TRI_ANGLE_DEG:
        say("W-COLMAP-NOBASELINE", ratio=float(baseline_ratio), angle=float(tri_angle))
    f_solved = float(camera.mean_focal_length())
    fov_long = float(np.degrees(2 * np.arctan(max(width, height) / (2 * f_solved))))
    if focal_px is None and fov_long < NARROW_FOV_DEG:
        # Long lens: nearly orthographic, so focal length and depth trade off against
        # each other (and against radial distortion): the solved focal can be 10-25 % off.
        say("W-COLMAP-LONGLENS", fov=fov_long, focal=f_solved)
    verified = pairs["parallax"] + pairs["rotation_or_planar"]
    if verified and pairs["rotation_or_planar"] > 0.5 * verified:
        say("W-COLMAP-ROTATION", rotation=pairs["rotation_or_planar"], verified=verified)
    if len(registered) < len(used):
        missing = len(used) - len(registered)
        say("W-COLMAP-UNREGISTERED", missing=missing, used=len(used))

    sparse = raw / "sparse"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir()
    rec.write(sparse)

    param_names = [s.strip() for s in camera.params_info.split(",")]
    fx, fy = float(camera.focal_length_x), float(camera.focal_length_y)
    cx, cy = float(camera.principal_point_x), float(camera.principal_point_y)
    reproj = float(rec.compute_mean_reprojection_error())
    cameras = {
        "width": int(camera.width),
        "height": int(camera.height),
        "model": camera.model.name,
        "params": [float(v) for v in camera.params],
        "param_names": param_names,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "frames": frame_numbers,
        "registered": registered,
        "poses": {k: v.tolist() for k, v in poses.items()},
        "convention": CONVENTION,
        "scale": "arbitrary",
        "mean_reprojection_error_px": reproj,
        "num_points": len(xyz),
    }
    (raw / "cameras.json").write_text(json.dumps(cameras, indent=1), encoding="utf-8")
    # The database (SIFT descriptors, ~1 MB per frame) and the mapper's own copy of the
    # model are not needed any more: raw/sparse has the result. Masks stay for checking.
    for scratch in ("images", "sparse", "database.db"):
        path = work / scratch
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    run.finish(
        frame_numbers,
        kind="camera_solve",
        cameras="cameras.json",
        points="points.npz",
        sparse="sparse",
        convention=CONVENTION,
        units="arbitrary (COLMAP scale)",
        registered=len(registered),
        num_frames=len(frame_numbers),  # (the standard `frames` is [first, last], like ViPE; every frame number is in cameras.json)
        solved_frames=len(used),
        step=step,
        mapper=params["mapper"],
        matcher=params["matcher"],
        camera_model=camera.model.name,
        focal_px=0.5 * (fx + fy),
        focal_source="user" if focal_px else "colmap",
        fov_long_side_deg=fov_long,
        mean_reprojection_error_px=reproj,
        num_points=len(xyz),
        mean_track_length=float(rec.compute_mean_track_length()),
        models=len(ranked),
        masked_frames=len(masked),
        frame_pairs=pairs,
        parallax={"baseline_over_depth": baseline_ratio, "median_triangulation_angle_deg": tri_angle},
        path=stats,
        colmap=pycolmap.COLMAP_build,
        sift="GPU (SiftGPU, non-commercial)" if sift_on_gpu else "CPU (VLFeat, BSD)",
        step_seconds={k: round(v, 1) for k, v in step_seconds.items()},
    )


def _median_triangulation_angle(rec, points, centers_by_image: dict) -> float:
    """Median over points of the widest angle between two cameras that see it."""
    angles = []
    for i, p in enumerate(points.values()):
        if i % 7:  # a sample is sufficient
            continue
        cs = np.array([centers_by_image[e.image_id] for e in p.track.elements if e.image_id in centers_by_image])
        if len(cs) < 2:
            continue
        rays = cs - p.xyz
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        # angle between the first observer and the one furthest from it (cheap proxy for the widest pair)
        cos = np.clip(rays @ rays[0], -1, 1)
        angles.append(np.degrees(np.arccos(cos.min())))
    return float(np.median(angles)) if angles else 0.0


if __name__ == "__main__":
    serve(main)
