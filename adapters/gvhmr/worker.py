"""GVHMR worker: world-grounded SMPL-X motion. Runs inside third_party/gvhmr/.venv-ada-blackwell
with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>        (node "gvhmr.solve")

Upstream demo (tools/demo/demo.py) step by step, on the job's PNG frames instead
of a video: upstream's own people tracker (YOLOv8x, hmr4d/utils/preproc/tracker.py,
demo.py:109 Tracker) finds the boxes -> box interpolation + smoothing
-> ViTPose-H 2D keypoints + HMR2.0a image features on the 256px crops ->
camera rotation (SimpleVO, identity for a locked-off camera, or the input camera)
-> GVHMR network + post-processing (static joints, IK). Every person is solved
on their own (as upstream does for its one person). The first (largest) person's
world is the scene's world and the camera is the one that person implies; the
others are moved into it as a whole (lab2shot_worker.world_humans.merge_worlds).

Parameters (job["params"]):
    focal_px       float | null   known focal length in pixels at the input resolution;
                                  null = upstream's guess (image diagonal, ~53 deg diagonal FOV)
    static_camera  bool  (false)  locked-off camera: no visual odometry, static-camera post-processing
    follow_camera  bool  (false)  true = place every body through the camera frame by frame (lines up
                                  with the plate exactly, depth can jitter, feet may slide); false = keep
                                  GVHMR's world motion (feet planted): extra people and an input camera
                                  are aligned to it as a whole
    vo_step        int 1-30 (8)   SimpleVO: solve the rotation every n-th frame, interpolate between

Every person gets one body shape for the shot (median of GVHMR's per-frame betas;
translations corrected so the pelvis stays where it was).
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch

from lab2shot_worker import (
    nothing,
    MemoryBound, limit_gpu_memory, link_file, offload, read_frame, reason, require_weights, resident, say, serve, stage,
)
from lab2shot_shared import motion as mo
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run
from lab2shot_worker.recon import load_camera

NODE = "gvhmr.solve"
CKPT = "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
YOLO_CKPT = "inputs/checkpoints/yolo/yolov8x.pt"  # upstream tracker.py:22 reads it at PROJ_ROOT/this
SMPLX_LINK = "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
MIN_FRAMES = 16  # shorter tracks are dropped (too short for the temporal network)
SCALE = 0.5  # upstream reads the video at half size for crops and visual odometry


# --------------------------------------------------------------------------- setup


def enter_runtime(weights: Path) -> None:
    """Upstream reads everything below inputs/checkpoints relative to its project root:
    make weights/ that root (working directory and hmr4d.PROJ_ROOT). The repo's own data
    files are read from <root>/hmr4d/...: weights/hmr4d links to the repo's package."""
    link_file(weights / "hmr4d", Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]) / "hmr4d")
    os.chdir(weights)
    import hmr4d

    hmr4d.PROJ_ROOT = weights


def half_frame(path: Path) -> np.ndarray:
    """RGB at half size, as upstream reads the video (ffmpeg scale: bicubic)."""
    rgb = read_frame(path)
    return cv2.resize(rgb, (round(rgb.shape[1] * SCALE), round(rgb.shape[0] * SCALE)), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------- people


@resident
def load_tracker(weights: Path):
    """Upstream's own people tracker: YOLOv8x behind hmr4d/utils/preproc/tracker.py Tracker,
    the checkpoint at <root>/inputs/checkpoints/yolo/yolov8x.pt (enter_runtime made weights/ the root).

    The file is checked first because ultralytics does not fail on a path that is not there: YOLO(path)
    falls through to attempt_download_asset (ultralytics/utils/downloads.py:408), and "yolov8x.pt" is in
    GITHUB_ASSETS_NAMES, so :449-450 fetches Ultralytics' own release copy instead, a different file from
    the pinned one (136,890,692 bytes from github vs the 136,867,539-byte copy in the pinned mirror the
    GVHMR authors published). In a real job that download is refused anyway (the SDK's _forbid_downloads
    closes every socket), but it is refused three times (safe_download retry=3) and the artist is then told
    「连不上 github.com」 rather than 「找不到权重」. Checking the file first reports the actual problem."""
    from hmr4d.utils.preproc import Tracker

    require_weights("gvhmr", weights / YOLO_CKPT, what="YOLOv8x 人物检测权重")
    return Tracker()


def track_people(run: Run, weights: Path, width: int, height: int) -> dict[int, dict[int, np.ndarray]]:
    """Who is in the plate, as upstream finds them (demo.py:108-112 Tracker): its own YOLOv8 tracking
    (Tracker.track) and its own sorting by how much of the frame each track covers
    (Tracker.sort_track_length), on the job's frame files instead of a video. Both call
    get_video_lwh(video_path) for the length and the frame size, which are taken from the frames directly.

    {person id: {frame index in the job: [x1, y1, x2, y2]}}, most prominent person first. Upstream's
    demo keeps only its top-1 track (get_one_track); here every track long enough for the temporal
    network is kept, so several people in one plate come out as several people.

    This is the path without a 人物框 input; with one wired (the official functions take bbx_xys),
    given_people() below is used instead and this tracker never runs.
    """
    from hmr4d.utils.preproc import tracker as tracker_module

    tracker = run.model("YOLOv8 人物跟踪模型", load_tracker, weights)
    run.stage("YOLOv8 找人")
    frames = [str(path) for _, path in run.job.frames]
    video_lwh = tracker_module.get_video_lwh
    tracker_module.get_video_lwh = lambda *_args, **_kwargs: (len(frames), width, height)
    try:
        history = tracker.track(frames)
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = tracker.sort_track_length(history, frames)
    finally:
        tracker_module.get_video_lwh = video_lwh  # the process stays for the next job
    offload(tracker)  # off the GPU for the rest of the job, as upstream deletes it (demo.py:113)

    people = {}
    for track_id in id_sorted:
        at = {int(i): np.asarray(b, np.float64) for i, b in zip(id_to_frame_ids[track_id], id_to_bbx_xyxys[track_id])}
        if len(at) >= MIN_FRAMES:
            people[int(track_id)] = at
        else:
            say("N-HUMANS-SHORTPERSON", person=int(track_id), frames=len(at), min_frames=MIN_FRAMES)
    return people


def given_people(job) -> dict[int, dict[int, np.ndarray]]:
    """The wired 人物框 input (「ViTDet 人物框」 -> 「选人」) as {person id: {frame index: box}}, the same shape
    track_people returns (upstream VitPoseExtractor.extract / DemoPL.predict take these boxes, nodes.py official
    note ①). Tracks shorter than MIN_FRAMES are skipped."""
    index = {int(f): i for i, (f, _) in enumerate(job.frames)}
    people = {}
    for pid, boxes in job.people().items():
        at = {index[f]: np.asarray(b[:4], np.float64) for f, b in boxes.items() if f in index}
        if len(at) >= MIN_FRAMES:
            people[int(pid)] = at
        else:
            say("N-HUMANS-SHORTPERSON", person=int(pid), frames=len(at), min_frames=MIN_FRAMES)
    return people


def box_track(boxes: dict[int, np.ndarray]) -> tuple[int, torch.Tensor]:
    """Upstream get_one_track, on the person's own frame range: gaps linearly
    interpolated, then twice a 5-frame moving average. Returns (first index, xyxy [L,4])."""
    from hmr4d.utils.net_utils import moving_average_smooth
    from hmr4d.utils.seq_utils import get_frame_id_list_from_mask, linear_interpolate_frame_ids

    first, last = min(boxes), max(boxes)
    mask = torch.zeros(last - first + 1, dtype=torch.bool)
    xyxy = torch.zeros(last - first + 1, 4)
    for i, b in boxes.items():
        mask[i - first] = True
        xyxy[i - first] = torch.from_numpy(b[:4]).float()
    if not mask.all():
        xyxy = linear_interpolate_frame_ids(xyxy, get_frame_id_list_from_mask(~mask))
    xyxy = moving_average_smooth(xyxy, window_size=5, dim=0)
    xyxy = moving_average_smooth(xyxy, window_size=5, dim=0)
    return first, xyxy


# --------------------------------------------------------------------------- camera


def simple_vo(half_frames: np.ndarray, focal_px: float, width: int, height: int, step: int) -> np.ndarray:
    """Upstream SimpleVO (SIFT + pycolmap two-view geometry) on the half-size frames. R_w2c [N,3,3]."""
    from hmr4d.utils.preproc.relpose import simple_vo as vo_module

    read_video = vo_module.read_video_np
    vo_module.read_video_np = lambda *args, **kwargs: half_frames  # frames instead of a video file
    diag = (width**2 + height**2) ** 0.5
    f_mm = focal_px / diag * (24**2 + 36**2) ** 0.5  # SimpleVO takes a full-frame equivalent focal length
    stage("SimpleVO 估计相机转动")
    try:
        t_w2c = vo_module.SimpleVO(None, scale=SCALE, step=step, method="sift", f_mm=f_mm).compute()
    finally:
        vo_module.read_video_np = read_video  # the process stays for the next job: do not keep this shot's frames
    return np.asarray(t_w2c, np.float64)[:, :3, :3]


# --------------------------------------------------------------------------- network


@resident
def load_extractors(weights: Path):
    """ViTPose-H and HMR2 (their checkpoints below weights/, the working directory)."""
    from hmr4d.utils.preproc import Extractor, VitPoseExtractor

    return VitPoseExtractor(tqdm_leave=False), Extractor(tqdm_leave=False)


@resident
def load_body_model(weights: Path):
    """Upstream's SMPL-X "supermotion" layer (files below weights/)."""
    from hmr4d.utils.smplx_utils import make_smplx

    return make_smplx("supermotion").cuda().eval()


@resident
def load_gvhmr(weights: Path, static: bool):
    import hydra
    from hydra import compose, initialize_config_module

    import importlib

    importlib.import_module("hmr4d.model.gvhmr.gvhmr_pl_demo")  # registers model/gvhmr/gvhmr_pl_demo (demo.py imports it)
    from hmr4d.configs import register_store_gvhmr

    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=["video_name=lab2shot", f"static_cam={static}", "use_dpvo=False"])
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    model.load_pretrained_model(str(weights / CKPT))
    return model.eval().cuda()


def crops_features(extractors, half_frames: np.ndarray, first: int, bbx_xys: torch.Tensor):
    """ViTPose keypoints [L,17,3] (full-resolution pixels) and HMR2 features [L,1024] for one person."""
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch

    vitpose, hmr2 = extractors
    frames = half_frames[first:first + len(bbx_xys)]
    imgs, _ = get_batch(frames, bbx_xys * SCALE, img_ds=1.0, path_type="np")  # = upstream's img_ds=0.5 crops
    kp2d = vitpose.extract(imgs, bbx_xys)
    feats = hmr2.extract_video_features(imgs, bbx_xys)
    return kp2d, feats


def solve_person(model, bm, pid, first, bbx_xys, kp2d, feats, K, r_w2c, static, frame_numbers) -> wh.Person:
    from hmr4d.utils.geo_transform import compute_cam_angvel

    length = len(bbx_xys)
    sl = slice(first, first + length)
    data = {
        "length": torch.tensor(length),
        "bbx_xys": bbx_xys.float(),
        "kp2d": kp2d.float(),
        "K_fullimg": torch.from_numpy(K[sl]).float(),
        "cam_angvel": compute_cam_angvel(torch.from_numpy(r_w2c[sl]).float()),
        "f_imgseq": feats.float(),
    }
    with torch.no_grad():
        pred = model.predict(data, static_cam=static)
    glob = {k: v.detach().float().cpu() for k, v in pred["smpl_params_global"].items()}
    cam = {k: v.detach().float().cpu() for k, v in pred["smpl_params_incam"].items()}

    betas = glob["betas"].numpy()  # [L,10] per frame
    root_rest = lambda b: bm.get_skeleton(torch.as_tensor(b, dtype=torch.float32, device="cuda"))[..., 0, :].cpu().numpy()  # noqa: E731
    j0_frames = root_rest(betas)
    go_w, tr_w = glob["global_orient"].numpy().astype(np.float64), glob["transl"].numpy().astype(np.float64)
    go_c, tr_c = cam["global_orient"].numpy().astype(np.float64), cam["transl"].numpy().astype(np.float64)
    # camera implied by the same body in the camera and in the world
    cam_to_world = wh.camera_from_body(mo.rotvec_to_matrix(go_c), j0_frames + tr_c, mo.rotvec_to_matrix(go_w), j0_frames + tr_w)

    betas_one, shift = wh.lock_shape(betas, root_rest)  # one body shape for the shot
    tr_w, tr_c = tr_w + shift, tr_c + shift
    j0 = root_rest(betas_one[None])[0]

    body_pose = glob["body_pose"].numpy()
    eval_betas = np.repeat(betas_one[None], length, 0)
    # SMPL-X (upstream's "supermotion" layer) -> vertices [F,10475,3], joints [F,55,3]
    verts_w, joints_w = wh.evaluate(bm, 55, global_orient=go_w, body_pose=body_pose, betas=eval_betas, transl=tr_w)
    verts_c, joints_c = wh.evaluate(bm, 55, global_orient=go_c, body_pose=body_pose, betas=eval_betas, transl=tr_c)
    hands = bm.bm  # SMPL-X layer: PCA hands at zero = the relaxed mean hand pose (flat_hand_mean False)
    left = np.repeat(hands.left_hand_mean.detach().cpu().numpy()[None], length, 0)
    right = np.repeat(hands.right_hand_mean.detach().cpu().numpy()[None], length, 0)
    return wh.Person(
        pid=pid, frames=np.asarray(frame_numbers[sl]), global_orient=go_w, body_pose=body_pose, betas=betas_one,
        transl=tr_w, vertices=verts_w, joints=joints_w, j0=j0,
        global_orient_cam=go_c, transl_cam=tr_c, vertices_cam=verts_c, joints_cam=joints_c,
        cam_to_world=cam_to_world, extra={"left_hand_pose": left, "right_hand_pose": right},
        # the 17 ViTPose keypoints at full resolution (crops_features): an upstream intermediate, also exported
        keypoints_2d=kp2d.cpu().numpy().astype(np.float32), keypoint_names=wh.COCO17_KEYPOINT_NAMES,
    )


# --------------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "GVHMR")
    job, params = run.job, run.params
    # an allocation beyond the free memory fails instead of spilling into RAM (WSL/Windows): before the tracker too,
    # not only the models run.model loads (it would set the cap later)
    run.gpu_cap_mb = limit_gpu_memory()
    follow_camera, vo_step, static = params["follow_camera"], params["vo_step"], params["static_camera"]

    weights = Path(job.weights_dir).resolve()
    raw = job.raw_dir.resolve()
    cam = load_camera(job)
    wh.body_model_file("smplx", "GVHMR", weights / SMPLX_LINK)  # before any heavy work
    enter_runtime(weights)

    frame_numbers, width, height = wh.plate(run, MIN_FRAMES, reason("I-WORKER-WHYTEMPORAL"))
    n = len(frame_numbers)

    # with 人物框 wired, use those boxes (the upstream functions take boxes, nodes.py official note ①);
    # otherwise track people as the upstream demo does (demo.py:108-112)
    people_boxes = list((given_people(job) if job.inputs.get("boxes") else track_people(run, weights, width, height)).items())
    if not people_boxes:
        nothing("N-GVHMR-NOPEOPLE", frames=MIN_FRAMES)

    run.stage("读取画面")
    half = np.stack([half_frame(path) for _, path in job.frames])

    # camera: focal and rotation (world-to-camera) for every frame
    from hmr4d.utils.geo.hmr_cam import estimate_focal_length

    focal, focal_source = wh.focal_per_frame(frame_numbers, params["focal_px"], cam,
                                             estimate_focal_length(width, height), "default (image diagonal)")
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = focal
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = width / 2.0, height / 2.0, 1.0
    if cam is not None:
        r_w2c = np.swapaxes(cam.at(frame_numbers)[1][:, :3, :3], -1, -2)
        rotation_source = "camera"
    elif static:
        r_w2c = np.repeat(np.eye(3)[None], n, 0)
        rotation_source = "static"
    else:
        r_w2c = simple_vo(half, float(np.median(focal)), width, height, vo_step)
        rotation_source = "SimpleVO"

    # 2D evidence per person
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    extractors = run.model("ViTPose-H 和 HMR2 模型", load_extractors, weights)
    run.stage("ViTPose 关键点 · HMR2 图像特征")

    def gather_evidence(extractors, half, _):
        evidence = []
        for k, (pid, boxes) in run.each(people_boxes, "关键点和特征"):
            first, xyxy = box_track(boxes)
            bbx_xys = get_bbx_xys_from_xyxy(xyxy, base_enlarge=1.2).float()
            kp2d, feats = crops_features(extractors, half, first, bbx_xys)
            evidence.append((pid, first, bbx_xys, kp2d, feats))
        return evidence

    # no parameter splits the shot: the input itself bounds the memory
    evidence = run.fit(MemoryBound.shot(n, len(people_boxes)), partial(gather_evidence, extractors, half))
    offload(extractors)  # off the GPU for the rest of the job, as upstream frees them
    del extractors, half

    # GVHMR
    model = run.model("GVHMR 模型", load_gvhmr, weights, static)
    bm = run.model("SMPL-X 身体模型", load_body_model, weights)
    run.stage("GVHMR 解算人体动作")

    def solve_people(_):
        people = []
        for k, (pid, first, bbx_xys, kp2d, feats) in run.each(evidence, "解算"):
            people.append(solve_person(model, bm, pid, first, bbx_xys, kp2d, feats, K, r_w2c, static, frame_numbers))
        return people

    people = run.fit(MemoryBound.shot(n, len(evidence)), solve_people)

    # one world: the camera track is only used internally to merge worlds and align to an input camera
    run.stage("统一世界坐标")
    # 「相机旋转」 is not a camera: `cam` holds only per-frame rotations here (`rotation_only` in `camera_in.npz`)
    # with zero translation, since upstream uses only the rotation
    # (`third_party/gvhmr/repo/tools/demo/demo.py:197 compute_cam_angvel(R_w2c)`). The world merge therefore runs
    # as if no input camera were given: no alignment as a whole, and the world stays the method's own gravity world.
    # Placing the people into a given camera requires an explicit 「相机空间转换」 node in the graph.
    as_camera = None if cam is None or cam.rotation_only else cam
    track, world_info = wh.one_world("GVHMR", people, frame_numbers, np.swapaxes(r_w2c, -1, -2), static, as_camera,
                                     follow_camera)

    run.stage("写出结果")
    out = [wh.save_person(raw, person, "smplx", bm.bm) for person in people]
    # raw/camera.npz: the reference camera of this result (the node's 「参照相机」 output,
    # families/humans.py `reference_camera`). It is not a production camera; its only use is as the reference for the
    # core node 「相机空间转换」, which computes the correction that moves the result into the world of the user's camera.
    # It comes from upstream's own results, not from the body:
    #   (1) rotation: upstream solves the camera rotation itself (SimpleVO by default, DPVO with `--use_dpvo`):
    #      `third_party/gvhmr/repo/tools/demo/demo.py:149-150`
    #      `simple_vo = SimpleVO(...)` / `vo_results = simple_vo.compute()  # (L, 4, 4), numpy`,
    #      `:186` `R_w2c = torch.from_numpy(traj[:, :3, :3])`, `:197` `compute_cam_angvel(R_w2c)`.
    #      Here this is `r_w2c` (simple_vo); the rotation of `track` comes from it.
    #      Upstream does not solve translation (SimpleVO gives relative rotation only; DPVO's translation is unused),
    #      so the position comes from:
    #   (2) upstream outputs the same body both in camera space and in world space:
    #      `third_party/gvhmr/repo/tools/demo/demo.py:215` `pred["smpl_params_incam"]` and
    #      `:259` `pred["smpl_params_global"]` (both saved to `:327` `paths.hmr4d_results`).
    #      The rigid transform between them is that camera, uniquely determined by these two official outputs
    #      (wh.camera_from_body, solve_person).
    wh.save_camera(raw, frame_numbers, focal, track)
    # 放人用的那台相机（节点的「相机」口，家族的 plate_camera）：原点、不动，每帧的焦距就是上面用的那个，主点在画面中心；
    # 人在它的空间里（相机空间那一份），参照相机把世界里的人和它连起来
    wh.save_camera(raw, frame_numbers, focal, name=wh.PLATE_CAMERA)

    wh.write_humans(
        run,
        frame_numbers,
        out,
        world_info["world"],
        method="GVHMR",
        body_model="SMPL-X neutral (SMPLX_NEUTRAL.npz), 10 betas; body_pose 21x3 axis-angle; hands: full 15x3 "
                   "axis-angle per hand (GVHMR predicts none: the relaxed mean hand pose, i.e. flat_hand_mean=False "
                   "with zero PCA); jaw / eyes / expression zero",
        # provenance of the lens and camera rotation used by the solve (metadata, not an output port)
        camera={"focal_source": focal_source, "rotation_source": rotation_source,
                "width": width, "height": height, "static": static and cam is None},
        up_axis="+Y" if as_camera is None else "input camera's world up",
        joints="the 55 SMPL-X joints (lab2shot_shared.smpl)",
        merge=world_info["merge"],
        alignment=world_info["alignment"],
        params={"focal_px": params["focal_px"], "static_camera": static, "follow_camera": follow_camera,
                "vo_step": vo_step},
    )


if __name__ == "__main__":
    serve(main)
