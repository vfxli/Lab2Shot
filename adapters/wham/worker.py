"""WHAM worker: world-grounded SMPL motion. Runs inside third_party/wham/.venv with
the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>        (node "wham.solve")

Upstream demo.py step by step, on the job's PNG frames instead of a video:
upstream's own DetectionModel (YOLOv8x person boxes + ViTPose-H keypoints + pose
tracking, lib/models/preproc/detector.py, demo.py:65 detector.track) -> HMR2.0a features
(with flipped copies: FLIP_EVAL) -> camera rotation (DPVO on half-size frames,
identity for a locked-off camera, or the input camera) -> WHAM (motion encoder,
trajectory decoder, foot-contact trajectory refinement), with optional Temporal
SMPLify. Each person comes out in a world of their own: the longest-tracked
person's world is the scene's world and the camera is the one that person implies;
the others are moved into it as a whole (lab2shot_worker.world_humans.merge_worlds).

Parameters (job["params"]):
    focal_px       float | null   known focal length in pixels at the input resolution;
                                  null = upstream's guess (image diagonal, CLIFF convention)
    static_camera  bool  (false)  locked-off camera: no DPVO (zero camera rotation)
    follow_camera  bool  (false)  true = place every body through the camera frame by frame (lines up
                                  with the plate exactly, depth can jitter, feet may slide); false = keep
                                  WHAM's world motion (feet planted): extra people and an input camera
                                  are aligned to it as a whole
    (people upstream tracks for fewer than 30 frames are dropped, as upstream does)
    smplify        bool  (false)  upstream --run_smplify: Temporal SMPLify refinement against the
                                  2D keypoints (better image alignment, slower)

Every person gets one body shape for the shot (median of WHAM's per-frame betas;
translations corrected so the pelvis stays where it was).
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from lab2shot_worker import nothing, offload, read_frame, reason, require_weights, resident, say, serve
from lab2shot_shared import motion as mo
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run
from lab2shot_worker.recon import load_camera

NODE = "wham.solve"
SMPL_LINK = "dataset/body_models/smpl/SMPL_NEUTRAL.pkl"  # relative to weights/ (the working directory)
YOLO_CKPT = "checkpoints/yolov8x.pt"  # upstream detector.py:39 reads it at ROOT_DIR/this
MIN_FRAMES = 30  # upstream MINIMUM_FRMAES: shorter tracks are dropped
AUX_TAR = "_downloads/body_models.tar.gz"  # fetch_demo_data.sh's SMPL-derived regressors (under weights/)
# what the demo path reads from it (smplx2smpl.pkl, 578 MB, is for evaluation only)
AUX_FILES = ("J_regressor_wham.npy", "J_regressor_feet.npy", "J_regressor_h36m.npy", "J_regressor_coco.npy",
             "smpl_mean_params.npz", "coco_aug_dict.pth")


class FrameList(list):
    """Image paths for FeatureExtractor.run: its video-or-images test calls
    os.path.isfile(video), which raises TypeError for a plain list."""

    def __fspath__(self) -> str:
        return "/nonexistent/lab2shot-frame-list"


# --------------------------------------------------------------------------- setup


def enter_runtime(weights: Path, repo: Path) -> None:
    """Upstream reads dataset/body_models/... relative to the working directory and
    checkpoints/... relative to the repo: point both at weights/."""
    os.chdir(weights)
    import lib.models.preproc.detector as detector
    import lib.models.preproc.extractor as extractor

    detector.ROOT_DIR = str(weights)  # checkpoints/; ViTPose configs stay in repo/third-party/ViTPose
    extractor.ROOT_DIR = str(weights)


def unpack_aux(weights: Path) -> None:
    """body_models.tar.gz -> weights/dataset/body_models/ (once)."""
    import tarfile

    target = weights / "dataset" / "body_models"
    if all((target / name).is_file() for name in AUX_FILES):
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(weights / AUX_TAR) as tar:
        for name in AUX_FILES:
            with tar.extractfile(tar.getmember(f"body_models/{name}")) as src:
                part = target / (name + ".part")
                part.write_bytes(src.read())
                part.replace(target / name)


def load_config(repo: Path, weights: Path):
    from configs.config import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(repo / "configs" / "yamls" / "demo.yaml"))
    cfg.MODEL_CONFIG = str(repo / "configs" / "yamls" / "model_base.yaml")
    cfg.TRAIN.CHECKPOINT = str(weights / "checkpoints" / "wham_vit_bedlam_w_3dpw.pth.tar")
    return cfg


# --------------------------------------------------------------------------- 2D evidence


@torch.no_grad()
def detect_and_track(run: Run, weights: Path, fps: float):
    """Who is in the plate and where, exactly as upstream's demo builds it (demo.py:59-72):
    DetectionModel.track frame by frame (its own YOLOv8x person boxes, ViTPose-H keypoints on
    them and its own pose tracking, lib/models/preproc/detector.py:79-120), then
    DetectionModel.process(fps), which drops tracks shorter than MINIMUM_FRMAES and median-filters
    the boxes. Upstream reads the frames from a VideoCapture; here the job's PNG files are read.

    The node has no 人物框 input (upstream demo.py takes --video and detects the boxes internally).
    To solve a single person, remove the others from the plate upstream of this node
    (「ViTDet 人物框」 -> 「选人」 -> 「人物框转遮罩」 -> 「图像合成」).
    """
    detector = run.model("YOLOv8 和 ViTPose-H 模型", load_detector, weights)
    run.stage("YOLOv8 找人 · ViTPose 关键点")
    detector.initialize_tracking()  # a new shot
    frames = run.job.frames
    for i, (_, path) in run.each(frames, "检测和关键点"):
        detector.track(read_frame(path, order="bgr"), fps, len(frames))
    if not detector.tracking_results["id"]:
        nothing("N-WHAM-NOPEOPLE")
    tracking = detector.process(fps)
    detector.initialize_tracking()  # the model stays loaded: without this shot's detections
    offload(detector)  # off the GPU for the rest of the job, as upstream frees it
    return tracking


@torch.no_grad()
def features(run: Run, weights: Path, tracking):
    extractor = run.model("HMR2 模型", load_extractor, weights)
    run.stage("HMR2 图像特征")
    tracking = extractor.run(FrameList(str(p) for _, p in run.job.frames), tracking)
    offload(extractor)  # off the GPU for the rest of the job, as upstream frees it
    return tracking


# --------------------------------------------------------------------------- camera


@torch.no_grad()
def run_dpvo(run: Run, repo: Path, weights: Path, focal: float, width: int, height: int) -> np.ndarray:
    """Upstream SLAMModel (DPVO, video_stream: half-size BGR frames cropped to a multiple
    of 16) on the job's frames. Returns (N,7): camera-to-world translation + quaternion xyzw."""
    from dpvo.config import cfg as dpvo_cfg
    from dpvo.dpvo import DPVO

    run.stage("DPVO 估计相机转动")
    dpvo_cfg.merge_from_file(str(repo / "third-party" / "DPVO" / "config" / "default.yaml"))
    dpvo_cfg.BUFFER_SIZE = 2048
    intrinsics = torch.tensor([focal * 0.5, focal * 0.5, width / 2 * 0.5, height / 2 * 0.5]).cuda()
    slam = None
    for t, (_, path) in run.each(run.job.frames, "DPVO"):
        image = cv2.resize(read_frame(path, order="bgr"), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = torch.from_numpy(image[:h - h % 16, :w - w % 16]).permute(2, 0, 1).cuda()
        if slam is None:
            slam = DPVO(dpvo_cfg, str(weights / "checkpoints" / "dpvo.pth"), ht=image.shape[1], wd=image.shape[2], viz=False)
        slam(t, image, intrinsics)
    for _ in range(12):
        slam.update()
    poses = slam.terminate()[0]
    del slam
    torch.cuda.empty_cache()
    return np.asarray(poses, np.float64)


def slam_from_rotations(c2w: np.ndarray) -> np.ndarray:
    """(N,7) in DPVO's output layout from cam_to_world poses."""
    from scipy.spatial.transform import Rotation

    out = np.zeros((len(c2w), 7))
    out[:, :3] = c2w[:, :3, 3]
    out[:, 3:] = Rotation.from_matrix(c2w[:, :3, :3]).as_quat()  # x, y, z, w
    return out


# --------------------------------------------------------------------------- network


def run_wham(cfg, network, smpl, dataset, subject: int, tracking, width, height, smplify: bool):
    """One person, exactly as upstream demo.run() (FLIP_EVAL, optional Temporal SMPLify)."""
    from lib.models.smplify import TemporalSMPLify
    from lib.utils.imutils import avg_preds

    with torch.no_grad():
        flipped_batch = dataset.load_data(subject, True)
        _id, x, inits, feats, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
        flipped_pred = network(x, inits, feats, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        batch = dataset.load_data(subject)
        _id, x, inits, feats, mask, init_root, cam_angvel, frame_id, kwargs = batch
        pred = network(x, inits, feats, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
        flipped_pose, flipped_shape = flipped_pred["pose"].squeeze(0), flipped_pred["betas"].squeeze(0)
        pose, shape = pred["pose"].squeeze(0), pred["betas"].squeeze(0)
        flipped_pose, pose = flipped_pose.reshape(-1, 24, 6), pose.reshape(-1, 24, 6)
        avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
        avg_pose = avg_pose.reshape(-1, 144)
        avg_contact = (flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2
        network.pred_pose = avg_pose.view_as(network.pred_pose)
        network.pred_shape = avg_shape.view_as(network.pred_shape)
        network.pred_contact = avg_contact.view_as(network.pred_contact)
        output = network.forward_smpl(**kwargs)
        pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
    if smplify:
        fitter = TemporalSMPLify(smpl, img_w=width, img_h=height, device=cfg.DEVICE)
        pred = fitter.fit(pred, tracking[_id]["keypoints"], **kwargs)
        with torch.no_grad():
            network.pred_pose = pred["pose"]
            network.pred_shape = pred["betas"]
            network.pred_cam = pred["cam"]
            output = network.forward_smpl(**kwargs)
            pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
    return _id, pred, np.asarray(frame_id)


def to_person(pid, pred, frame_ids, frame_numbers, body, j_wham, track=None) -> wh.Person:
    """WHAM's outputs place the hip centre (J_regressor_wham joints 11, 12); convert to
    standard SMPL parameters (transl) and evaluate the mesh. `track`: this person's entry of the tracking results,
    whose ViTPose keypoints (full-frame pixels + score) are kept as the person's own 2D keypoints."""
    r_root_w = pred["poses_root_world"].reshape(-1, 3, 3).cpu().numpy().astype(np.float64)
    r_root_c = pred["poses_root_cam"].reshape(-1, 3, 3).cpu().numpy().astype(np.float64)
    r_body = pred["poses_body"].reshape(-1, 23, 3, 3).cpu().numpy().astype(np.float64)
    betas = pred["betas"].reshape(-1, 10).cpu().numpy().astype(np.float64)
    trans_w = pred["trans_world"].reshape(-1, 3).cpu().numpy().astype(np.float64)
    trans_c = pred["trans_cam"].reshape(-1, 3).cpu().numpy().astype(np.float64)
    go_w, go_c = mo.matrix_to_rotvec(r_root_w), mo.matrix_to_rotvec(r_root_c)
    body_pose = mo.matrix_to_rotvec(r_body).reshape(-1, 69)

    def hip_centre(go):  # WHAM's anchor point of the un-translated body
        v, _ = body(go, body_pose, betas, np.zeros((len(go), 3)))
        return np.einsum("jv,fvc->fjc", j_wham[[11, 12]], v).mean(1)

    tr_w = trans_w - hip_centre(go_w)
    tr_c = trans_c - hip_centre(go_c)
    j0_frames = body.root_rest(betas)
    cam_to_world = wh.camera_from_body(r_root_c, j0_frames + tr_c, r_root_w, j0_frames + tr_w)
    betas_one, shift = wh.lock_shape(betas, body.root_rest)
    tr_w, tr_c = tr_w + shift, tr_c + shift
    b = np.repeat(betas_one[None], len(go_w), 0)
    verts_w, joints_w = body(go_w, body_pose, b, tr_w)
    verts_c, joints_c = body(go_c, body_pose, b, tr_c)
    kp2d = None
    if track is not None:  # the 17 ViTPose keypoints at full resolution, aligned to this person's solved frames, exported as their 2D keypoints
        at = {int(f): i for i, f in enumerate(track["frame_id"])}
        rows = [at[int(f)] for f in frame_ids if int(f) in at]
        if len(rows) == len(frame_ids):
            kp2d = np.asarray(track["keypoints"], np.float32)[rows]
    return wh.Person(
        pid=int(pid), frames=np.asarray(frame_numbers[frame_ids]), global_orient=go_w, body_pose=body_pose,
        betas=betas_one, transl=tr_w, vertices=verts_w, joints=joints_w, j0=body.root_rest(betas_one[None])[0],
        global_orient_cam=go_c, transl_cam=tr_c, vertices_cam=verts_c, joints_cam=joints_c, cam_to_world=cam_to_world,
        keypoints_2d=kp2d, keypoint_names=wh.COCO17_KEYPOINT_NAMES if kp2d is not None else (),
    )


class SmplBody:
    """Plain SMPL (smplx package) for the standard parameterisation: vertices, 24 joints, rest root joint."""

    def __init__(self, model_file: Path):
        import smplx

        self.model = smplx.SMPL(model_path=str(model_file), gender="neutral", num_betas=10).cuda().eval()

    def __call__(self, go, body_pose, betas, transl, chunk: int = 512):
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device="cuda")  # noqa: E731
        verts, joints = [], []
        with torch.no_grad():
            for s in range(0, len(go), chunk):
                e = slice(s, s + chunk)
                out = self.model(global_orient=t(go[e]), body_pose=t(body_pose[e]), betas=t(betas[e]), transl=t(transl[e]))
                verts.append(out.vertices.cpu().numpy())
                joints.append(out.joints[:, :24].cpu().numpy())
        return np.concatenate(verts).astype(np.float32), np.concatenate(joints).astype(np.float32)

    def root_rest(self, betas) -> np.ndarray:
        """Rest-pose root joint [F,3] for betas [F,10]."""
        with torch.no_grad():
            b = torch.as_tensor(np.asarray(betas), dtype=torch.float32, device="cuda")
            v = self.model.v_template + torch.einsum("fb,vcb->fvc", b, self.model.shapedirs[..., :b.shape[1]])
            return torch.einsum("v,fvc->fc", self.model.J_regressor[0], v).cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- models (kept loaded between jobs)


@resident
def load_detector(weights: Path):
    """Upstream's DetectionModel (YOLOv8x + ViTPose-H): each job starts it with initialize_tracking().

    Both checkpoints are checked first (enter_runtime pointed detector.ROOT_DIR at weights/, upstream reads
    them at ROOT_DIR/checkpoints, lib/models/preproc/detector.py:35 and :39). ViTPose says so itself when its
    file is not there; ultralytics does not. YOLO(path) falls through to attempt_download_asset
    (ultralytics/utils/downloads.py:408) and "yolov8x.pt" is in GITHUB_ASSETS_NAMES, so :449-450 fetches
    Ultralytics' own release copy, which is not the pinned file. A real job has no network (the SDK's
    _forbid_downloads closes every socket), so the weights cannot actually be replaced, but the download is refused
    three times (safe_download retry=3) and the artist reads 「连不上 github.com」 instead of 「找不到权重」."""
    from lib.models.preproc import detector as upstream

    require_weights("wham", weights / YOLO_CKPT, what="YOLOv8x 人物检测权重")
    return upstream.DetectionModel("cuda")


@resident
def load_extractor(weights: Path):
    from lib.models.preproc.extractor import FeatureExtractor

    return FeatureExtractor("cuda", flip_eval=True)


@resident
def load_wham(repo: Path, weights: Path):
    """(cfg, SMPL layer, WHAM network), as upstream demo.py builds them."""
    from lib.models import build_body_model, build_network

    cfg = load_config(repo, weights)
    smpl = build_body_model(cfg.DEVICE, cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN)
    network = build_network(cfg, smpl)
    network.eval()
    return cfg, smpl, network


@resident
def load_smpl(model_file: Path) -> SmplBody:
    return SmplBody(model_file)


# --------------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "WHAM")
    job, params = run.job, run.params
    follow_camera, smplify, static = params["follow_camera"], params["smplify"], params["static_camera"]

    weights = Path(job.weights_dir).resolve()
    repo = Path(job.repo_dir).resolve()
    raw = job.raw_dir.resolve()
    cam = load_camera(job)
    smpl_file = wh.body_model_file("smpl", "WHAM", weights / SMPL_LINK)  # before any heavy work
    unpack_aux(weights)
    enter_runtime(weights, repo)

    frame_numbers, width, height = wh.plate(run, MIN_FRAMES, reason("I-WORKER-WHYTEMPORAL"))
    n = len(frame_numbers)
    fps = float(job.fps)  # the plate's own frame rate (a frame-carrying packet always has one)

    focal, focal_source = wh.focal_per_frame(frame_numbers, params["focal_px"], cam,
                                             (width**2 + height**2) ** 0.5, "default (image diagonal)")

    # 2D evidence: upstream detects and tracks the people itself, the node has no 人物框 input
    tracking = detect_and_track(run, weights, fps)
    if not tracking:
        nothing("N-WHAM-NOLONGTRACK", frames=MIN_FRAMES)
    # longest track first: the world's reference; person ids are upstream's own track ids
    keep = sorted(tracking, key=lambda k: len(tracking[k]["frame_id"]), reverse=True)
    pid_of = {k: int(k) for k in keep}
    tracking = defaultdict(lambda: defaultdict(list), {k: tracking[k] for k in keep})
    tracking = features(run, weights, tracking)

    # camera rotation
    if cam is not None:
        c2w_in = cam.at(frame_numbers)[1]
        slam_results, rotation_source = slam_from_rotations(c2w_in), "camera"
    elif static:
        slam_results = np.zeros((n, 7))
        slam_results[:, 6] = 1.0
        rotation_source = "static"
    else:
        slam_results, rotation_source = run_dpvo(run, repo, weights, float(np.median(focal)), width, height), "DPVO"
    from scipy.spatial.transform import Rotation

    rot_hint = Rotation.from_quat(slam_results[:, 3:]).as_matrix()  # cam_to_world rotations, any world

    # WHAM
    from lib.data.datasets import CustomDataset

    cfg, smpl, network = run.model("WHAM 模型", load_wham, repo, weights)
    body = run.model("SMPL 身体模型", load_smpl, smpl_file)
    run.stage("WHAM 解算人体动作")
    dataset = CustomDataset(cfg, tracking, slam_results, width, height, fps)
    K = torch.eye(3)[None].float()
    K[:, 0, 0] = K[:, 1, 1] = float(np.median(focal))
    K[:, 0, 2], K[:, 1, 2] = width / 2.0, height / 2.0
    dataset.intrinsics = K  # the shot's focal instead of upstream's fixed diagonal guess
    if cam is not None and np.ptp(focal) > 0.01 * np.median(focal):
        say("W-WHAM-FOCALCHANGES", low=float(np.min(focal)), high=float(np.max(focal)), focal=float(np.median(focal)))

    j_wham = smpl.J_regressor_wham.cpu().numpy().astype(np.float64)
    people = []
    for s, _ in run.each(range(len(dataset)), "解算"):
        key, pred, frame_ids = run_wham(cfg, network, smpl, dataset, s, tracking, width, height, smplify)
        people.append(to_person(pid_of[key], pred, frame_ids, frame_numbers, body, j_wham, tracking.get(key)))
    people.sort(key=lambda person: keep.index(next(k for k, v in pid_of.items() if v == person.pid)))

    # the camera track is used to merge the people's worlds and align them to an input camera
    run.stage("统一世界坐标")
    # 「相机旋转」 is not a camera: `cam` holds only per-frame rotations here (`rotation_only` in `camera_in.npz`)
    # with zero translation, since upstream uses only the rotation
    # (`third_party/wham/repo/lib/data/datasets/dataset_custom.py:19 quat = traj[:, 3:]`). The world merge therefore
    # runs as if no input camera were given: no alignment as a whole, and the world stays the method's own gravity
    # world. Placing the people into a given camera requires an explicit 「相机空间转换」 node in the graph.
    as_camera = None if cam is None or cam.rotation_only else cam
    track, world_info = wh.one_world("WHAM", people, frame_numbers, rot_hint, static, as_camera,
                                     follow_camera)

    run.stage("写出结果")
    out = [wh.save_person(raw, person, "smpl", body.model) for person in people]
    # raw/camera.npz: the reference camera of this result (the node's 「参照相机」 output,
    # families/humans.py `reference_camera`). It is not a production camera; its only use is as the reference for
    # the core node 「相机空间转换」, which computes the correction that moves the result into the user's camera world.
    #
    # This camera comes entirely from upstream outputs, not from the body:
    #   (1) upstream runs SLAM to solve the camera: `third_party/wham/repo/demo.py:56`
    #      `slam = SLAMModel(video, output_pth, width, height, calib)`, `:76` `slam_results = slam.process()`,
    #      `:88` `joblib.dump(slam_results, …'slam_results.pth')`; its rotation is fed to the network through
    #      `third_party/wham/repo/lib/data/datasets/dataset_custom.py:15 convert_dpvo_to_cam_angvel`.
    #      `rot_hint` is that rotation, and the rotation of `track` comes from it
    #      (world_humans.camera_track -> fill_track).
    #   (2) the position: upstream outputs the same person both in camera space and in world space
    #      (`third_party/wham/repo/demo.py:162-167`: `results[_id]['pose'] / ['trans']` (camera space)
    #      and `['pose_world'] / ['trans_world']` (world)). The rigid transform between them is that camera,
    #      uniquely determined by these two official outputs (wh.camera_from_body).
    wh.save_camera(raw, frame_numbers, focal, track)
    wh.write_humans(
        run,
        frame_numbers,
        out,
        world_info["world"],
        method="WHAM",
        body_model="SMPL neutral (SMPL_NEUTRAL.pkl), 10 betas; body_pose 23x3 axis-angle",
        # provenance of the lens and camera rotation used by the solve (metadata, not an output port)
        camera={"focal_source": focal_source, "rotation_source": rotation_source,
                "width": width, "height": height, "static": static and cam is None},
        up_axis="+Y" if as_camera is None else "input camera's world up",
        joints="24 SMPL joints, parents from the model",
        merge=world_info["merge"],
        alignment=world_info["alignment"],
        params={"focal_px": params["focal_px"], "static_camera": static, "follow_camera": follow_camera,
                "smplify": smplify},
        fps=fps,
    )


if __name__ == "__main__":
    serve(main)
