"""TRAM worker: SMPL bodies placed in the world through the camera. Runs inside
third_party/tram/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>        (node "tram.solve")

Upstream scripts step by step, on the job's PNG frames (TRAM reads a folder of
*.jpg: the worker links the PNGs under those names, nothing is re-encoded):

    estimate_camera.py   ViTDet-H + SAM ViT-H + DEVA: people masks and tracks ->
                         focal search by SLAM reprojection error (or the given focal) ->
                         masked DROID-SLAM + ZoeDepth metric scale -> SPEC gravity
                         (pitch / roll) -> camera in a gravity-aligned world
    estimate_humans.py   VIMO per track: SMPL in the camera
    visualize_tram.py    world body = camera x camera-space body, root trajectory
                         smoothed (traj_filter, sigma 3 frames)

There is no camera input: upstream's VIMO only takes a focal length and a principal
point (lib/models/hmr_vimo.py:134), and the camera is what estimate_camera.py solves,
not something it is given. The 人物框 input is optional (the official
HMR_VIMO.inference takes each person's per-frame boxes, hmr_vimo.py:134): with one,
VIMO solves the people in those boxes (tracks_from_given); without one, the people are
found by upstream's own ViTDet + SAM + DEVA, exactly as the official scripts do.
Detection + segmentation run either way: the SLAM needs the people masks.

Parameters (job["params"]):
    focal_px       float | null   known focal length in pixels at the input resolution; null = upstream's
                                  search (500-1500 px by SLAM reprojection error: too short for long lenses)
    static_camera  bool  (false)  locked-off camera: no SLAM (upstream --static_camera; TRAM also detects it)
    max_people     int 1-20 (20)  most people solved (upstream --max_humans): longest DEVA tracks first, or the
                                  first ones of the given boxes (most prominent first, as the file orders them)

Every person gets one body shape for the shot (median betas, pelvis kept in place;
upstream averages the shape the same way for its visualisation). VIMO needs runs of
at least 16 consecutive frames: shorter pieces of a track are dropped (upstream).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import nothing, link_file, local_hub, offload, reason, resident, serve, say, stage, stub_module
from lab2shot_shared import motion as mo
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run

NODE = "tram.solve"
REPO_DATA = ("data/smpl/J_regressor_extra.npy", "data/smpl/J_regressor_h36m.npy", "data/smpl/kintree_table.pkl",
             "data/smpl/smpl_mean_params.npz", "data/pretrain/cascade_mask_rcnn_vitdet_h_75ep.py")
MIN_FRAMES = 16  # VIMO's window


# --------------------------------------------------------------------------- setup


def enter_runtime(weights: Path, repo: Path) -> None:
    """TRAM reads data/... relative to its working directory: run in weights/, where
    data/pretrain holds the checkpoints and data/smpl the repo's SMPL files + SMPL_NEUTRAL.pkl."""
    for rel in REPO_DATA:
        link_file(weights / rel, repo / rel)
    os.chdir(weights)
    # the scripts put these on sys.path relative to the repo root
    for folder in ("thirdparty/Tracking-Anything-with-DEVA", "thirdparty/DROID-SLAM/droid_slam", "thirdparty/DROID-SLAM", ""):
        if str(repo / folder) not in sys.path:
            sys.path.insert(0, str(repo / folder))
    # lib/pipeline/__init__.py also imports the video renderer (pytorch3d, not installed:
    # nothing is rendered here) and VIMO imports lib.pipeline: give it an empty stand-in.
    stub_module("lib.pipeline.visualization", visualize_tram=None)

    # ZoeDepth and its MiDaS backbone from the pinned local copies (never GitHub)
    local_hub({"ZoeDepth": os.environ["TRAM_ZOEDEPTH_DIR"], "MiDaS": os.environ["TRAM_MIDAS_DIR"]}, "TRAM")
    # as estimate_camera.py: lib.camera first (it sets the multiprocessing start method,
    # which fails once anything else has used multiprocessing)
    import importlib

    importlib.import_module("lib.camera")


def link_frames(job, folder: Path) -> list[str]:
    """TRAM globs <folder>/*.jpg; cv2 reads by content, so PNGs linked under .jpg names work unchanged."""
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for i, (_, path) in enumerate(job.frames):
        link = folder / f"{i:05d}.jpg"
        link_file(link, Path(path).resolve())
        files.append(str(link))
    return files


# --------------------------------------------------------------------------- camera


def detect_segment_track(imgfiles: list[str], seq_folder: Path):
    """Upstream estimate_camera.py: ViTDet + SAM + DEVA -> (boxes, masks as RLE, tracks)."""
    from lib.pipeline import detect_segment_track as upstream

    stage("ViTDet 检测 · SAM 遮罩 · DEVA 跟踪人物")
    return upstream(imgfiles, str(seq_folder), thresh=0.25, min_size=100, save_vos=False)


def solve_camera(img_folder: Path, imgfiles, masks_rle, focal_px, static: bool, width: int, height: int):
    """Upstream estimate_camera.py after detection. Returns (cam_to_world [N,4,4] in TRAM's
    gravity-aligned world, focal, is_static, focal_source, spec focal)."""
    from pycocotools import mask as masktool

    from lib.camera import align_cam_to_world, calibrate_intrinsics, run_metric_slam
    from lib.camera.masked_droid_slam import test_slam
    from lib.camera.slam_utils import preprocess_masks

    masks = None if masks_rle is None else torch.from_numpy(np.array([masktool.decode(m) for m in masks_rle]))
    folder = str(img_folder)
    if focal_px:
        cam_int = np.array([focal_px, focal_px, width / 2.0, height / 2.0])
        is_static, focal_source = static, "user"
        if not static:  # upstream's static-camera test (first step of calibrate_intrinsics)
            _, is_static = test_slam(folder, preprocess_masks(folder, masks[::10][:50]), stride=10, calib=cam_int,
                                     max_frame=50)
    else:
        stage("TRAM 搜索 Focal Length（SLAM 重投影误差）")
        cam_int, is_static = calibrate_intrinsics(folder, masks, is_static=static)
        # a locked-off camera keeps upstream's first guess (the image's longest side): nothing to search with
        focal_source = "TRAM default (longest image side)" if is_static else "TRAM focal search (500-1500 px)"
    if is_static and not static:
        say("W-TRAM-STATIC")
    stage("DROID-SLAM 解算相机 · ZoeDepth 定尺度" if not is_static else "固定机位")
    cam_r, cam_t = run_metric_slam(folder, masks=masks, calib=cam_int, is_static=is_static)
    stage("SPEC 估计重力方向")
    wd_r, wd_t, spec_f = align_cam_to_world(imgfiles[0], cam_r, cam_t)
    return wh.make_pose(wd_r.numpy(), wd_t.numpy()), float(cam_int[0]), bool(is_static), focal_source, float(spec_f)


# --------------------------------------------------------------------------- people


def tracks_from_given(job, frame_numbers: list[int], max_people: int):
    """接进来的「人物框」（「ViTDet 人物框」→「选人」）-> VIMO 的轨迹：每个人 (编号, 帧序号, 框 [N,5], 有效)，
    和 tracks_from_deva 交的一个形状（官方 HMR_VIMO.inference 收的就是这个：hmr_vimo.py:134；EMDB 评测脚本拿真值框调它）。
    帧序号是这一段画面里的序号（imgfiles 的下标），框后面补一个 1.0 的分数。
    「最多人数」在这条路上同样管用：按文件里的顺序
    （最显眼的在前，job.people()）取前 `max_people` 个有框的人。"""
    index = {int(f): i for i, f in enumerate(frame_numbers)}
    out = []
    for pid, boxes in list(job.people().items())[:max_people]:
        present = sorted(f for f in boxes if f in index)
        if not present:
            continue
        out.append((int(pid), np.array([index[f] for f in present]),
                    np.array([[*boxes[f][:4], 1.0] for f in present], np.float64), np.ones(len(present), bool)))
    return out


def tracks_from_deva(tracks_obj, max_people: int):
    """Upstream estimate_humans.py: DEVA tracks sorted by length, their detection boxes."""
    tracks = tracks_obj.item() if isinstance(tracks_obj, np.ndarray) else tracks_obj
    ranked = sorted(tracks.values(), key=len, reverse=True)[:max_people]
    out = []
    for k, trk in enumerate(ranked):
        valid = np.array([t["det"] for t in trk])
        boxes = np.concatenate([t["det_box"] for t in trk])
        frame = np.array([t["frame"] for t in trk])
        out.append((k + 1, frame, boxes, valid))
    return out


@resident
def load_vimo(weights: Path):
    """VIMO (ViTDet, SAM, DEVA, DROID-SLAM, ZoeDepth and SPEC are built by upstream's pipeline functions on every
    job)."""
    from lib.models import get_hmr_vimo

    return get_hmr_vimo(checkpoint=str(weights / "data/pretrain/vimo_checkpoint.pth.tar"))


@resident
def load_smpl(weights: Path):
    """The SMPL layer (data/smpl below weights/, the working directory)."""
    from lib.models.smpl import SMPL

    return SMPL().cuda().eval()


def solve_people(run: Run, tracks, imgfiles, focal, width, height, frame_numbers, cam_to_world, weights: Path):
    from scipy.ndimage import gaussian_filter

    model = run.model("VIMO 模型", load_vimo, weights)
    smpl = run.model("SMPL 身体模型", load_smpl, weights)
    run.stage("VIMO 估计人体")
    img_center = np.array([width / 2.0, height / 2.0])
    people = []
    for k, (pid, frame, boxes, valid) in run.each(tracks, "VIMO"):
        res = model.inference(imgfiles, boxes, valid=valid, frame=frame, img_focal=focal, img_center=img_center)
        if res is None:
            say("N-TRAM-SHORTTRACK", person=int(pid), frames=MIN_FRAMES)
            continue
        idx = res["frame"].numpy().astype(int)
        rotmat = res["pred_rotmat"].numpy().astype(np.float64)
        betas = res["pred_shape"].numpy().astype(np.float64)
        tr_c = res["pred_trans"].reshape(-1, 3).numpy().astype(np.float64)
        go_c = mo.matrix_to_rotvec(rotmat[:, 0])
        body_pose = mo.matrix_to_rotvec(rotmat[:, 1:]).reshape(-1, 69)

        def root_rest(b):
            with torch.no_grad():
                b = torch.as_tensor(np.asarray(b), dtype=torch.float32, device="cuda")
                v = smpl.v_template + torch.einsum("fb,vcb->fvc", b, smpl.shapedirs[..., :b.shape[1]])
                return torch.einsum("v,fvc->fc", smpl.J_regressor[0], v).cpu().numpy().astype(np.float64)

        betas_one, shift = wh.lock_shape(betas, root_rest)
        tr_c = tr_c + shift
        verts_c, joints_c = wh.evaluate(smpl, 24, chunk=512, global_orient=go_c, body_pose=body_pose,
                                        betas=np.repeat(betas_one[None], len(go_c), 0), transl=tr_c, default_smpl=True)
        person = wh.Person(
            pid=pid, frames=frame_numbers[idx], global_orient=go_c, body_pose=body_pose, betas=betas_one,
            transl=tr_c, vertices=verts_c, joints=joints_c, j0=root_rest(betas_one[None])[0],
            global_orient_cam=go_c, transl_cam=tr_c, vertices_cam=verts_c, joints_cam=joints_c,
        )
        person.place_through(cam_to_world[idx])
        # upstream traj_filter: smooth the world root trajectory (a translation per frame)
        root = person.joints[:, 0].astype(np.float64)
        delta = gaussian_filter(root, sigma=3, axes=0) - root
        person.transl = person.transl + delta
        person.vertices = (person.vertices + delta[:, None]).astype(np.float32)
        person.joints = (person.joints + delta[:, None]).astype(np.float32)
        people.append(person)
    offload(model)  # waits in RAM: the next job's detection and SLAM have the GPU as in a new process
    return people, smpl


# --------------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "TRAM")
    job, params = run.job, run.params
    max_people, static = params["max_people"], params["static_camera"]

    weights = Path(job.weights_dir).resolve()
    repo = Path(job.repo_dir).resolve()
    raw = job.raw_dir.resolve()
    work = job.dir
    # before any heavy work: fail early; linked where TRAM reads it
    wh.body_model_file("smpl", "TRAM", weights / "data" / "smpl" / "SMPL_NEUTRAL.pkl")
    enter_runtime(weights, repo)

    frame_numbers, width, height = wh.plate(run, MIN_FRAMES, reason("I-TRAM-VIMOWINDOW", need=MIN_FRAMES))
    n = len(frame_numbers)
    img_folder = work / "tram_images"
    imgfiles = link_frames(job, img_folder)

    # 上游自己找人：ViTDet + SAM + DEVA，既给 DEVA 跟踪（没接「人物框」时 VIMO 按它解人），也给 SLAM 要的人物遮罩——
    # 所以接了「人物框」这一步照样跑（SLAM 的遮罩来自它），只是 VIMO 那一步换成按框解（下面 tracks_from_given）
    _, masks_rle, tracks_obj = detect_segment_track(imgfiles, work / "tram")
    torch.cuda.empty_cache()

    c2w, focal, is_static, focal_source, spec_f = solve_camera(
        img_folder, imgfiles, masks_rle, params["focal_px"], static, width, height)
    camera_info = {"focal_source": focal_source, "pose_source": "static" if is_static else "masked DROID-SLAM + ZoeDepth scale",
                   "gravity_source": "SPEC (pitch / roll of the first frame)", "spec_focal_px": round(spec_f, 1),
                   "static": is_static}
    world = "TRAM world: first camera at the origin, gravity-aligned from SPEC (+Y up, gravity -Y), metres"
    torch.cuda.empty_cache()

    # 接了「人物框」就按框解（官方 HMR_VIMO.inference 收框，nodes.py official 注 ⓪）；没接照官方 demo 用 DEVA 的轨迹
    tracks = (tracks_from_given(job, frame_numbers, max_people) if job.inputs.get("boxes")
              else tracks_from_deva(tracks_obj, max_people))
    if not tracks:
        nothing("N-TRAM-NOPEOPLE")
    people, smpl = solve_people(run, tracks, imgfiles, focal, width, height, frame_numbers, c2w, weights)
    if not people:
        nothing("N-TRAM-NOLONGTRACK", frames=MIN_FRAMES)

    run.stage("写出结果")
    out = [wh.save_person(raw, person, "smpl", smpl) for person in people]
    wh.save_camera(raw, frame_numbers, np.full(n, focal), c2w)
    wh.write_humans(
        run,
        frame_numbers,
        out,
        world,
        method="TRAM",
        body_model="SMPL neutral (SMPL_NEUTRAL.pkl), 10 betas; body_pose 23x3 axis-angle",
        camera={"file": "camera.npz", "width": width, "height": height, "focal_px": focal, **camera_info},
        up_axis="+Y",
        joints="24 SMPL joints, parents from the model",
        params={"focal_px": params["focal_px"], "static_camera": static, "max_people": max_people},
    )


if __name__ == "__main__":
    serve(main)
