"""SMIRK worker: FLAME face shape and expression per frame. Runs inside
third_party/smirk/.venv with the pinned repo on PYTHONPATH and as working
directory (upstream loads assets/ relative to it); never imports Lab2Shot core.

    python worker.py <job.json>

Node smirk.face, per frame, the upstream demo's pipeline (demo.py / demo_video.py):
crop "auto": MediaPipe Face Landmarker (IMAGE mode, one face) -> square crop
around the landmarks, 1.4 x their extent -> 224 x 224; crop "none": the whole
frame, padded to a square (no stretching) -> 224 x 224 -> SMIRK encoder ->
FLAME 2020 (300 shape + 50 expression components, jaw, eyelids, head rotation).
SMIRK's camera is orthographic; it is turned into a pinhole camera at focal_px
(depth from the orthographic scale), so the head sits in camera space.

Output, the body-model family format (lab2shot_worker.world_humans):
raw/person_01.npz (one face; every input frame)；结果在相机空间；raw/plate_camera.npz 是放脸用的那台针孔相机（原点，focal_px，主点在画面中心）。

    frames          int  [F]
    body_model      str  "flame"
    solved          int  [S]        the frames where `found` is true
    found           bool [F]        MediaPipe found the face (crop auto); other frames reuse the nearest crop
    shape           f32  [300]      identity, locked for the shot (median of the frames)
    shape_per_frame f32  [F,300]    SMIRK's own per-frame estimates
    expression      f32  [F,50]     FLAME expression coefficients
    jaw             f32  [F,3]      jaw rotation, axis-angle
    eyelids         f32  [F,2]      SMIRK eyelid blend shape weights (left, right), 0..1
    pose            f32  [F,3]      SMIRK head rotation, axis-angle, FLAME space (+Y up, +Z out of the face)
    global_orient   f32  [F,3]      the head rotation in camera space (axis-angle)
    transl          f32  [F,3]      v = R(global_orient) (v_rest(pose) - root_rest) + root_rest + transl
    root_rest       f32  [3]
    vertices        f32  [F,5023,3] metres, camera space of each frame (OpenCV: +X right, +Y down, +Z forward)
    joints          f32  [F,5,3]    FLAME skeleton (joint_names / parents), same space
    faces           int  [9976,3];  joint_names, parents [5]
    rest_vertices   f32  [5023,3]   shaped rest pose (identity, zero expression, zero pose), FLAME space
    rest_joints     f32  [5,3]
    skin_weights    f32  [5023,5]   FLAME's LBS weights
    local_rotations f32  [F,5,3,3]  each joint relative to its parent; the root's in camera space
    blendshapes     f32  [52,5023,3] the 50 FLAME expression directions + SMIRK's left / right eyelid shapes
    blendshape_names [52]; blendshape_weights f32 [F,52] (= expression, eyelids)
    cam             f32  [F,3]      SMIRK orthographic camera (s, tx, ty) in the 224 crop
    crop_matrix     f64  [F,3,3]    frame pixels -> crop pixels (similarity)
    landmarks_2d    f32  [F,478,2]  MediaPipe landmarks in frame pixels (NaN where not found)
    focal_px        float           pinhole focal length behind transl; principal point W/2, H/2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from lab2shot_worker import (
    nothing,
    progress,
    read_frame,
    resident,
    save_npz,
    say,
    serve,
)
from lab2shot_shared import motion as mo
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run

IMAGE_SIZE = 224  # SMIRK input
CROP_SCALE = 1.4  # demo.py crop_face(scale=1.4)
BATCH = 64
# MediaPipe settings of utils/mediapipe_utils.py
MP_MIN_DETECTION, MP_MIN_PRESENCE = 0.1, 0.1


# ------------------------------------------------------------------ crop


def detect_landmarks(run: Run, frames, task: Path) -> dict[int, np.ndarray]:
    """frame -> [478,2] MediaPipe landmarks in pixels (x * width, y * height as upstream); missing if no face."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task), delegate=BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=MP_MIN_DETECTION,
        min_face_presence_confidence=MP_MIN_PRESENCE,
    )
    found = {}
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for n, (frame, path) in run.each(frames, "找脸"):
            rgb = read_frame(path)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if result.face_landmarks:
                h, w = rgb.shape[:2]
                found[frame] = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]], np.float64)
    return found


def crop_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """demo.py crop_face(): square around the landmarks' extent x 1.4 -> 3x3 similarity, frame px -> crop px."""
    from skimage.transform import estimate_transform

    left, right = landmarks[:, 0].min(), landmarks[:, 0].max()
    top, bottom = landmarks[:, 1].min(), landmarks[:, 1].max()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * CROP_SCALE)
    src = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]])
    dst = np.array([[0, 0], [0, IMAGE_SIZE - 1], [IMAGE_SIZE - 1, 0]])
    return estimate_transform("similarity", src, dst).params


def crop_whole_frame(width: int, height: int) -> np.ndarray:
    """crop none: the whole frame centred in a square (padding, no stretching)."""
    k = IMAGE_SIZE / max(width, height)
    return np.array([[k, 0, (IMAGE_SIZE - k * width) / 2], [0, k, (IMAGE_SIZE - k * height) / 2], [0, 0, 1]])


def warp_crop(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Same resampling as upstream: skimage warp with the inverse transform (bilinear), uint8."""
    from skimage.transform import SimilarityTransform, warp

    tform = SimilarityTransform(matrix=matrix)
    return warp(rgb, tform.inverse, output_shape=(IMAGE_SIZE, IMAGE_SIZE), preserve_range=True).astype(np.uint8)


def nearest_fill(frames: list[int], known: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    keys = np.array(sorted(known))
    return {f: known[int(keys[np.abs(keys - f).argmin()])] for f in frames}


# ------------------------------------------------------------------ model


@resident
def load_models(repo: Path, weights: Path, flame_file: Path, device):
    import timm
    import torch

    # The encoders are built with timm "pretrained=True" (ImageNet weights from the
    # Hub); SMIRK's checkpoint replaces all of them, so build them empty and offline.
    create_model = timm.create_model
    timm.create_model = lambda *a, **kw: create_model(*a, **{**kw, "pretrained": False})
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from src.FLAME.FLAME import FLAME
    from src.smirk_encoder import SmirkEncoder

    encoder = SmirkEncoder()
    timm.create_model = create_model
    ckpt = torch.load(weights / "SMIRK_em1.pt", map_location="cpu", weights_only=False)
    state = {k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k}
    encoder.load_state_dict(state)  # strict: every encoder parameter must come from the checkpoint
    flame = FLAME(flame_model_path=str(flame_file))
    return encoder.to(device).eval(), flame.to(device).eval()


def perspective(cam: np.ndarray, crop: np.ndarray, focal: float, cx: float, cy: float) -> np.ndarray:
    """Weak-perspective (orthographic) SMIRK camera -> camera translation for vertices @ diag(1,-1,-1) + t.

    Crop pixel u_c = 112 (1 + s (x + tx)); frame pixel = inverse(crop) @ crop pixel (a similarity with
    scale 1/k). Matching a pinhole camera at the head's depth: t_z = f k / (112 s), and t_x, t_y put the
    FLAME origin where the orthographic camera puts it.
    """
    s, tx, ty = (float(v) for v in cam)
    inv = np.linalg.inv(crop)
    k = float(np.sqrt(abs(np.linalg.det(crop[:2, :2]))))
    half = IMAGE_SIZE / 2
    origin = inv @ np.array([half * (1 + s * tx), half * (1 - s * ty), 1.0])  # FLAME origin in frame pixels
    tz = focal * k / (half * s)
    return np.array([(origin[0] - cx) * tz / focal, (origin[1] - cy) * tz / focal, tz])


# ------------------------------------------------------------------ rig


FLAME_JOINT_NAMES = ("root", "neck", "jaw", "left_eye", "right_eye")
TO_CAMERA = np.diag([1.0, -1.0, -1.0])  # FLAME space (+Y up, +Z to the viewer) -> OpenCV camera: 180 deg about X


def evaluate_flame(flame, shape: np.ndarray, p: dict[str, np.ndarray], device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FLAME.forward() of the repo with the locked shape, keeping the posed joints: vertices [F,V,3],
    joints [F,5,3] (FLAME space) and each frame's rest root joint j0 [F,3] (it moves slightly with expression)."""
    import torch
    from src.FLAME.lbs import lbs, vertices2joints

    n = len(p["pose"])
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)
    with torch.no_grad():
        betas = torch.cat([t(shape)[None].expand(n, -1), t(p["expression"])], 1)
        zeros = torch.zeros(n, 3, device=device)
        full_pose = torch.cat([t(p["pose"]), zeros, t(p["jaw"]), zeros, zeros], 1)  # neck and eyeballs fixed
        template = flame.v_template[None].expand(n, -1, -1)
        verts, joints = lbs(betas, full_pose, template, flame.shapedirs, flame.posedirs, flame.J_regressor,
                            flame.parents, flame.lbs_weights, dtype=torch.float32)
        eyelids = t(p["eyelids"])
        verts = verts + flame.r_eyelid * eyelids[:, 1:2, None] + flame.l_eyelid * eyelids[:, 0:1, None]
        shaped = template + torch.einsum("bl,mkl->bmk", betas, flame.shapedirs)
        j0 = vertices2joints(flame.J_regressor, shaped)[:, 0]
    return verts.cpu().numpy(), joints.cpu().numpy(), j0.cpu().numpy()


def face_rig(flame, shape: np.ndarray, p: dict[str, np.ndarray], cam_t: np.ndarray, device) -> dict:
    """The family arrays in camera space: v_cam = TO_CAMERA v_flame + cam_t, written as a rig whose root
    carries TO_CAMERA R(pose) and whose translation absorbs where the root joint sits."""
    import torch
    from src.FLAME.lbs import vertices2joints

    verts, joints, j0 = evaluate_flame(flame, shape, p, device)
    n_exp = p["expression"].shape[1]
    with torch.no_grad():
        dirs = flame.shapedirs.cpu().numpy()  # [V,3,n_shape+n_exp]
        rest = flame.v_template.cpu().numpy() + dirs[:, :, :len(shape)] @ shape
        rest_joints = vertices2joints(flame.J_regressor, torch.as_tensor(rest[None], device=device, dtype=torch.float32))[0].cpu().numpy()
        eyelid_dirs = np.concatenate([flame.l_eyelid.cpu().numpy(), flame.r_eyelid.cpu().numpy()])  # [2,V,3]
    root = TO_CAMERA @ mo.rotvec_to_matrix(p["pose"])  # [F,3,3]
    local = np.tile(np.eye(3), (len(root), len(FLAME_JOINT_NAMES), 1, 1))
    local[:, 0] = root
    local[:, 2] = mo.rotvec_to_matrix(p["jaw"])
    r0 = rest_joints[0]
    # exact for the root: TO_CAMERA (R (v - j0_f) + j0_f) + cam_t == root (v - r0) + r0 + transl
    transl = cam_t + j0 @ TO_CAMERA.T - r0 - np.einsum("fij,fj->fi", root, j0 - r0)
    return {
        "vertices": (verts @ TO_CAMERA.T + cam_t[:, None]).astype(np.float32),
        "joints": (joints @ TO_CAMERA.T + cam_t[:, None]).astype(np.float32),
        "global_orient": mo.matrix_to_rotvec(root).astype(np.float32),
        "transl": transl.astype(np.float32),
        "root_rest": r0.astype(np.float32),
        "rest_vertices": rest.astype(np.float32),
        "rest_joints": rest_joints.astype(np.float32),
        "skin_weights": flame.lbs_weights.cpu().numpy().astype(np.float32),
        "local_rotations": local.astype(np.float32),
        "parents": flame.parents.cpu().numpy().astype(np.int32),
        "faces": flame.faces_tensor.cpu().numpy().astype(np.int32),
        "joint_names": np.array(FLAME_JOINT_NAMES),
        "blendshapes": np.concatenate([dirs[:, :, len(shape):len(shape) + n_exp].transpose(2, 0, 1), eyelid_dirs]).astype(np.float32),
        "blendshape_names": np.array([f"expression_{i:02d}" for i in range(n_exp)] + ["eyelid_left", "eyelid_right"]),
        "blendshape_weights": np.concatenate([p["expression"], p["eyelids"]], 1).astype(np.float32),
    }


# ------------------------------------------------------------------ main


def main(job_path: str) -> None:
    import torch

    run = Run.start(job_path, "smirk.face", "SMIRK")
    job, params = run.job, run.params
    crop_mode = params["crop"]
    focal = params["focal_px"]  # the node always sends one (a 50 mm lens by default)
    flame_file = wh.body_model_file("flame", "SMIRK")  # before anything slow
    frames = run.frames()
    weights = Path(job.weights_dir)
    device = torch.device("cuda")
    width, height = frames.width, frames.height
    numbers = frames.numbers

    landmarks: dict[int, np.ndarray] = {}
    if crop_mode == "auto":
        run.stage("MediaPipe 找脸")
        landmarks = detect_landmarks(run, frames.pairs, weights / "face_landmarker.task")
        if not landmarks:
            nothing("N-SMIRK-NOFACE", frames=len(frames))
        if len(landmarks) < len(frames):
            say("N-SMIRK-FACEMISSING", missing=len(frames) - len(landmarks), frames=len(frames))
        crops = nearest_fill(numbers, {f: crop_from_landmarks(lm) for f, lm in landmarks.items()})
    else:
        crops = {f: crop_whole_frame(width, height) for f in numbers}

    encoder, flame = run.model("SMIRK 和 FLAME 模型", load_models, job.repo_dir, weights, flame_file, device)

    run.stage("估计表情")
    keys = {"shape_params": "shape", "expression_params": "expression", "jaw_params": "jaw",
            "eyelid_params": "eyelids", "pose_params": "pose", "cam": "cam"}
    acc: dict[str, list[np.ndarray]] = {k: [] for k in keys.values()}
    for start in range(0, len(frames), BATCH):
        chunk = frames.pairs[start:start + BATCH]
        imgs = np.stack([warp_crop(read_frame(p), crops[f]) for f, p in chunk])
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0).to(device)
        with torch.no_grad():
            out = encoder(x)
        for k, name in keys.items():
            acc[name].append(out[k].float().cpu().numpy())
        progress(min(start + BATCH, len(frames)), len(frames), "估计表情")
    p = {k: np.concatenate(v).astype(np.float32) for k, v in acc.items()}

    run.stage("写出结果")
    shape = np.median(p["shape"], axis=0).astype(np.float32)  # one identity for the shot
    crop_matrix = np.stack([crops[f] for f in numbers])
    cam_t = np.stack([perspective(c, m, focal, width / 2.0, height / 2.0) for c, m in zip(p["cam"], crop_matrix)])
    rig = face_rig(flame, shape, p, cam_t, device)
    lm = np.full((len(frames), 478, 2), np.nan, np.float32)
    for i, f in enumerate(numbers):
        if f in landmarks:
            lm[i] = landmarks[f]
    raw = job.raw_dir
    found = np.array([f in landmarks for f in numbers]) if crop_mode == "auto" else np.ones(len(numbers), bool)
    save_npz(
        raw / "person_01.npz",
        frames=np.array(numbers, np.int64),
        # 真正解出来的那些帧（契约：lab2shot_worker/world_humans.py）：就是 `found` 为真的那几帧，
        # 其余帧上的脸是按最近的裁切补出来的
        solved=np.asarray(numbers, np.int64)[found],
        body_model=np.array("flame"),
        found=found,
        shape=shape,
        shape_per_frame=p["shape"],
        expression=p["expression"],
        jaw=p["jaw"],
        eyelids=p["eyelids"],
        pose=p["pose"],
        cam=p["cam"],
        crop_matrix=crop_matrix,
        landmarks_2d=lm,
        focal_px=np.float64(focal),
        **rig,
    )
    # plate_camera.npz：脸所在的那台针孔相机（原点、不动；focal_px 就是上面用来把弱透视换成针孔的那个，主点在画面中心）。
    # 它不是上游解出来的（官方的 outputs['cam'] 是裁切上的弱透视三个数），是本节点放脸用的相机，交出去让三维视图
    # 透过它看背板、交付时和头对得上（nodes.py official ours）
    wh.save_camera(raw, numbers, focal, name=wh.PLATE_CAMERA)

    wh.write_humans(
        run,
        numbers,
        [{"name": "person_01_face", "file": "person_01.npz"}],
        None,
        body_model="flame",
        model="SMIRK (SMIRK_em1) + FLAME 2020 generic",
        rig=("v = R(global_orient) (v_rest(pose) - root_rest) + root_rest + transl; local_rotations[:, 0] is the head "
             "in camera space (TO_CAMERA @ R(pose)), [:, 2] the jaw, neck and eyes identity; rest_vertices are FLAME "
             "space (+Y up, +Z out of the face). Skinning rest_vertices + blendshapes @ blendshape_weights with "
             "skin_weights / local_rotations gives vertices up to FLAME's pose correctives and the small "
             "expression-dependent joint offsets; SMIRK adds the eyelid shapes after skinning (as here)"),
        camera_model=("SMIRK is orthographic: crop pixel u = 112 (1 + s (x + tx)), v = 112 (1 - s (y + ty)) with "
                      "cam = (s, tx, ty) and (x, y) the FLAME-space vertices; frame pixel = inverse(crop_matrix) @ "
                      "(u, v, 1). transl is the matching pinhole camera at focal_px: depth = f k / (112 s)"),
        flame="FLAME 2020 generic_model.pkl, 300 shape (locked, median) + 50 expression components; jaw and pose axis-angle",
        temporal="per frame (no smoothing); the identity is the median of all frames",
        crop=crop_mode,
        crop_scale=CROP_SCALE if crop_mode == "auto" else None,
        faces_found=int(found.sum()),
        focal_px=focal,
        width=width,
        height=height,
        frames=numbers,  # the whole list, not the standard [first, last]
    )


if __name__ == "__main__":
    serve(main)
