"""Pixel3DMM worker: a FLAME head tracked over a whole shot. Runs inside
third_party/pixel3dmm/.venv with the composed code base (codebase.py) on PYTHONPATH;
never imports Lab2Shot core.

    python worker.py <job.json>

Node pixel3dmm.face, upstream's own five steps, called in this process instead of through
its os.system scripts (scripts/run_preprocessing.py, network_inference.py, track.py):

  1. cropping    PIPNet's FaceBoxes detector on every frame -> ONE square crop box for the
                 whole shot (upstream's static_crop: the mean box, 1.42x), each frame
                 cropped and resized to 512 x 512, plus 98 WFLW landmarks per frame.
  2. MICA        an identity (300 FLAME shape coefficients) from about ten frames, averaged.
  3. facer/FaRL  a face parsing map per frame (which pixels are skin, brows, lips ...).
  4. Pixel3DMM   the two ViT priors per frame: a surface-normal map (in the head's own
                 FLAME frame) and a canonical-face UV map.
  5. tracking    FLAME fitted to the shot: frame by frame (`iters`), then all frames jointly
                 (`global_iters`) with one camera for the shot, an optimised focal length
                 and temporal smoothness on expression, jaw, neck, head rotation and
                 translation.

Frames are handed to upstream as 00000.png, 00001.png ... (symlinks to the job's normalized
PNGs, so nothing is re-encoded; its own unpack_images would rewrite them as JPEG); the real
frame numbers come back on the way out. Upstream's tracker caps a shot at 1000 frames and
needs a face on every frame.

Output, the body-model family format (lab2shot_worker.world_humans), everything in the
solved camera's space (space "camera"):

    person_01.npz
        frames          int  [F]
        body_model      str  "flame"
        shape           f32  [300]      identity, one for the shot (MICA, refined jointly)
        expression      f32  [F,100]    FLAME expression coefficients
        jaw, neck       f32  [F,3,3]    rotation relative to the parent joint
        eyes            f32  [F,2,3,3]  left, right eyeball rotation
        eyelids         f32  [F,2]      the two eyelid blend shape weights
        global_orient   f32  [F,3]      the head in the camera (axis-angle)
        transl          f32  [F,3]      v = R(global_orient) (v_rest - root_rest) + root_rest + transl
        root_rest       f32  [3]
        vertices        f32  [F,5023,3] metres, OpenCV camera space (+X right, +Y down, +Z forward)
        joints          f32  [F,5,3]    FLAME's five joints, same space
        faces           int  [9976,3];  joint_names, parents [5]
        rest_vertices   f32  [5023,3]   shaped rest pose, FLAME space (+Y up, +Z out of the face)
        rest_joints     f32  [5,3]
        skin_weights    f32  [5023,5]
        local_rotations f32  [F,5,3,3]  each joint relative to its parent; the root's in camera space
        blendshapes     f32  [102,5023,3]  FLAME's 100 expression directions + the two eyelid shapes
        blendshape_names [102]; blendshape_weights f32 [F,102]
        landmarks_2d    f32  [F,98,2]   PIPNet's WFLW landmarks in the plate's pixels; a frame its
                                   second pass was under 0.99 sure of has zeros, as upstream reads them
    camera.npz          frames, focal_px [F], cam_to_world [F,4,4] (identity), principal_px [F,2]
    frame_<n>.npz       normal [h,w,3] OpenCV-camera unit normals, uv [h,w,2] FLAME UV 0..1,
                        valid [h,w] — the crop's rectangle, at its size in the plate

Pixel3DMM solves one camera for the whole shot, with its own focal length and principal
point; the head moves in front of it. The principal point is the crop's centre, so it is off
the plate's centre whenever the face is: camera.npz carries it (principal_px) and the family
writes it as the delivered camera's lens centre offset.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from lab2shot_shared import smpl
from lab2shot_shared.motion import matrix_to_rotvec
from lab2shot_worker import (
    fail,
    link_file,
    progress,
    reason,
    save_npz,
    say,
    serve,
    stage,
)
from lab2shot_worker import world_humans as wh
from lab2shot_worker.run import Run

from codebase import Layout

# 三个档位：迭代次数和提前结束的阈值一起变（upstream configs/tracking.yaml 的 iters / global_iters /
# early_stopping_delta）。节点只给这三档，不给随便填的数字
QUALITY = {
    "fast": {"iters": 100, "global_iters": 1500, "early_stopping_delta": 10.0},
    "standard": {"iters": 200, "global_iters": 5000, "early_stopping_delta": 5.0},  # upstream's defaults
    "fine": {"iters": 400, "global_iters": 10000, "early_stopping_delta": 2.0},
}
LEAST_FRAMES = 16   # configs/tracking.yaml batch_size: the joint stage samples this many frames at a time
MOST_FRAMES = 1000  # tracker.py MAX_STEPS = min(frames, 1000)
FLAME_JOINT_NAMES = ("root", "neck", "jaw", "left_eye", "right_eye")
TO_CV = np.diag([1.0, -1.0, -1.0])  # the tracker's OpenGL camera axes -> OpenCV (ours)
VIDEO = "shot"  # upstream keys every folder by a "video name"


# ------------------------------------------------------------------ upstream, one process per step


def run_step(lay: Layout, step: str, **args) -> None:
    """One of upstream's steps, in a process of its own (steps.py says why: MICA, PIPNet and facer each carry a
    top-level `utils` / `configs` / `models` package and they collide in one interpreter). Its own prints go to
    this worker's log; a step that stops takes the cook with it, naming which one."""
    import subprocess

    runner = Path(__file__).resolve().parent / "steps.py"
    env = {**os.environ, "PIXEL3DMM_EXT_ROOT": str(lay.root)}  # PIXEL3DMM_PEAKS: set in main()
    done = subprocess.run([sys.executable, str(runner), step, json.dumps(args)], env=env)
    if done.returncode != 0:
        fail("E-PIXEL3DMM-STEP", step=STEP_NAMES[step], code=done.returncode)


STEP_NAMES = {"crop": "找脸、裁切、面部关键点", "mica": "MICA 估计脸型", "segment": "面部分割",
              "priors": "预测法线图和规范面部坐标", "track": "拟合 FLAME"}


def prepare_upstream(lay: Layout, work: Path) -> None:
    """env_paths reads these when it is imported (here and in every step process): where its code and this job's
    folders are. Only the package tree goes on this process's path — never MICA's or PIPNet's."""
    os.environ["PIXEL3DMM_CODE_BASE"] = str(lay.code_base)
    os.environ["PIXEL3DMM_PREPROCESSED_DATA"] = str(work / "data")
    os.environ["PIXEL3DMM_TRACKING_OUTPUT"] = str(work / "track") + "/"  # the tracker joins it by string
    path = str(lay.code_base / "src")
    if path not in sys.path:
        sys.path.insert(0, path)


def link_frames(frames, rgb: Path) -> None:
    """Upstream reads a folder of 00000.png, 00001.png ...: the job's frames, linked (its own unpack_images
    would rewrite them as JPEG)."""
    rgb.mkdir(parents=True, exist_ok=True)
    for i, (_, path) in enumerate(frames):
        link_file(rgb / f"{i:05d}.png", Path(path))


def run_cropping(lay: Layout, rgb: Path, count: int) -> np.ndarray:
    """Step 1. Returns the crop rectangle in the plate's pixels [ymin, ymax, xmin, xmax]."""
    stage(STEP_NAMES["crop"])
    run_step(lay, "crop", rgb=str(rgb))
    got = len(list((rgb.parent / "cropped").glob("*.png")))
    if got != count:  # a frame with no face, or one under its 0.75 confidence, stops its detector
        fail("E-PIXEL3DMM-NOFACE", reason=f"只裁出了 {got} / {count} 帧")
    return np.load(rgb.parent / "crop_ymin_ymax_xmin_xmax.npy").astype(np.int64)


def run_mica(lay: Layout, data: Path) -> None:
    """Step 2: the identity prior, from about ten frames of the shot."""
    stage(STEP_NAMES["mica"])
    run_step(lay, "mica", data=str(data))
    if not any((data / "mica").glob("*/identity.npy")):
        fail("E-PIXEL3DMM-NOFACE", reason="MICA 在抽查的那几帧里没找到脸")


def run_segmentation(lay: Layout, data: Path, count: int) -> None:
    """Step 3: the face parsing map the UV and normal losses are masked by."""
    stage(STEP_NAMES["segment"])
    run_step(lay, "segment")
    got = len(list((data / "seg_og").glob("*.png")))
    if got != count:
        fail("E-PIXEL3DMM-NOSEG", got=got, count=count)


def run_priors(lay: Layout, data: Path, count: int) -> None:
    """Step 4: the normal and UV predictions, one pass of the ViT for each."""
    for kind, label in (("normals", "预测法线图"), ("uv_map", "预测规范面部坐标")):
        stage(label)
        run_step(lay, "priors", kind=kind)
        got = len(list((data / "p3dmm" / kind).glob("*.png")))
        if got != count:
            fail("E-PIXEL3DMM-PRIOR", kind=kind, got=got, count=count)


def run_tracking(lay: Layout, work: Path, quality: str, focal_norm: float | None) -> tuple[Path, int]:
    """Step 5: the two-stage fit. Returns (the folder it wrote, its render size)."""
    stage(STEP_NAMES["track"])
    wrote = work / "tracked.json"
    run_step(lay, "track", settings=QUALITY[quality], focal_norm=focal_norm,
             out=os.environ["PIXEL3DMM_TRACKING_OUTPUT"], wrote=str(wrote))
    if not wrote.exists():
        fail("E-PIXEL3DMM-STEP", step=STEP_NAMES["track"], code=0)
    said = json.loads(wrote.read_text(encoding="utf-8"))
    return work / "track" / said["folder"] / "checkpoint", int(said["size"])


# ------------------------------------------------------------------ reading the result back


def rot6d(values) -> np.ndarray:
    """Upstream keeps every rotation as 6D: [...,6] -> [...,3,3]."""
    import torch
    from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix

    values = np.asarray(values, np.float32)
    flat = rotation_6d_to_matrix(torch.from_numpy(values.reshape(-1, 6))).numpy().astype(np.float64)
    return flat.reshape(*values.shape[:-1], 3, 3)


def read_checkpoints(folder: Path, count: int) -> dict:
    """The per-frame .frame files, as the joint stage last rewrote them."""
    import torch

    got: dict[str, list] = {k: [] for k in ("exp", "eyes", "eyelids", "jaw", "neck", "R", "t")}
    shape, camera = None, {}
    for i in range(count):
        path = folder / f"{i:05d}.frame"
        if not path.exists():
            fail("E-PIXEL3DMM-NOTRACK", frame=i)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        for key in got:
            got[key].append(np.asarray(ck["flame"][key], np.float64)[0])
        if shape is None:
            shape = np.asarray(ck["flame"]["shape"], np.float64)[0]
            camera = {k: np.asarray(v, np.float64) for k, v in ck["camera"].items()}
        progress(i + 1, count, "读取跟踪结果")
    return {"shape": shape, "camera": camera, **{k: np.stack(v) for k, v in got.items()}}


def flame_model(lay: Layout):
    """A plain FLAME layer (the tracker's own is torch.compile'd): the rest pose, the skinning weights and the
    expression blend shapes the 蒙皮角色 is made of, and the exact mesh of every frame."""
    from omegaconf import OmegaConf
    from pixel3dmm.tracking.flame.FLAME import FLAME

    return FLAME(OmegaConf.load(lay.code_base / "configs" / "tracking.yaml")).cuda().eval()


def head_in_camera(flame, ck: dict) -> tuple[dict, np.ndarray]:
    """Everything the family's npz needs, in the solved camera's OpenCV frame. Returns (arrays, M_rot [F,3,3]).

    The tracker fits the head in a world of its own: the camera is fixed (R_base_0, t_base_0, an OpenGL
    world-to-camera pose) and the head moves by (R, t). Upstream's own mesh comes out of FLAME with an identity
    global rotation, so

        v_camera = TO_CV (R_base (R_head v_flame + t_head) + t_base) = M_rot v_flame + M_t

    and the rig is that one transform on the root joint: local_rotations[:, 0] = M_rot, with transl chosen so that
    R (v - root_rest) + root_rest + transl is exactly the same map. What the rig cannot carry is FLAME's pose
    correctives and the small joint offsets an expression causes — under a millimetre, as for SMIRK.
    """
    import torch

    n = len(ck["exp"])
    r_base, t_base = ck["camera"]["R_base_0"][0], ck["camera"]["t_base_0"][0]
    m_rot = TO_CV @ r_base @ rot6d(ck["R"])                               # [F,3,3]
    m_t = (ck["t"] @ r_base.T + t_base) @ TO_CV.T                         # [F,3]

    def cuda(a):
        return torch.as_tensor(np.asarray(a, np.float32), device="cuda")

    with torch.no_grad():
        verts = flame(shape_params=cuda(ck["shape"])[None].expand(n, -1), cameras=None,
                      expression_params=cuda(ck["exp"]), eye_pose_params=cuda(ck["eyes"]),
                      jaw_pose_params=cuda(ck["jaw"]), neck_pose_params=cuda(ck["neck"]),
                      eyelid_params=cuda(ck["eyelids"]))[0].cpu().numpy().astype(np.float64)
        dirs = flame.shapedirs.cpu().numpy()                              # [V,3,300+100]
        rest = flame.v_template.cpu().numpy() + dirs[:, :, :300] @ ck["shape"]
        rest_joints = flame.J_regressor.cpu().numpy() @ rest              # [5,3]
        eyelid_dirs = np.concatenate([flame.l_eyelid.cpu().numpy(), flame.r_eyelid.cpu().numpy()])  # [2,V,3]
        parents = flame.parents.cpu().numpy().astype(np.int32)
        faces = flame.faces.cpu().numpy().astype(np.int32).reshape(-1, 3)
        skin_weights = flame.lbs_weights.cpu().numpy().astype(np.float32)

    r0 = rest_joints[0]
    transl = np.einsum("fij,j->fi", m_rot, r0) + m_t - r0
    local = np.tile(np.eye(3), (n, len(FLAME_JOINT_NAMES), 1, 1))
    local[:, 0] = m_rot
    local[:, 1] = rot6d(ck["neck"])
    local[:, 2] = rot6d(ck["jaw"])
    local[:, 3] = rot6d(ck["eyes"][:, :6])
    local[:, 4] = rot6d(ck["eyes"][:, 6:])
    arrays = {
        "shape": ck["shape"].astype(np.float32),
        "expression": ck["exp"].astype(np.float32),
        "neck": local[:, 1].astype(np.float32),
        "jaw": local[:, 2].astype(np.float32),
        "eyes": local[:, 3:5].astype(np.float32),
        "eyelids": ck["eyelids"].astype(np.float32),
        "global_orient": matrix_to_rotvec(m_rot).astype(np.float32),
        "transl": transl.astype(np.float32),
        "root_rest": r0.astype(np.float32),
        "vertices": (np.einsum("fij,fvj->fvi", m_rot, verts) + m_t[:, None]).astype(np.float32),
        "joints": smpl.world_of(parents, rest_joints, local, transl + r0)[:, :, :3, 3].astype(np.float32),
        "faces": faces,
        "joint_names": np.array(FLAME_JOINT_NAMES),
        "parents": parents,
        "rest_vertices": rest.astype(np.float32),
        "rest_joints": rest_joints.astype(np.float32),
        "skin_weights": skin_weights,
        "local_rotations": local.astype(np.float32),
        "blendshapes": np.concatenate([dirs[:, :, 300:400].transpose(2, 0, 1), eyelid_dirs]).astype(np.float32),
        "blendshape_names": np.array([f"expression_{i:03d}" for i in range(100)] + ["eyelid_left", "eyelid_right"]),
        "blendshape_weights": np.concatenate([ck["exp"], ck["eyelids"]], 1).astype(np.float32),
    }
    return arrays, m_rot


def intrinsics(ck: dict, rect: np.ndarray, size: int) -> tuple[float, np.ndarray]:
    """The solved camera in the plate's pixels: (focal length, principal point [2]).

    Upstream projects with x = f X/Z + (cx - 1), y = cy + f Y/Z in its `size` x `size` render, where f = fl * size
    and (cx, cy) = size/2 + 0.5 + pp * (size/2 + 0.5) — the -1 on x is its own `use_hack` in get_intrinsics, and the
    signs are read off project_points_screen_space. That render is the 512 crop, and the crop is the rectangle
    `rect` of the plate resized to it, so a pixel maps back as x_plate = xmin + (x + 0.5) * w / size - 0.5.
    """
    ymin, ymax, xmin, xmax = (float(v) for v in rect)
    w, h = xmax - xmin, ymax - ymin
    fl = float(np.asarray(ck["camera"]["fl"]).reshape(-1)[0])
    ppx, ppy = (float(v) for v in np.asarray(ck["camera"]["pp"]).reshape(-1)[:2])
    half = size / 2 + 0.5
    cx, cy = half + ppx * half - 1.0, half + ppy * half
    return fl * w, np.array([xmin + (cx + 0.5) * w / size - 0.5, ymin + (cy + 0.5) * h / size - 0.5])


def read_landmarks(folder: Path, count: int) -> np.ndarray:
    """PIPNet's 98 WFLW landmarks per frame, 0..1 in the crop. It writes no file for a frame its second pass is
    less than 0.99 sure of, and upstream's tracker reads those as zeros and masks them out of its landmark losses
    (tracker.read_data); the same here, so the shot carries the same frames either way."""
    out = np.zeros((count, 98, 2), np.float64)
    for i in range(count):
        path = folder / f"{i:05d}.npy"
        if path.exists():
            out[i] = np.load(path)
    return out


def write_maps(raw: Path, data: Path, numbers, rect: np.ndarray, m_rot: np.ndarray) -> None:
    """The two priors as the node's per-frame files, resized to the crop's rectangle in the plate (the node pastes
    them onto the plate's canvas, where the rest is 没有值).

    The normal map is in the head's own FLAME frame (upstream's README, and its normal loss, which compares the
    prediction against R^T x the rendered world normals): n_camera = M_rot n_head, the same M_rot the head is
    placed by. The UV map is FLAME's own UV, 0..1, as the tracker reads it.
    """
    import cv2

    stage("写出法线图和 UV 坐标图")
    ymin, ymax, xmin, xmax = (int(v) for v in rect)
    size = (xmax - xmin, ymax - ymin)
    for i, f in enumerate(numbers):
        normal = cv2.imread(str(data / "p3dmm" / "normals" / f"{i:05d}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        uv = cv2.imread(str(data / "p3dmm" / "uv_map" / f"{i:05d}.png"), cv2.IMREAD_COLOR)[:, :, ::-1]
        n = cv2.resize(normal.astype(np.float32) / 255.0 * 2.0 - 1.0, size, interpolation=cv2.INTER_LINEAR)
        n = np.einsum("ij,hwj->hwi", m_rot[i], n)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
        save_npz(raw / f"frame_{f}.npz", normal=n.astype(np.float16),
                 uv=cv2.resize(uv[:, :, :2].astype(np.float32) / 255.0, size,
                               interpolation=cv2.INTER_LINEAR).astype(np.float16),
                 valid=np.ones(size[::-1], np.float16))
        progress(i + 1, len(numbers), "写出法线图和 UV 坐标图")


# ------------------------------------------------------------------ main


def main(job_path: str) -> None:
    run = Run.start(job_path, "pixel3dmm.face", "Pixel3DMM")
    job = run.job
    quality = job.params["quality"]
    focal_px = job.params["focal_px"]  # None: Pixel3DMM solves the focal length itself
    lay = Layout(Path(job.repo_dir).parent)
    wh.body_model_file("flame", "Pixel3DMM", link=lay.flame_assets / "FLAME2020" / "generic_model.pkl")
    numbers, width, height = wh.plate(run, LEAST_FRAMES, reason("N-PIXEL3DMM-WHYFRAMES", least=LEAST_FRAMES))
    if len(numbers) > MOST_FRAMES:
        fail("E-PIXEL3DMM-TOOMANYFRAMES", most=MOST_FRAMES, have=len(numbers))

    work = Path(job.dir) / "p3dmm"
    work.mkdir(parents=True, exist_ok=True)
    os.environ["PIXEL3DMM_PEAKS"] = str(work / "gpu_peaks.jsonl")  # each step writes what it used of the GPU
    prepare_upstream(lay, work)
    data = work / "data" / VIDEO
    link_frames(job.frames, data / "rgb")

    rect = run_cropping(lay, data / "rgb", len(numbers))
    run_mica(lay, data)
    run_segmentation(lay, data, len(numbers))
    run_priors(lay, data, len(numbers))

    crop_w, crop_h = float(rect[3] - rect[2]), float(rect[1] - rect[0])
    checkpoints, size = run_tracking(lay, work, quality, float(focal_px) / crop_w if focal_px else None)

    flame = run.model("FLAME 模型", flame_model, lay)
    run.stage("整理结果")
    ck = read_checkpoints(checkpoints, len(numbers))
    person, m_rot = head_in_camera(flame, ck)
    focal_plate, principal = intrinsics(ck, rect, size)
    landmarks = read_landmarks(data / "PIPnet_landmarks", len(numbers)) * np.array([crop_w, crop_h]) \
        + np.array([rect[2], rect[0]])

    raw = Path(job.raw_dir)
    # `solved`：真正解出来的那些帧（契约：lab2shot_worker/world_humans.py）。Pixel3DMM 逐帧拟合，
    # 跟到哪几帧就交哪几帧（`numbers`），中间不补洞，所以和 frames 一样
    save_npz(raw / "person_01.npz", frames=np.asarray(numbers, np.int64), solved=np.asarray(numbers, np.int64),
             body_model=np.array("flame"), landmarks_2d=landmarks.astype(np.float32), **person)
    wh.save_camera(raw, numbers, focal_plate, principal_px=principal)
    write_maps(raw, data, numbers, rect, m_rot)

    # the models run in the step processes: their peaks, not this process's (the Run's own gpu_peak_mb would be this
    # process's, so the worker's value below overrides it)
    peaks: dict[str, int] = {}
    for line in Path(os.environ["PIXEL3DMM_PEAKS"]).read_text(encoding="utf-8").splitlines():
        said = json.loads(line)
        peaks[said["step"]] = max(peaks.get(said["step"], 0), said["mb"])
    off = float(np.hypot(principal[0] - width / 2, principal[1] - height / 2) / width)
    if job.params.get("has_camera") and off > 0.02:
        say("N-PIXEL3DMM-OFFCENTRE", percent=round(off * 100, 1))
    wh.write_humans(
        run, numbers, [{"name": "person_01_face", "file": "person_01.npz"}], None,
        body_model="flame",
        model="Pixel3DMM (uv.ckpt + normals.ckpt) + MICA + FLAME 2020 generic",
        crop=[int(v) for v in rect],
        rig=("v = R(global_orient) (v_rest(shape) - root_rest) + root_rest + transl; local_rotations[:, 0] is the head "
             "in camera space, [:, 1] the neck, [:, 2] the jaw, [:, 3:5] the eyeballs. Skinning rest_vertices + "
             "blendshapes @ blendshape_weights with skin_weights / local_rotations gives vertices up to FLAME's pose "
             "correctives and the small expression-dependent joint offsets"),
        camera_model=("one camera for the shot: focal_px at the plate's width, principal_px its lens centre (the crop's "
                      "centre, off the plate's centre when the face is); the head moves in front of it"),
        flame=("FLAME 2020 generic_model.pkl, 300 shape components (locked for the shot) + 100 expression, "
               "neck, jaw and eyeball rotations, two eyelid shapes"),
        temporal=("two stages: frame by frame, then all frames jointly with temporal smoothness on expression, jaw, "
                  "neck, head rotation and translation"),
        quality=quality, iters=QUALITY[quality]["iters"], global_iters=QUALITY[quality]["global_iters"],
        focal_px=focal_plate, focal_source="node" if focal_px else "pixel3dmm",
        principal_px=[float(v) for v in principal],
        width=width, height=height, frames=[int(f) for f in numbers],  # the whole list, not the standard [first, last]
        gpu_peak_mb=max(peaks.values(), default=0), gpu_peak_by_step=peaks,
    )


if __name__ == "__main__":
    serve(main)
