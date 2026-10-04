"""OpenDelight worker: face delighting. Runs inside third_party/opendelight/.venv-ada-blackwell
with the original repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

Per frame, the steps of upstream test.py (repo files untouched):
RetinaFace + FAN 68 landmarks -> DAViD soft foreground matte x FaRL (LaPa) face
parsing without hair = face matte -> FFHQ-style crop from the landmarks ->
delight network (MAE ViT-B + Sapiens decoder, 512) -> optional UNet enhancer
(resolution) -> warped back to the frame -> raw/frame_<n>.npz (basecolor, alpha).

Differences from test.py, all outside the network: the base colour is kept as float
(test.py quantizes it to 8 bit before warping back), the alpha is not rounded to
8 bit, one face per frame is chosen (the largest, or the one overlapping the
previous frame's face), and landmarks can be smoothed over time for video.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
# torch first: its CUDA 12.8 / cuDNN 9 libraries are the ones onnxruntime's CUDA provider loads.
import torch

from lab2shot_worker import fail, read_frame, require_weights, resident, say, serve
from lab2shot_worker.frame_io import Writer
from lab2shot_worker.run import Run

os.environ.setdefault("MPLBACKEND", "Agg")  # facer imports matplotlib

NODE = "opendelight.delight"

# test.py defaults
BORDER_SCALE = 0.85
NET_SIZE = 512  # the delight network always runs at 512; resolution is the crop / enhancer size
DATA_IDX = 0  # --data_idx: which register token of the mae_mix encoder
DET_THRESHOLD = 0.8
LAPA_HAIR = 10

# Temporal landmark smoothing (video): Gaussian over time, zero lag. A cut, a gap
# in the frame numbers or a jump of the face breaks the run, so unrelated photos
# in one job are never blended together (even aligned portraits, whose landmarks
# sit in the same place).
SMOOTH_SIGMA = 1.5  # frames
MAX_JUMP = 0.25  # mean landmark motion between two frames, relative to the face size
# Mean abs difference of 64x64 grey thumbnails (0..1) above which two frames are a cut.
# Consecutive video frames score 0.00-0.06 (handheld walking shot), different photos 0.20-0.31.
CUT_THRESHOLD = 0.12


def find_one(pattern_root: Path, pattern: str) -> Path:
    found = sorted(pattern_root.glob(pattern))
    if not found:
        require_weights("opendelight", pattern_root / pattern)
    return found[-1]


@resident(movable=False)  # an onnxruntime CUDA session: making room frees it
def load_matting(repo: Path, weights: Path):
    """DAViD soft foreground matte (matting/runtime). Its modules import each other as top-level names, and its
    `utils` clashes with the repo's own utils.py: import it in isolation, then forget the short names."""
    runtime = str(repo / "matting" / "runtime")
    sys.path.insert(0, runtime)
    saved = {k: sys.modules.pop(k) for k in ("utils",) if k in sys.modules}
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")  # CUDA / cuDNN from the pip nvidia-* packages (torch's)
        from soft_foreground_segmenter import SoftForegroundSegmenter
    finally:
        sys.path.remove(runtime)
        for k in ("utils", "pixelwise_estimator", "soft_foreground_segmenter"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    matting = SoftForegroundSegmenter(onnx_model=weights / "david" / "foreground-segmentation-model-vitl16_384.onnx")
    providers = matting.onnx_sess.get_providers()
    if providers[0] != "CUDAExecutionProvider":
        say("N-OPENDELIGHT-CPUMATTING", providers=list(providers))
    return matting


@resident
def load_face_models(weights: Path, device: str):
    """ibug RetinaFace + FAN (from the pinned source archives in weights/) and FaRL face parsing (LaPa, 448, the
    local TorchScript file)."""
    for folder in (find_one(weights / "ibug" / "face_detection", "face_detection-*"),
                   find_one(weights / "ibug" / "face_alignment", "face_alignment-*")):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    from ibug.face_alignment import FANPredictor
    from ibug.face_detection import RetinaFacePredictor

    r50 = RetinaFacePredictor.get_model("resnet50")
    r50.weights = str(weights / "ibug" / "Resnet50_Final.pth")  # the LFS object, fetched separately
    detector = RetinaFacePredictor(threshold=DET_THRESHOLD, device=device, model=r50)
    landmarks = FANPredictor(device=device, model=FANPredictor.get_model("2dfan2_alt"))
    import facer

    parser = facer.face_parser(
        "farl/lapa/448", device=device,
        model_path=str(weights / "facer" / "face_parsing.farl.lapa.main_ema_136500_jit191.pt"),
    )
    return detector, landmarks, parser


def repo_modules(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


@resident
def load_delight(repo: Path, weights: Path, device: str):
    """The delight network (repo code) -> (encoder type, network)."""
    repo_modules(repo)
    import yaml

    import encoder.mae_mix_encoder as mae_mix
    from model.delight_base_net import DelightBaseModel

    # get_mae_encoder() loads the MAE ImageNet weights (CC BY-NC 4.0) only as a training
    # initialization; the OpenDelight checkpoint then overwrites every parameter
    # (strict load below). Skip it: the MAE file is not needed at inference.
    mae_mix.load_part_mae_model = lambda ckpt_path, model: model
    cfg = yaml.safe_load((repo / "config" / "delight_base_network.yaml").read_text())
    model = DelightBaseModel(enc_cfg=cfg["encoder"], dec_cfg=cfg["decoder"], device=device).to(device)
    state = torch.load(weights / "opendelight" / "base_delight_network.pth", map_location=device)
    model.load_state_dict(state, strict=True)
    return cfg["encoder"]["type"], model.eval()


@resident
def load_enhancer(repo: Path, weights: Path, device: str):
    repo_modules(repo)
    from model.detail_enhance_unet import UNet_Enhancer

    enhancer = UNet_Enhancer(n_channels=6, n_classes=3, output_confidence=False, output_shadow_mask=False).to(device)
    state = torch.load(weights / "opendelight" / "unet_enhancer.pth", map_location=device)
    enhancer.load_state_dict(state, strict=True)
    return enhancer.eval()


class Models:
    """What this job works with (the models themselves stay loaded between jobs)."""

    def __init__(self, repo: Path, weights: Path, enhancer: bool, device: str = "cuda") -> None:
        self.device = device
        for name in ("base_delight_network.pth", "unet_enhancer.pth"):
            if not (weights / "opendelight" / name).is_file():
                fail("E-OPENDELIGHT-NOWEIGHTS", name=name)
        self.matting = load_matting(repo, weights)
        self.face_detector, self.landmark_detector, self.face_parser = load_face_models(weights, device)
        self.enc_type, self.model = load_delight(repo, weights, device)
        self.enhancer = load_enhancer(repo, weights, device) if enhancer else None
        repo_modules(repo)
        from synthetic.align_utils import align_face, inverse_align_face

        self.align_face = align_face
        self.inverse_align_face = inverse_align_face


# ----------------------------------------------------------------------------- landmarks


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detect_landmarks(models: Models, bgr: np.ndarray, prev_box: np.ndarray | None):
    """68 landmarks of one face: the one overlapping the previous frame's face, else the largest."""
    faces = models.face_detector(bgr, rgb=False)
    if len(faces) == 0:
        return None, None, 0
    areas = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1])
    pick = int(np.argmax(areas))
    if prev_box is not None:
        ious = [box_iou(f[:4], prev_box) for f in faces]
        if max(ious) > 0.3:
            pick = int(np.argmax(ious))
    landmarks, _ = models.landmark_detector(bgr, faces[pick : pick + 1], rgb=False)
    return landmarks[0].astype(np.float64), faces[pick, :4].astype(np.float64), len(faces)


def face_size(lm: np.ndarray) -> float:
    return float(np.ptp(lm, axis=0).max())


def thumbnail(bgr: np.ndarray) -> np.ndarray:
    import cv2

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def smooth_landmarks(frames: list[int], lms: dict[int, np.ndarray], sizes: dict[int, tuple],
                     thumbs: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], int]:
    """Returns the smoothed landmarks and how many frames were smoothed."""
    from scipy.ndimage import gaussian_filter1d

    diffs = np.diff(frames)
    step = int(diffs[diffs > 0].min()) if (diffs > 0).any() else 1  # e.g. 2 for a 1001-1100x2 range
    runs, run = [], []
    for f in frames:
        if f not in lms:
            if run:
                runs.append(run)
            run = []
            continue
        if run:
            p = run[-1]
            jump = np.linalg.norm(lms[f] - lms[p], axis=1).mean() / max(face_size(lms[p]), 1.0)
            cut = float(np.abs(thumbs[f] - thumbs[p]).mean()) > CUT_THRESHOLD
            if f - p != step or sizes[f] != sizes[p] or jump > MAX_JUMP or cut:
                runs.append(run)
                run = []
        run.append(f)
    if run:
        runs.append(run)
    out, smoothed = dict(lms), 0
    for run in runs:
        if len(run) < 3:
            continue
        seq = np.stack([lms[f] for f in run])  # [T,68,2]
        seq = gaussian_filter1d(seq, SMOOTH_SIGMA, axis=0, mode="nearest")
        out.update({f: seq[i] for i, f in enumerate(run)})
        smoothed += len(run)
    return out, smoothed


# ----------------------------------------------------------------------------- one frame


def face_matte(models: Models, bgr: np.ndarray, lm: np.ndarray) -> np.ndarray:
    """test.py mat_image + skin_mask_image: DAViD matte x (face parsing without hair), uint8 [H,W]."""
    fg = models.matting.estimate_foreground_segmentation(bgr)
    mat = (fg * 255.0).astype(np.uint8)  # as test.py writes it to disk

    kps = np.stack([
        lm[36:42].mean(axis=0),  # left eye
        lm[42:48].mean(axis=0),  # right eye
        lm[30],  # nose tip
        lm[48],  # mouth corners
        lm[54],
    ]).astype(np.float32)
    rgb = torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1])).permute(2, 0, 1)[None].to(models.device)
    with torch.inference_mode():
        faces = models.face_parser(rgb, {
            "points": torch.from_numpy(kps)[None].to(models.device),
            "image_ids": torch.tensor([0], device=models.device),
        })
        seg = faces["seg"]["logits"].softmax(dim=1)[0].argmax(dim=0)
        face_mask = (seg >= 1).float() - (seg == LAPA_HAIR).float()
        mat_t = torch.from_numpy(mat).to(models.device).float() / 255.0 * face_mask
        # torchvision save_image rounding, as test.py stores the masked matte
        mat = (mat_t * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return mat


def bgr_to_torch(img: np.ndarray, device: str) -> torch.Tensor:
    """test.py cv2_to_torch: HWC BGR (0..255) -> [1,3,H,W] RGB 0..1."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return t[None, [2, 1, 0]].to(device)


@torch.no_grad()
def delight(models: Models, bgr: np.ndarray, lm: np.ndarray, mat: np.ndarray, resolution: int):
    """test.py align_image + inference + inverse alignment. Returns the base colour [H,W,3], alpha [H,W]
    (float32 0..1). 基础色 = 去掉光照的漫反射颜色，CG 流程里也叫 albedo，项目里一律叫
    basecolor（lab2shot/nodes/kit/ports.py basecolor_port）。"""
    h, w = bgr.shape[:2]
    aligned_face, H = models.align_face(bgr.astype(np.float32), output_size=resolution, lm=lm, border_scale=BORDER_SCALE)
    aligned_face = aligned_face.astype(np.uint8)
    mask3 = np.repeat(mat[:, :, None], 3, axis=2).astype(np.float32)  # cv2.imread of the matte PNG
    aligned_mask, _ = models.align_face(mask3, output_size=resolution, lm=lm, border_scale=BORDER_SCALE)

    face_t = bgr_to_torch(aligned_face, models.device)
    mask_t = bgr_to_torch(aligned_mask, models.device)
    inp = face_t * mask_t
    inp_512 = torch.nn.functional.interpolate(inp, size=(NET_SIZE, NET_SIZE), mode="bicubic")
    data_source = torch.tensor([DATA_IDX]) if models.enc_type == "mae_mix" else None
    out = models.model(inp_512, data_source)
    out = torch.nn.functional.interpolate(out, size=(resolution, resolution), mode="bicubic")
    if models.enhancer is not None:
        out = models.enhancer(torch.cat([out, inp], dim=1))["diffuse"]

    basecolor_crop = out[0].permute(1, 2, 0).float().cpu().numpy()  # RGB, before any 8-bit quantization
    basecolor = models.inverse_align_face(np.ascontiguousarray(basecolor_crop), H, (w, h))
    basecolor = np.clip(basecolor, 0.0, 1.0).astype(np.float32)

    # alpha: warped-back matte x the area the crop covers (test.py final_mask)
    back = models.inverse_align_face(np.ascontiguousarray(aligned_mask[:, :, 0]), H, (w, h))
    covered = models.inverse_align_face(np.ones((resolution, resolution), np.float32), H, (w, h)) > 0
    alpha = (np.clip(back, 0.0, 255.0) / 255.0 * covered).astype(np.float32)
    return basecolor, alpha


# ----------------------------------------------------------------------------- job


def device_used_mb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2**20


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "OpenDelight")
    job, params = run.job, run.params
    use_enhancer, resolution, smooth = params["enhancer"], params["resolution"], params["smooth_landmarks"]
    frames = run.frames().pairs

    torch.cuda.init()
    base_used = device_used_mb()
    peak_used = base_used
    repo = job.repo_dir
    weights = job.weights_dir
    os.chdir(repo)  # the repo's modules append "." / ".." to sys.path

    models = run.model("load_model", Models, repo, weights, use_enhancer, stage_params={"model": "OpenDelight"})

    run.stage("detect_faces")
    lms: dict[int, np.ndarray] = {}
    sizes: dict[int, tuple] = {}
    thumbs: dict[int, np.ndarray] = {}
    prev_box = None
    many = []
    for _n, (frame, path) in run.each(frames, "detect_faces"):
        bgr = read_frame(path, order="bgr")
        sizes[frame] = bgr.shape[:2]
        thumbs[frame] = thumbnail(bgr)
        lm, box, count = detect_landmarks(models, bgr, prev_box)
        prev_box = box
        if lm is not None:
            lms[frame] = lm
            if count > 1:
                many.append(frame)
    missing = [f for f, _ in frames if f not in lms]
    if missing:
        say("W-OPENDELIGHT-NOFACE", count=len(missing), frames=missing[:20])
    if many:
        say("W-OPENDELIGHT-MANYFACES", count=len(many))
    smoothed = 0
    if smooth and len(lms) > 2:
        lms, smoothed = smooth_landmarks([f for f, _ in frames], lms, sizes, thumbs)

    run.stage("delight")
    delight_started = time.time()
    raw = job.raw_dir
    with Writer(threads=2, max_pending=8) as writer:
        for _n, (frame, path) in run.each(frames, "delight"):
            t = run.frame_started()  # only frames with a face count towards the time per frame
            if frame in lms:
                bgr = read_frame(path, order="bgr")
                mat = face_matte(models, bgr, lms[frame])
                basecolor, alpha = delight(models, bgr, lms[frame], mat, resolution)
                run.frame_done(t)
            else:
                h, w = sizes[frame]
                basecolor, alpha = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
            peak_used = max(peak_used, device_used_mb())
            # compressed: outside the face crop everything is zero
            writer.npz(raw / f"frame_{frame}.npz", 6, basecolor=basecolor, alpha=alpha)

    run.finish(
        [f for f, _ in frames],
        kind="opendelight",
        faces_found=len(lms),
        missing_frames=missing,
        enhancer=use_enhancer,
        resolution=resolution,
        smooth_landmarks=smooth,
        smoothed_frames=smoothed,  # frames inside continuous shots (photos / cuts are left alone)
        basecolor="display-referred sRGB-encoded diffuse base colour (albedo), float32 0..1, input frame size",
        seconds_frames=round(time.time() - delight_started, 1),  # the 去光照 stage, writing included
        # whole process (torch + onnxruntime), sampled after every frame; excludes the CUDA context itself
        # (overrides the standard gpu_peak_mb, which is torch's reserved memory only)
        gpu_peak_mb=round(peak_used - base_used),
        gpu_peak_torch_mb=round(torch.cuda.max_memory_allocated() / 2**20),
        models=["RetinaFace R50 (ibug)", "FAN 2DFAN2-alt (ibug)", "DAViD foreground ViT-L", "FaRL LaPa 448",
                "OpenDelight base (MAE ViT-B + Sapiens decoder)"] + (["OpenDelight UNet enhancer"] if use_enhancer else []),
    )


if __name__ == "__main__":
    serve(main)
