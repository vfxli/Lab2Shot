"""The DUSt3R family's model input, shared by the CUT3R and MonST3R workers (MonST3R requires this extension:
Extension.requires puts this folder on its worker's PYTHONPATH). Needs numpy, OpenCV and Pillow; never imports
Lab2Shot core.

  * frames -> model input: long side resized to `resolution`, centre crop to multiples of 16 (exactly what
    DUSt3R-family loaders do), and the way back (intrinsics, maps at the input resolution);
  * the stitched frames (lab2shot_worker.recon: chunks, stitching) -> the raw `reconstruction`
    contract at the input resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from lab2shot_worker import recon

POINTS = "points"  # the one extra map that is 3D coordinates, not a selection (write_outputs)


# ---------------------------------------------------------------------- model input geometry


@dataclass(frozen=True)
class Geometry:
    """Input frame (W x H) -> resized (rw x rh) -> centre crop (w x h) at (x0, y0)."""

    width: int
    height: int
    rw: int
    rh: int
    x0: int
    y0: int
    w: int
    h: int

    @classmethod
    def for_size(cls, width: int, height: int, resolution: int) -> Geometry:
        # DUSt3R / CUT3R / MonST3R load_images(size=512): long side -> size (rounded),
        # then crop around the centre to multiples of 16 (the ViT patch size).
        s = resolution / max(width, height)
        rw, rh = int(round(width * s)), int(round(height * s))
        cx, cy = rw // 2, rh // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if rw == rh:  # upstream: square frames become 4:3 (square_ok=False)
            halfh = 3 * halfw // 4
        return cls(width, height, rw, rh, cx - halfw, cy - halfh, 2 * halfw, 2 * halfh)

    @property
    def sx(self) -> float:
        return self.rw / self.width

    @property
    def sy(self) -> float:
        return self.rh / self.height

    def image(self, rgb: np.ndarray) -> np.ndarray:
        """uint8 [H,W,3] -> uint8 [h,w,3], resampled like upstream (PIL Lanczos down / bicubic up)."""
        img = Image.fromarray(rgb)
        interp = Image.LANCZOS if max(self.width, self.height) > max(self.rw, self.rh) else Image.BICUBIC
        img = img.resize((self.rw, self.rh), interp)
        return np.array(img.crop((self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)))

    def mask(self, mask: np.ndarray) -> np.ndarray:
        """bool [H,W] -> bool [h,w] (area average > 0.5)."""
        m = cv2.resize(mask.astype(np.float32), (self.rw, self.rh), interpolation=cv2.INTER_AREA)
        return m[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w] > 0.5

    def intrinsics(self, focal: float, pp: tuple[float, float]) -> np.ndarray:
        """Model focal / principal point -> K at the input resolution, pixel-corner convention.
        The models put the principal point at the centre of their (centre-cropped) input,
        w/2 x h/2; read as corner coordinates that is exactly the frame centre (W/2, H/2).
        (DUSt3R's own pixel grid has pixel centres at integers, half a model pixel away:
        ignored, below the models' accuracy.)"""
        cx = (pp[0] + self.x0) / self.sx
        cy = (pp[1] + self.y0) / self.sy
        return np.array([[focal / self.sx, 0, cx], [0, focal / self.sy, cy], [0, 0, 1]], np.float64)

    def focal_to_model(self, focal_input_px: float) -> float:
        return focal_input_px * 0.5 * (self.sx + self.sy)

    def to_input(self, arr: np.ndarray, nearest: bool = False, fill: float | None = None) -> np.ndarray:
        """[h,w] model map -> [H,W]. Outside the crop: edge values (or `fill`)."""
        pad = (self.y0, self.rh - self.h - self.y0, self.x0, self.rw - self.w - self.x0)
        a = arr.astype(np.float32)
        if fill is None:
            a = cv2.copyMakeBorder(a, *pad, cv2.BORDER_REPLICATE)
        else:
            a = cv2.copyMakeBorder(a, *pad, cv2.BORDER_CONSTANT, value=fill)
        return cv2.resize(a, (self.width, self.height), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)

    def describe(self) -> dict:
        return {"model_input": [self.w, self.h], "resized": [self.rw, self.rh], "crop_offset": [self.x0, self.y0]}


# ---------------------------------------------------------------------- reading a shot in


def read_images(used: list[tuple[int, Path]], geo: Geometry, read_frame, progress) -> list[np.ndarray]:
    """A shot read in at the network's size, reporting progress every 16 frames (stage read_frames: an extension using this
    module defines stage.read_frames in its own catalogue)."""
    images = []
    for i, (_, path) in enumerate(used):
        images.append(geo.image(read_frame(path)))
        if (i + 1) % 16 == 0 or i + 1 == len(used):
            progress(i + 1, len(used), "read_frames")
    return images


def moving_at_model_size(job, frames: list[int], geo: Geometry, width: int, height: int) -> np.ndarray:
    """可选的「运动物体遮罩」输入缩到网络尺寸：[帧数, h, w] 的布尔数组，True = 这个像素上有在动的东西。

    没接这个输入、或者某一帧没有遮罩时，那一帧整帧 False。

    只有 MonST3R 用它：MonST3R 的上游吃运动物体遮罩（demo.py:107 dynamic_mask_path），所以它的节点
    保留着那个输入口；CUT3R 的上游 demo.py 不吃遮罩，节点没有这个口，不调这里。这两个函数是 MonST3R 的
    依赖，不要因为 CUT3R 不用就删。
    """
    moving_in = recon.MovingMasks(job, width, height)
    out = np.zeros((len(frames), geo.h, geo.w), bool)
    for i, f in enumerate(frames):
        if f in moving_in:
            out[i] = geo.mask(moving_in.get(f))
    return out


# ---------------------------------------------------------------------- outputs


def model_K(focal: np.ndarray, pp: np.ndarray) -> np.ndarray:
    """[n] focal + [n,2] principal point (model pixels, DUSt3R grid) -> [n,3,3] K for recon.Chunk."""
    K = np.zeros((len(focal), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = focal
    K[:, :2, 2] = pp
    K[:, 2, 2] = 1.0
    return K


def write_outputs(raw: Path, frames: list[int], geo: Geometry, stitched: list[recon.Frame], focal: np.ndarray,
                  progress=None) -> dict:
    """raw/cameras.npz + raw/frame_<n>.npz at the input resolution from the stitched frames; `focal` [n] in model
    pixels (per frame, or one shared value). Extra maps are written too: `points` (the model's own three-channel
    point map, CUT3R) goes out as float, every other one is a selection and goes out as bool (MonST3R's `moving`).
    Returns recon.summary numbers."""
    c2w = np.stack([f.cam_to_world for f in stitched])
    K = np.stack([geo.intrinsics(float(fo), (float(f.K[0, 2]), float(f.K[1, 2]))) for fo, f in zip(focal, stitched)])
    recon.save_cameras(raw, frames, K, c2w, geo.width, geo.height)
    inside = geo.to_input(np.ones((geo.h, geo.w), np.float32), nearest=True, fill=0.0) > 0.5
    depth_median = []
    for i, (frame, f) in enumerate(zip(frames, stitched)):
        d = geo.to_input(f.depth)
        m = (geo.to_input(f.usable.astype(np.float32)) > 0.5) & inside & np.isfinite(d) & (d > 0)
        # 点图和深度走同一条边界规则（裁切外沿用边缘值），不然点云里会多出一圈落在原点的假点：
        # 深度在裁切外是复制边缘、并且是有效的（家族按 `isfinite 且 > 0` 判有效），点图要跟着它
        extra = {k: (np.stack([geo.to_input(v[..., c]) for c in range(v.shape[-1])], axis=-1)
                     if k == POINTS else geo.to_input(v.astype(np.float32), fill=0.0) > 0.5)
                 for k, v in f.extra.items()}
        recon.save_frame(raw, frame, d, geo.to_input(f.confidence), m, **extra)
        depth_median.append(float(np.median(d[m])) if m.any() else float("nan"))
        if progress:
            progress(i + 1, len(frames), "write_depth")
    return recon.summary(c2w, K, geo.width, depth_median)
