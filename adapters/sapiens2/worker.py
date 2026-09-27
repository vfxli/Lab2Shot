"""Meta Sapiens2 worker: body-part segmentation / surface normals per frame. Runs
inside third_party/sapiens2/.venv with the pinned repo on sys.path; never imports
Lab2Shot core.

    python worker.py <job.json>

job["node"] picks the task:

  sapiens2.segment -> raw/frame_<n>.npz
        labels  uint8   [H,W]    body-part class (raw/classes.json), 0 = background
        alpha   float32 [H,W]    person foreground 0..1: Sapiens2-1B matting model
                                 (params.matte, default) or 1 - P(background) of the
                                 segmentation model
        foreground float32 [H,W,3] the matting model's own foreground colour, premultiplied
                                 by alpha, 0..1 in the encoding of the frames it was given
                                 (only with params.matte: the segmentation model predicts
                                 no colour)
  sapiens2.normal  -> raw/frame_<n>.npz
        normal  float32 [H,W,3]  unit normals, OpenCV camera (+X right, +Y down,
                                 +Z forward); surfaces facing the camera have Z < 0.
                                 Every pixel, background included: upstream saves the
                                 unmasked prediction (vis_normal.py:122 np.save, and
                                 only the picture it draws afterwards is masked).

Sapiens2 works on 1024x768 (H x W). The whole frame goes in, as upstream's demos do (their argparse takes an image
and nothing else). Segmentation and matting stretch the frame to 1024x768; normals keep the aspect ratio and pad.
Each follows upstream's own test_pipeline (seg / matting: keep_ratio=False; normal: NormalResizePadImage), and the
result is scaled back to the full frame size. To process only one person, mask out the rest in the node graph
before this node (「人物框转遮罩」 -> 「图像相乘」); no cropping is done here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import fail, read_frame, require_weights, resident, say, serve
from lab2shot_worker.frame_io import FrameReader, Writer
from lab2shot_worker.run import Run

NODES = {"sapiens2.segment": "seg", "sapiens2.normal": "normal"}
MATTING_SIZE = "1b"  # the only matting checkpoint released
CONFIGS = {  # upstream model definitions, relative to the repo
    "seg": "sapiens/dense/configs/seg/shutterstock_goliath/sapiens2_{size}_seg_shutterstock_goliath-1024x768.py",
    "normal": "sapiens/dense/configs/normal/metasim_render_people/sapiens2_{size}_normal_metasim_render_people-1024x768.py",
    "matting": "sapiens/dense/configs/matting/gss_p3m_metasim/sapiens2_{size}_matting_gss_p3m_metasim-1024x768.py",
}
IN_H, IN_W = 1024, 768  # every Sapiens2 task model is trained at this size
ASPECT = IN_W / IN_H
# ImageNet statistics on 0..255 RGB, upstream's ImagePreprocessor.
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
# Sapiens2 predicts normals with +X right, +Y up, +Z towards the viewer (OpenGL
# camera; checked on silhouettes: the top of the head points +Y, a front-facing
# torso +Z). OpenCV camera = flip Y and Z.
TO_OPENCV = (1.0, -1.0, -1.0)
CONVENTION = ("OpenCV camera: +X right, +Y down, +Z forward; unit normals, surfaces facing the camera have Z < 0 "
              "(converted from Sapiens2's +X right, +Y up, +Z towards the viewer)")
# Left/Right in class names: the person's own (anatomical) side.
CLASSES_ZH = {
    "Background": "背景", "Apparel": "服饰配件", "Eyeglass": "眼镜", "Face_Neck": "脸和脖子", "Hair": "头发",
    "Left_Foot": "左脚", "Left_Hand": "左手", "Left_Lower_Arm": "左小臂", "Left_Lower_Leg": "左小腿",
    "Left_Shoe": "左鞋", "Left_Sock": "左袜", "Left_Upper_Arm": "左大臂", "Left_Upper_Leg": "左大腿",
    "Lower_Clothing": "下装", "Right_Foot": "右脚", "Right_Hand": "右手", "Right_Lower_Arm": "右小臂",
    "Right_Lower_Leg": "右小腿", "Right_Shoe": "右鞋", "Right_Sock": "右袜", "Right_Upper_Arm": "右大臂",
    "Right_Upper_Leg": "右大腿", "Torso": "躯干", "Upper_Clothing": "上装", "Lower_Lip": "下嘴唇",
    "Upper_Lip": "上嘴唇", "Lower_Teeth": "下牙", "Upper_Teeth": "上牙", "Tongue": "舌头",
}

@resident
def load_model(repo: Path, weights: Path, task: str, size: str, device: torch.device, dtype: torch.dtype):
    from safetensors.torch import load_file
    from sapiens.engine.config import Config
    from sapiens.registry import MODELS

    checkpoint = weights / task / f"sapiens2_{size}_{task}.safetensors"
    require_weights("sapiens2", checkpoint, what=f" Sapiens2 {task} {size} 权重")
    cfg = Config.fromfile(str(repo / CONFIGS[task].format(size=size)))
    cfg.model["backbone"].pop("init_cfg", None)  # never load the pretraining backbone
    # Built without memory (random init of a 1B model on the CPU takes ~10 s), then
    # every parameter and buffer is taken from the checkpoint: it holds all of them.
    with torch.device("meta"):
        model = MODELS.build(cfg.model)
    missing, unexpected = model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=False, assign=True)
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="Sapiens2", extension="sapiens2", model=checkpoint.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    if unexpected:
        say("I-WORKER-UNUSEDWEIGHTS", project="Sapiens2", model=checkpoint.name, count=len(unexpected),
            examples=list(unexpected[:3]))
    return model.to(device=device, dtype=dtype).eval()

def full_frame_region(width: int, height: int) -> tuple[int, int, int, int]:
    """The full frame padded to 3:4. Used for normals and albedo: upstream `NormalResizePadImage` /
    `AlbedoResizePadImage` keep the aspect ratio and pad."""
    cx, cy = width / 2, height / 2
    w, h = float(width), float(height)
    if w / h > ASPECT:
        h = w / ASPECT
    else:
        w = h * ASPECT
    rx, ry = int(round(cx - w / 2)), int(round(cy - h / 2))
    return rx, ry, rx + max(int(round(w)), 3), ry + max(int(round(h)), 4)


def stretched_frame_region(width: int, height: int) -> tuple[int, int, int, int]:
    """The full frame without padding; the subsequent resize stretches it to 1024x768.

    Used for segmentation and matting, as upstream does:
    the test_pipeline of `configs/seg/.../sapiens2_1b_seg_...py:201` and `configs/matting/.../sapiens2_1b_matting_...py:206`
    sets `keep_ratio=False` (in `SegResize` / `MattingResize`, False means
    `new_w, new_h = target_width, target_height`, a full stretch).
    With padding, a 1920x1080 frame would occupy only 432 of the 1024 rows, a 2.4x loss of effective resolution."""
    return 0, 0, int(width), int(height)

def read_rgb255(path: Path) -> np.ndarray:
    """float32 [H,W,3] RGB on the 0..255 scale the model normalises from (16-bit precision kept)."""
    return read_frame(path, "float32") * 255.0

class Region:
    """The region the full frame occupies in 1024x768 under this task's test_pipeline. The region may be larger than
    the frame (normals are padded), so the frame's location within it is recorded to map the result back."""

    def __init__(self, rect: tuple[int, int, int, int], width: int, height: int):
        self.x0, self.y0, self.x1, self.y1 = rect
        self.w, self.h = self.x1 - self.x0, self.y1 - self.y0
        # frame slice covered by the region, and the same pixels in region coordinates
        fx0, fy0 = max(self.x0, 0), max(self.y0, 0)
        fx1, fy1 = min(self.x1, width), min(self.y1, height)
        self.frame = (slice(fy0, fy1), slice(fx0, fx1))
        self.crop = (slice(fy0 - self.y0, fy1 - self.y0), slice(fx0 - self.x0, fx1 - self.x0))

def make_batch(image: torch.Tensor, r: Region, dtype: torch.dtype) -> torch.Tensor:
    """image [3,H,W] 0..255 on the GPU -> the normalized [1,3,1024,768] the model takes (black where the padded
    region falls outside the frame, as upstream pads)."""
    mean = torch.tensor(MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(STD, device=image.device).view(3, 1, 1)
    canvas = image.new_zeros((3, r.h, r.w))
    canvas[:, r.crop[0], r.crop[1]] = image[:, r.frame[0], r.frame[1]]
    scaled = F.interpolate(canvas[None], size=(IN_H, IN_W), mode="bilinear", align_corners=False,
                           antialias=r.h > IN_H)[0]
    return ((scaled - mean) / std)[None].to(dtype)

def run_model(model, image: torch.Tensor, r: Region, dtype: torch.dtype) -> torch.Tensor:
    """The model's [C,1024,768] output in float32."""
    with torch.inference_mode():
        return model(make_batch(image, r, dtype)).float()[0]

def to_frame(out: torch.Tensor, r: Region) -> torch.Tensor:
    """[C,1024,768] model output -> [C,h,w] over the part of the region inside the frame."""
    full = F.interpolate(out[None], size=(r.h, r.w), mode="bilinear", align_corners=False,
                         antialias=r.h < IN_H)[0]
    return full[:, r.crop[0], r.crop[1]]

def paste(shape: tuple[int, int], values: torch.Tensor, r: Region) -> torch.Tensor:
    """[C,h,w] over the region's part of the frame -> [C,H,W] over the whole frame (0 outside it: the padded
    region never reaches past the picture, so this only matters if a rounding leaves a pixel over)."""
    out = values.new_zeros((values.shape[0], *shape))
    out[:, r.frame[0], r.frame[1]] = values
    return out

def main(job_path: str) -> None:
    run = Run.start(job_path, tuple(NODES), "Sapiens2")
    job, params = run.job, run.params
    node = job.node
    task = NODES[node]
    size, fp16 = params["model_size"], params["fp16"]
    matte = task == "seg" and params["matte"]

    repo, weights = job.repo_dir, job.weights_dir
    sys.path.insert(0, str(repo))
    device = torch.device("cuda")
    dtype = torch.bfloat16 if fp16 else torch.float32  # bf16: what upstream trained with (fp16 overflows in the 5B)

    frames = run.frames()
    raw = job.raw_dir

    # each task runs only its own model: the normal task does not run segmentation to mask out non-person normals
    # (upstream vis_normal.py saves the unmasked full-frame normals: :122 np.save precedes the masking at :125)
    def load_models():
        seg = load_model(repo, weights, "seg", size, device, dtype) if task == "seg" else None
        normal = load_model(repo, weights, "normal", size, device, dtype) if task == "normal" else None
        matting = load_model(repo, weights, "matting", MATTING_SIZE, device, dtype) if matte else None
        return seg, normal, matting

    seg_model, normal_model, matting_model = run.model("Sapiens2 模型", load_models)

    if task == "seg":
        from sapiens.dense.datasets.seg.seg_utils import DOME_CLASSES_29

        classes = [{"index": i, "name": c["name"], "name_zh": CLASSES_ZH.get(c["name"], c["name"]),
                    "color": list(c["color"])} for i, c in DOME_CLASSES_29.items()]
        (raw / "classes.json").write_text(json.dumps(classes, indent=2, ensure_ascii=False), encoding="utf-8")

    run.stage({"seg": "逐帧分割身体部位", "normal": "逐帧估计法线"}[task])
    coverage: list[float] = []
    with FrameReader(frames.paths, read_rgb255, threads=2, ahead=2) as reader, \
            Writer(threads=2, max_pending=8) as writer:
        for i, (frame, _) in run.each(frames.pairs, "Sapiens2"):
            rgb = reader.get(i)
            h, w = rgb.shape[:2]
            started = run.frame_started()  # this frame's time, up to before the file is written
            image = torch.from_numpy(rgb).to(device).permute(2, 0, 1)
            if task == "seg":
                # segmentation / matting: full frame stretched to 1024x768 (upstream test_pipeline keep_ratio=False)
                r = Region(stretched_frame_region(w, h), w, h)
                logits = to_frame(run_model(seg_model, image, r, dtype), r)
                labels = paste((h, w), logits.argmax(0, keepdim=True).to(torch.uint8), r)[0]
                if matte:  # [fg rgb, alpha]: the model gives both, both are kept
                    got = paste((h, w), to_frame(run_model(matting_model, image, r, dtype), r).clamp(0, 1), r)
                    alpha, fg = got[3], got[:3]
                else:  # alpha without the matting model: the segmentation model's 'not background'
                    alpha = paste((h, w), (1.0 - logits.softmax(0)[:1]), r)[0]
                    fg = None
                result = {"labels": labels.cpu().numpy(),
                          "alpha": alpha.clamp(0, 1).cpu().numpy().astype(np.float32)}
                if fg is not None:
                    result["foreground"] = fg.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # HWC, like every map
                person = labels > 0
            else:
                # normals: aspect ratio kept, padded to 3:4 (upstream NormalResizePadImage). The raw upstream values
                # are delivered, background included, as upstream np.save stores them unmasked
                r = Region(full_frame_region(w, h), w, h)
                got = paste((h, w), to_frame(run_model(normal_model, image, r, dtype), r), r)
                normal = F.normalize(got, dim=0).permute(1, 2, 0) * torch.tensor(TO_OPENCV, device=device)
                result = {"normal": normal.cpu().numpy().astype(np.float32)}
                person = None
            if person is not None:
                coverage.append(float(person.float().mean()))
            run.frame_done(started)
            # segmentation is mostly 0 outside the person and compresses well (level 6); normals have values
            # everywhere (unmasked upstream values), so the fastest level (1) is used to keep writing faster than inference
            writer.npz(raw / f"frame_{frame}.npz", 6 if task == "seg" else 1, **result)

    models = [f"facebook/sapiens2-seg-{size}"] if seg_model is not None else []
    if normal_model is not None:
        models.insert(0, f"facebook/sapiens2-normal-{size}")
    if matting_model is not None:
        models.append(f"facebook/sapiens2-matting-{MATTING_SIZE}")
    info = {
        "kind": {"seg": "body_parts", "normal": "normals"}[task],
        "node": node,
        "model": models[0],
        "models": models,
        "model_size": size,
        "precision": "bf16" if fp16 else "fp32",
        "width": job.width or w,
        "height": job.height or h,
        "frames": frames.numbers,
        "crops": "full frame (seg/matting stretched to 1024x768 as upstream keep_ratio=False; "
                 "normal letterboxed to 3:4)",
        "model_input": [IN_H, IN_W],
        "person_coverage": [round(c, 5) for c in coverage] if coverage else None,
        # seconds / load_seconds / seconds_per_frame / gpu_model_mb / gpu_peak_mb / gpu_peak_allocated_mb: standard Run fields
    }
    if task == "seg":
        info.update(files="frame_<n>.npz: labels uint8 [H,W], alpha float32 [H,W]"
                          + (", foreground float32 [H,W,3] premultiplied colour" if matte else "") + "; classes.json",
                    classes="classes.json", num_classes=29,
                    alpha_source="sapiens2 matting 1b" if matte else "1 - P(background) of the segmentation model")
    else:
        info.update(files="frame_<n>.npz: normal float32 [H,W,3]", convention=CONVENTION,
                    masked="no: upstream saves the unmasked prediction (vis_normal.py:122)")
    run.finish(frames.numbers, **info)

if __name__ == "__main__":
    serve(main)
