"""SDMatte worker: interactive image matting, one frame at a time. Runs inside third_party/sdmatte/.venv with the
pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Contract: lab2shot_worker/matte.py. SDMatte is a Stable-Diffusion U-Net turned into a matting network: it takes the
frame plus one **visual prompt** saying which thing to matte, and predicts alpha in a single forward pass (no
denoising loop, `num_inference_steps=1`). No trimap. The 「粗遮罩」 input is the prompt, used either as the mask itself
（提示 = 遮罩）or as its bounding box（提示 = 框）, exactly as upstream's data pipeline builds them.

Upstream's test pipeline (data/dataset.py, phase="test", and inference.py):

* the frame is resized to a square (1024x1024 upstream) and scaled to -1..1; the prompt the same way, nearest;
* the prompt's bounding box, normalised 0..1, goes in as the coordinate embedding;
* `is_trans` says whether the subject is a transparent object (glass, smoke, veils) -- it drives the model's
  opacity embedding, and upstream sets it per dataset, so here the artist says it（半透明主体）;
* the alpha comes out at the square size and is resized back to the frame.

Upstream builds the model through detectron2's LazyConfig and loads the checkpoint with DetectionCheckpointer.
Neither is used here: the model class is imported straight from the repo file, and the checkpoint is a plain torch.save whose
["model"] is the state dict -- so detectron2 (and the training-only utils it drags in) is not installed at all.

Output: raw/frame_<n>.npz alpha float32 [H,W] 0..1 at the input resolution. SDMatte predicts alpha only
(no foreground colour).
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # transformers / diffusers read the weights folder, never the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402  (after the offline switches: transformers reads them when it is imported)

from lab2shot_worker import MemoryBound, fail, progress, resident, serve  # noqa: E402
from lab2shot_worker.matte import GUIDE_THRESHOLD, GuideMasks, Output, frame_reader  # noqa: E402
from lab2shot_worker.run import Run  # noqa: E402

MODEL_DIR = "SDMatte"  # weights/SDMatte/: SDMatte.pth beside the diffusers configs it is built from
CHECKPOINT = "SDMatte.pth"
# upstream's released configs/SDMatte.py hy_dict.model_kwargs, with load_weight False: the shapes come from the
# configs in the weights folder, every number from SDMatte.pth
MODEL_KWARGS = dict(
    conv_scale=3, num_inference_steps=1, load_weight=False, add_noise=False, use_dis_loss=True, use_aux_input=True,
    use_coor_input=True, use_attention_mask=True, residual_connection=False, use_encoder_hidden_states=True,
    use_attention_mask_list=[True, True, True], use_encoder_hidden_states_list=[False, True, False],
)
# what the memory grows with: 处理分辨率 within the node's settings (1024: 14.5 GB)
SIZE = MemoryBound.parameter("resolution", (1024, 768, 512))


def _import_upstream(repo: Path):
    """The repo's model class, without running the two package __init__ files that pull in training-only packages.

    `modeling/__init__.py` imports LiteSDMatte as well, and `utils/__init__.py` imports detectron2's training hooks
    and scikit-image's metric port. Neither is used to run the model, and detectron2 has to be compiled, so the
    `utils` package is built here with its two inference modules only and the model file is loaded on its own."""
    package = importlib.util.module_from_spec(importlib.machinery.ModuleSpec("utils", None, is_package=True))
    package.__path__ = [str(repo / "utils")]
    sys.modules["utils"] = package
    for name in ("replace", "utils"):  # utils/utils.py does `from .replace import ...`
        sys.modules[f"utils.{name}"] = importlib.import_module(f"utils.{name}")
    for name in ("replace_unet_conv_in", "replace_attention_mask_method", "add_aux_conv_in"):
        setattr(package, name, getattr(sys.modules["utils.utils"], name))
    package.replace = sys.modules["utils.replace"]

    spec = importlib.util.spec_from_file_location("sdmatte_meta_arch", repo / "modeling" / "SDMatte" / "meta_arch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SDMatte


@resident
def load_model(repo: Path, weights: Path, aux_input: str, device: torch.device):
    net = _import_upstream(repo)(pretrained_model_name_or_path=str(weights / MODEL_DIR), aux_input=aux_input,
                                 **MODEL_KWARGS)
    # mmap: the file also carries the trainer's optimiser state (about half of its 12 GB), which is never
    # read — memory-mapped storages keep it out of RAM instead of materialising the whole checkpoint
    state = torch.load(weights / MODEL_DIR / CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)["model"]
    own = net.state_dict()
    missing = [k for k in own if k not in state]
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="SDMatte", extension="sdmatte", model=CHECKPOINT,
             missing=len(missing), unexpected=len([k for k in state if k not in own]), examples=list(missing[:3]))
    net.load_state_dict(state)
    return net.to(device).eval()


def box_of(mask: np.ndarray) -> np.ndarray:
    """The prompt's bounding box as normalised (x_min, y_min, x_max, y_max), upstream's GenMask/GenBBox coordinate
    embedding. An empty prompt is upstream's (0, 0, 1, 1): the whole frame."""
    height, width = mask.shape
    where = np.argwhere(mask > 0)
    if where.size == 0:
        return np.array([0.0, 0.0, 1.0, 1.0], np.float32)
    (y_min, x_min), (y_max, x_max) = where.min(axis=0), where.max(axis=0)
    return np.array([x_min / width, y_min / height, x_max / width, y_max / height], np.float32)


class Matting:
    """One SDMatte pass over a frame, at `side` x `side` like upstream's test transform."""

    def __init__(self, net, device: torch.device, side: int, guide: str, transparent: bool, fp16: bool):
        self.net, self.device, self.side, self.guide = net, device, side, guide
        self.dtype = torch.float16 if fp16 else torch.float32
        self.is_trans = torch.tensor([1 if transparent else 0], device=device).long()

    def __call__(self, rgb: np.ndarray, guide: np.ndarray) -> np.ndarray:
        """rgb uint8 [H,W,3] and a 0..1 guide [H,W] -> alpha float32 [H,W]."""
        square = (self.side, self.side)
        image = cv2.resize(rgb, square, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        mask = cv2.resize((guide > GUIDE_THRESHOLD).astype(np.float32), square, interpolation=cv2.INTER_NEAREST)
        if self.guide == "box":  # upstream GenBBox: the box the prompt spans, filled
            box = box_of(mask)
            mask = np.zeros(square, np.float32)
            x0, y0, x1, y1 = (box * self.side).astype(int)
            mask[y0:y1, x0:x1] = 1.0
        coords = torch.from_numpy(box_of(mask))[None].to(self.device)
        # upstream Normalize(): every plane goes to -1..1
        data = {
            "image": torch.from_numpy(image.transpose(2, 0, 1))[None].to(self.device) * 2 - 1,
            "mask_coords" if self.guide == "mask" else "bbox_coords": coords,
            "mask" if self.guide == "mask" else "bbox_mask":
                torch.from_numpy(mask)[None, None].to(self.device) * 2 - 1,
            "is_trans": self.is_trans,
        }
        with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype is torch.float16):
            alpha = self.net(data).float()
        alpha = alpha.flatten(0, 2).clamp_(0, 1).cpu().numpy()
        return cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def main(job_path: str) -> None:
    run = Run.start(job_path, "sdmatte.matte", "SDMatte")
    job, params = run.job, run.params
    guide, side, transparent, fp16 = params["guide"], params["resolution"], params["transparent"], params["fp16"]
    weights = job.weights_dir
    run.weights(weights / MODEL_DIR / CHECKPOINT, weights / MODEL_DIR / "unet" / "config.json")

    frames = run.frames()
    numbers, height, width = frames.numbers, frames.height, frames.width
    guides = GuideMasks(job, height, width)
    guides.missing(numbers)

    device = torch.device("cuda")
    # the aux input is part of how the network is wired, so a changed 提示 loads the model again
    net = run.model("load_model", load_model, job.repo_dir, weights, "mask" if guide == "mask" else "bbox_mask", device,
                    stage_params={"model": "SDMatte"})

    run.stage("matte")
    reader = frame_reader(frames.paths, (height, width))

    def matte_shot(value: int):
        """The whole shot, frame by frame, each at value x value."""
        matting = Matting(net, device, value, guide, transparent, fp16)
        out = Output(job)
        try:
            with torch.inference_mode():
                for i, frame in enumerate(numbers):
                    mask = guides.get(frame)
                    mask = np.zeros((height, width), np.float32) if mask is None else mask
                    out.put(frame, matting(reader.get(i, list(range(i, len(frames)))), mask))
                    progress(i + 1, len(frames), "matte")
        except BaseException:
            out.close()  # its writing thread, not kept for a run that failed
            raise
        return value, out

    try:
        side, out = run.fit(SIZE, matte_shot, side)
    finally:
        reader.close()

    out.finish(
        run,
        numbers,
        model="SDMatte (vivoCameraResearch/SDMatte, SDMatte.pth)",
        guide=f"the coarse mask as a {guide} prompt",
        width=width,
        height=height,
        processing_size={"width": side, "height": side},
        params={"guide": guide, "resolution": side, "transparent": transparent, "fp16": fp16},
        foreground=False,
    )


if __name__ == "__main__":
    serve(main)
