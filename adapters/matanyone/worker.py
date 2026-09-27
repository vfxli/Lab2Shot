"""MatAnyone 2 worker: mask-guided video matting of people. Runs inside
third_party/matanyone/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Contract: lab2shot_worker/matte.py. MatAnyone 2 takes the mask of ONE
frame and propagates it with its memory (Cutie-style working memory of the last
5 memory frames + sensory memory, so GPU memory does not grow with the shot).
The anchor is the first frame whose guide mask has foreground; frames after it
are matted forwards, frames before it (if any) by a second pass backwards from
the anchor. Per upstream inference (inference_matanyone2.py):

* guide binarised (> 0.5), then dilated and eroded by `mask_close` px
  (elliptical kernel, upstream gen_dilate / gen_erosion, default 10);
* the anchor frame is repeated `warmup` times (default 10) as "first frame
  prediction" steps to refine the first alpha before propagation;
* frames scaled down (area) when the long side exceeds resolution; the mask with
  nearest neighbour; fp16 autocast like upstream's safe_autocast.

Output: raw/frame_<n>.npz alpha float32 [H,W] 0..1 at the input resolution.
MatAnyone 2 predicts alpha only (no foreground colour).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import MemoryBound, fail, fit_size, progress, require_weights, resident, say, serve
from lab2shot_worker.frame_io import FrameReader
from lab2shot_worker.matte import GUIDE_THRESHOLD, GuideMasks, Output, frame_reader
from lab2shot_worker.run import Run

MULTIPLE = 2  # the network pads to multiples of 16 itself; keep the size even
# what the memory grows with: 最大处理尺寸 within the node's 256..1920 (1920: 7.5 GB)
RESOLUTION = MemoryBound.parameter("resolution", (1920, 1280, 960, 640))


@resident
def load_model(checkpoint: Path, device: torch.device):
    """upstream utils/get_default_model.get_matanyone2_model, minus the ImageNet ResNet
    download (those layers are overwritten by the checkpoint anyway)."""
    from hydra import compose, initialize_config_module
    from omegaconf import open_dict

    from matanyone2.model.matanyone2 import MatAnyone2

    with initialize_config_module(version_base="1.3.2", config_module="matanyone2.config"):
        cfg = compose(config_name="eval_matanyone_config")
    with open_dict(cfg):
        cfg["weights"] = str(checkpoint)
        cfg.model["pretrained_resnet"] = False
    model = MatAnyone2(cfg, single_object=True).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    own = model.state_dict()
    missing = [k for k in own if k not in state]
    unexpected = [k for k in state if k not in own]
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="MatAnyone 2", extension="matanyone", model=checkpoint.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    model.load_weights(state)
    return model


class Pass:
    """One MatAnyone 2 propagation from the anchor frame through `order`."""

    def __init__(self, model, size: tuple[int, int], device: torch.device):
        from matanyone2.inference.inference_core import InferenceCore

        self.processor = InferenceCore(model, cfg=model.cfg, device=device)
        self.size, self.device = size, device

    def image(self, rgb: np.ndarray) -> torch.Tensor:
        """uint8 [H,W,3] -> float [3,h,w] 0..1 (upstream: float 0..255, area-resized, / 255)."""
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).float()
        if tuple(x.shape[-2:]) != self.size:
            x = F.interpolate(x[None], size=self.size, mode="area")[0]
        return x / 255.0

    def run(self, reader: FrameReader, order: list[int], guide: torch.Tensor, warmup: int, out_size, emit) -> None:
        """order[0] is the anchor; emit(index, alpha float32 [H,W] numpy) for every frame of `order`."""
        p = self.processor
        # upstream: [anchor] * warmup + frames; step 0 memorises the mask and predicts,
        # steps 1..warmup are "first frame prediction" again (the last of them is the
        # real anchor frame, whose output is kept); later steps propagate.
        sequence = [order[0]] * warmup + order
        for ti, index in enumerate(sequence):
            upcoming = [j for j in sequence[ti:] if j != index][: reader.ahead]
            image = self.image(reader.get(index, [index, *upcoming]))
            if ti == 0:
                p.step(image, guide, objects=[1])
                prob = p.step(image, first_frame_pred=True)
            elif ti <= warmup:
                prob = p.step(image, first_frame_pred=True)
            else:
                prob = p.step(image)
            if ti < warmup:
                continue
            alpha = p.output_prob_to_mask(prob).float()
            if tuple(alpha.shape) != tuple(out_size):
                alpha = F.interpolate(alpha[None, None], size=out_size, mode="bilinear", align_corners=False)[0, 0]
            emit(index, alpha.clamp_(0, 1).cpu().numpy())


def main(job_path: str) -> None:
    run = Run.start(job_path, "matanyone.matte", "MatAnyone 2")
    job, params = run.job, run.params
    resolution, warmup, mask_close, fp16 = params["resolution"], params["warmup"], params["mask_close"], params["fp16"]
    checkpoint = job.weights_dir / "matanyone2.pth"
    require_weights("matanyone", checkpoint)  # not run.weights: the extension is "matanyone", the project "MatAnyone 2"

    frames = run.frames()
    numbers, height, width = frames.numbers, frames.height, frames.width
    guides = GuideMasks(job, height, width)

    run.stage("读取引导遮罩")
    anchor, guide = guides.first_nonempty(numbers)
    if anchor > 0:
        say("N-MATANYONE-LATESTART", count=anchor, frame=numbers[anchor])
    # upstream: 0/255 mask -> gen_dilate(r) -> gen_erosion(r) (both elliptical, same kernel)
    from matanyone2.utils.inference_utils import gen_dilate, gen_erosion

    mask = (guide > GUIDE_THRESHOLD).astype(np.float32) * 255.0
    if mask_close > 0:
        mask = gen_dilate(mask, mask_close, mask_close)
        mask = gen_erosion(mask, mask_close, mask_close)
    device = torch.device("cuda")

    model = run.model("MatAnyone 2", load_model, checkpoint, device)

    run.stage("抠像")
    reader = frame_reader(frames.paths, (height, width))
    passes = [list(range(anchor, len(frames)))]
    if anchor > 0:
        passes.append(list(range(anchor, -1, -1)))

    def matte_shot(side: int):
        """The whole shot, its long side scaled down to `side`."""
        size = fit_size(width, height, side, MULTIPLE, upscale=False)[::-1]  # (h, w)
        mask_t = torch.from_numpy(mask).to(device)
        if tuple(mask_t.shape) != size:
            mask_t = F.interpolate(mask_t[None, None], size=size, mode="nearest")[0, 0]
        out = Output(job)
        done = 0

        def emit(index: int, alpha: np.ndarray) -> None:
            nonlocal done
            if numbers[index] in out.coverage:  # the anchor, already written by the forward pass
                return
            out.put(numbers[index], alpha)
            done += 1
            progress(done, len(frames), "抠像")

        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
                for order in passes:
                    Pass(model, size, device).run(reader, order, mask_t, warmup, (height, width), emit)
                    torch.cuda.empty_cache()
        except BaseException:
            out.close()  # its writing thread, not kept for a run that failed
            raise
        return side, size, out

    try:
        resolution, size, out = run.fit(RESOLUTION, matte_shot, resolution)
    finally:
        reader.close()

    out.finish(  # the Run's seconds_per_frame includes the warm-up steps and (if any) the backward pass
        run,
        numbers,
        model="MatAnyone 2 (pq-yang/MatAnyone2, matanyone2.pth v1.0.0)",
        guide="first non-empty frame only (memory propagation)",
        anchor_frame=numbers[anchor],
        backward_frames=anchor,
        width=width,
        height=height,
        processing_size={"width": size[1], "height": size[0]},
        params={"resolution": resolution, "warmup": warmup, "mask_close": mask_close, "fp16": fp16},
        foreground=False,
    )


if __name__ == "__main__":
    serve(main)
