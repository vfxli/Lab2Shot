"""LTX-2.5 Clean Plate worker: the plate with its people and vehicles removed and the background rebuilt behind them.
Runs in the ltx extension's environment (Extension.runs_in); never imports Lab2Shot core.

    python worker.py <job.json>

The whole job is the shared LTX runtime's (adapters/ltx/ltx_runtime.py run_feature) with this extension's LoRA and the
fixed recipe below: distilled base, LoRA strength 1.0, windows of at most 49 frames (the model card's training length;
longer windows risk the input showing through, as the Alpha Gen card warns), the model card's example prompt describing the empty scene,
stage 1 only at the native resolution, no guidance (ComfyUI's single-stage distilled IC-LoRA workflow samples at CFG 1,
where the card's negative prompt has no effect, so it is not encoded), 25 fps (the LoRA's training clips).

Processing, not generation. Nothing the artist sets reaches the model besides the plate and the processing size:
the prompt, seed, steps, LoRA and frame count are fixed here. Every input frame gets exactly one output frame of its
own size and number, and no other frame is made; any other count or size stops the job (E-CLEANPLATE-SHAPE).

Output: raw/frame_<n>.npz plate float32 [H,W,3] 0..1 display sRGB at the input resolution.
"""

from __future__ import annotations

import numpy as np

from lab2shot_worker import MemoryBound, serve
from lab2shot_worker.frame_io import Writer
from ltx_runtime import Feature, FrameOutput, Recipe, run_feature

# the model card's example positive prompt ("describe the desired empty scene"), never shown or changed
PROMPT = ("An empty clean plate of the exact same location: identical background, environment, structures and lighting "
          "as the source video, with no people, no humans, no figures, and no body parts such as arms, hands or legs "
          "anywhere in the frame. Static photorealistic footage, natural light, high detail.")
CLEAN = Feature(node="cleanplate.clean_plate", title="LTX-2.5 Clean Plate",
                lora="LTX-2.5-22b-IC-LoRA-Clean-Plate/ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors",
                recipe=Recipe(prompt=PROMPT, seed=1234, fps=25.0, max_frames=49, overlap=17, max_side=1920, max_short=1088),
                stage="clean_plate", shape_error="E-CLEANPLATE-SHAPE")
RESOLUTION = MemoryBound.parameter("resolution", (1920, 1280, 960))


class Plates(FrameOutput):
    """Each frame's plate as it came out (display sRGB), at the input size."""

    def __init__(self, job, numbers: list[int]):
        self.raw, self.numbers = job.raw_dir, numbers
        self.raw.mkdir(parents=True, exist_ok=True)
        self.writer = Writer(threads=2, max_pending=8)

    def put(self, index: int, rgb) -> None:
        plate = np.ascontiguousarray(rgb.cpu().numpy(), dtype=np.float32)
        self.writer.npz(self.raw / f"frame_{self.numbers[index]}.npz", plate=plate)

    def close(self) -> None:
        self.writer.close()

    def finish(self, run, numbers: list[int], **facts) -> None:
        self.close()
        run.finish(numbers, kind="cleanplate",
                   model="LTX-2.5 22B distilled (fp8) + IC-LoRA Clean Plate 1.0, stage 1 only, fixed prompt",
                   plate="display sRGB float32 0..1, the input frame's size; one per input frame", **facts)


def main(job_path: str) -> None:
    run_feature(job_path, CLEAN, RESOLUTION, Plates)


if __name__ == "__main__":
    serve(main)
