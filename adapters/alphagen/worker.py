"""LTX-2.5 Alpha Gen worker: an alpha matte from the plate alone (no mask, no prompt). Runs in the ltx extension's
environment (Extension.runs_in); never imports Lab2Shot core.

    python worker.py <job.json>

Contract: lab2shot_worker/matte.py (the matting family's), without a guide mask. The whole job is the shared LTX
runtime's (adapters/ltx/ltx_runtime.py run_feature) with this extension's LoRA and the fixed recipe below, as the
LoRA's model card recommends: distilled base, LoRA strength 1.0, empty prompt, stage 1 only at the native resolution
(ltx_pipelines.ic_lora --skip-stage-2), seed 1234, at most 1920×1088 and 145 frames per window (longer: "RGB content
starts leaking into the matte").

Matting only. Nothing the artist sets reaches the model besides the plate and the processing size: the prompt, seed,
steps, LoRA and frame count are fixed here. The output is the alpha only (the decoded grey picture averaged to one
channel); no picture the model made is ever written. Every input frame gets exactly one alpha of its own size; any
other count or size stops the job (E-ALPHAGEN-SHAPE).

Output: raw/frame_<n>.npz alpha float32 [H,W] 0..1 at the input resolution (no foreground colour).
"""

from __future__ import annotations

from lab2shot_worker import MemoryBound, serve
from lab2shot_worker.matte import Output
from ltx_runtime import Feature, FrameOutput, Recipe, run_feature

# the model card's recommended settings; the prompt is always empty ("the RGB video is the only guide")
MATTE = Feature(node="alphagen.matte", title="LTX-2.5 Alpha Gen",
                lora="LTX-2.5-22b-IC-LoRA-Alpha-Gen/ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors",
                recipe=Recipe(prompt="", seed=1234, fps=24.0, max_frames=145, overlap=17, max_side=1920, max_short=1088),
                stage="matte", shape_error="E-ALPHAGEN-SHAPE")
# what the memory grows with: the long side of the processing size, within the node's choices
RESOLUTION = MemoryBound.parameter("resolution", (1920, 1280, 960))


class Alphas(FrameOutput):
    """The matte comes out as a grey picture: one channel of it is written, never the RGB."""

    def __init__(self, job, numbers: list[int]):
        self.out, self.numbers = Output(job), numbers

    def put(self, index: int, rgb) -> None:
        self.out.put(self.numbers[index], rgb.mean(dim=-1).cpu().numpy())

    def close(self) -> None:
        self.out.close()

    def finish(self, run, numbers: list[int], **facts) -> None:
        self.out.finish(run, numbers, model="LTX-2.5 22B distilled (fp8) + IC-LoRA Alpha Gen 0.9, stage 1 only, empty prompt",
                        guide="none (the model picks the foreground)", foreground=False, **facts)


def main(job_path: str) -> None:
    run_feature(job_path, MATTE, RESOLUTION, Alphas)


if __name__ == "__main__":
    serve(main)
