"""Nodes provided by the LTX-2.5 Clean Plate extension (LTX-2.x Community License)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Cost, Licence, Measured, MissingFrames, NodeParams, Official, Port, WorkerNode, basecolor_map,
                          frame_maps, measured_param, rgb_port)

# Peak VRAM at the default processing size (1920 long side, a full window), RTX 5090 (worker docs.md): the fp8
# transformer alone is ~22 GB, so the node needs a 32 GB card. Its Cost and the 1920 setting use it.
DEFAULT_VRAM_GB = 28.5


class CleanPlate(WorkerNode):
    """Processing, not generation: the plate in, the same frames out with people and vehicles removed. No prompt,
    seed, step count, LoRA or length reaches the node; the worker fixes them (adapters/cleanplate/worker.py CLEAN) and
    checks that every input frame got exactly one output frame of its size and number."""

    id = "cleanplate.clean_plate"
    # upstream ltx_pipelines.ic_lora (stage 1 only, as for every IC-LoRA here): the shot goes in as
    # video_conditioning, the clean plate comes out of pipeline_output_from_chunks
    official = Official(
        cite="third_party/ltx/repo/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py:644-667",
        takes={"image": "video_conditioning"},
        gives={"plate": "pipeline_output_from_chunks"},
    )
    on_node = ("resolution",)
    inputs = (rgb_port(),)
    outputs = (Port("plate", "image.3"),)
    runtime = "cleanplate"
    streams = True  # frame by frame EXR (frame_maps): each frame written is final
    missing_frames = MissingFrames.SKIP
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=2.8, measured_on="RTX 5090 32 GB", note=True)
    licence = Licence(note=True)

    class Params(NodeParams):
        # the long side of the processing size (the model card validated up to 1920×1088); a larger plate is scaled
        # down to it and the result scaled back up
        resolution: Literal[960, 1280, 1920] = measured_param(
            {960: Measured(below=1920), 1280: Measured(below=1920), 1920: Measured(gb=DEFAULT_VRAM_GB)},
            default=1920, group="clean_plate")

    @classmethod
    def convert(cls, ctx, raw, job):
        # display sRGB 0..1 from the VAE: written as it is in the working space, as a model's base colour is
        return frame_maps(ctx, raw, job.plate, {"plate": basecolor_map("plate")}, stage="write_plate")


NODES = (CleanPlate,)
