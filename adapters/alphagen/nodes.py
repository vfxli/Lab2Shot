"""Nodes provided by the LTX-2.5 Alpha Gen extension (LTX-2.x Community License)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Cost, Licence, MatteNode, Measured, Official, NodeParams, measured_param

# Peak VRAM at the default processing size (1920 long side, a full 145-frame window), RTX 5090 (worker docs.md):
# the fp8 transformer alone is ~22 GB, so the node needs a 32 GB card. Its Cost and the 1920 setting use it.
DEFAULT_VRAM_GB = 28.5


class Matte(MatteNode):
    """Matting only: the plate in, an alpha out. No prompt, seed, step count, LoRA or length reaches the node; the
    worker fixes them (adapters/alphagen/worker.py MATTE) and checks that every input frame got one alpha of its size."""

    id = "alphagen.matte"
    # upstream ltx_pipelines.ic_lora (the model card's command, with --skip-stage-2): the reference clip goes in as
    # video_conditioning, the matte comes out of pipeline_output_from_chunks
    official = Official(
        cite="third_party/ltx/repo/packages/ltx-pipelines/src/ltx_pipelines/ic_lora.py:644-667",
        takes={"image": "video_conditioning"},
        gives={"alpha": "pipeline_output_from_chunks"},
    )
    # no guide mask: MatteNode (the unguided tier, as BiRefNet). Input RGB, output Alpha, written by matte()
    on_node = ("resolution",)
    runtime = "alphagen"
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=2.8, measured_on="RTX 5090 32 GB", note=True)
    licence = Licence(note=True)

    class Params(NodeParams):
        # the long side of the processing size; the model card's limit is 1920×1088, a larger plate is scaled down to
        # it and the alpha scaled back up
        resolution: Literal[960, 1280, 1920] = measured_param(
            {960: Measured(below=1920), 1280: Measured(below=1920), 1920: Measured(gb=DEFAULT_VRAM_GB)},
            default=1920, group="matte")


NODES = (Matte,)
