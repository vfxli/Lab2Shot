"""Diffusers worker: the generative image models run through the pinned diffusers checkout.

Runs inside third_party/diffusers/.venv with the pinned diffusers repo's src/ on sys.path
(Extension.import_repo; never imports Lab2Shot core). One worker, one node, one pipeline, three modes:

    python worker.py <job.json>

diffusers.generate (mode) -- one QwenImage21Pipeline run as the official README's three usages, chosen by the node's
                            mode: text-to-image sends no image and the width/height the node worked out from aspect
                            and resolution (the official WH_RATIO_TO_SIZE usage, passed explicitly as the README shows);
                            image editing sends image=[the job's own frame] (a subject with an alpha travels as its own
                            listing, straight-alpha PNGs, the picture the upstream examples edit); multi-reference sends
                            image=[subject, *references] (a JSON listing of display PNGs, in order); the output size
                            follows the subject (passed explicitly: the pipeline alone would follow the last picture).

The worker writes the family's raw contract (lab2shot/nodes/families/image_generation.py):

    raw/image.png    the generated picture, 8-bit sRGB PNG, straight alpha when the model made one (Qwen-Image 2.1's
                     VAE is natively 4-channel, so the output is RGBA: transparent when the prompt asks for it)
    raw/result.json  the standard fields plus width, height, channels, mode, model, the sampler settings, and
                     which loading served the pipeline (full_gpu, transformer_resident or cpu_offload) and vae_tiling

Determinism: every job seeds all generators from the node's seed (set_seed); use_kv_cache keeps its official
default (toggling it changes the sample in reduced precision, upstream's own note), so the same seed, parameters
and card give the same picture.
"""

from __future__ import annotations

import json

import torch
from PIL import Image

from lab2shot_worker import fail, progress, reason, resident, say, serve, set_seed
from lab2shot_worker.run import Run

NODES = ("diffusers.generate",)
MODEL = "Qwen/Qwen-Image-2.1"
WEIGHTS = "Qwen-Image-2.1"  # below the extension's weights folder: the snapshot's own layout
MAX_IMAGES = 10  # the upstream limit (README: "Support up to 10 reference images")
# The whole pipeline in bf16 needs about 32.4 GB of weights (text encoder 8B ~17.5 + transformer 7B ~14.2 + VAE
# ~0.7, measured from the safetensors headers: all three shard groups are BF16 except the VAE which is F32 and
# converted). Which loading a card allows depends on the output size (by resolution basis, <= key), since the peak is
# the VAE decode of the finished picture:
#   transformer_resident (load_pipeline): the transformer and the VAE on the GPU, the text encoder streamed. Measured
#     on the RTX 5090: 20.5 GB allocated at 1024; at 2048 the decode asks for 4.5 GB more on top of 25.0 GB allocated
#     (29.5 GB, with 4.0 GB more reserved but unallocated), past the 29.45 GB a 32 GB card allows (out of memory).
#     The official low-memory path (enable_model_cpu_offload) does not save it either: the decode alone
#     then asks for 2.25 GB on top of 23.5 GB allocated (RTX 5090, out of memory), since it is the 2048 picture's
#     VAE activations, not the weights. So at 2048 a card under VAE_TILE_BELOW_GB decodes in tiles (the VAE's own
#     enable_tiling: 256 px tiles, 64 px overlaps blended; the denoising is untouched), and with the decode small
#     the transformer stays resident down to RESIDENT_TILED_GB (the 2048 denoising peaked at 25.0 GB allocated).
#   full_gpu: everything at once: the resident threshold plus the text encoder's 17.5 GB (derived: no card here
#     holds it).
RESIDENT_GB = {1024: 22.0, 2048: 34.0}
FULL_LOAD_GB = {1024: 40.0, 2048: 52.0}
VAE_TILE_BELOW_GB = 34.0  # 2048 basis: the untiled decode needs a card with this much free
RESIDENT_TILED_GB = 28.0  # 2048 basis with a tiled decode: the transformer resident


@resident
def load_pipeline(snapshot, device: str, loading: str):
    """The official pipeline from the local snapshot (from_pretrained with torch_dtype=bfloat16, never the network).

    Three loadings, by what the card has free; the result is the same, only the time differs (a tiled 2048 decode,
    generate(), differs slightly where the tiles blend). Which one
    served is part of the result (result.json's `loading`):
      full_gpu             `.to(device)` (the README's usage) when the card holds everything;
      transformer_resident the transformer and the VAE stay on the GPU between jobs, and the 8B text encoder, which
                           runs once per picture, streams through it a block at a time (diffusers' group offloading,
                           blocks prefetched on a side stream): no transformer copy over PCIe per picture;
      cpu_offload          `enable_model_cpu_offload()` (the README's memory-optimization section): every component
                           goes to the GPU and back for every picture."""
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(str(snapshot), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=True)  # the worker reports its own steps through the protocol, never tqdm
    if loading == "cpu_offload":
        pipe.enable_model_cpu_offload()
    elif loading == "transformer_resident":
        from diffusers.hooks import apply_group_offloading

        pipe.transformer.to(device)
        pipe.vae.to(device)
        apply_group_offloading(pipe.text_encoder, onload_device=torch.device(device), offload_type="leaf_level",
                               use_stream=True)
    else:
        pipe = pipe.to(device)
    return pipe


def _subject(job) -> Image.Image | None:
    """The picture being edited (edit / reference nodes): the job's one frame, as the upstream examples read it.

    A subject with an alpha travels as its own listing (job input subject: straight-alpha RGBA PNGs — the picture
    this class of model edits transparent layers with); without one the display frame is the subject, the standard
    path of every worker."""
    if not job.frames:
        return None
    frame, path = job.frames[0]
    listing = job.inputs.get("subject")
    if listing is not None:
        named = json.loads(listing.read_text(encoding="utf-8"))["frames"].get(str(frame))
        if named is not None:
            return Image.open(listing.parent / named)
        say("W-DIFFUSERS-NOSUBJECT", frame=frame)
    return Image.open(path)


def _references(job) -> list[Image.Image]:
    """The job input references: display PNGs in order, each the one picture a vision model sees."""
    listing = job.inputs.get("references")
    if listing is None:
        return []
    names = json.loads(listing.read_text(encoding="utf-8"))["images"]
    return [Image.open(listing.parent / name).convert("RGB") for name in names]


def generate(job_path: str) -> None:
    run = Run.start(job_path, NODES, "QwenImage 2.1", extension="diffusers")
    job, params = run.job, run.params
    snapshot = job.weights_dir / WEIGHTS
    run.weights(snapshot / "model_index.json", what=reason("I-DIFFUSERS-WEIGHTS"))

    negative = params["negative"].strip()
    steps, seed = int(params["steps"]), int(params["seed"])
    resolution, true_cfg = int(params["resolution"]), float(params["true_cfg"])
    width, height = params.get("width"), params.get("height")  # the node's aspect arithmetic (text-to-image)

    images = [im for im in (_subject(job), *_references(job)) if im is not None]
    if len(images) > MAX_IMAGES:
        fail("E-DIFFUSION-MAXREFS", count=len(images), most=MAX_IMAGES)

    # The output follows the subject being edited. The pipeline itself takes the aspect ratio of the LAST picture
    # (pipeline_qwenimage21.py:620-622, image[-1]), which with references is a reference: the subject's size is
    # given explicitly instead, on the pipeline's own arithmetic (calculate_dimensions), so the order of the pictures
    # (the subject first, as the prompt names them) stays as it is. One picture: the same size the pipeline takes.
    if images and width is None:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        width, height, _ = calculate_dimensions(resolution * resolution, images[0].size[0] / images[0].size[1])

    device = "cuda"
    # what the card has for this job: its free memory plus what this worker already holds (a pipeline kept from the job
    # before, torch's cache): counted without it, the next job would see less, pick another loading, and the resident key
    # would change — the kept pipeline moved to RAM and a second copy loaded, every other picture
    free_gb = (torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()) / 2**30
    # which loading the card allows (results the same either way)
    basis = 1024 if resolution <= 1024 else 2048
    tiled = basis == 2048 and free_gb < VAE_TILE_BELOW_GB
    resident_gb = RESIDENT_TILED_GB if tiled else RESIDENT_GB[basis]
    loading = ("full_gpu" if free_gb >= FULL_LOAD_GB[basis] else
               "transformer_resident" if free_gb >= resident_gb else "cpu_offload")
    pipe = run.model("load_model", load_pipeline, snapshot, device, loading, stage_params={"model": "QwenImage 2.1"})
    # the pipeline is kept between jobs: each job sets the decode it needs
    if tiled:
        pipe.vae.enable_tiling()
    else:
        pipe.vae.disable_tiling()

    run.stage("generate")

    def _step(_pipe, i, _t, tensors):
        """Progress per denoising step. The pipeline pops its tensors off what this returns, so they go back
        unchanged (nothing here changes them)."""
        progress(i + 1, steps, "generate")
        return tensors

    with run.frame():
        result = pipe(
            prompt=params["prompt"],
            image=images or None,
            negative_prompt=negative or None,
            true_cfg_scale=true_cfg,
            width=width,
            height=height,
            output_resolution=resolution,  # the node's resolution: t2i's explicit width/height wins, edit/refs derive from it
            num_inference_steps=steps,
            generator=set_seed(seed, device),
            callback_on_step_end=_step,
        )
    image = result.images[0]
    seconds = run.frame_seconds[-1]  # this one picture's time, after the GPU synchronize in frame_done()

    image.save(job.raw_dir / "image.png", format="PNG")
    run.finish(
        [],
        kind="generation",
        model=MODEL,
        node=run.node,
        width=image.width,
        height=image.height,
        channels=len(image.getbands()),
        mode=image.mode,
        pictures=len(images),
        steps=steps,
        seed=seed,
        true_cfg=true_cfg,
        negative=bool(negative),
        resolution=resolution,
        generation_seconds=round(seconds, 2),
        loading=loading,
        vae_tiling=tiled,
        free_gb_at_load=round(free_gb, 1),
        files="image.png: the generated picture, 8-bit sRGB PNG (straight alpha when made)",
    )


def main(job_path: str) -> None:
    generate(job_path)


if __name__ == "__main__":
    serve(main)
