"""Diffusion image generation and editing: one model, a prompt and a picture (or several) in, one picture out.

The family is what every generative image model shares, whatever its backend (QwenImage 2.1 today, another one
tomorrow): the node runs its worker with the prompt and the pictures, and the worker writes the generated picture
plus result.json. Nothing of the model's insides (latents, text encoders, samplers) is a port or a type: the graph
only ever sees 图像进图像出, exactly like every other node.

The raw contract of the family's workers:

    raw/image.png   the generated picture, 8-bit sRGB PNG (straight alpha when the model made one: the display
                    colour this class of model is trained on, what it writes itself)
    raw/result.json the standard fields plus: width, height, channels (3 or 4), model, and the model's own settings
                    (seed, steps, ...)

The colour contract is the family's, in one place: the generated PNG is display-referred sRGB, which is the working
space bit for bit (io/color.py's first rule), so convert() hands the file on as it is and marks it with the working
space — the same thing a display-referred reader (and DiffusionRenderer's relight) does. The picture is a still
(Info.still): one image, no frame range, not a frame source.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import numpy as np

from ...data.contracts import NEW_PICTURE
from ...errors import Invalid
from ...messages import Msg
from ..applies import Param
from ..base import NodeParams, P, Port
from ..kit.ports import Measured, measured_param
from .base import Job, RawOutput, WorkerNode

# The official aspect table of QwenImage 2.1 (README「Supported Aspect Ratios」, the native 2K grid): what a
# text-to-image node sends, scaled from the native 2048 basis to the chosen one. (The pipeline's own
# calculate_dimensions arithmetic, used for image-conditioned sizes, lands one 32-step off this table on 4:3 / 3:4;
# the README's table is what the model is shown at its native resolution, so text-to-image follows the table.)
NATIVE_SIZE = {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
               "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)}
ASPECTS = tuple(NATIVE_SIZE)


def dimensions(resolution: int, ratio: float) -> tuple[int, int]:
    """(width, height) at the resolution basis following `ratio`: the pipeline's own arithmetic (calculate_dimensions —
    the sides that satisfy the ratio at the given area, on a 32-pixel grid), what an image-conditioned call makes."""
    width = round((resolution * resolution * ratio) ** 0.5 / 32) * 32
    return width, round(width / ratio / 32) * 32


def native_dimensions(resolution: int, aspect: str) -> tuple[int, int]:
    """(width, height) for text-to-image: the official aspect table (the native 2K grid) scaled to the resolution
    basis, on the same 32-pixel grid — the table itself at 2048."""
    width, height = NATIVE_SIZE[aspect]
    scale = resolution / 2048
    return round(width * scale / 32) * 32, round(height * scale / 32) * 32


class DiffusionImageParams(NodeParams):
    """What every image-generation node shares: the prompt, the sampler, the size policy, and the one pipeline's three
    ways of running (a `mode`: text-to-image, single-image editing, or a subject plus reference images). A model's own
    nodes add their specifics on top (a model choice, its own step range)."""

    prompt: str = P("", group="generation", lines=4, words="family.image_generation.prompt")
    negative: str = P("", group="generation", lines=2)
    # CFG only above 1 with a negative prompt: these models sample without guidance by design (1.0)
    true_cfg: float = P(1.0, ge=1.0, le=10.0, group="generation")
    steps: Literal[10, 20, 30, 40, 50] = measured_param(
        {n: Measured(flat=True) for n in (10, 20, 30, 40, 50)}, default=40, group="generation")
    seed: int = P(0, ge=0, le=2**31 - 1, group="generation", words="family.image_generation.seed")
    mode: Literal["text_to_image", "edit_image", "reference_images"] = P(
        "text_to_image", group="generation", words="family.image_generation.mode")
    # 官方 output_resolution：生成面积的边长基准（面积 = 基准²，长宽比另行决定）。默认 2048（官方 README
    # 「Default Parameters」的原生 2K 档，质量优先；offload 下很慢，见适配器 docs.md）。各档显存按实测填入。
    resolution: Literal[512, 768, 1024, 1536, 2048] = measured_param(
        {512: Measured(below=2048), 768: Measured(below=2048), 1024: Measured(below=2048),
                      1536: Measured(below=2048), 2048: Measured(gb=19.0)},
        default=2048, group="generation", words="family.image_generation.resolution")
    aspect: Literal[ASPECTS] = P(  # type: ignore[valid-type]
        "1:1", group="generation",
        applies=Param("mode").one_of("text_to_image"))


class DiffusionImage(WorkerNode):
    """An image made by a generative diffusion model: one still picture out, of its own size (not the plate's).

    A subclass declares `runtime` (its extension), its licence and cost, and its inputs (a subject to edit, a list of
    references); the worker gets the prompt and the pictures and writes the raw contract above. A node of this family
    never has frame results: what it makes is one picture, whichever frames came in."""

    MAX_REFERENCES = 10  # the upstream limit (QwenImage 2.1: "Support up to 10 reference images")

    outputs = (Port("image", "image", shape=NEW_PICTURE, words="family.image_generation.image"),)
    main = "image"
    picture = ""  # a picture of its own making: the result's size follows the model's policy, not any input's
    on_node = ("mode", "prompt", "resolution", "seed")

    @classmethod
    def info(cls, params: dict, inputs: dict[str, list]) -> "Info":
        from ..base import Info

        # The picture the model will make, a still with no frame range. Text-to-image follows the official aspect
        # table at the resolution basis; the image-conditioned modes follow the subject's aspect (`inputs` is port ->
        # list of Info, so the subject's size is its Info's width/height, not a packet's meta), both on the pipeline's
        # own 32-pixel grid.
        if params.get("mode") == "text_to_image":
            width, height = native_dimensions(params["resolution"], params["aspect"])
        else:
            ratio = 1.0
            for info in inputs.get("image", ()):
                if info.width:
                    ratio = info.width / max(1, info.height)
                    break
            width, height = dimensions(params["resolution"], ratio)
        return Info((), width, height, still=True)

    @classmethod
    def prepare(cls, ctx) -> Job:
        if not ctx.params["prompt"].strip():
            raise Invalid(Msg("E-DIFFUSION-NOPROMPT", node=ctx.label))
        image = ctx.input("image")
        inputs = cls._subject_alpha(ctx, image) if image is not None else {}
        pngs, skipped = cls.reference_images(ctx, cls._frame(ctx, image))
        if skipped:  # an item with no picture at the working frame: left out and said, never another frame silently
            ctx.say("N-DIFFUSION-REFSKIPPED", count=len(skipped), names=skipped[:5],
                    more="……" if len(skipped) > 5 else "")
        listing = cls._references_listing(ctx, pngs)
        if listing is not None:
            inputs["references"] = listing
        job = Job(image, inputs=inputs or {})
        if ctx.params.get("mode") == "text_to_image":
            # the width and height the worker passes the pipeline, from the aspect and the resolution basis (the
            # upstream WH_RATIO_TO_SIZE usage: the official table, scaled — one implementation, native_dimensions())
            width, height = native_dimensions(ctx.params["resolution"], ctx.params["aspect"])
            return job.with_(extra={"width": width, "height": height})
        return job

    @classmethod
    def _frame(cls, ctx, image) -> int:
        """The one frame the node works on: the subject's (a one-frame input, checked before submission), 0 without
        one (text-to-image: references pair with the one frame being made)."""
        if image is not None and image.meta.get("frames"):
            return image.meta["frames"][0]
        return 0

    @classmethod
    def _subject_alpha(cls, ctx, image) -> dict[str, Path]:
        """The subject's own job input, when it carries an alpha.

        The frames themselves go the standard way (display sRGB PNGs, the picture over black — the one path every
        worker's pictures take). A subject with an alpha is more than that: this class of model edits transparent
        layers with the alpha, so the whole RGBA picture travels along as straight-alpha 8-bit PNGs (what the
        upstream examples feed the model), one listing file naming them; the worker prefers it over the frames."""
        from ...data.payloads import file_at, has_alpha, read_picture
        from ...io import images

        if not has_alpha(image):
            return {}
        folder = ctx.work / "subject"
        folder.mkdir(exist_ok=True)
        files = {}
        for f in image.meta["frames"]:
            path = file_at(image, f)
            if path is None:
                continue
            files[f] = folder / f"frame.{f}.png"
            # read_picture gives premultiplied RGBA; write_png stores straight alpha (OpenImageIO divides)
            images.write_png(files[f], images.as_uint8(read_picture(path, True, None)))
        listing = ctx.work / "subject.json"
        listing.write_text(json.dumps({"frames": {str(f): p.name for f, p in files.items()}}), encoding="utf-8")
        return {"subject": listing}

    @classmethod
    def reference_images(cls, ctx, frame: int) -> tuple[list[Path], list[str]]:
        """The 参考图 list as one display picture per item, and the names of the items that could not be taken.

        Every item is whatever the graph produces — a picture, a matte, a depth map, a normal pass: each becomes the
        one image a vision model sees, the way the viewer shows it. A still is its own picture; a sequence is taken
        at `frame` (the frame being edited); an item that does not cover `frame` is left out and named, never
        silently replaced by another frame."""
        from ...data.packet import Packet, items_of, packet_dir

        references = ctx.input("references")
        if references is None:
            return [], []
        out: list[Path] = []
        skipped: list[str] = []
        for name, fp in items_of(references):
            # a list never holds its items: each is a packet of its own in the cache (data/packet.py items_of)
            item = Packet.load(packet_dir(fp))
            png = cls._reference_png(ctx, item, frame, name)
            if png is None:
                skipped.append(name)
            else:
                out.append(png)
        if len(out) > cls.MAX_REFERENCES:
            raise Invalid(Msg("E-DIFFUSION-MAXREFS", count=len(out), most=cls.MAX_REFERENCES))
        return out, skipped

    @classmethod
    def _reference_png(cls, ctx, item, frame: int, name: str) -> Path | None:
        """One item as the display PNG the model sees: a picture as the worker's own frames are (frames_for_worker),
        a data map normalized by its stated range (the viewer's own display), written beside the job. None: this item
        is a sequence that does not cover `frame` (a still, or a single picture read as a one-frame sequence, goes
        with any frame — the second material's own rule, RoMa v2's 参考图 the same way)."""
        from ...data.maps import map_at
        from ...data.payloads import file_at, frames_for_worker, image_files, is_data
        from ...io import images

        if not is_data(item):
            shown = frames_for_worker(item)
            path = file_at(shown, frame)
            if path is None and (item.meta.get("still") or len(image_files(shown)) == 1):
                path = next(iter(image_files(shown).values()))  # one picture: it goes with any frame
            return path
        got = map_at(item, frame)
        if got is None and (item.meta.get("still") or len(item.meta.get("frames") or ()) == 1):
            got = map_at(item, item.meta["frames"][0])
        if got is None:
            return None
        values = got[0]
        single = values[..., 0] if values.ndim == 3 else values
        lo, hi = item.meta.get("range") or item.meta.get("full_range") or (None, None)
        if lo is None or not hi > lo:  # no stated range: this frame's own extent, like the viewer's「贴合」
            lo, hi = float(np.nanmin(single)), float(np.nanmax(single))
        shown = np.clip((single - lo) / (hi - lo), 0.0, 1.0)
        folder = ctx.work / "references"
        folder.mkdir(exist_ok=True)
        safe = "".join(c if c not in "/\\:" and ord(c) >= 32 else "_" for c in name) or "reference"
        path = folder / f"{safe}.png"
        images.write_png(path, images.as_uint8(np.repeat(shown[..., None], 3, axis=-1)))
        return path

    @classmethod
    def _references_listing(cls, ctx, pngs: list[Path]) -> Path | None:
        """The references as one job input: a JSON naming the pictures, in order, relative to the listing's folder."""
        if not pngs:
            return None
        listing = ctx.work / "references.json"
        listing.write_text(json.dumps({"images": [os.path.relpath(p, ctx.work) for p in pngs]}), encoding="utf-8")
        return listing

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job):
        """raw/image.png + result.json -> the still packet. The PNG is the display colour the model wrote: the working
        space bit for bit, handed on as the file it is."""
        from ...data.payloads import still_packet
        from ...io.color import working_space

        result = raw.result()
        out = ctx.outputs["image"]
        png = out / "image.png"
        png.write_bytes(raw.file("image.png").read_bytes())
        packet = still_packet(out, png, [0], int(result["width"]), int(result["height"]),
                              channels=int(result.get("channels") or 3), colorspace=working_space())
        return {"image": packet}
