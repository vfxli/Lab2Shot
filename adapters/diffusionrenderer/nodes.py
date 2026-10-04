"""Nodes provided by the Cosmos DiffusionRenderer extension (Apache-2.0 code, NVIDIA Open Model License weights:
commercial use allowed, with NVIDIA's notices)."""

from __future__ import annotations

import shutil
from typing import Literal

from lab2shot.sdk import (rgb_port, Official, normal_port, UNIT, measured_param, MissingFrames, Param, Job, camera_normals, HighDynamicRange, Invalid, Msg, WorkerNode,
                          NodeParams, FrameCount, P, Port, basecolor_map, basecolor_port, frame_maps, image_files, image_packet, Cost,
                          Licence, Measured, working_space)

RUNTIME = "diffusionrenderer"
# Peak VRAM at the default settings (canvas 1280×704, 41 frames a window), RTX 4090: 19.1 GB on a 70-frame shot cut
# into two windows (docs.md; a single 24-frame window peaked at 17.5). The two measured settings and both nodes'
# Cost say this one number, so the scheduler and the node's rating never count on less.
DEFAULT_VRAM_GB = 19.1

MAX_FRAMES = (1, 9, 17, 25, 33, 41, 49, 57)  # 「每段最多帧数」的档（8k+1：tokenizer 按块编码）


class DiffusionParams(NodeParams):
    """What both models share: the working canvas, the windows the shot is cut into, the sampler."""

    resolution: Literal[1280, 1024, 960, 768, 640, 512] = measured_param({1280: Measured(gb=DEFAULT_VRAM_GB), 1024: Measured(below=1280), 960: Measured(below=1280), 768: Measured(below=1280), 640: Measured(below=1280), 512: Measured(below=1280)},
        default=1280, group="model")
    max_frames: Literal[MAX_FRAMES] = measured_param(  # type: ignore[valid-type]
        {1: Measured(below=41), 9: Measured(below=41), 17: Measured(below=41), 25: Measured(below=41), 33: Measured(below=41), 41: Measured(gb=DEFAULT_VRAM_GB), 49: Measured(below=57), 57: Measured(gb=17.8)},
        default=41, group="temporal")
    # 重叠最多是每段帧数的一半（worker 的分段 plan_windows 要求）：每一档只在「每段最多帧数」够两倍时可选，
    # 网页提交前就把这一档变灰、写清原因（option_applies，同 depthanything3 的做法）；计算时 _check 仍是最后一道
    overlap: Literal[0, 4, 8, 12] = measured_param({0: Measured(flat=True), 4: Measured(flat=True), 8: Measured(flat=True), 12: Measured(flat=True)}, default=8, group="temporal",
        option_applies={v: Param("max_frames").one_of(*(n for n in MAX_FRAMES if n >= 2 * v)) for v in (4, 8, 12)})
    steps: Literal[5, 10, 15] = measured_param({5: Measured(flat=True), 10: Measured(flat=True), 15: Measured(flat=True)},
        default=15, group="model")
    seed: int = P(1000, ge=0, le=2**31 - 1, group="model")


def _check(params: dict) -> None:
    """The overlap is at most half the frames per window: the page stops it before submitting (option_applies); this is
    the last check, for the command line and anything that bypasses the page."""
    if params["overlap"] * 2 > params["max_frames"]:
        raise Invalid(Msg("E-DIFFUSIONRENDERER-OVERLAP", overlap=params["overlap"], max_frames=params["max_frames"], half=params["max_frames"] // 2))


class Inverse(WorkerNode):
    id = "diffusionrenderer.inverse_render"
    # 上游 inference_inverse_renderer.py：--dataset_path（画面）-> --inference_passes 的五个通道
    official = Official(
        cite="third_party/diffusionrenderer/repo/cosmos_predict1/diffusion/inference/inference_inverse_renderer.py:60-94",
        takes={"image": "--dataset_path"},
        gives={"basecolor": "basecolor", "normal": "normal", "depth": "depth",
               "roughness": "roughness", "metallic": "metallic"},
    )
    on_node = ("resolution", "max_frames")
    version = 4  # 4：要裁三块及以上的画面（竖幅、方形）改回整幅放进画布、四周镜像填充；3：非 16:9 画面按官方居中裁切的方式分块算再拼回
    # ram_gb: the 7B checkpoint is read into memory before it goes to the GPU (22–29 GB)
    # vram_gb: RTX 4090，默认 1280×704 / 每段 41 帧（DEFAULT_VRAM_GB）
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=9.7, ram_gb=32)
    licence = Licence(note=True)
    # 默认每段 41 帧、段间重叠 8 帧，段内时序稳定；深度是每段各自归一化的相对值，不是米制
    inputs = (rgb_port(),)
    outputs = (
        basecolor_port(),  # 口名和标签在 kit/ports.py 写一次，和 OpenDelight 那个口是同一样东西
        normal_port(),
        Port("depth", "image.1", means=("scale",)),
        Port("roughness", "image.1"),
        Port("metallic", "image.1"),
    )
    runtime = RUNTIME

    class Params(DiffusionParams):
        pass

    missing_frames = MissingFrames.SKIP
    # the maps each output is written from (convert), and so which passes the worker runs (prepare)
    MAPS = {
        "basecolor": basecolor_map("basecolor"),  # the model gives it like a texture, sRGB-encoded
        "normal": camera_normals("normal"),  # the one turn into our camera (nodes/kit/maps.py)
        "depth": ("image.1", lambda d: (d["depth"], d["depth"] >= 0), {"scale": "affine"}),
        "roughness": ("image.1", "roughness", {"value_range": UNIT, "half": True}),  # 0..1
        "metallic": ("image.1", "metallic", {"value_range": UNIT, "half": True}),  # 0..1
    }

    @classmethod
    def prepare(cls, ctx) -> Job:
        _check(ctx.params)
        # each channel is a run of the 7B video model (about ten seconds a frame): only the ones something is wired to
        return Job(ctx.input("image"), extra={"passes": sorted(p for p in cls.MAPS if p in ctx.wanted)})

    @classmethod
    def convert(cls, ctx, raw, job):
        return frame_maps(ctx, raw, job.plate, cls.MAPS, stage="write_maps")


class Relight(WorkerNode):
    id = "diffusionrenderer.relight"
    # 上游 inference_forward_renderer.py：dataset_path（材质通道）+ envlight_path（HDRI）-> output（重新打光的画面）
    official = Official(
        cite="third_party/diffusionrenderer/repo/cosmos_predict1/diffusion/inference/inference_forward_renderer.py:204-258",
        takes={"image": "dataset_path", "hdri": "envlight_path"},
        gives={"image": "output"},
    )
    version = 4  # 4：竖幅、方形画面改回整幅放进画布、镜像填充；3：非 16:9 画面按官方裁切分块算再拼回；2：image packets carry whether they have an alpha
    on_node = ("env_rotate", "exposure")
    # vram_gb: RTX 4090，同一 7B 模型、同一套分段参数（DEFAULT_VRAM_GB），正向渲染而不是拆解
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=11.7, ram_gb=32)
    licence = Licence(note=True)
    # 默认每段 41 帧、段间重叠 8 帧；真实镜头上的结果仍有斑块和彩色噪点，属实验功能
    inputs = (rgb_port(), rgb_port(name="hdri", expects=(FrameCount(most=1), HighDynamicRange())))
    outputs = (Port("image", "image.3"),)
    runtime = RUNTIME

    class Params(DiffusionParams):
        env_rotate: float = P(0.0, unit="°", ge=-360, le=360, group="environment_light")
        exposure: float = P(0.0, unit="EV", ge=-20, le=20, group="environment_light")

    @classmethod
    def prepare(cls, ctx) -> Job:
        _check(ctx.params)
        hdri_files = image_files(ctx.input("hdri"))
        return Job(ctx.input("image"), inputs={"hdri": hdri_files[min(hdri_files)]})

    @classmethod
    def convert(cls, ctx, raw, job):
        image = job.plate
        ctx.stage("write_image")
        out = ctx.outputs["image"]
        files = {}
        for f in ctx.each(image.meta["frames"]):
            files[f] = out / f"frame.{f}.png"
            # a frame the worker did not write stops the cook naming the file (E-FAMILY-NORAW), not a traceback
            shutil.copyfile(raw.file(f"frame_{f}.png"), files[f])
        m = image.meta
        return {"image": image_packet(out, files, m["width"], m["height"], working_space())}  # PNG the model wrote: sRGB = the working space


NODES = (Inverse, Relight)
