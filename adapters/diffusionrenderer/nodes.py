"""Nodes provided by the Cosmos DiffusionRenderer extension (Apache-2.0 code, NVIDIA Open Model License weights:
commercial use allowed, with NVIDIA's notices)."""

from __future__ import annotations

import shutil
from typing import Literal

from lab2shot.sdk import (Official, UNIT, measured_param, MissingFrames, RawOutput, camera_normals, HighDynamicRange, Invalid, Msg, NodeDef,
                          NodeParams, FrameCount, P, Port, basecolor_map, basecolor_port, frame_maps, image_files, image_packet, Cost,
                          Licence, Measured, working_space)

RUNTIME = "diffusionrenderer"
# Peak VRAM at the default settings (canvas 1280×704, 41 frames a window), RTX 4090: 19.1 GB on a 70-frame shot cut
# into two windows (docs.md; a single 24-frame window peaked at 17.5). The two measured settings and both nodes'
# Cost say this one number, so the scheduler and the node's rating never count on less.
DEFAULT_VRAM_GB = 19.1
LICENSE = ("代码 Apache-2.0，模型 NVIDIA Open Model License：可以商用。对外分发模型要附许可证和 NVIDIA 声明；"
           "用到它的产品或服务要写明 “Built on NVIDIA Cosmos”。")

class DiffusionParams(NodeParams):
    """What both models share: the working canvas, the windows the shot is cut into, the sampler."""

    resolution: Literal[1280, 1024, 960, 768, 640, 512] = measured_param(
        "处理分辨率", {1280: Measured(gb=DEFAULT_VRAM_GB), 1024: Measured(below=1280), 960: Measured(below=1280), 768: Measured(below=1280), 640: Measured(below=1280), 512: Measured(below=1280)},
        default=1280, group="模型",
        option_labels={"1280": "1280×704", "1024": "1024×576", "960": "960×528", "768": "768×432", "640": "640×352", "512": "512×288"},
    )
    max_frames: Literal[1, 9, 17, 25, 33, 41, 49, 57] = measured_param(
        "每段最多帧数", {1: Measured(below=41), 9: Measured(below=41), 17: Measured(below=41), 25: Measured(below=41), 33: Measured(below=41), 41: Measured(gb=DEFAULT_VRAM_GB), 49: Measured(below=57), 57: Measured(gb=17.8)},
        default=41, group="时序", option_labels={str(n): f"{n} 帧" for n in (1, 9, 17, 25, 33, 41, 49, 57)},
    )
    overlap: Literal[0, 4, 8, 12] = measured_param(
        "段间重叠", {0: Measured(flat=True), 4: Measured(flat=True), 8: Measured(flat=True), 12: Measured(flat=True)}, default=8, group="时序")
    steps: Literal[5, 10, 15] = measured_param(
        "去噪步数", {5: Measured(flat=True), 10: Measured(flat=True), 15: Measured(flat=True)},
        default=15, group="模型")
    seed: int = P(1000, label="随机种子", ge=0, le=2**31 - 1, group="模型")


def _check(params: dict) -> None:
    if params["overlap"] * 2 > params["max_frames"]:
        raise Invalid(Msg("E-DIFFUSIONRENDERER-OVERLAP", overlap=params["overlap"], max_frames=params["max_frames"], half=params["max_frames"] // 2))


class Inverse(NodeDef):
    id = "diffusionrenderer.inverse"
    # 上游 inference_inverse_renderer.py：--dataset_path（画面）-> --inference_passes 的五个通道
    official = Official(
        cite="third_party/diffusionrenderer/repo/cosmos_predict1/diffusion/inference/inference_inverse_renderer.py:60-94",
        takes={"image": "--dataset_path"},
        gives={"basecolor": "basecolor", "normal": "normal", "depth": "depth",
               "roughness": "roughness", "metallic": "metallic"},
        note="五个通道就是上游 --inference_passes 的默认值（第 63 行）；每个通道各跑一遍 7B 视频模型。",
    )
    on_node = ("resolution", "max_frames")
    version = 2
    # ram_gb: the 7B checkpoint is read into memory before it goes to the GPU (22–29 GB)
    # vram_gb: RTX 4090，默认 1280×704 / 每段 41 帧（DEFAULT_VRAM_GB）
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=9.7, ram_gb=32)
    licence = Licence(note=LICENSE)
    # 默认每段 41 帧、段间重叠 8 帧，段内时序稳定；深度是每段各自归一化的相对值，不是米制
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (
        basecolor_port(),  # 口名和标签在 kit/ports.py 写一次，和 OpenDelight 那个口是同一样东西
        Port("normal", "image.3", "法线图", means=("space",)),
        Port("depth", "image.1", "深度图", means=("scale",)),
        Port("roughness", "image.1", "粗糙度"),
        Port("metallic", "image.1", "金属度"),
    )
    runtime = RUNTIME

    class Params(DiffusionParams):
        pass

    @classmethod
    def cook(cls, ctx):
        _check(ctx.params)
        image = ctx.input("image")
        maps = {
            "basecolor": basecolor_map("basecolor"),  # the model gives it like a texture, sRGB-encoded
            "normal": camera_normals("normal"),  # the one turn into our camera (nodes/kit/maps.py)
            "depth": ("image.1", lambda d: (d["depth"], d["depth"] >= 0), {"scale": "affine"}),
            "roughness": ("image.1", "roughness", {"value_range": UNIT, "half": True}),  # 0..1
            "metallic": ("image.1", "metallic", {"value_range": UNIT, "half": True}),  # 0..1
        }
        # each channel is a run of the 7B video model (about ten seconds a frame): only the ones something is wired to
        wanted = sorted(p for p in maps if p in ctx.wanted)
        raw = RawOutput(ctx.run_worker(image, extra={"passes": wanted}), MissingFrames.SKIP)
        return frame_maps(ctx, raw, image, maps, stage="写出材质通道")


class Relight(NodeDef):
    id = "diffusionrenderer.relight"
    # 上游 inference_forward_renderer.py：dataset_path（材质通道）+ envlight_path（HDRI）-> output（重新打光的画面）
    official = Official(
        cite="third_party/diffusionrenderer/repo/cosmos_predict1/diffusion/inference/inference_forward_renderer.py:204-258",
        takes={"image": "dataset_path", "hdri": "envlight_path"},
        gives={"image": "output"},
        note="上游的 --dataset_path「should point to the output of the inverse renderer」（第 88 行）："
             "我们的「RGB」口进来的是画面，材质通道由 worker 先跑一遍拆解再送进正向模型。"
             "HDRI 走 --use_custom_envmap 那一路（process_environment_map，第 232 行）。",
    )
    version = 2  # image packets carry whether they have an alpha
    on_node = ("env_rotate", "exposure")
    # vram_gb: RTX 4090，同一 7B 模型、同一套分段参数（DEFAULT_VRAM_GB），正向渲染而不是拆解
    cost = Cost(gpu=True, vram_gb=DEFAULT_VRAM_GB, seconds_per_frame=11.7, ram_gb=32)
    licence = Licence(note=LICENSE)
    # 默认每段 41 帧、段间重叠 8 帧；真实镜头上的结果仍有斑块和彩色噪点，属实验功能
    inputs = (Port("image", "image.3", "RGB"), Port("hdri", "image.3", "HDRI", expects=(FrameCount(most=1), HighDynamicRange())))
    outputs = (Port("image", "image.3", "重新打光"),)
    runtime = RUNTIME

    class Params(DiffusionParams):
        env_rotate: float = P(0.0, label="环境光旋转", unit="°", ge=-360, le=360, group="环境光")
        exposure: float = P(0.0, label="曝光", unit="EV", ge=-20, le=20, group="环境光")

    @classmethod
    def cook(cls, ctx):
        _check(ctx.params)
        image, hdri = ctx.input("image"), ctx.input("hdri")
        hdri_files = image_files(hdri)
        raw = ctx.run_worker(image, inputs={"hdri": hdri_files[min(hdri_files)]})
        ctx.stage("写出重新打光的画面")
        out = ctx.outputs["image"]
        files = {}
        for f in ctx.each(image.meta["frames"]):
            files[f] = out / f"frame.{f}.png"
            shutil.copyfile(raw / f"frame_{f}.png", files[f])
        m = image.meta
        return {"image": image_packet(out, files, m["width"], m["height"], working_space())}  # PNG the model wrote: sRGB = the working space


NODES = (Inverse, Relight)
