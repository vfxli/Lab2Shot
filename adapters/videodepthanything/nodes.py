"""Nodes provided by the Video Depth Anything extension (Small models Apache-2.0; Base / Large CC-BY-NC-4.0)."""

from __future__ import annotations

from typing import Literal


from .extension import OPTION_LICENCES
from lab2shot.sdk import (licence_traits, rgb_port, Official, All, measured_param, MissingFrames, Job, WorkerNode, NodeParams, P, Port, depth_maps,
                          fp16_param, frame_maps, Cost, Licence, OptionTrait, Param, Measured)


# Largest processing size per encoder that keeps ~4 GB free on a 24 GB card (measured on 16:9 plates, 32-frame
# window: Large 686 -> 19.6 GB, Base 868 -> 15.6 GB / 1036 -> 22.2 GB, Small 1036 -> 11 GB). Past the card's
# memory it spills into system RAM and a shot takes hours instead of minutes.
MAX_SIZE = {"large": 686, "base": 868}


class Depth(WorkerNode):
    id = "videodepthanything.depth"
    on_node = ("model", "resolution")
    # docs.md + worker.py：官方 32 帧重叠窗口在这里改写成流式，任何长度的镜头只在内存里留约 40 帧；
    # 默认「真实 · Small」，米制尺度只是估计（同一面墙各模型差 5.7 / 6.7 / 7.9 米）
    inputs = (rgb_port(),)
    # 一个口：真实尺度的模型给米制深度，相对的模型给相对视差——同一样官方数据（depths），是什么由数据自己说
    # （包的 scale：视差为 "disparity"，下游「深度对齐」按它自动认、当深度反投影时 expects.py 警告）。
    # 所以换模型不用改线，模板也不必按模型分卡。
    version = 2  # 2：相对模型的结果从「视差图」口挪到「深度图」口；原来这个口在相对模型下是空包，缓存不能再用
    outputs = (Port("depth", "image.1", means=("scale",)),)
    runtime = "videodepthanything"
    # 官方的输入等于解算器的输入、输出等于输出：
    # run.py:57 `depths, fps = video_depth_anything.infer_video_depth(frames, …)` —— 进去的是整段画面，
    # 出来的只有一样 depths；它是真实尺度的深度还是相对视差，由权重决定（run.py:49 `args.metric`）。
    official = Official(
        cite="third_party/videodepthanything/repo/run.py:49-57",
        takes={"image": "frames"},
        gives={"depth": "depths"},
    )
    # vram_gb: RTX 4090 上测得（docs.md），默认「真实 · Small」
    cost = Cost(gpu=True, vram_gb=2.8, seconds_per_frame=0.028)
    licence = Licence(note=True)
    traits = (
        *licence_traits(OPTION_LICENCES),
        # RTX 4090 上测得（docs.md，MAX_SIZE）：Base 518 5.2 GB、868 15.6 GB（1036 按 868 算）；Large 518 10.6 GB、686 19.6 GB（更大按 686 算）
        OptionTrait(Param('model').one_of('metric_base', 'base'), vram_gb=5.2),
        OptionTrait(All(Param('model').one_of('metric_base', 'base'), Param('resolution').one_of(868, 1036)), vram_gb=15.6),
        OptionTrait(Param('model').one_of('metric_large', 'large'), vram_gb=10.6),
        OptionTrait(All(Param('model').one_of('metric_large', 'large'), Param('resolution').one_of(686, 868, 1036)), vram_gb=19.6),
    )

    class Params(NodeParams):
        model: Literal["metric_small", "small", "metric_base", "base", "metric_large", "large"] = P(
            "metric_small", group="depth")
        resolution: Literal[518, 686, 868, 1036] = measured_param(
            {518: Measured(gb=2.8), 686: Measured(below=1036), 868: Measured(below=1036), 1036: Measured(gb=11.0)},
            default=518, group="depth")
        fp16: bool = fp16_param("depth")

    missing_frames = MissingFrames.SKIP

    @classmethod
    def prepare(cls, ctx) -> Job:
        """The processing size the worker gets: the chosen one, capped for the Base / Large encoders (MAX_SIZE)."""
        encoder = ctx.params["model"].rsplit("_", 1)[-1]
        limit = MAX_SIZE.get(encoder)
        size = ctx.params["resolution"]
        if limit and size > limit:
            ctx.say("W-VIDEODEPTHANYTHING-SIZELIMIT", param="resolution", size=size, model=encoder.capitalize(), limit=limit)
            size = limit
        return Job(ctx.input("image"), extra={"resolution": size})

    @classmethod
    def convert(cls, ctx, raw, job):
        image = job.plate
        if ctx.params["model"].startswith("metric_"):
            return {"depth": depth_maps(ctx, raw, image)}
        import numpy as np

        # 数据自己写着它是视差（不是「只知远近的深度」）：下游把它当深度反投影时按这句话警告
        disparity = {"depth": ("image.1", lambda d: (d["depth"], np.isfinite(d["depth"])), {"scale": "disparity"})}
        return frame_maps(ctx, raw, image, disparity, stage="write_disparity")


NODES = (Depth,)
