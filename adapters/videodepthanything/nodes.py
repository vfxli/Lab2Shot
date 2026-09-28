"""Nodes provided by the Video Depth Anything extension (Small models Apache-2.0; Base / Large CC-BY-NC-4.0)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (Official, All, measured_param, MissingFrames, RawOutput, NodeDef, NodeParams, P, Port, depth_maps,
                          empty_packet, fp16_param, frame_maps, Cost, Licence, OptionTrait, Param, Measured)


# Largest processing size per encoder that keeps ~4 GB free on a 24 GB card (measured on 16:9 plates, 32-frame
# window: Large 686 -> 19.6 GB, Base 868 -> 15.6 GB / 1036 -> 22.2 GB, Small 1036 -> 11 GB). Past the card's
# memory it spills into system RAM and a shot takes hours instead of minutes.
MAX_SIZE = {"large": 686, "base": 868}


class Depth(NodeDef):
    id = "videodepthanything.depth"
    on_node = ("model", "resolution")
    # docs.md + worker.py：官方 32 帧重叠窗口在这里改写成流式，任何长度的镜头只在内存里留约 40 帧；
    # 默认「真实 · Small」，米制尺度只是估计（同一面墙各模型差 5.7 / 6.7 / 7.9 米）
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("depth", "image.1", "深度图", means=("scale",)), Port("disparity", "image.1", "视差图", means=("scale",)))
    runtime = "videodepthanything"
    # 官方的输入等于解算器的输入、输出等于输出：
    # run.py:57 `depths, fps = video_depth_anything.infer_video_depth(frames, …)` —— 进去的是整段画面，
    # 出来的只有一样 depths；它是真实尺度的深度还是相对视差，由权重决定（run.py:49 `args.metric`）。
    official = Official(
        cite="third_party/videodepthanything/repo/run.py:49-57",
        takes={"image": "frames"},
        gives={"depth": "depths", "disparity": "depths"},
        note="上游只有一样输出 depths：选「真实」的权重时它是米制深度（走「深度图」口），"
             "选「相对」的权重时它是相对视差（走「视差图」口），另一个口空着 —— 两个口是同一样官方数据的两种情形，"
             "不是我们多加的第二种结果",
    )
    # vram_gb: RTX 4090 上测得（docs.md），默认「真实 · Small」
    cost = Cost(gpu=True, vram_gb=2.8, seconds_per_frame=0.028)
    licence = Licence(note="代码和 Small 两个模型都是 Apache-2.0，可以商用；Base 和 Large 四个模型是 CC-BY-NC-4.0，只能研究用。"
        "米制模型的训练数据含 Virtual KITTI（CC BY-NC-SA），有潜在的训练数据许可风险。")
    traits = (
        OptionTrait(Param('model').one_of('metric_base', 'base', 'metric_large', 'large'), noncommercial=True),
        # RTX 4090 上测得（docs.md，MAX_SIZE）：Base 518 5.2 GB、868 15.6 GB（1036 按 868 算）；Large 518 10.6 GB、686 19.6 GB（更大按 686 算）
        OptionTrait(Param('model').one_of('metric_base', 'base'), vram_gb=5.2),
        OptionTrait(All(Param('model').one_of('metric_base', 'base'), Param('resolution').one_of(868, 1036)), vram_gb=15.6),
        OptionTrait(Param('model').one_of('metric_large', 'large'), vram_gb=10.6),
        OptionTrait(All(Param('model').one_of('metric_large', 'large'), Param('resolution').one_of(686, 868, 1036)), vram_gb=19.6),
    )

    class Params(NodeParams):
        model: Literal["metric_small", "small", "metric_base", "base", "metric_large", "large"] = P(
            "metric_small", label="模型", group="深度",
            option_labels={
                "metric_small": "真实 · Small", "small": "相对 · Small",
                "metric_base": "真实 · Base", "base": "相对 · Base",
                "metric_large": "真实 · Large", "large": "相对 · Large",
            },
        )
        resolution: Literal[518, 686, 868, 1036] = measured_param(
            "处理分辨率", {518: Measured(gb=2.8), 686: Measured(below=1036), 868: Measured(below=1036), 1036: Measured(gb=11.0)},
            default=518, group="深度")
        fp16: bool = fp16_param("深度")

    @classmethod
    def cook(cls, ctx):
        image = ctx.input("image")
        encoder = ctx.params["model"].rsplit("_", 1)[-1]
        limit = MAX_SIZE.get(encoder)
        size = ctx.params["resolution"]
        if limit and size > limit:
            ctx.say("W-VIDEODEPTHANYTHING-SIZELIMIT", param="resolution", size=size, model=encoder.capitalize(), limit=limit)
            size = limit
        raw = RawOutput(ctx.run_worker(image, extra={"resolution": size}), MissingFrames.SKIP)
        if ctx.params["model"].startswith("metric_"):
            return {"depth": depth_maps(ctx, raw, image), "disparity": empty_packet(ctx, "disparity")}
        import numpy as np

        # 数据自己写着它是视差（不是「只知远近的深度」）：下游把它当深度反投影时按这句话警告
        disparity = {"disparity": ("image.1", lambda d: (d["depth"], np.isfinite(d["depth"])), {"scale": "disparity"})}
        return {"depth": empty_packet(ctx, "depth"), **frame_maps(ctx, raw, image, disparity, stage="写出视差图")}


NODES = (Depth,)
