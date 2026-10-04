"""Nodes provided by the DiffusionLight extension (MIT code and LoRAs; SDXL under Open RAIL++-M)."""

from __future__ import annotations

from lab2shot.sdk import (Official, LensParams, LightProbe as LightProbeBase, LightProbeParams, NEW_PICTURE, Port, Cost,
                          Licence)


class LightProbe(LightProbeBase):
    id = "diffusionlight.light_probe"
    # 上游 README 的三步：inpaint.py（画铬球，square）-> ball2envmap.py（展成经纬图）-> exposure2hdr.py（合成 hdr）
    official = Official(
        cite="third_party/diffusionlight/repo/README.md:55-87",
        takes={"image": "--dataset"},
        gives={"hdri": "hdr", "preview": "square"},
    )
    version = 2  # image packets always say whether they have an alpha; version 1 results do not
    # 只收一帧（前面接「FrameHold」选），挑环境最完整、遮挡最少的一帧；环境看得比较全的广一点的镜头最好
    outputs = tuple(p for p in LightProbeBase.outputs if p.name != "preview") + (Port("preview", "image.3", shape=NEW_PICTURE),)
    runtime = "diffusionlight"
    # vram_gb: RTX 4090，1280×534、默认参数、全新进程：PyTorch 保留峰值 13.69 GB（分配 12.46）
    cost = Cost(gpu=True, vram_gb=13.7, whole=True)
    licence = Licence(note=True)

    # 没有相机口：镜头就是两个普通参数
    class Params(LightProbeParams, LensParams):
        pass

    @classmethod
    def prepare(cls, ctx):
        """The lens unfolds the chrome ball correctly: its Focal Length (the node has no camera input), else the ball is
        unfolded as seen from infinitely far (a small error at usual focal lengths)."""
        job = super().prepare(ctx)
        return job.with_(extra={"fov_deg": job.lens.fov_x_deg})


NODES = (LightProbe,)
