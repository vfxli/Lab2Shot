"""Nodes provided by the MEMFOF extension (BSD-3: commercial use allowed)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, OpticalFlow, OpticalFlowParams, flow_resolution_param, Cost, Measured)


class Flow(OpticalFlow):
    id = "memfof.flow"
    version = 2  # 2：原尺寸低于约 1080p 的素材按官方评测先放大 2 倍再算（worker.py upscale_of）
    # 上游 demo.py：frames_tensor（连续三帧）-> output["flow"] 的前后两个方向
    official = Official(
        cite="third_party/memfof/repo/demo.py:111-115",
        takes={"image": "frames_tensor"},
        gives={"flow": "forward_flow"},
    )
    # 逐三帧算，每帧的矢量只描述到相邻一帧；帧数没有上限
    runtime = "memfof"
    # RTX 4090，原尺寸（resolution 留空）
    cost = Cost(gpu=True, vram_gb=2.3, seconds_per_frame=0.48)

    class Params(OpticalFlowParams):
        resolution: Literal[960, 1920] | None = flow_resolution_param(
            {960: Measured(gb=0.6), 1920: Measured(gb=2.3)})


NODES = (Flow,)
