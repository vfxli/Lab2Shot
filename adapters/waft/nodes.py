"""Nodes provided by the WAFT extension (code BSD-3; the weights state no license and were trained on research-only
datasets: non-commercial)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, OpticalFlow, OpticalFlowParams, flow_resolution_param, Cost, Licence, Measured)


class Flow(OpticalFlow):
    id = "waft.flow"
    version = 2  # 2：DINOv2 的注意力换成 torch SDPA（数学等价，浮点有细微差别），旧缓存重算
    # 上游 demo.py：model.calc_flow(image1, image2) -> output['flow']（两帧之间的矢量）
    official = Official(
        cite="third_party/waft/repo/demo.py:102-104",
        takes={"image": "image1"},
        gives={"flow": "flow"},
    )
    # 和 MEMFOF 同一个家族、同样的输出；两帧方法，前后两个方向各算一遍
    runtime = "waft"
    # vram_gb: RTX 4090，默认长边 960
    cost = Cost(gpu=True, vram_gb=2.2, seconds_per_frame=0.33)
    licence = Licence(note=True)

    class Params(OpticalFlowParams):
        resolution: Literal[960, 1920] | None = flow_resolution_param(
            {960: Measured(gb=2.2), 1920: Measured(gb=9.7)}, default=960)  # 1920：注意力走 SDPA 后实测（原 14.3）


NODES = (Flow,)
