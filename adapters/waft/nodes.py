"""Nodes provided by the WAFT extension (code BSD-3; the weights state no license and were trained on research-only
datasets: non-commercial)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, OpticalFlow, OpticalFlowParams, flow_resolution_param, Cost, Licence, Measured)


class Flow(OpticalFlow):
    id = "waft.flow"
    # 上游 demo.py：model.calc_flow(image1, image2) -> output['flow']（两帧之间的矢量）
    official = Official(
        cite="third_party/waft/repo/demo.py:102-104",
        takes={"image": "image1"},
        gives={"flow": "flow"},
        note="两帧法：image1、image2 是相邻两帧，前后各算一遍。每个像素的置信度是 output['info']"
             "（demo.py:84 get_heatmap 就是拿它算的），在我们的运动矢量包里当置信度通道，不是单独的口。",
    )
    # 和 MEMFOF 同一个家族、同样的输出；两帧方法，前后两个方向各算一遍
    runtime = "waft"
    # vram_gb: RTX 4090，默认长边 960
    cost = Cost(gpu=True, vram_gb=2.2, seconds_per_frame=0.33)
    licence = Licence(note="代码 BSD-3；权重没有写明许可，而且用只许研究使用的数据集训练，按非商用处理。")

    class Params(OpticalFlowParams):
        resolution: Literal[960, 1920] | None = flow_resolution_param(
            {960: Measured(gb=2.2), 1920: Measured(gb=14.3)}, default=960)


NODES = (Flow,)
