"""Nodes provided by the MatAnyone 2 extension (NTU S-Lab License: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, measured_param, GuidedMatte, NodeParams, P, Port, fp16_param, Cost, Measured,
                          plate_mask_port)


class Matte(GuidedMatte):
    id = "matanyone.matte"
    version = 2  # 2：处理分辨率按短边（官方 --max_size），非 16:9 的画面缩放结果和以前按长边不同
    # 上游 inference_matanyone2.py：image（逐帧）+ mask（第一帧遮罩）-> pha（alpha）
    official = Official(
        cite="third_party/matanyone/repo/inference_matanyone2.py:92-118",
        takes={"image": "image", "mask": "mask"},
        gives={"alpha": "pha"},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（CRGNN 实拍、VideoMatte 绿幕共 8 个人像镜头）：粗遮罩用 BiRefNet、第一帧真实遮罩、SAM 3 差不多（抠像误差 SAD 7.6、7.7、7.7）；SAM 3 在一个镜头上找错了人（SAD 59），换 BiRefNet 就好
    on_node = ("resolution", "mask_close")  # 家族默认的 erode_dilate 这个节点没有：它做的是闭运算
    # 只看第一帧遮罩往后传，整段时间上稳定；显存不随镜头长度增长（只记最近 5 个记忆帧，300 帧和 60 帧都是约 7.5 GB）
    every_frame = False  # only the first mask with foreground is read; the memory carries it through the shot
    # 家族统一叫「粗遮罩」；本节点只读首帧（上一行），标签写明，免得以为整段遮罩都参与。端口名仍是 mask，连线和缓存不变。
    inputs = (rgb_port(), plate_mask_port(optional=False, every_frame=False))
    runtime = "matanyone"
    # RTX 4090：显存不随镜头长度涨
    cost = Cost(gpu=True, vram_gb=7.5, seconds_per_frame=0.09)

    class Params(NodeParams):
        # 处理分辨率按**短边**（官方 inference_matanyone2.py:50-56 的 --max_size：min(h, w) 超过它才缩小）。
        # 显存随时间基本不涨：短边 1080 约 7.5 GB，1440 约 21.6 GB（RTX 5090）；官方默认的 4K 原尺寸 32 GB 也放不下，不给选
        resolution: Literal[540, 720, 1080, 1440] = measured_param(
            {540: Measured(below=1080), 720: Measured(below=1080), 1080: Measured(gb=7.5), 1440: Measured(gb=21.6)},
            default=1080, group="matte")
        warmup: int = P(10, ge=0, le=100, group="matte")
        mask_close: int = P(10, unit="px", ge=0, le=100, group="matte")
        fp16: bool = fp16_param("matte")

NODES = (Matte,)
