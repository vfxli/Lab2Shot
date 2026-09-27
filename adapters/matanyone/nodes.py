"""Nodes provided by the MatAnyone 2 extension (NTU S-Lab License: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, GuidedMatte, NodeParams, P, fp16_param, Cost, Licence, Measured)


class Matte(GuidedMatte):
    id = "matanyone.matte"
    # 上游 inference_matanyone2.py：image（逐帧）+ mask（第一帧遮罩）-> pha（alpha）
    official = Official(
        cite="third_party/matanyone/repo/inference_matanyone2.py:92-118",
        takes={"image": "image", "mask": "mask"},
        gives={"alpha": "pha"},
        note="上游还写了 fgr，但那是 com_np = image_np * pha + 绿幕 * (1 - pha) 的合成预览（第 111 行），"
             "不是模型的输出，所以不是一个口。",
    )
    # Benchmark figures shown as the inputs' tooltips.
    measured = {
        "mask": "实测（CRGNN 实拍、VideoMatte 绿幕共 8 个人像镜头）：粗遮罩用 BiRefNet、第一帧真实遮罩、SAM 3 差不多（抠像误差 SAD 7.6、7.7、7.7）；SAM 3 在一个镜头上找错了人（SAD 59），换 BiRefNet 就好",
    }
    on_node = ("resolution", "mask_close")  # 家族默认的 erode_dilate 这个节点没有：它做的是闭运算
    # 只看第一帧遮罩往后传，整段时间上稳定；显存不随镜头长度增长（只记最近 5 个记忆帧，300 帧和 60 帧都是约 7.5 GB）
    every_frame = False  # only the first mask with foreground is read; the memory carries it through the shot
    runtime = "matanyone"
    # RTX 4090：显存不随镜头长度涨
    cost = Cost(gpu=True, vram_gb=7.5, seconds_per_frame=0.09)
    licence = Licence(note="代码和模型是 NTU S-Lab License 1.0，只能研究用；商用要先得到作者许可。")

    class Params(NodeParams):
        # 24G 显卡上 1920 用 7.5 GB，显存随时间基本不涨；4096（4.4 倍面积）没有测过，不给填
        resolution: Literal[960, 1280, 1920] = measured_param(
            "处理分辨率", {960: Measured("比实测的一档省", below=1920), 1280: Measured("比实测的一档省", below=1920), 1920: Measured("1080×1920 原尺寸，显存不随镜头长度涨（300 帧 0.10 秒/帧）", gb=7.5)},
            default=1920, group="抠像",
            help="画面长边超过这个像素就先缩小再算（只缩不放）。1920 = 1080p 按原尺寸算，细节最好。"
                 "**官方默认是不缩**；1920 是我们实测过的最大一档（4K 原尺寸是它的 4.4 倍面积，没测过、不给选）。"
                 "拿 4K 素材测抠像时要知道：默认是先缩到 1920 再算的")
        warmup: int = P(10, label="首帧预热次数", help="开始前在第一帧上反复细化几次，让起点的 alpha 更干净（官方默认 10）。第一帧边缘不干净就调大（20–30），只是开头多花一点时间", ge=0, le=100, group="抠像")
        mask_close: int = P(10, label="遮罩修补", unit="px", help="先把粗遮罩膨胀再腐蚀这么多像素：填掉小洞、抹平锯齿（官方默认 10）。粗遮罩很干净可调小；粗遮罩有破洞、边缘锯齿就调大", ge=0, le=100, group="抠像")
        fp16: bool = fp16_param("抠像")


NODES = (Matte,)
