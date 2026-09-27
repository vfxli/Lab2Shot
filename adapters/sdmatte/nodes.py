"""Nodes provided by the SDMatte extension (MIT code and weights)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, GuidedMatte, NodeParams, P, fp16_param, Cost, Licence, Measured)


class Matte(GuidedMatte):
    id = "sdmatte.matte"
    # 上游 SDMatte.forward(data)：rgb = data["image"]，aux_input = data[mask / bbox_mask / point_mask] -> output
    official = Official(
        cite="third_party/sdmatte/repo/modeling/SDMatte/meta_arch.py:98-217",
        takes={"image": "rgb", "mask": "aux_input"},
        gives={"alpha": "output"},
        note="aux_input 就是「粗遮罩」那一路：aux_input_type 取 mask / bbox_mask / point_mask，"
             "节点的「提示方式」选的就是它。上游只出 alpha。",
    )
    on_node = ("guide", "resolution")  # 家族默认的 erode_dilate 不是这个节点的关键参数
    # 逐帧独立计算；官方按 1024×1024 方图计算再缩回原尺寸，所以 2K/4K 素材的发丝细节到此为止；
    # 任何主体都行（人、动物、物体、半透明物体各有训练数据）
    runtime = "sdmatte"
    # RTX 4090 / 5090（1920×1080、20 帧、默认参数）：保留峰值两张卡都是 14560 MB（分配 14474 MB）；
    # 0.672 秒/帧（4090）、1.03 秒/帧（5090，卡上还有别的活）。按结构估的 11 GB 偏低，照它派活会派到装不下的卡上。
    cost = Cost(gpu=True, vram_gb=14.5, seconds_per_frame=0.67)
    licence = Licence(note="代码和权重都是 MIT，可以商用；训练数据里有仅限研究的数据集，严格的商业交付前建议做一次法务确认。")

    class Params(NodeParams):
        guide: Literal["mask", "box"] = P(
            "mask", label="提示", group="抠像",
            option_labels={"mask": "遮罩", "box": "框"},
            help="遮罩：把粗遮罩整张交给模型指路，形状复杂、有镂空时更准；框：只取粗遮罩的外接矩形，粗遮罩本身很糙时反而更稳")
        # 1024（官方测试尺寸）保留峰值 14.5 GB（见 cost）；再大官方没验证过效果，不给填
        resolution: Literal[512, 768, 1024] = measured_param(
            "处理分辨率", {512: Measured("实测 0.15 秒/帧", below=1024), 768: Measured("比最大的一档省", below=1024),
                         1024: Measured("官方测试尺寸，最稳：0.67 秒/帧", gb=14.5)},
            default=1024, group="抠像",
            help="画面先压成这么大的方图再算，结果缩回原尺寸。1024 是官方尺寸，最稳；调小只为省显存，模型没在别的尺寸上验证过")
        transparent: bool = P(False, label="半透明主体", help="要抠的是玻璃、烟、纱、水这类整体半透明的主体就打开：模型换一套不透明度先验，边界不再当成实心物体。抠人、动物、实体道具保持关闭", group="抠像")
        fp16: bool = fp16_param("抠像", default=False)  # 上游 inference.py 全程 fp32：不替上游做这个决定


NODES = (Matte,)
