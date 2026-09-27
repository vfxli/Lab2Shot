"""BiRefNet 扩展包提供的节点（代码与权重均为 MIT）。"""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (Official, measured_param, MatteNode, NodeParams, P, fp16_param, Cost, Licence, Measured)


class Matte(MatteNode):
    id = "birefnet.matte"
    # 上游 inference.py：inputs（一批画面）-> scaled_preds（经 sigmoid 的 alpha）
    official = Official(
        cite="third_party/birefnet/repo/inference.py:33-36",
        takes={"image": "inputs"},
        gives={"alpha": "scaled_preds"},
        note="上游只吃画面，没有任何提示（遮罩、框、点）；模型自己判断主体。",
    )
    on_node = ("model", "resolution")
    # 逐帧独立计算，没有帧间约束，边缘可能出现轻微的呼吸闪烁；默认「抠像」模型在头发、运动模糊处产生半透明过渡
    # 输入「图像」、输出「Alpha」、SKIP 缺帧处理以及经 matte() 写出，均由抠像家族的 `MatteNode` 实现
    runtime = "birefnet"
    # vram_gb：RTX 4090 / 5090，1280×534，默认参数，每批 4 帧时保留显存峰值 9.46 GB（已分配 5.13 GB）
    cost = Cost(gpu=True, vram_gb=9.5, seconds_per_frame=0.12)
    licence = Licence(note="代码和权重都是 MIT；训练数据里有仅限研究的数据集，严格的商业交付前建议做一次法务确认。")

    class Params(NodeParams):
        model: Literal["matting", "general", "hr_matting", "hr", "dynamic"] = P(
            "matting", label="模型", group="抠像",
            option_labels={
                "matting": "抠像",
                "general": "通用分割",
                "hr_matting": "高分辨率抠像",
                "hr": "高分辨率分割",
                "dynamic": "任意比例",
            },
            help="抠像：头发、运动模糊有半透明过渡，人和动物用它；通用分割：边缘干净接近黑白，适合硬边物体；2K/4K 素材要头发细节选高分辨率抠像",
        )
        # 24 GB 显卡上 2048（批大小自动下调）约占 9 GB；4096 未经测试，因此不提供。
        # 默认值为具体数字，不设「自动」档（留空会在参数面板上显示为 null；见 lab2shot/nodes/kit/ports.py resolution_param）。
        # 1024 是「抠像」「通用分割」的训练尺寸；「高分辨率抠像 / 高分辨率分割」按 2048 训练，需手动设为 2048。
        resolution: Literal[512, 1024, 2048] = measured_param(
            "处理分辨率", {512: Measured("比实测的一档省", below=2048), 1024: Measured("训练尺寸：网络 86 毫秒/帧", below=2048), 2048: Measured("高分辨率模型：324 毫秒/帧", gb=9.0)},
            default=1024, group="抠像",
            help="画面缩到这个尺寸再计算。1024 是「抠像」「通用分割」的训练尺寸；"
                 "选了「高分辨率抠像」「高分辨率分割」就调到 2048（它们按 2048 训练）。调大不一定更好，模型没见过")
        fp16: bool = fp16_param("抠像", "（这里用 bf16，和上游 config.py 取的一样）")


NODES = (Matte,)
