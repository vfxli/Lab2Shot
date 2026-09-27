"""Nodes provided by the OpenDelight extension (code GPL-3.0, weights non-commercial)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (Official, measured_param, MissingFrames, RawOutput, NodeDef, NodeParams, P, Port, basecolor_map,
                          basecolor_port, frame_maps, Cost, Licence, Measured)


class Delight(NodeDef):
    id = "opendelight.delight"
    # 上游 test.py：img_path（人脸画面）-> output_face_torch（去光照的脸）+ final_mask（有效区域）
    official = Official(
        cite="third_party/opendelight/repo/test.py:248-300",
        takes={"image": "img_path"},
        gives={"basecolor": "output_face_torch", "mask": "final_mask"},
        note="上游把两样存成同一张 RGBA（save_image(torch.cat([res_torch, final_mask[:, :1]], dim=1))，第 293-300 行）；"
             "我们拆成两个口。上游自己跑抠像和人脸对齐，不吃遮罩。",
    )
    on_node = ("resolution", "enhancer")
    # 正面或接近正面、脸够大；一帧只处理一张脸，一帧一帧单独算（视频会闪）；
    # 遮罩按面部分割去掉头发，是 0 / 1 的选区
    inputs = (Port("image", "image.3", "RGB"),)
    # 「基础色」的口名和标签在 kit/ports.py 写一次：CG 流程里 basecolor 就是 albedo，
    # 和 DiffusionRenderer 的「基础色」是同一样东西
    outputs = (basecolor_port(), Port("mask", "image.1", "遮罩"))
    runtime = "opendelight"
    # RTX 4090 / 5090（1280×534 48 帧，默认参数）：峰值 5.36 / 5.43 GB（其中 PyTorch 2.79），逐帧算、不随帧数涨
    cost = Cost(gpu=True, vram_gb=5.5, seconds_per_frame=0.15)
    licence = Licence(note="OpenDelight 代码是 GPL-3.0，权重没有声明许可证且训练数据仅限学术：按非商用对待，只能用于研究和评估。")

    class Params(NodeParams):
        enhancer: bool = P(True, label="细节增强", help="多跑一个增强网络，毛孔、皱纹等细节更好，每帧慢约 20%；只要大致颜色时可以关", group="去光照")
        resolution: Literal[512] = measured_param(
            "处理分辨率", {512: Measured("772×855 0.15 秒/帧；1920×1080 0.34 秒/帧", gb=6.6)}, default=512,
            group="去光照",
            help="脸部裁切后送进网络的尺寸。512 是上游默认，也是目前唯一一档")
        smooth_landmarks: bool = P(True, label="关键点平滑", help="视频输入时平滑每帧的脸部定位，减少裁切框抖动；自动识别镜头切换和不相关的照片，不会把它们混在一起", group="去光照")

    @classmethod
    def cook(cls, ctx):
        image = ctx.input("image")
        raw = RawOutput(ctx.run_worker(image), MissingFrames.SKIP)
        return frame_maps(ctx, raw, image, {
            "basecolor": basecolor_map("basecolor"),  # display-referred sRGB-ish from the network
            "mask": ("image.1", "alpha", None),
        }, stage="写出基础色")


NODES = (Delight,)
