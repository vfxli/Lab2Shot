"""Nodes provided by the OpenDelight extension (code GPL-3.0, weights non-commercial)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (rgb_port, Official, measured_param, MissingFrames, WorkerNode, NodeParams, P, Port, basecolor_map,
                          basecolor_port, frame_maps, Cost, Measured)

# Peak VRAM, RTX 4090: 6.6 GB on a 1920×1080 plate (docs.md; 5.4 GB at 1280×534, 6.4 GB at 772×855). Frame by frame, so
# it does not grow with the shot; the measured setting and Cost say this one number.
PEAK_VRAM_GB = 6.6


class Delight(WorkerNode):
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
    inputs = (rgb_port(),)
    # 「基础色」的口名和标签在 kit/ports.py 写一次：CG 流程里 basecolor 就是 albedo，
    # 和 DiffusionRenderer 的「基础色」是同一样东西
    outputs = (basecolor_port(), Port("mask", "image.1", "遮罩"))
    runtime = "opendelight"
    # 峰值见 PEAK_VRAM_GB（1920×1080 最高）；逐帧算、不随帧数涨
    cost = Cost(gpu=True, vram_gb=PEAK_VRAM_GB, seconds_per_frame=0.15)

    class Params(NodeParams):
        enhancer: bool = P(True, label="细节增强", group="去光照")
        resolution: Literal[512] = measured_param(
            "处理分辨率", {512: Measured(gb=PEAK_VRAM_GB)}, default=512,
            group="去光照")
        smooth_landmarks: bool = P(True, label="关键点平滑", group="去光照")

    missing_frames = MissingFrames.SKIP

    @classmethod
    def convert(cls, ctx, raw, job):
        return frame_maps(ctx, raw, job.plate, {
            "basecolor": basecolor_map("basecolor"),  # display-referred sRGB-ish from the network
            "mask": ("image.1", "alpha", None),
        }, stage="写出基础色")


NODES = (Delight,)
