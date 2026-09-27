"""Nodes provided by the LuxDiT extension (NVIDIA OneWay Noncommercial; CogVideoX VAE under the CogVideoX License)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, LightProbe, LightProbeParams, P, Cost, Licence, Measured)


class Probe(LightProbe):
    id = "luxdit.light_probe"
    # 上游 README 的两步：inference_luxdit.py（--input_dir -> ldr_log）-> hdr_merger.py（-> hdr）
    official = Official(
        cite="third_party/luxdit/repo/README.md:106-119",
        takes={"image": "--input_dir"},
        gives={"hdri": "hdr", "preview": "ldr_log"},
        note="显示图就是上游第一步的 LDR 经纬图（ldr_log 目录里的 *_ldr.png，hdr_merger.py:60-81 把它和 log 图"
             "一起喂给 HDR 合成网络）；HDRI 是第二步写出的 .exr（hdr_merger.py:93）。上游不吃相机。",
    )
    version = 2  # image packets now always say whether they have an alpha
    # 只把取样那一帧发给它（单帧模型）；普通焦段（约 24–35 mm 全画幅等效）最好，长焦会被当成广角理解
    runtime = "luxdit"
    # RTX 4090 上量得的显存和时间
    cost = Cost(gpu=True, vram_gb=13.3, whole="只算一帧的环境光（约 29 秒一次），不是逐帧的活")
    licence = Licence(note="代码和权重是 NVIDIA OneWay Noncommercial 许可（只能研究和评估）；用到的 CogVideoX 视频 VAE 另有 CogVideoX 许可（商用需登记）。")

    class Params(LightProbeParams):
        lora_scale: float = P(0.8, label="贴近画面程度", help="真实场景 LoRA 的强度（0–1）：越高，环境图里画面看得见的部分越像原画面；越低越像模型想象的环境。一般 0.8", ge=0.0, le=1.0, group="环境光", widget="slider")
        steps: Literal[20, 30, 50] | None = measured_param(
            "去噪步数", {20: Measured("时间约为 50 步的 40%", flat=True), 30: Measured("时间约为 50 步的 60%", flat=True), 50: Measured("官方默认：生成 16 秒", flat=True)}, auto="默认",
            group="模型", help="扩散采样的步数。留空 = 官方默认（50）；调低更快、细节更少，调高很少有明显改善")
        guidance_scale: float = P(2.5, label="引导强度", help="模型多大程度上跟着画面走（官方 2.5）。环境太随意、和画面不连贯就调高（3–4）；颜色发怪、过饱和就调低", ge=1.0, le=10.0, group="模型")
        resolution: Literal["auto", "480x720", "512x512", "720x480"] = P(
            "auto", label="处理分辨率", group="模型",
            option_labels={"auto": "自动", "480x720": "竖幅 480×720", "512x512": "方形 512", "720x480": "横幅 720×480"},
            help="画面缩放到模型支持的哪种画幅。自动按画面比例选；画面很宽或很窄时可以手动试另一种",
        )


NODES = (Probe,)
