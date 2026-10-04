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
    )
    version = 2  # image packets always say whether they have an alpha; version 1 results do not
    # 只把取样那一帧发给它（单帧模型）；普通焦段（约 24–35 mm 全画幅等效）最好，长焦会被当成广角理解
    runtime = "luxdit"
    # RTX 4090 上量得的显存和时间
    cost = Cost(gpu=True, vram_gb=13.3, whole=True)
    licence = Licence(note=True)

    class Params(LightProbeParams):
        lora_scale: float = P(0.8, ge=0.0, le=1.0, group="environment_light", widget="slider")
        steps: Literal[20, 30, 50] | None = measured_param({20: Measured(flat=True), 30: Measured(flat=True), 50: Measured(flat=True)}, group="model")
        guidance_scale: float = P(2.5, ge=1.0, le=10.0, group="model")
        resolution: Literal["auto", "480x720", "512x512", "720x480"] = P("auto", group="model")


NODES = (Probe,)
