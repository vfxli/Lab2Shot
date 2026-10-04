"""Nodes provided by the π³ extension (weights CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, P, Param, WholeShotDepthCamera, WholeShotParams, loops_param, max_frames_param,
                          measured_param, Cost, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "pi3.reconstruct"
    metric = Param("model").one_of("pi3x")  # Pi3X gives metres; Pi3 has arbitrary units (DepthCamera.metric)
    # Pi3.forward 只吃 imgs，所以不声明 `takes_mask`，节点上没有遮罩口（那是给真的吃遮罩的上游留的，如 MonST3R）。
    # 想只重建画面的一部分，在送进「RGB」口之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）。
    native_points = "points"  # worker 交出 Pi3 自己的 local_points（官方 points 就是它乘上 camera_poses）
    # 结果变了就加一，让旧缓存作废（7：raw 里带官方的 points，运动像素不抹掉）
    version = 8  # 8: its metric output (Pi3X) is no longer scaled by 「尺度」
    # 模型看的是整幅画面，运动物体也在里面；长焦 Focal Length 偏小 10–15%；Pi3X 的米制只是大致；每段帧数上限 150
    runtime = "pi3"
    # Pi3.forward(imgs) 交出 points / local_points / conf / camera_poses（pi3.py:211-215）。
    official = Official(
        cite="third_party/pi3/repo/pi3/models/pi3.py:173-215",
        takes={"image": "imgs"},
        gives={"depth": "local_points", "camera": "camera_poses", "points": "points"},
    )
    confidence = Confidence("probability")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # 默认（官方 25.5 万像素，16:9 为 672x378）：RTX 5090 上 80 帧实测峰值 13.1 GB；150 帧一次按 worker 的每千个
    # patch 0.036 GB 推到约 15.5 GB，24 GB 的卡放得下（504 档 150 帧 RTX 4090 实测 13.8 GB）
    cost = Cost(gpu=True, vram_gb=15.5, seconds_per_frame=0.253)

    class Params(WholeShotParams):
        model: Literal["pi3x", "pi3"] = P("pi3x", group="solve")
        # 留空 = 官方做法（默认）：按画面比例取 25.5 万像素以内的尺寸（pi3/utils/basic.py:11 PIXEL_LIMIT，16:9 为
        # 672x378），worker 的 input_size(resolution=None)。三个数字档是长边，比官方小，只在要省显存时选
        resolution: Literal[280, 392, 504] | None = measured_param(
            {280: Measured(below=504), 392: Measured(below=504), 504: Measured(gb=13.8)}, auto=True, group="solve",
            words="kit.resolution")
        max_frames: Literal[50, 100, 150] = max_frames_param(
            {50: Measured(below=150), 100: Measured(below=150), 150: Measured(gb=13.8)},
            default=150)  # 24G 显卡：150 帧 13.8 GB，一次性看完整段（不是流式）
        loops: bool = loops_param()


NODES = (Reconstruct,)
