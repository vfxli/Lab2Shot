"""Nodes provided by the π³ extension (weights CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, P, WholeShotDepthCamera, WholeShotParams, loops_param, max_frames_param,
                          resolution_param, unit_cm_param, Cost, Licence, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "pi3.reconstruct"
    # Pi3.forward 只吃 imgs，所以不声明 `takes_mask`，节点上没有遮罩口（那是给真的吃遮罩的上游留的，如 MonST3R）。
    # 想只重建画面的一部分，在送进「RGB」口之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）。
    native_points = "points"  # worker 交出 Pi3 自己的 local_points（官方 points 就是它乘上 camera_poses）
    # 结果变了就加一，让旧缓存作废（7：raw 里带官方的 points，运动像素不抹掉）
    version = 7
    # 模型看的是整幅画面，运动物体也在里面；长焦 Focal Length 偏小 10–15%；Pi3X 的米制只是大致；每段帧数上限 150
    runtime = "pi3"
    # Pi3.forward(imgs) 交出 points / local_points / conf / camera_poses（pi3.py:211-215）。
    official = Official(
        cite="third_party/pi3/repo/pi3/models/pi3.py:173-215",
        takes={"image": "imgs"},
        gives={"depth": "local_points", "camera": "camera_poses", "points": "points"},
        note="深度是官方相机空间点图的 Z（pi3.py:212 local_points）；相机是 pi3.py:214 camera_poses。「点云」"
             "是官方的世界点图 points（pi3.py:211）：上游那一行的定义就是 camera_poses @ local_points，所以 "
             "worker 交出 local_points 本身（同一份数据，少乘一次每段自己的位姿），"
             "由家族按拼接好的相机摆回世界。没有遮罩或人物框输入口：上游 forward 只有 imgs "
             "一个入参（pi3.py:173），遮罩进不了模型；想局部重建，在送进去之前把其余部分涂黑：「ViTDet "
             "人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口",
    )
    confidence = Confidence("probability")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # RTX 4090，150 帧一次
    cost = Cost(gpu=True, vram_gb=13.8, seconds_per_frame=0.253)
    licence = Licence(note="π³ / π³X 权重是 CC-BY-NC-4.0，只能研究用；代码 BSD-3，但其中一个位置编码文件来自 Naver（CC-BY-NC-SA）。")

    class Params(WholeShotParams):
        model: Literal["pi3x", "pi3"] = P(
            "pi3x", label="模型", group="解算",
            option_labels={"pi3x": "Pi3X", "pi3": "Pi3 原版"},
        )
        resolution: Literal[280, 392, 504] = resolution_param(
            {280: Measured(below=504), 392: Measured(below=504), 504: Measured(gb=13.8)}, default=504)
        max_frames: Literal[50, 100, 150] = max_frames_param(
            {50: Measured(below=150), 100: Measured(below=150), 150: Measured(gb=13.8)},
            default=150)  # 24G 显卡：150 帧 13.8 GB，一次性看完整段（不是流式）
        loops: bool = loops_param()
        unit_cm: float = unit_cm_param()

    @classmethod
    def prepare(cls, ctx):
        return super().prepare(ctx).with_(notes={"scale": "metric" if ctx.params["model"] == "pi3x" else "relative"})


NODES = (Reconstruct,)
