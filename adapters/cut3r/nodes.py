"""Nodes provided by the CUT3R extension (CC-BY-NC-SA-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, P, WholeShotDepthCamera, WholeShotParams, conf_threshold_param,
                          resolution_param, Cost, max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "cut3r.reconstruct"
    metric = True  # metric pointmaps (approximate: trained on metric ground truth, arXiv 2501.12387) (DepthCamera.metric)
    # 上游 demo.py 的 argparse 一共 6 个参数（demo.py:42-77），没有任何遮罩，所以节点不声明 takes_mask，
    # 没有遮罩输入口。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个「RGB」口
    native_points = "points"  # worker 交出官方的 pts3d_in_other_view（放回每帧的相机空间）
    version = 8  # 8：度量输出不再被「尺度」缩放；算法或 raw 内容变化时加一，之前缓存的结果重算（NodeDef.version）
    # 普通焦段、有移动的手持 / 跟拍效果最好，固定机位能用但相机会轻微漂移；尺度接近米但不准，
    # 当成大致米制用；模型看的是整幅画面，运动物体也在里面；默认每段 64 帧
    runtime = "cut3r"
    # 上游 demo.py 只收一串画面（--seq_path → load_images），模型每帧交出 pts3d_in_self_view /
    # pts3d_in_other_view / conf_self / conf / camera_pose。
    official = Official(
        cite="third_party/cut3r/repo/demo.py:103-234",
        takes={"image": "images"},
        gives={"depth": "pts3d_in_self_view", "camera": "camera_pose", "points": "pts3d_in_other_view"},
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    on_node = ("update", "step", "max_frames")
    # 数值来自 RTX 4090；显存 O(1)（一次编码一帧），不随分段长度增长
    cost = Cost(gpu=True, vram_gb=3.6, seconds_per_frame=0.1)

    class Params(WholeShotParams):
        update: Literal["cut3r", "ttt3r"] = P(
            "cut3r", group="solve",
        )
        max_frames: Literal[32, 64, 120, 200, 792] = max_frames_param(
            {32: Measured(below=64), 64: Measured(gb=3.6), 120: Measured(below=200), 200: Measured(gb=4.3), 792: Measured(gb=7.4)},
            default=64)
        resolution: Literal[256, 384, 512] = resolution_param(
            {256: Measured(below=512), 384: Measured(below=512), 512: Measured(gb=4.3)}, default=512)
        overlap: Literal[8, 16, 32] = measured_param({8: Measured(flat=True), 16: Measured(flat=True), 32: Measured(gb=7.4)},
            default=16, group="solve")
        conf_threshold: float = conf_threshold_param(1.5)
        shared_focal: bool = P(True, group="solve")


NODES = (Reconstruct,)
