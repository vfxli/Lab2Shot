"""Nodes provided by the CUT3R extension (CC-BY-NC-SA-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, P, WholeShotDepthCamera, WholeShotParams, conf_threshold_param,
                          resolution_param, unit_cm_param, Cost, Licence, max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "cut3r.reconstruct"
    # 上游 demo.py 的 argparse 一共 6 个参数（demo.py:42-77），没有任何遮罩，所以节点不声明 takes_mask，
    # 没有遮罩输入口。想只重建画面的一部分，在送进去之前把其余部分涂黑：
    # 「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个「RGB」口
    native_points = "points"  # worker 交出官方的 pts3d_in_other_view（放回每帧的相机空间）
    version = 7  # 算法或 raw 内容变化时加一，之前缓存的结果重算（NodeDef.version）
    # 普通焦段、有移动的手持 / 跟拍效果最好，固定机位能用但相机会轻微漂移；尺度接近米但不准，
    # 当成大致米制用；模型看的是整幅画面，运动物体也在里面；默认每段 64 帧
    runtime = "cut3r"
    # 上游 demo.py 只收一串画面（--seq_path → load_images），模型每帧交出 pts3d_in_self_view /
    # pts3d_in_other_view / conf_self / conf / camera_pose。
    official = Official(
        cite="third_party/cut3r/repo/demo.py:103-234",
        takes={"image": "images"},
        gives={"depth": "pts3d_in_self_view", "camera": "camera_pose", "points": "pts3d_in_other_view"},
        note="深度是官方自视角点图的 Z（demo.py:210 pts3d_in_self_view）；相机是它的位姿头（demo.py:218 "
             "camera_pose）加 demo.py:234 estimate_focal_knowing_depth 估的 Focal Length。「点云」"
             "是官方的另一张点图 pts3d_in_other_view（demo.py:211，世界坐标系），"
             "和自视角点图同时算出来（heads/dpt_head.py:258）：worker 用这一帧官方自己的位姿把它放回相机空间，"
             "由家族按拼接好的相机摆回世界。没有遮罩或人物框输入口：demo.py 的 argparse 里只有 --seq_path / "
             "--size 等（demo.py:42-77），模型自己不吃遮罩；想局部重建，在送进去之前把其余部分涂黑：「ViTDet "
             "人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    on_node = ("update", "step", "max_frames")
    # 数值来自 RTX 4090；显存 O(1)（一次编码一帧），不随分段长度增长
    cost = Cost(gpu=True, vram_gb=3.6, seconds_per_frame=0.1)
    licence = Licence(note="代码和权重 CC-BY-NC-SA-4.0（含 DUSt3R / CroCo 代码），只能研究用；TTT3R 的记忆更新规则是 MIT，但仍用 CUT3R 的权重。")

    class Params(WholeShotParams):
        update: Literal["cut3r", "ttt3r"] = P(
            "cut3r", label="记忆更新", group="解算",
            option_labels={"cut3r": "CUT3R", "ttt3r": "TTT3R"},
        )
        max_frames: Literal[32, 64, 120, 200, 792] = max_frames_param(
            {32: Measured(below=64), 64: Measured(gb=3.6), 120: Measured(below=200), 200: Measured(gb=4.3), 792: Measured(gb=7.4)},
            default=64)
        resolution: Literal[256, 384, 512] = resolution_param(
            {256: Measured(below=512), 384: Measured(below=512), 512: Measured(gb=4.3)}, default=512)
        overlap: Literal[8, 16, 32] = measured_param(
            "段间重叠", {8: Measured(flat=True), 16: Measured(flat=True), 32: Measured(gb=7.4)},
            default=16, group="解算")
        conf_threshold: float = conf_threshold_param(1.5)
        shared_focal: bool = P(True, label="整段同一 Focal Length", group="解算")
        unit_cm: float = unit_cm_param()


NODES = (Reconstruct,)
