"""Nodes provided by the CUT3R extension (CC-BY-NC-SA-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, P, WholeShotDepthCamera, WholeShotParams, conf_threshold_param,
                          resolution_param, unit_cm_param, Cost, Licence, max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "cut3r.reconstruct"
    # 上游 demo.py 的 argparse 一共 6 个参数（demo.py:42-77），没有任何遮罩。家族的默认口里带着
    # 「运动物体遮罩」「人物框」，那是给真的吃遮罩的 MonST3R 留的，这里筛掉。想只重建画面的一部分，
    # 在送进去之前把其余部分涂黑：人物检测 → 人物框转遮罩 → 图像相乘 → 这个「图像」口
    native_points = "points"  # worker 交出官方的 pts3d_in_other_view（放回每帧的相机空间）
    # 7: 去掉上游没有的「运动物体遮罩」「人物框」输入口（结果里不再抹掉运动像素），补上上游自己算的
    #    三维点图。raw 里多了一张 points，结果变了，旧缓存作废
    version = 7
    # 普通焦段、有移动的手持 / 跟拍效果最好，固定机位能用但相机会轻微漂移；尺度接近米但不准，
    # 当成大致米制用；模型看的是整幅画面，运动物体也在里面；默认每段 64 帧
    runtime = "cut3r"
    # 上游 demo.py 只收一串画面（--seq_path → load_images），模型每帧交出 pts3d_in_self_view /
    # pts3d_in_other_view / conf_self / conf / camera_pose。
    official = Official(
        cite="third_party/cut3r/repo/demo.py:103-234",
        takes={"image": "images"},
        gives={"depth": "pts3d_in_self_view", "camera": "camera_pose", "points": "pts3d_in_other_view"},
        note="深度是官方自视角点图的 Z（demo.py:210 pts3d_in_self_view）；相机是它的位姿头"
             "（demo.py:218 camera_pose）加 demo.py:234 estimate_focal_knowing_depth 估的 Focal Length。"
             "「点云」是官方的另一张点图 pts3d_in_other_view（demo.py:211，世界坐标系）：它本来就和自视角点图"
             "同时算出来（heads/dpt_head.py:258），我们整份扔掉了，现在补上——worker 用"
             "这一帧官方自己的位姿把它放回相机空间，由家族按拼接好的相机摆回世界。"
             "「运动物体遮罩」「人物框」两个输入口是我们自己加的：demo.py 的 argparse 里只有 --seq_path /"
             " --size 等（demo.py:42-77），模型自己不吃遮罩，那两张图一个字节都没进模型，只在算完之后把"
             "运动像素从有效位里去掉。已删除，想局部重建改走"
             "「人物检测 → 人物框转遮罩 → 图像相乘」",
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
            help="CUT3R：原版，每读一帧就把新画面整个写进记忆，读得越久越漂，所以每 64 帧重新开始一段再拼接；"
                 "TTT3R：长镜头模式（ICLR 2026，不用重新训练），每个记忆单元按它和新画面有多相关决定写进多少，"
                 "能连续读得更久（默认 200 帧一段），固定机位漂得少，显存不变",
        )
        max_frames: Literal[32, 64, 120, 200, 792] = max_frames_param(
            {32: Measured("比实测的一档省", below=64), 64: Measured("CUT3R 默认（120 帧一段误差 11%，64 帧 7%）", gb=3.6), 120: Measured("介于 64 帧和 200 帧之间", below=200), 200: Measured("TTT3R 默认", gb=4.3), 792: Measured("一口气读完（TTT3R 会转丢方向）", gb=7.4)},
            default=64, note="。64 是 CUT3R 自己的默认（误差也最小）；换成 TTT3R 模型时它自己的默认是 200")
        resolution: Literal[256, 384, 512] = resolution_param(
            {256: Measured("比实测的一档省", below=512), 384: Measured("比实测的一档省", below=512), 512: Measured("训练尺寸：一段 200 帧 0.1 秒/帧", gb=4.3)}, default=512, note="；512 是模型的训练尺寸，最稳")
        overlap: Literal[8, 16, 32] = measured_param(
            "段间重叠", {8: Measured("每段少算 8 帧", flat=True), 16: Measured("全部实测用的值：0.1 秒/帧", flat=True), 32: Measured("每段多算 16 帧，显存看每段帧数", gb=7.4)},
            default=16, group="解算", help="相邻两段共用多少帧来对齐拼接。重叠越多拼得越稳、越慢；一般 16")
        conf_threshold: float = conf_threshold_param(1.5)
        shared_focal: bool = P(True, label="整段同一 Focal Length", help="整段镜头用一个 Focal Length（定焦镜头）。变焦镜头才关闭", group="解算")
        unit_cm: float = unit_cm_param("CUT3R 接近真实尺度，保持 100；和 ViPE 或实测距离对比后可以微调")


NODES = (Reconstruct,)
