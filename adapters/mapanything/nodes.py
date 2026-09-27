"""Nodes provided by the MapAnything extension (code Apache-2.0; weights Apache-2.0 or CC-BY-NC-4.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, LensWholeShotParams, P, WholeShotDepthCamera, Cost, Licence, OptionTrait, Param,
                          max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "mapanything.reconstruct"
    # Benchmark figures shown as the inputs' tooltips.
    measured = {
        "focal_mm": "实测（12 个镜头）：填真实 Focal Length，相机位置误差 ATE 中位数 2.37 → 2.03 cm，每个镜头典型少 25%（7 好 1 差），深度 AbsRel 0.108 → 0.093。接 AnyCalib 估的 Focal Length 没区别",
    }
    # MapAnything 的 view 字典里可选的只有内参 / 射线 / 深度 / 位姿（model.py:2059-2064），没有任何遮罩，
    # 所以家族默认的「运动物体遮罩」「人物框」输入口在这里不存在（它们是给真的吃遮罩的 MonST3R 的）。
    # 想只重建画面的一部分，在送进「图像」口之前把其余部分涂黑：人物检测 → 人物框转遮罩 → 图像相乘。
    native_points = "points"  # worker 交出 MapAnything 自己的 pts3d_cam（官方 pts3d 的相机空间形式）
    on_node = ("focal_mm", "step", "max_frames")
    # 7：去掉「运动物体遮罩」「人物框」输入口（结果里不再抹掉运动像素），raw 里多了一张 points；旧缓存作废
    version = 7
    runtime = "mapanything"
    # MapAnything.infer(views, …)：每个 view 必须有 'img'，可选 intrinsics / ray_directions / depth_z /
    # camera_poses；每个 view 交出 pts3d / pts3d_cam / intrinsics / depth_z / camera_poses / mask / conf 等
    # （model.py:2054-2116）。
    official = Official(
        cite="third_party/mapanything/repo/mapanything/models/mapanything/model.py:2028-2116",
        takes={"image": "img"},
        gives={"depth": "depth_z", "camera": "camera_poses", "points": "pts3d"},
        note="深度是 depth_z（model.py:2107），相机是 camera_poses（model.py:2110）加 intrinsics（model.py:2105）。"
             "「点云」是官方的世界点图 pts3d（model.py:2102），原来只被拿去算法线、结果本身丢了，"
             "现在补上：worker 交出它的相机空间形式 pts3d_cam（model.py:2103，同一份数据"
             "换个坐标系），由家族按拼接好的相机摆回世界。"
             "「运动物体遮罩」「人物框」两个输入口是我们自己加的：官方 view 字典里可选的只有相机内参 / 射线 /"
             "深度 / 位姿（model.py:2059-2064），没有任何遮罩，那两张图一个字节都没进模型，只在算完之后把"
             "运动像素从输出里去掉。已删除，想局部重建改走"
             "「人物检测 → 人物框转遮罩 → 图像相乘」",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # RTX 4090：默认一次约 150 帧；200 帧到 23.5 GB，几乎是 24G 卡的上限。1080×1920 150 帧约 35 秒。
    cost = Cost(gpu=True, vram_gb=18.5, seconds_per_frame=0.23)
    licence = Licence(note="代码 Apache-2.0。Apache 权重可以商用；主权重（13 个数据集训练，明显更准）是 CC-BY-NC-4.0，只能研究用。")
    traits = (
        OptionTrait(Param('model').one_of('main'), noncommercial=True),
    )

    class Params(LensWholeShotParams):
        model: Literal["main", "apache"] = P(
            "main", label="模型", group="解算",
            option_labels={"main": "主权重", "apache": "Apache"},
            help="主权重：13 个数据集训练，相机和深度都准得多，但只能研究用；Apache：只用可商用数据训练，长焦镜头上 Focal Length 会严重估小、相机偏差大，只适合普通焦段",
        )
        max_frames: Literal[50, 100, 150, 200] = max_frames_param(
            {50: Measured("比实测的一档省", below=150), 100: Measured("比实测的一档省", below=150), 150: Measured("1080×1920 约 35 秒", gb=18.5), 200: Measured("1080×1920 显存最多的一档", gb=23.5)},
            default=150, note="。16:9 一段 150 帧是实测过的一档；4:3 更吃显存，掉一档到 100")


NODES = (Reconstruct,)
