"""Nodes provided by the MapAnything extension (code Apache-2.0; weights Apache-2.0 or CC-BY-NC-4.0)."""

from __future__ import annotations

from typing import Literal

from .extension import MODELS, OPTION_LICENCES
from lab2shot.sdk import (Official, licence_traits, Confidence, LensWholeShotParams, P, WholeShotDepthCamera, Cost, Licence,
                          max_frames_param, Measured)


class Reconstruct(WholeShotDepthCamera):
    id = "mapanything.reconstruct"
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（12 个镜头）：填真实 Focal Length，相机位置误差 ATE 中位数 2.37 → 2.03 cm，每个镜头典型少 25%（7 好 1 差），深度 AbsRel 0.108 → 0.093。接 AnyCalib 估的 Focal Length 没区别
    # MapAnything 的 view 字典里可选的只有内参 / 射线 / 深度 / 位姿（model.py:2059-2064），没有任何遮罩，
    # 所以家族默认的「运动物体遮罩」「人物框」输入口在这里不存在（它们是给真的吃遮罩的 MonST3R 的）。
    # 想只重建画面的一部分，在送进「RGB」口之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）。
    native_points = "points"  # worker 交出 MapAnything 自己的 pts3d_cam（官方 pts3d 的相机空间形式）
    on_node = ("focal_mm", "step", "max_frames")
    # 7：结果里保留运动像素，raw 里带一张 points；更早版本的缓存不是这样，不能复用
    version = 7
    runtime = "mapanything"
    # MapAnything.infer(views, …)：每个 view 必须有 'img'，可选 intrinsics / ray_directions / depth_z /
    # camera_poses；每个 view 交出 pts3d / pts3d_cam / intrinsics / depth_z / camera_poses / mask / conf 等
    # （model.py:2054-2116）。
    official = Official(
        cite="third_party/mapanything/repo/mapanything/models/mapanything/model.py:2028-2116",
        takes={"image": "img"},
        gives={"depth": "depth_z", "camera": "camera_poses", "points": "pts3d"},
        note="深度是 depth_z（model.py:2107），相机是 camera_poses（model.py:2110）加 "
             "intrinsics（model.py:2105）。「点云」是官方的世界点图 pts3d（model.py:2102）：worker "
             "交出它的相机空间形式 pts3d_cam（model.py:2103，同一份数据换个坐标系），"
             "由家族按拼接好的相机摆回世界。没有遮罩或人物框输入口：官方 view 字典里可选的只有相机内参 / 射线 "
             "/ 深度 / 位姿（model.py:2059-2064），没有任何遮罩；想局部重建，在送进去之前把其余部分涂黑："
             "「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # RTX 4090：默认一次约 150 帧；200 帧到 23.5 GB，几乎是 24G 卡的上限。1080×1920 150 帧约 35 秒。
    cost = Cost(gpu=True, vram_gb=18.5, seconds_per_frame=0.23)
    licence = Licence(note="代码 Apache-2.0。Apache 权重可以商用；主权重（13 个数据集训练，明显更准）是 CC-BY-NC-4.0，非商用。")
    traits = licence_traits(OPTION_LICENCES)

    class Params(LensWholeShotParams):
        model: Literal[tuple(MODELS)] = P(  # type: ignore[valid-type]
            "main", label="模型", group="解算",
            option_labels={"main": "主权重", "apache": "Apache"},
        )
        max_frames: Literal[50, 100, 150, 200] = max_frames_param(
            {50: Measured(below=150), 100: Measured(below=150), 150: Measured(gb=18.5), 200: Measured(gb=23.5)},
            default=150)


    @classmethod
    def prepare(cls, ctx):
        """The worker gets the chosen weights' folder and licence from the one table (extension.py MODELS)."""
        model = ctx.params["model"]
        if model in OPTION_LICENCES["model"]:  # non-commercial weights: said once more at the cook
            ctx.say("N-MAPANYTHING-NONCOMMERCIAL")
        repo, _revision, _sha, licence = MODELS[model]
        return super().prepare(ctx).with_(extra={"weights_repo": repo, "weights_license": licence})


NODES = (Reconstruct,)
