"""Nodes provided by the Microsoft MoGe extension (MIT code and MoGe-3 weights)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Official, normal_port, PerFrameDepthCamera, P, Port, precision_level_param, Cost


class Geometry(PerFrameDepthCamera):
    id = "moge.geometry"
    # 模型自己就输出每个像素的三维点（worker 的 npz 里 points，相机空间、米）：「点云」口直接用它，
    # 不拿深度 + Focal Length 反投影绕一圈（families/base.py native_points）
    native_points = "points"
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（10 个公开镜头）：填真实 Focal Length，深度误差 AbsRel 中位数 0.175 → 0.134，每个镜头典型少 22%（6 好 3 差；ETH3D 上形状更准，米制尺度却偏了）。接 AnyCalib 估的 Focal Length 反而更差（3 好 5 差）
    version = 2
    # 每帧单独计算，没有前后帧约束，深度会轻微闪动；米制尺度是模型猜出来的，不是测量值
    # （长焦镜头不填 Focal Length 时视角会估宽一倍）
    outputs = PerFrameDepthCamera.outputs[:1] + (normal_port(),) + PerFrameDepthCamera.outputs[1:]
    runtime = "moge"
    # MoGeModel.infer(image, fov_x=…) 交出 points / intrinsics / depth / mask / normal 五样，这里的四个口
    # 就是其中四样（mask 在家族里当深度的有效位用，不另起一个口）。Focal Length 是参数（fov_x），不是相机输入。
    official = Official(
        cite="third_party/moge/repo/moge/model/v3.py:220-255",
        takes={"image": "image"},
        gives={"depth": "depth", "normal": "normal", "camera": "intrinsics", "points": "points"},
        note="「相机」口只有内参（intrinsics，v3.py:250 文档行），MoGe 不出外参；"
             "「点云」是模型自己的 points（v3.py:248），不是我们拿深度反投影的",
    )
    # RTX 4090，默认的标准 ViT-L
    cost = Cost(gpu=True, vram_gb=2.6, seconds_per_frame=0.15)

    class Params(PerFrameDepthCamera.Params):  # 家族的 Params：镜头 + 点云间隔 / 点的大小（口上接了东西才起作用）
        model: Literal["Ruicheng/moge-3-vitl", "Ruicheng/moge-3-vitg"] = P(
            "Ruicheng/moge-3-vitl", label="模型", group="几何",
            option_labels={"Ruicheng/moge-3-vitl": "标准 ViT-L", "Ruicheng/moge-3-vitg": "大模型 ViT-G"},
        )
        resolution_level: int = precision_level_param()


NODES = (Geometry,)
