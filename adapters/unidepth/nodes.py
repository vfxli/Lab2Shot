"""Nodes provided by the UniDepth extension (CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Official, Confidence, CameraLensParams, PerFrameDepthCamera, P, precision_level_param, Cost, Licence


class Geometry(PerFrameDepthCamera):
    id = "unidepth.geometry"
    # 模型自己就输出每个像素的三维点（worker 的 npz 里 points，相机空间、米）：「点云」口直接用它，
    # 不拿深度 + Focal Length 反投影绕一圈（families/base.py native_points）
    native_points = "points"
    # measured on public benchmarks; shown as the inputs' tooltips
    measured = {
        "focal_mm": "实测（10 个镜头）：填真实 Focal Length，AbsRel 中位数 0.081 → 0.069，但逐个镜头有好有坏（4 好 3 差），整体算没区别。接 AnyCalib 估的 Focal Length 更差（2 好 6 差）",
    }
    # docs.md：每帧单独计算，没有时序平滑，深度会轻微闪动（约 1% 尺度闪动）；
    # 填真实 Focal Length AbsRel 中位数 0.081 → 0.069（逐个镜头有好有坏），自己估的 Focal Length 在长焦上偏短 27%
    runtime = "unidepth"
    # 官方的输入等于解算器的输入、输出等于输出：
    # UniDepthV2.infer(rgb, camera=None) 交出 intrinsics / depth / points / rays / confidence / radius。
    # camera 那个参数只吃 3x3 内参（unidepthv2.py:272 断言「camera tensor should be of shape (..., 3, 3)」），
    # 不是完整相机，所以它在我们这儿是「Focal Length」「Filmback」两个参数，不是相机输入口。
    official = Official(
        cite="third_party/unidepth/repo/unidepth/models/unidepthv2/unidepthv2.py:241-338",
        takes={"image": "rgb"},
        gives={"depth": "depth", "camera": "intrinsics", "points": "points"},
        note="「相机」口只有内参（out[\"intrinsics\"], unidepthv2.py:330），没有外参；"
             "「点云」是模型自己的 out[\"points\"]（unidepthv2.py:336）",
    )
    confidence = Confidence("exp_error")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # vram_gb: RTX 4090 上测得（docs.md），默认 ViT-L
    cost = Cost(gpu=True, vram_gb=3.2, seconds_per_frame=0.065)
    licence = Licence(note="代码和权重 CC-BY-NC-4.0，只能研究用。")

    class Params(PerFrameDepthCamera.Params):  # 家族的 Params：镜头 + 点云间隔 / 点的大小（口上接了东西才起作用）
        model: Literal["unidepth-v2-vitl14", "unidepth-v2-vitb14", "unidepth-v2-vits14"] = P(
            "unidepth-v2-vitl14", label="模型", group="几何",
            option_labels={"unidepth-v2-vitl14": "ViT-L", "unidepth-v2-vitb14": "ViT-B", "unidepth-v2-vits14": "ViT-S"},
            help="越大越准：ViT-L 最稳；ViT-B / ViT-S 更快、更省，适合预览",
        )
        resolution_level: int = precision_level_param()


NODES = (Geometry,)
