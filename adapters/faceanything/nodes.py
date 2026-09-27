"""Nodes provided by the FaceAnything extension (CC BY-NC 4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, Invalid, PerFrameDepthCamera, Msg, NodeParams, P, Port, frame_maps, points_params,
                          Cost, Licence, OptionTrait, Param, Measured)

# 每段帧数 / 处理分辨率只给 24 GB 显卡上验证过的几档，不接受任意数字——
# 一个改过的节点图存了超出这些值的数字会被服务器拒绝（Params 的 Literal），不会真的跑起来撑爆显存。
CHUNK_CHOICES = {"8": "8", "16": "16", "24": "24"}
MAX_SIDE_CHOICES = {"504": "标准 504", "700": "更高 700", "1008": "最高 1008"}


class Solve(PerFrameDepthCamera):
    id = "faceanything.solve"
    on_node = ("mode", "max_frames", "resolution")
    # 适合脸占画面大的特写；默认「分段」16 帧一起算（前后帧一致）；深度没有真实尺度，是模型自己的单位；
    # 每帧估的 Focal Length 会飘，节点统一用整段的中位数当一个镜头；遮罩是 Robust Video Matting 抠的人物前景（软边）
    inputs = (Port("image", "image.3", "RGB"),)  # the model estimates its own lens: no camera, no Focal Length (Params)
    # two maps of its own on top of the family's (the point cloud is the family's: one unprojection)
    outputs = PerFrameDepthCamera.outputs + (Port("canonical", "image.3", "规范坐标"), Port("mask", "image.1", "面部遮罩"))
    runtime = "faceanything"
    # 上游 faceanything.predict.run_inference(model, frame_paths, mask_paths=None, …) 交出 FacePrediction：
    # depth / intrinsics / extrinsics / images / canonical / conf / valid（predict.py:11-20）。
    official = Official(
        cite="third_party/faceanything/repo/src/faceanything/predict.py:10-48",
        takes={"image": "frame_paths"},
        gives={"depth": "depth", "canonical": "canonical", "camera": "intrinsics", "mask": "valid"},
        note="「面部遮罩」就是上游那张前景遮罩（predict.py:19 valid）；上游自己用 Robust Video Matting 生成它"
             "（run_inference.py:--remove-background，src/faceanything/background.py:18-61 generate_masks），"
             "我们的 worker 照着做同一步。「相机」只用它的 intrinsics：上游默认 monocular=True，"
             "把预测的 extrinsics 换成单位阵（predict.py:42-45）",
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # vram_gb: RTX 4090，默认每段帧数 16（峰值有时到 21 GB）
    cost = Cost(gpu=True, vram_gb=15.0, seconds_per_frame=0.19)
    licence = Licence(note="FaceAnything 的代码和权重都是 CC BY-NC 4.0（非商用）：只能用于研究和评估，不能用于商业制作。")
    traits = (
        OptionTrait(Param('max_frames').one_of(8), vram_gb=13.2, seconds_per_frame=0.15),
        OptionTrait(Param('max_frames').one_of(16), vram_gb=15.0, seconds_per_frame=0.19),
        OptionTrait(Param('max_frames').one_of(24), vram_gb=18.8, seconds_per_frame=0.19),
    )
    # 每段帧数三档各自的显存（处理分辨率 504）：选更贵的一档时节点上的计算量档位跟着变

    class Params(NodeParams):
        # 「点云」口上接了东西才起作用（points_params 的 applies=WiredOut("points")）
        point_step: int = points_params()["point_step"]
        point_size: float = points_params()["point_size"]
        mode: Literal["chunk", "one_by_one"] = P(
            "chunk", label="方式", group="解算",
            option_labels={"chunk": "分段", "one_by_one": "逐帧"},
            help="分段：几帧一起算，前后帧一致，视频推荐；逐帧：每帧单独算，细节略多但会闪，适合单张照片，显存最省",
        )
        max_frames: Literal[8, 16, 24] = measured_param(
            "每段最多帧数", {8: Measured("0.15 秒/帧", gb=13.2), 16: Measured("0.19 秒/帧", gb=15.0), 24: Measured("显存最多的一档", gb=18.8)}, default=16,
            group="解算", option_labels=CHUNK_CHOICES, applies=Param("mode").one_of("chunk"),
            help="分段处理时一次送进模型的帧数（处理分辨率 504 实测）；更大的没测过、不给选，显存不够就选小一档")
        resolution: Literal[504] = measured_param(
            "处理分辨率", {504: Measured("训练尺寸：显存看每段帧数", flat=True)}, default=504,
            group="解算", option_labels=MAX_SIDE_CHOICES,
            help="长边像素。504 是模型训练尺寸，也是 24 GB 显卡上实测过的唯一一档；更高的没测过、不给选")

    @classmethod
    def prepare(cls, ctx):
        params = ctx.params
        if params["resolution"] >= 1008 and params["mode"] != "one_by_one":
            raise Invalid(Msg("E-FACEANYTHING-ONEBYONEONLY"))
        if params["resolution"] > 504 and params["mode"] == "chunk" and params["max_frames"] != 8:
            raise Invalid(Msg("E-FACEANYTHING-CHUNKTOOBIG"))
        return super().prepare(ctx)

    @classmethod
    def convert(cls, ctx, raw, job):
        """The geometry family's outputs, plus each pixel's position on the canonical face (same for every frame
        where the same point of the face is seen: for tracking and texture transfer), where the face is, and a
        colored point cloud (depth + camera, through the family-shared depth-to-points logic)."""
        import numpy as np

        image = job.plate
        out = super().convert(ctx, raw, job)
        face = {
            "canonical": ("image.3", lambda d: (d["canonical"], d["mask"].astype(bool)), {"space": "canonical"}),
            "mask": ("image.1", lambda d: d["mask"].astype(np.float32), None),
        }
        out |= frame_maps(ctx, raw, image, face, stage="写出规范坐标和面部遮罩")
        return out


NODES = (Solve,)
