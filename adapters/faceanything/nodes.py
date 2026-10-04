"""Nodes provided by the FaceAnything extension (CC BY-NC 4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, measured_param, Confidence, PerFrameDepthCamera, NodeParams, P, Port, frame_maps,
                          Cost, OptionTrait, Param, Measured)

# 每段帧数 / 处理分辨率只给 24 GB 显卡上验证过的几档，不接受任意数字——
# 一个改过的节点图存了超出这些值的数字会被服务器拒绝（Params 的 Literal），不会真的跑起来撑爆显存。


class Solve(PerFrameDepthCamera):
    id = "faceanything.face_maps"
    metric = False  # depth in the model's own units (DepthCamera.metric)
    on_node = ("mode", "max_frames", "resolution")
    # 适合脸占画面大的特写；默认「分段」16 帧一起算（前后帧一致）；深度没有真实尺度，是模型自己的单位；
    # 每帧估的 Focal Length 会飘，节点统一用整段的中位数当一个镜头；遮罩是 Robust Video Matting 抠的人物前景（软边）
    inputs = (rgb_port(),)  # the model estimates its own lens: no camera, no Focal Length (Params)
    # two maps of its own on top of the family's (the point cloud is the family's: one unprojection)
    outputs = PerFrameDepthCamera.outputs + (Port("canonical", "image.3", data=True), Port("mask", "image.1"))
    runtime = "faceanything"
    # 上游 faceanything.predict.run_inference(model, frame_paths, mask_paths=None, …) 交出 FacePrediction：
    # depth / intrinsics / extrinsics / images / canonical / conf / valid（predict.py:11-20）。
    official = Official(
        cite="third_party/faceanything/repo/src/faceanything/predict.py:10-48",
        takes={"image": "frame_paths"},
        gives={"depth": "depth", "canonical": "canonical", "camera": "intrinsics", "mask": "valid"},
    )
    confidence = Confidence("exp_plus_one")  # how its model gives its confidence (CONFIDENCE_SCALES)
    # vram_gb: RTX 4090，默认每段帧数 16（峰值有时到 21 GB）
    cost = Cost(gpu=True, vram_gb=15.0, seconds_per_frame=0.19)
    traits = (
        OptionTrait(Param('max_frames').one_of(8), vram_gb=13.2, seconds_per_frame=0.15),
        OptionTrait(Param('max_frames').one_of(16), vram_gb=15.0, seconds_per_frame=0.19),
        OptionTrait(Param('max_frames').one_of(24), vram_gb=18.8, seconds_per_frame=0.19),
    )
    # 每段帧数三档各自的显存（处理分辨率 504）：选更贵的一档时节点上的计算量档位跟着变

    class Params(NodeParams):
        mode: Literal["chunk", "one_by_one"] = P(
            "chunk", group="solve")
        max_frames: Literal[8, 16, 24] = measured_param({8: Measured(gb=13.2), 16: Measured(gb=15.0), 24: Measured(gb=18.8)}, default=16,
            group="solve", applies=Param("mode").one_of("chunk"))
        resolution: Literal[504] = measured_param({504: Measured(flat=True)}, default=504, group="solve")

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
        out |= frame_maps(ctx, raw, image, face, stage="write_maps")
        return out


NODES = (Solve,)
