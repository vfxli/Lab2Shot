"""Nodes provided by the SMIRK extension (MIT code; FLAME 2020 and the training data research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import rgb_port, Official, CameraLensParams, P, Port, WorldHumans, curves_packet, Cost, window_of


class Face(WorldHumans):
    id = "smirk.face_solve"
    version = 2  # 2：位移按画面中心（principal_px）反推，不按画布中心（带不对称 overscan 时脸不再整体偏移）
    # 引的是官方的解算器本体 SmirkEncoder：一张裁好的脸（img）进去，FLAME 的一套参数出来
    # （pose_params / cam / shape_params / expression_params / eyelid_params / jaw_params）。
    official = Official(
        cite=("third_party/smirk/repo/src/smirk_encoder.py:34-133",
              "third_party/smirk/repo/src/FLAME/FLAME.py:160-190"),
        takes={"image": "img"},
        gives={"character": "shape_params", "expressions": "expression_params"},
        # 「相机」不是上游的输出：它是本节点把弱透视换成针孔时用的那台相机（原点、不动，Focal Length 为节点上的「Focal
        # Length」，不填按 50 mm），交出来让三维视图透过它看背板、交付时头和画面对得上
        ours={"camera": ""},
    )
    on_node = ("focal_mm", "crop")
    # 脸够大、正脸到大半侧脸；每帧单独计算，没有时序平滑；
    # Focal Length 只决定头离镜头的远近（不填按全画幅 50 mm 估算）。
    # 只有「RGB」一个输入口：官方的 SmirkEncoder 只吃一张裁好的脸，相机一个字节都进不去。
    # 「相机」输出口（家族的 plate_camera）：放脸用的那台原点静止针孔相机，见 official 的 ours
    inputs = (rgb_port(),)
    plate_camera = True
    outputs = WorldHumans.outputs + (Port("expressions", "curves"),)
    runtime = "smirk"
    camera_to_worker = None
    default_focal_mm = 50.0  # SMIRK's camera is orthographic: the pinhole it is turned into needs a lens
    # RTX 4090
    cost = Cost(gpu=True, vram_gb=0.6, seconds_per_frame=0.13)

    class Params(CameraLensParams):
        crop: Literal["auto", "none"] = P("auto", group="face")

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px（同 gvhmr）：去畸变画面带 overscan 时送去的是整块
        画布，画面中心不在画布中心；放脸用的针孔相机主点在画面中心，位移按它反推，不按画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})

    @classmethod
    def convert(cls, ctx, raw, job):
        """The world-humans outputs, plus the face's 52 expression curves (held over frames without a face)."""
        import numpy as np

        image = job.plate
        out = super().convert(ctx, raw, job)
        d = raw.arrays("person_01.npz")
        frames, own = image.meta["frames"], d["frames"]
        weights = d["blendshape_weights"]
        values = np.stack([np.interp(frames, own, weights[:, k]) for k in range(weights.shape[1])], 1)  # frames without a face: held
        out["expressions"] = curves_packet(ctx.outputs["expressions"], frames,
                                           [str(n) for n in d["blendshape_names"]], values, extension=cls.runtime)
        return out


NODES = (Face,)
