"""UniK3D 扩展提供的节点（CC-BY-NC-SA-4.0，仅限研究用途）。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Confidence, PerFrameDepthCamera, P, Port, camera_normals, frame_maps, precision_level_param,
                          M_TO_CM, Cost)


class Geometry(PerFrameDepthCamera):
    id = "unik3d.depth"
    metric = True  # metric monocular 3D (DepthCamera.metric)
    # 模型直接输出每个像素的三维点（worker npz 中的 points，相机空间，单位米）：「点云」口直接使用，
    # 不经由深度 + Focal Length 反投影（families/base.py native_points）
    native_points = "points"
    # 上游不输出内参，因此不提供「相机」输出口（解算器的输出仅限官方输出）：
    # third_party/unik3d/repo/unik3d/models/unik3d.py:392-397 仅返回
    # confidence / distance / depth / points / rays / lowres_features，不含 intrinsics。
    # worker 对 `rays` 做最小二乘拟合得到的针孔属于二次加工，仅在家族内部使用，不作为输出。
    # 官方的 `rays` 和 `distance` 各有输出口；需要相机时由使用者自行拟合或提供。
    solves_camera = False
    lens = "any"  # 逐像素射线，可处理畸变镜头（「射线场」输出即模型自身的镜头）
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（10 个镜头）：填真实 Focal Length 略好，AbsRel 0.098 → 0.087（5 好 4 差）。接 AnyCalib 估的 Focal Length 更差（3 好 7 差）
    # docs.md：逐帧独立计算，无时序平滑；适用于鱼眼及未去畸变的素材；
    # 填写真实 Focal Length 略有改善（AbsRel 0.098 → 0.087，5 好 4 差）；自身针孔近似的 Focal Length 在常规焦段偏长、长焦偏短
    runtime = "unik3d"
    # 官方接口的输入输出与解算器一致：
    # UniK3D.infer(rgb, camera=None, rays=None) 输出 confidence / distance / depth / points / rays
    # （unik3d.py:375、394-397），不含 intrinsics：其相机模型即射线场本身。
    official = Official(
        cite="third_party/unik3d/repo/unik3d/models/unik3d.py:284-398",
        takes={"image": "rgb"},
        gives={"depth": "depth", "points": "points", "distance": "distance", "rays": "rays"},
    )
    confidence = Confidence("log_error")  # 模型置信度的表示方式（CONFIDENCE_SCALES）
    # vram_gb：在 RTX 4090 上以默认 ViT-L 测得（docs.md）
    cost = Cost(gpu=True, vram_gb=3.3, seconds_per_frame=0.065)

    # 官方计算的另外两项输出（unik3d.py:394、397），与「深度图」是同一次推理的三种表示：
    # depth 为相机坐标 Z，distance 为沿视线的距离（鱼眼、广角下两者差异很大），rays 为视线方向本身
    outputs = (*PerFrameDepthCamera.outputs,
               Port("distance", "image.1", means=("scale",)),
               Port("rays", "image.3", means=("space",), data=True))

    class Params(PerFrameDepthCamera.Params):  # 家族的 Params：镜头 + 点云间隔 / 点的大小（仅在相应输出口有连接时生效）
        model: Literal["unik3d-vitl", "unik3d-vitb", "unik3d-vits"] = P("unik3d-vitl", group="geometry")
        resolution_level: int = precision_level_param()

    @classmethod
    def convert(cls, ctx, raw, job):
        """家族的输出，另加模型自身计算的两项：沿视线的距离和射线场（unik3d.py:394、397）。
        仅在有下游需要时写出。"""
        out = super().convert(ctx, raw, job)
        if not ({"distance", "rays"} & ctx.wanted):
            return out
        scale = "metric" if cls.is_metric(ctx.params) else "relative"  # 与家族写「深度图」时使用相同的判断（节点的 metric 声明）
        maps = {
            # 米 → 厘米，与「深度图」单位一致；有效区域取 worker 的 mask
            "distance": ("image.1", lambda d: (d["distance"] * M_TO_CM, d["mask"].astype(bool)), {"scale": scale}),
            # 相机空间单位向量：与法线采用相同处理（OpenCV → GL 轴，缩放后重新归一化）
            "rays": camera_normals("rays", "mask"),
        }
        return {**out, **frame_maps(ctx, raw, job.plate, maps, stage="write_maps")}


NODES = (Geometry,)
