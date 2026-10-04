"""Nodes provided by the Track4World extension (research only: its licence forbids commercial and production use)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, measured_param, frame_maps, Handle, NodeParams, opencv_points_to_usd, P, PointTracker3D,
                          point_size_param, points_packet, points_params, Port, Cost, Licence, Measured)


class Track(PointTracker3D):
    id = "track4world.track"
    # 有效区域按素材比例还原，相机 fx / fy 分方向换算，跟踪点按像素中心及交付相机转换到世界坐标。
    # 输出指纹不含 worker 代码；结果语义改变时需更新 NodeDef.version，避免复用旧缓存。
    version = 2
    # 上游 demo.py forward_video3d_ff：rgbs -> traj_2d、world_points、camera_poses（自己解算相机和世界）
    official = Official(
        cite="third_party/track4world/repo/demo.py:477-525",
        takes={"image": "rgbs"},
        gives={"tracks": "traj_2d", "tracks3d": "traj_3d", "camera": "camera_poses",
               "points": "world_points", "valid": "masks"},
    )
    # 只要画面；米制尺度来自 Depth Anything 3 的米制分支，只是估计；
    # 相机是它自己解出来的（每帧一台，Depth Anything 3 骨干）；默认每段 64 帧，段间共用 8 帧
    runtime = "track4world"
    # 上游自己解相机、自己出 2D 轨迹，所以这两个口是官方的（families/tracks3d.py 的两条声明）：
    # third_party/track4world/repo/demo.py:510 `camera_poses`、:519 `traj_2d`。
    solves_camera = True
    gives_tracks2d = True
    # RTX 4090，默认 640 处理分辨率（上游默认）、每段 64 帧：1280×534 上显卡峰值约 14 GB（和 512 档同一个量级，
    # 显存主要看「每段最多帧数」和模型本身，不看分辨率档），60 帧 52.2 秒（含一次模型加载，加载本身 12–39 秒）；
    # ram_gb：建模型时 5.5 GB 的检查点和 6.8 GB 的骨干都读进内存
    cost = Cost(gpu=True, vram_gb=14.0, seconds_per_frame=0.15, ram_gb=20)
    licence = Licence(note=True)
    on_node = ("query_frame", "track_step", "max_frames")

    class Params(NodeParams):
        query_frame: int | None = P(None, group="tracking")
        track_step: int = P(8, ge=1, le=64, group="tracking")
        # 上游默认就是 640（`repo/demo.py:1145 --image_size`，用法是 `scale = min(size/H, size/W)`，
        # 也就是长边 640；`scripts/demo.sh` 没有覆盖它）。
        # 缩完之后短边至少要 256：相关金字塔 5 层，短边不够模型自己会算到 0 像素
        # （worker.py 的 MIN_SIDE 那一段写了推导）。画面越扁越容易不够：
        # 1920×1080 在 512 档是 512×256 刚好够，1280×534 在同一档是 512×192 就崩。
        resolution: Literal[256, 384, 512, 640] = measured_param(
            {256: Measured(flat=True),
             384: Measured(flat=True),
             512: Measured(flat=True),
             # 这一行不写显存数字（Measured 的说明只讲时间、尺寸、含义）：显存在 `Cost.vram_gb` 里
             640: Measured(flat=True)},
            default=640, group="model")
        max_frames: Literal[24, 64, 120] = measured_param(
            {24: Measured(gb=10.6), 64: Measured(gb=14.0), 120: Measured(gb=18.7)}, default=64, group="model")
        picks: list[str] = P([], widget="picks", group="tracking", worker=False)
        point_size: float = point_size_param(0.5)
        point_step: int = points_params()["point_step"]  # 「点云」那一口的取样间隔（接了才起作用）

    handles = (Handle("points", {"points": "picks"}),)
    queries = "dense"  # every track_step-th pixel of 参考帧; points clicked on other frames are left out (with a notice)
    # 官方一次 infer 就出这些，别的口一个都没有（official 上面）：跟踪出的 3D / 2D 点和它自己解的相机（家族的三口），
    # 外加同一个 return 里每帧稠密的世界坐标点图和有效区域
    inputs = (rgb_port(),)
    outputs = PointTracker3D.outputs + (
        Port("points", "scene.points"),
        Port("valid", "image.1"),
    )

    @classmethod
    def convert(cls, ctx, raw, job):
        """家族那三口（3D 跟踪点、2D 跟踪点、相机）之外，再把官方同一次 return 里的 world_points 和 masks 交出去。"""
        out = super().convert(ctx, raw, job)
        image = job.plate
        # 「有效区域」读 worker 另存的 valid：模型输入两边各取 64 的倍数、画面被压扁（640×480 在 640 档送进去是 640×448），
        # worker 已把 mask 拉回素材比例；缩到素材尺寸照旧由 frame_maps 做（mask 本身留在模型网格上给「点云」取样）
        out |= frame_maps(ctx, raw, image, {"valid": ("image.1", "valid", None)}, stage="write_valid")
        if "points" in ctx.wanted:
            out["points"] = cls._cloud(ctx, raw, image)
        return out

    @classmethod
    def _cloud(cls, ctx, raw, image):
        """上游 world_points（米、OpenCV 轴、方法自己的世界）→ 一片每帧的点云（厘米、Y 轴向上），
        颜色取自 worker 存下的那一帧画面。取样间隔是「点云间隔」，只取官方 masks 里的像素。"""
        import numpy as np

        frames = image.meta["frames"]
        step = int(ctx.params.get("point_step") or 4)
        xyz_of, mask_of, rgb_of = raw.maps("world_points"), raw.maps("mask"), raw.maps("rgb")
        ctx.stage("make_points")
        pts, cols = [], []
        for f in ctx.each(frames):
            world, valid, rgb = xyz_of(f), mask_of(f), rgb_of(f)
            if world is None:
                pts.append(np.zeros((0, 3), np.float32))
                cols.append(np.zeros((0, 3), np.float32))
                continue
            keep = np.asarray(valid, bool)[::step, ::step]
            r, c = np.nonzero(keep)
            r, c = r * step, c * step
            pts.append(opencv_points_to_usd(np.asarray(world, np.float64)[r, c]).astype(np.float32))
            cols.append((np.asarray(rgb, np.float32)[r, c] / 255.0) if rgb is not None
                        else np.full((len(r), 3), 0.5, np.float32))
        return points_packet(ctx.outputs["points"], frames, "track4world_points", pts, cols,
                             scale="metric", width_cm=ctx.params["point_size"],
                             info={"extension": cls.runtime}, width=image.meta["width"],
                             height=image.meta["height"])


NODES = (Track,)
