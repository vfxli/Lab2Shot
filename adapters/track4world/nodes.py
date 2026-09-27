"""Nodes provided by the Track4World extension (research only: its licence forbids commercial and production use)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, frame_maps, Handle, NodeParams, opencv_points_to_usd, P, PointTracker3D,
                          point_size_param, points_packet, points_params, Port, Cost, Licence, Measured)

from .extension import LICENSE_NOTE


class Track(PointTracker3D):
    id = "track4world.track"
    # 上游 demo.py forward_video3d_ff：rgbs -> traj_2d、world_points、camera_poses（自己解算相机和世界）
    official = Official(
        cite="third_party/track4world/repo/demo.py:477-525",
        takes={"image": "rgbs"},
        gives={"tracks": "traj_2d", "tracks3d": "traj_3d", "camera": "camera_poses",
               "points": "world_points", "valid": "masks"},
        note="上游 model.infer(rgbs_selected, iters, sw, is_training, tracking3d)（demo.py:499-505）只吃画面："
             "没有相机、没有遮罩、没有查询点，相机是它自己解出来的（camera_poses）。"
             "原来节点上有「遮罩」和「相机」两个输入，都是我们自己加的：遮罩只在 worker 里决定网格点撒在哪几个"
             "像素上，相机只在算完之后把结果搬进那台相机的世界（worker.py 原 304-323 行），两样都一个字节没进模型。"
             "已删除——要只跟一块区域，在图上接「人物框转遮罩 → 图像相乘」把画面挡住再送进来；"
             "要把结果摆进自己那台相机的世界，用显式的小工具节点。"
             "同一天补上两个被我们丢掉的官方输出：「点云」= 上游 return 里的 world_points（每帧稠密的世界坐标点图，"
             "demo.py:519；上游 world_points = c2w @ points，nets/model.py:193，所以同一个 return 里的 points"
             "（相机空间）是同一份数据换个坐标系，不另立一个口），「有效区域」= 同一个 return 里的 masks"
             "（demo.py:520，这一帧哪些像素有可用的三维点）。",
    )
    # 只要画面；米制尺度来自 Depth Anything 3 的米制分支，只是估计；
    # 相机是它自己解出来的（每帧一台，Depth Anything 3 骨干）；默认每段 64 帧，段间共用 8 帧
    runtime = "track4world"
    # 上游自己解相机、自己出 2D 轨迹，所以这两个口是官方的（families/tracks3d.py 的两条声明）：
    # third_party/track4world/repo/demo.py:510 `camera_poses`、:519 `traj_2d`。
    solves_camera = True
    gives_tracks2d = True
    # RTX 4090，默认 640 处理分辨率（上游默认）、每段 64 帧：1280×534 上显卡峰值约 14 GB（和 512 档同一个量级，
    # 显存主要看「每段帧数」和模型本身，不看分辨率档），60 帧 52.2 秒（含一次模型加载，加载本身 12–39 秒）
    cost = Cost(gpu=True, vram_gb=14.0, seconds_per_frame=0.15, ram_gb=20)
    licence = Licence(note=LICENSE_NOTE + "：Track4World 的代码和权重只许学术研究，任何情况下都不能用于商业或生产；骨干 Depth Anything 3 权重是 CC BY-NC 4.0。")
    on_node = ("query_frame", "track_step", "max_frames")  # the 5.5 GB checkpoint and the 6.8 GB backbone are read into RAM while the model is built

    class Params(NodeParams):
        query_frame: int | None = P(
            None, label="参考帧", group="跟踪", placeholder="第一帧",
            help="从哪一帧的像素开始跟（帧号）：这一帧上每隔几个像素一个点，前后都会跟过去。选要跟的东西清楚可见的一帧",
        )
        track_step: int = P(
            8, label="跟踪点间隔", ge=1, le=64, group="跟踪",
            help="在处理分辨率的画面上每隔几个像素取一个点。8：512 宽的画面约 4 千个点，发射粒子够用；调小点更密（4 约 1.6 万个），"
                 "USD 文件和 Houdini 里更重",
        )
        # 上游默认就是 640（`repo/demo.py:1145 --image_size`，用法是 `scale = min(size/H, size/W)`，
        # 也就是长边 640；`scripts/demo.sh` 没有覆盖它）。
        # 缩完之后短边至少要 256：相关金字塔 5 层，短边不够模型自己会算到 0 像素
        # （worker.py 的 MIN_SIDE 那一段写了推导）。画面越扁越容易不够：
        # 1920×1080 在 512 档是 512×256 刚好够，1280×534 在同一档是 512×192 就崩。
        resolution: Literal[256, 384, 512, 640] = measured_param(
            "处理分辨率",
            {256: Measured("最省的一档；画面扁的时候缩完短边不到 256，算之前会被拦下", flat=True),
             384: Measured("比 640 省；同样要看缩完短边够不够 256", flat=True),
             512: Measured("比 640 省；16:9 缩成 512×256 刚好够，更扁的画面不够", flat=True),
             # 这一行不写显存数字（`tests/test_card_info.py` 守着）：显存在 `Cost.vram_gb` 里
             640: Measured("官方默认；1280×534、每段 64 帧实测：60 帧连一次模型加载共 52 秒", flat=True)},
            default=640, group="模型",
            help="长边缩到这个像素再跟踪，上游默认就是 640。缩完之后短边至少要 256（模型的相关金字塔 5 层），"
                 "不够时算之前就会说清楚要调大；画面越扁（宽高比越大）越要调大")
        max_frames: Literal[24, 64, 120] = measured_param(
            "每段最多帧数", {24: Measured("最省的一档", gb=10.6), 64: Measured("每帧 0.13–0.17 秒", gb=14.0), 120: Measured("显存最多的一档", gb=18.7)}, default=64, group="模型",
            help="一次送进模型的帧数，更长的镜头分段跟、在重叠的 8 帧上接起来并对齐到同一个世界。越长越连贯，但显存越高；显存不够时停下来，列出更小的安全选项让你选")
        picks: list[str] = P(
            [], label="手动点", widget="picks", group="跟踪", worker=False, placeholder="显示本节点，在参考帧上点要跟踪的位置",
            help="显示这个节点时，在 2D 视图里点哪就在哪加一个跟踪点（点在参考帧上），和整帧的点一起跟踪，导出时排在最前面、名字是 user_01…；"
                 "Track4World 只从参考帧出发，点在别的帧上的会忽略",
        )
        point_size: float = point_size_param(0.5)
        point_step: int = points_params()["point_step"]  # 「点云」那一口的取样间隔（接了才起作用）

    handles = (Handle("points", {"points": "picks"}),)
    queries = "dense"  # every track_step-th pixel of 参考帧; points clicked on other frames are left out (with a notice)
    # 官方一次 infer 就出这些，别的口一个都没有（official 上面）：跟踪出的 3D / 2D 点和它自己解的相机（家族的三口），
    # 外加同一个 return 里每帧稠密的世界坐标点图和有效区域
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = PointTracker3D.outputs + (
        Port("points", "scene.points", "点云",
             help="官方同一次计算顺带出的每帧稠密世界坐标点（上游 world_points = 相机 × 每帧相机空间的点；镜头分了几段算时，"
                  "按拼接后的相机放回同一个世界）：按「点云间隔」取样，"
                  "只取「有效区域」里的像素，颜色取自这一帧的画面。和「3D 跟踪点」不是一回事——这一份每帧各算各的，"
                  "点和点之间没有对应关系"),
        Port("valid", "image.1", "有效区域",
             help="官方 masks：这一帧哪些像素有可用的三维点（天空、无纹理的地方是 0）"),
    )

    @classmethod
    def convert(cls, ctx, raw, job):
        """家族那三口（3D 跟踪点、2D 跟踪点、相机）之外，再把官方同一次 return 里的 world_points 和 masks 交出去。"""
        import numpy as np

        out = super().convert(ctx, raw, job)
        image = job.plate
        out |= frame_maps(ctx, raw, image, {"valid": ("image.1", lambda d: d["mask"].astype(np.float32), None)},
                          stage="写出有效区域")
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
        ctx.stage("生成点云")
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
