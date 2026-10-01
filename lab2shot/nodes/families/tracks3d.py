"""三维点跟踪：点在整段镜头里的三维位置，坐标在画面那台相机的世界里。"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np

from ..kit.ports import rgb_port
from ...data.packet import Packet
from ...data.units import M_TO_CM
from ...errors import Invalid, NothingToCook
from ...messages import Msg
from ..base import Port
from ..handles import Handle, entries
from .base import Job, RawOutput, WorkerNode
from ..kit.cameras import camera_port, send_camera, solved_camera
from ..kit.ports import plate_mask_port, point_size_param
from .tracks import TrackParams, track_queries, tracks
from ..applies import Cost
from ...data.contracts import meant


class PointTracks3DParams(TrackParams):
    """Parameters every 3D point tracker shares (the 2D trackers', plus how big the points are drawn)."""

    point_size: float = point_size_param()


class PointTracker3D(WorkerNode):
    """3D point tracking: points (grid, or clicked in the viewer) followed through the shot in 3D, in the world of the
    plate's camera: a tracked point cloud (each point keeps its id, says where it is seen, carries its velocity:
    locators to pin CG to in Houdini / Maya), and, when the method gives them, the same points on the picture and its
    own camera. A method
    that tracks in a given depth and camera takes them from any depth and camera node (TAPIP3D: its inputs are
    required); one that solves its own lands in a connected camera's world when there is one (Track4World).

    `queries`, where the points start: "grid" an n×n grid (网格点数) on 参考帧 and points clicked on any frame (TAPIP3D);
    "dense" every few pixels of 参考帧 (the method's own spacing) and points clicked on that frame only (Track4World).

    Raw contract (lab2shot_worker.point_tracks, 3D): raw/tracks3d.npz xyz [N,F,3] metres, and raw/tracks.npz the same
    points on the picture (the 2D contract; its visibility is theirs); raw/result.json "world": "camera" (the world of
    the camera the node sent: the shot's, Y up) or "own" (the method's own, OpenCV axes, its cameras in
    raw/cameras.npz, the reconstruction contract's), "metric" true or false.

    Outputs the tracked points (cm; every point on every frame with its id, whether it is seen and its velocity,
    coloured from the plate where it starts; their scale the depth's they were tracked in, else the method's, relative
    when brought into a connected camera's world), the 2D tracks (gives_tracks2d), and the method's own camera
    (solves_camera). Job.notes: none."""
    lens = "pinhole"  # treats the plate as a lens without distortion: says it needs undistorted plates

    inputs = (rgb_port(), plate_mask_port("遮罩", every_frame=False), camera_port())
    main = "tracks3d"
    # 口名叫 `tracks3d`，不叫 `points`：`points` 在别的口上是「点云」（每帧各算各的一片点），这里是「3D 跟踪点」
    # （每个点有编号、整段跟着同一个点）；Track4World 两样都出，同名会撞。`tracks3d` 和 2D 的 `tracks` 成对。
    outputs = (Port("tracks3d", "scene.points", "3D 跟踪点"), Port("tracks", "tracks2d", "2D 跟踪点"),
               Port("camera", "scene.camera", "相机"))
    cost = Cost(gpu=True)
    handles = (Handle("points", {"points": "picks"}),)
    Params = PointTracks3DParams
    queries: ClassVar[Literal["grid", "dense"]] = "grid"
    # 上游自己出不出这两样；只有上游真的算出来的东西才作为输出口摆在节点上。
    # `solves_camera`：上游自己解相机（Track4World 的 `camera_poses`）。False 的（TAPIP3D）那台相机只是接进来的
    #   相机原样透传——它 npz 里的 intrinsics / extrinsics 是把输入写回去给可视化用的，不是算出来的。
    # `gives_tracks2d`：上游自己出 2D 轨迹（Track4World 的 `traj_2d`）。False 的（TAPIP3D）那份 2D 点是三维点
    #   投影出来的。
    # 两样家族内部照样要用（三维点就是从 2D 点和深度来的），只是不当成输出口。
    solves_camera: bool = False
    gives_tracks2d: bool = False
    min_frames = 2  # tracking in 3D needs a second frame (refused before cooking, nodes/expects.py FrameCount)

    def __init_subclass__(cls, **kw):
        # 上游没有的那两个口不长出来（上面 solves_camera / gives_tracks2d 两条声明）
        drop = {n for n, on in (("camera", cls.solves_camera), ("tracks", cls.gives_tracks2d)) if not on}
        if drop:
            cls.outputs = tuple(p for p in cls.outputs if p.name not in drop)
        super().__init_subclass__(**kw)
    # 给出多少个点：下面的节点靠它说自己哪一档能用（applies.py Incoming）。和 2D 的点跟踪同一条声明
    fact_labels = {"points": "跟踪点数"}

    @classmethod
    def facts(cls, params: dict) -> dict:
        """How many points it gives. 「dense」 methods track every pixel: not a number known beforehand, so the
        declaration says nothing and a condition on it stays 'not known'."""
        from ..applies import Fact
        from ..base import parse_picks

        if cls.queries == "dense":
            return {}
        grid = int(params.get("grid") or 0)
        picked = len(parse_picks(cls, "picks", list(params.get("picks") or [])))  # an entry that does not read is no point
        return {"points": Fact(grid * grid + picked, cls.fact_labels["points"])}

    @classmethod
    def prepare(cls, ctx) -> Job:
        image, camera = ctx.input("image"), ctx.input("camera")
        frames = image.meta["frames"]
        picks = ctx.params["picks"]
        if cls.queries == "dense":
            ref = ctx.params["query_frame"] if ctx.params["query_frame"] is not None else frames[0]
            # frames by the one reading of handle entries (handles.entries): an entry that does not read stays in and
            # track_queries warns about it (W-HANDLE-BADPICK), never a ValueError here
            elsewhere = [p for p in picks if (got := entries(cls, "picks", [p])[0]) and got[0][0] != ref]
            if elsewhere:
                ctx.say("N-TRACKS3D-PICKSIGNORED", param="picks", count=len(elsewhere), frame=ref, node=cls.label)
            picks = [p for p in picks if p not in elsewhere]
        inputs = track_queries(ctx, image, picks)
        if cls.queries == "grid" and not ctx.params["grid"] and "points" not in inputs:  # nothing asked for: empty
            raise NothingToCook(Msg("N-TRACKS-NOQUERY"))
        # 深度口按名字认：单通道图的类型本身分不出深度和遮罩，口名才说明接上去的是什么
        depths = [p.name for p in cls.inputs if p.name == "depth" and ctx.input(p.name) is not None]
        for name in depths:
            if ctx.input(name).meta.get("scale") == "affine":
                raise Invalid(Msg("E-TRACKS3D-AFFINE"))
        inputs |= ctx.input_files(*depths)
        if camera is not None:
            from ...data.camera import CameraSamples

            inputs["camera"] = send_camera(ctx, camera, frames, CameraSamples.from_packet(camera, frames).focal_px(image.meta["width"]))
        return Job(image, inputs=inputs, camera=camera)

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from ...data.payloads import read_tracks, start_colours, tracked_points_packet
        from ...data.units import opencv_points_to_usd, opencv_poses_to_usd

        image = job.plate
        # 2D 点：上游自己出的才当输出口交出去；不出的（TAPIP3D）家族内部照样要它算三维点，写进临时文件夹
        if cls.gives_tracks2d:
            out = tracks(ctx, raw, image, cls.runtime, ctx.params.get("min_confidence"))
            tracks2d = out["tracks"]
        else:
            out = {}
            scratch = ctx.work / "_tracks2d"
            scratch.mkdir(parents=True, exist_ok=True)
            tracks2d = tracks(ctx, raw, image, cls.runtime, ctx.params.get("min_confidence"), into=scratch)["tracks"]
        flat = read_tracks(tracks2d)
        xyz = raw.arrays("tracks3d.npz")["xyz"].astype(np.float64)
        frames = image.meta["frames"]
        result = raw.result()
        if result["world"] == "own":
            xyz = opencv_points_to_usd(xyz)  # the method's OpenCV world -> Y up, cm (as its cameras: opencv_poses_to_usd)
            cams = raw.arrays("cameras.npz")
            K = np.asarray(cams["K"], np.float64)
            out["camera"] = solved_camera(ctx, image, cams["frames"], K[:, 0, 0], opencv_poses_to_usd(cams["cam_to_world"]),
                                          info={"extension": cls.runtime}, fy_px=K[:, 1, 1],
                                          principal_px=K[:, :2, 2])
        else:
            xyz = xyz * M_TO_CM  # 接进来那台相机的世界：相机本身不交出去（solves_camera=False，从它的来源接）
        depth = next((ctx.input(p.name) for p in cls.inputs if p.name == "depth"), None)
        if depth is not None:
            scale = meant(ctx, depth, "scale", "深度图")
        else:
            scale = "metric" if result.get("metric") and result["world"] == "own" else "relative"
        out["tracks3d"] = tracked_points_packet(ctx.outputs["tracks3d"], frames, "tracks", xyz,
                                                flat["visible"], scale=scale, colors=start_colours(image, flat, frames),
                                                confidence=flat.get("confidence"),
                                                width_cm=ctx.params["point_size"],
                                                info={"extension": cls.runtime}, width=image.meta["width"],
                                                height=image.meta["height"], extension=cls.runtime)
        return out
