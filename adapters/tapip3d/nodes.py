"""Nodes provided by the TAPIP3D extension (Apache-2.0 code, MIT weights: commercial use allowed)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, OwnCamera, P, PointTracker3D, PointTracks3DParams, Port, SameShot, camera_port,
                          plate_mask_port, Cost, Measured, measured_param)


class Track(PointTracker3D):
    id = "tapip3d.track"
    # 上游 inference.py：video + depths + intrinsics + extrinsics + query_point -> coords、visibs
    official = Official(
        cite="third_party/tapip3d/repo/inference.py:114-156",
        takes={"image": "video", "depth": "depths", "camera": "extrinsics", "mask": "query_point"},
        gives={"tracks3d": "coords"},
        note="**输入**侧都是官方的：上游 inference(model, video, depths, intrinsics, extrinsics, query_point)"
             "（utils/inference_utils.py:108-118）**每帧外参和内参都吃**，没有外参时才退回单位阵"
             "（inference.py:68-70「No extrinsics provided, using identity matrix」）。"
             "遮罩只决定 query_point 撒在哪。"
             "**输出侧只有「3D 跟踪点」= coords**：上游 inference() 只 return (coords, visibs)"
             "（utils/inference_utils.py:119 的返回类型注解、inference.py:124-133 的调用），存的 npz 里"
             "intrinsics / extrinsics 是**把输入原样写回去**给可视化用的（inference.py:137-156），不是它算出来的。"
             "家族 PointTracker3D 上那两个上游没有的口——「相机」（接进来那台原样透传）和「2D 跟踪点」"
             "（我们自己把三维点投影出来的）——这个节点不长出来（lab2shot/nodes/families/tracks3d.py 的"
             "solves_camera / gives_tracks2d 两条声明）。"
             "官方的 visibs 我们没丢，它装在「3D 跟踪点」每个点的可见性里。",
    )
    # 深度和相机都必须接、而且要对得上，尺度跟着接进来的相机走；整段一起跟，前后双向；
    # 所有帧的特征都放在显卡上，显存和帧数成正比（300 帧以内放得下）
    runtime = "tapip3d"
    # RTX 4090，默认的「精细」处理分辨率
    cost = Cost(gpu=True, vram_gb=10.0, seconds_per_frame=0.08)
    on_node = ("grid", "query_frame", "resolution")
    inputs = (
        Port("image", "image.3", "RGB"),
        Port("depth", "image.1", "深度图", expects=(OwnCamera(), SameShot("image"))),
        camera_port(optional=False),
        plate_mask_port("遮罩", every_frame=False),
    )

    class Params(PointTracks3DParams):
        grid: Literal[0, 10, 16] = measured_param(
            "网格点数", {0: Measured(below=16), 10: Measured(below=16), 16: Measured(gb=10.0)}, default=0, group="跟踪")
        resolution: Literal["standard", "fine"] = P(
            "fine", label="处理分辨率", group="模型", option_labels={"standard": "标准", "fine": "精细"},
        )


NODES = (Track,)
