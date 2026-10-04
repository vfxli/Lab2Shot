"""Nodes provided by the TAPIP3D extension (Apache-2.0 code, MIT weights: commercial use allowed)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, OwnCamera, P, PointTracker3D, PointTracks3DParams, Port, SameShot, camera_port,
                          plate_mask_port, Cost, Measured, measured_param)


class Track(PointTracker3D):
    id = "tapip3d.track"
    version = 3  # 3：不足 16 帧的镜头补到一个窗口再算（原来一个窗口都不跑，点原地不动）；2：查询点用滤掉飞点后的深度抬升到三维
    # 上游 inference.py：video + depths + intrinsics + extrinsics + query_point -> coords、visibs
    official = Official(
        cite="third_party/tapip3d/repo/inference.py:114-156",
        takes={"image": "video", "depth": "depths", "camera": "extrinsics", "mask": "query_point"},
        gives={"tracks3d": "coords"},
    )
    # 深度和相机都必须接、而且要对得上，尺度跟着接进来的相机走；整段一起跟，前后双向；
    # 所有帧的特征都放在显卡上，显存和帧数成正比（300 帧以内放得下）
    runtime = "tapip3d"
    # RTX 4090，默认的「精细」处理分辨率
    cost = Cost(gpu=True, vram_gb=10.0, seconds_per_frame=0.08)
    on_node = ("grid", "query_frame", "resolution")
    inputs = (
        rgb_port(),
        Port("depth", "image.1", expects=(OwnCamera(), SameShot("image"))),
        camera_port(optional=False),
        plate_mask_port(every_frame=False),
    )

    class Params(PointTracks3DParams):
        grid: Literal[0, 10, 16] = measured_param(
            {0: Measured(below=16), 10: Measured(below=16), 16: Measured(gb=10.0)}, default=0, group="tracking")
        resolution: Literal["standard", "fine"] = P("fine", group="model")


NODES = (Track,)
