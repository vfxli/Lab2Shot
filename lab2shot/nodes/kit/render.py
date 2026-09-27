"""透过相机看场景：逐帧把场景的网格和点云栅格化成图层（raster 在工作进程里算）。"""

from __future__ import annotations

import numpy as np
from pxr import Usd

from ...data.packet import Packet
from ...data.scene import camera_of, open_scene


def camera_for(stage: Usd.Stage, cam_prim: Usd.Prim, width: int, height: int, time: Usd.TimeCode):
    """The raster camera looking through a USD camera prim at `time`."""
    from ...data.camera import CameraSamples
    from .raster import PinholeCamera

    s = CameraSamples.from_prim(cam_prim, [time.GetValue()])
    return PinholeCamera(np.linalg.inv(s.cam_to_world[0]), float(s.focal_px(width)[0]), width, height)


def render_layer_frames(scenes: list[Packet], camera: Packet | None, frames: list[int], width: int, height: int):
    """What the camera sees of the scenes' meshes and point clouds, frame by frame: yields (frame, raster.Layers,
    camera-to-world [4,4]) in order of completion. The scene is evaluated (USD) in this process; rasterization runs
    in worker processes."""

    from ...data.evaluate import scene_meshes, scene_points
    from ...io.parallel import in_flight, process_pool
    from .raster import Mesh, PointCloud, render_layers

    stage = open_scene(scenes)
    cam_stage, cam_prim = camera_of(scenes, camera)
    views = {}

    def jobs():
        for frame in frames:
            t = Usd.TimeCode(frame)
            cam = camera_for(cam_stage, cam_prim, width, height, t)
            views[frame] = np.linalg.inv(cam.world_to_cam)
            meshes = [Mesh(pts, faces) for _, pts, faces in scene_meshes(stage, t)]
            clouds = [PointCloud(pts, widths) for _, pts, widths, _ in scene_points(stage, t)]
            yield frame, render_layers, cam, meshes, clouds

    with process_pool() as pool:
        for frame, layers in in_flight(pool, jobs(), most=16, ordered=False):
            yield frame, layers, views[frame]
