"""The native reconstruction outputs camera calibrations and sparse points."""
from lab2shot.sdk import (Cost, Job, LensParams, Licence, MissingFrames, Not, Official, P, Param, Port, WorkerNode,
                          opencv_points_to_usd, opencv_poses_to_usd, plate_lens, points_packet,
                          rgb_port, solved_camera, unit_cm_param, window_of)


class CameraSolve(WorkerNode):
    id = "vidmap.camera_solve"
    runtime = "vidmap"
    lens = "pinhole"
    main = "camera"
    inputs = (rgb_port(),)
    outputs = (Port("camera", "scene.camera"), Port("points", "scene.points"))
    on_node = ("focal_mm", "step")
    missing_frames = MissingFrames.SKIP
    min_frames, min_frames_step = 3, "step"
    official = Official(cite=["third_party/vidmap/repo/vidmap/run.py:12-130",
                              "third_party/vidmap/repo/README.md:75-140"],
                        takes={"image": "input_data"}, gives={"camera": "reconstruction", "points": "COLMAP"})
    cost = Cost(gpu=True, vram_gb=25, ram_gb=23, measured_on="RTX 5090 32 GB", note=True)
    licence = Licence(note=True)

    class Params(LensParams):
        step: int = P(1, ge=1, le=10, group="solve")
        # 官方的「平滑轨迹」只有不给焦距（uncalib）这一套配置：填了焦距时用不上，置灰并写原因
        smooth: bool = P(False, group="solve", applies=Not(Param("focal_mm").set()))
        unit_cm: float = unit_cm_param()

    @classmethod
    def prepare(cls, ctx):
        image = ctx.input("image")
        used = plate_lens(ctx, image)
        return Job(image, lens=used, extra={"focal_px": used.focal_px})

    @classmethod
    def convert(cls, ctx, raw, job):
        import numpy as np
        data = raw.arrays("cameras.npz")
        scale = ctx.params["unit_cm"]
        K = data["K"]
        left, top, _, _ = window_of(job.plate).overscan
        out = {"camera": solved_camera(ctx, job.plate, list(data["frames"]), K[:, 0, 0],
                                       opencv_poses_to_usd(data["cam_to_world"], scale),
                                       fy_px=K[:, 1, 1], principal_px=K[:, :2, 2] - [left, top],
                                       filmback_mm=job.lens.filmback_mm, info={"extension": "vidmap", "scale": "relative"})}
        if "points" in ctx.wanted:
            points = raw.arrays("points.npz")
            out["points"] = points_packet(ctx.outputs["points"], job.plate.meta["frames"], "vidmap_points",
                                          [opencv_points_to_usd(points["xyz"], scale)], [points["rgb"] / 255.0], scale="relative")
        return out


NODES = (CameraSolve,)
