"""3D 高斯重建家族：画面 → 一片 3D 高斯（静态一份，或逐帧一份）+ 同一个世界里的相机。

模型的推理在接入层（HY-World 2.0 / WorldMirror 2.0、NeoVerse），数据种类、单位与坐标在这里统一：worker 只按下面的
原始契约写出模型自己的数值，换算成项目的厘米、Y 向上都由 convert() 做，和深度与相机家族（families/depth_camera.py）
的相机一样经 data/units.py 的换算。

原始契约（worker 写、convert 读）：

    raw/gaussian_<帧>.npz   points [N,3]、scales [N,3]（线性标准差）、rotations [N,4]（w,x,y,z）、opacity [N]（线性）、
                            sh [N,C,3]（Inria 实球谐次序，DC 在前）；OpenCV 世界轴向（第一台相机在原点），模型单位。
                            静态重建只写第一帧那一份
    raw/cameras.npz         frames、K [F,3,3]（原画面像素，含外扩边缘时按数据窗口）、cam_to_world [F,4,4]（OpenCV）
    raw/result.json         标准字段 + "dynamic"（逐帧变化与否）+ "frames"（写了高斯的帧）

模型单位换厘米用「尺度」（`unit_cm_param`，节点上换算，不送 worker：改它不重算模型）。轴向换算烘进高斯自身
（位置、协方差、球谐一起，data/gaussian.py transform_sample），结果里没有附加变换，与直接读一份 PLY 得到的一样。
"""

from __future__ import annotations

import numpy as np

from .base import Job, WorkerNode
from ..base import NodeParams, P, Port
from ..kit.cameras import solved_camera
from ..kit.ports import unit_cm_param
from ...data.gaussian import transform_sample, write
from ...data.payloads import SCENE_FILE, scene_packet, window_of
from ...data.units import CV_TO_GL, opencv_poses_to_usd
from ...io import usd

FIELDS = ("points", "scales", "rotations", "opacity", "sh")


class GaussianReconstruction(WorkerNode):
    """输入画面，输出「3D 高斯」与「相机」。成员只声明运行环境、官方引文、显存和自己的推理参数。"""

    inputs = (Port("image", "image.3", data=False),)
    outputs = (Port("gaussians", "scene.gaussian"), Port("camera", "scene.camera"))
    main = "gaussians"

    class Params(NodeParams):
        step: int = P(1, ge=1, group="solve")
        unit_cm: float = unit_cm_param()

    @classmethod
    def prepare(cls, ctx) -> Job:
        return Job(plate=ctx.input("image"))

    @classmethod
    def convert(cls, ctx, raw, job):
        result = raw.result()
        dynamic = bool(result["dynamic"])
        written = [int(f) for f in result["frames"]]
        frames = written if dynamic else []  # a static reconstruction is one splat set, not a per-frame one
        unit = float(ctx.params["unit_cm"])
        to_usd = np.diag([*(CV_TO_GL * unit), 1.0])  # OpenCV axes, model units -> Y up, centimetres (a uniform scale)
        samples = []
        for _, arr in raw.frames(ctx, written if dynamic else written[:1], pattern="gaussian_{}.npz"):
            with arr:
                samples.append(transform_sample({k: np.asarray(arr[k]) for k in FIELDS}, to_usd))
        stage = usd.create_stage(frames, {"algorithm": cls.id})
        write(stage, f"{usd.ROOT_PATH}/gaussian", frames, samples)
        folder = ctx.outputs["gaussians"]
        usd.save_stage(stage, folder / SCENE_FILE)
        out = {"gaussians": scene_packet(folder, frames, "scene.gaussian")}
        with raw.arrays("cameras.npz") as cams:
            k = np.asarray(cams["K"], np.float64)
            left, top, _, _ = window_of(job.plate).overscan  # the worker's pixels include the overscan
            out["camera"] = solved_camera(ctx, job.plate, [int(f) for f in cams["frames"]], k[:, 0, 0],
                                          opencv_poses_to_usd(cams["cam_to_world"], unit),
                                          fy_px=k[:, 1, 1], principal_px=k[:, :2, 2] - [left, top],
                                          info={"extension": cls.runtime})
        return out
