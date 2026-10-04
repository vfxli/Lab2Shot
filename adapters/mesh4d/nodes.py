"""Mesh4D 扩展提供的节点。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (rgb_port, Official, DEFORMING, ROOT_PATH, SCENE_FILE, Cost, Job, Licence, NodeParams, P, Port, RawOutput,
                          WorkerNode, create_stage, plate_mask_port, save_stage, scene_packet, unit_cm_param,
                          write_mesh)

from .model_spec import WINDOW  # 模型一次看的帧数，worker.py 用同一个

# 「面数」：该参数对显存和耗时影响很大，因此不接受任意数值，仅提供经过验证的几档。
# 默认「4 万」即官方流程：infer.py 先把生成的网格交给贴图流程（paint_pipeline，textureGenPipeline.py 的
# use_remesh=True → hy3dpaint/utils/simplify_mesh_utils.py remesh_mesh：mesh_simplify_trimesh(target_count=40000)），
# 形变用的是这份 4 万面的网格（之后 not_simplify=True 不再减）。「原样」不减面，其余档用同一个官方减面函数，
# 只改目标面数。减面在形变之前执行一次，之后整段共用同一拓扑。
FACES = {"full": 0, "100k": 100_000, "40k": 40_000, "20k": 20_000}
OFFICIAL_FACES = "40k"
# 「质量」：扩散采样步数，官方 infer.py 使用 50。
STEPS = {"fast": 25, "standard": 50, "fine": 75}


class Solve(WorkerNode):
    """抠像物体及其遮罩 -> 逐帧变形的网格（点缓存）。

    先用 Hunyuan3D-2.1 从第一帧生成网格，再由 Mesh4D 的形变网络将同一网格的顶点
    变形到每一帧；拓扑只确定一次，之后仅更新顶点位置，因此输出即 DCC 中的点缓存。
    """

    id = "mesh4d.reconstruct"
    version = 2  # 2: 默认按官方减到 4 万面
    runtime = "mesh4d"
    # Mesh4D 的形变管线输入 batch['image'] 和 batch['mask']（pipelines_video_newvae_all_nonalign_infer.py:
    # 1030-1035，遮罩属于条件输入 cond_inputs），输出逐帧形变后的顶点 deformed_verts_gen（同一拓扑，
    # 写为 gen_<帧>.obj，:1144-1158）以及生成的静止网格 registered_gen_mesh.obj（:953）。
    official = Official(
        cite="third_party/mesh4d/repo/hy3dshape/hy3dshape/pipelines_video_newvae_all_nonalign_infer.py:953-1160",
        takes={"image": "batch['image']", "mask": "batch['mask']"},
        gives={"mesh": "deformed_verts_gen", "rest": "registered_gen_mesh"},
    )
    on_node = ("faces", "quality", "unit_cm")
    inputs = (
        rgb_port(),
        plate_mask_port(optional=False),
    )
    outputs = (
        Port("mesh", "scene.model", kinds=(DEFORMING,)),
        Port("rest", "scene.model"),
    )
    main = "mesh"
    min_frames = WINDOW  # 少于 6 帧时该方法无法计算，提交前拒绝
    # 在 RTX 4090 上测得
    cost = Cost(gpu=True, vram_gb=21.5, seconds_per_frame=2.4, note=True)
    licence = Licence(note=True)

    class Params(NodeParams):
        quality: Literal["fast", "standard", "fine"] = P("standard", group="compute")
        faces: Literal["full", "100k", "40k", "20k"] = P(OFFICIAL_FACES, group="compute")
        seed: int = P(0, ge=0, le=2**31 - 1, group="compute")
        unit_cm: float = unit_cm_param()

    @classmethod
    def prepare(cls, ctx) -> Job:
        return Job(ctx.input("image"), inputs=ctx.input_files("mask"), extra={
            "steps": STEPS[ctx.params["quality"]],
            "target_faces": FACES[ctx.params["faces"]],
            "seed": int(ctx.params["seed"]),
        })

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict:
        import numpy as np

        image = job.plate
        frames = image.meta["frames"]
        result = raw.result()
        unit = float(ctx.params["unit_cm"])

        got = raw.arrays("mesh.npz")
        faces = np.asarray(got["faces"], np.int32)
        # worker 输出的顶点位于上游形变空间：Z 轴朝上（上游的 convert_matrix 将
        # Hunyuan3D 的 Y 轴朝上转换为 Z 轴朝上），包围盒最小角位于原点，最长边为 0.9。
        # 此处转换为本项目的 Y 轴朝上并按「尺度」换算为厘米；摆放位置按第一帧计算一次，整段共用
        deform = np.asarray(got["vertices"], np.float64)  # [F, V, 3]
        rest = np.asarray(got["rest"], np.float64)  # [V, 3]，同一拓扑
        place = _placement(deform[0], unit)
        vertices = place(deform).astype(np.float32)

        ctx.stage("write_mesh")
        info = {"extension": cls.runtime, "faces": ctx.params["faces"], "quality": ctx.params["quality"],
                "seed": int(ctx.params["seed"]), "unit_cm": unit,
                "windows": int(result["windows"]), "scale": "relative"}
        w, h = image.meta["width"], image.meta["height"]

        stage = create_stage(frames, info)
        write_mesh(stage, f"{ROOT_PATH}/mesh4d", vertices, faces, frames)
        save_stage(stage, ctx.outputs["mesh"] / SCENE_FILE)
        mesh = scene_packet(ctx.outputs["mesh"], frames, "scene.model", width=w, height=h)

        still = create_stage(frames, info)
        write_mesh(still, f"{ROOT_PATH}/mesh4d_rest", place(rest).astype(np.float32), faces)
        save_stage(still, ctx.outputs["rest"] / SCENE_FILE)
        rest_out = scene_packet(ctx.outputs["rest"], frames, "scene.model", width=w, height=h)

        _say(ctx, result, len(frames), int(vertices.shape[1]), int(faces.shape[0]), unit)
        return {"mesh": mesh, "rest": rest_out}


def _placement(first: "object", unit_cm: float):
    """返回将上游形变空间中的点变换到场景坐标（Y 轴朝上，厘米）的函数；变换按第一帧计算一次，整段共用。

    上游形变空间为 Z 轴朝上（其 convert_matrix 将 Hunyuan3D 的 Y 轴朝上转换为 Z 轴朝上），
    包围盒最小角位于原点，最长边为 0.9。转换为 Y 轴朝上后底面位于 Y=0，再将 X、Z 居中到原点，
    最后乘以「尺度」换算为厘米。输入 [..., 3] 数组，输出形状相同。
    """
    import numpy as np

    def to_y_up(points):
        p = np.asarray(points, np.float64)
        # (x, y, z)_形变空间 -> (x, z, -y)_场景：上游 convert_matrix 的逆变换
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

    ref = to_y_up(first)
    low, high = ref.reshape(-1, 3).min(0), ref.reshape(-1, 3).max(0)
    offset = np.array([(low[0] + high[0]) / 2, low[1], (low[2] + high[2]) / 2])

    def place(points):
        return (to_y_up(points) - offset) * unit_cm

    return place


def _say(ctx, result: dict, frames: int, verts: int, faces: int, unit: float) -> None:
    """计算完成后在节点上显示的消息：网格规模、尺度为估计值、窗口段数；每次都输出，以便说明各值的来源。"""
    windows = int(result["windows"])
    ctx.say("I-MESH4D-DONE", verts=verts, faces=faces, frames=frames, windows=windows, unit=unit)
    ctx.say("N-MESH4D-NOSCALE", unit=unit, param="unit_cm")
    if windows > 1:
        ctx.say("W-MESH4D-SEAM", frames=frames, windows=windows, seams=[int(f) for f in result["seams"]])
    if ctx.params["faces"] != OFFICIAL_FACES:
        ctx.say("N-MESH4D-SIMPLIFIED", choice=ctx.params["faces"], faces=faces, param="faces")
    megabytes = verts * frames * 3 * 4 / 1e6
    if megabytes > 200:
        ctx.say("W-MESH4D-BIGMESH", verts=verts, frames=frames, mb=megabytes, param="faces")


NODES = (Solve,)
