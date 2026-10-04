"""The Alembic format module's nodes (the interface: lab2shot/nodes/formats.py): 「导入 Alembic」 lists and reads an .abc
through this extension's worker (worker.py alembic.import: every item of the file as scene arrays), 「Alembic 输出设置」
writes the scene to .abc through it (alembic.output). Alembic records neither its unit nor its up axis: the import node
says them (单位, 上轴). Alembic stores seconds, turned into frame numbers at the file's rate: its DCC FPS hint (the one
the writer sets), else the rate its samplings are at (Blender and Houdini write no hint; a motion-blurred file's
sub-samples cycle once a frame), else 24, said (worker.py import_file)."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from lab2shot.sdk import (Axes, DEFAULT_WIDTH, Msg, NodeParams, DistinctNames, Format, OutputSettings, P, Param, Port, Reads, WorkerImport, Writes, fps_param, import_file_param, name_param, pack, scene_arrays, selection_param, selection_ports, to_cm)

SUFFIXES = (".abc",)
LISTED_FROM = ("path",)


class ImportAlembic(WorkerImport):
    id = "alembic.import"
    # 6：没记帧率时按采样间隔（均匀或运动模糊的周期采样）认帧率，再没有才按 24
    version = 6
    # 相机读自文件，并非由本节点解算；Alembic 的点缓存是单个随时间变化的对象，导入节点原样输出文件内容
    runtime = "alembic"
    suffixes = SUFFIXES
    reads = Reads(rank=1)  # 点云、曲线只有它读；其余 FBX 在前
    on_node = ("unit",)
    # seconds_per_frame：仅使用 CPU，读取 300 帧的相机耗时不足 1 秒
    cost = replace(WorkerImport.cost, seconds_per_frame=0.003, note=True)

    class Params(NodeParams):  # Alembic holds cameras, models, point clouds and curves: no skeletons
        path: str = import_file_param(SUFFIXES)
        camera: str = selection_param("camera", LISTED_FROM)
        models: list[str] = selection_param("models", LISTED_FROM)
        points: list[str] = selection_param("points", LISTED_FROM)
        curves: list[str] = selection_param("curves", LISTED_FROM)
        unit: Literal["cm", "m"] = P("cm", group="alembic", worker=False)
        up: Literal["y", "z"] = P("y", group="alembic", 
                                  worker=False)
        width: int = P(DEFAULT_WIDTH, unit="px", gt=0, group="alembic", worker=False, applies=Param("camera").set())
    outputs = selection_ports(Params, fps=True)  # its words: node.alembic.import.port.fps.help

    @classmethod
    def axes(cls, params, top):
        return Axes.of(to_cm(params["unit"]), params["up"])

    @classmethod
    def picture_width(cls, params):
        return params["width"]


class AlembicOutput(OutputSettings):
    id = "alembic.output"
    format = Format("alembic")
    version = 4  # 4：原名写进用户属性 lab2shot:name；分区名也转写并同层去重（带「/」或转写后同名时不再整个写不出）
    category = "out_scene"
    inputs = (Port("scene", "scene|scene[]", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name", "unit")
    runtime = "alembic"
    cost = replace(OutputSettings.cost, whole=True)
    writes = {
        "model": Writes.full(),
        "camera": Writes.full(),
        "points": Writes.full(),
        "curves": Writes.full(),
        "skeleton": Writes.no(Msg("I-ALEMBIC-NOBONES")),
        "character": Writes.no(Msg("I-ALEMBIC-NOBONES"), via="bake_geometry"),
        "gaussian": Writes.no(Msg("I-ALEMBIC-NOGAUSSIAN")),
        "light": Writes.no(Msg("I-ALEMBIC-NOLIGHT")),
    }

    class Params(NodeParams):
        name: str = name_param("alembic")
        unit: Literal["cm", "m"] = P("cm", 
                                     group="file")
        fps: float = fps_param()  # Alembic 以秒记录时间，按此帧率与帧号换算

    @classmethod
    def write(cls, ctx) -> str:
        out = cls.out_file(ctx, ".abc")
        ctx.stage("gather_scene")
        npz = scene_arrays(pack(ctx.inputs["scene"], ctx.work / "scene"), 1.0 / to_cm(ctx.params["unit"]),
                          fps=float(ctx.params["fps"])).save(ctx.work / "scene.npz")
        ctx.run_worker(None, extra={"file": str(out)}, inputs={"scene": npz}, reuse=False)  # it writes the file
        return out.name


NODES = (ImportAlembic, AlembicOutput)
