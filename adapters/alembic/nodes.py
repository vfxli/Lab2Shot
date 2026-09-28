"""The Alembic format module's nodes (the interface: lab2shot/nodes/formats.py): 「导入 Alembic」 lists and reads an .abc
through this extension's worker (worker.py alembic.import: every item of the file as scene arrays), 「Alembic 输出设置」
writes the scene to .abc through it (alembic.output). Alembic records neither its unit nor its up axis: the import node
says them (单位, 上轴). Alembic stores seconds; the archive's DCC FPS hint (the one the writer sets) turns them into
frame numbers, and a file without one reads at 24."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from lab2shot.sdk import (Axes, DEFAULT_WIDTH, NodeParams, DistinctNames, OutputSettings, P, Param, Port, WorkerImport, Writes, fps_param, import_file_param, name_param, pack, scene_arrays, selection_param, selection_ports, to_cm)

SUFFIXES = (".abc",)
UNIT_LABELS = {"cm": "厘米 · Maya", "m": "米 · Houdini"}
LISTED_FROM = ("path",)


class ImportAlembic(WorkerImport):
    id = "alembic.import"
    # 相机读自文件，并非由本节点解算；Alembic 的点缓存是单个随时间变化的对象，导入节点原样输出文件内容
    runtime = "alembic"
    suffixes = SUFFIXES
    on_node = ("unit",)
    # seconds_per_frame：仅使用 CPU，读取 300 帧的相机耗时不足 1 秒
    cost = replace(WorkerImport.cost, seconds_per_frame=0.003, note="只用 CPU")

    class Params(NodeParams):  # Alembic holds cameras, models, point clouds and curves: no skeletons
        path: str = import_file_param(SUFFIXES)
        camera: str = selection_param("camera", LISTED_FROM)
        models: list[str] = selection_param("models", LISTED_FROM)
        points: list[str] = selection_param("points", LISTED_FROM)
        curves: list[str] = selection_param("curves", LISTED_FROM)
        unit: Literal["cm", "m"] = P("cm", label="单位", group="Alembic", option_labels=UNIT_LABELS, worker=False)
        up: Literal["y", "z"] = P("y", label="上轴", group="Alembic", option_labels={"y": "Y 轴向上", "z": "Z 轴向上"},
                                  worker=False)
        width: int = P(DEFAULT_WIDTH, label="画面宽度", unit="px", gt=0, group="Alembic", worker=False, applies=Param("camera").set())
    outputs = selection_ports(Params)

    @classmethod
    def axes(cls, params, top):
        return Axes.of(to_cm(params["unit"]), params["up"])

    @classmethod
    def picture_width(cls, params):
        return params["width"]


class AlembicOutput(OutputSettings):
    id = "alembic.output"
    category = "out_scene"
    inputs = (Port("scene", "scene|scene[]", "场景", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name", "unit")
    runtime = "alembic"
    cost = replace(OutputSettings.cost, whole="写一个文件，时间看写多少东西（300 帧的网格和相机约 1 秒），不按帧算")
    writes = {
        "model": Writes.full(),
        "camera": Writes.full(),
        "points": Writes.full(),
        "curves": Writes.full(),
        "skeleton": Writes.no("Alembic 没有骨骼"),
        "character": Writes.no("Alembic 没有骨骼", via="core.bake_model"),
    }

    class Params(NodeParams):
        name: str = name_param("alembic")
        unit: Literal["cm", "m"] = P("cm", label="单位",
                                     group="文件", option_labels=UNIT_LABELS)
        fps: float = fps_param()  # Alembic 以秒记录时间，按此帧率与帧号换算

    @classmethod
    def write(cls, ctx) -> str:
        out = cls.out_file(ctx, ".abc")
        ctx.stage("整理场景")
        npz = scene_arrays(pack(ctx.inputs["scene"], ctx.work / "scene"), 1.0 / to_cm(ctx.params["unit"]),
                          fps=float(ctx.params["fps"])).save(ctx.work / "scene.npz")
        ctx.run_worker(None, extra={"file": str(out)}, inputs={"scene": npz}, reuse=False)  # it writes the file
        return out.name


NODES = (ImportAlembic, AlembicOutput)
