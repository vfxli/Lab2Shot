"""The PLY module's node (interface: lab2shot/nodes/formats.py): 「导入 PLY」 lists and reads a .ply in the core
environment (reader.py); 「PLY 输出设置」 writes 3D gaussians (Inria layout), 「点云 PLY 输出设置」 point clouds with their
colours (writer.py)."""

from __future__ import annotations

from typing import Literal
import numpy as np

from ...data.units import to_cm
from ...messages import Msg
from ...nodes.base import NodeParams, P, Port, Reads
from ...nodes.output import Format, OutputSettings, Writes, name_param
from ...data.types import SCENE_KINDS
from ...nodes.formats import ArraysImport, import_file_param, selection_param, selection_ports
from . import SUFFIXES



class ImportPly(ArraysImport):
    id = "ply.import"
    version = 4  # Inria 高斯识别与原生数据输出
    suffixes = SUFFIXES
    on_node = ("unit",)
    reads = Reads(rank=2)  # 三维高斯只有它读；模型、点云 FBX / Alembic 在前

    class Params(NodeParams):
        path: str = import_file_param(SUFFIXES)
        points: list[str] = selection_param("points")
        gaussians: list[str] = selection_param("gaussians")
        models: list[str] = selection_param("models")
        unit: Literal["cm", "m"] = P("m", group="ply", worker=False)
        up: Literal["y", "z"] = P("y", group="ply",
                                  worker=False)
    outputs = selection_ports(Params)

    @classmethod
    def axes(cls, params, top):
        from ...data.scene_arrays import Axes

        return Axes.of(to_cm(params["unit"]), params["up"])

    @classmethod
    def arrays(cls, params, ctx=None):
        from . import reader

        return reader.arrays(cls.path(params))


class GaussianPlyOutput(OutputSettings):
    id = "ply.output_gaussian"
    version = 1
    format = Format("gaussian_ply")
    category = "out_scene"
    inputs = (Port("gaussians", "scene.gaussian"),)
    on_node = ("name", "unit")
    writes = {k: Writes.full() if k == "gaussian" else Writes.no(Msg("I-PLY-ONLYGAUSSIAN")) for k in SCENE_KINDS}

    class Params(NodeParams):
        name: str = name_param("gaussian")
        unit: Literal["cm", "m"] = P("m", group="file")

    @classmethod
    def write(cls, ctx):
        from ...data.scene_arrays import scene_arrays
        from .writer import write

        from ...data.gaussian import transform_sample
        from ...errors import Invalid
        from ...messages import Msg

        src = ctx.input("gaussians")
        items = scene_arrays(src, 1.0 / to_cm(ctx.params["unit"])).items["gaussian"]
        if len(items) != 1:
            raise Invalid(Msg("E-GAUSSIAN-PLYONE", node=ctx.label, count=len(items)))
        item = items[0]
        frames = [int(f) for f in item["frames"]]
        cuts = np.cumsum(item["counts"])[:-1]
        samples = {k: np.split(item[k], cuts) for k in ("points", "scales", "rotations", "opacity", "sh")}
        # Inria PLY has no transform: the prim's local-to-world (one per frame, or one sample for a still splat set
        # under an animated transform) is baked into each written sample. One file per frame when either changes.

        count = max(len(item["counts"]), len(item["world"]))
        for i in range(count):
            values = {k: v[min(i, len(v) - 1)] for k, v in samples.items()}
            values = transform_sample(values, item["world"][min(i, len(item["world"]) - 1)])
            out = cls.out_file(ctx, ".ply", frames[i] if count > 1 else None)
            write(out, values)
        return cls.sequence_main(ctx, ".ply") if count > 1 else out.name


class PointsPlyOutput(OutputSettings):
    id = "ply.output"
    version = 1
    format = Format("points_ply")
    category = "out_scene"
    inputs = (Port("points", "scene.points"),)
    on_node = ("name", "unit")
    writes = {k: Writes.full() if k == "points" else Writes.no(Msg("I-PLY-ONLYPOINTS")) for k in SCENE_KINDS}

    class Params(NodeParams):
        name: str = name_param("points")
        unit: Literal["cm", "m"] = P("m", group="file")

    @classmethod
    def write(cls, ctx):
        from ...data.scene_arrays import scene_arrays
        from ...errors import Invalid
        from ...messages import Msg
        from .writer import write_points

        items = scene_arrays(ctx.input("points"), 1.0 / to_cm(ctx.params["unit"])).items["points"]
        if len(items) != 1:
            raise Invalid(Msg("E-POINTS-PLYONE", node=ctx.label, count=len(items)))
        item = items[0]
        frames = [int(f) for f in item["frames"]]
        cuts = np.cumsum(item["counts"])[:-1]
        points, colors = np.split(item["points"], cuts), np.split(item["colors"], cuts)
        # PLY has no transform: the prim's local-to-world (columns as vectors) is baked into each written sample. One
        # file per frame when the points or their placement change, as 「PLY 输出设置」 does for gaussians.
        count = max(len(points), len(item["world"]))
        for i in range(count):
            m = np.asarray(item["world"][min(i, len(item["world"]) - 1)], np.float64)
            p = np.asarray(points[min(i, len(points) - 1)], np.float64)
            out = cls.out_file(ctx, ".ply", frames[i] if count > 1 else None)
            write_points(out, p @ m[:3, :3].T + m[:3, 3], colors[min(i, len(colors) - 1)])
        return cls.sequence_main(ctx, ".ply") if count > 1 else out.name


NODES = (ImportPly, GaussianPlyOutput, PointsPlyOutput)
