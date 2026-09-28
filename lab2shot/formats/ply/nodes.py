"""The PLY module's node (interface: lab2shot/nodes/formats.py): 「导入 PLY」 lists and reads a .ply in the core
environment (reader.py). PLY export is not provided; point clouds are written by an output-settings node that supports
point clouds."""

from __future__ import annotations

from typing import Literal

from ...data.units import to_cm
from ...nodes.base import NodeParams, P
from ...nodes.formats import ArraysImport, import_file_param, selection_param, selection_ports
from . import SUFFIXES

UNIT_LABELS = {"cm": "厘米", "m": "米"}


class ImportPly(ArraysImport):
    id = "core.import_ply"
    suffixes = SUFFIXES
    on_node = ("unit",)

    class Params(NodeParams):
        path: str = import_file_param(SUFFIXES)
        points: list[str] = selection_param("points")
        models: list[str] = selection_param("models")
        unit: Literal["cm", "m"] = P("m", label="单位", group="PLY", option_labels=UNIT_LABELS, worker=False)
        up: Literal["y", "z"] = P("y", label="上轴", group="PLY", option_labels={"y": "Y 轴向上", "z": "Z 轴向上"},
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


NODES = (ImportPly,)
