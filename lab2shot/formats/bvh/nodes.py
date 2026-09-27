"""The BVH module's node (the interface: lab2shot/nodes/formats.py): 「导入 BVH」 lists and reads a .bvh in the core's
environment (reader.py). Writing BVH is not offered: a skeleton animation goes out through an output setting that
holds skeletons."""

from __future__ import annotations

from typing import Literal

from ...data.units import to_cm
from ...nodes.base import NodeParams, P
from ...nodes.formats import ArraysImport, import_file_param, selection_param, selection_ports
from . import SUFFIXES


class ImportBvh(ArraysImport):
    id = "core.import_bvh"
    suffixes = SUFFIXES
    on_node = ("unit",)

    class Params(NodeParams):
        path: str = import_file_param(SUFFIXES, " BVH 文件（.bvh）")
        skeletons: list[str] = selection_param("skeletons")
        unit: Literal["cm", "m"] = P("cm", label="单位", group="BVH", option_labels={"cm": "厘米", "m": "米"},
                                     worker=False, help="BVH 不记录单位：多数动捕软件导出的是厘米。读进来都换成厘米")
    outputs = selection_ports(Params)

    @classmethod
    def axes(cls, params, top):
        from ...data.scene_arrays import Axes

        return Axes.of(to_cm(params["unit"]), "y")  # BVH is Y up

    @classmethod
    def arrays(cls, params, ctx=None):
        from . import reader

        return reader.arrays(cls.path(params))


NODES = (ImportBvh,)
