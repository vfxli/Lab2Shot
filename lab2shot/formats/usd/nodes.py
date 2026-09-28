"""The USD module's nodes (interface: lab2shot/nodes/formats.py and nodes/output.py): 「导入 USD」 reads a DCC's USD
file by kind (reader.py), and 「USD 输出设置」 writes a scene as one flattened file (data/scene.py export). USD is
Lab2Shot's native representation, so both nodes run in the core environment and support every kind."""

from __future__ import annotations

from typing import Literal

from ...nodes.base import NodeParams, P, Port
from ...nodes.expects import DistinctNames
from ...nodes.formats import ImportNode, import_file_param, selection_param, selection_ports
from ...nodes.output import OutputSettings, Writes, fps_param, name_param
from ...nodes.services import services
from . import SUFFIXES


class ImportUsd(ImportNode):
    id = "core.import_usd"
    suffixes = SUFFIXES

    class Params(NodeParams):  # USD can hold every kind found in a DCC file
        path: str = import_file_param(SUFFIXES)
        camera: str = selection_param("camera")
        models: list[str] = selection_param("models")
        points: list[str] = selection_param("points")
        curves: list[str] = selection_param("curves")
        skeletons: list[str] = selection_param("skeletons")
        characters: list[str] = selection_param("characters")
    outputs = selection_ports(Params)

    @classmethod
    def listing(cls, params):
        from . import reader

        return reader.listing(str(cls.path(params)))

    @classmethod
    def read(cls, ctx, chosen):
        from ...io import usd
        from . import reader

        path = cls.path(ctx.params)
        group = usd.import_group(ctx.label, cls.label, path.name)  # /shot/<node name or file stem>/...
        out = {}
        for port, entries in chosen.items():
            if port == "camera":
                samples = reader.camera(path, entries[0], cls.picture_size(ctx.params, entries[0]))
                out[port] = samples.write(ctx.outputs[port], name=samples.info["object"].rsplit("/", 1)[-1] or "camera",
                                          path=usd.import_path(group, entries[0].path), source=entries[0].path,
                                          customize=lambda cam: usd.mark_group(cam.GetPrim().GetStage(), group, ctx.fingerprint))
            else:
                out[port] = reader.copied(path, entries, entries[0].kind, ctx.outputs[port], group, ctx.fingerprint)
        return out


class UsdOutput(OutputSettings):
    id = "core.output_usd"
    category = "out_scene"
    inputs = (Port("scene", "scene|scene[]", "场景", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name", "unit")
    writes = {
        "model": Writes.full(),
        "camera": Writes.full(),
        "points": Writes.full(),
        "curves": Writes.full(),
        "skeleton": Writes.full(),
        "character": Writes.full(),
    }

    class Params(NodeParams):
        name: str = name_param("scene")
        format: Literal["usd", "usda"] = P(
            "usd", label="格式", group="文件", option_labels={"usd": "二进制 .usd", "usda": "文本 .usda"},
        )
        unit: Literal["cm", "m"] = P(
            "cm", label="单位", group="文件",
            option_labels={"cm": "厘米 · Maya", "m": "米 · Houdini"},
        )
        fps: float = fps_param()  # USD's timeCodesPerSecond / framesPerSecond

    @classmethod
    def write(cls, ctx) -> str:
        from ...data.scene import export, pack

        p = ctx.params
        scene = pack(ctx.inputs["scene"], ctx.work / "scene")  # multiple inputs are packed into one scene, as 「合成场景」 does
        return export(scene, cls.out_file(ctx, f".{p['format']}"), p["unit"], float(p["fps"]), services().plan.may_draw_on,
                      ctx.provenance).name


NODES = (ImportUsd, UsdOutput)
