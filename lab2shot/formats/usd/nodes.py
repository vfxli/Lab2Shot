"""The USD module's nodes (interface: lab2shot/nodes/formats.py and nodes/output.py): 「导入 USD」 reads a DCC's USD
file by kind (reader.py), and 「USD 输出设置」 writes a scene as one flattened file (data/scene.py export). USD is
Lab2Shot's native representation, so both nodes run in the core environment and support every kind."""

from __future__ import annotations

from typing import Literal

from ...nodes.base import NodeParams, P, Port, Reads
from ...nodes.expects import DistinctNames
from ...nodes.formats import ImportNode, import_file_param, selection_param, selection_ports
from ...nodes.output import Format, OutputSettings, Writes, fps_param, name_param
from ...nodes.services import services
from . import SUFFIXES


class ImportUsd(ImportNode):
    id = "usd.import"
    # 原名从 customData 读取；外来文件里像转写的名字（Bone_u0041）保持原样。
    version = 6  # 原生高斯种类与逐帧外观导入
    suffixes = SUFFIXES
    reads = Reads(rank=9)  # 什么种类都读：别的格式都读不了时才用它

    class Params(NodeParams):  # USD can hold every kind found in a DCC file
        path: str = import_file_param(SUFFIXES)
        camera: str = selection_param("camera")
        models: list[str] = selection_param("models")
        points: list[str] = selection_param("points")
        gaussians: list[str] = selection_param("gaussians")
        curves: list[str] = selection_param("curves")
        skeletons: list[str] = selection_param("skeletons")
        characters: list[str] = selection_param("characters")
    # timeCodesPerSecond: import reads one time code as one frame (reader.py _frames)
    outputs = selection_ports(Params, fps=True)

    @classmethod
    def listing(cls, params):
        from . import reader

        return reader.listing(str(cls.path(params)))

    @classmethod
    def read(cls, ctx, chosen):
        from ...io import usd
        from . import reader

        path = cls.path(ctx.params)
        group = usd.import_group(ctx.node_id, cls.id, path.name)  # /shot/<file stem>/...
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
    id = "usd.output"
    format = Format("usd")
    # 相机片门偏移取镜头中心偏移的相反数；换单位时同步换算相机、基本体和实例的位置尺寸。
    version = 4  # 输出单位同步换算高斯尺度
    category = "out_scene"
    inputs = (Port("scene", "scene|scene[]", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name", "unit")
    writes = {
        "model": Writes.full(),
        "camera": Writes.full(),
        "points": Writes.full(),
        "curves": Writes.full(),
        "skeleton": Writes.full(),
        "character": Writes.full(),
        "gaussian": Writes.full(),
        "light": Writes.full(),  # 穹顶灯的 HDRI 随文件拷进 <名字>_textures/（data/scene.py _localize_textures）
    }

    class Params(NodeParams):
        name: str = name_param("scene")
        format: Literal["usd", "usda"] = P(
            "usd", group="file",
        )
        unit: Literal["cm", "m"] = P(
            "cm", group="file",
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
