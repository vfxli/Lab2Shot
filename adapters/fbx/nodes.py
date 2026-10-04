"""The FBX format module's nodes (the interface: lab2shot/nodes/formats.py): 「导入 FBX」 lists and reads an .fbx through
this extension's worker (worker.py fbx.import: every item of the file as scene arrays, with the axis system and unit
the file records), 「FBX 输出设置」 writes our scene to .fbx through it (fbx.output: centimetres, the Y-up axis system
written explicitly). FBX holds cameras, still models and skinned characters with blend shapes; it has no point clouds,
no per-frame vertex caches and no dome lights."""

from __future__ import annotations

from dataclasses import replace

from lab2shot.sdk import (Msg, NodeParams, DistinctNames, Format, OutputSettings, P, Port, Reads, WorkerImport, Writes, fps_param, import_file_param, name_param, pack, scene_arrays, selection_param, selection_ports)

SUFFIXES = (".fbx",)


class ImportFbx(WorkerImport):
    id = "fbx.import"
    # 6：角色的 Visibility 关键帧读进来（隐藏的帧在 USD 里也隐藏）
    version = 6
    runtime = "fbx"
    suffixes = SUFFIXES
    reads = Reads(rank=0)  # 相机、骨架、角色、模型：DCC 交来的文件先用 FBX
    cost = replace(WorkerImport.cost, whole=True)

    class Params(NodeParams):  # FBX holds cameras, models, skeletons and skinned characters: no point clouds
        path: str = import_file_param(SUFFIXES)
        take: str = P("", widget="choice", group="file", choices_from=("path",))
        camera: str = selection_param("camera")
        models: list[str] = selection_param("models")
        skeletons: list[str] = selection_param("skeletons")
        characters: list[str] = selection_param("characters")
    outputs = selection_ports(Params, fps=True)  # its words: node."fbx.import".port.fps.help


class FbxOutput(OutputSettings):
    id = "fbx.output"
    format = Format("fbx")
    version = 7  # 7：帧号不连续的角色在空档上隐藏（可见性关键帧），不再读回插值出的姿势；6：同层重名按原名写
    category = "out_scene"
    runtime = "fbx"
    cost = replace(OutputSettings.cost, whole=True)
    inputs = (Port("scene", "scene|scene[]", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name",)
    writes = {
        "model": Writes.static(Msg("I-FBX-NOVERTEXCACHE"), lost=Msg("I-FBX-NOGROUPS")),
        "camera": Writes.full(),
        "points": Writes.no(Msg("I-FBX-NOPOINTS")),
        "curves": Writes.no(Msg("I-FBX-NOCURVES")),
        "skeleton": Writes.full(),
        "character": Writes.full(lost=Msg("I-FBX-NOGROUPS")),
        "gaussian": Writes.no(Msg("I-FBX-NOGAUSSIAN")),
        "light": Writes.no(Msg("I-FBX-NOLIGHT")),
    }

    class Params(NodeParams):
        name: str = name_param("fbx")
        fps: float = fps_param()  # FBX 记着自己的帧率，动画按它播

    @classmethod
    def write(cls, ctx) -> str:
        out = cls.out_file(ctx, ".fbx")
        ctx.stage("prepare_scene")
        npz = scene_arrays(pack(ctx.inputs["scene"], ctx.work / "scene"),
                           fps=float(ctx.params["fps"])).save(ctx.work / "scene.npz")
        ctx.run_worker(None, extra={"file": str(out)}, inputs={"scene": npz}, reuse=False)  # it writes the file
        return out.name


NODES = (ImportFbx, FbxOutput)
