"""The FBX format module's nodes (the interface: lab2shot/nodes/formats.py): 「导入 FBX」 lists and reads an .fbx through
this extension's worker (worker.py fbx.import: every item of the file as scene arrays, with the axis system and unit
the file records), 「FBX 输出设置」 writes our scene to .fbx through it (fbx.output: centimetres, the Y-up axis system
written explicitly). FBX holds cameras, still models and skinned characters with blend shapes; it has no point clouds,
no per-frame vertex caches and no dome lights."""

from __future__ import annotations

from dataclasses import replace

from lab2shot.sdk import (NodeParams, DistinctNames, OutputSettings, P, Port, WorkerImport, Writes, fps_param, import_file_param, name_param, pack, scene_arrays, selection_param, selection_ports)

SUFFIXES = (".fbx",)


class ImportFbx(WorkerImport):
    id = "fbx.import"
    runtime = "fbx"
    suffixes = SUFFIXES
    cost = replace(WorkerImport.cost, whole="读一个文件，时间看文件里有多少东西，不按帧算（只用 CPU）")

    class Params(NodeParams):  # FBX holds cameras, models, skeletons and skinned characters: no point clouds
        path: str = import_file_param(SUFFIXES)
        take: str = P("", label="动画段", widget="choice", group="文件", choices_from=("path",), placeholder="最长的一段")
        camera: str = selection_param("camera")
        models: list[str] = selection_param("models")
        skeletons: list[str] = selection_param("skeletons")
        characters: list[str] = selection_param("characters")
    outputs = selection_ports(Params)


class FbxOutput(OutputSettings):
    id = "fbx.output"
    category = "out_scene"
    runtime = "fbx"
    cost = replace(OutputSettings.cost, whole="写一个文件，时间看写多少东西，不按帧算（只用 CPU）")
    inputs = (Port("scene", "scene|scene[]", "场景", multi=True,
                   expects=(DistinctNames(),)),)
    on_node = ("name",)
    writes = {
        "model": Writes.static("FBX 没有逐帧的顶点缓存", lost="网格的分区。FBX 里没有面集，只能借材质分面，那会凭空造出材质；要保住分区写 USD 或 Alembic"),
        "camera": Writes.full(),
        "points": Writes.no("FBX 没有点云"),
        "curves": Writes.no("FBX 这一版不读写曲线"),
        "skeleton": Writes.full(),
        "character": Writes.full(lost="网格的分区，同「模型」"),
    }

    class Params(NodeParams):
        name: str = name_param("fbx")
        fps: float = fps_param()  # FBX 记着自己的帧率，动画按它播

    @classmethod
    def write(cls, ctx) -> str:
        out = cls.out_file(ctx, ".fbx")
        ctx.stage("整理场景")
        npz = scene_arrays(pack(ctx.inputs["scene"], ctx.work / "scene"),
                           fps=float(ctx.params["fps"])).save(ctx.work / "scene.npz")
        ctx.run_worker(None, extra={"file": str(out)}, inputs={"scene": npz}, reuse=False)  # it writes the file
        return out.name


NODES = (ImportFbx, FbxOutput)
