"""分区（网格的一部分）：USD GeomSubset 的读取与按分区取出。

分区即网格中的一组面：FLAME 自带的 scalp / face / neck 等、Maya 的面集、Houdini 的图元组、
文件中按材质划分的区块，在 USD 中都是同一种原语 `UsdGeom.Subset`（读写见 io/usd.py 的 write_subset、
mesh_subsets）。分区不是新的数据类型，而是网格自身的一部分，随「模型」数据传递，写入 USD 时随数据保存，
在 DCC 中对应为组。

    scene_subsets(packet)          场景中的分区及各自的面数（生成数据包时统计一次，写入 meta）
    only_subset(packet, 名字, 出)   「按分区取出」：仅保留该分区的面，点、UV、法线和逐点属性随之裁剪
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Vt

from ..io import usd
from .packet import Packet
from .payloads import SCENE_FILE, scene_packet


def stage_subsets(stage: Usd.Stage) -> dict[str, int]:
    """stage 上每个分区名 -> 其面数（多个网格上的同名分区相加），按出现顺序排列。"""
    out: dict[str, int] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        for name, faces in usd.mesh_subsets(UsdGeom.Mesh(prim)).items():
            out[name] = out.get(name, 0) + len(faces)
    return out


def scene_subsets(src: Packet) -> dict[str, int]:
    """场景数据包中的分区及各自的面数（「按分区取出」的选项和数据信息中的对应行均读取此结果）。"""
    from .scene import open_scene

    return stage_subsets(open_scene([src]))


def only_subset(src: Packet, name: str, out: Path) -> tuple[Packet, int, int] | None:
    """「按分区取出」：场景中每个含该分区的网格仅保留该分区的面（其余面和其余网格均删除），
    返回 (数据包, 保留的网格数, 保留的面数)。场景中没有该分区时返回 None；分区名从输入数据自带的分区中选取，
    因此仅在更换上游后才会出现这种情况。

    点按保留的面重新编号（未使用的点不输出，与 Houdini 的 Blast 相同）；UV、法线以及逐点、逐面属性
    按各自的插值方式随之裁剪，逐帧变形的点缓存逐帧裁剪。保留的网格上不再有分区（整个网格即为该分区）。"""
    from .scene import keep_only, open_scene

    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    kept, faces = [], 0
    meshes = [prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    for path in meshes:  # 按路径处理：裁剪网格会删除其分区 prim，已持有的对象随之失效
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        chosen = usd.mesh_subsets(mesh).get(name)
        if chosen is None:
            continue
        faces += _cut_mesh(mesh, chosen)
        kept.append(path)
    if not kept:
        return None
    keep_only(stage, kept)  # 祖先的变换和绑定的材质一并保留，与「按种类取出」共用同一实现（data/scene.py）
    out.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height")}
    return scene_packet(out, src.meta["frames"], "scene.model", **meta), len(kept), faces


def _cut_mesh(mesh: UsdGeom.Mesh, chosen: np.ndarray) -> int:
    """网格仅保留 `chosen` 中的面，返回保留的面数。"""
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)])
    chosen = np.unique(chosen[(chosen >= 0) & (chosen < len(counts))])
    corners = np.concatenate([np.arange(starts[f], starts[f + 1]) for f in chosen]) if len(chosen) else np.zeros(0, np.int64)
    used = np.unique(indices[corners]) if len(corners) else np.zeros(0, np.int64)
    renumber = np.full(int(indices.max()) + 1 if len(indices) else 1, -1, np.int64)
    renumber[used] = np.arange(len(used))

    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts[chosen].astype(np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(renumber[indices[corners]].astype(np.int32)))
    _cut_points(mesh, used)
    _cut_normals(mesh, used, corners, chosen)
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        _cut_primvar(pv, used, corners, chosen)
    for prim in list(mesh.GetPrim().GetChildren()):  # 整个网格即为该分区，分区不再有意义
        if prim.IsA(UsdGeom.Subset):
            mesh.GetPrim().GetStage().RemovePrim(prim.GetPath())
    return len(chosen)


def _cut_points(mesh: UsdGeom.Mesh, used: np.ndarray) -> None:
    """点（静止的一份，或逐帧变形的每一帧）仅保留被使用的点，并重新计算包围盒。"""
    attr, extent = mesh.GetPointsAttr(), mesh.CreateExtentAttr()
    times = attr.GetTimeSamples()
    for t in [Usd.TimeCode(t) for t in times] or [Usd.TimeCode.Default()]:
        points = np.asarray(attr.Get(t) or [], np.float32).reshape(-1, 3)[used]
        attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points)), t)
        if len(points):
            extent.Set(Vt.Vec3fArray.FromNumpy(np.stack([points.min(0), points.max(0)])), t)


def _cut_normals(mesh: UsdGeom.Mesh, used: np.ndarray, corners: np.ndarray, chosen: np.ndarray) -> None:
    values = mesh.GetNormalsAttr().Get()
    if values is None:
        return
    take = _take_of(mesh.GetNormalsInterpolation(), used, corners, chosen)
    if take is None:
        return
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(np.asarray(values, np.float32)[take])))


def _cut_primvar(pv: UsdGeom.Primvar, used: np.ndarray, corners: np.ndarray, chosen: np.ndarray) -> None:
    """按裁剪结果处理一条逐点、逐面或逐角的属性（UV、颜色、权重等）。带索引的属性（faceVarying 的 UV 常见）
    只裁剪索引，值保持不变：这样处理是无损的，也无需判断哪些值仍在使用。"""
    take = _take_of(pv.GetInterpolation(), used, corners, chosen)
    if take is None or not pv.HasValue():
        return
    if pv.IsIndexed():
        pv.SetIndices(Vt.IntArray.FromNumpy(np.asarray(pv.GetIndices(), np.int32)[take]))
        return
    values = pv.Get()
    if values is not None and len(values) > (int(take.max()) if len(take) else -1):
        pv.Set(type(values)(np.asarray(values)[take].tolist()))


def _take_of(interpolation, used: np.ndarray, corners: np.ndarray, chosen: np.ndarray):
    """按插值方式确定一条属性应保留的值：逐点属性按保留的点，逐角属性按保留的角，逐面属性按保留的面；
    整条只有一个值的（constant）不变（None）。"""
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying):
        return used
    if interpolation == UsdGeom.Tokens.faceVarying:
        return corners
    if interpolation == UsdGeom.Tokens.uniform:
        return chosen
    return None
