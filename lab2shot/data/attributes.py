"""点和曲线上附带的属性：列出现有属性，并按名称删除。

推理结果中点上附带的信息（置信度、法线、类别、跟踪编号等）一律保留在数据包中，保存前可按名称删除。
属性的收集在构建数据包时完成（`data/payloads.py point_attributes`：名称、类型、逐点或逐条，记入 meta；
摘要和节点右下角的感叹号只读取 meta，不打开文件）。本模块负责按名称删除。
对应 Houdini 中的 AttribDelete。
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from ..io import usd
from .packet import Packet
from .payloads import SCENE_FILE, point_attributes, scene_packet


def named(src: Packet) -> list[dict]:
    """数据中点和曲线上附带的属性（[{name, type, per}]），取自构建数据包时记入 meta 的列表。"""
    return list(src.meta.get("attributes") or ())


def matching(have: list[dict], patterns: str) -> list[str]:
    """`patterns`（以空格或逗号分隔，支持 * 通配，与 Houdini 的 AttribDelete 一致）在 `have` 中匹配到的名称。"""
    want = [p for p in patterns.replace(chr(0xFF0C), ",").replace(",", " ").split() if p]
    names = [a["name"] for a in have]
    return sorted({n for p in want for n in names if fnmatch.fnmatchcase(n, p)})


def drop(src: Packet, names: list[str], out: Path) -> tuple[Packet, int]:
    """删除场景中点云和三维曲线上具有指定名称的属性，返回 (数据包, 实际删除的数量)。

    点、面和坐标均不修改，仅删除点上附带的值。宽度、朝向等 UsdGeom 自有属性同样可以删除
    （与 primvar 一样属于逐点的值）。"""
    from pxr import Usd, UsdGeom

    from .scene import open_scene

    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    wanted, gone = set(names), 0
    for prim in stage.Traverse():
        if not (prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.BasisCurves)):
            continue
        api = UsdGeom.PrimvarsAPI(prim)
        for pv in api.GetPrimvars():
            if pv.GetPrimvarName() in wanted:
                api.RemovePrimvar(pv.GetPrimvarName())
                gone += 1
        built: list[tuple[str, object]] = []
        if prim.IsA(UsdGeom.Points):
            cloud = UsdGeom.Points(prim)
            built = [("ids", cloud.GetIdsAttr()), ("velocities", cloud.GetVelocitiesAttr())]
        else:
            curves = UsdGeom.BasisCurves(prim)
            built = [("widths", curves.GetWidthsAttr()), ("normals", curves.GetNormalsAttr())]
        for name, attr in built:
            if name in wanted and attr and attr.HasAuthoredValue():
                prim.RemoveProperty(attr.GetName())
                gone += 1
    out.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height", "scale", "space")}
    left = point_attributes({"points": [p for p in stage.Traverse() if p.IsA(UsdGeom.Points)],
                             "curves": [p for p in stage.Traverse() if p.IsA(UsdGeom.BasisCurves)]})
    return scene_packet(out, src.meta["frames"], src.type,
                        **meta, **({"attributes": left} if left else {})), gone


def attribute_said(a: dict) -> str:
    """One attribute as said (name, kind, per point or per curve), in the language now."""
    from .. import i18n

    def said() -> str:
        kind = i18n.lookup(f"attr.type.{a.get('type')}") or str(a.get("type", ""))
        per = i18n.lookup(f"attr.per.{a.get('per')}") or str(a.get("per", ""))
        return i18n.t("attributes.said", name=a.get("name", ""), type=kind, per=per)

    return i18n.Both.of(said)  # every language: a message naming it reads in its reader's
