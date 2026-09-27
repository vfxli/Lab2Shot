"""三维曲线（scene.curves）的几何处理：重采样。

核心中只放通用的曲线运算，不出现任何项目名或格式名。曲线自身的属性
（宽度、颜色、朝向及其他逐点 primvar）随点一同重采样：这些属性是逐点的值，点变化后必须随之更新，
否则发丝重采样后宽度将无法对应。

每条曲线在自身弧长上等距取点（与 Houdini 的 Resample 相同）：曲线形状不变，
仅描述它的点数增加或减少。只有一个点或长度为 0 的曲线重采样为同一个点重复 N 次
（此类曲线没有方向，无法凭空构造形状）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Vt

from .packet import Packet
from .payloads import SCENE_FILE, scene_packet


def resample_curve(values: np.ndarray, lengths: np.ndarray, want: np.ndarray) -> np.ndarray:
    """一条曲线上的逐点值 [k, ...]，按弧长 `lengths` [k] 线性插值到 `want` [n] 指定的弧长处。"""
    flat = np.asarray(values, np.float64).reshape(len(values), -1)
    out = np.stack([np.interp(want, lengths, flat[:, c]) for c in range(flat.shape[1])], -1)
    return out.reshape((len(want), *np.asarray(values).shape[1:]))


def arc_lengths(points: np.ndarray) -> np.ndarray:
    """一条曲线在每个点处的累计弧长 [k]（第一个点为 0）。"""
    steps = np.linalg.norm(np.diff(points, axis=0), axis=-1) if len(points) > 1 else np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(steps)])


def resample_sample(counts: np.ndarray, arrays: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """将一个时间采样上的全部曲线（`counts` [C] 为每条曲线的点数，`arrays` 为各逐点数组）重采样为每条 `n` 个点。
    返回新的 counts 和新的逐点数组。"""
    cuts = np.cumsum(np.asarray(counts, np.int64))[:-1]
    parts = {name: np.split(np.asarray(values), cuts) for name, values in arrays.items()}
    points = parts["points"]
    out: dict[str, list[np.ndarray]] = {name: [] for name in arrays}
    for i, pts in enumerate(points):
        lengths = arc_lengths(np.asarray(pts, np.float64))
        total = float(lengths[-1]) if len(lengths) else 0.0
        for name, values in parts.items():
            v = np.asarray(values[i])
            if len(v) < 2 or total <= 0.0:  # 单个点或所有点重合：没有方向，按原值重复
                out[name].append(np.repeat(v[:1] if len(v) else np.zeros((1, *v.shape[1:]), v.dtype), n, 0))
            else:
                out[name].append(resample_curve(v, lengths, np.linspace(0.0, total, n)))
    return (np.full(len(points), n, np.int32),
            {name: np.concatenate(rows).astype(np.float32) if rows else np.zeros((0, *np.asarray(arrays[name]).shape[1:]), np.float32)
             for name, rows in out.items()})


# 逐点属性：曲线自身的宽度、朝向，以及所有 vertex / varying 插值的 primvar（包括颜色）
PER_POINT = (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying)


def resample(src: Packet, points_per_curve: int, out: Path) -> tuple[Packet, int, int]:
    """将场景中每组三维曲线的每条曲线按等弧长重采样为 `points_per_curve` 个点；其余内容（相机、模型、
    组及其变换）保持不变。返回 (新的数据包, 重采样的曲线数, 结果的总点数)。"""
    from ..io import usd
    from .scene import open_scene

    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    strands = points = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.BasisCurves):
            continue
        done = _resample_prim(prim, points_per_curve)
        strands += done[0]
        points += done[1]
    out.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height")}
    return scene_packet(out, src.meta["frames"], "scene.curves", **meta), strands, points


def _resample_prim(prim, n: int) -> tuple[int, int]:
    """对一组曲线的每个时间采样进行重采样；返回 (第一个采样上的曲线数, 结果的总点数)。"""
    curves = UsdGeom.BasisCurves(prim)
    counts_attr, points_attr = curves.GetCurveVertexCountsAttr(), curves.GetPointsAttr()
    per_point = [(curves.GetWidthsAttr(), curves.GetWidthsInterpolation(), Vt.FloatArray),
                 (curves.GetNormalsAttr(), curves.GetNormalsInterpolation(), Vt.Vec3fArray)]
    per_point += [(pv.GetAttr(), pv.GetInterpolation(), None) for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]
    times = points_attr.GetTimeSamples() or counts_attr.GetTimeSamples()
    stamps = [Usd.TimeCode(t) for t in times] or [Usd.TimeCode.Default()]
    strands = total = 0
    for k, t in enumerate(stamps):
        counts = np.asarray(counts_attr.Get(t) or [], np.int64).reshape(-1)
        pts = np.asarray(points_attr.Get(t) or [], np.float64).reshape(-1, 3)
        arrays = {"points": pts}
        taken = []
        for attr, interpolation, array_type in per_point:
            values = attr.Get(t) if attr and attr.HasAuthoredValue() else None
            if values is None or interpolation not in PER_POINT or len(values) != len(pts):
                continue
            taken.append((attr, array_type or _array_type(values)))
            arrays[attr.GetName()] = np.asarray(values, np.float64)
        new_counts, new_arrays = resample_sample(counts, arrays, n)
        counts_attr.Set(Vt.IntArray.FromNumpy(new_counts), t)
        points_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(new_arrays["points"], np.float32)), t)
        for attr, array_type in taken:
            attr.Set(array_type.FromNumpy(np.ascontiguousarray(new_arrays[attr.GetName()], np.float32)), t)
        if k == 0:
            strands, total = len(new_counts), int(new_counts.sum())
    return strands, total


def _array_type(values):
    """逐点 primvar 的数组类型，依据每个值的分量数确定（颜色和朝向为三个分量，其余按一个分量处理）。"""
    return Vt.Vec3fArray if np.asarray(values).ndim == 2 else Vt.FloatArray
