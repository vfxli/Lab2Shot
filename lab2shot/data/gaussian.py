"""3D 高斯（scene.gaussian）的场景几何：USD Points 加一个标记（customData lab2shot:gaussian）和四组 primvars，
不依赖任何厂商 schema 或渲染器，Houdini / Omniverse 读到的就是一片带属性的点。

单位与约定（Inria 3DGS）：位置与尺度为厘米（尺度是线性的标准差），旋转为单位四元数 (w, x, y, z)，透明度线性 0–1，
球谐系数 [N, C, 3] 按 Inria 实球谐次序、DC 在前（C = 1 / 4 / 9 / 16）。协方差、球谐基与烘焙变换的算术在
lab2shot_shared/gaussians.py，核心与重建 worker 共用这一份；本模块只管 USD 读写与校验。
"""
from __future__ import annotations

from .. import i18n

import numpy as np
from pxr import Sdf, Usd, UsdGeom, Vt

from lab2shot_shared.gaussians import C0, covariance, transform

from ..errors import Invalid
from ..messages import Msg

MARKER = "lab2shot:gaussian"
__all__ = ["C0", "MARKER", "covariance", "is_gaussian", "validate", "write", "rewrite", "sample", "changing",
           "transform_sample"]


def is_gaussian(prim) -> bool:
    return bool(prim.GetCustomDataByKey(MARKER))


def validate(points, scales, rotations, opacity, sh):
    """The five arrays as float32, rotations normalized; anything that is not a valid splat set is refused
    (E-GAUSSIAN-BAD with the reason), wherever it came from (a PLY, a worker, a USD file)."""
    arrays = [np.asarray(v, np.float32) for v in (points, scales, rotations, opacity, sh)]
    p, s, r, a, h = arrays
    n = len(p)
    if p.shape != (n, 3) or s.shape != (n, 3) or r.shape != (n, 4) or a.shape != (n,) or h.ndim != 3 or h.shape[0] != n or h.shape[2] != 3:
        raise Invalid(Msg("E-GAUSSIAN-BAD", why=i18n.Word("gaussian.bad.shapes")))
    if h.shape[1] not in (1, 4, 9, 16):
        raise Invalid(Msg("E-GAUSSIAN-BAD", why=i18n.Word("gaussian.bad.sh", count=h.shape[1])))
    if not all(np.isfinite(v).all() for v in arrays) or (s <= 0).any() or ((a < 0) | (a > 1)).any():
        raise Invalid(Msg("E-GAUSSIAN-BAD", why=i18n.Word("gaussian.bad.values")))
    norm = np.linalg.norm(r, axis=1, keepdims=True)
    if (norm < 1e-8).any():
        raise Invalid(Msg("E-GAUSSIAN-BAD", why=i18n.Word("gaussian.bad.rotation")))
    return p, s, r / norm, a, h


# the gaussian primvars, in the order `validate` returns them (scales, rotations, opacity, sh)
FIELDS = (("scales", Sdf.ValueTypeNames.Float3Array, Vt.Vec3fArray.FromNumpy),
          ("rotations", Sdf.ValueTypeNames.Float4Array, Vt.Vec4fArray.FromNumpy),
          ("opacity", Sdf.ValueTypeNames.FloatArray, Vt.FloatArray.FromNumpy),
          ("sh", Sdf.ValueTypeNames.Float3Array, Vt.Vec3fArray.FromNumpy))


def _checked(samples):
    return [validate(*(s[k] for k in ("points", "scales", "rotations", "opacity", "sh"))) for s in samples]


def _set_primvars(cloud, frames, checked):
    """The gaussian primvars (scales, rotations, opacity, sh) of `checked` samples, replacing whatever `cloud` held."""
    api = UsdGeom.PrimvarsAPI(cloud)
    for j, (name, kind, make) in enumerate(FIELDS, 1):
        pv = api.CreatePrimvar("gaussian:" + name, kind, UsdGeom.Tokens.vertex)
        pv.GetAttr().Clear()
        if name == "sh":
            pv.SetElementSize(checked[0][4].shape[1])
        for i, values in enumerate(checked):
            time = Usd.TimeCode.Default() if len(checked) == 1 else Usd.TimeCode(frames[i])
            arr = values[j].reshape(-1, 3) if name == "sh" else values[j]
            pv.Set(make(np.ascontiguousarray(arr, dtype=np.float32)), time)


def write(stage, path, frames, samples):
    """Write full geometry per sample, including changing counts and appearance."""
    from ..io import usd

    checked = _checked(samples)
    if len(checked) != max(1, len(frames)) and len(checked) != 1:
        raise Invalid(Msg("E-GAUSSIAN-BAD", why=i18n.Word("gaussian.bad.samples", samples=len(checked), frames=len(frames))))
    cloud = usd.write_points(stage, path, frames, [v[0] for v in checked],
                             [np.maximum(0, C0 * v[4][:, 0] + 0.5) for v in checked])
    cloud.GetPrim().SetCustomDataByKey(MARKER, True)
    _set_primvars(cloud, frames, checked)
    return cloud


def rewrite(prim, frames, samples):
    """Set a gaussian's geometry (points, scales, rotations, opacity, sh) on an existing UsdGeom.Points prim, replacing
    what it held. Its display colour (SH DC, unchanged by a rigid or scaled transform) and its own transform are left
    alone. The one writer the camera-space bake uses, so a gaussian under 「相机空间转换」 carries its new world
    placement in its own data rather than a USD transform on /shot."""
    from pxr import Gf

    checked = _checked(samples)
    cloud = UsdGeom.Points(prim)
    pos, ext = cloud.GetPointsAttr(), cloud.GetExtentAttr()
    pos.Clear()
    ext.Clear()
    times = [Usd.TimeCode.Default()] if len(checked) == 1 else [Usd.TimeCode(f) for f in frames]
    for i, t in enumerate(times):
        p = np.ascontiguousarray(checked[i][0], dtype=np.float32)
        pos.Set(Vt.Vec3fArray.FromNumpy(p), t)
        if len(p):
            ext.Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p.min(0))), Gf.Vec3f(*map(float, p.max(0)))]), t)
    _set_primvars(cloud, frames, checked)


def sample(prim, frame, scale=1.0):
    api, t = UsdGeom.PrimvarsAPI(prim), Usd.TimeCode(frame)
    points = np.asarray(UsdGeom.Points(prim).GetPointsAttr().Get(t), np.float32).reshape(-1, 3) * scale
    def get(name, shape):
        return np.asarray(api.GetPrimvar("gaussian:" + name).Get(t), np.float32).reshape(shape)
    return dict(zip(("points", "scales", "rotations", "opacity", "sh"),
                    validate(points, get("scales", (-1, 3)) * scale, get("rotations", (-1, 4)),
                             get("opacity", (-1,)), get("sh", (len(points), api.GetPrimvar("gaussian:sh").GetElementSize(), 3)))))


def changing(prim):
    return any(a.ValueMightBeTimeVarying() for a in prim.GetAttributes()
               if a.GetName() == "points" or a.GetName().startswith("primvars:gaussian:"))


def transform_sample(values: dict, matrix) -> dict:
    """Bake a 4x4 transform into one sample (lab2shot_shared.gaussians.transform): the one bake 「相机空间转换」 and
    「PLY 输出设置」 share. SH above degree 0 cannot follow a non-uniform scale: refused with the way out."""
    try:
        return transform(values, matrix)
    except ValueError:
        raise Invalid(Msg("E-GAUSSIAN-NONUNIFORM")) from None
