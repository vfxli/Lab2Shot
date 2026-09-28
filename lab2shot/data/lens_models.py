"""Lens distortion: the model registry, a lens on the plate it was defined on, and ST-maps. Numbers in, numbers
out: no files, no packets.

Pixels: pixel (i, j) covers [i, i+1] x [j, j+1], y down, centres at +0.5. OpenCV's own pixel convention puts
centres on integers (c_opencv = c - 0.5); COLMAP's matches this one.

A lens is a pair of maps between the plate as shot (distorted) and the nominal undistorted picture of the same raster,
a pinhole whose principal point is the lens centre:

- the lens centre in pixels: c_x = W/2 + o_x W / F_w, c_y = H/2 - o_y H / F_h (o in mm, +y up);
- the film back height, when only the width is known: F_h = F_w H / (W a), a the pixel aspect;
- 3DE4 models are normalised by the film back's half diagonal r_fb = sqrt(F_w^2 + F_h^2) / 2 around the lens centre,
  y up: x = (px - c_x) (F_w / W) / r_fb, y = -(py - c_y) (F_h / H) / r_fb (LDPK ldpk_ldp_builtin.h map_unit_to_dn: the
  unit square, origin bottom left, minus the lens centre offset); their formula is distorted -> undistorted;
- OpenCV models are normalised by the focal length in pixels: x = (px - c_x) / f_x, y = (py - c_y) / f_y with
  f_x = f W / F_w, f_y = f H / F_h; their formula is undistorted -> distorted, except COLMAP's two division models,
  which are closed-form the other way (distorted -> undistorted, as COLMAP writes them).

The other direction of every model is Newton's method on its formula, with the Jacobian taken by complex step (exact to
rounding: every formula here is analytic in x and y, written without abs, hypot or atan2 so it takes complex numbers). It
converges when the residual is below TOL in the model's normalised units (1e-10: 1e-7 to 3e-7 pixels on a 4K plate for a
3DE4 model, 1e-10 f_x pixels for an OpenCV one); a point that does not converge, or whose solution is off the part of the formula
reached from the lens centre without folding (`_folded`), has no solution (nan).

ST-maps, as Nuke's STMap reads them: R = u, G = v, 0..1 across the plate frame, origin bottom left, pixel
centres at +0.5. An undistort ST-map covers the canvas (the undistorted picture with overscan) and points at the plate;
a distort ST-map covers the plate and points into the canvas in the plate frame's normalisation (overscan: below 0 or
above 1). Where a map has no value both channels are INVALID. The canvas is the undistorted picture grown
by whole pixels on each side, its principal point in the middle; `window` is where the plate frame's top left corner
sits in it.

A parameter is a number, or one per frame: {"frames": [...], "values": [...]} (linear between frames, held outside
them, as a per-frame 浮点 in data/values.py).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from dataclasses import dataclass, field, replace

import numpy as np

from ..errors import Invalid
from ..messages import Msg
from . import units
from .maps import alpha_of, bands, sample_band
from .windows import Window

TOL = 1e-10  # Newton's residual, in the model's normalised units
MAX_ITER = 60
_HALVINGS = 12  # backtracking: a Newton step that makes the residual worse is halved at most this many times
_STEP = 1e-30  # complex step for the Jacobian
_CHUNK = 1 << 19  # points per batch (complex temporaries stay well under a gigabyte)

INVALID = -1.0  # an ST-map entry with no value: R = G = INVALID
OVERSCAN_CAP = 0.25  # automatic overscan grows the canvas by at most this much of the plate, per axis
OVERSCAN_MARGIN_PX = 2  # automatic overscan's sampling margin on each side, pixels

LEVELS = ("measured", "solved", "estimated")  # where a distortion came from, most trusted first
STMAP_MODEL = "stmap"  # a lens whose distortion is given as ST-maps only: no formula (the node samples the maps)


# ------------------------------------------------------------------ the models


@dataclass(frozen=True)
class ParamSpec:
    """A model parameter, named as the software names it. `typical`: the range the software's own controls cover (LDPK's
    Nuke nodes; None: no published range): a value outside it is taken and noted (N-LENS-PARAMRANGE). `above`: a value
    must be greater than this to mean anything physically (a squeeze of 0), else it is refused. `limit`: where a value a
    person types or a tracker's file gives stops being this parameter at all (a typo, a wrong unit or column): what a
    person types and what a file says are refused outside it (in_limit, E-LENS-COEFF)."""

    name: str
    default: float = 0.0
    typical: tuple[float, float] | None = (-0.5, 0.5)
    above: float | None = None
    limit: tuple[float, float] = (-10.0, 10.0)


@dataclass(frozen=True)
class Model:
    """A distortion model: its id, the name the software gives it, how it is normalised ("3de4" / "opencv"), its
    parameters in the software's order, and its formula in the direction it is written in (`analytic`)."""

    id: str
    label: str
    family: str
    analytic: str  # "undistort" (3DE4: distorted -> undistorted) or "distort" (OpenCV: undistorted -> distorted)
    params: tuple[ParamSpec, ...]
    formula: Callable  # (x, y, params: dict, pixel_aspect) -> (x, y); complex-safe

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)


def _identity(x, y, p, aspect):
    return x, y


def _classic(x, y, p, aspect):
    """3DE Classic LD Model (LDPK 2.14.0 ldpk_classic_ld_model_distortion.h, tde4_ldp_classic_ld_model.h): undistort, no
    decentering."""
    ld, sq, cx, cy, qu = (p["Distortion"], p["Anamorphic Squeeze"], p["Curvature X"], p["Curvature Y"], p["Quartic Distortion"])
    x2, y2 = x * x, y * y
    xu = x * (1 + (ld / sq) * x2 + ((ld + cx) / sq) * y2 + (qu / sq) * x2 * x2 + (2 * qu / sq) * x2 * y2 + (qu / sq) * y2 * y2)
    yu = y * (1 + (ld + cy) * x2 + ld * y2 + qu * x2 * x2 + 2 * qu * x2 * y2 + qu * y2 * y2)
    return xu, yu


def _radial_std_deg4(x, y, p, aspect):
    """3DE4 Radial - Standard, Degree 4 (LDPK 2.14.0 tde4_ldp_radial_standard_degree_4.h:
    ldpk_radial_decentered_distortion.h, ldpk_cylindric_extender.h): radial and
    decentered degree 2 and 4, then the cylindric bending matrix. Undistort."""
    c2, u2, v2 = p["Distortion - Degree 2"], p["U - Degree 2"], p["V - Degree 2"]
    c4, u4, v4 = p["Quartic Distortion - Degree 4"], p["U - Degree 4"], p["V - Degree 4"]
    r2 = x * x + y * y
    radial = 1 + c2 * r2 + c4 * r2 * r2
    xr = x * radial + (r2 + 2 * x * x) * (u2 + u4 * r2) + 2 * x * y * (v2 + v4 * r2)
    yr = y * radial + (r2 + 2 * y * y) * (v2 + v4 * r2) + 2 * x * y * (u2 + u4 * r2)
    phi, b = math.radians(p["Phi - Cylindric Direction"]), p["B - Cylindric Bending"]
    q, c, s = math.sqrt(1 + b), math.cos(phi), math.sin(phi)
    m00, m01, m11 = c * c * q + s * s / q, (q - 1 / q) * c * s, c * c / q + s * s * q
    return m00 * xr + m01 * yr, m01 * xr + m11 * yr


def _anamorphic_std_deg4(x, y, p, aspect):
    """3DE4 Anamorphic - Standard, Degree 4 (LDPK 2.14.0 tde4_ldp_anamorphic_standard_degree_4.h:
    ldpk_generic_anamorphic_distortion.h, the rotation, squeeze and pixel aspect extenders): p_u = R Sx Sy PA . A(R^-1 PA^-1 p). The angular terms are written as polynomials (r^2 cos 2phi =
    x^2 - y^2, r^4 cos 4phi = (x^2 - y^2)^2 - 4 x^2 y^2) so the formula stays analytic. Undistort."""
    a = math.radians(p["Lens Rotation"])
    ca, sa = math.cos(a), math.sin(a)
    xs = x / aspect
    xi, yi = ca * xs + sa * y, -sa * xs + ca * y
    x2, y2 = xi * xi, yi * yi
    r2, d2 = x2 + y2, x2 - y2
    c44 = d2 * d2 - 4 * x2 * y2

    def one(k):
        return 1 + p[f"C{k}02 - Degree 2"] * r2 + p[f"C{k}22 - Degree 2"] * d2 + p[f"C{k}04 - Degree 4"] * r2 * r2 \
            + p[f"C{k}24 - Degree 4"] * r2 * d2 + p[f"C{k}44 - Degree 4"] * c44

    ax = xi * one("x") * aspect * p["Squeeze-X"]
    ay = yi * one("y") * p["Squeeze-Y"]
    return ca * ax - sa * ay, sa * ax + ca * ay


def _brown(x, y, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
    """OpenCV's pinhole distortion with k1 k2 p1 p2 k3 (calib3d.hpp, "Detailed Description"): distort. COLMAP's
    SIMPLE_RADIAL / RADIAL / OPENCV are this with the terms they have (src/colmap/sensor/models.h, the Distortion()
    of each: SimpleRadial :1393, Radial :1475, OpenCV :1560)."""
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
            y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)


def _simple_radial(x, y, p, aspect):
    return _brown(x, y, k1=p["k"])


def _radial(x, y, p, aspect):
    return _brown(x, y, k1=p["k1"], k2=p["k2"])


def _opencv(x, y, p, aspect):
    return _brown(x, y, k1=p["k1"], k2=p["k2"], p1=p["p1"], p2=p["p2"])


def _atan_over_r(s):
    """atan(sqrt(s)) / sqrt(s), analytic in s = r^2 (so the complex step works at the centre too)."""
    small = np.real(s) < 1e-12
    with np.errstate(all="ignore"):
        r = np.sqrt(np.where(small, 1.0, s))
        far = np.arctan(r) / r
    return np.where(small, 1 - s / 3 + s * s / 5, far)


def _fisheye(x, y, k1=0.0, k2=0.0, k3=0.0, k4=0.0):
    """OpenCV's fisheye (Kannala-Brandt, calib3d.hpp namespace fisheye): theta = atan(r), theta_d = theta (1 + k1
    theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), x' = (theta_d / r) x. Distort. COLMAP's OPENCV_FISHEYE is this
    (src/colmap/sensor/models.h OpenCVFisheye :1661), SIMPLE_RADIAL_FISHEYE / RADIAL_FISHEYE the first terms of it
    (:2014, :2107)."""
    s = x * x + y * y
    f = _atan_over_r(s)
    t2 = s * f * f
    scale = f * (1 + k1 * t2 + k2 * t2 * t2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    return x * scale, y * scale


def _simple_radial_fisheye(x, y, p, aspect):
    return _fisheye(x, y, k1=p["k"])


def _radial_fisheye(x, y, p, aspect):
    return _fisheye(x, y, k1=p["k1"], k2=p["k2"])


def _opencv_fisheye(x, y, p, aspect):
    return _fisheye(x, y, k1=p["k1"], k2=p["k2"], k3=p["k3"], k4=p["k4"])


def _thin_prism_fisheye(x, y, p, aspect):
    """COLMAP's THIN_PRISM_FISHEYE (src/colmap/sensor/models.h ThinPrismFisheyeCameraModel::Distortion): the equidistant
    angle, then radial k1..k4 in r^2, tangential p1 p2 and thin prism sx1 sy1 on it. Distort."""
    f = _atan_over_r(x * x + y * y)
    u, v = x * f, y * f
    r2 = u * u + v * v
    radial = p["k1"] * r2 + p["k2"] * r2 ** 2 + p["k3"] * r2 ** 3 + p["k4"] * r2 ** 4
    du = u * radial + 2 * p["p1"] * u * v + p["p2"] * (r2 + 2 * u * u) + p["sx1"] * r2
    dv = v * radial + 2 * p["p2"] * u * v + p["p1"] * (r2 + 2 * v * v) + p["sy1"] * r2
    return u + du, v + dv


def _full_opencv(x, y, p, aspect):
    """COLMAP's FULL_OPENCV (src/colmap/sensor/models.h FullOpenCVCameraModel::Distortion :1760-1781): OpenCV's rational
    model, radial (1 + k1 r^2 + k2 r^4 + k3 r^6) / (1 + k4 r^2 + k5 r^4 + k6 r^6), tangential p1 p2 as OPENCV. Distort."""
    r2 = x * x + y * y
    r4, r6 = r2 * r2, r2 * r2 * r2
    radial = (1 + p["k1"] * r2 + p["k2"] * r4 + p["k3"] * r6) / (1 + p["k4"] * r2 + p["k5"] * r4 + p["k6"] * r6)
    return (x * radial + 2 * p["p1"] * x * y + p["p2"] * (r2 + 2 * x * x),
            y * radial + 2 * p["p2"] * x * y + p["p1"] * (r2 + 2 * y * y))


def _fov(x, y, p, aspect):
    """COLMAP's FOV (src/colmap/sensor/models.h FOVCameraModel::Distortion :1853-1889; Devernay & Faugeras 2001): omega the
    field of view in radians, x' = x atan(2 r tan(omega/2)) / (r omega), written as (c / omega) atan(c r) / (c r) with
    c = 2 tan(omega/2) so the centre is analytic (_atan_over_r). Distort. COLMAP shortcuts its own formula with truncated
    series below omega^2 < 1e-4 and below r^2 < 1e-4 (:1864-1880); the exact formula those approximate is used throughout
    (at the edge of the r^2 branch the two agree to ~3e-9 relative; the omega^2 branch (:1870) carries the signs of the
    Undistortion's series (:1909), so it is not the formula's own limit). omega = 0 is the pinhole limit (c / omega -> 1)."""
    omega = p["omega"]
    if omega == 0:
        return x, y
    c = 2 * math.tan(omega / 2)
    factor = c / omega * _atan_over_r(c * c * (x * x + y * y))
    return x * factor, y * factor


def _rad_tan_thin_prism_fisheye(x, y, p, aspect):
    """COLMAP's RAD_TAN_THIN_PRISM_FISHEYE (src/colmap/sensor/models.h RadTanThinPrismFisheyeModel::Distortion :2333-2378;
    Project Aria's Fisheye624, :655-661): the equidistant angle (u, v), the radial polynomial 1 + k0 theta^2 + k1 theta^4
    + ... + k5 theta^12 on it, then, in the radially scaled point, tangential p0 p1 (p0 with r^2 + 2 x^2, the other way
    round from OPENCV's p1 p2) and thin prism s0 s1 on x (r^2, r^4), s2 s3 on y. Distort."""
    f = _atan_over_r(x * x + y * y)
    u, v = x * f, y * f
    t2 = u * u + v * v
    radial = 1 + p["k0"] * t2 + p["k1"] * t2 ** 2 + p["k2"] * t2 ** 3 + p["k3"] * t2 ** 4 + p["k4"] * t2 ** 5 + p["k5"] * t2 ** 6
    xr, yr = u * radial, v * radial
    r2 = xr * xr + yr * yr
    r4 = r2 * r2
    dx = 2 * p["p1"] * xr * yr + p["p0"] * (r2 + 2 * xr * xr) + p["s0"] * r2 + p["s1"] * r4
    dy = 2 * p["p0"] * xr * yr + p["p1"] * (r2 + 2 * yr * yr) + p["s2"] * r2 + p["s3"] * r4
    return xr + dx, yr + dy


def _division(x, y, p, aspect):
    """COLMAP's SIMPLE_DIVISION / DIVISION (src/colmap/sensor/models.h, the CamFromImg of each :2438-2456, :2528-2547;
    Fitzgibbon 2001): x_u = x_d / (1 + k r_d^2), closed form in this direction only. Its ImgFromCam (:2406-2436) is the
    smaller root of the quadratic this gives, the one Newton's method reaches from the centre. Undistort."""
    k = p["k"]
    denom = 1 + k * (x * x + y * y)
    return x / denom, y / denom


def _equidistant(x, y, p, aspect):
    """COLMAP's SIMPLE_FISHEYE / FISHEYE (src/colmap/sensor/models.h :714-746, the FisheyeFromNormal :429-438 of every
    fisheye model): the equidistant projection theta = atan(r) alone, no coefficients. Distort."""
    return _fisheye(x, y)


def _eucm(x, y, p, aspect):
    """COLMAP's EUCM (src/colmap/sensor/models.h EUCMCameraModel::ImgFromCam :2758-2794; Khomutenko, Garcia & Martinet
    2018) on the normalised plane (w = 1): rho = sqrt(beta r^2 + 1), x' = x / (alpha rho + 1 - alpha). Its domain is
    alpha in [0, 1], beta > 0 (its HasBogusExtraParams :2746-2748): then rho >= 1, the denominator >= 1, and the formula
    is analytic everywhere; alpha = 0 is the pinhole. Distort."""
    alpha, beta = p["alpha"], p["beta"]
    rho = np.sqrt(beta * (x * x + y * y) + 1)
    den = alpha * rho + (1 - alpha)
    return x / den, y / den


_C = ParamSpec
_SQUEEZE = dict(default=1.0, above=0.0, limit=(0.1, 10.0))
_ANGLE = (-180.0, 180.0)
_OPENCV = dict(typical=None)


def _cv(*names: str) -> tuple[ParamSpec, ...]:
    """COLMAP / OpenCV coefficients, named exactly as COLMAP names them (k, k1, p1, sx1, omega …)."""
    return tuple(_C(n, **_OPENCV) for n in names)


# 模型 id 与界面名称均采用 COLMAP 官方名称，括号内附一句适用镜头的说明，供使用者选择时参考。COLMAP 的「镜头模型」
# 与「LensDistortion」的「镜头模型」读取同一张表、使用同一名称，参数名完全一致，两处选择相同即可对应。
# 各档参数名及顺序依照上游 third_party/colmap/repo/src/colmap/sensor/models.h 的 InitializeParamsInfo()
# （SimplePinhole :1202、Pinhole :1263、SimpleRadial :1326、Radial :1408、OpenCV :1491、OpenCVFisheye :1579、
# FullOpenCV :1680、FOV :1786、SimpleRadialFisheye :1931、RadialFisheye :2027、ThinPrismFisheye :2122、
# RadTanThinPrismFisheye :2242、SimpleDivision :2383、Division :2473、SimpleFisheye :2564、Fisheye :2642、EUCM :2722），
# 去掉开头的 f / fx fy 与 cx cy：Focal Length、Filmback 各有独立输入，主点与像素比是镜头表自身的项，此处只登记畸变系数。
# 上游 18 档中仅 EQUIRECTANGULAR（:2837，球面全景，参数为 w h）不收录：它不是镜头，本管线不处理此类画面。
# 括号内的说明并非上游原文；上游仅有注释或公式的几档，依据其内容写一句中性说明。
# 另收录 3DE 的三个模型（跟踪环节提供的镜头表使用这些模型）。
COLMAP_MODELS = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV", "THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE",
                 "SIMPLE_DIVISION", "DIVISION", "SIMPLE_FISHEYE", "FISHEYE", "EUCM")
PINHOLE_MODELS = ("SIMPLE_PINHOLE", "PINHOLE")  # 无畸变的两档：镜头表上无系数，ST-map 为恒等映射
# FOV 的 omega 为视场角（弧度，Devernay & Faugeras 2001）：取 pi 时 tan(omega/2) 为无穷大，负值与正值结果相同，因此只接受 [0, pi]；
# 0 为针孔极限（上游初值为 0.01，:1805）。EUCM 的 alpha 必须在 [0, 1] 内、beta 必须 > 0（上游 HasBogusExtraParams
# :2746-2748 将范围外的值判为无效参数），beta 默认值取上游初值 1.0（:2754），取 0 会使公式退化。
_FOV_OMEGA = _C("omega", typical=None, limit=(0.0, math.pi))
_EUCM = (_C("alpha", typical=None, limit=(0.0, 1.0)), _C("beta", default=1.0, typical=None, above=0.0))
MODELS: Mapping[str, Model] = {m.id: m for m in (
    Model("SIMPLE_PINHOLE", "SIMPLE_PINHOLE（无畸变，已去畸变的画面）", "opencv", "distort", (), _identity),
    Model("PINHOLE", "PINHOLE（无畸变，fx≠fy）", "opencv", "distort", (), _identity),
    Model("SIMPLE_RADIAL", "SIMPLE_RADIAL（普通相机、手机）", "opencv", "distort", _cv("k"), _simple_radial),
    Model("RADIAL", "RADIAL（畸变稍大的普通镜头）", "opencv", "distort", _cv("k1", "k2"), _radial),
    Model("OPENCV", "OPENCV（带切向畸变，和 OpenCV 标定兼容）", "opencv", "distort", _cv("k1", "k2", "p1", "p2"), _opencv),
    Model("FULL_OPENCV", "FULL_OPENCV（畸变复杂的广角镜头）", "opencv", "distort",
          _cv("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"), _full_opencv),
    Model("OPENCV_FISHEYE", "OPENCV_FISHEYE（鱼眼，等距投影）", "opencv", "distort", _cv("k1", "k2", "k3", "k4"), _opencv_fisheye),
    Model("SIMPLE_RADIAL_FISHEYE", "SIMPLE_RADIAL_FISHEYE（鱼眼，径向 k）", "opencv", "distort", _cv("k"), _simple_radial_fisheye),
    Model("RADIAL_FISHEYE", "RADIAL_FISHEYE（鱼眼，径向 k1 k2）", "opencv", "distort", _cv("k1", "k2"), _radial_fisheye),
    Model("FOV", "FOV（部分广角 / 鱼眼）", "opencv", "distort", (_FOV_OMEGA,), _fov),
    Model("THIN_PRISM_FISHEYE", "THIN_PRISM_FISHEYE（高精度鱼眼，VR / SLAM 设备）", "opencv", "distort",
          _cv("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"), _thin_prism_fisheye),
    # 上游 :655-661：Project Aria 的 FisheyeRadTanThinPrism（Fisheye624）
    Model("RAD_TAN_THIN_PRISM_FISHEYE", "RAD_TAN_THIN_PRISM_FISHEYE（鱼眼，径向六项加切向、薄棱镜；Project Aria 设备）", "opencv",
          "distort", _cv("k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3"), _rad_tan_thin_prism_fisheye),
    # 上游 :678-712：Fitzgibbon 2001 的单参数除法模型，两个方向均有闭式解；此处公式为其去畸变方向
    Model("SIMPLE_DIVISION", "SIMPLE_DIVISION（除法模型，单系数）", "opencv", "undistort", _cv("k"), _division),
    Model("DIVISION", "DIVISION（除法模型，单系数，fx≠fy）", "opencv", "undistort", _cv("k"), _division),
    # 上游 :714-746：仅含等距投影 theta = r、无系数的鱼眼，「畸变可忽略或已校正的鱼眼」
    Model("SIMPLE_FISHEYE", "SIMPLE_FISHEYE（等距鱼眼，无系数：畸变可忽略或已校正）", "opencv", "distort", (), _equidistant),
    Model("FISHEYE", "FISHEYE（等距鱼眼，无系数，fx≠fy）", "opencv", "distort", (), _equidistant),
    # 上游 :748-757：Khomutenko, Garcia & Martinet 2018 的增强统一相机模型
    Model("EUCM", "EUCM（增强统一模型，广角到鱼眼，alpha beta 两系数）", "opencv", "distort", _EUCM, _eucm),
    Model("3de4_classic", "3DE Classic LD Model", "3de4", "undistort", (
        _C("Distortion"), _C("Anamorphic Squeeze", typical=(0.25, 4.0), **_SQUEEZE), _C("Curvature X"), _C("Curvature Y"),
        _C("Quartic Distortion")), _classic),
    Model("3de4_radial_std_deg4", "3DE4 Radial - Standard, Degree 4", "3de4", "undistort", (
        _C("Distortion - Degree 2"), _C("U - Degree 2"), _C("V - Degree 2"), _C("Quartic Distortion - Degree 4"),
        _C("U - Degree 4"), _C("V - Degree 4"), _C("Phi - Cylindric Direction", typical=(-90.0, 90.0), limit=_ANGLE),
        _C("B - Cylindric Bending", typical=(-0.1, 0.1), above=-1.0)), _radial_std_deg4),
    Model("3de4_anamorphic_std_deg4", "3DE4 Anamorphic - Standard, Degree 4", "3de4", "undistort", (
        _C("Cx02 - Degree 2"), _C("Cy02 - Degree 2"), _C("Cx22 - Degree 2"), _C("Cy22 - Degree 2"), _C("Cx04 - Degree 4"),
        _C("Cy04 - Degree 4"), _C("Cx24 - Degree 4"), _C("Cy24 - Degree 4"), _C("Cx44 - Degree 4"), _C("Cy44 - Degree 4"),
        _C("Lens Rotation", typical=(-2.0, 2.0), limit=_ANGLE), _C("Squeeze-X", typical=(0.9, 1.1), **_SQUEEZE),
        _C("Squeeze-Y", typical=(0.9, 1.1), **_SQUEEZE)), _anamorphic_std_deg4),
)}


def distorts(model: str) -> bool:
    """该模型是否含有需去除的畸变：公式非恒等时才有畸变（SIMPLE_PINHOLE / PINHOLE 没有，其 ST-map 为恒等映射；SIMPLE_FISHEYE /
    FISHEYE 虽无系数，但等距投影本身即构成畸变，且须有焦距才能计算）。"""
    m = MODELS.get(model)
    return m is not None and m.formula is not _identity


def model_labels() -> Mapping[str, str]:
    """镜头模型 id → 界面名称。自行估计镜头的适配层（AnyCalib、COLMAP 等）以此为选项命名，
    使其「镜头模型」与「LensDistortion」使用同一份列表和同一名称，使用者无需在两份列表之间对照。"""
    return MappingProxyType({k: m.label for k, m in MODELS.items()})


def in_limit(x: float, p: ParamSpec) -> float:
    """A parameter's value inside its limit (E-LENS-COEFF): what a person types and what a tracker's file says alike."""
    low, high = p.limit
    if not low <= x <= high:
        raise Invalid(Msg("E-LENS-COEFF", name=p.name, value=x, low=low, high=high))
    return x


# Other programs' names for these models: name -> (model id, its parameter names in the program's order -> the names in MODELS).
# COLMAP's names are the model ids (MODELS above), so its rows are the identity; Kalibr's / OpenCV's names for the
# fisheye are aliases. A dataset's calibration, a COLMAP camera read or written: through this table only. A name
# missing here has no model computing the same: refused, never approximated.
EXTERNAL_MODELS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    # Kannala & Brandt 2006 (eq. 6, four terms) is OpenCV's cv::fisheye and Kalibr's "equidistant" (pinhole-equi, the
    # TUM-VI calibration): theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), the same k1..k4
    "kannala_brandt": ("OPENCV_FISHEYE", ("k1", "k2", "k3", "k4")),
    "equidistant": ("OPENCV_FISHEYE", ("k1", "k2", "k3", "k4")),
    # COLMAP's camera models (src/colmap/sensor/models.h), parameters after fx fy cx cy (or f cx cy)
    **{name: (name, MODELS[name].names) for name in COLMAP_MODELS if name in MODELS},
}
# the models above (COLMAP's) whose parameter list starts with one focal length (f cx cy: the InitializeParamsInfo()
# cited above, and their InitializeFocalLengthIdxs() is {0} alone); the others: fx fy cx cy
ONE_FOCAL_MODELS = frozenset({"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",
                              "SIMPLE_DIVISION", "SIMPLE_FISHEYE"})


def external_camera(model: str, params: list[float]) -> dict | None:
    """A camera another program names (its model name and parameter list: COLMAP's cameras.txt and pycolmap) as {fx, fy, cx, cy
    (pixels; COLMAP's pixel centres match this module's), distortion}; None when no model here computes the same (EQUIRECTANGULAR,
    a name from elsewhere) or the list is not that model's."""
    values = [float(v) for v in params]
    n = 3 if model in ONE_FOCAL_MODELS else 4
    if len(values) < n:
        return None
    distortion = external_distortion(model, values[n:])
    if distortion is None:
        return None
    fx, fy = (values[0], values[0]) if n == 3 else (values[0], values[1])
    return {"fx": fx, "fy": fy, "cx": values[n - 2], "cy": values[n - 1], "distortion": distortion}


def distortion(model: str, values: Mapping[str, float]) -> dict:
    """A distortion of MODELS from some of its parameters by name, the rest at their defaults ({"model", "params"}): how
    an extension that computes one of these models hands its numbers over. A name the model does not have is refused
    (E-LENS-PARAMS)."""
    m = MODELS.get(model)
    if m is None:
        raise Invalid(Msg("E-LENS-MODEL", model=str(model)[:40], models=list(MODELS)))
    unknown = sorted(set(values) - set(m.names))
    if unknown:
        raise Invalid(Msg("E-LENS-PARAMS", model=m.label, missing="无", unknown=unknown))
    return {"model": model, "params": {p.name: float(values.get(p.name, p.default)) for p in m.params}}


def external_distortion(name: str, values: list[float]) -> dict | None:
    """A distortion another program names ({"model", "params"} in MODELS, parameters it does not have at 0); None when
    no model here computes the same, or the values are not the program's parameters for it."""
    got = EXTERNAL_MODELS.get(name)
    if got is None:
        return None
    model, names = got
    if len(values) != len(names):
        return None
    params = {p.name: p.default for p in MODELS[model].params}
    params.update({n: float(v) for n, v in zip(names, values, strict=True)})
    return {"model": model, "params": params}


# ------------------------------------------------------------------ Newton's method on a formula


def _jacobian(f: Callable, x: np.ndarray, y: np.ndarray):
    """f at (x, y) and its Jacobian (d fx/dx, d fx/dy, d fy/dx, d fy/dy), by complex step."""
    ax, ay = f(x + 1j * _STEP, y.astype(complex))
    bx, by = f(x.astype(complex), y + 1j * _STEP)
    return (np.real(ax), np.real(ay)), (np.imag(ax) / _STEP, np.imag(bx) / _STEP, np.imag(ay) / _STEP, np.imag(by) / _STEP)


def _determinant(f: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    _, (j00, j01, j10, j11) = _jacobian(f, x, y)
    return j00 * j11 - j01 * j10


_FOLD_SAMPLES = 8  # points checked on the way from the lens centre to a position


def _folded(f: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Where a position is off the part of the formula that corresponds one to one with the picture: the part reached
    from the lens centre without the Jacobian's determinant going non-positive on the way. A polynomial that turns back
    (a strong pincushion past its fold, a fisheye past 90 degrees) also has solutions beyond the turn, some with a
    positive determinant again (through the centre to the other side): those are not the lens."""
    with np.errstate(all="ignore"):
        bad = ~(_determinant(f, x, y) > 0)
        for k in range(1, _FOLD_SAMPLES):
            t = k / _FOLD_SAMPLES
            bad |= ~(_determinant(f, x * t, y * t) > 0)
    return bad


def _solve(f: Callable, tx: np.ndarray, ty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y) with f(x, y) = (tx, ty), from the target itself; nan where it does not converge to TOL. Also where the
    solution lies on a fold (the determinant not positive): the third array marks those."""
    with np.errstate(all="ignore"):  # a point running off to infinity is expected (outside the model's range): nan
        return _newton(f, tx, ty)


def _newton(f: Callable, tx: np.ndarray, ty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = tx.copy(), ty.copy()
    for _ in range(MAX_ITER):
        (fx, fy), (j00, j01, j10, j11) = _jacobian(f, x, y)
        rx, ry = fx - tx, fy - ty
        res = np.hypot(rx, ry)
        live = np.isfinite(res) & (res >= TOL)
        if not live.any():
            break
        with np.errstate(all="ignore"):
            det = j00 * j11 - j01 * j10
            dx = np.where(live, (j11 * rx - j01 * ry) / det, 0.0)
            dy = np.where(live, (-j10 * rx + j00 * ry) / det, 0.0)
        lam = np.ones_like(x)
        for _ in range(_HALVINGS):
            nx, ny = x - lam * dx, y - lam * dy
            gx, gy = f(nx, ny)
            worse = live & ~(np.hypot(gx - tx, gy - ty) <= res)  # nan is worse
            if not worse.any():
                break
            lam = np.where(worse, lam / 2, lam)
        x, y = np.where(live, nx, x), np.where(live, ny, y)
    fx, fy = f(x, y)
    ok = np.hypot(fx - tx, fy - ty) < TOL
    folded = ok & _folded(f, x, y)
    ok &= ~folded
    return np.where(ok, x, np.nan), np.where(ok, y, np.nan), folded


# ------------------------------------------------------------------ a lens on its plate


def value_at(value, frame: int | None) -> float:
    """A parameter at a frame: a number, or {"frames": [...], "values": [...]} (linear between frames, held outside;
    no frame: the first)."""
    if isinstance(value, Mapping):
        frames, values = [float(f) for f in value["frames"]], [float(v) for v in value["values"]]
        order = np.argsort(frames)
        frames, values = list(np.asarray(frames)[order]), list(np.asarray(values)[order])
        return float(values[0] if frame is None else np.interp(frame, frames, values))
    return float(value)


def _animated(value) -> bool:
    return isinstance(value, Mapping) and len({float(v) for v in value["values"]}) > 1


def _each_value(value) -> list[float]:
    return [float(v) for v in value["values"]] if isinstance(value, Mapping) else [float(value)]


@dataclass(frozen=True)
class Mapped:
    """Pixel positions a lens mapped: `points` [..., 2] (nan: no solution), `folded` [...]: where the formula folds over
    (the two directions do not correspond one to one there)."""

    points: np.ndarray
    folded: np.ndarray


def _mapped(points: np.ndarray, one: Callable) -> Mapped:
    """Pixel positions [..., 2] through `one` ([n, 2] -> (positions [n, 2], folded [n])) in batches of _CHUNK."""
    pts = np.asarray(points, np.float64)
    flat = pts.reshape(-1, 2)
    out, folded = np.empty_like(flat), np.zeros(len(flat), bool)
    for start in range(0, len(flat), _CHUNK):
        out[start:start + _CHUNK], folded[start:start + _CHUNK] = one(flat[start:start + _CHUNK])
    return Mapped(out.reshape(*pts.shape[:-1], 2), folded.reshape(pts.shape[:-1]))


@dataclass(frozen=True)
class Lens:
    """A lens distortion on the raster it was defined on: the model and its parameters (numbers or per frame),
    the raster [W, H] of the plate as shot, the focal length in mm (OpenCV models need it; a number or per frame), the
    film back [width, height] in mm (a height of None: from the raster and pixel aspect; no film back: 36 mm wide), the
    pixel aspect, the lens centre offset in mm (+x right, +y up) and where the distortion came from ({level, by})."""

    model: str
    params: Mapping[str, object]
    raster: tuple[int, int]
    focal_mm: object = None
    filmback_mm: tuple[float, float | None] | None = None
    pixel_aspect: float = 1.0
    center_mm: tuple[float, float] = (0.0, 0.0)
    source: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        m = MODELS.get(self.model)
        if m is None:
            raise Invalid(Msg("E-LENS-MODEL", model=str(self.model)[:40], models=list(MODELS)))
        given = set(self.params)
        if given != set(m.names):
            raise Invalid(Msg("E-LENS-PARAMS", model=m.label, missing=[n for n in m.names if n not in given] or "无",
                              unknown=sorted(given - set(m.names)) or "无"))
        for spec in m.params:
            for v in _each_value(self.params[spec.name]):
                if not math.isfinite(v) or (spec.above is not None and v <= spec.above):
                    raise Invalid(Msg("E-LENS-PHYSICAL", what=spec.name, value=v))
        w, h = self.raster
        back = self.filmback_mm or (units.FILMBACK_MM, None)
        for what, v in (("raster", min(w, h)), ("filmback_mm", back[0]), ("filmback_mm", back[1] or 1.0),
                        ("pixel_aspect", self.pixel_aspect)):
            if not v > 0:
                raise Invalid(Msg("E-LENS-PHYSICAL", what=what, value=float(v)))
        if m.family == "opencv" and distorts(self.model) and self.focal_mm is None:
            raise Invalid(Msg("E-LENS-NOFOCAL", model=m.label))
        if self.source.get("level", "measured") not in LEVELS:
            raise Invalid(Msg("E-LENS-PHYSICAL", what="source.level", value=str(self.source.get("level"))[:20]))

    # -- description

    @classmethod
    def from_meta(cls, meta: Mapping, plate: tuple[int, int] | None = None) -> Lens | None:
        """A lens described as a camera's meta writes it (`meta()` below): None when it has no distortion, or its
        distortion is given as ST-maps only. `plate`: the raster when the description has none."""
        dist = meta.get("distortion")
        if not dist or dist.get("model") == STMAP_MODEL:
            return None
        raster = meta.get("raster") or plate
        if not raster:
            raise Invalid(Msg("E-LENS-NORASTER"))
        back = meta.get("filmback_mm")
        back = None if back is None else (float(back), None) if isinstance(back, (int, float)) else (float(back[0]), None if back[1] is None else float(back[1]))
        return cls(dist["model"], dict(dist.get("params") or {}), (int(raster[0]), int(raster[1])), meta.get("focal_mm"), back,
                   float(meta.get("pixel_aspect") or 1.0), tuple(float(v) for v in meta.get("center_mm") or (0.0, 0.0)),  # type: ignore[arg-type]
                   dict(meta.get("source") or {}))

    def meta(self) -> dict:
        fw, fh = self.back
        return {"focal_mm": self.focal_mm, "filmback_mm": [fw, fh], "pixel_aspect": self.pixel_aspect,
                "center_mm": list(self.center_mm), "raster": list(self.raster),
                "distortion": {"model": self.model, "params": dict(self.params)}, "source": dict(self.source)}

    @property
    def label(self) -> str:
        return MODELS[self.model].label

    @property
    def animated(self) -> bool:
        """Whether anything that moves a pixel changes over the shot (a zoom, focus breathing)."""
        opencv = MODELS[self.model].family == "opencv" and distorts(self.model)
        return any(_animated(v) for v in self.params.values()) or (opencv and _animated(self.focal_mm))

    def range_notes(self) -> list[Msg]:
        """N-LENS-PARAMRANGE for every parameter outside the range the software's controls cover (taken as it is)."""
        out = []
        for spec in MODELS[self.model].params:
            if spec.typical is None:
                continue
            lo, hi = spec.typical
            bad = [v for v in _each_value(self.params[spec.name]) if not lo <= v <= hi]
            if bad:
                out.append(Msg("N-LENS-PARAMRANGE", model=self.label, name=spec.name, value=bad[0], low=lo, high=hi))
        return out

    def on_plate(self, width: int, height: int) -> Lens:
        """The lens on a plate of this size: the same shape at another resolution (a proxy) scales with it (the
        normalisation is in film back units), another shape is refused (B-LENS-RASTER)."""
        w0, h0 = self.raster
        if (width, height) == (w0, h0):
            return self
        if width * h0 != height * w0:
            raise Invalid(Msg("B-LENS-RASTER", width=width, height=height, lens_width=w0, lens_height=h0))
        return replace(self, raster=(width, height))

    # -- geometry

    @property
    def back(self) -> tuple[float, float]:
        w, h = self.raster
        fw, fh = self.filmback_mm or (units.FILMBACK_MM, None)
        return float(fw), float(fh) if fh is not None else float(fw) * h / (w * self.pixel_aspect)

    def centre_px(self) -> tuple[float, float]:
        """The lens centre on the plate, pixels: the principal point of the nominal undistorted picture."""
        (w, h), (fw, fh) = self.raster, self.back
        return w / 2 + self.center_mm[0] * w / fw, h / 2 - self.center_mm[1] * h / fh

    def focal_px(self, frame: int | None = None) -> tuple[float, float]:
        """(f_x, f_y) in pixels at a frame."""
        if self.focal_mm is None:
            raise Invalid(Msg("E-LENS-NOFOCAL", model=self.label))
        (w, h), (fw, fh), f = self.raster, self.back, value_at(self.focal_mm, frame)
        return f * w / fw, f * h / fh

    def _normalisation(self, frame: int | None):
        """(pixels -> model coordinates, model coordinates -> pixels) at a frame."""
        cx, cy = self.centre_px()
        m = MODELS[self.model]
        if m.family == "opencv":
            fx, fy = self.focal_px(frame) if distorts(self.model) else (1.0, 1.0)
            return (lambda u, v: ((u - cx) / fx, (v - cy) / fy)), (lambda x, y: (x * fx + cx, y * fy + cy))
        (w, h), (fw, fh) = self.raster, self.back
        r = math.hypot(fw, fh) / 2
        sx, sy = fw / w / r, fh / h / r
        return (lambda u, v: ((u - cx) * sx, -(v - cy) * sy)), (lambda x, y: (x / sx + cx, -y / sy + cy))

    def _formula(self, frame: int | None) -> Callable:
        m = MODELS[self.model]
        p = {k: value_at(v, frame) for k, v in self.params.items()}
        return lambda x, y: m.formula(x, y, p, self.pixel_aspect)

    # -- the two maps

    def map_px(self, points: np.ndarray, direction: str, frame: int | None = None) -> Mapped:
        """Pixel positions [..., 2] of one picture -> where they are in the other: "distort" takes undistorted positions
        to the plate, "undistort" the plate's to the undistorted picture."""
        to, back = self._normalisation(frame)
        f = self._formula(frame)
        analytic = MODELS[self.model].analytic == direction

        def one(part: np.ndarray):
            x, y = to(part[:, 0], part[:, 1])
            if analytic:
                (ox, oy), fold = f(x, y), _folded(f, x, y)
            else:
                ox, oy, fold = _solve(f, x, y)
            return np.stack(back(np.asarray(ox, np.float64), np.asarray(oy, np.float64)), -1), fold
        return _mapped(points, one)

    def undistort_px(self, points: np.ndarray, frame: int | None = None) -> np.ndarray:
        return self.map_px(points, "undistort", frame).points


# ------------------------------------------------------------------ canvas and overscan


def centred_canvas(lens: Lens, size: tuple[int, int]) -> Window:
    """The canvas of this size for a lens, as one Window (data/windows.py): the plate frame placed so the principal
    point (the lens centre) sits in the middle within half a pixel, kept inside. How fit_canvas places
    it, and how a canvas camera (its size and its lens) gives its canvas back."""
    return Window.centred(lens.raster, size, lens.centre_px())


def _even_up(x: float) -> int:
    return 2 * math.ceil(x / 2 - 1e-9)


def fit_canvas(lens: Lens, frames=(None,), mode: str | float = "auto", cap: float = OVERSCAN_CAP) -> tuple[Window, bool]:
    """The canvas the undistorted plate is put on, and whether automatic overscan hit its cap. "none": the
    plate frame itself; a fraction (0.05): that much bigger per axis; "auto": every plate pixel centre's undistorted
    position inside, OVERSCAN_MARGIN_PX more on each side, at most `cap` bigger. The principal point sits in the middle
    of the canvas, within half a pixel."""
    w, h = lens.raster
    if mode == "none":
        return Window(w, h), False
    cx, cy = lens.centre_px()
    capped = False
    if mode == "auto":
        xs, ys = (np.arange(w) + 0.5), (np.arange(h) + 0.5)
        border = np.concatenate([np.stack([xs, np.full(w, 0.5)], -1), np.stack([xs, np.full(w, h - 0.5)], -1),
                                 np.stack([np.full(h, 0.5), ys], -1), np.stack([np.full(h, w - 0.5), ys], -1)])
        gx, gy = np.meshgrid(np.linspace(0.5, w - 0.5, 33), np.linspace(0.5, h - 0.5, 33))
        pts = np.concatenate([border, np.stack([gx.ravel(), gy.ravel()], -1)])
        half_x, half_y = max(cx, w - cx), max(cy, h - cy)
        for frame in frames:
            u = lens.undistort_px(pts, frame)
            if not np.isfinite(u).all():
                capped = True
            u = u[np.isfinite(u).all(-1)]
            if len(u):
                half_x = max(half_x, cx - u[:, 0].min() + 0.5, u[:, 0].max() - cx + 0.5)
                half_y = max(half_y, cy - u[:, 1].min() + 0.5, u[:, 1].max() - cy + 0.5)
        size_w, size_h = _even_up(2 * (half_x + OVERSCAN_MARGIN_PX)), _even_up(2 * (half_y + OVERSCAN_MARGIN_PX))
        top_w, top_h = _even_up(w * (1 + cap) + 1), _even_up(h * (1 + cap) + 1)
        capped = capped or size_w > top_w or size_h > top_h
        size_w, size_h = min(size_w, top_w), min(size_h, top_h)
    else:
        grow = float(mode)
        size_w, size_h = _even_up(w * (1 + grow)), _even_up(h * (1 + grow))
    size_w, size_h = max(size_w, _even_up(w)), max(size_h, _even_up(h))
    return centred_canvas(lens, (size_w, size_h)), capped


# ------------------------------------------------------------------ ST-maps


def pixel_centres(width: int, height: int) -> np.ndarray:
    """[H, W, 2] the pixel centres (x, y) of a raster."""
    y, x = np.mgrid[:height, :width] + 0.5
    return np.stack([x, y], -1)


def encode(positions: np.ndarray, plate: tuple[int, int], window: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Pixel positions [..., 2] in a raster where the plate frame starts at `window` -> ST-map entries normalised to
    the plate frame (R = x / W, G = 1 - y / H); nan -> INVALID."""
    (w, h), (left, top) = plate, window
    st = np.stack([(positions[..., 0] - left) / w, 1.0 - (positions[..., 1] - top) / h], -1)
    return np.where(np.isfinite(st).all(-1, keepdims=True), st, INVALID)


def decode(st: np.ndarray, plate: tuple[int, int], window: tuple[int, int] = (0, 0)) -> np.ndarray:
    """ST-map entries [..., 2] -> pixel positions in the raster where the plate frame starts at `window`; INVALID ->
    nan."""
    (w, h), (left, top) = plate, window
    st = np.asarray(st, np.float64)
    invalid = (st[..., 0] == INVALID) & (st[..., 1] == INVALID)
    pos = np.stack([st[..., 0] * w + left, (1.0 - st[..., 1]) * h + top], -1)
    return np.where(invalid[..., None], np.nan, pos)


def identity_stmap(width: int, height: int) -> np.ndarray:
    return encode(pixel_centres(width, height), (width, height))


@dataclass(frozen=True)
class StmapPair:
    """Both ST-maps of a lens on a canvas at one frame (float64; INVALID where there is no value) and how many plate
    pixels fold or fall outside the canvas."""

    undistort: np.ndarray  # [H', W', 2] canvas pixel -> where on the plate
    distort: np.ndarray  # [H, W, 2] plate pixel -> where in the canvas, plate frame normalisation
    folded: int
    outside: int


def lens_stmaps(lens: Lens, canvas: Window, frame: int | None = None) -> StmapPair:
    """The undistort and distort ST-maps of a lens on a canvas, from its formula (exact to Newton's tolerance)."""
    (cw, ch), (left, top), plate = canvas.canvas, canvas.offset, canvas.plate
    und = lens.map_px(pixel_centres(cw, ch) - (left, top), "distort", frame)
    dis = lens.map_px(pixel_centres(*plate), "undistort", frame)
    inside = (dis.points[..., 0] >= -left) & (dis.points[..., 0] <= cw - left) & \
             (dis.points[..., 1] >= -top) & (dis.points[..., 1] <= ch - top)
    return StmapPair(encode(und.points, plate), encode(dis.points, plate), int(dis.folded.sum() + und.folded.sum()),
                     int((~inside).sum()))


def _keys(t: np.ndarray) -> np.ndarray:
    """Keys' cubic kernel (a = -0.5) at distances `t` from the sample point (|t| in [0, 2]): 1.5|x|³ - 2.5|x|² + 1 on
    [0, 1], -0.5|x|³ + 2.5|x|² - 4|x| + 2 on [1, 2], 0 past 2. Sums to one over any four unit taps (the one
    bicubic kernel, shared by 「LensDistortion」 and 「STMap」 through apply_stmap)."""
    t = np.abs(t)
    return np.where(t <= 1.0, 1.5 * t ** 3 - 2.5 * t ** 2 + 1.0,
                    np.where(t <= 2.0, -0.5 * t ** 3 + 2.5 * t ** 2 - 4.0 * t + 2.0, 0.0))


def _bicubic_band(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """values [H,W,C] at (xs, ys) -> (values, where they hold one), bicubic (Keys a = -0.5), weighted by `a` (the
    source's alpha as float32, maps.alpha_of — built once a frame) so no invalid neighbour bleeds in; taps past the
    picture hold the edge pixel. Positions outside the source hold nothing, like maps.sample (one sampler for every
    warp, the kernels only here).

    One row band of an `apply_stmap` (maps.bands), like maps.sample_band: sixteen taps over a whole 1080p frame at
    once is over half a gigabyte of intermediate arrays and none of it stays in cache. Nothing here looks past its
    own pixels, so the bands add up bit for bit to the whole frame."""
    h, w = values.shape[:2]
    inside = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
    gx = np.clip(np.where(inside, xs, 0.0) - 0.5, 0.0, w - 1.0)
    gy = np.clip(np.where(inside, ys, 0.0) - 0.5, 0.0, h - 1.0)
    i0, j0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    tx, ty = gx - i0, gy - j0
    ii = np.stack([np.clip(i0 + k, 0, w - 1) for k in (-1, 0, 1, 2)])  # the four taps each way, clamped to the picture
    jj = np.stack([np.clip(j0 + k, 0, h - 1) for k in (-1, 0, 1, 2)])
    # the weights are the nominal unit-spaced distances (t + 1, t, t - 1, t - 2), so they always sum to one even where
    # a tap past the picture is clamped onto the edge pixel (the edge pixel then takes both its own and that weight)
    wx = _keys(np.stack([tx + 1.0, tx, tx - 1.0, tx - 2.0]))
    wy = _keys(np.stack([ty + 1.0, ty, ty - 1.0, ty - 2.0]))
    out = np.zeros((*xs.shape, values.shape[-1]), np.float32)
    total = np.zeros(xs.shape, np.float32)
    for k in range(4):
        for m in range(4):
            wgt = wx[m] * wy[k] * a[jj[k], ii[m]]  # [h,w]: the geometric weight times the tap's alpha
            out += wgt[..., None] * values[jj[k], ii[m]]
            total += wgt
    got = np.where(inside, total, 0.0)
    out = np.where(got[..., None] > 0, out / np.maximum(got, 1e-12)[..., None], 0.0)
    # the colour is the weighted average (divided by the weights that made it, so a bicubic stays as sharp as it is),
    # but what comes out as validity is an alpha: Keys' kernel has negative lobes, so with an alpha weighting the
    # taps the weights can sum past 1 (a transparent neighbour on a negative lobe) or below 0, and an alpha outside
    # 0..1 is not one. Only the validity is clamped, never the colour
    return out.astype(np.float32), np.clip(got, 0.0, 1.0).astype(np.float32)


def apply_stmap(values: np.ndarray, st: np.ndarray, plate: tuple[int, int], window: tuple[int, int] = (0, 0),
                alpha: np.ndarray | None = None, nearest: bool = False, bicubic: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Nuke's STMap with overscan: output pixel takes `values` (a raster where the plate frame starts at `window`) at
    the position its ST-map entry names -> (values, where they hold one). `nearest` for labels, `bicubic` (Keys a = -0.5)
    for pictures, neither: bilinear (maps.sample). With the plate as the source and no window it is exactly Nuke's STMap.
    The one sampling kernel for 「LensDistortion」 and 「STMap」 lives here.

    A row band at a time (maps.bands): the ST-map's own decode is banded too, so the float64 positions of a whole
    1080p frame (33 MB, read once and thrown away) never exist either. Bit for bit the same as one pass."""
    st = np.asarray(st)
    a = alpha_of(values, alpha, nearest)
    out = np.empty((*st.shape[:-1], *values.shape[2:]), np.float32)
    valid = np.empty(st.shape[:-1], np.float32)
    for band in bands(st.shape[:-1]):
        pos = decode(st[band], plate, window)
        out[band], valid[band] = _bicubic_band(values, pos[..., 0], pos[..., 1], a) if bicubic else \
            sample_band(values, pos[..., 0], pos[..., 1], a, nearest)
    return out, valid


