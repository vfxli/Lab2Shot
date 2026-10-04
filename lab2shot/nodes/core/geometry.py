"""深度与几何：深度转世界位置、深度转点云、深度转法线、法线空间转换、跟踪点转 3D、深度对齐。各项目的深度节点（包括今后新增的）
只输出深度及其观察相机；世界位置、点云和法线统一在本模块生成，相对深度、视差等各类深度也在本模块对齐到公制参考。"""

from __future__ import annotations

from typing import Literal

import numpy as np
from lab2shot_shared.motion import orthonormal

from ..kit.ports import values_port
from ..kit.ports import normal_port
from ...errors import Invalid
from ... import i18n
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..expects import NotAlready, NotDisparity, OwnCamera, SameShot
from ..applies import Cost, Wired
from ...data.units import M_TO_CM, PERCENT
from ...data.contracts import meant


def _depth_and_camera(ctx):
    """深度与相机两个输入；只知道差一个偏移的深度无法反投影。（相对深度只适配其自身解算得到的相机；覆盖其他帧的
    相机取最近的一帧。这些情况由各端口的用法检查提示，见 nodes/expects.py。）"""
    depth, camera = ctx.input("depth"), ctx.input("camera")
    if depth.meta.get("scale") == "affine":
        raise Invalid(Msg("E-DEPTH-AFFINE"))
    if depth.meta.get("scale") == "disparity":
        # 视差反投影得到的三维是错误的，且无法从数值上察觉。接线时已给出黄色提示（NotDisparity）；
        # 实际计算时在此停止，不把 scale='disparity' 写入点云（否则违反点云约定，被报告为开发者错误）。
        raise Invalid(Msg("E-DEPTH-DISPARITY"))
    return depth, camera


def _depth_port(scale_matters: bool = True) -> Port:
    """节点要转成三维的深度。scale_matters 为 False 时只关心形状（法线与尺度无关）。

    输入图自身标明为视差时给出提示（NotDisparity）：视差直接反投影得到的三维是变形的，且无法从数值上察觉。"""
    return Port("depth", "image.1", means=("scale",), expects=(OwnCamera(), NotDisparity()) if scale_matters else (NotDisparity(),))


def _camera_port(of: str, optional: bool = False) -> Port:
    """二维结果（输入 `of`）所对应的观察相机。"""
    return Port("camera", "scene.camera", optional=optional, expects=(SameShot(of),))


def _view(camera, frames: list[int], width: int, world: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`frames` 各帧的相机：以 `width` 为宽度时的像素焦距 (fx, fy) [F,2]（像素不是正方形时 fy ≠ fx）、相机到世界矩阵
    [F,4,4]（相机空间时为单位阵），以及以 `width` 为宽度时的像素主点 [F,2]（相机未写出主点时取画面中心）。"""
    from ...data.camera import CameraSamples

    samples = CameraSamples.from_packet(camera, frames)
    mats = np.repeat(np.eye(4)[None], len(frames), 0) if not world else samples.cam_to_world
    return samples.focal_xy_px(width), mats, samples.principal_px(width)


def normals_from_positions(p: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """相机空间位置图的相机空间法线 [H,W,3]（朝向相机）及其有效位置。每条轴上，像素从左右（上下）两个邻居中
    取深度更接近的那个，即同一表面上的邻居，使法线不会在物体边缘处倾斜。"""
    def tangent(axis: int) -> tuple[np.ndarray, np.ndarray]:
        fwd = np.roll(p, -1, axis) - p
        bwd = p - np.roll(p, 1, axis)
        fwd_ok = valid & np.roll(valid, -1, axis)
        bwd_ok = valid & np.roll(valid, 1, axis)
        last, first = [slice(None)] * 2, [slice(None)] * 2
        last[axis], first[axis] = -1, 0
        fwd_ok[tuple(last)] = False  # 画面边缘之外没有邻居
        bwd_ok[tuple(first)] = False
        use_fwd = fwd_ok & (~bwd_ok | (np.abs(fwd[..., 2]) <= np.abs(bwd[..., 2])))
        return np.where(use_fwd[..., None], fwd, bwd), fwd_ok | bwd_ok

    tx, ok_x = tangent(1)
    ty, ok_y = tangent(0)
    n = np.cross(ty, tx)  # 行号沿画面向下增加：结果指向表面外侧、朝向观察者
    n *= np.where(np.einsum("hwc,hwc->hw", n, -p) < 0, -1.0, 1.0)[..., None]
    length = np.linalg.norm(n, axis=-1)
    ok = ok_x & ok_y & (length > 0)
    return np.where(ok[..., None], n / np.maximum(length, 1e-12)[..., None], 0.0).astype(np.float32), ok


NORMAL_SPACES = i18n.Words("normals.space.", ("world", "camera"))  # value -> what normals in that space are called


class WorldPosition(NodeDef):
    id = "position_from_depth"
    on_node = ("space",)
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    inputs = (_depth_port(), _camera_port("depth"))
    outputs = (Port("position", "image.3", means=("space",), data=True),)

    class Params(NodeParams):
        space: Literal["world", "camera"] = P("world", group="position")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter, image_files, read_map
        from ..kit.unproject import unproject_depth

        depth, camera = _depth_and_camera(ctx)
        frames, w, h = depth.meta["frames"], depth.meta["width"], depth.meta["height"]
        focal, mats, pp = _view(camera, frames, w, ctx.params["space"] == "world")
        rows, cols = (a.ravel() for a in np.indices((h, w)))
        out = ExrWriter(ctx.outputs["position"], 3, validity=True, space=ctx.params["space"])
        files = image_files(depth)
        ctx.stage("compute_position")

        def position(job):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            i, f = job
            z, alpha = read_map(files[f])
            out.add(f, unproject_depth(z[..., 0], focal[i], mats[i], rows, cols, pp[i]).reshape(h, w, 3), alpha)

        list(ctx.each_done(enumerate(frames), position))
        return {"position": out.packet()}


def _depth_shade(z: np.ndarray) -> np.ndarray:
    """无颜色输入时点云的颜色：按深度的灰阶，近亮远暗。

    范围取 2%–98% 分位而非最小值 / 最大值：深度图边缘常有少量飞点，按极值拉伸会使整片点呈现同一颜色。
    深度全部相同（如一面墙）时取中性灰，不放大数值噪声。"""
    if not len(z):
        return np.zeros((0, 3), np.float32)
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
        return np.full((len(z), 3), 0.7, np.float32)
    near = 1.0 - np.clip((z - lo) / (hi - lo), 0.0, 1.0)  # 近处更亮
    grey = (0.15 + 0.8 * near).astype(np.float32)
    return np.repeat(grey[:, None], 3, axis=1)


# 飞点：深度跳变与法线折角同时成立才判为边缘。
#
# 依照 Pi3 官方的 `pi3/utils/geometry.py depth_normal_edge`（其自述为 conservative geometry edge mask），
# 默认值与之一致：相对深度差 3%、法线夹角 5 度、3×3 邻域。仅依据深度跳变会把掠射角下的斜面整片误判为边缘
# （一个合成房间中 27% 的有效像素被删除）；加入法线条件后，斜面（深度变化而法线不折）得以保留。
#
# 去飞点只在此处进行，不作用于深度图：深度图输出官方原值，去飞点只影响点云。
EDGE_RTOL = 0.03
EDGE_NORMAL_DEG = 5.0


def _neighbours(a: np.ndarray) -> list[np.ndarray]:
    """3×3 邻域的九个平移（边界按复制填充），形状与 `a` 相同。"""
    pad = ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2)
    p = np.pad(a, pad, mode="edge")
    h, w = a.shape[:2]
    return [p[y:y + h, x:x + w] for y in range(3) for x in range(3)]


def surface_normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """点图 [H,W,3] -> 每个像素的法线及可求出法线的位置，依照官方 `points_to_normals`：
    上左、左下、下右、右上四组叉积求和后归一化；任一组的两个邻居均有效即可求出。"""
    p = np.pad(np.where(valid[..., None], points, 0.0), ((1, 1), (1, 1), (0, 0)))
    ok = np.pad(valid, ((1, 1), (1, 1)))
    centre = p[1:-1, 1:-1]
    up, left, down, right = p[:-2, 1:-1] - centre, p[1:-1, :-2] - centre, p[2:, 1:-1] - centre, p[1:-1, 2:] - centre
    parts = [np.cross(up, left), np.cross(left, down), np.cross(down, right), np.cross(right, up)]
    pairs = [ok[:-2, 1:-1] & ok[1:-1, :-2], ok[1:-1, :-2] & ok[2:, 1:-1],
             ok[2:, 1:-1] & ok[1:-1, 2:], ok[1:-1, 2:] & ok[:-2, 1:-1]]
    pairs = [q & ok[1:-1, 1:-1] for q in pairs]
    total = np.zeros_like(centre)
    for part, has in zip(parts, pairs):
        unit = part / np.maximum(np.linalg.norm(part, axis=-1, keepdims=True), 1e-12)
        total += np.where(has[..., None], unit, 0.0)
    got = np.any(pairs, axis=0)
    normals = total / np.maximum(np.linalg.norm(total, axis=-1, keepdims=True), 1e-12)
    return np.where(got[..., None], normals, 0.0), got


def depth_edges(z: np.ndarray, points: np.ndarray | None = None, rtol: float = EDGE_RTOL,
                tol_deg: float = EDGE_NORMAL_DEG) -> np.ndarray:
    """位于真实几何边缘上的像素：深度跳变（3×3 邻域内最大值比最小值高出 `rtol` 以上）
    且法线折角（邻域内法线方向差超过 `tol_deg`）。

    `points`：该帧的三维点图 [H,W,3]（任意一致的坐标系均可，仅用于求法线方向）。
    未提供点图时退化为只看深度，这会把斜面一并删除，因此调用方应提供点图。"""
    d = np.maximum(np.asarray(z, np.float32), 1e-6)
    shifts = _neighbours(d)
    jump = np.max(shifts, axis=0) / np.min(shifts, axis=0) > 1.0 + rtol
    if points is None:
        return jump
    valid = np.isfinite(d) & (d > 0)
    normals, got = surface_normals(np.asarray(points, np.float32), valid)
    angle = np.zeros_like(d)
    for other, other_ok in zip(_neighbours(normals), _neighbours(got)):
        dot = np.clip(np.sum(normals * other, axis=-1), -1.0, 1.0)
        angle = np.maximum(angle, np.where(other_ok, np.arccos(dot), 0.0))
    angle = np.max(_neighbours(angle), axis=0)  # 对应官方实现中的 max_pool2d
    return jump & (angle > np.radians(float(tol_deg))) & got


def camera_grid(z: np.ndarray, focal_px) -> np.ndarray:
    """整幅深度图在相机空间中的三维点 [H,W,3]（针孔模型，主点位于画面中心），仅用于求法线。`focal_px`：(fx, fy)。"""
    h, w = z.shape
    y, x = np.mgrid[0:h, 0:w]
    return np.stack([(x + 0.5 - w / 2) * z / focal_px[0], (y + 0.5 - h / 2) * z / focal_px[1], z], axis=-1)


def points_from_depth(ctx, depth, camera, image, mask, step: int, point_size: float, space: str, confidence=None,
                      min_confidence: float = 0.5, native=None, scale_cm: float = M_TO_CM):
    """深度图 + 相机（+ 可选的颜色、排除遮罩、置信度）→ 点云 Packet（单位厘米，颜色取自 `image`）。这是 `DepthToPoints`
    的核心逻辑，单独拆出供家族节点自带的点云输出口（kit/maps.py family_points）和扩展（lab2shot/sdk）直接调用。`mask`：值 > 0.5
    的像素不进入点云（用于排除运动物体等）；`confidence`：置信度低于 `min_confidence` 的像素不进入点云；
    `depth`/`camera`/`image` 为已计算好的 Packet，不经过 ctx.input。

    `scale_cm`：模型一个单位对应的厘米数。原生点图必须与深度、相机使用同一数值：节点上的「尺度」参数
    （`unit_cm`）改变时三者必须同步改变，否则点云与相机不一致。

    `native`：(帧) -> 模型自身计算的相机空间点图 [H,W,3]（OpenCV 轴向、模型自身单位），没有时为 None。提供时使用模型的点，
    仅将其转换为内部单位和轴向并按相机放入世界；未提供时由深度和焦距反投影。采样、着色、排除遮罩、置信度门槛、
    点大小和坐标系均相同，区别仅在于三维坐标的来源。"""
    from ...data.payloads import display_rgb, file_at, image_files, points_packet, read_map
    from ...data.maps import map_at, same_size
    from ..kit.unproject import unproject_depth
    from ...data.units import CV_TO_GL

    same_size({ctx.node_type.port_label("depth"): depth, ctx.node_type.port_label("mask"): mask, ctx.node_type.port_label("confidence"): confidence})
    frames, w, h = depth.meta["frames"], depth.meta["width"], depth.meta["height"]
    focal, mats, pp = _view(camera, frames, w, space == "world")
    files = image_files(depth)
    ctx.stage("make_points")

    def points(job):
        """一帧的点（及颜色、是否取自模型的点图）：各帧互不相干，由引擎逐帧并行（ctx.each_done），按帧序收集。"""
        i, f = job
        z, alpha = read_map(files[f])
        keep = alpha[::step, ::step] > 0
        # 飞点：前后景交界处反投影出的点悬在半空。判据需要整幅三维点（法线条件需要邻居），因此先按模型点图或
        # 针孔反投影生成一张点图（仅用于求法线方向，量纲和轴向不影响夹角）。
        model_grid = np.asarray(native(f), np.float64) if native is not None else None
        grid = model_grid if model_grid is not None else camera_grid(z[..., 0], focal[i])
        keep &= ~depth_edges(z[..., 0], grid)[::step, ::step]
        excluded = map_at(mask, f) if mask is not None else None  # 静态遮罩在每一帧都生效
        if excluded is not None:
            keep &= excluded[0][::step, ::step, 0] <= 0.5
        sure = map_at(confidence, f) if confidence is not None else None  # 置信度缺少该帧时不剔除任何点
        if sure is not None:
            keep &= sure[0][::step, ::step, 0] >= min_confidence
        r, c = np.nonzero(keep)
        r, c = r * step, c * step
        model = model_grid
        if model is not None:
            # 模型的点：模型单位 -> 厘米（scale_cm，与深度、相机相同），
            # OpenCV 轴向（+Y 向下、+Z 向前）-> 内部轴向（+Y 向上、+Z 朝向相机），再按该帧相机放入世界。
            # 与 camera_points 反投影之后的步骤相同，因此两条路径落在同一坐标系中。
            got = np.asarray(model, np.float64)[r, c] * CV_TO_GL * scale_cm
            placed = (got @ mats[i][:3, :3].T + mats[i][:3, 3]).astype(np.float32)
        else:
            placed = unproject_depth(z[..., 0], focal[i], mats[i], r, c, pp[i]).astype(np.float32)
        colored = image is not None and file_at(image, f) is not None
        # 未连接「颜色」时按深度着色（中性灰无法体现远近）：近亮远暗，
        # 范围取该帧保留点的 2%–98% 分位，以避开个别飞点。
        rgb = display_rgb(image, f, w, h)[r, c] if colored else _depth_shade(z[..., 0][r, c])
        return placed, rgb.astype(np.float32), model is not None

    made = list(ctx.each_done(enumerate(frames), points))
    pts, cols = [p for p, _, _ in made], [c for _, c, _ in made]
    from_native = sum(own for _, _, own in made)
    # 声明了原生点图却有帧未写出时，静默改用反投影会掩盖差异，因此不这样处理。点的来源不在此处以消息说明：
    # 它是该数据恒定的属性，写入 meta 的 points_from，由「数据信息」常驻显示；计算消息在命中缓存时不会再出现。
    if native is not None and from_native != len(frames):
        ctx.say("W-POINTS-NONATIVE", native=from_native, frames=len(frames))
    points_from = "native" if from_native else "depth"
    made_from = {"depth": depth.fingerprint, "camera": camera.fingerprint, "step": int(step), "space": space,
                 "source": points_from,
                 "mask": mask.fingerprint if mask is not None else None,
                 "confidence": confidence.fingerprint if confidence is not None else None,
                 "min_confidence": float(min_confidence) if confidence is not None else None}
    return points_packet(ctx.outputs["points"], frames, "points", pts, cols,
                         scale=meant(ctx, depth, "scale", ctx.node_type.port_label("depth")), width_cm=point_size, width=w, height=h,
                         depth_grid=made_from, points_from=points_from)


class DepthToPoints(NodeDef):
    id = "points_from_depth"
    on_node = ("point_step", "space")
    version = 5  # 结果变化时递增，work/ 中的旧结果随之不再命中缓存（engine/cook.py）
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    inputs = (
        _depth_port(),
        _camera_port("depth"),
        Port("image", "image.3", optional=True, data=False, expects=(SameShot("depth"),)),
        Port("mask", "image.1", optional=True, expects=(SameShot("depth"),)),
        Port("confidence", "image.1", optional=True, expects=(SameShot("depth"),)),
    )
    outputs = (Port("points", "scene.points"),)

    class Params(NodeParams):
        point_step: int = P(4, ge=1, le=64, group="point_cloud")
        point_size: float = P(0.5, unit="cm", gt=0, le=100, group="point_cloud")
        space: Literal["world", "camera"] = P("world", group="point_cloud")
        conf_threshold: float = P(0.5, ge=0, le=1, group="point_cloud", applies=Wired("confidence"))

    @classmethod
    def cook(cls, ctx):
        depth, camera = _depth_and_camera(ctx)
        image, mask = ctx.input("image"), ctx.input("mask")
        return {"points": points_from_depth(ctx, depth, camera, image, mask, ctx.params["point_step"],
                                            ctx.params["point_size"], ctx.params["space"],
                                            ctx.input("confidence"), ctx.params["conf_threshold"])}


class DepthNormal(NodeDef):
    id = "normal_from_depth"
    on_node = ("space",)
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    inputs = (_depth_port(scale_matters=False), _camera_port("depth"))
    outputs = (normal_port(),)

    class Params(NodeParams):
        space: Literal["camera", "world"] = P("camera", group="normals")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import SIGNED, ExrWriter, image_files, read_map
        from ..kit.unproject import unproject_depth

        depth, camera = _depth_and_camera(ctx)
        frames, w, h = depth.meta["frames"], depth.meta["width"], depth.meta["height"]
        world = ctx.params["space"] == "world"
        focal, mats, pp = _view(camera, frames, w, world)
        rotations = orthonormal(np.asarray(mats, np.float64)[:, :3, :3])
        rows, cols = (a.ravel() for a in np.indices((h, w)))
        out = ExrWriter(ctx.outputs["normal"], 3, validity=True, value_range=SIGNED, half=True, space=ctx.params["space"])
        files = image_files(depth)
        ctx.stage("compute_normals")

        def normal(job):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            i, f = job
            z, alpha = read_map(files[f])
            p = unproject_depth(z[..., 0], focal[i], np.eye(4), rows, cols, pp[i]).reshape(h, w, 3)
            n, ok = normals_from_positions(p, alpha > 0)
            out.add(f, n @ rotations[i].T if world else n, ok)

        list(ctx.each_done(enumerate(frames), normal))
        return {"normal": out.packet()}


class NormalSpace(NodeDef):
    id = "normal_space"
    on_node = ("space",)
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    inputs = (values_port("normal", "image.3", means=("space",), expects=(NotAlready("space", NORMAL_SPACES),)),
              _camera_port("normal"))
    outputs = (normal_port(),)

    class Params(NodeParams):
        space: Literal["world", "camera"] = P("world", group="normals")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import SIGNED, ExrWriter, image_files, read_map

        normal, camera = ctx.input("normal"), ctx.input("camera")
        frames, w, target = normal.meta["frames"], normal.meta["width"], ctx.params["space"]
        source = meant(ctx, normal, "space", ctx.node_type.port_label("normal"))
        rotations = orthonormal(np.asarray(_view(camera, frames, w, True)[1], np.float64)[:, :3, :3])
        out = ExrWriter(ctx.outputs["normal"], 3, validity=True, value_range=SIGNED, half=True, space=target)
        files = image_files(normal)
        ctx.stage("convert_normals")

        def turn(job):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            i, f = job
            n, alpha = read_map(files[f])
            n = n[..., :3]  # 四通道输入可接入三通道端口（第四通道 alpha 随行）：此处只转换前三个通道
            r = np.eye(3) if source == target else rotations[i] if target == "world" else rotations[i].T
            out.add(f, n @ r.T, alpha)

        list(ctx.each_done(enumerate(frames), turn))
        return {"normal": out.packet()}


# ------------------------------------------------------------------ 跟踪点转 3D


def depth_under(z: np.ndarray, valid: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """像素 xy [N,2]（像素中心在 +0.5）处的深度：取该像素自身的值；无效时取其 3 x 3 邻域内有效值的中位数
    （边缘或空洞上的点）；仍无则为 nan。"""
    h, w = z.shape
    c = np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 1)
    out = np.where(valid[r, c], z[r, c], np.nan)
    for i in np.flatnonzero(~valid[r, c]):
        patch, ok = z[max(0, r[i] - 1):r[i] + 2, max(0, c[i] - 1):c[i] + 2], valid[max(0, r[i] - 1):r[i] + 2, max(0, c[i] - 1):c[i] + 2]
        out[i] = np.median(patch[ok]) if ok.any() else np.nan
    return out


class TracksToPoints(NodeDef):
    id = "points_from_tracks"
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    inputs = (
        Port("tracks", "tracks2d"),
        _depth_port(),
        _camera_port("tracks"),
        Port("image", "image.3", optional=True, data=False, expects=(SameShot("tracks"),)),
    )
    # 端口名为 `tracks3d` 而非 `points`：`points` 在全项目中指「点云」，此处输出的是带编号、整段跟随同一点的
    # 「3D 跟踪点」，与 families/tracks3d.py 中的端口同名。
    outputs = (Port("tracks3d", "scene.points"),)
    on_node = ("space",)

    class Params(NodeParams):
        space: Literal["world", "camera"] = P("world", group="points")
        point_size: float = P(1.0, unit="cm", gt=0, le=100, group="points")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import image_files, read_map, read_tracks, start_colours, tracked_points_packet
        from ...data.maps import same_size
        from ..kit.unproject import camera_points

        tracks, image = ctx.input("tracks"), ctx.input("image")
        depth, camera = _depth_and_camera(ctx)
        same_size({ctx.node_type.port_label("depth"): depth, ctx.node_type.port_label("tracks"): tracks})
        t = read_tracks(tracks)
        frames, w, h = tracks.meta["frames"], tracks.meta["width"], tracks.meta["height"]
        files = image_files(depth)
        have = [f for f in frames if f in files]
        if not have:
            raise Invalid(Msg("E-POINTS-NOCOMMON"))
        if len(have) < len(frames):
            ctx.say("N-POINTS-DEPTHGAP", count=len(frames) - len(have), port="depth")
        focal, mats, pp = _view(camera, frames, w, ctx.params["space"] == "world")
        n = len(t["tracks"])
        xyz = np.full((n, len(frames), 3), np.nan)
        seen = t["visible"].copy()
        ctx.stage("depth_to_camera_world")

        def place(job):
            """一帧：（该帧可见的点，其中取到深度的，它们的位置）；该帧没有深度时为 None。各帧互不相干，
            由引擎逐帧并行（ctx.each_done），结果按帧序填入。"""
            j, f = job
            if f not in files:
                return None
            z, alpha = read_map(files[f])
            on = np.flatnonzero(t["visible"][:, j])
            d = depth_under(z[..., 0], alpha > 0, t["tracks"][on, j])
            ok = np.isfinite(d)
            xy = t["tracks"][on[ok], j]
            return on, ok, camera_points(d[ok], xy[:, 0], xy[:, 1], focal[j], w, h, mats[j], pp[j])

        for j, got in enumerate(ctx.each_done(enumerate(frames), place)):
            if got is None:
                seen[:, j] = False
                continue
            on, ok, placed = got
            seen[on[~ok], j] = False
            xyz[on[ok], j] = placed
        lost = int((~np.isfinite(xyz).all(-1).any(1)).sum())
        if lost:
            ctx.say("N-POINTS-NODEPTH", count=lost)
        colours = start_colours(image, t, frames) if image is not None else None
        return {"tracks3d": tracked_points_packet(ctx.outputs["tracks3d"], frames, "tracks", xyz, seen,
                                                  scale=meant(ctx, depth, "scale", ctx.node_type.port_label("depth")), colors=colours, width_cm=ctx.params["point_size"],
                                                  confidence=t.get("confidence"), width=w, height=h)}


# ------------------------------------------------------------------ 深度对齐

KEEP = 0.8  # 拟合保留的点对比例：其余（运动物体、天空、边缘、被遮挡的点）被剔除
SAMPLES_PER_FRAME = 40_000
MIN_PAIRS = 64
MODEL_TEXT = i18n.Words("depth_align.model.", ("scale", "affine", "inverse"))


def _evenly(n: int, k: int) -> np.ndarray:
    return np.linspace(0, n - 1, min(n, k)).astype(np.int64) if n else np.zeros(0, np.int64)


def _trimmed_offset(d: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """`d` 的稳健中心：取离中心最近的 KEEP 比例的加权（`w`）均值，从中位数开始迭代直至收敛。"""
    c = float(np.median(d))
    inliers = np.ones(len(d), bool)
    for _ in range(20):
        r = np.abs(d - c)
        inliers = r <= np.quantile(r, KEEP)
        new = float(np.average(d[inliers], weights=w[inliers]))
        if abs(new - c) < 1e-12:
            break
        c = new
    return c, inliers


def _trimmed_line(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
    """最小二乘拟合 y ≈ s·x + t（每个点对按 `w` 加权），只用拟合最好的 KEEP 比例点对，迭代直至该集合不再变化
    （截尾最小二乘）。"""
    a = np.stack([x, np.ones_like(x)], 1)
    root = np.sqrt(w)
    inliers = np.ones(len(x), bool)
    for _ in range(20):
        (s, t), *_ = np.linalg.lstsq(a[inliers] * root[inliers, None], y[inliers] * root[inliers], rcond=None)
        r = np.abs(y - (s * x + t))
        new = r <= np.quantile(r, KEEP)
        if np.array_equal(new, inliers):
            break
        inliers = new
    return float(s), float(t), inliers


def align_model(src, ctx=None) -> str:
    """源数据到深度的映射方式：视差按 1/z = s·x + t，差一个偏移的深度按 z = s·x + t，其他深度按比例因子。

    深度和视差都是单通道图，无法按通道数区分；可区分的是数据自带的尺度：视差图标有 scale = disparity，
    深度图标有「真实尺度 / 只知比例 / 还差一个偏移」。未标明的数据（其他软件渲染的 EXR、手工绘制的图）按节点上的
    「待对齐的是」确定；该参数也为「自动」时按视差处理（只知远近，是更安全的假设），并在节点上给出提示。
    不得静默采用某种猜测，否则结果错误时使用者会误以为素材有问题。"""
    scale = src.meta.get("scale")
    if not scale and ctx is not None:
        said = ctx.params.get("depth_is", "auto")
        scale = None if said == "auto" else said
        if scale is None:
            ctx.say("N-DEPTHALIGN-ASSUMEDDISPARITY")
    if not scale or scale == "disparity":
        return "inverse"
    return "affine" if scale == "affine" else "scale"


def fit_depth(model: str, x: np.ndarray, z: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    """(s, t, inliers)：把源值 `x` 映射到参考深度 `z`（厘米）。"scale" 为 z = s·x（在对数深度上拟合，使远近权重相同），
    "affine" 为 z = s·x + t，"inverse" 为 1/z = s·x + t（视差）。`w`：每个点对的权重（置信度；默认相同）。"""
    w = np.ones(len(x)) if w is None else w
    if model == "scale":
        c, inliers = _trimmed_offset(np.log(z) - np.log(x), w)
        return float(np.exp(c)), 0.0, inliers
    if model == "affine":
        return _trimmed_line(x, z, w)
    return _trimmed_line(x, 1.0 / z, w)


def apply_fit(model: str, s: float, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """源值对齐后的深度（厘米）及其有效位置（位于相机前方且有限）。"""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if model == "inverse":
            inv = s * x + t
            z = 1.0 / inv
            ok = inv > 1e-7  # 距离小于 100 km
        else:
            z = s * x + t
            ok = z > 0
    ok &= np.isfinite(z)
    return np.where(ok, z, 0.0), ok


class DepthAlign(NodeDef):
    id = "depth_align"
    version = 2  # 2: the summary names its reference and what the source was (reference, source_scale)
    on_node = ("fit",)
    keeps_overscan = False  # 通过相机在画面框内计算：框外像素在三维中没有对应位置
    category = "geometry_tools"
    # 对齐后的深度采用参考的尺寸；每帧都对参考拟合，与画面无关
    inputs = (
        Port("depth", "image.1"),
        Port("reference", "image.1", optional=True, expects=(SameShot("depth"),)),
        Port("points", "scene.points", optional=True),
        _camera_port("depth", optional=True),
        Port("mask", "image.1", optional=True, expects=(SameShot("depth"),)),
        Port("confidence", "image.1", optional=True, expects=(SameShot("depth"),)),
    )
    outputs = (Port("depth", "image.1", means=("scale",)),)

    class Params(NodeParams):
        # 参数名为 depth_is 而非 source：同一参数名在全项目中只能指一件事，「自动落地」上的 source 已表示「地面依据」
        depth_is: Literal["auto", "disparity", "relative", "affine"] = P(
            "auto", group="align",
        )
        fit: Literal["shot", "frame"] = P(
            "shot", group="align",
        )

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter
        from ...data.maps import map_at, same_size

        src, ref, points = ctx.input("depth"), ctx.input("reference"), ctx.input("points")
        if (ref is None) == (points is None):
            raise Invalid(Msg("E-DEPTHALIGN-NOREF"))
        model = align_model(src, ctx)
        same_size({ctx.node_type.port_label("depth"): src, ctx.node_type.port_label("reference"): ref, ctx.node_type.port_label("mask"): ctx.input("mask"), ctx.node_type.port_label("confidence"): ctx.input("confidence")})
        if ref is not None:
            if ref.meta.get("scale") in ("affine", "disparity", None):
                raise Invalid(Msg("E-DEPTHALIGN-AFFINEREF"))
            out_scale = ref.meta["scale"]  # 参考必须是真实深度（上一行已拦下视差和差偏移的深度）
            pairs = cls._pairs_from_depth(ctx, src, ref, model)
        else:
            out_scale = points.meta.get("scale", "relative")  # 点云单位为厘米，或只知比例（COLMAP）
            pairs = cls._pairs_from_points(ctx, src, points, model)
        if sum(len(x) for x, _, _ in pairs.values()) < MIN_PAIRS:
            raise Invalid(Msg("E-DEPTHALIGN-NOOVERLAP"))

        x_all = np.concatenate([x for x, _, _ in pairs.values()])
        z_all = np.concatenate([z for _, z, _ in pairs.values()])
        s, t, inliers = fit_depth(model, x_all, z_all, np.concatenate([w for _, _, w in pairs.values()]))
        if s <= 0:
            raise Invalid(Msg("E-DEPTHALIGN-NEGATIVE"))
        frames = src.meta["frames"]
        fits = dict.fromkeys(frames, (s, t))
        errors = [cls._error(model, s, t, x_all[inliers], z_all[inliers])]
        if ctx.params["fit"] == "frame":
            own = {f: fit_depth(model, x, z, w) for f, (x, z, w) in pairs.items() if len(x) >= MIN_PAIRS}
            own = {f: fit for f, fit in own.items() if fit[0] > 0}
            fits.update({f: (fs, ft) for f, (fs, ft, _) in own.items()})
            errors = [cls._error(model, fs, ft, pairs[f][0][inl], pairs[f][1][inl]) for f, (fs, ft, inl) in own.items()] or errors
            if len(own) < len(frames):
                ctx.say("N-DEPTHALIGN-FEWFRAMES", count=len(frames) - len(own))

        ctx.stage("write_aligned_depth")
        # reference: what it was aligned to (the reference depth's or point cloud's fingerprint), so results aligned to
        # one reference can be told from others (expects.py SameReference); source_scale: what the source was (its
        # own scale meaning)
        summary = {"model": model, "fit": ctx.params["fit"], "scale": s, "offset": t, "frames": len(pairs),
                   "pixels": int(len(x_all)), "error_pct": float(np.median(errors)),
                   "reference": (ref if ref is not None else points).fingerprint, "source_scale": src.meta.get("scale")}
        out = ExrWriter(ctx.outputs["depth"], 1, validity=True, scale=out_scale, aligned=summary)

        def align(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            x, alpha = map_at(src, f)
            z, ok = apply_fit(model, *fits[f], x[..., 0].astype(np.float64))
            out.add(f, z.astype(np.float32), (alpha > 0) & ok)

        list(ctx.each_done(frames, align))
        code, said = cls._message(model, s, t, summary, sorted({fs for fs, _ in fits.values()}))
        ctx.say(code, **said)
        return {"depth": out.packet()}

    @staticmethod
    def _usable(model: str, x: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.isfinite(x) & (x > 0) if model == "scale" else np.isfinite(x)

    @classmethod
    def _region(cls, ctx, frame: int) -> np.ndarray | None:
        """`frame` 处的对齐区域遮罩（> 0.5），仅当已连接且包含该帧时。"""
        from ...data.maps import map_at

        mask = ctx.input("mask")
        got = map_at(mask, frame) if mask is not None else None
        return None if got is None else got[0][..., 0] > 0.5

    @classmethod
    def _weights(cls, ctx, frame: int, shape: tuple[int, int]) -> np.ndarray:
        """`frame` 处每个像素点对的权重：已连接的置信度（缺少该帧时权重相同）。"""
        from ...data.maps import map_at

        confidence = ctx.input("confidence")
        got = map_at(confidence, frame) if confidence is not None else None
        return np.ones(shape) if got is None else got[0][..., 0].astype(np.float64)

    @classmethod
    def _pairs_from_depth(cls, ctx, src, ref, model: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """逐帧：两者均有值（且置信度不为 0）的像素处的（源值、参考深度、权重）采样。"""
        from ...data.maps import map_at

        ctx.stage("read_reference_depth")

        def sample(f):  # 一帧的点对：各帧互不相干，由引擎逐帧并行（ctx.each_done），按帧序收集
            got = map_at(ref, f)
            if got is None:
                return None
            (x, xa), (z, za) = map_at(src, f), got
            weight = cls._weights(ctx, f, x.shape[:2])
            ok = (xa > 0) & (za > 0) & (z[..., 0] > 0) & cls._usable(model, x[..., 0]) & (weight > 0)
            region = cls._region(ctx, f)
            if region is not None:
                ok &= region
            pick = _evenly(int(ok.sum()), SAMPLES_PER_FRAME)
            return x[..., 0][ok][pick].astype(np.float64), z[..., 0][ok][pick].astype(np.float64), weight[ok][pick]

        frames = src.meta["frames"]
        return {f: got for f, got in zip(frames, ctx.each_done(frames, sample)) if got is not None}

    @classmethod
    def _pairs_from_points(cls, ctx, src, points, model: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """逐帧：通过相机可见的点（每像素取最近者）、其沿视线的距离、其所在像素的源值及权重（该处的置信度）。"""
        from pxr import Usd

        from ...data.evaluate import scene_points
        from ...data.maps import map_at
        from ..kit.raster import nearest
        from ...data.scene import open_scene

        camera = ctx.input("camera")
        if camera is None:
            raise Invalid(Msg("E-DEPTHALIGN-NOCAMERA"))
        w, h = src.meta["width"], src.meta["height"]
        frames = [f for f in src.meta["frames"] if not points.meta["frames"] or f in points.meta["frames"]]
        focal, mats, pp = _view(camera, frames, w, True)
        stage = open_scene([points])
        pairs = {}
        ctx.stage("project_points")
        for i, f in enumerate(ctx.each(frames)):
            clouds = [p for _, p, _, _ in scene_points(stage, Usd.TimeCode(f))]
            if not clouds:
                continue
            world = np.concatenate(clouds)
            inv = np.linalg.inv(mats[i])
            pc = world @ inv[:3, :3].T + inv[:3, 3]
            z = -pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = focal[i][0] * pc[:, 0] / z + pp[i][0]  # 经过相机主点，与反投影一致
                v = -focal[i][1] * pc[:, 1] / z + pp[i][1]
            seen = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            pixel = (np.floor(v[seen]).astype(np.int64) * w + np.floor(u[seen]).astype(np.int64))
            z = z[seen]
            keep = nearest(pixel, 1.0 / z)  # 被更近的点遮挡的点不能说明该像素的情况
            pixel, z = pixel[keep], z[keep]
            x, xa = map_at(src, f)
            weight = cls._weights(ctx, f, x.shape[:2]).reshape(-1)[pixel]
            ok = (xa.reshape(-1)[pixel] > 0) & cls._usable(model, x[..., 0].reshape(-1)[pixel]) & (weight > 0)
            region = cls._region(ctx, f)
            if region is not None:
                ok &= region.reshape(-1)[pixel]
            pick = _evenly(int(ok.sum()), SAMPLES_PER_FRAME)
            pairs[f] = (x[..., 0].reshape(-1)[pixel][ok][pick].astype(np.float64), z[ok][pick], weight[ok][pick])
        return pairs

    @staticmethod
    def _error(model: str, s: float, t: float, x: np.ndarray, z: np.ndarray) -> float:
        """对齐后深度相对参考深度的中位相对误差（%）。"""
        aligned, ok = apply_fit(model, s, t, x)
        return float(np.median(np.abs(aligned[ok] - z[ok]) / z[ok]) * PERCENT) if ok.any() else float("nan")

    @staticmethod
    def _message(model: str, s: float, t: float, summary: dict, scales: list[float]) -> tuple[str, dict]:
        """拟合结果对应的消息：代码及参数（数值保留四位有效数字，偏移带符号）。"""
        if model == "scale":
            how = Msg("I-DEPTHALIGN-SCALE", scale=f"{s:.4g}")
        elif model == "affine":
            how = Msg("I-DEPTHALIGN-AFFINE", scale=f"{s:.4g}", offset=f"{t:+.4g}")
        else:  # 逆深度以 1/m 表示比 1/cm 更易读
            how = Msg("I-DEPTHALIGN-INVERSE", scale=f"{s * M_TO_CM:.4g}", offset=f"{t * M_TO_CM:+.4g}")
        per_frame = ""
        if summary["fit"] == "frame" and len(scales) > 1:
            per_frame = Msg("I-DEPTHALIGN-PERFRAME", low=f"{min(scales):.4g}", high=f"{max(scales):.4g}")
        return "I-DEPTHALIGN-RESULT", dict(model=MODEL_TEXT[model], how=how, per_frame=per_frame, frames=summary["frames"],
                                          pixels=summary["pixels"], dropped=round((1 - KEEP) * PERCENT), error=summary["error_pct"])


# ------------------------------------------------------------------ 点云融合


def _voxelize(xyz: np.ndarray, rgb: np.ndarray | None, voxel: float) -> tuple[np.ndarray, np.ndarray | None]:
    """把所有点拼在一起后按体素去重：每个体素一个质心（+ 平均颜色）。体素边长 `voxel`（厘米）。

    按体素整数坐标的线性索引分组（比在 [N,3] 上 unique 快、省内存）：同一体素内的点坐标累加后除以点数。"""
    idx = np.floor(xyz / voxel).astype(np.int64)
    lo = idx.min(0)
    shape = idx.max(0) - lo + 1
    if (shape <= 0).any():
        return np.zeros((0, 3), np.float32), (np.zeros((0, 3), np.float32) if rgb is not None else None)
    lin = (idx[:, 0] - lo[0]) * shape[1] * shape[2] + (idx[:, 1] - lo[1]) * shape[2] + (idx[:, 2] - lo[2])
    uniq, inverse, counts = np.unique(lin, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3), np.float64)
    np.add.at(sums, inverse, xyz)
    centroid = (sums / counts[:, None]).astype(np.float32)
    if rgb is None:
        return centroid, None
    cs = np.zeros((len(uniq), 3), np.float32)
    np.add.at(cs, inverse, rgb)
    return centroid, (cs / counts[:, None]).astype(np.float32)


SLAB_VOXELS = 2_000_000  # voxels handled at once (integration and extraction): a few hundred MB of temporaries
# bytes per voxel the fusion keeps for the whole run: tsdf + weight (float32), + colour (3 float32) when kept
VOXEL_BYTES, COLOR_BYTES = 8, 12
FUSE_MEMORY_SHARE = 0.5  # of the memory free when the cook starts: the rest stays for the slabs and everyone else
# Largest grid fused at the asked voxel size; a bigger scene (a street, a landscape) is fused with a coarser voxel so
# the grid stays a few GB and the integration finishes in minutes. The cook says which voxel it used.
FUSE_MAX_VOXELS = 1 << 27


def _slabs(lo: np.ndarray, hi: np.ndarray):
    """[lo, hi] (inclusive voxel indices) cut along x into slabs of at most SLAB_VOXELS voxels: (x0, x1) half-open."""
    plane = int(np.prod(hi[1:] - lo[1:] + 1))
    step = max(1, SLAB_VOXELS // max(plane, 1))
    for x0 in range(int(lo[0]), int(hi[0]) + 1, step):
        yield x0, min(x0 + step, int(hi[0]) + 1)


def _tsdf_surface(tsdf: np.ndarray, weight: np.ndarray, color: np.ndarray | None,
                  origin: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray | None]:
    """TSDF 过零提取表面点：沿每根体素边，TSDF 值变号（两侧都观测到）时按数值线性插值出表面点，
    颜色同样插值。截断 TSDF 里表面就是 TSDF 从正（体素在表面前方）到负（后方）穿过零的位置。

    按 x 切片做（每片多带一层，x 方向的边跨片也不漏）：临时数组只和一片一样大，坐标只为过零的边算。"""
    pts, cols = [], []
    dims = np.asarray(tsdf.shape)
    for x0, x1 in _slabs(np.zeros(3, np.int64), dims - 1):
        hi = min(x1 + 1, int(dims[0]))  # one more layer: the x-edges leaving this slab
        t, wt = tsdf[x0:hi], weight[x0:hi]
        c = color[x0:hi] if color is not None else None
        own = x1 - x0  # the layers this slab owns; an x-edge from its last layer ends in the extra one
        for a in range(3):
            sl0, sl1 = [slice(0, own), slice(None), slice(None)], [slice(0, own), slice(None), slice(None)]
            if a == 0:
                n = min(own, len(t) - 1)
                sl0[0], sl1[0] = slice(0, n), slice(1, n + 1)
            else:
                sl0[a], sl1[a] = slice(0, -1), slice(1, None)
            t0, t1 = t[tuple(sl0)], t[tuple(sl1)]
            cross = (np.sign(t0) * np.sign(t1) < 0) & (wt[tuple(sl0)] > 0) & (wt[tuple(sl1)] > 0)
            at = np.nonzero(cross)
            if not len(at[0]):
                continue
            v0, v1 = t0[at], t1[at]
            f = v0 / (v0 - v1 + 1e-12)  # 线性插值系数：0 在 t0 侧、1 在 t1 侧
            center = origin + (np.stack(at, -1) + [x0, 0, 0] + 0.5) * voxel
            center[:, a] += f * voxel
            pts.append(center.astype(np.float32))
            if c is not None:
                cols.append((c[tuple(sl0)][at] * (1 - f[:, None]) + c[tuple(sl1)][at] * f[:, None]).astype(np.float32))
    if not pts:
        return np.zeros((0, 3), np.float32), (np.zeros((0, 3), np.float32) if color is not None else None)
    return np.concatenate(pts), (np.concatenate(cols) if cols else None)


def _tsdf_fuse(read, frames, focal, mats, pp, w, h, voxel, trunc_cm, want_color, paint=None):
    """逐帧深度 + 相机 → TSDF 融合出的表面点、颜色、融合前的总观测点数（去重对比用）、实际用的体素大小。

    `read(f)` 返回 (深度 z [H,W] 厘米, 有效 alpha [H,W])；`focal` [F,2]（fx, fy）、`mats` [F,4,4]（世界）、`pp` [F,2]
    为每帧的像素焦距 / 相机到世界矩阵 / 主点。`paint(i, f)` 返回该帧画面的颜色 [H,W,3]（显示用 sRGB 0..1）或 None
    （没接画面、或画面缺这一帧：按远近着色）。体素的颜色取它投到的那个像素——正是给它 SDF 的那个深度像素，深度与
    颜色同一像素一一对应；多帧按 TSDF 权重平均。体素网格常驻（每体素 8 字节，带颜色 20 字节），开之前按这次计算
    开始时的空闲内存核过，放不下就请使用者调大体素；逐帧积分按 x 切片做，临时数组只和一片一样大。"""
    from lab2shot_shared.memory import available_gb

    from ..kit.unproject import unproject_depth

    def bounds(i, f):
        z, alpha = read(f)
        z = z.astype(np.float64)
        valid = (alpha > 0) & (z > 0) & np.isfinite(z)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            return None, 0
        pts = unproject_depth(z, focal[i], mats[i], rows, cols, pp[i])
        # 包围盒取每帧 0.5%–99.5% 分位而非最小 / 最大：深度图边缘的飞点、结构光的噪声点会把盒子撑到十几米外，
        # 体素网格随之爆炸；真实表面都在中间 99% 里。
        return np.percentile(pts, [0.5, 99.5], axis=0), int(len(rows))

    found = [bounds(i, f) for i, f in enumerate(frames)]
    robust = [r for r, _ in found if r is not None]
    seen = sum(n for _, n in found)
    if not robust:
        raise Invalid(Msg("E-FUSE-NOPOINTS"))
    lo = np.min([r[0] for r in robust], axis=0) - trunc_cm
    hi = np.max([r[1] for r in robust], axis=0) + trunc_cm
    span = np.prod((hi - lo) / voxel + 1)
    if span > FUSE_MAX_VOXELS:  # coarser voxel, same truncation in voxels
        grow = float(np.cbrt(span / FUSE_MAX_VOXELS)) * 1.01
        voxel *= grow
        trunc_cm *= grow
        lo -= trunc_cm * (1 - 1 / grow)
        hi += trunc_cm * (1 - 1 / grow)
    origin = lo
    dims = np.ceil((hi - lo) / voxel).astype(np.int64) + 1
    voxels = int(np.prod(dims))
    budget = available_gb() * FUSE_MEMORY_SHARE * (1 << 30)
    if voxels * (VOXEL_BYTES + (COLOR_BYTES if want_color else 0)) > budget:
        raise Invalid(Msg("E-FUSE-TOOBIG", voxels=voxels))
    tsdf = np.full(dims, trunc_cm, np.float32)
    weight = np.zeros(dims, np.float32)
    color = np.zeros((*dims, 3), np.float32) if want_color else None
    flat_t, flat_w = tsdf.reshape(-1), weight.reshape(-1)
    flat_c = color.reshape(-1, 3) if color is not None else None

    for i, f in enumerate(frames):
        z, alpha = read(f)
        zf = z.astype(np.float32)
        valid = (alpha > 0) & (zf > 0) & np.isfinite(zf)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            continue
        pts = unproject_depth(zf, focal[i], mats[i], rows, cols, pp[i])
        ilo = np.clip(np.floor((pts.min(0) - trunc_cm - origin) / voxel).astype(np.int64), 0, dims - 1)
        ihi = np.clip(np.floor((pts.max(0) + trunc_cm - origin) / voxel).astype(np.int64), 0, dims - 1)
        inv = np.linalg.inv(mats[i])
        shade = None
        if flat_c is not None:  # [H*W,3]：画面的颜色；不接画面就按远近着色（与「深度转点云」一致）
            rgb = paint(i, f) if paint is not None else None
            shade = (np.asarray(rgb, np.float32).reshape(-1, 3) if rgb is not None
                     else _depth_shade(zf.reshape(-1)))
        for x0, x1 in _slabs(ilo, ihi):
            gx, gy, gz = (g.reshape(-1) for g in np.mgrid[x0:x1, ilo[1]:ihi[1] + 1, ilo[2]:ihi[2] + 1])
            centers = origin + (np.stack([gx, gy, gz], -1) + 0.5) * voxel
            pc = centers @ inv[:3, :3].T + inv[:3, 3]
            zv = -pc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = focal[i][0] * pc[:, 0] / zv + pp[i][0]
                v = -focal[i][1] * pc[:, 1] / zv + pp[i][1]
            uf, vf = np.floor(u), np.floor(v)
            ok = (zv > 0) & (uf >= 0) & (uf < w) & (vf >= 0) & (vf < h)
            ui = np.clip(np.nan_to_num(uf), 0, w - 1).astype(np.int64)
            vi = np.clip(np.nan_to_num(vf), 0, h - 1).astype(np.int64)
            d_at = np.where(ok, zf[vi, ui], 0.0)
            ok &= d_at > 0
            sdf = d_at - zv
            ok &= sdf > -trunc_cm
            if not ok.any():
                continue
            lin = (gx[ok] * dims[1] + gy[ok]) * dims[2] + gz[ok]
            old_w = flat_w[lin]
            new_w = old_w + 1.0
            flat_t[lin] = (flat_t[lin] * old_w + np.clip(sdf[ok], -trunc_cm, trunc_cm)) / new_w
            flat_w[lin] = new_w
            if flat_c is not None:
                flat_c[lin] = (flat_c[lin] * old_w[:, None] + shade[vi[ok] * w + ui[ok]]) / new_w[:, None]

    return _tsdf_surface(tsdf, weight, color, origin, voxel) + (seen, voxel)


class PointCloudFuse(NodeDef):
    id = "pointcloud_fuse"
    version = 2  # 结果变化时递增，work/ 中的旧结果随之不再命中缓存（engine/cook.py）；2：「画面」输入给点云上色
    keeps_overscan = False  # TSDF 反投影只在画面框内：框外像素无三维
    category = "geometry_tools"
    cost = Cost(whole=True)  # node.pointcloud_fuse.cost.whole
    # 「逐帧点云」与「逐帧深度」二选一（input_choice：至少接一路，另一路自动灰显）。接点云走体素拼接；
    # 接深度走 TSDF，此时必须接「相机」（camera 口按 Wired("depth") 只在接深度时可用，cook 里再校验一次）。
    inputs = (
        Port("points", "scene.points", optional=True),
        Port("depth", "image.1", optional=True, means=("scale",),
             expects=(OwnCamera(), NotDisparity())),
        Port("camera", "scene.camera", optional=True, applies=Wired("depth"), expects=(SameShot("depth"),)),
        Port("image", "image.3", optional=True, data=False, applies=Wired("depth"), expects=(SameShot("depth"),)),
    )
    input_choice = (("points",), ("depth",))
    outputs = (Port("points", "scene.points"),)
    main = "points"

    class Params(NodeParams):
        voxel_size: float = P(0.5, unit="cm", gt=0, le=100, group="blend")
        # TSDF truncation, in voxels: only voxels this close to the surface count, the rest are unobserved
        truncation: float = P(2.0, gt=0, le=20, group="blend", applies=Wired("depth"))
        color: bool = P(True, group="blend")

    @classmethod
    def cook(cls, ctx):
        points = ctx.input("points")
        depth = ctx.input("depth")
        if points is not None:
            return cls._voxel(ctx, points)
        if depth is None:
            raise Invalid(Msg("E-FUSE-NOINPUT"))
        return cls._tsdf(ctx, depth)

    @classmethod
    def _voxel(cls, ctx, points_in):
        from pxr import Usd

        from ...data.evaluate import scene_points
        from ...data.payloads import points_packet
        from ...data.scene import open_scene

        frames = points_in.meta["frames"]
        stage = open_scene([points_in])
        ctx.stage("merge_frame_points")
        pts, cols = [], []
        for f in ctx.each(frames):
            for _, p, _, c in scene_points(stage, Usd.TimeCode(f)):
                if len(p):
                    pts.append(np.asarray(p, np.float64))
                    cols.append(np.asarray(c, np.float32) if c is not None else None)
        if not pts:
            raise Invalid(Msg("E-FUSE-NOPOINTS"))
        xyz = np.concatenate(pts)
        keep_color = ctx.params["color"] and all(c is not None and len(c) == len(p) for c, p in zip(cols, pts))
        rgb = np.concatenate([c for c in cols if c is not None]).astype(np.float32) if keep_color else None
        voxel = float(ctx.params["voxel_size"])
        centroid, color = _voxelize(xyz, rgb, voxel)
        ctx.say("I-FUSE-RESULT", method=i18n.Word("fuse.voxels"), before=int(len(xyz)), after=int(len(centroid)))
        return {"points": points_packet(ctx.outputs["points"], [], "points", [centroid],
                                        [color] if color is not None else None,
                                        scale=points_in.meta.get("scale", "relative"),
                                        width_cm=max(voxel, 0.2))}

    @classmethod
    def _tsdf(cls, ctx, depth):
        from ...data.maps import map_at
        from ...data.payloads import display_rgb, file_at, points_packet

        if ctx.input("camera") is None:
            raise Invalid(Msg("E-FUSE-NOCAMERA"))
        depth, camera = _depth_and_camera(ctx)  # 视差 / 差偏移的深度无法反投影，在这里拒绝
        voxel = float(ctx.params["voxel_size"])
        trunc_cm = float(ctx.params["truncation"]) * voxel
        frames, w, h = depth.meta["frames"], depth.meta["width"], depth.meta["height"]
        focal, mats, pp = _view(camera, frames, w, True)

        ctx.stage("fuse_tsdf")

        def read(f):
            z, alpha = map_at(depth, f)
            return z[..., 0].astype(np.float32), alpha

        image = ctx.input("image")

        def paint(i, f):  # 画面缺这一帧时按远近着色（与「深度转点云」一致）
            return display_rgb(image, f, w, h) if image is not None and file_at(image, f) is not None else None

        surf, col, seen, used = _tsdf_fuse(read, frames, focal, mats, pp, w, h, voxel, trunc_cm,
                                           bool(ctx.params["color"]), paint)
        if used > voxel:
            ctx.say("W-FUSE-VOXELUP", asked=round(voxel, 2), used=round(used, 2))
            voxel = used
        if not len(surf):
            raise Invalid(Msg("E-FUSE-NOSURFACE"))
        ctx.say("I-FUSE-RESULT", method="TSDF", before=seen, after=int(len(surf)))
        return {"points": points_packet(ctx.outputs["points"], [], "points", [surf],
                                        [col] if col is not None else None,
                                        scale=meant(ctx, depth, "scale", ctx.node_type.port_label("depth")),
                                        width_cm=max(voxel, 0.2))}


NODES = (WorldPosition, DepthToPoints, DepthNormal, NormalSpace, TracksToPoints, DepthAlign, PointCloudFuse)
