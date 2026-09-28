"""相机：创建相机、相机去抖、锁定 Focal Length、相机对比。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...messages import Msg
from ..base import Info, NodeDef, NodeParams, P, Port, empty_packet
from ...errors import Invalid
from ..handles import Places
from ..lens import filmback_param, focal_param
from ..kit.align import align_paths
from ..kit.deshake import CUTOFF_DEFAULT, CUTOFF_MAX, CUTOFF_MIN
from ...data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH, FILMBACK_MM, PERCENT
from ...availability import Not
from ..applies import Wired


class CreateCamera(NodeDef):
    id = "core.create_camera"
    lens = "given"  # the lens comes from parameter values; a wired image supplies only its size
    on_node = ("focal_mm", "filmback_mm")
    category = "camera_tools"
    # Focal Length 与 Filmback 为可接线的数值参数，相机为输出
    inputs = (Port("image", "image.3", "RGB", optional=True),)
    outputs = (Port("camera", "scene.camera", "相机"),)
    handles = (Places(translate="translate", rotate="rotate"),)  # viewport gizmo for the camera placement
    # 与镜头标定节点（AnyCalib）的四个数值输出一一对应的常驻接线口，节点创建后即可连接
    wired_ports = ("focal_mm", "filmback_mm", "center_x_mm", "center_y_mm")

    class Params(NodeParams):
        focal_mm: float | None = focal_param()
        filmback_mm: float | None = filmback_param()
        center_x_mm: float = P(0.0, label="主点 X", unit="mm", group="镜头")
        center_y_mm: float = P(0.0, label="主点 Y", unit="mm", group="镜头")
        translate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="位置", unit="cm", widget="vec3", group="相机")
        rotate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="旋转", unit="°", widget="vec3", group="相机")
        width: int = P(DEFAULT_WIDTH, label="画面宽度", unit="px", gt=0, group="画面", applies=Not(Wired("image")))
        height: int = P(DEFAULT_HEIGHT, label="画面高度", unit="px", gt=0, group="画面", applies=Not(Wired("image")))

    @classmethod
    def info(cls, params, inputs):
        """Frames and size of the wired image; without one, no frames and the size set in the parameters."""
        return Info.merge(inputs["image"]) if inputs.get("image") else Info((), params["width"], params["height"])

    @classmethod
    def cook(cls, ctx):
        from ...data.camera import CameraSamples
        from ..lens import lens

        image, p = ctx.input("image"), ctx.params
        if image is not None:
            frames, w, h = image.meta["frames"], image.meta["width"], image.meta["height"]
        else:
            frames, w, h = [], p["width"], p["height"]
        used = lens(ctx, w, frames, port="image")
        if used.focal_px is None:  # no focal length available: output no camera; downstream nodes run without one
            ctx.say("W-CAMERA-NOFOCAL", param="focal_mm")
            return {"camera": empty_packet(ctx, "camera")}
        pose = cls.places.matrix(p)  # the declared placement, also previewed by the viewer while the handle is dragged
        # 主点：参数为相对画面中心的偏移（毫米，向上为正），相机中存储像素位置
        per_mm = w / (used.filmback_mm or FILMBACK_MM)
        principal = (w / 2 + p["center_x_mm"] * per_mm, h / 2 - p["center_y_mm"] * per_mm)
        # 接入画面时将其作为该相机的背板（即 DCC 中相机的图像平面），三维视图通过该相机观察时显示此画面
        samples = CameraSamples.solved(frames, w, h, used.focal_px, pose, filmback_mm=used.filmback_mm,
                                       principal_px=principal, info={"lens": used.said}, plate=image.fingerprint if image is not None else "")
        return {"camera": samples.write(ctx.outputs["camera"])}


class DeshakeCamera(NodeDef):
    id = "core.deshake_camera"
    on_node = ("strength", "keep_sudden")
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera", "相机"),)
    outputs = (Port("camera", "scene.camera", "相机"), Port("curves", "curves", "前后对比"))

    class Params(NodeParams):
        strength: float = P(CUTOFF_DEFAULT, label="强度", unit="帧", group="去抖", widget="slider", ge=CUTOFF_MIN, le=CUTOFF_MAX)
        keep_sudden: bool = P(True, label="保留急停", group="去抖")
        rotation: bool = P(True, label="平滑转动", group="去抖")
        focal: Literal["keep", "smooth"] = P(
            "keep", label="Focal Length", group="去抖", option_labels={"keep": "原样", "smooth": "一起去抖"},
        )

    @classmethod
    def cook(cls, ctx):
        from ..kit.deshake import deshake_camera
        from ..kit.cameras import pass_camera

        p, camera = ctx.params, ctx.input("camera")
        frames = [int(f) for f in camera.meta["frames"]]
        if len(frames) < 3:  # with two poses or fewer, a second difference cannot separate shake from real motion
            ctx.say("N-CAMERA-ONEPOSE")
            return {"camera": pass_camera(ctx, camera), "curves": empty_packet(ctx, "curves")}
        out, report = deshake_camera(camera, ctx.outputs["camera"], p["strength"], p["keep_sudden"], p["rotation"], p["focal"])
        ctx.say("I-DESHAKE-RESULT", moved=report["moved_cm"], turned=report["turned_deg"],
                distance=report["distance_cm"], shift=report["shift_px"])
        if report["shift_px"] > OFF_PLATE_PX:
            ctx.say("W-DESHAKE-OFFPLATE", shift=report["shift_px"], frame=report["shift_frame"],
                    distance=report["distance_cm"], param="strength")
        if report["sudden"]:
            ctx.say("N-DESHAKE-SUDDEN", count=len(report["sudden"]), frames=report["sudden"][:SUDDEN_LISTED],
                    param="keep_sudden")
        return {"camera": out, "curves": before_after(ctx.outputs["curves"], frames, camera_pairs(report))}


OFF_PLATE_PX = 2.0  # plate shift (px) at or above which the removed shake may have been genuine hand-held motion


SUDDEN_LISTED = 8  # maximum number of preserved sudden-motion frames named in the message


ORIGINAL = "原始"  # suffix of the input ("before") channel in a before/after curves packet


def before_after(out, frames: list[int], pairs: list[tuple[str, np.ndarray, np.ndarray | None]]):
    """Build a before/after curves packet. Each item of `pairs` is (channel name, output values, input values), where
    the input values are None for channels without a "before" (such as residuals). The packet's `before` map pairs
    each channel with its original, which is sufficient for the curve editor to draw them together; both 「相机去抖」
    and 「锁定 Focal Length」 use this function, and the editor has no knowledge of either node."""
    from ...data.payloads import curves_packet

    names, columns, before = [], [], {}
    for label, after, original in pairs:
        names.append(label)
        columns.append(np.asarray(after, np.float64).reshape(-1))
        if original is None:
            continue
        names.append(f"{label} {ORIGINAL}")
        columns.append(np.asarray(original, np.float64).reshape(-1))
        before[label] = f"{label} {ORIGINAL}"
    return curves_packet(out, frames, names, np.stack(columns, axis=1), before=before)


def camera_pairs(report: dict) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Return the six before/after channels of a camera tool: translation (cm) and rotation (degrees) per axis."""
    axes = (("translate", 0, "位置 X"), ("translate", 1, "位置 Y"), ("translate", 2, "位置 Z"),
            ("rotate", 0, "旋转 X"), ("rotate", 1, "旋转 Y"), ("rotate", 2, "旋转 Z"))
    return [(label, np.asarray(report["after"][key])[:, axis], np.asarray(report["before"][key])[:, axis])
            for key, axis, label in axes]


class LockFocal(NodeDef):
    id = "core.lock_focal"
    on_node = ("focal_mm",)
    category = "camera_tools"
    inputs = (
        Port("camera", "scene.camera", "相机"),
        # 必需：缺少场景点到相机的距离则无法求出锁定 Focal Length 后的相机位置。未接线时由 B-GRAPH-NOWIRE 在提交前拦截
        Port("points", "scene.points", "点云",
             recommend="core.depth_points",
             help="拿来当锚点的三维点：解算节点的「点云」输出，或者「3D 跟踪点」"),
        Port("tracks", "tracks2d", "2D 跟踪点", optional=True,
             help="和「点云」出自同一个节点、一一对应的画面观测。接上就用真实观测代替「按原相机投出来的位置」"),
    )
    outputs = (Port("camera", "scene.camera", "相机"), Port("curves", "curves", "前后对比"))
    wired_ports = ("focal_mm", "filmback_mm")

    class Params(NodeParams):
        # 留空时锁定到解算 Focal Length 的整段中值（nodes/lens.py lens()：未填写时取输入相机的值，即中位数）；
        # 填写或接线时锁定到该值。与其他读取镜头的节点规则一致，不另设「来源」选项
        focal_mm: float | None = focal_param(overrides=("camera",))
        filmback_mm: float | None = filmback_param(overrides=("camera",))

    @classmethod
    def cook(cls, ctx):
        from ..kit.lock_focal import lock_focal
        from ..lens import lens

        from ..kit.cameras import pass_camera

        camera, points, tracks = ctx.input("camera"), ctx.input("points"), ctx.input("tracks")
        frames = [int(f) for f in camera.meta["frames"]]
        w = int(camera.meta.get("width") or DEFAULT_WIDTH)
        if not frames:  # a static camera without frames already has a single focal length and no poses to re-solve
            ctx.say("N-FOCAL-NOFRAMES")
            return {"camera": pass_camera(ctx, camera), "curves": empty_packet(ctx, "curves")}
        used = lens(ctx, w, frames)
        if not used.given:
            ctx.say("W-CAMERA-NOFOCAL", param="focal_mm")
            return {"camera": empty_packet(ctx, "camera"), "curves": empty_packet(ctx, "curves")}
        out, report = lock_focal(camera, points, tracks, ctx.outputs["camera"], used.focal_px,
                                 info={"focal_mm": used.focal_mm, "source": used.said})
        low, high = report["range_mm"]
        ctx.say("I-FOCAL-RESULT", low=low, high=high, locked=report["focal_mm_after"], said=used.said,
                moved=report["moved_cm"], median=report["residual_median"], max=report["residual_max"])
        if report["measured"]:
            ctx.say("I-FOCAL-MEASURED", count=report["measured"])
        if report["zoomlike"]:
            ctx.say("N-FOCAL-ZOOMLIKE", low=low, high=high)
        if report["few"] or report["empty"]:
            ctx.say("N-FOCAL-FEWANCHORS", count=len(report["few"]) + len(report["empty"]),
                    least=FEW_ANCHORS, port="points")
        if report["residual_median"] > REPROJ_MEDIAN_PX or report["residual_max"] > REPROJ_MAX_PX:
            ctx.say("W-FOCAL-REPROJ", low=low, high=high, locked=report["focal_mm_after"],
                    median=report["residual_median"], max=report["residual_max"], frame=report["worst_frame"])
        pairs = [("Focal Length", np.full(len(frames), report["focal_mm_after"]), report["focal_mm_before"]),
                 ("沿镜头方向", report["along_cm"], None), ("重投影残差", report["residual_px"], None)]
        return {"camera": out, "curves": before_after(ctx.outputs["curves"], frames, pairs)}


FEW_ANCHORS = 30  # the threshold of nodes/kit/lock_focal.py FEW_ANCHORS, reported in the message
REPROJ_MEDIAN_PX, REPROJ_MAX_PX = 1.0, 4.0  # above either, repositioning the camera could not compensate for the lock


ALIGN_LABELS = {"similarity": "比例+旋转+位置", "rigid": "旋转+位置", "none": "不对齐"}


class CompareCameras(NodeDef):
    id = "core.compare_cameras"
    on_node = ("align",)
    version = 2  # a collinear path is aligned using the cameras' orientations
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera", "相机"), Port("reference", "scene.camera", "参考相机"))
    outputs = (Port("camera", "scene.camera", "对齐后的相机"), Port("errors", "curves", "逐帧误差"))

    class Params(NodeParams):
        align: Literal["similarity", "rigid", "none"] = P(
            "similarity", label="对齐", group="对比", option_labels=ALIGN_LABELS,
        )

    @classmethod
    def cook(cls, ctx):
        from lab2shot_worker.recon import rotation_deg

        from ...data.camera import CameraSamples
        from ...data.payloads import curves_packet
        from ...data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH

        camera, reference = ctx.input("camera"), ctx.input("reference")
        frames = sorted(set(camera.meta["frames"]) & set(reference.meta["frames"]))
        if not frames:
            from ...io.sequence import format_frame_range

            raise Invalid(Msg("E-COMPARE-NOCOMMON", camera=format_frame_range(camera.meta["frames"]),
                              reference=format_frame_range(reference.meta["frames"])))
        if len(frames) < 2:
            raise Invalid(Msg("E-COMPARE-ONECOMMON", frame=frames[0]))
        a, b = CameraSamples.from_packet(camera, frames), CameraSamples.from_packet(reference, frames)
        pa, pb = a.cam_to_world[:, :3, 3], b.cam_to_world[:, :3, 3]
        ra, rb = a.cam_to_world[:, :3, :3], b.cam_to_world[:, :3, :3]
        path = float(np.linalg.norm(np.diff(pb, axis=0), axis=1).sum())
        mode = ctx.params["align"]
        s, R, t = 1.0, np.eye(3), np.zeros(3)
        if mode != "none":
            s, R, t, note = align_paths(pa, pb, ra, rb)
            if note:
                ctx.say(note)
            if mode == "rigid":
                s = 1.0
                t = pb.mean(0) - R @ pa.mean(0)
        aligned = np.repeat(np.eye(4)[None], len(frames), 0)
        aligned[:, :3, :3] = R @ ra
        aligned[:, :3, 3] = s * pa @ R.T + t
        pos_err = np.linalg.norm(aligned[:, :3, 3] - pb, axis=1)
        rot_err = rotation_deg(aligned[:, :3, :3] @ np.swapaxes(rb, 1, 2))
        focal_a = a.focal_mm / a.h_aperture_mm  # focal length over filmback, independent of resolution
        focal_b = b.focal_mm / b.h_aperture_mm
        focal_err = (focal_a / focal_b - 1.0) * PERCENT

        rms = float(np.sqrt(np.mean(pos_err**2)))
        share = Msg("I-COMPARE-SHARE", share=rms / path * PERCENT) if path > 1.0 else ""
        scale = Msg("I-COMPARE-SCALE", scale=s) if mode == "similarity" and s != 1.0 else ""
        ctx.say("I-COMPARE-RESULT", frames=len(frames), rms=rms, share=share, max=float(pos_err.max()),
                angle=float(np.median(rot_err)), angle_max=float(rot_err.max()), focal=f"{np.median(focal_err):+.1f}", scale=scale)
        w, h = int(camera.meta.get("width") or DEFAULT_WIDTH), int(camera.meta.get("height") or DEFAULT_HEIGHT)
        summary = {"frames": len(frames), "align": mode, "scale": s, "position_rms_cm": rms, "path_cm": path,
                   "rotation_median_deg": float(np.median(rot_err)), "rotation_max_deg": float(rot_err.max()),
                   "focal_median_pct": float(np.median(focal_err))}
        solved = CameraSamples.solved(frames, w, h, focal_a * w, aligned,
                                      filmback_mm=float(np.median(a.h_aperture_mm)), info={"aligned_to_reference": summary})
        return {
            "camera": solved.write(ctx.outputs["camera"]),
            "errors": curves_packet(ctx.outputs["errors"], frames,
                                    ["position_error_cm", "rotation_error_deg", "focal_error_pct"],
                                    np.stack([pos_err, rot_err, focal_err], axis=1), summary=summary),
        }




NODES = (CreateCamera, DeshakeCamera, LockFocal, CompareCameras)
