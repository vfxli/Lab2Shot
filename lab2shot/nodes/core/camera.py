"""相机：创建相机、设置背板、相机去抖、锁定 Focal Length、相机对比。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..kit.ports import rgb_port
from ... import i18n
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
    id = "camera"
    version = 2  # an unknown focal length gives no camera without a warning
    same_on_cards = True  # 各卡上公开的这块参数一样（NodeDef.same_on_cards）
    lens = "given"  # the lens comes from parameter values; a wired image supplies only its size
    on_node = ("focal_mm", "filmback_mm")
    category = "camera_tools"
    # Focal Length 与 Filmback 为可接线的数值参数，相机为输出
    inputs = (rgb_port(optional=True),)
    outputs = (Port("camera", "scene.camera", may_be_empty=True),)
    handles = (Places(translate="translate", rotate="rotate"),)  # viewport gizmo for the camera placement
    # 与镜头标定节点（AnyCalib）的四个数值输出一一对应的常驻接线口，节点创建后即可连接
    wired_ports = ("focal_mm", "filmback_mm", "center_x_mm", "center_y_mm")

    class Params(NodeParams):
        focal_mm: float | None = focal_param()
        filmback_mm: float | None = filmback_param()
        center_x_mm: float = P(0.0, unit="mm", group="lens")
        center_y_mm: float = P(0.0, unit="mm", group="lens")
        translate: tuple[float, float, float] = P((0.0, 0.0, 0.0), unit="cm", widget="vec3", group="camera")
        rotate: tuple[float, float, float] = P((0.0, 0.0, 0.0), unit="°", widget="vec3", group="camera")
        width: int = P(DEFAULT_WIDTH, unit="px", gt=0, group="image", applies=Not(Wired("image")))
        height: int = P(DEFAULT_HEIGHT, unit="px", gt=0, group="image", applies=Not(Wired("image")))

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
        # no focal length: no camera, as if unwired. The port may be empty (may_be_empty), so its empty packet is no news
        # downstream (engine/cook.py expected): what reads it directly goes without or quietly gives nothing
        if used.focal_px is None:
            return {"camera": empty_packet(ctx, "camera")}
        pose = cls.places.matrix(p)  # the declared placement, also previewed by the viewer while the handle is dragged
        # 主点：参数为相对画面中心的偏移（毫米，向上为正），相机中存储像素位置
        per_mm = w / (used.filmback_mm or FILMBACK_MM)
        principal = (w / 2 + p["center_x_mm"] * per_mm, h / 2 - p["center_y_mm"] * per_mm)
        # 接入画面时将其作为该相机的背板（即 DCC 中相机的图像平面），三维视图通过该相机观察时显示此画面
        samples = CameraSamples.solved(frames, w, h, used.focal_px, pose, filmback_mm=used.filmback_mm,
                                       principal_px=principal, info={"lens": used.said}, plate=image.fingerprint if image is not None else "")
        return {"camera": samples.write(ctx.outputs["camera"])}



class SetPlate(NodeDef):
    """Give a camera a plate: the same camera out, its plate (the picture the 3D view shows through it, as in a DCC's
    camera image plane) now the wired image. Any camera (imported from abc / USD / FBX, solved, created) and any
    image (colour, grey, with alpha). The camera's file is copied unchanged apart from that one attribute, so an
    imported camera stays exactly what the user brought in. The plate is looked up by frame number, so the camera's
    and the image's frames should agree."""

    id = "image_plane"
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera"), Port("image", "image", alpha=True, data=False))
    outputs = (Port("camera", "scene.camera"),)

    class Params(NodeParams):
        pass

    @classmethod
    def info(cls, params, inputs):
        """The camera's frames and size: the image only lends its picture."""
        return Info.merge(inputs["camera"])

    @classmethod
    def cook(cls, ctx):
        import shutil

        from pxr import Usd

        from ...data.packet import Packet
        from ...data.payloads import SCENE_FILE
        from ...data.scene import the_camera
        from ...io import usd

        camera, image = ctx.input("camera"), ctx.input("image")
        target = ctx.outputs["camera"] / SCENE_FILE
        shutil.copyfile(camera.path(SCENE_FILE), target)
        stage = Usd.Stage.Open(str(target))
        the_camera(stage, i18n.t("scene.where.camera_input")).SetCustomDataByKey(usd.PLATE, image.fingerprint)
        stage.GetRootLayer().Save()
        return {"camera": Packet(ctx.outputs["camera"], camera.type, {**camera.meta, "plate": image.fingerprint})}


class DeshakeCamera(NodeDef):
    id = "deshake_camera"
    on_node = ("strength", "keep_sudden")
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera"),)
    outputs = (Port("camera", "scene.camera"), Port("curves", "curves", may_be_empty=True))

    class Params(NodeParams):
        strength: float = P(CUTOFF_DEFAULT, unit="frame", group="stabilize", widget="slider", ge=CUTOFF_MIN, le=CUTOFF_MAX)
        keep_sudden: bool = P(True, group="stabilize")
        rotation: bool = P(True, group="stabilize")
        focal: Literal["keep", "smooth"] = P(
            "keep", group="stabilize",
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


ORIGINAL = "original"  # suffix of the input ("before") channel in a before/after curves packet (a data name)


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
        names.append(f"{label}_{ORIGINAL}")
        columns.append(np.asarray(original, np.float64).reshape(-1))
        before[label] = f"{label}_{ORIGINAL}"
    return curves_packet(out, frames, names, np.stack(columns, axis=1), before=before)


def camera_pairs(report: dict) -> list[tuple[str, np.ndarray, np.ndarray | None]]:
    """Return the six before/after channels of a camera tool: translation (cm) and rotation (degrees) per axis."""
    axes = (("translate", 0, "translate_x"), ("translate", 1, "translate_y"), ("translate", 2, "translate_z"),
            ("rotate", 0, "rotate_x"), ("rotate", 1, "rotate_y"), ("rotate", 2, "rotate_z"))  # curve names: data
    return [(label, np.asarray(report["after"][key])[:, axis], np.asarray(report["before"][key])[:, axis])
            for key, axis, label in axes]


class LockFocal(NodeDef):
    id = "lock_focal"
    on_node = ("focal_mm",)
    category = "camera_tools"
    inputs = (
        Port("camera", "scene.camera"),
        # 必需：缺少场景点到相机的距离则无法求出锁定 Focal Length 后的相机位置。未接线时由 B-GRAPH-NOWIRE 在提交前拦截
        Port("points", "scene.points",
             recommend="points_from_depth"),
        Port("tracks", "tracks2d", optional=True),
    )
    outputs = (Port("camera", "scene.camera"), Port("curves", "curves", may_be_empty=True))
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
        used = lens(ctx, w, frames)  # the wired camera always has a focal length, so one is always known
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
        pairs = [("focal_length", np.full(len(frames), report["focal_mm_after"]), report["focal_mm_before"]),
                 ("along_lens", report["along_cm"], None), ("reprojection_residual", report["residual_px"], None)]
        return {"camera": out, "curves": before_after(ctx.outputs["curves"], frames, pairs)}


FEW_ANCHORS = 30  # the threshold of nodes/kit/lock_focal.py FEW_ANCHORS, reported in the message
REPROJ_MEDIAN_PX, REPROJ_MAX_PX = 1.0, 4.0  # above either, repositioning the camera could not compensate for the lock




class CompareCameras(NodeDef):
    id = "compare_cameras"
    on_node = ("align",)
    version = 2  # a collinear path is aligned using the cameras' orientations
    category = "camera_tools"
    inputs = (Port("camera", "scene.camera"), Port("reference", "scene.camera"))
    outputs = (Port("camera", "scene.camera"), Port("errors", "curves"))

    class Params(NodeParams):
        align: Literal["similarity", "rigid", "none"] = P(
            "similarity", group="compare",
        )

    @classmethod
    def cook(cls, ctx):
        from lab2shot_shared.poses import rotation_deg

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




NODES = (CreateCamera, SetPlate, DeshakeCamera, LockFocal, CompareCameras)
