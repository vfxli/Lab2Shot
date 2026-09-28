"""A camera's samples over a shot: the single way to read a camera (from a packet or a USD prim), build one from a
solved result, and write it out.

`frames=()` means a single, non-animated sample (「创建相机」 without a picture, a locked-off camera in a family that
has no plate of its own): from_packet/from_prim then give every array exactly one row. A CameraSamples built by
solved() keeps its cam_to_world exactly as given (a single [4,4], or one per frame, or None: no position, at the
origin looking down -Z) instead of broadcasting it: write() passes it directly to usd.write_camera, whose own check
distinguishes a constant transform from a per-frame one. Because that is the only place the decision is made, a
locked-off camera with no plate has no transform authored at all (rather than an identity one), and a genuine
per-frame solve is keyed on every frame even where neighbouring frames coincide.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..errors import Invalid
from ..messages import Msg
from . import units

if TYPE_CHECKING:
    from .packet import Packet

# The parts of a lens description a camera keeps: the distortion, the raster it was defined on and its source.
# The focal length, film back, pixel aspect and lens centre belong to the camera itself and are not duplicated.
CARRIED_LENS = ("distortion", "raster", "source")
LENS = "lab2shot:lens:"  # prefix of the named properties storing a camera's lens; identical in every file format
PARAMS = LENS + "params:"


def carried_lens(meta) -> dict:
    """The part of a lens description (as a picture carries it) a camera takes with it: CARRIED_LENS."""
    return {k: meta[k] for k in CARRIED_LENS if (meta or {}).get(k)}


@dataclass(frozen=True)
class CameraSamples:
    """A camera's samples over some frames, in Lab2Shot's conventions: centimetres, Y up, GL camera axes (looks down
    -Z), column-vector matrices. Always a pinhole on a definite picture: a distortion is carried in `lens` for
    consumers that reapply it (compositing, 3DE) and does not affect how the camera projects."""

    frames: tuple[int, ...]
    cam_to_world: np.ndarray | None  # [F,4,4] (a solved() camera: a single [4,4], or None: at the origin, -Z)
    focal_mm: np.ndarray  # [F] (or [1])
    h_aperture_mm: np.ndarray  # [F] (or [1])
    v_aperture_mm: np.ndarray  # [F] (or [1])
    width: int = 0  # 0: the picture size is not known
    height: int = 0
    info: dict = field(default_factory=dict)  # where it came from: imported_from, object, extension, lens ...
    center_mm: np.ndarray = field(default_factory=lambda: np.zeros((1, 2)))  # [F,2] (or [1,2]): the lens centre off the picture's centre, mm, +x right +y up
    pixel_aspect: float = 1.0  # a pixel's width over its height (2.0: a 2x anamorphic squeeze)
    overscan: tuple[int, int, int, int] = (0, 0, 0, 0)  # pixels past the picture: left, top, right, bottom (the one window of data/windows.py, as `window` gives it)
    lens: dict = field(default_factory=dict)  # the lens description: distortion {model, params, focus_cm?}, raster [W,H], source; {}: none described
    # the picture this camera belongs to (the packet fingerprint of what it was solved on, or one assigned manually):
    # its plate, an attribute of the camera as in every DCC (io/usd.py PLATE). "": none recorded, no plate shown in 3D
    plate: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "center_mm", np.asarray(self.center_mm, np.float64).reshape(-1, 2))
        object.__setattr__(self, "overscan", tuple(int(v) for v in self.overscan))
        if not (self.pixel_aspect > 0 and np.isfinite(self.center_mm).all()):
            raise Invalid(Msg("E-LENS-PHYSICAL", what="pixel_aspect" if not self.pixel_aspect > 0 else "center_mm",
                              value=float(self.pixel_aspect) if not self.pixel_aspect > 0 else "NaN"))
        if len(self.overscan) != 4 or min(self.overscan) < 0 or (any(self.overscan) and not (self.width and self.height)):
            raise Invalid(Msg("E-CAMERA-OVERSCAN", value=list(self.overscan)))
        if self.lens:
            from .lens_models import Lens

            Lens.from_meta(self.lens_meta(), (self.width, self.height) if self.width and self.height else None)  # rejects an invalid lens description

    @property
    def window(self):
        """The picture this camera belongs to as one window (data/windows.py Window): its plate frame and the canvas
        its overscan forms around it. A camera and the picture it was solved on describe it identically."""
        from .windows import Window

        left, top, right, bottom = self.overscan
        return Window.canvas_of((self.width, self.height), (self.width + left + right, self.height + top + bottom), (left, top))

    @property
    def filmback_mm(self) -> float:
        return float(np.median(self.h_aperture_mm))

    def focal_px(self, width: int | None = None) -> np.ndarray:
        """Per frame at `width` (or the camera's own picture width): an animated aperture keeps its field of view."""
        return units.focal_px(self.focal_mm, self.h_aperture_mm, width if width is not None else self.width)

    def principal_px(self, width: int | None = None) -> np.ndarray:
        """Per frame [F,2]: the principal point (cx, cy) in pixels at `width` (the picture scaled to it), +y down — the
        picture's centre moved by the lens centre (center_mm, +y up). A solver that wrote its principal point (COLMAP,
        MapAnything, VGGT …) is unprojected through it; one that did not is at the centre."""
        w = float(width if width is not None else self.width)
        h = float(self.height) * w / float(self.width) if self.width else 0.0
        n = max(len(self.frames), 1)
        per_mm = np.broadcast_to(w / np.asarray(self.h_aperture_mm, np.float64), (n,))
        c = np.broadcast_to(np.asarray(self.center_mm, np.float64).reshape(-1, 2), (n, 2))
        return np.stack([w / 2 + c[:, 0] * per_mm, h / 2 - c[:, 1] * per_mm], -1)

    def rotations(self) -> np.ndarray:
        """The rotation part of cam_to_world, its columns made unit length."""
        return units.rotations(self.poses())

    def is_still(self, tolerance_cm: float = 0.05) -> bool:
        """Whether the camera is static: position within `tolerance_cm`, orientation within 0.01 degree."""
        m = self.poses()
        moved = np.abs(m[:, :3, 3] - m[:1, :3, 3]).max(initial=0.0)
        turned = np.abs(m[:, :3, :3] - m[:1, :3, :3]).max(initial=0.0)
        return bool(moved <= tolerance_cm and turned <= np.radians(0.01))

    def opencv_m(self) -> np.ndarray:
        """cam_to_world the way a worker reads it (lab2shot_worker.recon.load_camera): OpenCV camera axes, metres."""
        return units.usd_poses_to_opencv_m(self.poses())

    def at(self, frames) -> CameraSamples:
        """The samples of only these frames (each must already be one of self.frames)."""
        wanted = [int(f) for f in frames]
        index = {f: i for i, f in enumerate(self.frames)}
        try:
            rows = [index[f] for f in wanted]
        except KeyError as exc:
            raise Invalid(Msg("E-CAMERA-NOFRAME", frame=exc.args[0])) from None
        def at_rows(values: np.ndarray) -> np.ndarray:
            """One value per wanted frame; a value given once applies to every frame (the class's own rule: [F] or
            [1])."""
            return values[rows] if len(values) > 1 else values

        return replace(self, frames=tuple(wanted), cam_to_world=self.poses()[rows], focal_mm=at_rows(self.focal_mm),
                        h_aperture_mm=at_rows(self.h_aperture_mm), v_aperture_mm=at_rows(self.v_aperture_mm),
                        center_mm=at_rows(self.center_mm))

    # ------------------------------------------------------------------ the lens

    def lens_meta(self) -> dict:
        """The lens description as lens_models.Lens.from_meta reads it: this camera's focal length, film back,
        centre and pixel aspect with the distortion, raster and source it carries."""
        focal = np.asarray(self.focal_mm, np.float64).reshape(-1)
        animated = len(focal) > 1 and np.ptp(focal) > 0
        return {"focal_mm": {"frames": list(self.frames), "values": focal.tolist()} if animated else float(focal[0]),
                "filmback_mm": [float(np.median(self.h_aperture_mm)), float(np.median(self.v_aperture_mm))],
                "pixel_aspect": float(self.pixel_aspect), "center_mm": np.median(self.center_mm, axis=0).tolist(), **self.lens}

    def lens_properties(self) -> dict:
        """The lens as the named properties every file format keeps it under: texts, numbers,
        and a parameter that changes over the shot as {frames, values}."""
        out: dict = {}
        if self.lens:
            dist = self.lens["distortion"]
            out[LENS + "model"] = str(dist["model"])
            out[LENS + "source"] = json.dumps(self.lens.get("source") or {}, ensure_ascii=False, sort_keys=True)
            if self.lens.get("raster"):
                out[LENS + "raster"] = json.dumps([int(v) for v in self.lens["raster"]])
            if dist.get("focus_cm") is not None:
                out[LENS + "focusCm"] = float(dist["focus_cm"])
            out.update({PARAMS + name: value for name, value in dist["params"].items()})
        return out

    @staticmethod
    def lens_from_properties(props: dict) -> dict:
        """The lens from a file's named properties (lens_properties written out; none: no lens)."""
        lens: dict = {}
        if props.get(LENS + "model") is not None:
            params = {name[len(PARAMS):]: value if isinstance(value, dict) else float(value)
                      for name, value in props.items() if name.startswith(PARAMS)}
            dist = {"model": str(props[LENS + "model"]), "params": params}
            if props.get(LENS + "focusCm") is not None:
                dist["focus_cm"] = float(props[LENS + "focusCm"])
            lens = {"distortion": dist, "source": json.loads(props.get(LENS + "source") or "{}")}
            if props.get(LENS + "raster"):
                lens["raster"] = json.loads(props[LENS + "raster"])
        return {"lens": lens}

    def poses(self) -> np.ndarray:
        """cam_to_world as one [4,4] per frame (as read by every writer of the camera): identity at every frame when a
        solved() camera was given no position, the same matrix repeated when it was given a single constant one."""
        n = max(1, len(self.frames))
        if self.cam_to_world is None:
            return np.repeat(np.eye(4)[None], n, 0)
        if self.cam_to_world.ndim == 2:
            return np.repeat(self.cam_to_world[None], n, 0)
        return self.cam_to_world

    # ------------------------------------------------------------------ building

    @classmethod
    def from_packet(cls, camera: Packet, frames=None) -> CameraSamples:
        """A camera packet's samples at `frames` (its own frames when not given; a still with none: one sample)."""
        from .scene import open_scene, the_camera

        want = list(frames) if frames is not None else list(camera.meta.get("frames", []))
        stage = open_scene([camera])  # kept alive while the prim below is read
        prim = the_camera(stage, "相机输入")
        width, height = int(camera.meta.get("width") or 0), int(camera.meta.get("height") or 0)
        return cls.from_prim(prim, want, width=width, height=height)

    @classmethod
    def from_prim(cls, prim, frames, *, width: int = 0, height: int = 0, info: dict | None = None) -> CameraSamples:
        """A USD camera prim's samples at `frames` (empty: one, at the prim's default time)."""
        from pxr import Usd, UsdGeom

        from ..io import usd

        times = list(frames) or [0]
        cam = UsdGeom.Camera(prim)
        mats, focal, h_ap, v_ap = [], [], [], []
        for t in times:
            tc = Usd.TimeCode(t)
            mats.append(np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(tc)).T)
            focal.append(cam.GetFocalLengthAttr().Get(tc))
            h_ap.append(cam.GetHorizontalApertureAttr().Get(tc))
            v_ap.append(cam.GetVerticalApertureAttr().Get(tc))
        lens = usd.camera_lens(prim, times, LENS)
        return cls(tuple(frames), np.asarray(mats, np.float64), np.asarray(focal, np.float64),
                   np.asarray(h_ap, np.float64), np.asarray(v_ap, np.float64), width, height, dict(info or {}),
                   lens["center_mm"], lens["pixel_aspect"], lens["overscan"], plate=usd.camera_plate(prim),
                   **cls.lens_from_properties(lens["properties"]))

    @classmethod
    def solved(cls, frames, width: int, height: int, focal_px, cam_to_world: np.ndarray | None = None, *,
               filmback_mm: float = units.FILMBACK_MM, info: dict | None = None, fy_px=None, principal_px=None,
               lens: dict | None = None,
               overscan: tuple[int, int, int, int] = (0, 0, 0, 0), plate: str = "") -> CameraSamples:
        """A solved camera: `focal_px` (pixels at `width`, one value or one per frame) and `cam_to_world` (a single
        [4,4], one [4,4] per frame, or None: no position at all, left at the origin looking down -Z) at `frames`.
        `cam_to_world` is kept exactly as given, not broadcast: write() hands it straight to usd.write_camera, whose
        own ndim check distinguishes a constant transform (a single [4,4]: an un-animated camera, regardless of
        `frames`) from a per-frame one and authors nothing when it is None; that is the only place this is decided. The callers
        (CreateCamera and the other camera nodes, nodes/kit/cameras.py solved_camera) write a solved CameraSamples out
        without reading its cam_to_world again.

        The intrinsics beyond the focal length, pixels at `width` x `height` with centres at +0.5: `fy_px` (None: fx)
        becomes the pixel aspect, f_y / f_x (one for the camera: the median over the frames; f_y = a f_x), as every
        file format keeps it; `principal_px` ((cx, cy), or one per frame; None: the picture's centre) becomes the lens
        centre offset in mm. `lens`: the distortion the solve found.
        `plate`: the picture it was solved on (its packet fingerprint), the camera's plate from then on."""
        n = max(1, len(frames))
        focal_mm = np.broadcast_to(np.atleast_1d(units.focal_mm(focal_px, filmback_mm, width)), (n,))
        aspect = 1.0 if fy_px is None else float(np.median(np.broadcast_to(np.asarray(fy_px, np.float64), (n,))
                                                            / np.broadcast_to(np.asarray(focal_px, np.float64), (n,))))
        h_ap = np.full(n, float(filmback_mm))
        # the film back's height: F_h = F_w H / (W a) -- a pixel is as wide as F_w / W mm and as high as
        # that over the pixel aspect, so an anamorphic squeeze (a = 2) makes the back half as high as the proportions
        # alone would
        v_ap = np.full(n, float(filmback_mm) * height / (width * aspect))
        center = np.zeros((1, 2))
        if principal_px is not None:
            c = np.asarray(principal_px, np.float64).reshape(-1, 2)
            # per pixel, as the file formats keep the apertures (io/usd.py write_camera) and read the lens back
            # through them (lens_meta): mm per pixel across is F_w / W, up it is F_h / H = F_w / (W a)
            mm_x = float(filmback_mm) / width
            mm_y = mm_x / aspect
            center = np.stack([(c[:, 0] - width / 2) * mm_x, -(c[:, 1] - height / 2) * mm_y], -1)
        mats = None if cam_to_world is None else np.asarray(cam_to_world, np.float64)
        return cls(tuple(frames), mats, np.array(focal_mm), h_ap, v_ap, width, height, dict(info or {}),
                   center, aspect, tuple(overscan), lens=dict(lens or {}), plate=str(plate or ""))

    def write(self, out: Path, name: str = "camera", customize=None, path: str = "", source: str = "", **meta) -> Packet:
        """/shot/<name> written out as a scene.camera packet; an imported camera at `path`, where its file
        (`source`) had it (io/usd.py import_path)."""
        from ..io import usd
        from .payloads import SCENE_FILE, scene_packet

        stage = usd.create_stage(list(self.frames), self.info)
        cam = usd.write_camera(stage, name, usd.CameraData(self.width, self.height, self.focal_mm, self.filmback_mm,
                                                            self.cam_to_world, center_mm=self.center_mm,
                                                            pixel_aspect=self.pixel_aspect, overscan=self.overscan,
                                                            properties=self.lens_properties()),
                               list(self.frames), path, source, LENS)
        if self.plate:  # stored on the prim (io/usd.py PLATE) so it survives merging, conversion and pass-through
            cam.GetPrim().SetCustomDataByKey(usd.PLATE, self.plate)
        if customize:
            customize(cam)
        usd.save_stage(stage, out / SCENE_FILE)
        # Facts for the summary (data/summary.py reads meta only): the lens, the picture the camera belongs to, and
        # whether it carries a distortion to reapply.
        facts = {"focal_mm": [float(np.min(self.focal_mm)), float(np.max(self.focal_mm))],
                 "filmback_mm": [self.filmback_mm, float(np.median(self.v_aperture_mm))],
                 "pixel_aspect": self.pixel_aspect,
                 "distortion": (self.lens.get("distortion") or {}).get("model", "") if self.lens else "",  # model id; empty when there is no distortion
                 "plate": self.plate}  # the plate: the picture it was solved on (packet fingerprint; empty when not recorded)
        return scene_packet(out, list(self.frames), "scene.camera", width=self.width, height=self.height,
                            **{**facts, **meta})
