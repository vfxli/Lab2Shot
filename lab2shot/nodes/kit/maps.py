"""Per-frame EXR outputs converted directly from the raw npz files written by workers; shared by several families."""

from __future__ import annotations

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ...data.packet import Packet
from ...data.payloads import ExrWriter, window_of
from ...data.units import M_TO_CM
from ..base import empty_packet
from ..families.base import RawOutput


def turn_to_camera(normals: np.ndarray) -> np.ndarray:
    """A model's normals in the OpenCV camera (X right, Y down, Z away) in ours (X right, Y up, Z towards the camera).
    The one turn: no extension and no other family writes it again."""
    from ...data.units import CV_TO_GL

    return np.asarray(normals, np.float32) * CV_TO_GL.astype(np.float32)


def camera_normals(array: str = "normal", valid: str = "mask"):
    """A frame_maps entry for a model's normals (turn_to_camera), with where it has them when its raw file says."""
    def turned(d, array=array, valid=valid):
        got = turn_to_camera(d[array])
        return (got, d[valid].astype(bool)) if valid in d else got

    return "image.3", turned, {"space": "camera"}, NORMALIZE


def points_params(default_step: int = 4):
    """The parameters of a family's point cloud (点云) output: they have no effect until something is wired to it."""
    from ..applies import WiredOut
    from ..base import P

    return {
        "point_step": P(default_step, label="点云间隔", ge=1, le=64, group="点云", worker=False, applies=WiredOut("points"),
                        help="每隔几个像素取一个点。4 够预览和导出；调小点更密、文件更大（1 = 每个像素都要）"),
        "point_size": P(0.5, label="点的大小", unit="cm", gt=0, le=100, group="点云", worker=False, applies=WiredOut("points"),
                        help="点在视图和 DCC 里画多大（Houdini 里是点的 pscale / width）"),
    }


def native_points_of(node, raw):
    """(frame) -> the camera-space point map computed by this node's worker, present only when the node declares
    `native_points`; otherwise None. One declaration, one path: the family only asks whether the node's worker outputs
    point maps and knows no project names."""
    return raw.maps(node.native_points) if getattr(node, "native_points", "") else None


def family_points(ctx, depth, camera, image=None, mask=None, confidence=None, native=None):
    """The point cloud (点云) a family gives: the model's own 3D points when its node declares them (`native_points`,
    read with `native_points_of`), else the same unprojection 「深度转点云」 performs, never a second implementation
    (what several projects share is an output of the family). Sampling, colouring, exclusion masks, confidence, point
    size and coordinate system follow the same path either way. Computed only when something wants it."""
    from ..core.geometry import points_from_depth

    from ...data.units import M_TO_CM

    sure = confidence if confidence is not None and not confidence.meta.get("empty") else None
    # The 「尺度」 parameter (unit_cm) is the factor this family uses to convert depth and camera; native point maps
    # must use the same factor, otherwise changing the scale would misalign the point cloud with the camera and depth.
    return points_from_depth(ctx, depth, camera, image, mask, ctx.params.get("point_step", 4),
                             ctx.params.get("point_size", 0.5), "world", sure, native=native,
                             scale_cm=float(ctx.params.get("unit_cm", M_TO_CM)))


ASPECT_TOLERANCE = 0.01  # how far a model's result may differ in proportions before it is refused (it cropped or padded)

# The four ways a result map is resized. The channel count cannot distinguish them (an image.1 may be depth or
# segmentation ids, an image.3 a picture or normals), so the node writing the map states it rather than it being
# inferred from the type.
LINEAR = "linear"    # bilinear: pictures, depth, masks, confidence, position maps; interpolating between values is correct
NEAREST = "nearest"  # nearest: segmentation ids, since interpolating between two ids yields a third id that does not exist
NORMALIZE = "normalize"  # bilinear, then renormalized: unit vectors such as normals
MOTION = "motion"    # bilinear, then scaled by the magnification: motion vectors are in pixels and scale with the picture
RESAMPLING = (LINEAR, NEAREST, NORMALIZE, MOTION)


def fit(values: np.ndarray, window, resample: str = LINEAR, valid: np.ndarray | None = None):
    """A worker's result brought back to the picture it was sent: a model that must run small gives its
    result at its own size, and it goes back to the plate's window here, in one place. `resample` says how (RESAMPLING
    above: the node writing the map knows what its values mean, the channel count does not). Its proportions must match
    (a model that cropped or padded is refused, E-FAMILY-ASPECT). Returns (values, valid, whether it was resized)."""
    from ...data.maps import resize

    if resample not in RESAMPLING:
        raise ValueError(f"resample must be one of {RESAMPLING}, not {resample!r}")
    want_w, want_h = window.canvas
    h, w = values.shape[:2]
    if (w, h) == (want_w, want_h):
        return values, valid, False
    if abs((w / h) - (want_w / want_h)) > ASPECT_TOLERANCE * (want_w / want_h):
        raise Invalid(Msg("E-FAMILY-ASPECT", width=w, height=h, want_width=want_w, want_height=want_h))
    if resample == NEAREST:
        ys = np.clip(((np.arange(want_h) + 0.5) * h / want_h).astype(np.int64), 0, h - 1)
        xs = np.clip(((np.arange(want_w) + 0.5) * w / want_w).astype(np.int64), 0, w - 1)
        out = values[ys][:, xs]
    else:
        out = resize(values, want_w, want_h)
        if resample == NORMALIZE:
            out = out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-6)
        elif resample == MOTION:  # pixels moved: as many times more as the picture is wider
            out = out * np.array([want_w / w, want_h / h, want_w / w, want_h / h], np.float32)[: out.shape[-1]]
    if valid is not None:
        valid = resize(np.asarray(valid, np.float32), want_w, want_h)
    return out, valid, True


def _entry(entry: tuple) -> tuple:
    """A frame_maps entry padded to (kind, value, writer options, how it resamples)."""
    kind, value, opts, *rest = entry
    opts = dict(opts or {})
    # A map carrying a class table (类别表) is an id map (segmentation): ids must not be interpolated. The class table
    # is a property of the data, not of the type: an image.1 alone does not tell depth from ids.
    return kind, value, opts, (rest[0] if rest else (NEAREST if "classes" in opts else LINEAR))


def frame_maps(ctx, raw: RawOutput, image: Packet, maps: dict, pattern: str = "frame_{}.npz", stage: str = "写出结果") -> dict[str, Packet]:
    """raw/<pattern> per frame -> one EXR sequence per output port. `maps`: port -> (kind, value, writer options), and
    optionally how it resamples (RESAMPLING; if omitted, NEAREST for a map carrying a 类别表, since ids never interpolate,
    and LINEAR otherwise). value is an array name in the frame's npz, or a function of the npz returning the array or
    (array, valid pixels). A frame without a file follows the RawOutput's MissingFrames."""
    from ...data.types import channels_of

    ctx.stage(stage)
    window = window_of(image)  # the plate the worker was sent: its results land on the same window
    maps = {port: _entry(entry) for port, entry in maps.items() if port in ctx.wanted}  # only what something wants
    writers: dict[str, ExrWriter] = {}

    def writer_for(port: str, validity: bool) -> ExrWriter:
        """The writer is created when the first frame arrives: whether the map has a validity channel is known only
        when it is written (whether the value step provides `valid`), so nobody needs to declare it in advance."""
        if port not in writers:
            kind, _, opts, _ = maps[port]
            writers[port] = ExrWriter(ctx.outputs[port], channels_of(kind),
                                      validity=validity, window=window, **opts)
        return writers[port]

    resized = False
    for f, d in raw.frames(ctx, image.meta["frames"], pattern):
        for port, (_, value, _, resample) in maps.items():
            got = d[value] if isinstance(value, str) else value(d)
            values, valid = got if isinstance(got, tuple) else (got, None)
            values, valid, did = fit(values, window, resample, valid)  # the model's own size back to the plate's
            resized = resized or did
            writer_for(port, valid is not None).add(f, values, valid)
    if resized:
        ctx.say("I-FAMILY-RESIZED", width=window.canvas[0], height=window.canvas[1])
    return {port: writer_for(port, False).packet() for port in maps}  # ports without any frame still produce an empty packet


def model_picture() -> tuple:
    """A picture a model answers in (display sRGB 0..1: a base colour like a texture, a matting model's foreground
    colour) is the working space (io/color.py): no conversion, the values clipped to 0..1 and written half-float,
    tagged the working space. Returns (convert, the writer options of a picture written in it)."""
    from ...io.color import load_config, working_space

    def convert(rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.clip(rgb, 0.0, 1.0), dtype=np.float32)

    return convert, {"half": True, "colorspace": working_space(load_config())}


def basecolor_map(array: str) -> tuple:
    """A frame_maps entry for the 基础色 (base colour, what CG also calls albedo; the project says basecolor
    everywhere) a model gives sRGB-encoded like a texture: written as it is, in the working space (sRGB, io/color.py);
    an output node converts
    it to ACEScg when the file is an EXR. The port it feeds is declared once too: kit/ports.py basecolor_port()."""
    convert, options = model_picture()
    return "image.3", lambda d: convert(d[array]), options


def depth_maps(ctx, raw: RawOutput, image: Packet, pattern: str = "frame_{}.npz", port: str = "depth") -> Packet:
    """raw/<pattern>: metric depth [H,W] per frame in metres -> a depth map packet (cm). Pixels that are not finite
    or not > 0 are invalid."""
    def depth(d):
        z = d["depth"]
        return z * M_TO_CM, np.isfinite(z) & (z > 0)

    # When nothing wants this port it is not computed (frame_maps writes only wanted ports); an empty packet is returned
    # instead of raising KeyError (when downstream wants only the disparity map, this port is absent from the result).
    made = frame_maps(ctx, raw, image, {port: ("image.1", depth, {"scale": "metric"})}, pattern, "写出深度图")
    return made[port] if port in made else empty_packet(ctx, port)
