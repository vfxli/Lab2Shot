"""EXR writer shared by the core and the workers (OpenEXR, falling back to OpenImageIO): named channels, half or full
float, header attributes and the compression modes offered by Nuke. Files are written to <name>.part.exr and renamed."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .protocol import Failure, shown

def has_module(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


EXR_COMPRESSIONS = {
    "none": ("none", "NO_COMPRESSION"),
    "zips": ("zips", "ZIPS_COMPRESSION"),  # ZIP, one scanline per block
    "zip": ("zip", "ZIP_COMPRESSION"),  # ZIP, 16 scanlines per block
    "piz": ("piz", "PIZ_COMPRESSION"),
    "pxr24": ("pxr24", "PXR24_COMPRESSION"),
    "dwaa": ("dwaa", "DWAA_COMPRESSION"),
    "dwab": ("dwab", "DWAB_COMPRESSION"),
}


# Per-channel storage types (the three OpenEXR pixel types); names accepted by `write_exr`'s `types`
PIXEL_TYPES = ("half", "float", "uint")
_NUMPY_OF = {"half": np.float16, "float": np.float32, "uint": np.uint32}


def write_exr(path: str | Path, data: np.ndarray, channels: str | Sequence[str] = "RGB", attrs: dict | None = None,
              chromaticities: tuple | None = None, half: bool | Sequence[bool] = False, header: dict | None = None,
              compression: str = "zip", windows: tuple | None = None, types: Sequence[str] | None = None) -> Path:
    """Write an EXR with one channel per entry of `channels` and return its path.

    `channels` is either a string of single-letter names ("RGBA", "ZA") or a list of full names ("R", "depth.Z",
    "N.X", ...); `data` has shape [H, W, len(channels)] ([H, W] for a single channel). Channels are stored as float32,
    or float16 where `half` is set (a single flag for all channels or one flag per channel). `compression` is a key
    of EXR_COMPRESSIONS (default "zip", 16-scanline ZIP; output nodes expose the same choices as Nuke). `attrs` are
    written as lab2shot:<key> header attributes; `header` entries are written under their own names (e.g. a
    Cryptomatte manifest, cryptomatte/<key>/manifest). `windows` is (display window (x, y, width, height), data window
    (x, y relative to the display window origin, width, height)) as returned by Lab2Shot's io/images.windows, with
    `data` holding the data-window pixels (preserving render overscan); None uses the size of `data` for both. The
    file is written to <name>.part.exr and renamed so that readers never observe a partial file. Uses OpenEXR or
    OpenImageIO.

    `types` gives one of PIXEL_TYPES per channel ("half" / "float" / "uint") in place of `half`, for channels that
    must be stored as OpenEXR 32-bit unsigned integers (e.g. an ID pass). When `types` is given, `data` may be
    float64, the only NumPy float type that represents every uint32 exactly (float32 is exact only up to 2**24).
    Pixels are converted per channel on output, so a half read as float32 or a uint read as float64 round-trips
    exactly. Only channel-level uploads (lab2shot/transfer/uploads.py add_planes) write user channels this way."""
    path = Path(path)
    if compression not in EXR_COMPRESSIONS:
        raise ValueError(f"compression must be one of {sorted(EXR_COMPRESSIONS)}, not {compression!r}")
    names = list(channels)
    if types is not None:
        kinds = list(types)
        if len(kinds) != len(names) or any(k not in PIXEL_TYPES for k in kinds):
            raise ValueError(f"types: one of {PIXEL_TYPES} per channel, got {kinds!r} for {len(names)} channels")
    else:
        halves = [half] * len(names) if isinstance(half, bool) else list(half)
        if len(halves) != len(names):
            raise ValueError(f"half: {len(halves)} flags for {len(names)} channels")
        kinds = ["half" if hf else "float" for hf in halves]
    carrier = np.float64 if "uint" in kinds else np.float32
    data = np.asarray(data, carrier).reshape(*np.shape(data)[:2], len(names))
    extra = {**{f"lab2shot:{k}": v for k, v in (attrs or {}).items()}, **(header or {})}
    h, w = data.shape[:2]
    (dx, dy, dw, dh), (ox, oy, ow, oh) = windows or ((0, 0, w, h), (0, 0, w, h))
    if (ow, oh) != (w, h):
        raise ValueError(f"windows: a data window of {ow}x{oh} for pixels of {w}x{h}")
    tmp = path.with_name(path.stem + ".part.exr")
    if has_module("OpenEXR"):
        import OpenEXR

        head = {"compression": getattr(OpenEXR, EXR_COMPRESSIONS[compression][1]), "type": OpenEXR.scanlineimage, **extra}
        if windows is not None:  # OpenEXR windows use inclusive corners
            head["displayWindow"] = (np.array([dx, dy], np.int32), np.array([dx + dw - 1, dy + dh - 1], np.int32))
            head["dataWindow"] = (np.array([dx + ox, dy + oy], np.int32), np.array([dx + ox + w - 1, dy + oy + h - 1], np.int32))
        if chromaticities:
            head["chromaticities"] = chromaticities
        pixels = {n: np.ascontiguousarray(data[..., i], _NUMPY_OF[kinds[i]]) for i, n in enumerate(names)}
        with OpenEXR.File(head, pixels) as exr:
            exr.write(str(tmp))
    else:
        import OpenImageIO as oiio

        c = data.shape[2]
        oiio_of = {"half": oiio.HALF, "float": oiio.FLOAT, "uint": oiio.UINT32}
        spec = oiio.ImageSpec(w, h, c, oiio_of[kinds[0]])
        spec.x, spec.y, spec.full_x, spec.full_y, spec.full_width, spec.full_height = dx + ox, dy + oy, dx, dy, dw, dh
        spec.channelnames = tuple(names)
        if len(set(kinds)) > 1:
            spec.channelformats = tuple(oiio.TypeDesc(oiio_of[k]) for k in kinds)
        spec.attribute("compression", EXR_COMPRESSIONS[compression][0])
        if chromaticities:
            spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), chromaticities)
        for key, value in extra.items():
            spec.attribute(key, value)
        out = oiio.ImageOutput.create(str(tmp))
        if out is None or not out.open(str(tmp), spec) or not out.write_image(data) or not out.close():
            raise Failure("E-FILES-WRITE", path=shown(path), detail=oiio.geterror())
    tmp.replace(path)
    return path
