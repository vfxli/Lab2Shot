"""EXR writer shared by the core and the workers (OpenImageIO or OpenEXR, whichever is installed; see `write_exr`):
named channels of the three OpenEXR pixel types, header attributes and the compression modes offered by Nuke. Files are
written to <name>.part.exr and renamed."""

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


def write_exr(path: str | Path, data: np.ndarray | Sequence[np.ndarray], channels: str | Sequence[str] = "RGB",
              attrs: dict | None = None, chromaticities: tuple | None = None, half: bool | Sequence[bool] = False,
              header: dict | None = None, compression: str = "zip", windows: tuple | None = None,
              types: Sequence[str] | None = None, level: int | None = None) -> Path:
    """Write an EXR with one channel per entry of `channels` and return its path.

    `channels` is either a string of single-letter names ("RGBA", "ZA") or a list of full names ("R", "depth.Z",
    "N.X", ...); `data` has shape [H, W, len(channels)] ([H, W] for a single channel), or is a list of [H, W] planes,
    one per channel, each in its own dtype. Channels are stored as float32, or float16 where `half` is set (a single
    flag for all channels or one flag per channel). `compression` is a key of EXR_COMPRESSIONS (default "zip",
    16-scanline ZIP; output nodes expose the same choices as Nuke). `attrs` are written as lab2shot:<key> header
    attributes; `header` entries are written under their own names (e.g. a Cryptomatte manifest,
    cryptomatte/<key>/manifest). `windows` is (display window (x, y, width, height), data window (x, y relative to the
    display window origin, width, height)) as returned by Lab2Shot's io/images.windows, with `data` holding the
    data-window pixels (preserving render overscan); None uses the size of `data` for both. The file is written to
    <name>.part.exr and renamed so that readers never observe a partial file.

    `types` gives one of PIXEL_TYPES per channel ("half" / "float" / "uint") in place of `half`, for channels that
    must be stored as OpenEXR 32-bit unsigned integers (e.g. an ID pass). A uint channel holds integer values, not
    0..1: they are written as they are (uint32 planes, as channel-level uploads keep them, or integral floats), so
    pass uint channels as uint32 planes in a list (a float32 stack is exact only up to 2**24). Only channel-level
    uploads (lab2shot/transfer/uploads.py add_planes) write user channels this way.

    Backend: OpenImageIO when installed (the core; it takes `level` and the core sizes its thread pool), else OpenEXR
    (the extension environments that have it). OpenImageIO converts between pixel types through float32 (an integer
    type normalised to 0..1), so a uint channel beside channels of another type is written with OpenEXR, which the
    core also installs; all-uint files and files without uint are exact with either. The OpenEXR backend uses the
    File class of the bindings 3.3 and later: an environment whose upstream pins older bindings (ViPE: OpenEXR<3.3)
    reads EXR (lab2shot_worker/files.py reads with the API every version has) but cannot write it this way.

    `level`: the deflate level of "zip" / "zips" (1 fastest .. 9 smallest; None: the library's default). It changes
    only how hard the writer tries, not the format: every reader reads the file the same way. OpenImageIO takes it;
    the OpenEXR bindings have no setting for it and write their default level.

    `header` entries with an empty value are not written: OpenImageIO adds DateTime (the time of writing) unless it
    is given empty, and the OpenEXR bindings write no attribute for it either."""
    path = Path(path)
    if compression not in EXR_COMPRESSIONS:
        raise ValueError(f"compression must be one of {sorted(EXR_COMPRESSIONS)}, not {compression!r}")
    if level is not None and (compression not in ("zip", "zips") or not 1 <= level <= 9):
        raise ValueError(f"level: 1-9, for zip or zips only (got {level!r} for {compression!r})")
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
    # Values reach the libraries as float32 (half / float channels) or uint32 (uint channels, taken as integers)
    carrier = [np.uint32 if k == "uint" else np.float32 for k in kinds]
    if isinstance(data, np.ndarray):
        data = data.reshape(*data.shape[:2], len(names))
        planes = [data[..., i] for i in range(len(names))]
    else:
        planes = list(data)
        if len(planes) != len(names) or any(np.shape(p) != np.shape(planes[0]) or np.ndim(p) != 2 for p in planes):
            raise ValueError(f"data: {len(names)} planes of one [H, W] size, got {[np.shape(p) for p in planes]}")
    extra = {**{f"lab2shot:{k}": v for k, v in (attrs or {}).items()}, **(header or {})}
    h, w = np.shape(planes[0])
    (dx, dy, dw, dh), (ox, oy, ow, oh) = windows or ((0, 0, w, h), (0, 0, w, h))
    if (ow, oh) != (w, h):
        raise ValueError(f"windows: a data window of {ow}x{oh} for pixels of {w}x{h}")
    tmp = path.with_name(path.stem + ".part.exr")
    mixed_uint = "uint" in kinds and len(set(kinds)) > 1
    if mixed_uint and not has_module("OpenEXR"):
        raise ValueError("a uint channel beside channels of another type needs the OpenEXR bindings")
    if mixed_uint or not has_module("OpenImageIO"):
        import OpenEXR

        head = {"compression": getattr(OpenEXR, EXR_COMPRESSIONS[compression][1]), "type": OpenEXR.scanlineimage,
                **{k: v for k, v in extra.items() if v != ""}}
        if windows is not None:  # OpenEXR windows use inclusive corners
            head["displayWindow"] = (np.array([dx, dy], np.int32), np.array([dx + dw - 1, dy + dh - 1], np.int32))
            head["dataWindow"] = (np.array([dx + ox, dy + oy], np.int32), np.array([dx + ox + w - 1, dy + oy + h - 1], np.int32))
        if chromaticities:
            head["chromaticities"] = chromaticities
        # A plane already in its stored type is written as it is (no copy of a channel upload's planes)
        stored = [np.asarray(p) for p in planes]
        pixels = {n: np.ascontiguousarray(p if p.dtype == _NUMPY_OF[k] else np.asarray(p, c), _NUMPY_OF[k])
                  for n, p, k, c in zip(names, stored, kinds, carrier)}
        with OpenEXR.File(head, pixels) as exr:
            exr.write(str(tmp))
    else:
        import OpenImageIO as oiio

        c = len(names)
        # One carrier for all channels (OpenImageIO takes a single array): uint32 for an all-uint file, else float32
        if isinstance(data, np.ndarray):
            data = np.asarray(data, carrier[0])
        else:
            data = np.stack([np.asarray(p, carrier[0]) for p in planes], axis=-1)
        oiio_of = {"half": oiio.HALF, "float": oiio.FLOAT, "uint": oiio.UINT32}
        spec = oiio.ImageSpec(w, h, c, oiio_of[kinds[0]])
        spec.x, spec.y, spec.full_x, spec.full_y, spec.full_width, spec.full_height = dx + ox, dy + oy, dx, dy, dw, dh
        spec.channelnames = tuple(names)
        if len(set(kinds)) > 1:
            spec.channelformats = tuple(oiio.TypeDesc(oiio_of[k]) for k in kinds)
        spec.attribute("compression", EXR_COMPRESSIONS[compression][0] + ("" if level is None else f":{level}"))
        if chromaticities:
            spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), chromaticities)
        for key, value in extra.items():
            spec.attribute(key, value)
        out = oiio.ImageOutput.create(str(tmp))
        if out is None or not out.open(str(tmp), spec) or not out.write_image(data) or not out.close():
            raise Failure("E-FILES-WRITE", path=shown(path), detail=oiio.geterror())
    tmp.replace(path)
    return path
