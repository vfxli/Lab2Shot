"""Frames, masks and result files for workers: one reader per kind, whatever image library the environment has
(OpenCV, Pillow, OpenImageIO; OpenEXR for masks), and atomic .npz writes so the core never sees half a file."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from lab2shot_shared.exr import EXR_COMPRESSIONS, has_module, write_exr  # noqa: F401
from lab2shot_shared.units import M_TO_CM

from . import Failure, shown  # defined before the package imports this module


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """16-bit -> 8-bit, rounded (65535 / 257 = 255)."""
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 257.0 + 0.5).astype(np.uint8)
    return img


def _read_pixels(path: Path) -> np.ndarray:
    """[H, W, 3] RGB (uint8 or uint16), alpha dropped, grey expanded."""
    if has_module("cv2"):
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise Failure("E-FILES-FRAME", path=shown(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if has_module("PIL"):
        from PIL import Image

        with Image.open(path) as im:
            if im.mode in ("I;16", "I;16B", "I"):
                grey = np.asarray(im, dtype=np.uint16)
                return np.repeat(grey[..., None], 3, axis=2)
            return np.asarray(im.convert("RGB"))
    import OpenImageIO as oiio

    buf = oiio.ImageBuf(str(path))
    pixels = buf.get_pixels(oiio.UINT16 if buf.spec().format == oiio.UINT16 else oiio.UINT8)
    if pixels is None or buf.has_error:
        raise Failure("E-FILES-FRAMEDETAIL", path=shown(path), detail=buf.geterror())
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    return pixels[..., :3]


def read_frame(path: str | Path, dtype: str = "uint8", order: str = "rgb") -> np.ndarray:
    """A job frame (display-referred sRGB PNG) -> contiguous [H, W, 3].

    dtype "uint8": 0..255 (16-bit PNGs rounded to 8 bits); "float32": 0..1 (full 16-bit precision kept).
    order "rgb" or "bgr" (OpenCV models)."""
    img = _read_pixels(Path(path))
    if dtype == "uint8":
        img = _to_uint8(img)
    elif dtype == "float32":
        img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    else:
        raise ValueError(f"dtype must be uint8 or float32, not {dtype!r}")
    if order == "bgr":
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def read_mask(path: str | Path) -> np.ndarray:
    """A mask / matte -> float32 [H, W], 0..1 for mattes. EXR: channel A, else the only channel, else Y / R /
    mask / matte; PNG / JPG: grey 0..255 (or 16-bit) scaled to 0..1."""
    path = Path(path)
    if path.suffix.lower() == ".exr":
        if has_module("OpenEXR"):
            return next(iter(_exr_planes(path, lambda names: [_mask_channel(path, names)]).values()))
        if not has_module("OpenImageIO"):
            raise Failure("E-FILES-NOEXRMASK")
        import OpenImageIO as oiio

        buf = oiio.ImageBuf(str(path))
        names = list(buf.spec().channelnames)
        pixels = buf.get_pixels(oiio.FLOAT)
        if pixels is None or buf.has_error:
            raise Failure("E-FILES-MASKDETAIL", path=shown(path), detail=buf.geterror())
        return np.ascontiguousarray(pixels[..., names.index(_mask_channel(path, names))])
    if has_module("cv2"):
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise Failure("E-FILES-MASK", path=shown(path))
    else:
        from PIL import Image

        with Image.open(path) as im:
            img = np.asarray(im if im.mode in ("I;16", "I;16B") else im.convert("L"))
    return img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)


def read_channels(path: str | Path, names: tuple[str, ...] = ("R", "G")) -> np.ndarray:
    """Named channels of a multi-channel EXR written by Lab2Shot -> float32 [H, W, len(names)].

    `read_mask` reads only one channel; maps with two or more (an orientation field's angle + confidence, an ST-map's
    U + V, any image.2 / image.3 data map) need this function. Channel names follow data/payloads.py channel_names:
    R G B A, plus an optional valid.
    """
    path = Path(path)
    if path.suffix.lower() != ".exr":
        raise Failure("E-FILES-NOTEXR", path=shown(path))
    if not has_module("OpenEXR"):
        raise Failure("E-FILES-NOEXRMASK")

    def pick(channels: list[str]) -> list[str]:
        missing = [n for n in names if n not in channels]
        if missing:
            raise Failure("E-FILES-CHANNELS", file=path.name, want=missing, channels=sorted(channels))
        return list(names)

    planes = _exr_planes(path, pick)
    return np.stack([planes[n] for n in names], axis=-1)


def read_depth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """A depth map a node sent -> (metres [H, W] float32, valid [H, W] bool).

    Every float map Lab2Shot writes names its channel R (for one channel) plus a valid channel
    (`lab2shot/data/payloads.py channel_names`: there is essentially one kind of float map, differing only in channel
    count, with no second naming scheme by meaning). The `depth.Z` scheme is used only when delivering to Nuke
    (`lab2shot/data/layers.py`) and follows other applications' conventions. `R` is therefore checked first, since
    accepting only `Z` would fail to read every depth map Lab2Shot hands to a worker; `Z` / `A` are still accepted,
    because a user may connect a depth EXR rendered by a DCC directly (as lenient as `read_mask`)."""
    if not has_module("OpenEXR"):
        raise Failure("E-FILES-NOEXRDEPTH")
    path = Path(path)
    # the depth channel, then the file's validity channel when it has one
    planes = list(_exr_planes(path, lambda channels: [_depth_channel(path, channels),
                                                      *[n for n in ("valid", "A") if n in channels][:1]]).values())
    z = planes[0]
    a = planes[1] if len(planes) > 1 else np.ones_like(z)
    valid = (a > 0) & np.isfinite(z) & (z > 0)
    return np.where(valid, z / M_TO_CM, 0.0).astype(np.float32), valid


def _exr_planes(path: Path, pick) -> dict[str, np.ndarray]:
    """Channels of an EXR as float32 [H, W] planes (its data window), in the order `pick(names)` chooses them from
    the file's channel names; the library converts every pixel type (half, float, uint) to float32.

    Read with the OpenEXR bindings' InputFile, the reading API of every version of the bindings: an extension's
    environment has the version its upstream pins, and the File class exists only from 3.3 on (ViPE pins OpenEXR<3.3)."""
    import Imath
    import OpenEXR

    exr = OpenEXR.InputFile(str(path))
    try:
        header = exr.header()
        box = header["dataWindow"]
        h, w = box.max.y - box.min.y + 1, box.max.x - box.min.x + 1
        as_float = Imath.PixelType(Imath.PixelType.FLOAT)
        # frombuffer shares the library's read-only bytes: copied so that callers may change the pixels in place
        return {n: np.frombuffer(exr.channel(n, as_float), np.float32).reshape(h, w).copy()
                for n in pick(list(header["channels"]))}
    finally:
        exr.close()


def _depth_channel(path: Path, channels: list[str]) -> str:
    """The channel holding depth values: R for Lab2Shot's own maps, Z for DCC renders, or the only channel when there is one."""
    for name in ("R", "Z"):
        if name in channels:
            return name
    value = [c for c in channels if c not in ("valid", "A")]
    if len(value) == 1:
        return value[0]
    raise Failure("E-FILES-DEPTHCHANNELS", file=path.name, channels=sorted(channels))


def _mask_channel(path: Path, channels: list[str]) -> str:
    if "A" in channels:
        return "A"
    if len(channels) == 1:
        return channels[0]
    for name in ("Y", "R", "mask", "matte"):
        if name in channels:
            return name
    raise Failure("E-FILES-MASKCHANNELS", file=path.name, channels=sorted(channels))


def save_npz(path: str | Path, compression: int = 0, **arrays) -> Path:
    """np.savez, written to <name>.part and renamed: a crash never leaves half a file under the real name.
    compression: 0 = stored (fastest), 1 = fast zlib (masks shrink ~20x), 6 = np.savez_compressed."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    mode = zipfile.ZIP_DEFLATED if compression else zipfile.ZIP_STORED
    kwargs = {"compresslevel": compression} if compression else {}
    with zipfile.ZipFile(part, "w", compression=mode, allowZip64=True, **kwargs) as zf:
        for key, value in arrays.items():
            with zf.open(key + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(value), allow_pickle=False)
    part.replace(path)
    return path


def fit_size(width: int, height: int, long_side: float, multiple: int = 1, *, upscale: bool = True,
             minimum: int | None = None) -> tuple[int, int]:
    """(w, h) with the aspect ratio kept, the long side ~ `long_side` (never above the input when upscale=False),
    both sides rounded to `multiple` and at least `minimum` (default: one multiple)."""
    scale = long_side / max(width, height)
    if not upscale:
        scale = min(1.0, scale)
    low = multiple if minimum is None else minimum
    return (max(low, int(round(width * scale / multiple)) * multiple),
            max(low, int(round(height * scale / multiple)) * multiple))


def link_file(link: Path, target: Path) -> None:
    """Make `link` a symlink to `target` (replacing whatever is there): weights where an upstream repo expects them."""
    import os

    link, target = Path(link), Path(target)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == str(target):
            return
        link.unlink()
    link.symlink_to(target)
