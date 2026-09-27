"""Read and write EXR / PNG / JPG with OpenImageIO.

Alpha follows the Nuke convention: in memory and in EXR files, colour is premultiplied by alpha (OpenEXR associated
alpha). PNG stores straight alpha (per the PNG standard); OpenImageIO premultiplies on read and divides on write, so
straight values never reach the code unexpectedly."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from ..messages import Msg
from . import FileProblem


def said(path: str | Path, detail: str) -> str:
    """返回读写库（OpenImageIO / OpenEXR）的原始错误信息，其中的服务器路径替换为文件名。

    原始信息常含完整路径（`"/home/…/uploads/sets/…/plate.exr" is not an OpenEXR file`），
    该路径是文件在服务器上的位置，对使用者无意义，且不应暴露。"""
    text = str(detail or "").strip()
    folder = str(Path(path).parent)
    return text.replace(folder + "/", "").replace(folder, "") if folder not in ("", ".") else text


def read_rgb(path: str | Path, box: tuple | None = None) -> np.ndarray:
    """Read the picture as float32 HxWx3: channels R, G, B by name (a multi-layer EXR may hold other layers alongside
    them). A file without them yields its first three channels; a single-channel file repeats that channel three
    times. `box`: as in read_named."""
    names = channel_names(path)
    pick = ["R", "G", "B"] if all(c in names for c in "RGB") else names[:3] if len(names) >= 3 else names[:1] * 3
    named = read_named(path, list(dict.fromkeys(pick)), box=box)
    return np.stack([named[c] for c in pick], axis=-1)


ALPHA_NAMES = ("A", "alpha")  # the picture's own alpha: an unprefixed A (a PNG's fourth channel, an EXR's rgba layer)


def has_alpha(path: str | Path) -> bool:
    """Return whether the picture has its own alpha channel."""
    return any(n in ALPHA_NAMES for n in channel_names(path))


def read_rgba(path: str | Path, box: tuple | None = None) -> np.ndarray:
    """Read the picture as float32 HxWx4: R, G, B premultiplied by A, then A (1 when the file has no alpha)."""
    rgb = read_rgb(path, box)
    names = channel_names(path)
    a = next((n for n in ALPHA_NAMES if n in names), None)
    alpha = read_named(path, [a], box=box)[a] if a else np.ones(rgb.shape[:2], np.float32)
    return np.concatenate([rgb, alpha[..., None]], axis=-1).astype(np.float32)


def read_picture(path: str | Path, alpha: bool, box: tuple | None = None) -> np.ndarray:
    """Read the picture with alpha (HxWx4, premultiplied) when `alpha` is set, otherwise its colour as stored (HxWx3);
    covers the display window, or `box` as in read_named."""
    return read_rgba(path, box) if alpha else read_rgb(path, box)


def read_own(path: str | Path, *, channels: int | None = None, scale8: bool = True,
             scale16: bool = True) -> tuple[np.ndarray, tuple[int, int, int, str]]:
    """Read the picture as stored, without conversion to three channels as in read_rgb: the file's own channel count
    (capped at `channels`), pixels as float32. `scale8` / `scale16`: whether integers of an 8- or 16-bit file are
    scaled to 0..1 (colour) or kept as raw values (instance ids, 16-bit depth); floating-point files are always
    returned as stored. Also returns the file header: (width, height, channel count, type). The data QA
    (datasets/qa.py) and the ground-truth reading blocks (datasets/blocks/image.py) use this function as their single
    entry point to OpenImageIO."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        spec = inp.spec()
        base = spec.format.basetype
        n = spec.nchannels if channels is None else min(channels, spec.nchannels)
        keep = base in (oiio.UINT8, oiio.UINT16) and not (scale8 if base == oiio.UINT8 else scale16)
        a = inp.read_image(0, 0, 0, n, base if keep else oiio.FLOAT)  # read as float: integers are normalised to 0..1
        if a is None:
            raise FileProblem(Msg("E-IMAGE-READ", file=Path(path).name, detail=said(path, inp.geterror())))
        px = np.asarray(a, np.float32)
        header = (spec.width, spec.height, spec.nchannels, str(spec.format))
    finally:
        inp.close()
    if px.ndim == 2:
        px = px[..., None]
    return px, header


def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    """Convert premultiplied HxWx4 to straight colour HxWx3 (where alpha is 0 the colour is left unchanged, i.e. black)."""
    a = rgba[..., 3:4]
    return np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-8), rgba[..., :3]).astype(np.float32)


def header(path: str | Path) -> tuple[int, int, int, str]:
    """Return the file header: width, height, channel count and type name ("uint8", "half" ...), with a single open
    for checks that need several fields. Used by the data QA's detection; pixels are read by read_own / read_rgb."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        spec = inp.spec()
        return spec.width, spec.height, spec.nchannels, str(spec.format)
    finally:
        inp.close()


def image_size(path: str | Path) -> tuple[int, int]:
    """Return the picture's format, i.e. its display window size (an EXR whose data window covers only part of the
    frame, or varies between frames, still has the format Nuke displays)."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        spec = inp.spec()
        return spec.full_width, spec.full_height
    finally:
        inp.close()


def stores_float(path: str | Path) -> bool:
    """Return whether the picture's pixels are floating point (half, float, double: an EXR, a Radiance .hdr), per its header."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        return inp.spec().format.basetype in (oiio.HALF, oiio.FLOAT, oiio.DOUBLE)
    finally:
        inp.close()


Window = tuple[int, int, int, int]  # x, y, width, height (data/windows.py Window describes what a packet stores)


def _windows(inp) -> tuple[Window, Window]:
    """Return an open file's display window and the union of its parts' data windows, both in file coordinates."""
    inp.seek_subimage(0, 0)
    spec = inp.spec()
    full = (spec.full_x, spec.full_y, spec.full_width, spec.full_height)
    x0, y0, x1, y1 = spec.x, spec.y, spec.x + spec.width, spec.y + spec.height
    i = 1
    while inp.seek_subimage(i, 0):
        s = inp.spec()
        x0, y0, x1, y1 = min(x0, s.x), min(y0, s.y), max(x1, s.x + s.width), max(y1, s.y + s.height)
        i += 1
    return full, (x0, y0, x1 - x0, y1 - y0)


def windows(path: str | Path) -> tuple[Window, Window]:
    """Return the picture's display window (its format, in file coordinates) and its data window (the union over all
    parts, i.e. the pixels the file stores), with the data window's x and y relative to the display window's top-left
    corner. Pixels outside the display window are overscan."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        full, (x, y, w, h) = _windows(inp)
        return full, (x - full[0], y - full[1], w, h)
    finally:
        inp.close()


def _placed(data: np.ndarray, spec, target: Window) -> np.ndarray:
    """Place a part's pixels [h,w,C] (its data window) at their position within `target` (x, y, width, height in file
    coordinates), filling the remainder with 0; pixels outside `target` are dropped."""
    tx, ty, tw, th = target
    if (spec.x, spec.y, spec.width, spec.height) == target:
        return data
    out = np.zeros((th, tw, data.shape[-1]), np.float32)
    x0, y0 = spec.x - tx, spec.y - ty
    sx, sy = max(0, -x0), max(0, -y0)
    ex, ey = min(data.shape[1], tw - x0), min(data.shape[0], th - y0)
    if ex > sx and ey > sy:
        out[y0 + sy:y0 + ey, x0 + sx:x0 + ex] = data[sy:ey, sx:ex]
    return out


def write_image(path: str | Path, rgb: np.ndarray, *, exr_half: bool = True, exr_compression: str = "zip",
               png_bit_depth: int = 8, png_compress_level: int = 6, alpha: str = "auto", jpg_quality: int = 95,
               windows: tuple | None = None) -> None:
    """Write float RGB, or premultiplied RGBA, to EXR (`exr_half`: 16-bit half or 32-bit float; `exr_compression`:
    one of worker_sdk's EXR_COMPRESSIONS), or to PNG / JPG (values clamped to 0..1, `png_bit_depth` 8 or 16,
    `jpg_quality` 1-100). PNG stores alpha straight (`alpha`: "auto" writes it when the picture has one, "yes" always
    writes it, adding an opaque alpha if absent, "no" never writes it); JPG has no alpha (only colour is written, and
    callers must state this). This is the single implementation for every format a settings node offers
    (序列图输出设置, core/image_output.py); all other image writes (view/frames.py, internal conversions) call it with
    the defaults."""
    from lab2shot_worker.files import write_exr  # the single EXR writer, shared with the workers

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    has_alpha = rgb.shape[-1] == 4
    if path.suffix.lower() == ".exr":
        write_exr(path, rgb, "RGBA" if has_alpha else "RGB", half=exr_half, compression=exr_compression, windows=windows)  # EXR preserves overscan (windows())
        return
    if path.suffix.lower() in (".jpg", ".jpeg"):
        write_png(path, as_uintN(rgb[..., :3], 8), quality=jpg_quality)
        return
    give_alpha = has_alpha if alpha == "auto" else alpha == "yes"
    if give_alpha:
        rgba = rgb if has_alpha else np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=-1)
        # PNG stores straight alpha: unpremultiply in float before rounding so that faint edges keep their colour
        write_png(path, as_uintN(np.concatenate([unpremultiply(rgba), rgba[..., 3:]], axis=-1), png_bit_depth),
                  straight=True, compress_level=png_compress_level)
    else:
        write_png(path, as_uintN(rgb[..., :3], png_bit_depth), compress_level=png_compress_level)


def as_uint8(img: np.ndarray) -> np.ndarray:
    """Convert 0..1 floats to 8-bit integers, rounded."""
    return as_uintN(img, 8)


def as_uintN(img: np.ndarray, bits: int) -> np.ndarray:
    """Convert 0..1 floats to `bits`-bit integers (8 or 16), rounded."""
    top = (1 << bits) - 1
    dtype = np.uint8 if bits == 8 else np.uint16
    return (np.clip(img, 0.0, 1.0) * top + 0.5).astype(dtype)


def views(path: str | Path) -> list[str]:
    """Return the views of a stereo (multi-view) EXR in file order: each part's view (multi-part file), or a single
    part's multiView list (the first view's channels unprefixed, the others prefixed with the view name); [] for a
    single-view picture."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        found = [str(v) for v in (inp.spec().getattribute("multiView") or ())]
        i = 0
        while inp.seek_subimage(i, 0):
            own = str(inp.spec().getattribute("view") or "")
            if own and own not in found:
                found.append(own)
            i += 1
        return found
    finally:
        inp.close()


def _view_of(channel: str, multi_view: list[str]) -> str:
    """返回单部分多视角 EXR 中该通道所属的视角，遵循 OpenEXR 的约定（ImfMultiView viewFromChannelName）：
    通道名按点分段，倒数第二段为视角名时属于该视角，否则属于第一个视角（hero view）。
    因此 left.R、forward.left.u、whitebarmask.left.mask 属于左眼，disparityL.x 和 Z 属于 hero 视角。
    不得只判断第一段，否则 forward.left.u 这类通道会被错误归入 hero 视角。"""
    parts = channel.split(".")
    return parts[-2] if len(parts) >= 2 and parts[-2] in multi_view else multi_view[0]


def _without_view(channel: str, view: str) -> str:
    """去掉通道名中的视角名一段，返回该通道在此视角内的原名：forward.left.u -> forward.u，left.R -> R。"""
    parts = channel.split(".")
    if len(parts) >= 2 and parts[-2] == view:
        del parts[-2]
    return ".".join(parts)


def _parts(inp, view: str | None = None) -> list[list[str | None]]:
    """Return the channel names of every part of an open file. A multi-part EXR may store one layer per part:
    unprefixed channels of any part other than the first take the part name as their layer ("depth" part: Z ->
    depth.Z). When a part's channel names collide with those of an earlier part (a stereo pair: forward_left and
    forward_right both hold forward.u), every channel of that part is prefixed with the part name
    (forward_right.forward.u), so all channels across parts are distinct. `view` (stereo files): only that view's
    channels are returned, with None in place of every other channel (so each name's position still matches its
    channel index); the first kept part is unprefixed and part names drop the view suffix (depth.right -> depth), so
    each view reads as an independent picture."""
    parts: list[list[str | None]] = []
    taken: set[str] = set()
    inp.seek_subimage(0, 0)
    together = [str(v) for v in (inp.spec().getattribute("multiView") or ())]  # a single part holding every view
    kept = 0
    while inp.seek_subimage(len(parts), 0):
        spec = inp.spec()
        own = str(spec.getattribute("view") or "")
        if view is not None and own and own != view:
            parts.append([None] * spec.nchannels)
            continue
        part = str(spec.getattribute("name") or "") if kept else ""
        raw: list[str | None] = list(spec.channelnames)
        if view is not None:
            part = re.sub(rf"[._]{re.escape(own)}$", "", part) if own else part
            if not own and len(together) > 1:  # 单个部分包含所有视角：按 OpenEXR 的约定选出该视角的通道
                raw = [(_without_view(n, view) if _view_of(n, together) == view else None) for n in raw]
        names = [None if n is None else n if "." in n or not part else f"{part}.{n}" for n in raw]
        present = {n for n in names if n is not None}
        if part and taken & present:
            names = [None if n is None else f"{part}.{n}" for n in raw]
            present = {n for n in names if n is not None}
        taken |= present
        kept += 1
        parts.append(names)
    return parts


def channel_names(path: str | Path, view: str | None = None) -> list[str]:
    """Return every channel name of an image file (all parts of a multi-part EXR; one view of a stereo file), read
    from the header only."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        return [n for names in _parts(inp, view) for n in names if n is not None]
    finally:
        inp.close()


def header_attributes(path: str | Path) -> dict[str, str]:
    """Return the text attributes in an image file's header (first part): Cryptomatte manifests, lab2shot:* ..."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        return {a.name: a.value for a in inp.spec().extra_attribs if isinstance(a.value, str)}
    finally:
        inp.close()


def read_named(path: str | Path, names: list[str] | None = None, view: str | None = None,
               box: tuple | None = None) -> dict[str, np.ndarray]:
    """Read channels by name (EXR files store channels by name, in no fixed order): {"Z": [H,W], "A": [H,W]}. Reads
    every channel of every part, or only `names`, so one layer of a render with dozens of channels can be read alone.
    `box` (x, y, w, h) is relative to the display window's top-left corner, as in a packet's data window
    (data/windows.py): every channel is returned at that size, with the file's pixels in place and 0 elsewhere. None
    selects the display window (the picture's format, as used by nodes operating on the plate frame)."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:  # 文件无法打开（区别于打开后读不到像素）
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    out: dict[str, np.ndarray] = {}
    try:
        full, _ = _windows(inp)
        target = full if box is None else (full[0] + int(box[0]), full[1] + int(box[1]), int(box[2]), int(box[3]))
        for sub, channels in enumerate(_parts(inp, view)):
            wanted = [i for i, n in enumerate(channels) if n is not None and (names is None or n in names)]
            if not wanted:
                continue
            lo, hi = wanted[0], wanted[-1] + 1  # read the whole span at once (a layer's channels are contiguous)
            pixels = inp.read_image(sub, 0, lo, hi, oiio.FLOAT)
            if pixels is None:
                raise FileProblem(Msg("E-IMAGE-READ", file=Path(path).name, detail=said(path, inp.geterror())))
            data = np.asarray(pixels, np.float32).reshape(pixels.shape[0], pixels.shape[1], hi - lo)
            inp.seek_subimage(sub, 0)
            data = _placed(data, inp.spec(), target)
            out.update({channels[i]: data[..., i - lo] for i in wanted})
    finally:
        inp.close()
    missing = [n for n in names or () if n not in out]
    if missing:
        raise FileProblem(Msg("E-IMAGE-NOCHANNEL", file=Path(path).name, channels=", ".join(missing)))
    return out


def write_png(path: str | Path, rgb: np.ndarray, straight: bool = False, *, compress_level: int = 6, quality: int = 95) -> None:
    """Write 8- or 16-bit (per `rgb`'s dtype) RGB or RGBA to PNG (or JPG, per the suffix), creating the folder. RGBA is
    either premultiplied (OpenImageIO divides so the PNG stores straight alpha) or already `straight` (written as-is).
    `compress_level` (PNG, 0-9) and `quality` (JPG, 1-100) apply only to their respective formats. A file that cannot
    be opened raises an error, because writing into an ImageOutput that failed to open crashes the process."""
    h, w, c = rgb.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise FileProblem(Msg("E-IMAGE-WRITE", file=Path(path).name, detail=said(path, oiio.geterror())))
    spec = oiio.ImageSpec(w, h, c, oiio.UINT16 if rgb.dtype == np.uint16 else oiio.UINT8)
    if straight:
        spec.attribute("oiio:UnassociatedAlpha", 1)
    if path.suffix.lower() == ".png":
        spec.attribute("png:compressionLevel", compress_level)
    elif path.suffix.lower() in (".jpg", ".jpeg"):
        spec.attribute("CompressionQuality", quality)
    try:
        if not out.open(str(path), spec):
            raise FileProblem(Msg("E-IMAGE-WRITE", file=Path(path).name, detail=said(path, out.geterror())))
        out.write_image(np.ascontiguousarray(rgb))
    finally:
        out.close()
