"""Read and write EXR / PNG / JPG with OpenImageIO.

Alpha follows the Nuke convention: in memory and in EXR files, colour is premultiplied by alpha (OpenEXR associated
alpha). PNG stores straight alpha (per the PNG standard); OpenImageIO premultiplies on read and divides on write, so
straight values never reach the code unexpectedly."""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

from ..messages import Msg
from . import FileProblem


def said(path: str | Path, detail: str) -> str:
    """The image library's (OpenImageIO / OpenEXR) own error text, with the server's folder taken out of it.

    That text often holds the full path (`"/home/…/uploads/sets/…/plate.exr" is not an OpenEXR file`): where the
    file is on the server means nothing to the user and is not shown to them."""
    text = str(detail or "").strip()
    folder = str(Path(path).parent)
    return text.replace(folder + "/", "").replace(folder, "") if folder not in ("", ".") else text


# What one read of a picture may make in memory (float32), whatever its header claims: a file of a few hundred bytes can
# declare a display window of 30000x30000, and the read places its pixels in an array of that size. 1 << 28 pixels is a
# 16384x16384 picture; 4 GB is such a picture's RGB in float32. Checked before any pixel is read (_fits).
PIXELS_MAX = 1 << 28
READ_MAX = 4 << 30


def _fits(path: str | Path, spec, target: Window, channels: int) -> None:
    """Refuse (E-IMAGE-TOOBIG) a read whose data window or target window would be more than PIXELS_MAX pixels, or
    more than READ_MAX bytes over `channels` float32 channels."""
    for w, h in ((spec.width, spec.height), (target[2], target[3])):
        if w * h > PIXELS_MAX or w * h * channels * 4 > READ_MAX:
            raise FileProblem(Msg("E-IMAGE-TOOBIG", file=Path(path).name, width=w, height=h, channels=channels,
                                  pixels=PIXELS_MAX / 1e6, gb=READ_MAX / (1 << 30)))


@contextmanager
def _utf8_names(path: str | Path):
    """Around a read of a file's part, view and channel names: one that is not UTF-8 (GBK from software on a Chinese
    system, say) fails the read with E-IMAGE-NAMEENCODING naming the file. OpenImageIO's Python binding decodes
    them strictly and raises UnicodeDecodeError; these names are what ports and connections go by, so none is guessed
    at by replacing its bytes (unlike the header texts nothing reads, header_attributes). Used where the names are
    read: _parts (every reader's channel names) and views."""
    try:
        yield
    except UnicodeDecodeError as exc:
        raise FileProblem(Msg("E-IMAGE-NAMEENCODING", file=Path(path).name)) from exc


def _open(path: str | Path):
    """An open OpenImageIO input for `path`; close it when done (a file that cannot be opened raises)."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:  # the file cannot be opened (as opposed to opened with no pixels to read)
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    return inp


def _rgb_names(names: list[str], path: str | Path = "") -> list[str]:
    """The three channels read_rgb returns, from a file's channel names: R, G, B by name; else those of a colour
    layer (rgba.R …, the beauty of a multi-part EXR whose first part is another pass: its channels are prefixed with
    its part name, _parts); else, in a file of plain channel names only, its first three, or its one channel three
    times (a grey picture). A file whose channels are all in named layers and none of them R G B has no colour:
    E-IMAGE-NOCHANNEL, never a depth or a normal read as red."""
    if all(c in names for c in "RGB"):
        return ["R", "G", "B"]
    layers = [n.rsplit(".", 1)[0] for n in names if n.endswith(".R")]
    colour = sorted((l for l in layers if all(f"{l}.{c}" in names for c in "GB")),
                    key=lambda l: (l.lower() not in ("rgba", "rgb", "beauty"), names.index(f"{l}.R")))
    if colour:
        return [f"{colour[0]}.{c}" for c in "RGB"]
    if any("." in n for n in names):
        raise FileProblem(Msg("E-IMAGE-NOCHANNEL", file=Path(path).name, channels="R G B"))
    return names[:3] if len(names) >= 3 else names[:1] * 3


def read_rgb(path: str | Path, box: tuple | None = None) -> np.ndarray:
    """Read the picture as float32 HxWx3: channels R, G, B by name (a multi-layer EXR may hold other layers alongside
    them). A file without them yields its first three channels; a single-channel file repeats that channel three
    times. `box`: as in read_named. The file is opened once (its channel names and its pixels from the same open)."""
    inp = _open(path)
    try:
        parts = _parts(inp, path=path)
        return _read_block(inp, path, parts, _rgb_names([n for names in parts for n in names if n is not None], path), box)
    finally:
        inp.close()


ALPHA_NAMES = ("A", "alpha")  # the picture's own alpha: an unprefixed A (a PNG's fourth channel, an EXR's rgba layer)


def has_alpha(path: str | Path) -> bool:
    """Return whether the picture has its own alpha channel."""
    return any(n in ALPHA_NAMES for n in channel_names(path))


def read_rgba(path: str | Path, box: tuple | None = None) -> np.ndarray:
    """Read the picture as float32 HxWx4: R, G, B premultiplied by A, then A (1 when the file has no alpha). One open
    of the file, one read of its colour and its alpha."""
    inp = _open(path)
    try:
        parts = _parts(inp, path=path)
        names = [n for part in parts for n in part if n is not None]
        pick = _rgb_names(names, path)
        layer = pick[0].rsplit(".", 1)[0] + "." if "." in pick[0] else ""  # the alpha of the same layer as the colour
        a = next((n for n in (f"{layer}{x}" for x in ALPHA_NAMES) if n in names), None)
        block = _read_block(inp, path, parts, pick + [a] if a else pick, box)
    finally:
        inp.close()
    if a is None:
        block = np.concatenate([block, np.ones((*block.shape[:2], 1), np.float32)], axis=-1)
    return block.astype(np.float32, copy=False)


def read_picture(path: str | Path, alpha: bool, box: tuple | None = None) -> np.ndarray:
    """Read the picture with alpha (HxWx4, premultiplied) when `alpha` is set, otherwise its colour as stored (HxWx3);
    covers the display window, or `box` as in read_named."""
    return read_rgba(path, box) if alpha else read_rgb(path, box)


def unpremultiply(rgba: np.ndarray) -> np.ndarray:
    """Convert premultiplied HxWx4 to straight colour HxWx3 (where alpha is 0 the colour is left unchanged, i.e. black)."""
    a = rgba[..., 3:4]
    return np.where(a > 0, rgba[..., :3] / np.maximum(a, 1e-8), rgba[..., :3]).astype(np.float32)


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


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def is_rgb_png(path: str | Path) -> bool:
    """Return whether the file is a PNG of three samples per pixel, 8 or 16 bit: colour type 2 in its IHDR chunk, not
    grey, grey + alpha, palette or RGBA. Read from the file's first 26 bytes (signature, IHDR length and name, width,
    height, bit depth, colour type), nothing decoded: payloads.worker_ready asks it of every frame of a plate."""
    with open(path, "rb") as f:
        head = f.read(26)
    return len(head) == 26 and head[:8] == PNG_SIGNATURE and head[12:16] == b"IHDR" and head[24] in (8, 16) and head[25] == 2


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
        with _utf8_names(path):
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
    """The view a channel of a single-part multi-view EXR belongs to, by OpenEXR's rule (ImfMultiView
    viewFromChannelName): split at the dots, a channel whose second-to-last segment is a view name belongs to that view,
    any other to the first view (the hero view). So left.R, forward.left.u and whitebarmask.left.mask are the left
    eye's, disparityL.x and Z the hero view's. Looking at the first segment alone would put forward.left.u in the hero
    view."""
    parts = channel.split(".")
    return parts[-2] if len(parts) >= 2 and parts[-2] in multi_view else multi_view[0]


def _without_view(channel: str, view: str) -> str:
    """The channel's name within its view, the view segment taken out: forward.left.u -> forward.u, left.R -> R."""
    parts = channel.split(".")
    if len(parts) >= 2 and parts[-2] == view:
        del parts[-2]
    return ".".join(parts)


def _parts(inp, view: str | None = None, path: str | Path = "") -> list[list[str | None]]:
    """Return the channel names of every part of an open file. A multi-part EXR may store one layer per part:
    unprefixed channels of any part other than the first take the part name as their layer ("depth" part: Z ->
    depth.Z). When a part's channel names collide with those of an earlier part (a stereo pair: forward_left and
    forward_right both hold forward.u), every channel of that part is prefixed with the part name
    (forward_right.forward.u), so all channels across parts are distinct. `view` (stereo files): only that view's
    channels are returned, with None in place of every other channel (so each name's position still matches its
    channel index); the first kept part is unprefixed and part names drop the view suffix (depth.right -> depth), so
    each view reads as an independent picture. `path`: the file, named when a name is not UTF-8 (_utf8_names)."""
    with _utf8_names(path):
        return _part_names(inp, view)


def _part_names(inp, view: str | None) -> list[list[str | None]]:
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
            if not own and len(together) > 1:  # one part holds every view: this view's channels, by OpenEXR's rule
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
        return [n for names in _parts(inp, view, path) for n in names if n is not None]
    finally:
        inp.close()


# the header texts the program reads (data/layers.py describe_file): what this project writes (lab2shot:layers) and
# Cryptomatte's names and manifests (a layer's name is a port's)
READ_ATTRIBUTES = ("lab2shot:", "cryptomatte/")


def header_attributes(path: str | Path) -> dict[str, str]:
    """Return the text attributes in an image file's header (first part): Cryptomatte manifests, lab2shot:* ...

    The header holds texts of every kind: a camera's EXIF (ImageDescription, Artist, Make), a PNG's tEXt, a JPEG
    comment, an EXR's owner and comments, in whatever encoding the software that wrote them used, GBK from a Chinese
    Windows for one. OpenImageIO's Python binding decodes each strictly (UnicodeDecodeError on a byte that is not
    UTF-8), and nothing here reads those, so one that is not UTF-8 is left out instead of failing the whole file (as
    open_video does for a video's tags, io/sources.py). One the program does read (READ_ATTRIBUTES) is not guessed at:
    the file is refused naming it (E-IMAGE-ATTRENCODING)."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FileProblem(Msg("E-IMAGE-OPEN", file=Path(path).name, detail=said(path, oiio.geterror())))
    try:
        out: dict[str, str] = {}
        for a in inp.spec().extra_attribs:
            try:
                name = a.name
            except UnicodeDecodeError:  # a name that is not UTF-8 is none of READ_ATTRIBUTES (all ASCII)
                continue
            try:
                value = a.value
            except UnicodeDecodeError as exc:
                if name.startswith(READ_ATTRIBUTES):
                    raise FileProblem(Msg("E-IMAGE-ATTRENCODING", file=Path(path).name, name=name)) from exc
                continue
            if isinstance(value, str):
                out[name] = value
        return out
    finally:
        inp.close()


def read_named(path: str | Path, names: list[str] | None = None, view: str | None = None,
               box: tuple | None = None) -> dict[str, np.ndarray]:
    """Read channels by name (EXR files store channels by name, in no fixed order): {"Z": [H,W], "A": [H,W]}. Reads
    every channel of every part, or only `names`, so one layer of a render with dozens of channels can be read alone.
    `box` (x, y, w, h) is relative to the display window's top-left corner, as in a packet's data window
    (data/windows.py): every channel is returned at that size, with the file's pixels in place and 0 elsewhere. None
    selects the display window (the picture's format, as used by nodes operating on the plate frame)."""
    inp = _open(path)
    try:
        return _read(inp, path, _parts(inp, view, path), names, box)
    finally:
        inp.close()


def _target(inp, box: tuple | None) -> Window:
    """What read_named returns of an open file, in file coordinates: the display window of its first part (that of
    _windows), or `box` placed in it."""
    inp.seek_subimage(0, 0)
    spec = inp.spec()
    full = (spec.full_x, spec.full_y, spec.full_width, spec.full_height)
    return full if box is None else (full[0] + int(box[0]), full[1] + int(box[1]), int(box[2]), int(box[3]))


def read_stack(path: str | Path, pick, box: tuple | None = None) -> tuple[np.ndarray, list[str]]:
    """Channels of one file as a float32 [H, W, C] array in a single open: `pick(names)` chooses them (a list, a name
    may repeat) from every channel name the file has (channel_names); returns the array and the names it chose. The
    values are read_named's (`box` as there), without a copy per channel when they lie side by side in the file."""
    inp = _open(path)
    try:
        parts = _parts(inp, path=path)
        chosen = pick([n for part in parts for n in part if n is not None])
        return _read_block(inp, path, parts, chosen, box), chosen
    finally:
        inp.close()


def _read(inp, path: str | Path, parts: list[list[str | None]], names: list[str] | None,
          box: tuple | None) -> dict[str, np.ndarray]:
    """read_named on an open file whose parts' channel names (`_parts`) are known: each part holding a wanted channel
    is read once, over the span of its wanted channels."""
    target = _target(inp, box)
    out: dict[str, np.ndarray] = {}
    for sub, channels in enumerate(parts):
        wanted = [i for i, n in enumerate(channels) if n is not None and (names is None or n in names)]
        if not wanted:
            continue
        lo, hi = wanted[0], wanted[-1] + 1  # read the whole span at once (a layer's channels are contiguous)
        inp.seek_subimage(sub, 0)
        _fits(path, inp.spec(), target, hi - lo)
        pixels = inp.read_image(sub, 0, lo, hi, oiio.FLOAT)
        if pixels is None:
            raise FileProblem(Msg("E-IMAGE-READ", file=Path(path).name, detail=said(path, inp.geterror())))
        data = np.asarray(pixels, np.float32).reshape(pixels.shape[0], pixels.shape[1], hi - lo)
        inp.seek_subimage(sub, 0)
        data = _placed(data, inp.spec(), target)
        out.update({channels[i]: data[..., i - lo] for i in wanted})
    missing = [n for n in names or () if n not in out]
    if missing:
        raise FileProblem(Msg("E-IMAGE-NOCHANNEL", file=Path(path).name, channels=", ".join(missing)))
    return out


def _read_block(inp, path: str | Path, parts: list[list[str | None]], names: list[str],
                box: tuple | None) -> np.ndarray:
    """`names` (a name may repeat) of an open file as one float32 [H, W, len(names)] array, the values read_named gives.
    When they are one part's channels in the file's own order (R G B, R G B A: nearly every picture and map) that span
    is read and returned as it is, without taking it apart into channels and stacking them again."""
    for sub, channels in enumerate(parts):
        if names and names[0] in channels:
            lo = channels.index(names[0])
            if channels[lo:lo + len(names)] == names:
                target = _target(inp, box)
                inp.seek_subimage(sub, 0)
                _fits(path, inp.spec(), target, len(names))
                pixels = inp.read_image(sub, 0, lo, lo + len(names), oiio.FLOAT)
                if pixels is None:
                    raise FileProblem(Msg("E-IMAGE-READ", file=Path(path).name, detail=said(path, inp.geterror())))
                data = np.asarray(pixels, np.float32).reshape(pixels.shape[0], pixels.shape[1], len(names))
                inp.seek_subimage(sub, 0)
                return _placed(data, inp.spec(), target)
            break  # every name is in one part only (_parts): the others need not be looked at
    named = _read(inp, path, parts, list(dict.fromkeys(names)), box)
    return np.stack([named[n] for n in names], axis=-1)


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
