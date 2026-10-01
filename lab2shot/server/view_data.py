"""What the 3D viewer is sent: 3D data in the form a DCC gets it, so a shot crosses a slow network (the site may be
shared through a tunnel) as a few megabytes instead of hundreds, and the viewer still shows exactly what the data holds
(what is shown must be the data itself; any compression is lossless, and any thinning is reported to the user):

- a 蒙皮角色 as its bind-pose mesh, skin weights and blend-shape offsets once and, per frame, its joints' transforms
  and blend-shape weights: the browser's graphics card skins it with the same arithmetic as the server's evaluation
  (webui/src/model/viewFormat.ts). A mesh with more than four joints per vertex (what a graphics card takes) comes
  evaluated per frame instead, never cut down;
- a 模型 once in its own space, placed per frame by its transform; only a deforming one (a point cache) per frame;
- a set of 三维曲线 as how many points each curve has and its points, once when it never changes, per frame when it does;
  the viewer draws it while its points are within the display budget and otherwise says so and draws its box alone,
  never thinned (webui/src/view/curves3d.tsx);
- a point cloud, a still model, a still camera once, whatever the shot's length;
- a point cloud made from a depth map and a camera (深度转点云) as the depth map (float32, the pixels it dropped as
  NaN) and its colours as bytes, with the camera per frame once: the viewer's graphics card rebuilds the points with the
  same float32 arithmetic (viewFormat.ts gridPoints). Every frame is rebuilt here first and compared with the cloud's
  own points; a frame that does not give them (within GRID_TOLERANCE: float32's own rounding) goes as its points;
- nothing rounded: every point sent is float32 as the evaluation gives it (only a point cloud over the 「点云上限」 is
  thinned, and the viewer says so: see below). Compression is lossless only: the
  bytes of each float array sorted by significance (all first bytes, then all second ...), a point cache's frames as
  the bit differences (XOR) from the frame before, then gzip. Colours that are all whole 255ths within float32
  precision (from 8-bit pictures; colours() says why "within") go as those bytes;
- in parts: the base (everything that is not per frame; its large arrays in parts of their own), then the per-frame
  samples in chunks of frames, read from the scene only when asked for: the viewer asks for the chunk of the frame it
  is on first and keeps only as many chunks as it has room for.

The description (JSON, gzipped on the wire) gives every item's frames as runs ([first, count] each), every part's URL
with its content's hash (a part may be kept by the browser permanently) and
every base array as {"part", "o": byte offset, "n": elements, "t": type, "e": encoding}. A chunk part starts with its
own description: a 4-byte length, then JSON {"pieces": [{"item": "models/0", "array": "points", "samples": [first,
end], "sizes": [elements per sample], "t", "e", "o"}]}, then the data."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..data.camera import CameraSamples
from ..errors import Invalid, MessageError, NotFound, message_of
from ..messages import Msg
from ..text import decimal

# Colours as bytes within float32 precision, depth clouds as grids, distance maps as grids with a ramp, a
# per-frame cloud's colours once in the base, fixed-count clouds as XOR differences. The chunk addresses carry it
# (v=), so bytes the browser kept (immutable) or the disk kept (store_chunk) under the old format are never read as new.
VIEW_FORMAT = 6
GRID_TOLERANCE = (1e-3, 1e-5)  # a depth cloud rebuilt as the viewer does may differ from its points by this: cm + relative
BASE_MAX = 512 * 1024  # a base array bigger than this gets a part of its own, so the small things come first
# per-frame samples in one chunk part (before gzip), about. Several frames a chunk: a whole shot is fetched (the viewer
# caches it all, webui/src/view/scene.ts), and at 4 MB a frame of a dense cloud was a
# chunk of its own, one request per frame. 16 MB is 8-13 frames of a 640×480 depth cloud, and at most ~4 frames of one at
# the 「点云上限」 (a chunk made on request is still well within the view worker's ANSWER_S).
CHUNK_BYTES = 16_000_000
KEEP_BYTES = 512 * 2**20  # gzipped chunk parts kept in this process, for every view together
# gzip level of a chunk made on request, which the viewer waits for: on these arrays (float32 byte planes, colour
# bytes) level 1 is within 3% of level 5 in size and 25-35% faster
LIVE_GZIP = 1

GRID_BYTES_PER_CELL = 7  # about 7 bytes sent per grid cell: a float32 depth (4) plus three colour bytes (the same figure as PerFrame's estimate)
CLOUD_BYTES_PER_POINT = 15  # about 15 bytes sent per point: three float32 coordinates plus three colour bytes (the same figure as PerFrame's estimate)

# A point cloud over the 「点云上限」 (setting `view.points_max_mb`, editable by administrators and effective at once) is
# thinned with a fixed step; there is no second tier and no switch. Only the copy sent to the viewer is thinned: the
# kept points' coordinates are bit-identical (`_thin` / `drop_points` take the original values), and computation,
# cache and delivered files keep every point. The amount thinned must be visible to the user: the viewer's notice area
# always shows 「显示了 N / 共 M 点」 (N-VIEW-CLOUDPROXY), so `every` / `proxy` are sent with the description.
#
# No lossy compression such as zeroing low mantissa bits: its savings apply to data already thinned to within the
# limit, so they are bounded, while 3D depth is centimetres that artists measure distances with, and nothing could
# tell the user that precision was reduced.

TYPES = {np.dtype(np.float32): "f32", np.dtype(np.uint32): "u32", np.dtype(np.uint16): "u16", np.dtype(np.uint8): "u8"}


# ------------------------------------------------------------------ lossless packing


def pack(a: np.ndarray, xor_size: int = 0) -> tuple[bytes, str]:
    """An array's bytes as sent, and how they were packed: float32 as byte planes ("s"), a run of equal-sized samples
    (`xor_size` elements each) first as the bit differences from the sample before ("x"); integers as they are ("")."""
    a = np.ascontiguousarray(a).reshape(-1)
    if a.dtype != np.float32:
        return a.tobytes(), ""
    bits = a.view(np.uint32)
    code = "s"
    if xor_size and bits.size > xor_size:
        rows = bits.reshape(-1, xor_size)
        diff = rows.copy()
        diff[1:] ^= rows[:-1]
        bits, code = diff.reshape(-1), "x"
    return bits.view(np.uint8).reshape(-1, 4).T.tobytes(), code


COLOUR_WORDS = ((255, np.uint8), (65535, np.uint16))  # whole steps of an 8-bit or a 16-bit picture


def colours(c: np.ndarray) -> np.ndarray:
    """Display colours as sent, the same numbers in fewer bytes: bytes when every value equals k/255 within float32
    precision (an 8-bit picture's), 16-bit words when k/65535 (a 16-bit picture's), else float32. The viewer reads k
    back as k/255 or k/65535.

    「在 float32 精度内等于 k/255」：与 k/255 相差不超过 1 个 float32 精度单位（ulp）。8-bit 画面的颜色经读图 / 显示
    转换按 float32 算出（例如 k × (1/255)），与 f64 算出的 k/255 常差 1 ulp（实测只有 58% 逐位相同，最大偏差 1.9e-5），
    这是浮点舍入，不是颜色的差别；逐位比对会让整段颜色按 float32 发，每点多 9 字节。"""
    c32 = np.ascontiguousarray(np.asarray(c, np.float32).reshape(-1))
    for top, kind in COLOUR_WORDS:
        k = np.rint(c32.astype(np.float64) * top)
        if not c32.size or k.min() < 0 or k.max() > top:
            continue
        exact = (k / top).astype(np.float32)
        # 非负 float32 的位模式按整数单调：两者位模式之差就是相差几个 ulp
        if np.all(np.abs(exact.view(np.int32).astype(np.int64) - c32.view(np.int32).astype(np.int64)) <= 1):
            return k.astype(kind)
    return c32


def rows(matrices: np.ndarray, keep: int = 4) -> np.ndarray:
    """Matrices [..., 4, 4] (column vectors) as their first `keep` rows, row after row, float32."""
    return np.ascontiguousarray(np.asarray(matrices, np.float64)[..., :keep, :], np.float32).reshape(-1)


def faces(counts, indices) -> np.ndarray:
    from ..data.evaluate import triangulate

    tris = triangulate(np.asarray(counts, np.int64), np.asarray(indices, np.int64)).reshape(-1)
    return tris.astype(np.uint16 if tris.max(initial=0) < 65535 else np.uint32)


def indices(a) -> np.ndarray:
    a = np.asarray(a, np.int64).reshape(-1)
    return a.astype(np.uint16 if a.max(initial=0) < 65535 else np.uint32)


# ------------------------------------------------------------------ a view: its base now, its chunks when asked


@dataclass
class PerFrame:
    """Something with a sample per frame, read when its chunk is asked for: `read(frame)` gives its arrays at that
    frame; `xor`: its samples are the same size (a point cache), sent as differences."""

    ref: str  # "models/0", "clouds/2"
    frames: list[int]
    read: Callable[[int], dict[str, np.ndarray]]
    bytes_per_frame: int
    xor: bool = False


@dataclass
class View:
    description: dict
    parts: dict[str, tuple[bytes, int, str]] = field(default_factory=dict)  # built: name -> (gzipped, raw size, hash)
    chunks: list[tuple[int, int]] = field(default_factory=list)
    per_frame: list[PerFrame] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    token: str = field(default_factory=lambda: uuid.uuid4().hex)  # this view's chunks among those kept
    source: object = None  # what the readers read from, kept open as long as the view (a USD stage)

    def json(self, url: str) -> str:
        """The description, each part with its URL (`url` with {part}) and, once built, its size."""
        def at(name: str, query: str) -> str:
            u = url.format(part=name)
            return u + ("&" if "?" in u else "?") + query

        d = dict(self.description)
        parts = {name: {"url": at(name, f"h={h}"), "bytes": len(gz), "raw": raw} for name, (gz, raw, h) in self.parts.items()}
        # Chunk addresses include the 「点云上限」 (`mb=`): a chunk's bytes are thinned by it, so changing the limit
        # changes the bytes of the same chunk. The browser caches chunks permanently as content addresses (`respond`
        # marks them immutable); without the limit in the address, raising it would leave the browser showing its
        # cached, thinned bytes while the description says nothing was thinned.
        mb = _budget(POINTS_LIMIT) // 2**20
        for k, (lo, hi) in enumerate(self.chunks):
            parts[f"c{k}"] = {"url": at(f"c{k}", f"v={VIEW_FORMAT}&f={lo}-{hi}&mb={mb}"), "frames": [lo, hi]}  # its bytes: kept permanently
        d["parts"] = parts
        for kind in ("models", "characters", "clouds", "curves", "cameras"):  # thousands of props need not spell 250 frames each
            d[kind] = [{**it, "frames": runs(it["frames"])} for it in d.get(kind, [])]
        return json.dumps(d, separators=(",", ":"))

    def part(self, name: str) -> bytes:
        """A part, gzipped: the base's at once, a chunk read from the scene the first time it is asked for. One address
        is one sequence of bytes: point thinning is done in the readers, so there is no second (preview) variant."""
        if name in self.parts:
            return self.parts[name][0]  # the base: small, and not per-frame display data
        if not name.startswith("c") or decimal(name[1:]) is None or decimal(name[1:]) >= len(self.chunks):
            raise KeyError(name)
        key = (self.token, name)
        with _kept_lock:
            if key in _kept:
                _kept.move_to_end(key)
                return _kept[key]
        with self.lock:  # two requests for the same chunk: read once
            with _kept_lock:
                if key in _kept:
                    return _kept[key]
            data = gzip.compress(self._chunk(*self.chunks[int(name[1:])]), LIVE_GZIP)
        with _kept_lock:
            _kept[key] = data
            while sum(len(v) for v in _kept.values()) > KEEP_BYTES and len(_kept) > 1:
                _kept.popitem(last=False)
        return data

    def _chunk(self, lo: int, hi: int) -> bytes:
        pieces, blobs, offset = [], [], 0
        for pf in self.per_frame:
            idx = [i for i, f in enumerate(pf.frames) if lo <= f <= hi]
            if not idx:
                continue
            # Thinning is already done in `pf.read`: what arrives here is the copy that is sent
            samples = [pf.read(pf.frames[i]) for i in idx]
            for name in dict.fromkeys(n for sample in samples for n in sample):  # a sample without it: an empty one
                given = [np.asarray(sample[name]) for sample in samples if name in sample]
                dtype = given[0].dtype if all(a.dtype == given[0].dtype for a in given) else np.dtype(np.float32)

                def sent(a: np.ndarray) -> np.ndarray:  # colour words among float colours: the same numbers, k/255 or k/65535
                    words = dict((np.dtype(kind), top) for top, kind in COLOUR_WORDS)
                    return a if a.dtype == dtype else (a.astype(np.float64) / words[a.dtype] if a.dtype in words else a).astype(dtype)

                arrays = [sent(np.asarray(sample[name])) if name in sample else np.zeros(0, dtype) for sample in samples]
                sizes = [int(a.size) for a in arrays]
                same = pf.xor and len(set(sizes)) == 1
                data, code = pack(np.concatenate([a.reshape(-1).astype(dtype, copy=False) for a in arrays]), sizes[0] if same else 0)
                pad = (-offset) % 4
                blobs.append(b"\0" * pad + data)
                offset += pad
                pieces.append({"item": pf.ref, "array": name, "samples": [idx[0], idx[-1] + 1], "sizes": sizes,
                               "t": TYPES[np.dtype(dtype)], "e": code, "o": offset})
                offset += len(data)
        head = json.dumps({"pieces": pieces}).encode()
        head += b" " * ((-len(head)) % 4)
        return len(head).to_bytes(4, "little") + head + b"".join(blobs)


# ------------------------------------------------------------------ chunks made ahead and kept on disk
#
# A view's chunks are made once, in the background (view_worker.prebuild_later: when the node that computed the packet
# is through, farm/queue.py _proxies_of, and when a view is first described), and kept in the packet's own folder
# beside its image proxies (_view/). The web server then answers a chunk by sending that file (server/packets.py
# scene_part, server/view.py points_part): no view worker lane, nothing made on request. A chunk not there yet is made
# on request as before. The file's name says everything its bytes depend on: the generation of every packet the view
# reads (a recompute swaps in a new folder anyway; the name also keeps a late writer of the old generation from being
# read as the new one), the view format, the 「点云上限」 and the chunk's frames — exactly what the chunk's address says,
# so one address is still one sequence of bytes.

STORED_GZIP = 5  # made ahead: nobody waits for it, and level 5 is a little smaller than the live path's level 1


def _stamp(created) -> str:
    return re.sub(r"[^0-9A-Za-z]", "", str(created or "0")) or "0"


def chunk_file(how: tuple, stamps: tuple, lo: int, hi: int, mb: int) -> Path:
    """Where the chunk (frames lo..hi, made under a 「点云上限」 of `mb`) of the view `how` is kept: in the scene packet's
    folder, or the depth / position map's (a point preview: one set of files per camera it is seen from)."""
    from ..data.packet import packet_dir

    kind = "scene" if how[0] == "scene" else f"points.{how[2] or 'none'}"
    stamp = "-".join(_stamp(s) for s in stamps)
    return packet_dir(how[1]) / "_view" / kind / f"{stamp}.v{VIEW_FORMAT}.mb{int(mb)}.f{int(lo)}_{int(hi)}.gz"


_FRAMES = re.compile(r"^(-?[0-9]+)-(-?[0-9]+)$")


def stored_chunk(how: tuple, stamps: tuple, part: str, query) -> Path | None:
    """The file of the chunk a page asks for (its address: part c{k}, v=format, f=lo-hi, mb=limit), when it has been
    made; None otherwise (it is then made on request)."""
    if not (part.startswith("c") and decimal(part[1:]) is not None) or query.get("v") != str(VIEW_FORMAT):
        return None
    got, mb = _FRAMES.match(query.get("f") or ""), query.get("mb") or ""
    if got is None or decimal(mb) is None:
        return None
    path = chunk_file(how, stamps, int(got[1]), int(got[2]), decimal(mb))
    return path if path.is_file() else None


def store_chunk(view: View, how: tuple, stamps: tuple, k: int) -> int:
    """Make chunk `k` of `view` and keep it on disk (chunk_file), unless it is there already; the bytes written."""
    from ..view.encode import write_once

    lo, hi = view.chunks[k]
    path = chunk_file(how, stamps, lo, hi, _budget(POINTS_LIMIT) // 2**20)
    if path.is_file():
        return 0
    data = gzip.compress(view._chunk(lo, hi), STORED_GZIP)
    write_once(path, lambda part: part.write_bytes(data))
    return len(data)


def runs(frames: list[int]) -> list[list[int]]:
    """Frames as runs of consecutive ones, [first, count] each (webui/src/model/viewFormat.ts: expandRuns)."""
    out: list[list[int]] = []
    for f in frames:
        if out and out[-1][0] + out[-1][1] == f:
            out[-1][1] += 1
        else:
            out.append([f, 1])
    return out


_kept: OrderedDict = OrderedDict()
_kept_lock = threading.Lock()


class Base:
    """The base arrays, packed into parts; each becomes a descriptor."""

    def __init__(self) -> None:
        self.parts: dict[str, bytearray] = {"base": bytearray()}
        self.own = 0

    def array(self, arr, part: str = "base") -> dict:
        a = np.ascontiguousarray(arr).reshape(-1)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        data, code = pack(a)
        if part == "base" and len(data) > BASE_MAX:
            part, self.own = f"a{self.own}", self.own + 1
        buf = self.parts.setdefault(part, bytearray())
        buf += b"\0" * ((-len(buf)) % 4)
        d = {"part": part, "o": len(buf), "n": int(a.size), "t": TYPES[a.dtype], "e": code}
        buf += data
        return d

    def built(self) -> dict[str, tuple[bytes, int, str]]:
        return {name: (gzip.compress(bytes(buf), 5), len(buf), hashlib.sha1(bytes(buf)).hexdigest()[:12])
                for name, buf in self.parts.items() if buf or name == "base"}


def chunk_plan(frames: list[int], bytes_per_frame: int, one_frame_chunks: bool = False) -> list[tuple[int, int]]:
    """The shot's frames in runs of about CHUNK_BYTES of per-frame samples: [(first frame, last frame), ...].

    `one_frame_chunks`: one frame per chunk, for a partial result (partial_points_view). When chunks are filled by
    bytes, their boundaries follow the total frame count and frame size, which keep changing while computing; any
    change of boundaries changes every chunk address and invalidates everything the browser holds. A one-frame chunk's
    address depends only on the frame number and never changes."""
    if not frames:
        return []
    if one_frame_chunks:
        return [(f, f) for f in frames]
    if bytes_per_frame <= 0:
        return []
    per = max(1, CHUNK_BYTES // bytes_per_frame)
    return [(frames[i], frames[min(i + per, len(frames)) - 1]) for i in range(0, len(frames), per)]


def _text(v) -> str:
    v = np.asarray(v)
    x = v.item() if v.ndim == 0 else v.reshape(-1)[0]
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)  # a label to show: never fails the view


def _uv(b: Base, mesh: dict) -> dict | None:
    """A mesh's UVs as they are (face-varying) with, per corner of its triangles (the faces' fan), which UV it takes;
    in the "uv" part, asked for only when the UV checker is on."""
    from ..data.evaluate import triangulate

    if "uv" not in mesh or "uv_indices" not in mesh:
        return None
    uv_idx = np.asarray(mesh["uv_indices"], np.int64)
    corners = triangulate(np.asarray(mesh["counts"], np.int64), np.arange(len(uv_idx))).reshape(-1)
    return {"values": b.array(np.asarray(mesh["uv"], np.float32), "uv"), "indices": b.array(indices(uv_idx[corners]), "uv")}


def _cloud_arrays(points, cols, widths, every: int = 1) -> dict[str, np.ndarray]:
    """One sample of a cloud as it is sent. `every` > 1: over the 「点云上限」, every `every`-th point is kept (drop_points)
    first, and only the kept points are converted and have their colours checked (colours()): at 4K that check on every
    point would cost about 0.4 s a frame before thinning threw most of them away."""
    kept = drop_points({"points": np.asarray(points), "colors": np.asarray(cols), "widths": np.asarray(widths)}, every)
    points, cols, widths = kept["points"], kept["colors"], kept["widths"]
    out = {"points": np.asarray(points, np.float32), "colors": colours(cols)}
    w = np.asarray(widths, np.float32).reshape(-1)
    if w.size and not np.all(w == w[0]):
        out["widths"] = w  # a size per point
    return out


def _curve_arrays(vertex_counts, points, cols, widths) -> dict[str, np.ndarray]:
    """One sample of a set of 三维曲线 as it is sent: how many points each curve has, the points, their colours, and
    the widths only when they differ (hair is thick at the root and thin at the tip; one width throughout is a number
    in the description, exactly as a point cloud's is)."""
    out = {"vertex_counts": indices(vertex_counts), "points": np.asarray(points, np.float32), "colors": colours(cols)}
    w = np.asarray(widths, np.float32).reshape(-1)
    if w.size and not np.all(w == w[0]):
        out["widths"] = w
    return out


def encode(frames: list[int], items: dict[str, list[dict]], resolution: tuple[int, int],
           lights: list[dict] | None = None, reader: Callable[[str, dict], Callable[[int], dict]] | None = None,
           one_frame_chunks: bool = False) -> View:
    """A scene's items (data/scene_arrays.py, with samples=False: per-frame things marked `per_frame`) as a view.
    `reader(kind, item)` gives the function that reads a per-frame item at a frame."""
    frames = [int(f) for f in frames]
    b = Base()
    per_frame: list[PerFrame] = []

    def frames_of(it) -> list[int]:
        return [int(f) for f in np.asarray(it["frames"]).reshape(-1)]

    models = []
    for i, it in enumerate(items.get("model", [])):
        pts = np.asarray(it["points"], np.float32)
        pts = pts.reshape(1, -1, 3) if pts.ndim == 2 else pts
        model = {"name": _text(it["name"]), "path": _text(it["path"]), "frames": frames_of(it), "world": b.array(rows(it["world"])),
                 "faces": b.array(faces(it["counts"], it["indices"])), "vertices": int(pts.shape[1]), "uv": _uv(b, it),
                 "per_frame": bool(it.get("per_frame", False))}
        if model["per_frame"]:
            per_frame.append(PerFrame(f"models/{i}", model["frames"], reader("model", it), pts.shape[1] * 12, xor=True))
        else:
            model["points"] = b.array(pts[0])
        models.append(model)

    characters = []
    for i, it in enumerate(items.get("character", [])):
        joints = len(np.asarray(it["joints"]).reshape(-1))
        meshes = []
        for m in it.get("meshes", []):
            idx = np.asarray(m["joint_indices"], np.int64)
            wts = np.asarray(m["joint_weights"], np.float32)
            if idx.shape[1] < 4:  # fewer influences: padded with nothing
                pad = 4 - idx.shape[1]
                idx, wts = np.pad(idx, ((0, 0), (0, pad))), np.pad(wts, ((0, 0), (0, pad)))
            mesh = {"name": _text(m["name"]), "faces": b.array(faces(m["counts"], m["indices"])), "vertices": int(len(m["points"])),
                    "points": b.array(np.asarray(m["points"], np.float32)), "uv": _uv(b, m), "shapes": [], "influences": int(idx.shape[1])}
            if idx.shape[1] == 4:
                mesh["joint_indices"] = b.array(idx.astype(np.uint8 if joints <= 256 else np.uint16))
                mesh["joint_weights"] = b.array(wts)
            else:  # more joints per vertex than a graphics card takes: the server's evaluation, frame by frame
                mesh["per_frame"] = True
                per_frame.append(PerFrame(f"characters/{i}/meshes/{len(meshes)}", frames_of(it),
                                          reader("skinned", {"path": it["path"], "mesh": m["name"]}), len(m["points"]) * 12, xor=True))
            if "shapes" in m and len(np.asarray(m["shapes"]).reshape(-1)):
                mesh["shapes"] = [str(s) for s in np.asarray(m["shapes"]).reshape(-1)]
                mesh["shape_offsets"] = b.array(np.asarray(m["shape_offsets"], np.float32))
                mesh["shape_weights"] = b.array(np.asarray(m["shape_weights"], np.float32))
            meshes.append(mesh)
        characters.append({"name": _text(it["name"]), "path": _text(it["path"]), "frames": frames_of(it),
                           **({"person": int(it["person"])} if it.get("person") is not None else {}),
                           "joints": [str(j) for j in np.asarray(it["joints"]).reshape(-1)],
                           "parents": [int(x) for x in np.asarray(it["parents"]).reshape(-1)],
                           "bind": b.array(rows(it["bind"])), "anim": b.array(rows(it["anim"], 3)), "meshes": meshes})

    clouds = []
    for i, it in enumerate(items.get("points", [])):
        w = np.asarray(it.get("widths", []), np.float32).reshape(-1)
        cloud = {"name": _text(it["name"]), "path": _text(it["path"]), "frames": frames_of(it), "world": b.array(rows(it["world"])),
                 "width": float(w[0]) if w.size else None, "per_frame": bool(it.get("per_frame", False)),
                 # a changing cloud that is not a depth map's pixels has a speed, so the viewer offers 着色 · 速度
                 # (webui/src/model/viewControls.ts). A per-frame cloud comes with its first sample only, so its counts
                 # cannot tell whether later frames keep the same points: the viewer measures that per sample
                 # (points3d.tsx speedsOf) and colours by 颜色 where they differ
                 "speed": bool(it.get("per_frame", False) and not it.get("depth_grid"))}
        if cloud["per_frame"] and it.get("depth_grid"):
            info, read = it["depth_grid"]
            cloud["grid"] = {**{k: info[k] for k in ("width", "height", "step", "gw", "gh")},
                             "focal": b.array(np.asarray(info["focal"], np.float32)), "cam": b.array(rows(info["cam"])),
                             "principal": b.array(np.asarray(info["principal"], np.float32).reshape(-1)),  # (cx, cy) per sample, pixels
                             "proxy": int(info.get("proxy", 1)),
                             # 距离图的色带（_distance_grid_view）：[近, 远] 的取值范围，网页着色器按它算颜色
                             **({"ramp": info["ramp"], "ramp_colour": info["ramp_colour"]} if "ramp" in info else {})}
            per_frame.append(PerFrame(f"clouds/{i}", cloud["frames"], read,
                                      info["gw"] * info["gh"] * info.get("bytes_per_cell", GRID_BYTES_PER_CELL), xor=True))
        elif cloud["per_frame"]:
            # The first frame's point count: chunk byte estimates use it, and so does the viewer's 「显示了 N / 共 M 点」
            # (viewFormat.ts CloudRef.count)
            count = max(int(np.asarray(it["counts"])[0]), 1)
            # Over the 「点云上限」 every `every`-th point is kept; 1: nothing is dropped. Thinning happens at read time,
            # and chunk byte estimates use the thinned count.
            every = _point_step(count)
            cloud["count"] = count
            if every > 1:
                cloud["every"] = every   # the viewer computes the displayed count from it (webui/src/view/kinds3d.ts)
            # the reader thins as it reads (_cloud_arrays `every`), before converting and checking colours
            read_points = reader("points", {**it, "every": every})
            # 颜色只发一次（点缓存：同一批点、颜色不变）：首帧的颜色放进基础数据（cloud "colors"），
            # 之后每一帧只有颜色与它逐字节相同时才不发；不同的帧照常带自己的颜色，网页有自己的就用自己的
            # （webui/src/view/scene.ts apply），所以不会有哪一帧画错颜色
            if frames_of(it):
                first = read_points(frames_of(it)[0]).get("colors")
                if first is not None and np.asarray(first).size:
                    first = np.asarray(first)
                    cloud["colors"] = b.array(first)
                    read_points = (lambda f, r=read_points, c0=first: {k: v for k, v in r(f).items()
                                                                       if not (k == "colors" and np.asarray(v).dtype == c0.dtype
                                                                               and np.array_equal(v, c0))})
            # xor：点数不变的逐帧点云（真正的点缓存，如 3D 跟踪点）位置按与上一帧的位差发，同一块里点数不一样的
            # 照常发（View._chunk 只在一块内各帧大小相同时才差分），所以对点数逐帧变的点云没有影响
            per_frame.append(PerFrame(f"clouds/{i}", cloud["frames"], read_points, -(-count // every) * CLOUD_BYTES_PER_POINT,
                                      xor=True))
        else:
            arrays = _cloud_arrays(it["points"], it.get("colors", np.full((len(np.asarray(it["points"])), 3), 0.7)), w)
            cloud.update({k: b.array(v) for k, v in arrays.items()})
            cloud["count"] = int(len(np.asarray(it["points"]).reshape(-1, 3)))
        clouds.append(cloud)

    curves = []
    for i, it in enumerate(items.get("curves", [])):
        w = np.asarray(it.get("widths", []), np.float32).reshape(-1)
        strands = np.asarray(it["curve_counts"], np.int64).reshape(-1)
        counted = np.asarray(it["counts"], np.int64).reshape(-1)
        curve = {"name": _text(it["name"]), "path": _text(it["path"]), "frames": frames_of(it), "world": b.array(rows(it["world"])),
                 "width": float(w[0]) if w.size else None, "per_frame": bool(it.get("per_frame", False)),
                 # 「N 条曲线 / M 个点」: the viewer compares the total point count with its display budget and, when over,
                 # draws only the bounding box without thinning
                 "strands": int(strands[0]) if strands.size else 0, "count": int(counted[0]) if counted.size else 0}
        if curve["per_frame"]:
            per_frame.append(PerFrame(f"curves/{i}", curve["frames"], reader("curves", it), max(int(counted[0]), 1) * 17))
        else:
            arrays = _curve_arrays(it["curve_vertex_counts"], it["points"],
                                   it.get("colors", np.full((curve["count"], 3), 0.7)), w)
            curve.update({k: b.array(v) for k, v in arrays.items()})
        curves.append(curve)

    cameras = []
    for it in items.get("camera", []):
        res = np.asarray(it["resolution"]).reshape(-1).tolist() if "resolution" in it else list(resolution)
        # Its distortion (model id; empty when none): when the 3D stage looks through this camera, the background must
        # be undistorted with its lens (`through` in server/packets.py frame); a camera with a distortion model cannot
        # use the original picture as its background
        lens = CameraSamples.lens_from_properties(json.loads(_text(it["properties"])))["lens"] if "properties" in it else {}
        cameras.append({"name": _text(it["name"]), "path": _text(it["path"]), "frames": frames_of(it),
                        "distortion": str((lens.get("distortion") or {}).get("model") or ""),
                        # its plate (io/usd.py PLATE): the picture the 3D stage places behind it when looking through
                        # it; empty when none is recorded
                        "plate": _text(it["plate"]) if "plate" in it else "",
                        "world": b.array(rows(it["world"])), "width": int(res[0]), "height": int(res[1]),
                        "focal_mm": b.array(np.asarray(it["focal_mm"], np.float32)),
                        "h_aperture_mm": b.array(np.asarray(it["h_aperture_mm"], np.float32)),
                        "v_aperture_mm": b.array(np.asarray(it["v_aperture_mm"], np.float32)),
                        # the lens centre off the picture's centre, mm, +x right +y up (data/camera.py center_mm): [F,2] or [1,2]
                        "center_mm": b.array(np.asarray(it["center_mm"], np.float32).reshape(-1, 2))})

    view = View({"format": VIEW_FORMAT, "frames": frames, "models": models, "characters": characters,
                 "clouds": clouds, "curves": curves, "cameras": cameras, "lights": lights or []}, b.built(),
                chunk_plan(frames, sum(pf.bytes_per_frame for pf in per_frame), one_frame_chunks), per_frame)
    return view


def scene_view(p) -> View:
    """A scene packet's view: its items as scene_arrays evaluates them for the writers, the per-frame ones read from
    the scene a chunk at a time."""
    from pxr import Usd, UsdGeom

    from ..data.evaluate import mesh_world_points
    from ..data.scene import open_scene
    from ..data.scene_arrays import cloud_sample, curve_sample, model_points, scene_arrays
    from ..io.usd import PERSON_ID, name_of
    from ..data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH

    size = (int(p.meta.get("width", DEFAULT_WIDTH)), int(p.meta.get("height", DEFAULT_HEIGHT)))
    if p.meta.get("empty"):  # nothing this time (「创建相机」 without a focal length): nothing to show
        return encode([], {}, size)
    frames = [int(f) for f in p.meta["frames"]]
    stage = open_scene([p])  # kept open by the readers below while the view is kept
    arrays = scene_arrays(p, samples=False, stage=stage)

    def reader(kind: str, item: dict) -> Callable[[int], dict]:
        if kind == "model":
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(_text(item["path"])))
            return lambda f: {"points": model_points(mesh, [f])[0]}
        if kind == "points":
            prim = stage.GetPrimAtPath(_text(item["path"]))
            return lambda f: _cloud_arrays(*cloud_sample(prim, f), every=int(item.get("every", 1)))
        if kind == "curves":
            prim = stage.GetPrimAtPath(_text(item["path"]))
            return lambda f: _curve_arrays(*curve_sample(prim, f))
        root, name = _text(item["path"]), _text(item["mesh"])  # a skinned mesh evaluated: its points in the world
        return lambda f: {"points": next(pts for path, pts in mesh_world_points(stage, Usd.TimeCode(f))
                                         if str(path).startswith(root) and name_of(stage.GetPrimAtPath(path)) == name).astype(np.float32)}

    for it in arrays.items.get("character", []):  # which person a character is (the boxes it was solved from), so the
        # 3D view colours it like that person's box in 2D (viewFormat.ts CharacterRef.person); a character that is not
        # one person (a prop, a hand-made rig) has none
        person = stage.GetPrimAtPath(_text(it["path"])).GetCustomDataByKey(PERSON_ID)
        if person is not None:
            it["person"] = int(person)
    for it in arrays.items.get("points", []):  # made from a depth map: sent as the depth map (checked frame by frame)
        if bool(it.get("per_frame", False)):
            it["depth_grid"] = depth_grid(stage.GetPrimAtPath(_text(it["path"])), [int(f) for f in np.asarray(it["frames"]).reshape(-1)])
    view = encode(frames, arrays.items, size, None, reader)
    view.source = stage  # the readers hold its prims, which do not keep it open
    return view


def grid_points(z: np.ndarray, idx: np.ndarray, gw: int, step: int, width: int, height: int, focal: float, cam: np.ndarray,
                principal=None) -> np.ndarray:
    """Points [M,3] of a depth cloud's kept pixels (`idx`: row-major places in the grid of every `step`-th pixel, `gw`
    wide; `z` their depths) rebuilt in float32 in the viewer's order of operations (viewFormat.ts gridPoints and the
    shader in pointShaders.tsx): a pinhole at `principal` ((cx, cy) pixels; the picture's centre when None, the same
    point the cook unprojects through, nodes/kit/unproject.py), pixel centres at +0.5, OpenCV to GL axes, then `cam`
    (camera-to-world, 4x4 with column vectors)."""
    f = np.float32
    c = (idx % gw).astype(f) * f(step)
    r = (idx // gw).astype(f) * f(step)
    z = np.asarray(z, f)
    cx, cy = (f(width) * f(0.5), f(height) * f(0.5)) if principal is None else (f(principal[0]), f(principal[1]))
    x = (c + f(0.5) - cx) / f(focal) * z
    y = (r + f(0.5) - cy) / f(focal) * z
    m = np.asarray(cam, f)
    vy, vz = -y, -z
    return np.stack([m[k, 0] * x + m[k, 1] * vy + m[k, 2] * vz + m[k, 3] for k in range(3)], 1)


def depth_grid(prim, frames: list[int]):
    """A cloud made from a depth map (its DEPTH_GRID customData) as the viewer is sent it: (the grid's description with
    the camera per frame, read(frame) -> that frame's arrays), or None (not made from one, or what it was made from is
    no longer in the cache). read() rebuilds the frame's points from the depth map as the viewer will and gives
    {"depth": the grid's depths, NaN where dropped, "grid_colors": the points' colours as colours() sends them} (or
    "grid_tint" for one colour) only when
    they are the cloud's own points, in its order; otherwise the points themselves."""
    from ..engine import Packet
    from ..data.camera import CameraSamples
    from ..data.payloads import DEPTH_GRID, image_files, read_map
    from ..data.maps import map_at
    from ..data.packet import packet_dir
    from ..data.scene_arrays import cloud_sample

    given = dict(prim.GetCustomDataByKey(DEPTH_GRID) or {}) if prim else {}
    if not given or not frames:
        return None
    if given.get("source") == "native":
        # The model's own 3D points (the family's native_points) are not a depth + pinhole unprojection: a grid rebuilt
        # by the browser would never match them, and the per-frame rebuild and comparison below would be wasted before
        # falling back to sending points. The points are sent directly, skipping that pass.
        return None

    def load(key: str):
        fp = given.get(key)
        return Packet.load(packet_dir(fp)) if fp and Packet.exists(packet_dir(fp)) else None

    try:
        depth, camera, mask, confidence = load("depth"), load("camera"), load("mask"), load("confidence")
        if depth is None or camera is None or (given.get("mask") and mask is None) or (given.get("confidence") and confidence is None):
            return None
        step, width, height = int(given["step"]), int(depth.meta["width"]), int(depth.meta["height"])
        gw, gh = len(range(0, width, step)), len(range(0, height, step))
        # Proxy display: below the 「点云上限」 every point is sent; above it, every `proxy`-th cell is kept in each
        # direction. The limit is applied to what is actually sent, depth plus colour (about 7 bytes per cell); a
        # 150-frame 1080p depth cloud sent in full is about 136 MB, too much for a site behind a traffic-billed tunnel.
        # The proxy must be visible to the user: the viewer's notice area always shows 「代理显示 · 每 N 点取 1」.
        full_step, full_gw, full_gh = step, gw, gh
        proxy = _proxy_step(gw * gh)
        if proxy > 1:
            step, gw, gh = step * proxy, len(range(0, gw, proxy)), len(range(0, gh, proxy))
        cam = CameraSamples.from_packet(camera, frames)
        focal, mats, principal = cam.focal_px(width), cam.cam_to_world, cam.principal_px(width)
        if given.get("space") == "camera":
            mats = np.repeat(np.eye(4)[None], len(frames), 0)
        files = image_files(depth)
    except Exception:  # noqa: BLE001 (anything it was made from unreadable: sent as its points)
        return None
    at = {f: i for i, f in enumerate(frames)}
    least = float(given.get("min_confidence", 0.5))

    def _cells_of(mine: np.ndarray, keep: np.ndarray, i: int) -> np.ndarray:
        """点云的点所在的格（行优先序号），按 grid_points 的逆运算投回：世界 → 相机（OpenCV 轴），像素中心 +0.5。
        找不到一一对应（越界、不递增、不在 keep 里）时返回空，调用方据此按点发。"""
        if not len(mine):
            return np.zeros(0, np.int64)
        m = np.linalg.inv(np.asarray(mats[i], np.float64))
        p = np.asarray(mine, np.float64) @ m[:3, :3].T + m[:3, 3]
        z = -p[:, 2]
        if not np.all(z > 0):
            return np.zeros(0, np.int64)
        f = float(focal[i])
        cx, cy = float(principal[i][0]), float(principal[i][1])  # principal_px：未写出主点时即画面中心
        c = np.rint((p[:, 0] / z * f + cx - 0.5) / full_step).astype(np.int64)
        r = np.rint((-p[:, 1] / z * f + cy - 0.5) / full_step).astype(np.int64)
        if c.min() < 0 or r.min() < 0 or c.max() >= full_gw or r.max() >= full_gh:
            return np.zeros(0, np.int64)
        cells = r * full_gw + c
        if np.any(np.diff(cells) <= 0) or not np.all(keep.reshape(-1)[cells]):
            return np.zeros(0, np.int64)
        return cells

    def read(frame: int) -> dict[str, np.ndarray]:
        points, cols, widths = cloud_sample(prim, frame)
        if frame not in files or (widths.size and not np.all(widths == widths[0])):
            return _cloud_arrays(points, cols, widths)
        z, alpha = read_map(files[frame])
        keep = alpha[::full_step, ::full_step] > 0
        if mask is not None and (m := map_at(mask, frame)) is not None:
            keep &= m[0][::full_step, ::full_step, 0] <= 0.5
        if confidence is not None and (sure := map_at(confidence, frame)) is not None:
            keep &= sure[0][::full_step, ::full_step, 0] >= least
        idx = np.flatnonzero(keep)
        mine = points.astype(np.float32)
        if keep.shape == (full_gh, full_gw) and len(idx) != len(mine):
            # 「深度转点云」计算时还剔除了飞点（nodes/core/geometry.py points_from_depth 的 depth_edges），剩下的像素比
            # 上面的 keep 少，只按 keep 数点就永远对不上、网格发法不会生效。不在这里
            # 重算 depth_edges（整幅求法线，640×480 每帧约 0.25 秒，4K 十秒量级）：把点云的点按该帧相机投回画面，
            # 得到每个点所在的格；这些格按行优先严格递增、且都在 keep 里，才当作点云保留的格，其余照旧按点发。
            # 下面逐点重建比对（GRID_TOLERANCE）仍然把关：格找错了，重建出的点就对不上，同样退回按点发。
            idx = _cells_of(mine, keep, at[frame])
        one = len(cols) and np.all(cols == cols[0])
        if keep.shape != (full_gh, full_gw) or len(idx) != len(mine):
            return _cloud_arrays(points, cols, widths)
        rgb = colours(cols).reshape(-1, 3) if len(cols) else np.zeros((0, 3), np.uint8)
        zk = np.asarray(z[::full_step, ::full_step, 0], np.float32).reshape(-1)[idx]
        i = at[frame]
        rebuilt = grid_points(zk, idx, full_gw, full_step, width, height, float(focal[i]), mats[i], principal[i])
        if len(mine) and not np.all(np.abs(rebuilt - mine) <= GRID_TOLERANCE[0] + GRID_TOLERANCE[1] * np.abs(mine)):
            return _cloud_arrays(points, cols, widths)
        grid = np.full(full_gw * full_gh, np.nan, np.float32)
        grid[idx] = zk
        if one:
            tint = np.asarray(cols[0] if len(cols) else [0.7] * 3, np.float32)
            return {"depth": _thin(grid, full_gw, full_gh, proxy), "grid_tint": tint}
        if proxy > 1:
            # Proxy: spread the colours back over the whole grid first, then keep every q-th cell together with depth,
            # so colours and kept points still correspond one to one
            spread = np.zeros((full_gw * full_gh, 3), rgb.dtype)
            spread[idx] = rgb
            thin_depth = _thin(grid, full_gw, full_gh, proxy)
            thin_rgb = spread.reshape(full_gh, full_gw, 3)[::proxy, ::proxy].reshape(-1, 3)
            return {"depth": thin_depth, "grid_colors": thin_rgb[np.isfinite(thin_depth)]}
        return {"depth": grid, "grid_colors": rgb}  # the kept pixels' colours in their order: the cloud's own

    info = {"width": width, "height": height, "step": step, "gw": gw, "gh": gh, "focal": focal, "cam": mats, "principal": principal,
            # proxy > 1: every `proxy`-th cell is kept. The viewer states 「显示了 N / 共 M 点」 in its notice area; thinning
            # is never silent
            "proxy": proxy}
    return info, read


POINTS_LIMIT = "view.points_max_mb"   # one setting governs all point thinning (grids and point lists alike)


def _budget(key: str) -> int:
    from ..config import settings

    return max(1, int(settings()[key])) * 2**20


def _proxy_step(cells: int, per_cell: int = GRID_BYTES_PER_CELL) -> int:
    """The step, per direction, at which a frame's grid cells are kept so that the frame stays within the 「点云上限」
    (setting view.points_max_mb, editable by administrators). 1: every point is sent. `per_cell`: bytes sent per cell."""
    limit = _budget(POINTS_LIMIT)
    if cells * per_cell <= limit:
        return 1
    return int(np.ceil(np.sqrt(cells * per_cell / limit)))


def _point_step(count: int) -> int:
    """The step at which a frame's points (a point list, not a grid) are kept to stay within the 「点云上限」. 1: nothing is dropped."""
    limit = _budget(POINTS_LIMIT)
    if count * CLOUD_BYTES_PER_POINT <= limit:
        return 1
    return int(np.ceil(count * CLOUD_BYTES_PER_POINT / limit))


def _thin(grid: np.ndarray, gw: int, gh: int, proxy: int) -> np.ndarray:
    """Every `proxy`-th cell of a grid (returned as is when proxy is 1). The kept values are the original ones, bit for bit."""
    return grid if proxy <= 1 else np.ascontiguousarray(grid.reshape(gh, gw)[::proxy, ::proxy].reshape(-1))


def drop_points(sample: dict[str, np.ndarray], every: int) -> dict[str, np.ndarray]:
    """Every `every`-th point of a frame's cloud: points, colours and widths are thinned together, and kept point
    coordinates are bit-identical.

    When the frame read is a depth grid rather than points (`depth`), it is left alone here: grids are thinned by
    `_thin` in `depth_grid`."""
    points = sample.get("points")
    if every <= 1 or points is None:
        return sample
    pts = np.asarray(points).reshape(-1, 3)
    n = len(pts)
    out = dict(sample)
    out["points"] = np.ascontiguousarray(pts[::every])
    cols = sample.get("colors")
    if cols is not None and np.asarray(cols).size == n * 3:
        out["colors"] = np.ascontiguousarray(np.asarray(cols).reshape(-1, 3)[::every].reshape(-1))
    widths = sample.get("widths")
    if widths is not None and np.asarray(widths).size == n:
        out["widths"] = np.ascontiguousarray(np.asarray(widths).reshape(-1)[::every])
    return out


def cloud_per_frame(frames: list[int], name: str, read: Callable[[int], tuple[np.ndarray, np.ndarray]],
                    first: int | None = None, one_frame_chunks: bool = False) -> View:
    """One point cloud, already in the world, read frame by frame (`read(frame)`: points, colours): a depth or
    position map's point preview.

    `first`: the frame used as the sample for the point count (chunk byte estimates, 「显示了 N / 共 M 点」 and the
    thinning step all use it). A partial result may not have written its first frame yet, so the first frame that has a
    file is passed rather than frames[0]; None counts as empty (0 points, no thinning)."""
    at = frames[0] if first is None and frames else first
    sample = read(at)[0] if at is not None else np.zeros((0, 3))
    item = {"name": np.array(name), "path": np.array(name), "frames": np.array(frames), "world": np.eye(4)[None],
            "counts": np.array([len(sample)]), "points": np.asarray(sample), "per_frame": np.array(True)}
    return encode(frames, {"points": [item]}, (0, 0), reader=lambda kind, it: lambda f: _cloud_arrays(*read(f), [], every=int(it.get("every", 1))),
                  one_frame_chunks=one_frame_chunks)


# ------------------------------------------------------------------ the views of packets (built in the view worker)


class Refused(Exception):
    """A view that cannot be made, as the MessageError the viewer reads (its kind and message carried across the pipe
    to the web server, which raises it as it is: its status named in its docstring, answered in one place,
    server/app.py; an error itself does not pickle, only its args do)."""

    def __init__(self, error: MessageError) -> None:
        super().__init__(error.message.code)
        self.error = error


def _load(fp: str):
    from ..engine import Packet
    from ..data.packet import packet_dir

    try:
        d = packet_dir(fp)
    except ValueError as exc:
        raise Refused(Invalid(message_of(exc))) from None
    if not Packet.exists(d):
        raise Refused(NotFound(Msg("E-ACCESS-NORESULT")))
    return Packet.load(d)


def points_view(depth_fp: str, camera_fp: str | None) -> View:
    """A finished depth or position map viewed as a point cloud (together with the camera it is seen from)."""
    from ..data.payloads import image_files

    p = _load(depth_fp)
    # what is not a picture has no frames nor width: _points_view refuses it by its type (E-VIEW-NOTPOINTS)
    frames = [int(f) for f in p.meta.get("frames") or []]
    files = image_files(p) if frames else {}
    return _points_view(type_=p.type, width=p.meta.get("width", 0), space=p.meta.get("space"),
                        span=p.meta.get("range") or p.meta.get("full_range"),
                        frames=[f for f in frames if f in files], files_of=lambda: files, camera_fp=camera_fp)


def partial_points_view(directory: str, type_: str, width: int, space: str,
                        frames: tuple[int, ...], camera_fp: str | None) -> View:
    """The frames a node still being computed has written so far, viewed as a point cloud (each frame is viewable as
    soon as it is written).

    It differs from the finished view in two ways only; everything else (unprojection, colouring, thinning) is the same:

    1. which frames have files is checked each time (`partial_frames` scans the folder) rather than fixed as a table
       when the view is built, since the node is still writing and the table would change;
    2. one frame per chunk (`one_frame_chunks`). The finished view packs chunks of about CHUNK_BYTES whose boundaries follow
       the total frame count; while computing the frame count keeps changing, and every boundary change would change
       all addresses and invalidate everything the browser holds. One-frame chunk addresses depend only on the frame
       number, and a written frame's bytes never change afterwards (that is what `streams=True` means for this node
       family), so a frame already fetched is never fetched again.

    This code knows neither jobs nor nodes: the web server works out which folder, type, size and frame count
    (server/farm.py partial_points) and passes them in; the view worker only turns the frames present in the folder
    into points."""
    from ..data.payloads import partial_frames

    d = Path(directory)
    return _points_view(type_=type_, width=width, space=space or None, span=None,
                        frames=list(frames), files_of=lambda: {f: d / n for f, n in partial_frames(d).items()},
                        camera_fp=camera_fp, one_frame_chunks=True)


def _points_view(*, type_: str, width: int, space: str | None, span, frames: list[int],
                 files_of: Callable[[], dict[int, Path]], camera_fp: str | None,
                 one_frame_chunks: bool = False) -> View:
    """One set of points per frame, coloured by value: every pixel with a value is included, and only the frames the
    viewer asks for are read.

    The channel count and the packet's own metadata decide, not the type name:
    - one channel = one distance per pixel, unprojected along the view direction, which requires a camera;
    - three channels = one coordinate per pixel, which needs a camera only when the packet states camera space
      (meta space = camera).

    `files_of()` gives the file of each frame on every call: a fixed table for a finished result, a fresh folder scan
    for a partial one.
    """
    from ..data.camera import CameraSamples
    from ..data.payloads import read_map
    from ..nodes.kit.unproject import unproject_depth
    from ..data.types import channels_of

    camera = _load(camera_fp) if camera_fp else None
    count = channels_of(type_)
    if count not in (1, 3):
        raise Refused(Invalid(Msg("E-VIEW-NOTPOINTS")))
    distances = count == 1  # one distance per pixel rather than one coordinate
    in_camera = distances or space == "camera"
    if in_camera:
        if camera is None:
            raise Refused(Invalid(Msg("E-VIEW-NEEDCAMERA")))
        cam = CameraSamples.from_packet(camera, frames)
        focal, mats, principal = cam.focal_px(width), cam.cam_to_world, cam.principal_px(width)
    # The data's value range: taken from the packet when recorded (`range` is 1%-99%, `full_range` is uncropped);
    # otherwise computed from the first frame that has a file. The range only affects colour, not point positions, so a
    # missing range must not make the whole 3D display fail. Partial results take this path: their first frame may not
    # be written yet, so the first frame that exists is used.
    here = files_of()
    present = [f for f in frames if f in here]
    if span is None and present:
        first, alpha0 = read_map(here[present[0]])
        vals = first[..., 0][alpha0 > 0] if count == 1 else first[alpha0 > 0][..., :3]
        span = (float(np.min(vals)), float(np.max(vals))) if getattr(vals, "size", 0) else (0.0, 1.0)
    lo, hi = span if span is not None else (0.0, 1.0)
    at = {f: i for i, f in enumerate(frames)}
    if distances and present:
        return _distance_grid_view(frames, files_of, read_map, width, here[present[0]], focal, mats, principal, (lo, hi),
                                   one_frame_chunks)

    def read(f: int) -> tuple[np.ndarray, np.ndarray]:
        i = at[f]
        path = files_of().get(f)
        if path is None:  # this frame is not written yet (the partial result): an empty frame, not an error
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)
        data, alpha = read_map(path)
        r, c = np.nonzero(alpha > 0)
        if distances:
            pts = unproject_depth(data[..., 0], float(focal[i]), mats[i], r, c, principal[i])
            col = ramp_colours(data[r, c, 0], lo, hi)
        else:
            pts = data[r, c, :3]
            if in_camera:
                pts = pts @ mats[i][:3, :3].T + mats[i][:3, 3]
            col = np.clip((pts - lo) / max(hi - lo, 1e-6), 0, 1)
        return np.asarray(pts, np.float32), np.clip(col, 0, 1).astype(np.float32)

    return cloud_per_frame(frames, POINTS_NAME, read, first=present[0] if present else None,
                           one_frame_chunks=one_frame_chunks)


RAMP_BYTES_PER_CELL = 4  # a distance map sent as its grid: the depth alone (float32); its colour is the ramp, drawn on the GPU

# 距离图的色带（近暖远冷）：v = (距离 − 近) / (远 − 近)，颜色 = clamp(NEAR + v × SLOPE, 0, 1)。唯一的定义：网格发法把这两组数
# 放进描述（grid 的 ramp_colour），网页着色器照它算（webui/src/view/pointShaders.tsx uRampNear / uRampSlope）；
# 逐点发的那条路（_points_view，边算边看在还没有任何一帧写出时建的视图）用 ramp_colours 算，同一组数
RAMP_NEAR = (1.0, 0.75, 0.45)
RAMP_SLOPE = (-0.6, -0.35, 0.4)


def ramp_colours(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """距离 [N] 在色带上的颜色 [N, 3]（RAMP_NEAR / RAMP_SLOPE）。"""
    v = (np.asarray(z, np.float64) - lo) / max(hi - lo, 1e-6)
    return np.clip(np.asarray(RAMP_NEAR) + v[:, None] * np.asarray(RAMP_SLOPE), 0.0, 1.0)


def _distance_grid_view(frames: list[int], files_of: Callable[[], dict[int, Path]], read_map, width: int, sample: Path,
                        focal, mats, principal, ramp: tuple[float, float], one_frame_chunks: bool) -> View:
    """「深度直接看点云」（一通道距离图 + 相机）按网格发：每帧只发深度（float32，alpha 为 0 的像素记 NaN），点由网页的
    显卡按 grid_points 的同一算法重建（与「深度转点云」的网格发法同一条路：viewFormat.ts gridPoints、pointShaders.tsx
    GRID），颜色是按距离的色带（近暖远冷，`ramp` = 取值范围），也在着色器里算，与按点发时算的色带同一公式。

    每像素只发 4 字节（按显式点发要每点 12 字节坐标 + 12 字节颜色，640×480 每帧 7.4 MB；gzip 后约 0.69 对 4.32 MB/帧）。
    超过「点云上限」时与网格一样每 proxy 格取一格。"""
    z0, _ = read_map(sample)
    height, full_w = int(z0.shape[0]), int(z0.shape[1])
    gw, gh = full_w, height
    proxy = _proxy_step(gw * gh, RAMP_BYTES_PER_CELL)
    step = proxy
    tgw, tgh = len(range(0, gw, proxy)), len(range(0, gh, proxy))

    def read(f: int) -> dict[str, np.ndarray]:
        path = files_of().get(f)
        if path is None:  # this frame is not written yet (the partial result): an empty frame, not an error
            return {"points": np.zeros((0, 3), np.float32)}
        data, alpha = read_map(path)
        z = np.array(data[..., 0], np.float32)
        z[~(alpha > 0)] = np.nan
        return {"depth": _thin(z.reshape(-1), gw, gh, proxy)}

    info = {"width": full_w, "height": height, "step": step, "gw": tgw, "gh": tgh, "focal": focal, "cam": mats,
            "principal": principal, "proxy": proxy, "ramp": [float(ramp[0]), float(ramp[1])],
            "ramp_colour": [list(RAMP_NEAR), list(RAMP_SLOPE)],
            "bytes_per_cell": RAMP_BYTES_PER_CELL}
    item = {"name": np.array(POINTS_NAME), "path": np.array(POINTS_NAME), "frames": np.array(frames), "world": np.eye(4)[None],
            "counts": np.array([0]), "points": np.zeros((0, 3)), "per_frame": np.array(True), "depth_grid": (info, read)}
    return encode(frames, {"points": [item]}, (0, 0), one_frame_chunks=one_frame_chunks)


POINTS_NAME = "points"  # the name of the points in the scene tree (one set per view; the port name states what it is)


def build(how: tuple) -> View:
    """The view `how` names:

    - `("scene", packet)`: a scene packet;
    - `("points", depth or position map, camera or None)`: a finished point cloud preview;
    - `("points_partial", folder, type, width, coordinate system, all frames, camera or None)`: a partial result
      (streaming preview, partial_points_view). The view worker knows neither jobs nor nodes, so the web server works
      these out and passes them in.
    """
    if how[0] == "scene":
        from ..data.types import accepts

        p = _load(how[1])
        if not accepts("scene", p.type):  # asked of a picture or a value: said so, never a KeyError of its meta
            raise Refused(Invalid(Msg("E-VIEW-NOTSCENE")))
        return scene_view(p)
    if how[0] == "points_partial":
        return partial_points_view(*how[1:])
    return points_view(how[1], how[2])


def respond_description(text: str, accept_encoding: str):
    """The 3D view's description, gzipped (thousands of props make several MB of JSON, a few hundred KB compressed).

    Cache-Control is `no-cache` (FRESH), not `no-store`: the description is not a content address, since it follows
    settings (changing `view.points_max_mb` changes the sampled grid cells), so it cannot be marked immutable; but the
    browser may keep the previous copy and send `If-None-Match`, getting a zero-byte 304 when unchanged (the ETag is
    computed from the response bytes by the server/wire.py middleware). With `no-store`, the whole description would
    be downloaded again every time the displayed node changes.

    Switching nodes back and forth in the same tab does not even make that request: the whole scene is in the page's
    cache (webui/src/view/sceneData.ts)."""
    from fastapi.responses import Response

    from .wire import FRESH

    headers = {"Cache-Control": FRESH, "Vary": "Accept-Encoding"}
    if "gzip" in (accept_encoding or ""):
        return Response(gzip.compress(text.encode(), 5), media_type="application/json", headers={**headers, "Content-Encoding": "gzip"})
    return Response(text, media_type="application/json", headers=headers)


def respond(gz: bytes, accept_encoding: str, kept: bool = True):
    """A part as the browser takes it: gzipped when it can read gzip, kept by the browser permanently (a base part's URL
    carries its content's hash; a chunk's, the result's fingerprint and generation, the format, its frames and the
    「点云上限」: the same URL is always the same bytes), so a chunk let go and asked for again comes from the browser's own cache, not the network.

    `kept=False`: a partial result (server/farm.py partial_points_part). Its address carries the job id, node and frame
    number rather than a content address, so it cannot be marked immutable; avoiding re-downloads relies on the page's
    own cache (webui/src/view/scenePart.ts), which releases the whole prefix when the node finishes."""
    from fastapi.responses import Response

    from .wire import IMMUTABLE, NEVER

    headers = {"Cache-Control": IMMUTABLE if kept else NEVER, "Vary": "Accept-Encoding"}
    if "gzip" in (accept_encoding or ""):
        return Response(gz, media_type="application/octet-stream", headers={**headers, "Content-Encoding": "gzip"})
    return Response(gzip.decompress(gz), media_type="application/octet-stream", headers=headers)
