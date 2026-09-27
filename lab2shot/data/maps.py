"""Operations on 2D data maps, in numpy (the main environment has neither OpenCV nor SciPy): grow / shrink with a
round brush, blur, soft ranges, and a map's values at a frame.

Grey-level morphology works on soft masks too (a matte's soft edge moves out or in as a whole), like Nuke's Erode.
"""

from __future__ import annotations

import math

import numpy as np

from ..errors import Invalid
from ..messages import Msg
from .payloads import file_at, read_map, window_of
from .packet import Packet


def map_at(p: Packet, frame: int, box: tuple | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(values [H,W,C], validity [H,W]) of a map packet at `frame`, every pixel it keeps (its data window, so a
    canvas comes whole); a still gives its one picture at any frame. `box`: read it in another packet's window
    instead (a node combining two of them reads both in the same box). None when the packet has no such frame."""
    path = file_at(p, frame)
    return None if path is None else read_map(path, box or window_of(p).data)


BAND_PIXELS = 1 << 18  # maximum output pixels per row band (1080p: 136 rows, 8 bands per frame); see the table above alpha_of


def bands(shape: tuple[int, ...], budget: int | None = None):
    """Split an output of `shape` into row bands (slices along axis 0) of at most `budget` pixels each (default
    BAND_PIXELS). An output that fits in one band (small images, test data) yields a single full slice, following
    exactly the same path as unbanded processing."""
    rows = shape[0]
    per_row = 1
    for n in shape[1:]:
        per_row *= int(n)
    each = max(1, (BAND_PIXELS if budget is None else budget) // max(per_row, 1))
    if each >= rows:
        yield slice(0, rows)
        return
    for top in range(0, rows, each):
        yield slice(top, min(top + each, rows))


def resize(values: np.ndarray, width: int, height: int) -> np.ndarray:
    """values [h,w] or [h,w,C] at width x height, bilinear with the pixel centres lined up (like OpenCV's INTER_LINEAR
    and torch's align_corners=False): a model's result at its processing size brought to the plate. The values are
    not scaled (vectors in pixels are, by their caller)."""
    h, w = values.shape[:2]
    if (w, h) == (width, height):
        return values

    def axis(n_out: int, n_in: int):
        x = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        i0 = np.floor(x).astype(np.int64)
        return i0, np.minimum(i0 + 1, n_in - 1), (x - i0).astype(np.float32)

    flat = values.reshape(h, w, -1).astype(np.float32)
    y0, y1, ty = axis(height, h)
    x0, x1, tx = axis(width, w)
    # the same row bands as the sampling kernels (bands): processing the full frame at once builds two full-size
    # intermediate arrays (`flat[y0]` is [height, w, C], then [height, width, C]), tens of MB for 1080p with three
    # channels, and every model result returning to format size passes through here, more often than the sampling
    # kernels. Banded, the intermediates are one band in size and the result is bit-identical (the arithmetic per
    # output pixel is unchanged)
    out = np.empty((height, width, flat.shape[2]), np.float32)
    for band in bands(out.shape):
        yb0, yb1, tb = y0[band], y1[band], ty[band]
        rows = flat[yb0] * (1 - tb)[:, None, None] + flat[yb1] * tb[:, None, None]
        out[band] = rows[:, x0] * (1 - tx)[None, :, None] + rows[:, x1] * tx[None, :, None]
    return out.reshape(height, width, *values.shape[2:])


# Resampling processes one row band at a time. Processing a whole frame at once builds, for every tap, an intermediate
# array as large as the output: peak temporary memory (tracemalloc) for one 1080p three-channel frame is 595 MB for
# bicubic, 310 MB for bilinear and 146 MB for nearest, missing the cache throughout. Banded, the peaks are
# 114 / 78 / 50 MB, and eight frames in parallel save eight times that; a full 4K bicubic frame needs over 2 GB.
#
# Banding changes only the order of computation, not any floating-point operation: each output pixel depends only on
# the source and its own sampling position, with no interaction between bands, so results are bit-identical with and
# without banding (a requirement).
#
# Band size, measured on a 32-core machine, 1080p, 8 frames per sampling mode, minimum of 7 alternating rounds, ms/frame:
#     px/band      bilinear x3     bicubic x3      bilinear x1     nearest x3
#                1 thr   8 thr    1 thr   8 thr    1 thr   8 thr    1 thr   8 thr
#      65536     158.4   135.4    586.7   332.3     91.0   150.1     48.3    53.2
#     131072     160.4    66.7    583.1   218.5     96.0    76.5     48.3    28.9
#     262144     166.8    57.6    655.2   245.6    100.4    41.2     48.3    17.1   <- current
#     524288     176.0    64.0    755.2   279.7    118.6    50.3     52.8    21.4
#     full frame  226.5    92.2    923.7   350.3    149.2    87.1     72.8    32.5
# Smaller bands are faster single-threaded (intermediates stay in cache), but the number of numpy calls per frame
# multiplies, and with eight frames in parallel their interpreter overhead contends (the 65536 row is slower with 8
# threads than with 1). Nodes always process several frames in parallel (engine/cook.py FRAME_THREADS = 8), so the
# choice follows the 8-thread columns: 262144 (1080p split into 8 bands of 136 rows) is fastest in three of the four
# modes and 12% slower than the fastest for bicubic; single-threaded it is 5-12% slower than the fastest.
def alpha_of(values: np.ndarray, alpha: np.ndarray | None, nearest: bool = False) -> np.ndarray | None:
    """Per-pixel validity of the source, used as weights by the sampling kernels: all ones without an alpha (nearest
    sampling does not need it, as it only checks presence). Built once per frame and shared by all bands (building a
    full-size copy per band would be slower than no banding)."""
    if alpha is not None:
        return np.asarray(alpha, np.float32)
    return None if nearest else np.ones(values.shape[:2], np.float32)


def sample(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, alpha: np.ndarray | None = None,
           nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """values [H,W,C] at the continuous pixel positions (xs, ys) [h,w] (+x right, +y down, pixel centres at +0.5) ->
    (values [h,w,C], where they hold a value [h,w] 0..1). Bilinear, weighted by the source's `alpha` (where it holds
    values) so no invalid neighbour bleeds in; `nearest` for labels. Positions outside the source hold nothing. One
    sampler for every warp: ST-maps (Nuke's STMap), motion vectors (IDistort), chains of them.

    A row band at a time (`bands`): the same arithmetic per pixel, bit for bit, with the taps' intermediate arrays
    small enough to stay in cache."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    a = alpha_of(values, alpha, nearest)
    out = np.empty((*xs.shape, *values.shape[2:]), np.float32)
    valid = np.empty(xs.shape, np.float32)
    for band in bands(xs.shape):
        out[band], valid[band] = sample_band(values, xs[band], ys[band], a, nearest)
    return out, valid


def sample_band(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, a: np.ndarray | None,
                nearest: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`sample` on one row band: `a` is the source's alpha already as float32 (`alpha_of`, built once a frame; None
    only on the nearest path, where nothing weights the taps). Nothing here looks past its own pixels, which is why
    the bands add up bit for bit to the whole frame."""
    h, w = values.shape[:2]
    inside = np.isfinite(xs) & np.isfinite(ys) & (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
    xs, ys = np.where(inside, xs, 0.0), np.where(inside, ys, 0.0)
    if nearest:
        xi, yi = np.clip(np.floor(xs).astype(np.int64), 0, w - 1), np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
        valid = inside & ((a[yi, xi] > 0) if a is not None else True)
        return np.where(valid[..., None], values[yi, xi], 0.0).astype(np.float32), valid.astype(np.float32)
    x, y = np.clip(xs - 0.5, 0, w - 1).astype(np.float32), np.clip(ys - 0.5, 0, h - 1).astype(np.float32)
    fx, fy = np.floor(x), np.floor(y)  # kept in float32: the weights stay single precision (4x faster on HD frames)
    x0, y0 = fx.astype(np.int64), fy.astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    snap = lambda t: np.where(t < 1e-4, 0.0, np.where(t > 1 - 1e-4, 1.0, t))[..., None]  # noqa: E731  float noise at a centre
    tx, ty = snap(x - fx), snap(y - fy)
    corners = [(y0, x0, (1 - tx) * (1 - ty)), (y0, x1, tx * (1 - ty)), (y1, x0, (1 - tx) * ty), (y1, x1, tx * ty)]
    total = sum(wgt * a[r, c][..., None] * values[r, c] for r, c, wgt in corners)
    weight = sum(wgt * a[r, c][..., None] for r, c, wgt in corners)[..., 0]
    valid = np.where(inside, weight, 0.0)
    out = np.where(valid[..., None] > 0, total / np.maximum(weight, 1e-12)[..., None], 0.0)
    return out.astype(np.float32), valid.astype(np.float32)


def same_size(packets: dict[str, Packet | None]) -> None:
    """2D data used together comes from the same plate (rule 2: maps are the plate's size): packets (what the user
    calls them -> packet, None: not connected) of another size than the first are refused, saying which."""
    sized = [(label, p.meta["width"], p.meta["height"]) for label, p in packets.items()
             if p is not None and not p.meta.get("empty")]  # a packet with nothing in it has no size to compare
    for label, w, h in sized[1:]:
        first, w0, h0 = sized[0]
        if (w, h) != (w0, h0):
            raise Invalid(Msg("E-MAPS-SIZE", input=label, width=w, height=h, first=first, first_width=w0, first_height=h0))


def _shifted(a: np.ndarray, dy: int, dx: int, fill: float) -> np.ndarray:
    """out[y, x] = a[y + dy, x + dx]; pixels that come from outside the picture are `fill`."""
    h, w = a.shape
    out = np.full_like(a, fill)
    ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
    xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
    out[yd, xd] = a[ys, xs]
    return out


def _line_max(a: np.ndarray, half: int, dy: int, dx: int) -> np.ndarray:
    """Max over the 2 * half + 1 pixels centred on each pixel along the step (dy, dx), in O(log n) passes: windows of
    doubling length, then two overlapping ones cover any length (like a sparse table)."""
    if half <= 0:
        return a
    py, px = half * abs(dy), half * abs(dx)
    padded = np.pad(a, ((py, py), (px, px)), constant_values=-np.inf)  # every window starts inside
    n = 2 * half + 1
    m, window = 1, padded
    while 2 * m <= n:
        window = np.maximum(window, _shifted(window, m * dy, m * dx, -np.inf))  # [p, p + 2m)
        m *= 2
    full = np.maximum(window, _shifted(window, (n - m) * dy, (n - m) * dx, -np.inf))  # [p, p + n)
    centred = _shifted(full, -half * dy, -half * dx, -np.inf)
    return centred[py : py + a.shape[0], px : px + a.shape[1]]


def dilate(a: np.ndarray, radius: float) -> np.ndarray:
    """Grey-level dilation with a round brush of `radius` pixels: a regular octagon, the sum of lines across,
    down and along both diagonals (Minkowski sum). Outside the picture counts as nothing."""
    r = int(round(radius))
    if r <= 0:
        return a
    diag = int(round(r / (2 + math.sqrt(2))))  # half-length of each diagonal line; each of its steps is (1, 1)
    axis = r - 2 * diag  # half-length across and down: a regular octagon has axis = sqrt(2) * diag
    out = a.astype(np.float32)
    for (dy, dx), half in (((0, 1), axis), ((1, 0), axis), ((1, 1), diag), ((1, -1), diag)):
        out = _line_max(out, half, dy, dx)
    return out


def grow(a: np.ndarray, pixels: float) -> np.ndarray:
    """Grow (pixels > 0) or shrink (< 0) a mask by a round brush."""
    if pixels >= 0:
        return dilate(a, pixels)
    return -dilate(-a.astype(np.float32), -pixels)


def _box(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Mean over the 2k + 1 pixels centred on each pixel along `axis`; at the picture's edge only the pixels inside
    count (no darkening towards the border)."""
    if k <= 0:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    cs = np.concatenate([np.zeros((1, *a.shape[1:]), np.float64), np.cumsum(a, axis=0, dtype=np.float64)])
    i = np.arange(n)
    lo, hi = np.clip(i - k, 0, n), np.clip(i + k + 1, 0, n)
    count = (hi - lo).reshape(-1, *([1] * (a.ndim - 1)))
    return np.moveaxis(((cs[hi] - cs[lo]) / count).astype(np.float32), 0, axis)


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-like blur of `sigma` pixels: three box passes each way (the boxes' variances add up to sigma²)."""
    if sigma <= 0.25:
        return a
    k = max(1, int(round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2)))  # 3 boxes of width 2k+1: 3((2k+1)²-1)/12 = σ²
    out = a.astype(np.float32)
    for _ in range(3):
        out = _box(_box(out, k, 0), k, 1)
    return out


def feather(a: np.ndarray, pixels: float) -> np.ndarray:
    """Soften a mask's edges so they go from 0 to 1 over about `pixels` (10–90 %)."""
    return blur(a, pixels / 2.56)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """0 below edge0, 1 above edge1, a smooth ramp between (a hard step when they are equal)."""
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def polygon_coverage(shapes: list[list[tuple[float, float]]], width: int, height: int, samples: int = 4) -> np.ndarray:
    """Coverage of each pixel by hand-drawn closed outlines (image pixels, top-left 0,0), 0..1, height x width.

    Coverage is computed from samples x samples points per pixel, giving soft edges (the same anti-aliasing as
    「人物框转遮罩」, not a separate implementation). Several outlines are combined as a union (each computed separately,
    then the maximum), so overlaps do not cut holes. Each outline is evaluated only over its own bounding region (a small
    garbage mask on a 4K frame costs only that region).
    """
    out = np.zeros((height, width), np.float32)
    if width <= 0 or height <= 0:
        return out
    for shape in shapes:
        pts = np.asarray(shape, np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        ax, ay = pts[:, 0], pts[:, 1]
        bx, by = np.roll(ax, -1), np.roll(ay, -1)
        keep = ay != by  # horizontal edges produce no intersections
        ax, ay, bx, by = ax[keep], ay[keep], bx[keep], by[keep]
        if not len(ay):
            continue
        x0, x1 = max(0, int(np.floor(pts[:, 0].min()))), min(width, int(np.ceil(pts[:, 0].max())) + 1)
        y0, y1 = max(0, int(np.floor(pts[:, 1].min()))), min(height, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:  # entirely outside the frame
            continue
        cols = x1 - x0
        centres = x0 + (np.arange(cols * samples, dtype=np.float64) + 0.5) / samples
        one = np.zeros((y1 - y0, cols), np.float32)
        for r in range(y0 * samples, y1 * samples):  # one scanline at a time, only over the rows the outline covers
            y = (r + 0.5) / samples
            hit = (ay > y) != (by > y)
            if not hit.any():
                continue
            xs = np.sort(ax[hit] + (y - ay[hit]) * (bx[hit] - ax[hit]) / (by[hit] - ay[hit]))
            inside = np.searchsorted(xs, centres, side="right") % 2 == 1  # even-odd rule, as used by the viewer for point-in-shape tests
            one[r // samples - y0] += inside.reshape(cols, samples).sum(1)
        block = out[y0:y1, x0:x1]
        np.maximum(block, one / (samples * samples), out=block)
    return np.clip(out, 0.0, 1.0)
