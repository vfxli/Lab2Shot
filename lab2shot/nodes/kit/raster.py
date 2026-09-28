"""Small CPU renderer: meshes and point clouds seen through the shot camera.

Vectorized rasterization with a z-buffer (numpy only), so the core environment needs no OpenGL. It renders a scene's
silhouette, depth and normals as data maps (render_layers). Triangles are clipped at the near plane (a ground plane
reaching behind the camera still renders), triangles and points become fragments (a pixel, its inverse depth, what it
shows) in batches of bounded size, and a z-buffer keeps each pixel's nearest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEAR = 1.0  # scene units (cm): nearer than this is behind the lens
MAX_POINT_RADIUS = 6  # pixels (per sub-sample): nearer points are drawn at this size
BATCH = 4_000_000  # candidate pixels per batch: bounds memory whatever the triangles' size
POINT = -2  # z-buffer owner of a pixel a point cloud covers (triangles: their index; nothing: -1)


@dataclass
class PinholeCamera:
    """World -> pixels. `world_to_cam` is a column-vector matrix, camera looks down -Z."""

    world_to_cam: np.ndarray  # [4,4]
    focal_px: float
    width: int
    height: int

    def to_camera(self, p: np.ndarray) -> np.ndarray:
        return p @ self.world_to_cam[:3, :3].T + self.world_to_cam[:3, 3]

    def project(self, p_cam: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Camera-space points -> (pixel xy [N,2], depth [N]) at `scale` x resolution."""
        depth = -p_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.focal_px * p_cam[:, 0] / depth + self.width / 2
            y = -self.focal_px * p_cam[:, 1] / depth + self.height / 2
        return np.stack([x, y], 1) * scale, depth


@dataclass
class Mesh:
    points_world: np.ndarray  # [V,3]
    faces: np.ndarray  # [T,3]


@dataclass
class PointCloud:
    points_world: np.ndarray  # [N,3]
    widths: np.ndarray  # [N] diameters, scene units


# ------------------------------------------------------------------ geometry


def vertex_normals(p: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Smooth normals: each vertex averages its faces' normals, weighted by their area."""
    v = p[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    out = np.zeros(p.shape, np.float64)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def _soup(camera: PinholeCamera, meshes: list[Mesh]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All meshes as one triangle soup in camera space, so they occlude each other: (points, faces, the mesh each
    face comes from)."""
    pts, tris, owner, offset = [np.zeros((0, 3))], [np.zeros((0, 3), np.int64)], [np.zeros(0, np.int64)], 0
    for k, m in enumerate(meshes):
        pts.append(camera.to_camera(np.asarray(m.points_world, np.float64)))
        tris.append(np.asarray(m.faces, np.int64) + offset)
        owner.append(np.full(len(m.faces), k))
        offset += len(m.points_world)
    return np.concatenate(pts), np.concatenate(tris), np.concatenate(owner)


def clip_near(p: np.ndarray, faces: np.ndarray, extra: np.ndarray | None = None, near: float = NEAR):
    """Triangles cut at the near plane (camera space, looking down -Z): the parts in front of it, winding kept. A
    triangle with one vertex in front becomes a smaller triangle, one with two a quad (two triangles). Per-vertex
    `extra` attributes are interpolated onto the new vertices. Returns (points, faces, extra, the source face of
    each face)."""
    depth = -p[:, 2]
    front = depth[faces] > near  # [T,3]
    count = front.sum(1)
    keep = np.flatnonzero(count == 3)
    out_faces, source = [faces[keep]], [keep]
    new_p, new_x, base = [p], [extra] if extra is not None else [], len(p)

    def cut(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """New vertices where edges a -> b of triangles t cross the near plane; their indices."""
        nonlocal base
        s = ((depth[a] - near) / (depth[a] - depth[b]))[:, None]
        new_p.append(p[a] + s * (p[b] - p[a]))
        if extra is not None:
            new_x.append(extra[a] + s * (extra[b] - extra[a]))
        idx = base + np.arange(len(t))
        base += len(t)
        return idx

    for n_front, odd_is_front in ((1, True), (2, False)):
        t = np.flatnonzero(count == n_front)
        if not len(t):
            continue
        k = np.argmax(front[t] == odd_is_front, axis=1)  # the vertex on its own side of the plane comes first
        rot = faces[t[:, None], (k[:, None] + np.arange(3)) % 3]
        v0, v1, v2 = rot.T
        i01, i02 = cut(t, v0, v1), cut(t, v0, v2)
        if odd_is_front:  # (v0, i01, i02)
            out_faces.append(np.stack([v0, i01, i02], 1))
            source.append(t)
        else:  # quad (i01, v1, v2, i02)
            out_faces += [np.stack([i01, v1, v2], 1), np.stack([i01, v2, i02], 1)]
            source += [t, t]
    return (np.concatenate(new_p), np.concatenate(out_faces).astype(np.int64),
            np.concatenate(new_x) if extra is not None else None, np.concatenate(source))


# ------------------------------------------------------------------ fragments and the z-buffer


def triangle_fragments(xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, width: int, height: int, batch: int = BATCH):
    """Every pixel centre inside a projected triangle, in batches of at most about `batch` candidate pixels (a big
    triangle is split into bands of rows): yields (pixel index y * width + x, triangle, perspective-correct
    barycentric weights [M,3], inverse depth). Triangles must be in front of the camera (clip_near)."""
    p = xy[faces]  # [T,3,2]
    z = depth[faces]
    ok = np.all(np.isfinite(p).reshape(len(faces), -1), axis=1) & np.all(z > 0, axis=1)
    ok &= ~((p[..., 0].max(1) < 0) | (p[..., 0].min(1) > width) | (p[..., 1].max(1) < 0) | (p[..., 1].min(1) > height))
    area2 = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    ok &= np.abs(area2) > 1e-9  # degenerate
    idx = np.flatnonzero(ok)
    p, z, area2 = p[idx], z[idx], area2[idx]
    x0 = np.clip(np.floor(p[..., 0].min(1)), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(p[..., 0].max(1)), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(p[..., 1].min(1)), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.ceil(p[..., 1].max(1)), 0, height - 1).astype(np.int64)
    w = x1 - x0 + 1
    # pieces of at most `batch` pixels: a triangle, or a band of rows of a big one
    rows = np.maximum(1, batch // w)
    bands = (y1 - y0) // rows + 1
    piece = np.repeat(np.arange(len(idx)), bands)
    first = y0[piece] + (np.arange(len(piece)) - np.repeat(np.cumsum(bands) - bands, bands)) * rows[piece]
    last = np.minimum(first + rows[piece] - 1, y1[piece])
    sizes = w[piece] * (last - first + 1)
    ends = np.cumsum(sizes)
    start = 0
    while start < len(piece):
        stop = max(start + 1, int(np.searchsorted(ends, ends[start] - sizes[start] + batch, side="right")))
        yield _fragments(p, z, area2, x0, w, piece[start:stop], first[start:stop], last[start:stop], idx, width)
        start = stop


def _fragments(p, z, area2, x0, w, t, first, last, idx, width):
    counts = w[t] * (last - first + 1)
    t_rep = np.repeat(t, counts)
    local = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    px = x0[t_rep] + local % w[t_rep]
    py = np.repeat(first, counts) + local // w[t_rep]
    sx, sy = px + 0.5, py + 0.5
    a, b, c = p[t_rep, 0], p[t_rep, 1], p[t_rep, 2]
    w0 = (b[:, 0] - sx) * (c[:, 1] - sy) - (c[:, 0] - sx) * (b[:, 1] - sy)
    w1 = (c[:, 0] - sx) * (a[:, 1] - sy) - (a[:, 0] - sx) * (c[:, 1] - sy)
    w2 = (a[:, 0] - sx) * (b[:, 1] - sy) - (b[:, 0] - sx) * (a[:, 1] - sy)
    bary = np.stack([w0, w1, w2], 1) / area2[t_rep, None]
    inside = np.all(bary >= -1e-6, axis=1)
    t_rep, px, py, bary = t_rep[inside], px[inside], py[inside], bary[inside]
    per_z = bary / z[t_rep]  # 1/z is linear in screen space
    iz = per_z.sum(1)
    return py * width + px, idx[t_rep], per_z / iz[:, None], iz


def point_fragments(xy: np.ndarray, depth: np.ndarray, radius: np.ndarray, width: int, height: int, batch: int = BATCH):
    """Every pixel centre within `radius` pixels of a projected point (at least the pixel it falls in), in batches:
    yields (pixel index, inverse depth). Points are flat discs facing the camera."""
    idx = np.flatnonzero((depth > NEAR) & np.all(np.isfinite(xy), axis=1))
    r = np.clip(radius[idx], 0.0, MAX_POINT_RADIUS)
    R = int(np.ceil(r.max())) if len(r) else 0
    dy, dx = (a.ravel() for a in np.mgrid[-R : R + 1, -R : R + 1])
    step = max(1, batch // len(dx))
    for s in range(0, len(idx), step):
        i, rr = idx[s : s + step], r[s : s + step, None]
        x, y = xy[i, 0:1], xy[i, 1:2]
        px, py = np.floor(x).astype(np.int64) + dx, np.floor(y).astype(np.int64) + dy
        near = ((px + 0.5 - x) ** 2 + (py + 0.5 - y) ** 2 <= rr * rr) | ((dx == 0) & (dy == 0))
        near &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
        yield (py * width + px)[near], 1.0 / np.broadcast_to(depth[i][:, None], near.shape)[near]


def nearest(pixel: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Indices of the fragments that win their pixel: the nearest (largest inverse depth)."""
    order = np.lexsort((-iz, pixel))
    first = np.ones(len(order), bool)
    first[1:] = pixel[order][1:] != pixel[order][:-1]
    return order[first]


class ZBuffer:
    """Per pixel: the nearest fragment's inverse depth (0: nothing), its owner (a triangle, POINT, or -1) and its
    barycentric weights."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.iz = np.zeros(width * height)
        self.owner = np.full(width * height, -1, np.int64)
        self.bary = np.zeros((width * height, 3), np.float32)

    def add(self, pixel: np.ndarray, iz: np.ndarray, owner: np.ndarray | int, bary: np.ndarray | None = None) -> None:
        keep = nearest(pixel, iz)
        keep = keep[iz[keep] > self.iz[pixel[keep]]]
        at = pixel[keep]
        self.iz[at] = iz[keep]
        self.owner[at] = owner[keep] if isinstance(owner, np.ndarray) else owner
        if bary is not None:
            self.bary[at] = bary[keep]

    def triangles(self, xy: np.ndarray, depth: np.ndarray, faces: np.ndarray) -> None:
        for pixel, tri, bary, iz in triangle_fragments(xy, depth, faces, self.width, self.height):
            self.add(pixel, iz, tri, bary)


def supersampling(width: int, height: int) -> int:
    """Samples per pixel along each axis: 2 for plates up to 2K, for clean edges."""
    return 2 if max(width, height) <= 2048 else 1


# ------------------------------------------------------------------ data layers


@dataclass
class Layers:
    coverage: np.ndarray  # [H,W] 0..1: how much of the pixel something covers (antialiased silhouette)
    depth: np.ndarray  # [H,W] along the view, scene units; 0 where nothing is hit
    normal: np.ndarray  # [H,W,3] camera space (X right, Y up, Z toward the camera); 0 where not shaded
    hit: np.ndarray  # [H,W] bool: something is seen there
    shaded: np.ndarray  # [H,W] bool: a surface with a normal is seen there (points have none)


def render_layers(camera: PinholeCamera, meshes: list[Mesh], clouds: list[PointCloud]) -> Layers:
    """What the camera sees of meshes and point clouds at its resolution: silhouette coverage, depth and smooth
    normals of the nearest surface. Each pixel takes its nearest sub-sample: no depth is mixed across an edge."""
    w, h = camera.width, camera.height
    ss = supersampling(w, h)
    W, H = w * ss, h * ss
    zb = ZBuffer(W, H)
    p_cam, faces, _ = _soup(camera, meshes)
    p_cam, faces, normals, _ = clip_near(p_cam, faces, vertex_normals(p_cam, faces))
    if len(faces):
        zb.triangles(*camera.project(p_cam, ss), faces)
    for cloud in clouds:
        xy, depth = camera.project(camera.to_camera(np.asarray(cloud.points_world, np.float64)), ss)
        with np.errstate(divide="ignore", invalid="ignore"):
            radius = np.nan_to_num(np.asarray(cloud.widths, np.float64) / 2 * camera.focal_px * ss / depth)
        for pixel, iz in point_fragments(xy, depth, radius, W, H):
            zb.add(pixel, iz, POINT)

    # smooth normals where a triangle won, turned to face the camera (two-sided)
    tri = np.flatnonzero(zb.owner >= 0)
    corners = faces[zb.owner[tri]]
    bary = zb.bary[tri].astype(np.float64)
    n = np.einsum("mk,mkc->mc", bary, normals[corners])
    at = np.einsum("mk,mkc->mc", bary, p_cam[corners])
    n *= np.where(np.einsum("mc,mc->m", n, -at) < 0, -1.0, 1.0)[:, None]
    nrm = np.full((W * H, 3), np.nan)
    nrm[tri] = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore"):
        z = np.where(zb.iz > 0, 1.0 / zb.iz, np.inf)

    def blocks(a: np.ndarray) -> np.ndarray:  # [H*W, ...] -> [h, w, ss*ss, ...]: each pixel's sub-samples
        return a.reshape(h, ss, w, ss, *a.shape[1:]).swapaxes(1, 2).reshape(h, w, ss * ss, *a.shape[1:])

    zs, ns = blocks(z), blocks(nrm)
    pick = np.argmin(zs, axis=2)[..., None]
    depth = np.take_along_axis(zs, pick, 2)[..., 0]
    normal = np.take_along_axis(ns, pick[..., None], 2)[:, :, 0]
    hit = np.isfinite(depth)
    shaded = hit & np.isfinite(normal).all(-1)
    return Layers(np.isfinite(zs).mean(2).astype(np.float32), np.where(hit, depth, 0.0).astype(np.float32),
                  np.where(shaded[..., None], normal, 0.0).astype(np.float32), hit, shaded)
