"""Open3D worker: 点云转网格 (Poisson / Ball Pivoting / Alpha Shape). Runs inside third_party/open3d/.venv with
open3d on PYTHONPATH from pip; never imports Lab2Shot core.

    python worker.py <job.json>

Input:  job.inputs["points"] -> points.npz {points [N,3] float32 cm, colors [N,3] float32 0..1 when the cloud has
        them}, written by the node's prepare().
Output: raw/mesh.npz {vertices [V,3] float32 cm, triangles [T,3] int32, colors [V,3] float32 when the input had
        colors} + result.json.

Official Open3D calls (open3d pinned in requirements.txt), used exactly as upstream documents them:

    Poisson   o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=...)          -> (mesh, densities)
    Ball      o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)        -> mesh
    Alpha     o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)          -> mesh

Poisson and Ball Pivoting need normals (the docs state "Has to contain normals."), so the worker estimates them
first with the standard two calls (estimate_normals + orient_normals_consistent_tangent_plane); Poisson's quality
depends on consistently oriented normals. Alpha Shape uses no normals: none are estimated for it. The normal radius
is derived from the cloud's own average nearest-neighbour distance, so it adapts to the input density.

Poisson closes the surface everywhere, also where there are no points; as Open3D's own tutorial does
(docs/jupyter/geometry/surface_reconstruction.ipynb), the vertices whose density is in the lowest 1 % are removed
(remove_vertices_by_mask(densities < quantile(densities, 0.01))).

On an open scene (a street) that is not enough: Poisson still extrapolates large surfaces between and beyond the
points (buildings hanging down like stalactites, the ground pulled into a ramp). So after the density trim every
vertex farther from the input cloud than `trim_distance` x the point spacing (the median nearest-neighbour distance)
is removed with its triangles, then the connected pieces left with fewer than 1 % of the triangles are dropped.
trim_distance 0 turns this off.

Vertex colours: when the cloud has colours, every mesh vertex takes the colour of its nearest input point.

Ball radius and alpha left empty: taken from the cloud's mean nearest-neighbour distance (BALL_NN, ALPHA_NN times
it), so a cloud in any unit (a relative-scale solve) gets a mesh; a value given is used as it is (cm).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from lab2shot_worker import fail, serve
from lab2shot_worker.run import Run

METHOD_LABELS = {"poisson": "Poisson", "ball": "Ball Pivoting", "alpha": "Alpha Shape"}
DENSITY_QUANTILE = 0.01  # Open3D's tutorial: Poisson vertices below this density quantile are removed
BALL_NN = 2.0  # empty ball radius: this many mean nearest-neighbour distances (radii r, 2r, 4r)
ALPHA_NN = 4.0  # empty alpha: this many mean nearest-neighbour distances
SMALL_PIECE = 0.01  # after the distance trim, connected pieces with fewer than this share of the triangles go


def _nearest(xyz: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every query point: the index of the nearest point of `xyz` and the distance to it (one vectorized
    Open3D nearest-neighbour search)."""
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(xyz, np.float32)))
    nns.knn_index()
    idx, d2 = nns.knn_search(o3d.core.Tensor(np.ascontiguousarray(query, np.float32)), 1)
    return idx.numpy()[:, 0].astype(np.int64), np.sqrt(np.maximum(d2.numpy()[:, 0], 0.0))


def _trim_far(mesh: o3d.geometry.TriangleMesh, xyz: np.ndarray, limit: float) -> dict:
    """Removes the vertices farther than `limit` from the cloud (with their triangles), then the small pieces."""
    before = len(mesh.triangles)
    _, dist = _nearest(xyz, np.asarray(mesh.vertices))
    far = dist > limit
    mesh.remove_vertices_by_mask(far)
    after_far = len(mesh.triangles)
    pieces, sizes, _ = mesh.cluster_connected_triangles()
    pieces, sizes = np.asarray(pieces), np.asarray(sizes)
    small = sizes[pieces] < SMALL_PIECE * max(after_far, 1) if len(pieces) else np.zeros(0, bool)
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    return {"far_vertices": int(far.sum()), "far_triangles": int(before - after_far),
            "small_piece_triangles": int(small.sum()), "trimmed_share": float(1.0 - len(mesh.triangles) / max(before, 1))}


def _spacing(pcd: o3d.geometry.PointCloud) -> float:
    """The cloud's mean nearest-neighbour distance (its own unit)."""
    dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    return max(float(np.mean(dists)) if len(dists) else 0.01, 1e-9)


def _estimate_normals(pcd: o3d.geometry.PointCloud, spacing: float) -> o3d.geometry.PointCloud:
    """Normals the official reconstruction functions require: radius from the cloud's own mean nearest-neighbour
    distance (the usual 3x rule of thumb), then oriented consistently for Poisson."""
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(3.0 * spacing, 1e-6), max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def _reconstruct(pcd: o3d.geometry.PointCloud, method: str, params) -> tuple[o3d.geometry.TriangleMesh, dict]:
    spacing = _spacing(pcd)
    used: dict = {"spacing": spacing}
    xyz = np.asarray(pcd.points).copy()
    if method == "poisson":
        pcd = _estimate_normals(pcd, spacing)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(params["depth"]))
        densities = np.asarray(densities)
        low = densities < np.quantile(densities, DENSITY_QUANTILE)
        mesh.remove_vertices_by_mask(low)
        used["trimmed_vertices"] = int(low.sum())
        k = float(params.get("trim_distance") or 0.0)
        if k > 0:
            nn = np.asarray(pcd.compute_nearest_neighbor_distance())
            median = max(float(np.median(nn)) if len(nn) else spacing, 1e-9)
            used["trim_limit"] = k * median
            used.update(_trim_far(mesh, xyz, k * median))
    elif method == "ball":
        pcd = _estimate_normals(pcd, spacing)
        r = float(params["radius"]) if params.get("radius") else BALL_NN * spacing
        used["radius"] = r
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([r, r * 2.0, r * 4.0]))
    else:  # alpha: no normals needed
        a = float(params["alpha"]) if params.get("alpha") else ALPHA_NN * spacing
        used["alpha"] = a
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, a)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()
    return mesh, used


def main(job_path: str) -> None:
    run = Run.start(job_path, "open3d.mesh_from_points", "Open3D", gpu=False)
    job, params = run.job, run.params
    method = params["method"]
    with np.load(job.inputs["points"]) as d:
        xyz = np.asarray(d["points"], np.float32)
        colors = np.asarray(d["colors"], np.float32) if "colors" in d.files else None
    if not len(xyz):
        fail("E-OPEN3D-NOPOINTS")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    run.stage("reconstruct_mesh")
    mesh, used = _reconstruct(pcd, method, params)
    if not len(mesh.vertices) or not len(mesh.triangles):
        fail("E-OPEN3D-FAILED")
    verts = np.asarray(mesh.vertices, np.float32)
    tris = np.asarray(mesh.triangles, np.int32)
    out = {"vertices": verts, "triangles": tris}
    if colors is not None and len(colors) == len(xyz):
        idx, _ = _nearest(xyz, verts)
        out["colors"] = colors[idx]
        used["colored"] = True
    np.savez_compressed(job.raw_dir / "mesh.npz", **out)
    run.finish([], vertices=int(len(verts)), triangles=int(len(tris)), method=METHOD_LABELS[method], **used)


if __name__ == "__main__":
    serve(main)
