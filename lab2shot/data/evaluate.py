"""Evaluate a scene the way a DCC would: meshes (skinned where bound), point clouds, skeletons and the camera at a
time, in world space. Overlays and 2D projections of a scene render what this gives.

Skinned meshes go through UsdSkel's own skinning, so a correct overlay confirms that units, axes, skinning and lens
were all preserved in the conversion to USD.
"""

from __future__ import annotations

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdSkel

from ..io.usd import deformed_points


def skin_bindings(stage: Usd.Stage):
    """Every skeleton binding of the stage with its skeleton's query: (UsdSkel.Binding, UsdSkel.SkeletonQuery). The
    skeleton roots are listed first, so the caller may edit the stage between them."""
    cache = UsdSkel.Cache()
    for prim in [p for p in stage.Traverse() if p.IsA(UsdSkel.Root)]:
        root = UsdSkel.Root(prim)
        cache.Populate(root, Usd.PrimDefaultPredicate)
        for binding in cache.ComputeSkelBindings(root, Usd.PrimDefaultPredicate):
            yield binding, cache.GetSkelQuery(binding.GetSkeleton())


def changes(prim: Usd.Prim) -> bool:
    """Whether a prim changes over time: its points (a point cloud, a mesh) or its placement (its own transform or a
    parent's)."""
    if prim.IsA(UsdGeom.PointBased) and UsdGeom.PointBased(prim).GetPointsAttr().ValueMightBeTimeVarying():
        return True
    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying():
            return True
        prim = prim.GetParent()
    return False


def mesh_world_points(stage: Usd.Stage, time: Usd.TimeCode) -> list[tuple[Sdf.Path, np.ndarray]]:
    """(mesh path, world-space points) of every mesh, skinned where bound."""
    out: list[tuple[Sdf.Path, np.ndarray]] = []
    xf_cache = UsdGeom.XformCache(time)
    bound: set = set()
    for binding, skel_query in skin_bindings(stage):
        for target in binding.GetSkinningTargets():
            mesh_prim = target.GetPrim()
            points = deformed_points(target, skel_query, time)
            if points is not None:
                # skinned points are in skeleton space; place with the skeleton's world transform
                to_world = np.array(xf_cache.GetLocalToWorldTransform(binding.GetSkeleton().GetPrim()))
                out.append((mesh_prim.GetPath(), _apply(np.array(points), to_world)))
                bound.add(mesh_prim.GetPath())
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath() not in bound:
            points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(time))
            out.append((prim.GetPath(), _apply(points, np.array(xf_cache.GetLocalToWorldTransform(prim)))))
    return out


def world_points(stage: Usd.Stage, time: Usd.TimeCode) -> list[np.ndarray]:
    """World-space points (scene units) of every mesh, skinned where bound."""
    return [pts for _, pts in mesh_world_points(stage, time)]


def scene_points(stage: Usd.Stage, time: Usd.TimeCode) -> list[tuple[Sdf.Path, np.ndarray, np.ndarray, np.ndarray | None]]:
    """(path, world-space points [N,3], widths [N] in scene units, display colours [N,3] or None) of every point
    cloud."""
    xf_cache = UsdGeom.XformCache(time)
    clouds = [(prim.GetPath(), *cloud_at(prim, time, xf_cache)) for prim in stage.Traverse() if prim.IsA(UsdGeom.Points)]
    return [c for c in clouds if len(c[1])]


def cloud_at(prim: Usd.Prim, time: Usd.TimeCode, xf_cache: UsdGeom.XformCache | None = None):
    """One point cloud at `time`: world-space points [N,3] (none: [0,3]), widths [N] in scene units, display colours
    [N,3] or None."""
    cloud = UsdGeom.Points(prim)
    raw = cloud.GetPointsAttr().Get(time)
    if raw is None or not len(raw):
        return np.zeros((0, 3)), np.zeros(0), None
    to_world = (xf_cache or UsdGeom.XformCache(time)).GetLocalToWorldTransform(prim)
    points = _apply(np.asarray(raw, np.float64), np.array(to_world))
    widths = np.asarray(cloud.GetWidthsAttr().Get(time) or [1.0], np.float64)
    widths = widths if len(widths) == len(points) else np.full(len(points), float(widths[0]))
    color = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    rgb = np.asarray(color.ComputeFlattened(time), np.float64) if color and color.HasValue() else None
    if rgb is not None and len(rgb) != len(points):
        rgb = np.repeat(rgb[:1], len(points), 0) if len(rgb) else None
    return points, widths, rgb


def scene_meshes(stage: Usd.Stage, time: Usd.TimeCode) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(owner path, world points, triangle faces) per mesh; owner = the mesh's parent (e.g. a person)."""
    out = []
    for path, pts in mesh_world_points(stage, time):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), np.int64)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get(), np.int64)
        out.append((str(path.GetParentPath()), pts, triangulate(counts, idx)))
    return out


def triangulate(counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Polygons (corners per face, their indices one face after another) as triangles [T,3]: a fan per polygon."""
    if len(counts) and np.all(counts == 3):
        return idx.reshape(-1, 3)
    tris, start = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((idx[start], idx[start + k], idx[start + k + 1]))
        start += c
    return np.array(tris, np.int64).reshape(-1, 3)


def skeleton_joints(stage: Usd.Stage, time: Usd.TimeCode) -> list[tuple[Usd.Prim, list[str], np.ndarray, np.ndarray]]:
    """Every skeleton at `time`: (its prim, joint names, world-space joint positions [J,3], parent indices [J])."""
    cache = UsdSkel.Cache()
    xf_cache = UsdGeom.XformCache(time)
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdSkel.Skeleton):
            continue
        skel = UsdSkel.Skeleton(prim)
        query = cache.GetSkelQuery(skel)
        xforms = query.ComputeJointWorldTransforms(xf_cache)
        if not xforms:
            continue
        joints = [str(j) for j in query.GetJointOrder()]
        names = [str(n) for n in (skel.GetJointNamesAttr().Get() or [])] or [j.rsplit("/", 1)[-1] for j in joints]
        pos = np.array([np.array(m)[3, :3] for m in xforms])
        out.append((prim, names, pos, np.array(query.GetTopology().GetParentIndices())))
    return out


def _apply(points: np.ndarray, row_matrix: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return (homo @ row_matrix)[:, :3]
