"""Scene arrays (lab2shot_shared/scene_arrays.py: the layout) and Lab2Shot's packets, both ways, for every
format module read or written by its extension's worker:

- items_to_packets: what a reader found, the items the user selected, as packets: centimetres, Y up, every parent
  composed (a camera made rigid: no parent's scale reaches it), the file's own frame numbers;
- scene_arrays: a scene packet as the arrays a writer takes, in the unit it chose, Y up.

The format modules only give the file's units and axes (Axes); nothing here knows a format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lab2shot_shared import scene_arrays as sa
from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

from ..io import usd
from .camera import CameraSamples
from .packet import Packet
from .payloads import SCENE_FILE, scene_packet
from .units import DEFAULT_FPS

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # (x, y, z) -> (x, z, -y)


@dataclass(frozen=True)
class Axes:
    """How a file's lengths and axes become Lab2Shot's: centimetres per unit of the file, and the turn [3,3] that
    stands its world up (Y up, right-handed)."""

    to_cm: float = 1.0
    turn: np.ndarray = field(default_factory=lambda: np.eye(3))

    @staticmethod
    def of(to_cm: float, up: str) -> Axes:
        """A Y-up or Z-up file (what a format that records neither lets the user say)."""
        return Axes(float(to_cm), Z_UP_TO_Y_UP if up == "z" else np.eye(3))

    def matrices(self, m) -> np.ndarray:
        """Local-to-world [...,4,4] of the file -> Lab2Shot's, for local geometry scaled by `to_cm` (points(), which the
        matrices then place): the turn on the left, translations in cm."""
        out = np.array(m, np.float64)
        out[..., :3, 3] *= self.to_cm
        turn = np.eye(4)
        turn[:3, :3] = self.turn
        return turn @ out

    def points(self, p) -> np.ndarray:
        """Local lengths (points, offsets, widths) -> cm."""
        return np.asarray(p, np.float64) * self.to_cm

    def world_points(self, p) -> np.ndarray:
        """Points in the file's world -> Lab2Shot's."""
        return self.points(p) @ self.turn.T


def rigid(m: np.ndarray) -> np.ndarray:
    """Camera-to-world matrices [F,4,4] with only position and orientation (a camera under a scaled group sees the same
    as without the scale: Maya, Houdini and USD ignore a camera's scale)."""
    out = np.array(m, np.float64)
    u, _, vt = np.linalg.svd(out[:, :3, :3])
    out[:, :3, :3] = u @ vt
    return out


# ------------------------------------------------------------------ read: arrays -> packets


def item_frames(item: dict) -> list[int]:
    return [int(f) for f in np.asarray(item["frames"]).reshape(-1)]


def frames_of(items: list[dict]) -> list[int]:
    return sorted({f for item in items for f in item_frames(item)})


def _place(prim, frames: list[int], world: np.ndarray) -> None:
    """A transform op on `prim`: its local-to-world, one sample (still) or one per frame."""
    op = UsdGeom.Xformable(prim).AddTransformOp()
    if len(world) == 1:
        op.Set(Gf.Matrix4d(world[0].T.tolist()))
    else:
        for f, m in zip(frames, world):
            op.Set(Gf.Matrix4d(m.T.tolist()), Usd.TimeCode(f))


def hide(prim, item: dict) -> None:
    """The frames the item's file hides it on (`visible`: a delivery of a solve with gaps, an animator's hide), every
    kind alike."""
    if "visible" in item:
        usd.set_visible(prim, item_frames(item), np.asarray(item["visible"]).reshape(-1))


def item_subsets(item: dict) -> dict[str, np.ndarray]:
    """The subsets of a mesh item: {name: its faces} (from the format module's subset_names / subset_counts /
    subset_faces; empty when absent). Subsets are part of the mesh, not new data; written back to USD they are
    GeomSubsets."""
    if "subset_names" not in item:
        return {}
    names = [sa.text(n) for n in np.asarray(item["subset_names"]).reshape(-1)]
    cuts = np.cumsum(np.asarray(item["subset_counts"], np.int64).reshape(-1))[:-1]
    return dict(zip(names, np.split(np.asarray(item["subset_faces"], np.int64).reshape(-1), cuts)))


def model_stage(stage: Usd.Stage, items: list[dict], axes: Axes, group: str) -> None:
    """Models where their file had them, under their import's folder (io/usd.py import_path): each a Mesh with its local
    points (one shape, or one per frame: deforming) and its local-to-world as a transform op."""
    for item in items:
        frames, pts = item_frames(item), axes.points(item["points"]).astype(np.float32)
        uv = (item["uv"], item["uv_indices"]) if "uv" in item else (None, None)
        source = sa.text(item["path"])
        path = usd.free_path(stage, usd.import_path(group, source))
        usd.place(stage, path)
        mesh = usd.write_mesh(stage, path, pts[0] if len(pts) == 1 else pts,
                              np.asarray(item["indices"]), frames, *uv, counts=np.asarray(item["counts"]),
                              subsets=item_subsets(item))
        if "normals" in item:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(item["normals"], np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
        _place(mesh.GetPrim(), frames, axes.matrices(item["world"]))
        hide(mesh.GetPrim(), item)
        mesh.GetPrim().SetCustomDataByKey("lab2shot:object", source)
        usd.place(stage, path, source)


class _PerPoint:
    """One item's per-point arrays, cut into its samples and turned into Lab2Shot's units: the one way a 点云's and a
    三维曲线's arrays are read (they are cut the same way, by `counts`), and where each is placed in the scene."""

    def __init__(self, item: dict, axes: Axes, stage: Usd.Stage, group: str):
        self.item, self.axes = item, axes
        self.frames = item_frames(item)
        self.cuts = np.cumsum(np.asarray(item["counts"]).reshape(-1))[:-1]
        self.source = sa.text(item["path"])
        self.path = usd.free_path(stage, usd.import_path(group, self.source))
        usd.place(stage, self.path)

    def split(self, key: str, dtype, cm: bool = False):
        """One per-point array per sample ("" it has none): lengths (points, widths, velocities) turned into cm."""
        if key not in self.item:
            return None
        values = np.asarray(self.item[key], np.float64)
        return np.split((self.axes.points(values) if cm else values).astype(dtype), self.cuts)

    def place(self, stage: Usd.Stage, geom) -> None:
        _place(geom.GetPrim(), self.frames, self.axes.matrices(self.item["world"]))
        hide(geom.GetPrim(), self.item)
        geom.GetPrim().SetCustomDataByKey("lab2shot:object", self.source)
        usd.place(stage, self.path, self.source)


def points_stage(stage: Usd.Stage, items: list[dict], axes: Axes, group: str) -> None:
    """Point clouds where their file had them, under their import's folder: their local points per sample, colours,
    width, and a tracked cloud's ids, velocities and visible; local-to-world as a transform op."""
    for item in items:
        got = _PerPoint(item, axes, stage, group)
        widths = axes.points(item["widths"]) if "widths" in item else np.ones(0)
        visible = got.split("point_visible", np.int32)
        cloud = usd.write_points(stage, got.path, got.frames, got.split("points", np.float32, cm=True), got.split("colors", np.float32),
                                 primvars={"visible": visible} if visible is not None else None,
                                 width_cm=float(np.median(widths)) if len(widths) else 1.0,
                                 ids=got.split("ids", np.int64), velocities=got.split("velocities", np.float32, cm=True))
        got.place(stage, cloud)


def curves_stage(stage: Usd.Stage, items: list[dict], axes: Axes, group: str) -> None:
    """三维曲线 where their file had them, under their import's folder: each sample's curves (how many points each
    curve has) and their local points, widths one per point, colours and normals; local-to-world as a transform op.
    Cut exactly as a point cloud's per-point arrays are (`counts`), with `curve_counts` cutting the per-curve ones."""
    for item in items:
        got = _PerPoint(item, axes, stage, group)
        per_curve = np.cumsum(np.asarray(item["curve_counts"]).reshape(-1))[:-1]
        curves = usd.write_curves(stage, got.path, got.frames,
                                  np.split(np.asarray(item["curve_vertex_counts"], np.int32).reshape(-1), per_curve),
                                  got.split("points", np.float32, cm=True), got.split("widths", np.float32, cm=True),
                                  got.split("colors", np.float32), got.split("normals", np.float32))
        got.place(stage, curves)


def gaussian_stage(stage: Usd.Stage, items: list[dict], axes: Axes, group: str) -> None:
    from .gaussian import write

    for item in items:
        got = _PerPoint(item, axes, stage, group)
        arrays = {k: got.split(k, np.float32, cm=k in ("points", "scales"))
                  for k in ("points", "scales", "rotations", "opacity", "sh")}
        samples = [{k: v[i] for k, v in arrays.items()} for i in range(len(arrays["points"]))]
        got.place(stage, write(stage, got.path, got.frames, samples))


def character_stage(stage: Usd.Stage, items: list[dict], axes: Axes, group: str) -> None:
    """骨架动画 and 蒙皮角色 where their file had them, under their import's folder: each a skeleton root with its
    skeleton, animation (a sample at each of its frames: an animator's keys stay keys), and the meshes skinned to it
    with their blend shapes (none: bones alone)."""
    for item in items:
        frames = item_frames(item)
        meshes = [usd.SkinnedMesh(sa.text(m["name"]), axes.world_points(m["points"]), np.asarray(m["counts"]), np.asarray(m["indices"]),
                                  np.asarray(m["joint_indices"]), np.asarray(m["joint_weights"]),
                                  m.get("uv"), m.get("uv_indices"), sa.texts(m["shapes"]) if "shapes" in m else [],
                                  axes.points(m["shape_offsets"]) @ axes.turn.T if "shape_offsets" in m else None,
                                  np.asarray(m["shape_weights"]) if "shape_weights" in m else None)
                  for m in item["meshes"]]
        source = sa.text(item["path"])
        root = usd.write_rig(stage, sa.text(item["name"]), sa.texts(item["joints"]), np.asarray(item["parents"]),
                             axes.matrices(item["bind"]), axes.matrices(item["anim"]), frames, meshes,
                             path=usd.free_path(stage, usd.import_path(group, source)), source=source)
        root.GetPrim().SetCustomDataByKey("lab2shot:object", source)
        hide(root.GetPrim(), item)
        skins = [p for p in root.GetPrim().GetChildren() if p.IsA(UsdGeom.Mesh)]  # written in the item's order
        for prim, mesh in zip(skins, item["meshes"]):  # one hidden on its own (a LOD)
            hide(prim, {"frames": item["frames"], **({"visible": mesh["visible"]} if "visible" in mesh else {})})
        # the reader said whether that path ends at the root joint itself or at a group of its own: kept here, so
        # writing it back never has to guess from the names
        root.GetPrim().SetCustomDataByKey(usd.ROOT_AT_PATH, bool(np.asarray(item.get("root_at_path", False)).reshape(-1)[0])
                                          if "root_at_path" in item else False)


def camera_from_arrays(item: dict, axes: Axes, width: int, info: dict) -> CameraSamples:
    """A camera item as a CameraSamples: rigid, cm, Y up; the picture size the file records, else `width` and the
    filmback's proportions."""
    frames = item_frames(item)
    h_ap, v_ap = np.asarray(item["h_aperture_mm"], np.float64), np.asarray(item["v_aperture_mm"], np.float64)
    if "resolution" in item:
        size = tuple(int(v) for v in np.asarray(item["resolution"]).reshape(-1)[:2])
    else:
        size = (width, int(round(width * float(np.median(v_ap)) / float(np.median(h_ap)))))
    mats = rigid(axes.matrices(item["world"]))
    overscan = np.asarray(item["overscan"], np.float64).reshape(4) * [size[0], size[1], size[0], size[1]]
    return CameraSamples(tuple(frames), mats, np.asarray(item["focal_mm"], np.float64), h_ap, v_ap, *size,
                         {**info, "object": sa.text(item["path"])}, np.asarray(item["center_mm"], np.float64),
                         float(item["pixel_aspect"]), tuple(int(round(v)) for v in overscan),
                         plate=sa.text(item["plate"]) if "plate" in item else "",
                         **CameraSamples.lens_from_properties(json.loads(sa.text(item["properties"]))))


def camera_arrays(s: CameraSamples) -> dict:
    """A camera's lens beyond the focal length and film back as a writer's arrays: the centre offset per frame (mm),
    the pixel aspect, the overscan as parts of the picture's width and height (left, top, right, bottom) and the
    named properties as JSON text."""
    size = np.array([s.width, s.height, s.width, s.height], np.float64)
    return {"center_mm": s.center_mm, "pixel_aspect": np.float64(s.pixel_aspect),
            "overscan": np.asarray(s.overscan, np.float64) / size if any(s.overscan) else np.zeros(4),
            "properties": np.array(json.dumps(s.lens_properties(), ensure_ascii=False)),
            "plate": np.array(s.plate)}  # its backplate (io/usd.py PLATE), carried with the camera like the other facts


def items_to_packets(npz: Path, chosen: dict[str, list[str]], axes: Axes, width: int, outs: dict[str, Path],
                     info: dict, group: str, owner: str) -> dict[str, Packet]:
    """The chosen items of a reader's arrays (kind -> item paths) as packets, one per kind in `outs` (kind ->
    its packet folder): 相机 (one), 模型, 点云, 骨架动画, 蒙皮角色; each where its file had it under the import's folder
    `group` (io/usd.py import_group, import_path)."""
    top, items = sa.load(npz)
    stored = {"skeleton": "character"}  # bones alone are a character item without meshes in the arrays
    picked = {kind: [i for i in items[stored.get(kind, kind)] if sa.text(i["key"]) in paths] for kind, paths in chosen.items()}
    out: dict[str, Packet] = {}
    for kind, folder in outs.items():
        these = picked.get(kind) or []
        if kind == "camera":
            samples = camera_from_arrays(these[0], axes, width, info)
            source = samples.info["object"]
            def customize(cam, item=these[0]):
                usd.mark_group(cam.GetPrim().GetStage(), group, owner)
                hide(cam.GetPrim(), item)

            out[kind] = samples.write(folder, name=source.rsplit("/", 1)[-1] or "camera", path=usd.import_path(group, source), source=source,
                                      customize=customize)
            continue
        frames = frames_of(these)
        stage = usd.create_stage(frames, info)
        {"model": model_stage, "points": points_stage, "gaussian": gaussian_stage, "curves": curves_stage,
         "skeleton": character_stage, "character": character_stage}[kind](stage, these, axes, group)
        usd.mark_group(stage, group, owner)  # the folder of this very import cook
        usd.save_stage(stage, folder / SCENE_FILE)
        meta = {"scale": "metric"} if kind == "points" else {}
        out[kind] = scene_packet(folder, frames, f"scene.{kind}", **meta)
    return out


# ------------------------------------------------------------------ write: a scene packet -> arrays


def _frames_where(src: Packet, changing: bool) -> list[int]:
    frames = [int(f) for f in src.meta["frames"]]
    return frames if changing and frames else frames[:1] or [0]


class _Worlds:
    """Prims' world matrices at the frames asked ([F,4,4], columns as vectors, lengths × `scale`), each group's worked
    out once and shared by everything under it: a transform that never changes is read once, so a set of thousands
    of props under a moving group costs one product per prop, not one USD evaluation per prop and frame."""

    def __init__(self, scale: float) -> None:
        self.scale = scale
        self.memo: dict = {}

    def __call__(self, prim, frames: list[int]) -> np.ndarray:
        out = self._unscaled(prim, tuple(frames)).copy()
        out[:, :3, 3] *= self.scale
        return out

    def _unscaled(self, prim, frames: tuple[int, ...]) -> np.ndarray:
        key = (prim.GetPath(), frames)
        if key in self.memo:
            return self.memo[key]
        parent = prim.GetParent()
        xf = UsdGeom.Xformable(prim) if prim.IsA(UsdGeom.Xformable) else None
        if xf and xf.TransformMightBeTimeVarying():
            local = np.stack([np.array(xf.GetLocalTransformation(Usd.TimeCode(f))).T for f in frames])
        else:
            local = np.array(xf.GetLocalTransformation(Usd.TimeCode(frames[0]))).T[None] if xf else np.eye(4)[None]
        if not parent or parent.IsPseudoRoot() or (xf and xf.GetResetXformStack()):
            world = np.broadcast_to(local, (len(frames), 4, 4)).copy()
        else:
            world = self._unscaled(parent, frames) @ local
        self.memo[key] = world
        return world


def _moves(prim) -> bool:
    from .evaluate import changes

    return changes(prim)


def _hides(prim) -> bool:
    """Whether the prim's visibility, its own or an ancestor's, is authored as anything but shown (an animator's hide,
    a baked person on the frames its solve did not reach): then its item is sampled on every frame and carries
    `visible`, as a character does."""
    while prim and not prim.IsPseudoRoot():
        attr = UsdGeom.Imageable(prim).GetVisibilityAttr() if prim.IsA(UsdGeom.Imageable) else None
        if attr and attr.HasAuthoredValue() and (attr.ValueMightBeTimeVarying() or attr.Get() == UsdGeom.Tokens.invisible):
            return True
        prim = prim.GetParent()
    return False


def _shown(prim, frames: list[int]) -> dict:
    """The item's `visible` on its frames when it is hidden on some of them (usd.visible_at), else nothing."""
    on = usd.visible_at(prim, frames)
    return {} if on.all() else {"visible": on.astype(np.int32)}


def _mesh_arrays(mesh: UsdGeom.Mesh, times, scale: float) -> dict:
    pts = model_points(mesh, times, scale)
    out = {"counts": np.asarray(mesh.GetFaceVertexCountsAttr().Get(), np.int32),
           "indices": np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int32), "points": pts}
    st = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if st and st.HasValue():
        corners = len(out["indices"])
        if st.GetInterpolation() == UsdGeom.Tokens.faceVarying:
            idx = st.GetIndices()
            out["uv"], out["uv_indices"] = np.asarray(st.Get(), np.float32), np.asarray(idx if idx else range(corners), np.int32)
        elif st.GetInterpolation() == UsdGeom.Tokens.vertex:
            out["uv"], out["uv_indices"] = np.asarray(st.ComputeFlattened(), np.float32), out["indices"].copy()
    normals = mesh.GetNormalsAttr().Get()
    if normals is not None and mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying and len(normals) == len(out["indices"]):
        out["normals"] = np.asarray(normals, np.float32)
    parts = usd.mesh_subsets(mesh)  # the mesh's subsets, exported by the format modules as subset_names / subset_counts / subset_faces
    if parts:
        out["subset_names"] = np.array(list(parts))
        out["subset_counts"] = np.array([len(f) for f in parts.values()], np.int64)
        out["subset_faces"] = np.concatenate([f for f in parts.values()]).astype(np.int32)
    return out


def model_points(mesh: UsdGeom.Mesh, times, scale: float = 1.0) -> np.ndarray:
    """A mesh's points in its own space at `times` (frames or time codes) [S,V,3], × `scale`."""
    return np.stack([np.asarray(mesh.GetPointsAttr().Get(Usd.TimeCode(t)), np.float32) for t in times]) * scale


def cloud_sample(prim, frame: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A point cloud at `frame` in its own space: points [N,3], display colours [N,3] (0.7 grey when it has none) and
    widths [N], lengths × `scale`."""
    from .evaluate import cloud_at

    t = Usd.TimeCode(frame)
    local = np.asarray(UsdGeom.Points(prim).GetPointsAttr().Get(t) or [], np.float64).reshape(-1, 3)
    _, widths, colors = cloud_at(prim, t, UsdGeom.XformCache(t))
    colors = colors if colors is not None else np.full((len(local), 3), 0.7)
    widths = np.broadcast_to(widths, (len(local),)) if len(local) else np.zeros(0)
    return local * scale, np.asarray(colors, np.float32), np.asarray(widths, np.float32) * scale


def curve_sample(prim, frame: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A set of 三维曲线 at `frame` in its own space: how many points each curve has [C] int, the points [N,3],
    display colours [N,3] (0.7 grey when it has none) and widths [N], lengths x `scale`. A constant width is spread
    over the points here, so everything downstream sees one layout."""
    from pxr import UsdGeom

    t = Usd.TimeCode(frame)
    curves = UsdGeom.BasisCurves(prim)
    counts = np.asarray(curves.GetCurveVertexCountsAttr().Get(t) or [], np.int32).reshape(-1)
    local = np.asarray(curves.GetPointsAttr().Get(t) or [], np.float64).reshape(-1, 3)
    colour = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
    colors = np.asarray(colour.ComputeFlattened(t) if colour and colour.HasValue() else [], np.float32).reshape(-1, 3)
    if len(colors) != len(local):
        colors = np.full((len(local), 3), 0.7, np.float32)
    widths = np.asarray(curves.GetWidthsAttr().Get(t) or [], np.float64).reshape(-1)
    if len(widths) != len(local):
        widths = np.full(len(local), float(widths[0]) if len(widths) else 0.2)
    return counts, local * scale, colors, widths.astype(np.float32) * scale


def _sampled_at(src: Packet, prim, samples: bool) -> tuple[list[int], list[int], bool]:
    """(the frames a 点云 or a set of 三维曲线 has, the frames read now, whether its points change): read at every
    frame when they do, else once; with `samples=False` (the 3D viewer, which asks for a chunk at a time) only its
    first sample comes with the base."""
    frames = _frames_where(src, _moves(prim) or _hides(prim))
    changing = UsdGeom.PointBased(prim).GetPointsAttr().ValueMightBeTimeVarying()
    return frames, (frames if changing and samples else frames[:1]), changing


def scene_arrays(src: Packet, scale: float = 1.0, samples: bool = True, stage: Usd.Stage | None = None,
                 fps: float = DEFAULT_FPS) -> sa.SceneArrays:
    """A scene packet as a writer's arrays (lengths × `scale`: 0.01 for metres), Y up: every model, point cloud,
    set of 3D curves, camera and character in it, each at the frames it changes on (one sample when it never does).

    `samples=False` (the 3D viewer, which reads them a few frames at a time with model_points / cloud_sample): a
    deforming model or a changing point cloud comes with its first sample only, marked `per_frame`."""
    from .evaluate import skin_bindings
    from .scene import open_scene

    stage = stage or open_scene([src])
    out = sa.SceneArrays(float(fps), 1.0 / scale if scale else 1.0)
    bound = {b.GetSkeleton().GetPrim().GetPath(): b.GetSkinningTargets() for b, _ in skin_bindings(stage)}
    skinned = {t.GetPrim().GetPath() for targets in bound.values() for t in targets}
    cache = UsdSkel.Cache()
    worlds = _Worlds(scale)
    for prim in stage.Traverse():  # every skeleton, with the meshes skinned to it (a rig alone is a character too)
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            _character(out, stage, skel, cache.GetSkelQuery(skel), bound.get(prim.GetPath(), []), _frames_where(src, True), scale)
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Mesh) and prim.GetPath() not in skinned:
            mesh = UsdGeom.Mesh(prim)
            deforming = mesh.GetPointsAttr().ValueMightBeTimeVarying()
            frames = _frames_where(src, _moves(prim) or _hides(prim))
            times = [Usd.TimeCode(f) for f in frames] if deforming and samples else [Usd.TimeCode(frames[0])]
            item = out.add("model", usd.name_of(prim), path, frames, worlds(prim, frames), shown=usd.shown_path(prim),
                           **_mesh_arrays(mesh, times, scale), **_shown(prim, frames))
            if deforming and not samples:
                item["per_frame"] = np.array(True)
        elif prim.IsA(UsdGeom.Points) and prim.GetCustomDataByKey("lab2shot:gaussian"):
            from .gaussian import changing, sample

            moves = changing(prim)
            frames = _frames_where(src, moves or _moves(prim) or _hides(prim))
            taken = frames if moves and samples else frames[:1]
            read = [sample(prim, f, scale) for f in taken]
            arrays = {k: np.concatenate([s[k] for s in read]) for k in read[0]}
            item = out.add("gaussian", usd.name_of(prim), path, frames, worlds(prim, frames),
                           shown=usd.shown_path(prim), counts=np.array([len(s["points"]) for s in read]),
                           **arrays, **_shown(prim, frames))
            if moves and not samples:
                item["per_frame"] = np.array(True)
        elif prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.BasisCurves):
            curves = prim.IsA(UsdGeom.BasisCurves)
            frames, taken, changing = _sampled_at(src, prim, samples)
            if curves:
                read = [curve_sample(prim, f, scale) for f in taken]
                counts = [len(c) for c, _, _, _ in read]
                arrays = {"curve_counts": np.array(counts, np.int64),
                          "curve_vertex_counts": np.concatenate([c for c, _, _, _ in read]).astype(np.int32)}
                read = [(p, c, w) for _, p, c, w in read]
            else:
                read = [cloud_sample(prim, f, scale) for f in taken]
                arrays = _tracked(prim, taken, [len(p) for p, _, _ in read], scale)
            item = out.add("curves" if curves else "points", usd.name_of(prim), path, frames, worlds(prim, frames),
                           shown=usd.shown_path(prim),
                           counts=np.array([len(p) for p, _, _ in read], np.int64), points=np.concatenate([p for p, _, _ in read]),
                           colors=np.concatenate([c for _, c, _ in read]), widths=np.concatenate([w for _, _, w in read]),
                           **arrays, **_shown(prim, frames))
            if changing and not samples:
                item["per_frame"] = np.array(True)
        elif prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            lens_moves = any(a.ValueMightBeTimeVarying() for a in (cam.GetFocalLengthAttr(), cam.GetHorizontalApertureOffsetAttr(),
                                                                     cam.GetVerticalApertureOffsetAttr()))
            frames = _frames_where(src, _moves(prim) or lens_moves or _hides(prim))
            res = usd.camera_resolution(prim)
            s = CameraSamples.from_prim(prim, frames, width=res[0] if res else 0, height=res[1] if res else 0)
            mats = s.cam_to_world.copy()
            mats[:, :3, 3] *= scale
            out.add("camera", usd.name_of(prim), path, frames, mats, shown=usd.shown_path(prim), focal_mm=s.focal_mm, h_aperture_mm=s.h_aperture_mm,
                    v_aperture_mm=s.v_aperture_mm, resolution=np.array(res, np.int64) if res else None, **camera_arrays(s),
                    **_shown(prim, frames))
    return out


def _tracked(prim, frames: list[int], counts: list[int], scale: float) -> dict:
    """A point cloud's per-point identity and motion, when it has them (3D tracks, locators): ids, velocities (local,
    lengths × `scale` per second) and point_visible (its primvar visible), each only when every sample has one for every
    point."""
    cloud = UsdGeom.Points(prim)
    visible = UsdGeom.PrimvarsAPI(prim).GetPrimvar("visible")
    found = {"ids": (cloud.GetIdsAttr(), np.int64, 1.0), "velocities": (cloud.GetVelocitiesAttr(), np.float32, scale),
             "point_visible": (visible.GetAttr() if visible else None, np.int32, 1.0)}
    out = {}
    for name, (attr, dtype, factor) in found.items():
        if attr is None or not attr.HasAuthoredValue():
            continue
        values = [np.asarray(attr.Get(Usd.TimeCode(f)) or [], np.float64) for f in frames]
        if all(len(v) == n for v, n in zip(values, counts)):
            joined = np.concatenate(values) if values else np.zeros(0)
            out[name] = (joined.reshape(-1, 3) if name == "velocities" else joined) * factor
            out[name] = out[name].astype(dtype)
    return out


def _character(out: sa.SceneArrays, stage: Usd.Stage, skel: UsdSkel.Skeleton, query, targets, frames: list[int],
               scale: float) -> None:
    """One skeleton with the meshes skinned to it (`targets`): joints in world space at the bind pose and per frame
    (their names: the skeleton's jointNames, else the last part of each joint's path), each mesh in world space at the
    bind pose (its geomBindTransform and the skeleton's placement applied), its skin in the skeleton's joint order, its
    blend shapes and their weights per frame."""
    prim = skel.GetPrim()
    order = [str(j) for j in query.GetJointOrder()]
    names = usd.joint_names(skel)
    topo = query.GetTopology()
    parents = np.asarray(topo.GetParentIndices(), np.int32)
    first = Usd.TimeCode(frames[0])
    skel_world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(first)).T
    authored = skel.GetBindTransformsAttr().Get()
    bind = np.array([np.array(m).T for m in authored]) if authored and len(authored) == len(order) else \
        np.array([np.array(m).T for m in query.ComputeJointSkelTransforms(Usd.TimeCode.Default(), True)])  # the rest pose
    bind_world = skel_world @ bind
    anim = []
    for f in frames:
        t = Usd.TimeCode(f)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)).T
        anim.append(xf @ np.array([np.array(m).T for m in query.ComputeJointSkelTransforms(t)]))
    bind_world[..., :3, 3] *= scale
    anim = np.stack(anim)
    anim[..., :3, 3] *= scale
    owner = prim.GetParent() if prim.GetParent().IsA(UsdSkel.Root) else prim
    # shown where its root is and any of its meshes is; a mesh hidden on its own (a LOD, a proxy) carries its `visible`
    meshes_on = [usd.visible_at(t.GetPrim(), frames) for t in targets]
    shown_on = usd.visible_at(owner, frames) & (np.any(meshes_on, axis=0) if meshes_on else True)
    item = out.add("character", usd.name_of(owner), str(owner.GetPath()), frames, shown=usd.shown_path(owner),
                   visible=None if shown_on.all() else shown_on.astype(np.int32), joints=np.array(names),
                   parents=parents, bind=bind_world, anim=anim,
                   root_at_path=bool(owner.GetCustomDataByKey(usd.ROOT_AT_PATH)))
    anim_query = query.GetAnimQuery()
    anim_shapes = [str(s) for s in (anim_query.GetBlendShapeOrder() if anim_query else [])]
    shape_weights = np.stack([np.asarray(anim_query.ComputeBlendShapeWeights(Usd.TimeCode(f)), np.float64) for f in frames]) \
        if anim_shapes else np.zeros((len(frames), 0))
    for target, mesh_on in zip(targets, meshes_on):
        mesh = UsdGeom.Mesh(target.GetPrim())
        arrays = _mesh_arrays(mesh, [first], 1.0)
        geom = np.array(target.GetGeomBindTransform(first)).T
        to_world = skel_world @ geom
        pts = arrays["points"][0].astype(np.float64)
        arrays["points"] = (pts @ to_world[:3, :3].T + to_world[:3, 3]) * scale
        idx, wts = target.ComputeVaryingJointInfluences(len(pts), first)
        k = target.GetNumInfluencesPerComponent()
        idx, wts = np.asarray(idx, np.int32).reshape(-1, k), np.asarray(wts, np.float32).reshape(-1, k)
        mapper = target.GetJointMapper()
        if mapper and not mapper.IsIdentity():
            own = [str(j) for j in target.GetJointOrder()]
            idx = np.array([order.index(own[i]) for i in idx.reshape(-1)], np.int32).reshape(idx.shape)
        arrays.update(joint_indices=idx, joint_weights=wts)
        if target.HasBlendShapes():
            api = UsdSkel.BindingAPI(mesh.GetPrim())
            names = [str(s) for s in api.GetBlendShapesAttr().Get()]
            shapes = [UsdSkel.BlendShape(stage.GetPrimAtPath(p)) for p in api.GetBlendShapeTargetsRel().GetTargets()]
            offsets = np.zeros((len(names), len(pts), 3))
            for b, shape in enumerate(shapes):
                idx_b = shape.GetPointIndicesAttr().Get()
                off = np.asarray(shape.GetOffsetsAttr().Get(), np.float64)
                offsets[b, np.asarray(idx_b) if idx_b else np.arange(len(off))] = off
            arrays.update(shapes=np.array(names), shape_offsets=offsets @ to_world[:3, :3].T * scale,
                          shape_weights=np.stack([shape_weights[:, anim_shapes.index(n)] if n in anim_shapes else np.zeros(len(frames))
                                                  for n in names], -1))
        if not (mesh_on == shown_on).all():
            arrays["visible"] = mesh_on.astype(np.int32)
        sa.SceneArrays.add_mesh(item, usd.name_of(mesh.GetPrim()), **arrays)
