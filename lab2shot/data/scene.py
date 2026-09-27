"""USD scene packets: pack, transform, ground, bake, export, the kinds a scene holds, cameras, what the camera sees
(overlays, 2D layers). Reading and writing DCC files is the format modules' (lab2shot/nodes/formats.py).

Scene packets hold scene.usd in Lab2Shot units (cm, Y-up, source frame
numbers). Pack and transform only author small layers on top of their inputs
(sublayers), so they are instant; export flattens everything into one file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

from ..errors import Invalid
from ..io import FileProblem, usd
from ..messages import Msg
from ..io.usd import ROOT_PATH, STAGE_FPS, rescale_stage
from .frames import union
from .packet import Packet
from .payloads import SCENE_FILE, scene_packet
from .units import CM_TO_M


def _layer_under(layer: Sdf.Layer, files: list[Path]) -> None:
    """Scene files as sublayers of `layer`, the first strongest, their frames lined up by number (frames are the
    shot's identity). Every Lab2Shot stage has the same time base (io/usd.py STAGE_FPS: a time code is a
    frame), so USD never retimes one under another and no offset is needed."""
    layer.subLayerPaths = [str(p) for p in files]


def _new_layer_stage(path: Path, files: list[Path], frames: list[int]) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    usd.apply_conventions(stage, frames)
    _layer_under(stage.GetRootLayer(), files)
    return stage


def _file(p: Packet) -> Path:
    return p.path(SCENE_FILE)


def open_scene(packets: list[Packet]) -> Usd.Stage:
    """Read-only composed view of several scene packets (first is strongest), every packet's frames at their own
    numbers."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.timeCodesPerSecond = STAGE_FPS
    layer.framesPerSecond = STAGE_FPS
    _layer_under(layer, [_file(p) for p in packets])
    return Usd.Stage.Open(layer)


def pack(inputs: list[Packet], out: Path) -> Packet:
    """The inputs as one scene, each file layered in as it is, in order.

    A list of scenes among them is unpacked into its items, each under /shot/<its name> (the name the artist sees, from
    「逐项开始」 or 「命名」). Nothing is ever renamed here: two things of one name are refused (B-NAME-SAME), before the
    cook by the check on the input (nodes/expects.py DistinctNames) and here by `items_meta`, so no result ever carries
    a name nobody chose. A transform on an input's /shot itself (「3D 变换」, 「自动落地」, a Z-up file turned upright)
    would move what every other input has under /shot too: it is moved onto that input's own prims, the input going in
    as a copy of its file made so (_prepared)."""
    from .types import DATA_TYPES

    parts = scenes(inputs)
    frames = union([p for _, p in parts])
    under = []  # per input: its names under /shot, what to rename, whether /shot carries a transform, whose import each is
    for name, p in parts:
        holds, moved = _shot_of(p)
        # an item holding several things (a person's two hands) is gathered into one group `/shot/<item name>` rather
        # than refused (refusing would make the whole 「手部动作 · HaMeR · 多人」 card fail)
        group = name if name and len(holds) != 1 else ""
        under.append(([name] if name else holds,
                      {holds[0]: name} if name and not group and holds and holds[0] != name else {}, moved,
                      _import_groups(p), group))
    seen: dict[str, str] = {}  # name -> the cook that made it, for an import's folder ("" anything else)
    same = []
    for names, _renamed, _moved, groups, _group in under:
        for n in names:
            owner = groups.get(n, "")
            # one import node's outputs share its folder (its camera and its models are one thing); two things of
            # one name are said before anything is written, never renamed on the quiet
            if n in seen and not (owner and seen[n] == owner):
                same.append(n)
            seen.setdefault(n, owner)
    if same:
        raise Invalid(Msg("B-NAME-SAME", kind=DATA_TYPES["scene"].label, names=sorted(set(same))))
    files = []
    for i, ((_names, renamed, moved, _groups, group), (_name, p)) in enumerate(zip(under, parts)):
        files.append(_prepared(p, renamed, out / f"input_{i}.usda", group) if renamed or moved or group else _file(p))
    stage = _new_layer_stage(out / SCENE_FILE, files, frames)
    stage.GetRootLayer().Save()
    meta = {k: v for _, p in parts for k, v in p.meta.items() if k in ("width", "height")}
    return scene_packet(out, frames, **meta)


def _import_groups(p: Packet) -> dict[str, str]:
    """The children of a scene packet's /shot that are an import's folder -> the cook that made it (io/usd.py
    mark_group): one import node's camera and models are one folder, not two things of one name."""
    stage = Usd.Stage.Open(str(p.path(SCENE_FILE)))
    shot = stage.GetPrimAtPath(ROOT_PATH)
    return {c.GetName(): str(c.GetCustomDataByKey(usd.IMPORT_GROUP)) for c in shot.GetChildren()
            if c.GetCustomDataByKey(usd.IMPORT_GROUP)} if shot else {}


def scenes(inputs: list[Packet]) -> list[tuple[str, Packet]]:
    """The single place a 「场景」 input becomes scenes to work with, whether its wires bring one scene
    each or a list of them: (the name to put it under, its packet), with "" for a whole scene, which keeps its names;
    one entry per item of a list, under the item's own name. A list packet holds no scene of its own (no frames, no
    frame rate: data/packet.py), so nothing that reads a scene ever reads one; `pack` and every output-settings node
    (nodes/output.py) go through here."""
    from .packet import Packet as P
    from .packet import items_of, packet_dir
    from .types import is_list

    out_parts: list[tuple[str, Packet]] = []
    for p in inputs:
        if not is_list(p.type):
            out_parts.append(("", p))
            continue
        for name, fp in items_of(p):
            if not P.exists(packet_dir(fp)):
                raise Invalid(Msg("E-CONTRACT-NOITEM", name=name))
            out_parts.append((name, P.load(packet_dir(fp))))
    return out_parts


def as_group(src: Packet, name: str, out: Path) -> Packet:
    """The scene with everything it holds under one group /shot/<name> (「命名」): one thing already there is renamed,
    several are gathered under a group of that name. The one place a scene is named, and only where the artist said so."""
    if usd.valid_name(name) != name:  # a hierarchy name goes into the 3D file as it is: never changed here
        raise Invalid(Msg("E-NAME-HIERARCHY", name=name))
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    shot = stage.GetPrimAtPath(ROOT_PATH)
    children = [c for c in shot.GetChildren()] if shot else []
    editor = Usd.NamespaceEditor(stage)
    if len(children) == 1:
        editor.RenamePrim(children[0], name)
        editor.ApplyEdits()
    elif children:
        stage.DefinePrim(f"{ROOT_PATH}/{name}", "Xform")
        for child in children:
            editor.MovePrimAtPath(str(child.GetPath()), f"{ROOT_PATH}/{name}/{child.GetName()}")
            editor.ApplyEdits()
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height", "scale")}
    return scene_packet(out, src.meta["frames"], src.type, **meta)


def _shot_of(p: Packet) -> tuple[list[str], bool]:
    """What a scene packet has under its /shot (the prims' names), and whether /shot itself carries a transform."""
    stage = Usd.Stage.Open(str(p.path(SCENE_FILE)))
    shot = stage.GetPrimAtPath(ROOT_PATH)
    if not shot:
        return [], False
    return [c.GetName() for c in shot.GetChildren()], bool(UsdGeom.Xformable(shot).GetOrderedXformOps())


def _prepared(p: Packet, names: dict[str, str], path: Path, group: str = "") -> tuple[Path, float]:
    """A copy of a scene packet's file ready to be packed with others: prims under /shot renamed (old -> new name;
    relationships, a skinned mesh's skeleton or a material binding, follow), and the transform of /shot moved onto
    each prim under it (first in its transform order, so it moves the prim as /shot did).

    `group`: when the item holds several things, they are gathered into one group `/shot/<group>` (in
    「手部动作 · HaMeR · 多人」 one person yields two hands, which must become `/shot/person_01/left_hand` and
    `/shot/person_01/right_hand`: one group per person, neither refused nor flattened into two names). The /shot
    transform is moved after the group is built, so it lands on the group and the hands keep their relative placement."""
    stage = Usd.Stage.Open(Usd.Stage.Open(str(p.path(SCENE_FILE))).Flatten())
    editor = Usd.NamespaceEditor(stage)
    for old, new in names.items():
        editor.RenamePrim(stage.GetPrimAtPath(f"{ROOT_PATH}/{old}"), new)
        editor.ApplyEdits()
    if group:
        # the group is created under a temporary name and renamed after the children are moved in. Creating it
        # directly as `group` fails when a child has the same name (the item is `person_02` and the solved person is
        # also `person_02`): `Xform.Define` then returns that child itself, turns it from SkelRoot into Xform and moves
        # the other child inside it: `/shot/person_02`(Xform) ⊃ {skeleton, anim, body, person_01(SkelRoot)}, and the
        # nested one receives the outer transform twice, ending up scaled and offset.
        # Slashes and spaces are invalid in names, so a prefix that cannot collide is used.
        temp = f"{ROOT_PATH}/_pack_group"
        holder = UsdGeom.Xform.Define(stage, temp).GetPrim()
        for child in list(stage.GetPrimAtPath(ROOT_PATH).GetChildren()):
            if child != holder:
                editor.ReparentPrim(child, holder)
                editor.ApplyEdits()
        editor.RenamePrim(holder, group)
        editor.ApplyEdits()
    shot = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT_PATH))
    ops = shot.GetOrderedXformOps()
    times = sorted({t for op in ops for t in op.GetTimeSamples()}) or [Usd.TimeCode.Default()]
    for child in shot.GetPrim().GetChildren() if ops else []:
        if not child.IsA(UsdGeom.Xformable):
            child.SetTypeName("Xform")  # a scope holding things: now it can carry the transform
        xf = UsdGeom.Xformable(child)
        if xf.GetResetXformStack():  # it ignores its parents' transforms already
            continue
        own = xf.GetOrderedXformOps()
        parent = usd.free_transform_op(xf, "shot")   # :shot1 when :shot already exists (the same data processed a second time)
        for t in times:
            parent.Set(shot.GetLocalTransformation(t), t)
        xf.SetXformOpOrder([parent, *own])
    shot.ClearXformOpOrder()
    stage.GetRootLayer().Export(str(path))
    return path


def euler_xyz_matrix(rotate_deg) -> np.ndarray:
    """Houdini / Maya "XYZ" order: rotate about X first, then Y, then Z (column vectors)."""
    rx, ry, rz = np.radians(rotate_deg)
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def xyz_euler_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotations [F,3,3] -> "XYZ" angles in degrees [F,3], the inverse of euler_xyz_matrix (at a gimbal lock, Z is 0),
    unwrapped over the frames so a turn past 180° goes on instead of jumping."""
    r = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    ry = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
    locked = np.abs(np.cos(ry)) < 1e-8
    rx = np.where(locked, np.arctan2(-r[:, 1, 2], r[:, 1, 1]), np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    rz = np.where(locked, 0.0, np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    return np.degrees(np.unwrap(np.stack([rx, ry, rz], -1), axis=0))


def trs_matrix(translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale: float = 1.0) -> np.ndarray:
    """4x4 (column vectors): scale, then rotate (XYZ order, degrees), then translate (cm)."""
    m = np.eye(4)
    m[:3, :3] = euler_xyz_matrix(rotate) * scale
    m[:3, 3] = translate
    return m


def transform(src: Packet, out: Path, m: np.ndarray) -> Packet:
    """The scene moved as a whole by `m` (4x4, column vectors) about the world origin, on top of any transform its
    /shot has. It stays the kind of scene it was: a camera moved is still a camera."""
    stage = _new_layer_stage(out / SCENE_FILE, [_file(src)], src.meta["frames"])
    root = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT_PATH))
    existing = np.array(root.GetLocalTransformation()).T  # row-vector -> column-vector
    combined = np.asarray(m, np.float64) @ existing
    root.ClearXformOpOrder()
    root.MakeMatrixXform().Set(Gf.Matrix4d(combined.T.tolist()))
    stage.GetRootLayer().Save()
    return scene_packet(out, src.meta["frames"], src.type,
                        **{k: v for k, v in src.meta.items() if k != "frames"})


def place_through(src: Packet, out: Path, mats: np.ndarray, frames: list[int]) -> Packet:
    """A camera-space scene -> world: each frame is moved by `mats[i]` (that frame's cam_to_world, 4x4, column vectors).

    The same operation as `transform`, with the matrix given per frame (time samples), so the result follows the
    camera's motion. Bodies, hands and faces from solvers are in camera space (as upstream outputs them); placing them
    in the world requires a camera, which is not a solver input (upstream does not take it), so this step is an
    explicit, separate node (a solver's inputs equal the upstream inputs)."""
    stage = _new_layer_stage(out / SCENE_FILE, [_file(src)], src.meta["frames"])
    root = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT_PATH))
    existing = np.array(root.GetLocalTransformation()).T  # row-vector -> column-vector
    root.ClearXformOpOrder()
    op = root.MakeMatrixXform()
    for f, m in zip(frames, np.asarray(mats, np.float64)):
        op.Set(Gf.Matrix4d((m @ existing).T.tolist()), Usd.TimeCode(int(f)))
    stage.GetRootLayer().Save()
    return scene_packet(out, src.meta["frames"], src.type,
                        **{k: v for k, v in src.meta.items() if k != "frames"})


def export(src: Packet, path: Path, unit: str, fps: float, provenance: dict | None = None) -> Path:
    """Flatten to one standalone file, as it is (what kinds it holds is the graph's: 「烘焙成模型」 makes point caches);
    cm -> m. `provenance` goes into the layer's metadata.

    `fps`: the one place a frame rate is stated, the output-settings node's 「帧率」 parameter. Everything inside
    Lab2Shot runs on frame numbers (io/usd.py STAGE_FPS), so this only writes the time base the DCC reads: frame 1001
    stays time code 1001, and the file says how many of those go in a second."""
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])  # flattening an anonymous root drops them
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    stage.GetRootLayer().documentation = ""
    if unit == "m":
        rescale_stage(stage, CM_TO_M)
    data = dict(stage.GetRootLayer().customLayerData or {})
    info = dict(data.get("lab2shot", {}))
    info.update(unit=unit)
    if provenance is not None:
        info["commercial"] = provenance["commercial"]
        info["sources"] = [f"{s['project']} ({s['node']}){'' if s['commercial'] else ' 非商用'}" for s in provenance["sources"]]
    data["lab2shot"] = usd.layer_data(info)
    stage.GetRootLayer().customLayerData = data
    path.parent.mkdir(parents=True, exist_ok=True)
    _localize_textures(stage, path)
    if not stage.GetRootLayer().Export(str(path)):
        raise FileProblem(Msg("E-USD-WRITE", path=str(path)))
    return path


def _localize_textures(stage: Usd.Stage, path: Path) -> None:
    """Textures referenced from the cache (e.g. a dome light's HDRI) are copied next to the file, linked relatively."""
    import shutil

    tex_dir = path.parent / f"{path.stem}_textures"
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attr.Get()
            src = Path(value.path) if value and value.path else None
            if src is None or not src.is_absolute() or not src.is_file():
                continue
            tex_dir.mkdir(exist_ok=True)
            target = tex_dir / f"{prim.GetName()}_{src.name}"
            shutil.copyfile(src, target)
            attr.Set(Sdf.AssetPath(f"./{tex_dir.name}/{target.name}"))


def _bake_point_cache(stage: Usd.Stage, frames: list[int]) -> None:
    """Replace skinning with per-frame points (exact model result incl. correctives if it had them)."""
    from .evaluate import skin_bindings

    for binding, query in skin_bindings(stage):
        for target in binding.GetSkinningTargets():
            mesh = UsdGeom.Mesh(target.GetPrim())
            baked = {}
            for f in frames:
                pts = usd.deformed_points(target, query, Usd.TimeCode(f))
                if pts is not None:
                    baked[f] = pts
            api = UsdSkel.BindingAPI(mesh.GetPrim())
            for name in ("primvars:skel:jointIndices", "primvars:skel:jointWeights", "primvars:skel:geomBindTransform",
                         "skel:blendShapes"):
                mesh.GetPrim().RemoveProperty(name)
            api.GetSkeletonRel().ClearTargets(True)
            api.GetBlendShapeTargetsRel().ClearTargets(True)
            mesh.GetPrim().RemoveAPI(UsdSkel.BindingAPI)
            points = mesh.GetPointsAttr()
            points.Clear()
            for f, pts in baked.items():
                points.Set(pts, Usd.TimeCode(f))


def bake(src: Packet, out: Path) -> Packet:
    """「烘焙成模型」: the meshes of a 蒙皮角色 as deforming models, a point cache per frame (blend shapes and skinning
    evaluated as a DCC does), where they were; the skeletons, their animation and the blend shapes are gone. A 模型
    packet: what a format without skeletons holds."""
    frames = [int(f) for f in src.meta["frames"]]
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, frames)  # flattening an anonymous root drops them
    _bake_point_cache(stage, frames)
    for prim in list(stage.Traverse()):
        if not prim.IsValid():
            continue
        if prim.IsA(UsdSkel.Root):
            prim.SetTypeName("Xform")
        elif prim.GetPath() != Sdf.Path(ROOT_PATH) and not any(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(prim)):
            stage.RemovePrim(prim.GetPath())  # the skeleton, its animation, the blend shapes, a camera: not the models
    if not any(p.IsA(UsdGeom.Mesh) for p in stage.Traverse()):
        raise Invalid(Msg("E-BAKE-NOSKIN"))
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    return scene_packet(out, frames, "scene.model",
                        **{k: v for k, v in src.meta.items() if k in ("width", "height")})


def skeleton_only(src: Packet, out: Path) -> Packet:
    """「提取骨架」: a 蒙皮角色's skeletons and their animation, without the meshes (their skin and blend shapes go with
    them): a 骨架动画, keyed where the character was."""
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    skeletons = [p for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    if not skeletons:
        raise Invalid(Msg("E-SKELETON-NONE"))
    keep_only(stage, [s.GetPath() for s in skeletons]
              + [t for s in skeletons for t in UsdSkel.BindingAPI(s).GetAnimationSourceRel().GetTargets()])
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    return scene_packet(out, src.meta["frames"], "scene.skeleton",
                        **{k: v for k, v in src.meta.items() if k in ("width", "height")})


@dataclass(frozen=True)
class Meshes:
    """The meshes of a 模型 packet combined into the single triangle mesh used by auto-rigging.

    `points` [V,3] and `faces` [T,3] are the combined mesh in world coordinates (cm, Y up); `owner` [V] gives the source
    mesh of each vertex, `sizes` the vertex count of each mesh, `paths` their USD paths. After weights are computed they
    are split back by these (`split`) and written to the original meshes by path, not by order."""

    points: np.ndarray
    faces: np.ndarray
    owner: np.ndarray
    sizes: tuple[int, ...]
    paths: tuple[str, ...]

    def split(self, per_vertex: np.ndarray) -> dict[str, np.ndarray]:
        """Split per-vertex data (weights [V,J], influence indices [V,K]) back per mesh: USD path -> that mesh's slice."""
        return dict(zip(self.paths, np.split(np.asarray(per_vertex), np.cumsum(self.sizes)[:-1])))


def mesh_arrays(src: Packet, frame: int | None = None) -> Meshes:
    """All meshes of a 模型 packet combined into one triangle mesh (world coordinates, cm, Y up) for the auto-rigging
    family (nodes/families/rigging.py one_mesh).

    Returns Meshes: combined points [V,3], triangles [T,3], the source mesh of each point (`owner`), the point count of
    each mesh (`sizes`) and each mesh's USD path (`paths`, used to write weights back). Quads and polygons are
    triangulated with the same fan split as the io code (data/evaluate.py triangulate); point indices are unchanged,
    so per-vertex weights map back directly."""
    from .evaluate import triangulate

    time = Usd.TimeCode(float(frame if frame is not None else src.meta["frames"][0]))
    stage = open_scene([src])
    cache = UsdGeom.XformCache(time)
    points, faces, sizes, paths = [], [], [], []
    base = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        raw = mesh.GetPointsAttr().Get(time)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], np.int64)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], np.int64)
        if raw is None or not len(raw) or not len(counts):
            continue
        world = np.asarray(np.array(cache.GetLocalToWorldTransform(prim)), np.float64)
        pts = np.asarray(raw, np.float64)
        points.append(np.concatenate([pts, np.ones((len(pts), 1))], 1) @ world[:, :3])
        faces.append(triangulate(counts, idx) + base)
        sizes.append(len(pts))
        paths.append(str(prim.GetPath()))
        base += len(pts)
    if not points:
        raise Invalid(Msg("E-RIG-NOMESH"))
    owner = np.repeat(np.arange(len(sizes)), sizes)
    return Meshes(np.concatenate(points), np.concatenate(faces), owner, tuple(sizes), tuple(paths))


def bind_skin(src: Packet, out: Path, *, joint_names: list[str], parents, bind_world: np.ndarray,
              joint_indices: dict[str, np.ndarray], joint_weights: dict[str, np.ndarray], name: str = "",
              info: dict | None = None) -> Packet:
    """Auto-rigging delivery (nodes/families/rigging.py AutoRig): bind a skeleton to the meshes of a 模型 -> 蒙皮角色.

    The original USD is modified rather than rebuilt: UVs, normals, per-point attributes, subsets (GeomSubset) and
    material bindings are kept; only a SkelRoot, its Skeleton, a static SkelAnimation and skin weights on each mesh are
    added. This is the inverse of 「烘焙成模型」 and lives next to it (scene data edits in one place). The skeleton itself
    is written by io/usd.py write_rig, the same writer as for solver characters.

    Each mesh's world transform is baked into its points and its own and its ancestors' transforms are reset to
    identity: skinning results in USD are in skeleton space and the mesh's own transform no longer applies (UsdSkel
    convention), so after baking the shape is identical to before binding. `joint_indices` / `joint_weights` are keyed
    by the meshes' USD paths (the `paths` from mesh_arrays), each [V,K]."""
    frames = [int(f) for f in src.meta["frames"]]
    rest = frames[:1]
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, rest)
    top = [p for p in stage.GetPrimAtPath(ROOT_PATH).GetChildren()] if stage.GetPrimAtPath(ROOT_PATH) else []
    bind_world = np.asarray(bind_world, np.float64)
    # when the model is already under a single group and no name was given, that group becomes the SkelRoot, keeping
    # its name and leaving every mesh in place. Otherwise (several groups, or a user-given name) a new SkelRoot is created
    # and they are moved into it: skinnable prims in USD must be inside a SkelRoot
    in_place = len(top) == 1 and not name
    label = usd.valid_name(name or (top[0].GetName() if in_place else "rig"))
    if not in_place:
        while stage.GetPrimAtPath(f"{ROOT_PATH}/{label}"):
            label += "_"
    root_path = f"{ROOT_PATH}/{label}"
    skel_root = usd.write_rig(stage, label, list(joint_names), np.asarray(parents, np.int64), bind_world,
                              bind_world[None], rest, path=root_path, custom_data=usd.layer_data(info or {}))
    if not in_place:
        edit = Sdf.BatchNamespaceEdit()
        for prim in top:
            if prim.GetPath() != skel_root.GetPrim().GetPath():
                edit.Add(str(prim.GetPath()), f"{root_path}/{prim.GetName()}")
        if not stage.GetRootLayer().Apply(edit):
            raise Invalid(Msg("E-RIG-NOMESH"))
    cache = UsdGeom.XformCache(Usd.TimeCode(rest[0]))
    for path, idx in joint_indices.items():
        prim = stage.GetPrimAtPath(path if in_place else f"{root_path}/{path[len(ROOT_PATH) + 1:]}")
        mesh = UsdGeom.Mesh(prim)
        world = np.asarray(np.array(cache.GetLocalToWorldTransform(prim)), np.float64)
        pts = np.asarray(mesh.GetPointsAttr().Get(Usd.TimeCode(rest[0])), np.float64)
        mesh.GetPointsAttr().Clear()
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(
            (np.concatenate([pts, np.ones((len(pts), 1))], 1) @ world[:, :3]).astype(np.float32)))
        for above in [prim, *_ancestors(prim)]:  # ancestor transforms are already baked into the points and must not apply again
            if above.GetPath().HasPrefix(root_path) and above.IsA(UsdGeom.Xformable):
                UsdGeom.Xformable(above).ClearXformOpOrder()
        idx, w = np.asarray(idx), np.asarray(joint_weights[path])
        binding = UsdSkel.BindingAPI.Apply(prim)
        binding.CreateSkeletonRel().SetTargets([f"{root_path}/skeleton"])
        binding.CreateJointIndicesPrimvar(False, idx.shape[1]).Set(Vt.IntArray.FromNumpy(idx.astype(np.int32).reshape(-1)))
        binding.CreateJointWeightsPrimvar(False, w.shape[1]).Set(Vt.FloatArray.FromNumpy(w.astype(np.float32).reshape(-1)))
        binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    return scene_packet(out, rest, "scene.character",
                        **{k: v for k, v in src.meta.items() if k in ("width", "height")})


def _ancestors(prim):
    p = prim.GetParent()
    while p and not p.IsPseudoRoot():
        yield p
        p = p.GetParent()


def keep_only(stage: Usd.Stage, paths) -> None:
    """Keep these prims, everything below them and their ancestors (ancestors carry transforms; removing them would
    move things), and remove everything else. 「提取骨架」, 「按种类取出」 and 「按分区取出」 are the same operation and
    share this implementation. Materials bound to kept meshes are kept too; without them, delivered meshes would lose
    their shading groups."""
    from pxr import UsdShade

    keep = [Sdf.Path(str(p)) for p in paths]
    keep += [m.GetPrim().GetPath() for k in list(keep) if (root := stage.GetPrimAtPath(k))
             for p in Usd.PrimRange(root) if (m := UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0])]
    above = {a.GetPath() for k in keep if (p := stage.GetPrimAtPath(k)) for a in _ancestors(p)}
    for prim in list(stage.Traverse()):
        path = prim.GetPath()
        if prim.IsValid() and path not in above and not any(path.HasPrefix(k) for k in keep):
            stage.RemovePrim(path)


# ------------------------------------------------------------------ what a 3D output-settings node gets


def prims_by_kind(stage: Usd.Stage) -> dict[str, list[Usd.Prim]]:
    """Every prim that is one thing of a kind of 3D data (types.SCENE_KINDS), by kind: a camera; a skeleton, a 蒙皮角色
    when a mesh is skinned to it, else a 骨架动画; a mesh not skinned, a 模型; a 点云; a 三维曲线 (BasisCurves); a light.
    The one classification of a scene's prims (kinds_held, only_kind)."""
    from pxr import UsdLux

    from ..data.types import SCENE_KINDS
    from .evaluate import skin_bindings

    bound = {b.GetSkeleton().GetPrim().GetPath(): b.GetSkinningTargets() for b, _ in skin_bindings(stage)}
    skinned = {t.GetPrim().GetPath() for targets in bound.values() for t in targets}
    found: dict[str, list[Usd.Prim]] = {k: [] for k in SCENE_KINDS}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            found["camera"].append(prim)
        elif prim.IsA(UsdSkel.Skeleton):
            found["character" if bound.get(prim.GetPath()) else "skeleton"].append(prim)
        elif prim.IsA(UsdGeom.Mesh) and prim.GetPath() not in skinned:
            found["model"].append(prim)
        elif prim.IsA(UsdGeom.Points):
            found["points"].append(prim)
        elif prim.IsA(UsdGeom.BasisCurves):
            found["curves"].append(prim)
        elif prim.HasAPI(UsdLux.LightAPI):
            found["light"].append(prim)
    return found


def kinds_held(src: Packet) -> tuple[str, ...]:
    """The kinds of 3D data a cooked scene holds (types.SCENE_KINDS, and types.DEFORMING for a 模型 whose points
    change), in the table's order: what an output-settings node checks before it writes (the graph derived the same
    before cooking: Graph.scene_kinds)."""
    from ..data.types import DEFORMING, KIND_ORDER

    stage = open_scene([src])  # kept while its prims are read
    found = prims_by_kind(stage)
    held = {k for k, prims in found.items() if prims}
    if any(UsdGeom.Mesh(p).GetPointsAttr().ValueMightBeTimeVarying() for p in found["model"]):
        held.add(DEFORMING)
    return tuple(k for k in KIND_ORDER if k in held)


def only_kind(src: Packet, kind: str, out: Path) -> Packet | None:
    """「按种类取出」: one kind of 3D data of a scene where it is: its parents with their transforms, a 骨架动画's or 蒙皮角色's
    whole skeleton root (animation, skinned meshes, blend shapes), the materials its meshes are bound to (keep_only);
    nothing else. None when the scene holds none of that kind. A 点云 keeps the scene's scale claim, or claims no more than
    "relative" when the scene records none."""
    from ..data.types import SCENE_KINDS

    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    chosen = prims_by_kind(stage)[kind]
    if not chosen:
        return None
    roots = [next((a for a in _ancestors(p) if a.IsA(UsdSkel.Root)), p) if kind in ("skeleton", "character") else p for p in chosen]
    keep = [r.GetPath() for r in roots]
    keep += [t for p in chosen if p.IsA(UsdSkel.Skeleton) for t in UsdSkel.BindingAPI(p).GetAnimationSourceRel().GetTargets()]
    keep_only(stage, keep)  # bound materials and ancestor transforms are kept as well (keep_only)
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height")}
    if kind == "points":
        meta["scale"] = src.meta.get("scale", "relative")
    return scene_packet(out, src.meta["frames"], SCENE_KINDS[kind].type, **meta)


def only_group(src: Packet, name: str, out: Path) -> Packet | None:
    """One group under /shot of a scene, with everything under it (data/items.py: a scene's items are its groups).
    None when the scene has no such group."""
    stage = Usd.Stage.Open(open_scene([src]).Flatten())
    usd.apply_conventions(stage, src.meta["frames"])
    shot = stage.GetPrimAtPath(usd.ROOT_PATH)
    if not shot or not any(c.GetName() == name for c in shot.GetChildren()):
        return None
    for child in list(shot.GetChildren()):
        if child.GetName() != name:
            stage.RemovePrim(child.GetPath())
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    meta = {k: v for k, v in src.meta.items() if k in ("width", "height", "scale")}
    return scene_packet(out, src.meta["frames"], src.type, **meta)


def join_named(parts: list[tuple[str, Packet]], out: Path, type_: str = "scene") -> Packet:
    """Named scenes into one, each under /shot/<its name> (data/items.py merge): every part holds one thing, and its
    name is the artist's; nothing is renamed here (two of one name are refused before this is called). `type_`: what
    the result carries (data/items.py _scene_merge works it out: three cameras merged are still 相机)."""
    files = []
    for i, (name, packet) in enumerate(parts):
        holds = _shot_of(packet)[0]
        # an item holding several things becomes one group /shot/<name> (the same rule as pack)
        group = name if len(holds) != 1 else ""
        renamed = {} if group else ({holds[0]: name} if holds[0] != name else {})
        files.append(_prepared(packet, renamed, out / f"input_{i}.usda", group))
    frames = union([p for _, p in parts])
    stage = _new_layer_stage(out / SCENE_FILE, files, frames)
    stage.GetRootLayer().Save()
    meta = {k: v for _, p in parts for k, v in p.meta.items() if k in ("width", "height")}
    return scene_packet(out, frames, type_, **meta)


# ------------------------------------------------------------------ cameras


def cameras_in(stage: Usd.Stage) -> list[Usd.Prim]:
    return [p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)]


def the_camera(stage: Usd.Stage, where: str) -> Usd.Prim:
    """The one camera of a stage; none or several is the user's to resolve, never a silent pick. `where` names what
    the stage is in the message (相机输入, 场景)."""
    found = cameras_in(stage)
    if len(found) == 1:
        return found[0]
    if not found:
        raise Invalid(Msg("E-CAMERA-NONE", where=where))
    paths = "、".join(str(p.GetPath()) for p in found[:4]) + ("……" if len(found) > 4 else "")
    raise Invalid(Msg("E-CAMERA-SEVERAL", where=where, count=len(found), paths=paths))


# ------------------------------------------------------------------ seen through the camera


def camera_of(scenes: list[Packet], camera: Packet | None = None) -> tuple[Usd.Stage, Usd.Prim]:
    """The camera to look through: the connected camera, else the scenes' one camera (several: the user wires one)."""
    stage = open_scene([camera] if camera is not None else scenes)  # keep the stage alive while its prims are used
    return stage, the_camera(stage, "相机输入" if camera is not None else "场景")
