"""Reading a DCC's USD file: its entries per kind, and the selected ones as packets.

USD is also what packets are, so an entry is copied as it is (its whole subtree: a skeleton with its animation,
blend shapes and skinned meshes; a mesh with its UVs, normals and primvars), placed where it was in the file: its
parents' transforms composed into one op before its own, the file's unit (metersPerUnit) and up axis (upAxis) turned
into Lab2Shot's (cm, Y up). A camera is read as a camera track (rigid: no parent's scale reaches it).

Only a flattened file is taken: one that needs no other file to compose (open_stage). A production file that
references, pays in, sublayers or clips other files is refused before any of them is opened, saying how to flatten
it: the server never follows a path out of the file it was given (a wrong read there could reach a studio's whole
asset tree). A .usdz is one file: the layers inside it are its own. Asset-valued attributes (textures) are not
composition: they stay as they are.
"""

from __future__ import annotations

import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdSkel

from ...errors import Invalid
from ...io import usd
from ...messages import Msg
from ...nodes.formats import Entry, Listing, count

if TYPE_CHECKING:
    from ...data.camera import CameraSamples
    from ...data.packet import Packet

SHOWN = 5  # external files named in the message, then how many more


def open_stage(path: Path) -> Usd.Stage:
    """The file as a stage, when it is flattened (external_files: nothing outside it); its root layer alone is read
    first, so a file it refers to is never opened."""
    try:
        layer = Sdf.Layer.FindOrOpen(str(path))
    except Tf.ErrorException as exc:  # a broken file, or not USD: said plainly (USD's message names server paths)
        raise Invalid(Msg("E-USD-UNREADABLE", file=path.name)) from exc
    if layer is None:
        raise Invalid(Msg("E-USD-UNREADABLE", file=path.name))
    if outside := external_files(layer, str(path) if path.suffix.lower() == ".usdz" else ""):
        if len(outside) > SHOWN:
            raise Invalid(Msg("E-USD-MANYREFERENCES", file=path.name, files=outside[:SHOWN], count=len(outside)))
        raise Invalid(Msg("E-USD-REFERENCES", file=path.name, files=outside))
    try:
        return Usd.Stage.Open(layer)
    except Tf.ErrorException as exc:
        raise Invalid(Msg("E-USD-UNREADABLE", file=path.name)) from exc


def external_files(layer: Sdf.Layer, package: str = "", seen: set[str] | None = None) -> list[str]:
    """The files a layer needs from outside itself to compose, as the layer writes them: its sublayers, references,
    payloads and value clips. Inside a .usdz (`package`: its path) a layer of the package is followed the same way (it
    is part of the file); nothing outside is opened. Asset paths on attributes (textures) are not composition."""
    seen = set() if seen is None else seen
    out: list[str] = []
    for asset in [*layer.GetCompositionAssetDependencies(), *_clip_assets(layer)]:
        if not asset:  # an internal reference (an instance's prototype in the same file): not another file
            continue
        where = layer.ComputeAbsolutePath(asset)
        if package and where.startswith(package + "["):
            if where not in seen and (inner := Sdf.Layer.FindOrOpen(where)) is not None:
                seen.add(where)
                out += external_files(inner, package, seen)
        elif asset not in out:
            out.append(asset)
    return out


def _clip_assets(layer: Sdf.Layer) -> list[str]:
    """The files a layer's value clips read their samples from (every clip set's assetPaths, manifest, template)."""
    found: list[str] = []

    def visit(p: Sdf.Path) -> None:
        if not p.IsPrimPath():
            return
        spec = layer.GetPrimAtPath(p)
        if spec is None or not spec.HasInfo("clips"):
            return
        for clip in dict(spec.GetInfo("clips")).values():
            for key in ("assetPaths", "manifestAssetPath", "templateAssetPath"):
                value = clip.get(key)
                for a in value if isinstance(value, (list, tuple)) or type(value).__name__.endswith("Array") else [value]:
                    text = getattr(a, "authoredPath", a) if a is not None else ""
                    if text:
                        found.append(str(text))

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return found


def axes_of(stage: Usd.Stage):
    from ...data.scene_arrays import Axes

    return Axes.of(UsdGeom.GetStageMetersPerUnit(stage) / usd.METERS_PER_UNIT, "z" if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z else "y")


# ------------------------------------------------------------------ what the file holds


def _xform_times(prim) -> set[float]:
    """The time samples of a prim's own transform and every parent's."""
    times: set[float] = set()
    p = prim
    while p and p.IsValid() and not p.IsPseudoRoot():
        if p.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(p).GetOrderedXformOps():
                times.update(op.GetTimeSamples())
        p = p.GetParent()
    return times


def _frames(stage: Usd.Stage, times: set[float]) -> tuple[int, ...]:
    """Every frame from the first sample to the last (USD interpolates between samples: keys every few frames still
    move it every frame); nothing animated: one, the stage's first."""
    if not times:
        return (int(round(stage.GetStartTimeCode())),)
    first, last = math.ceil(min(times) - 1e-6), math.floor(max(times) + 1e-6)
    return tuple(range(first, last + 1)) if last >= first else (int(round(min(times))),)


def _attr_times(*attrs) -> set[float]:
    return {t for a in attrs if a for t in a.GetTimeSamples()}


def _skel_parts(root: Usd.Prim) -> tuple[list, list]:
    """A skeleton root's skeletons (a rig without a skinned mesh too) and the meshes skinned in it."""
    cache = UsdSkel.Cache()
    skel_root = UsdSkel.Root(root)
    cache.Populate(skel_root, Usd.PrimDefaultPredicate)
    bindings = cache.ComputeSkelBindings(skel_root, Usd.PrimDefaultPredicate)
    return [UsdSkel.Skeleton(p) for p in Usd.PrimRange(root) if p.IsA(UsdSkel.Skeleton)], [t for b in bindings for t in b.GetSkinningTargets()]


def _character_times(root: Usd.Prim) -> set[float]:
    """The samples of a character's animation and blend-shape weights, and of every transform above it."""
    times = _xform_times(root)
    for skel in _skel_parts(root)[0]:
        source = UsdSkel.BindingAPI(skel.GetPrim()).GetAnimationSourceRel().GetTargets()
        anim = UsdSkel.Animation(root.GetStage().GetPrimAtPath(source[0])) if source else None
        if anim:
            times |= _attr_times(anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr(), anim.GetBlendShapeWeightsAttr())
    return times


def _resolution(stage: Usd.Stage, prim) -> tuple[int, int] | None:
    found = usd.camera_resolution(prim)
    if found:
        return found
    for p in stage.Traverse():
        attr = p.GetAttribute("resolution")
        if p.GetTypeName() == "RenderSettings" and attr and attr.Get() is not None:
            r = attr.Get()
            return int(r[0]), int(r[1])
    return None


@lru_cache(maxsize=32)
def listing(path: str) -> Listing:
    """Every camera, model, point cloud, set of 3D curves, skeleton and character of a USD file (uploads never
    change: each is listed once): cameras, point clouds, curves and meshes by their prim paths; a skeleton root by its path, a 蒙皮角色 when meshes are
    skinned in it (they and any other mesh under it come with it, not as models), else a 骨架动画 (bones alone)."""
    stage = open_stage(Path(path))
    entries = []
    # instances too (instance proxies): an instanced set lists every mesh under every instance, where that instance is
    it = iter(stage.Traverse(Usd.TraverseInstanceProxies()))
    for prim in it:
        where = str(prim.GetPath())
        if prim.IsA(UsdSkel.Root):
            skels, meshes = _skel_parts(prim)
            joints = sum(len(s.GetJointsAttr().Get() or []) for s in skels)
            shapes = sum(len(UsdSkel.BindingAPI(t.GetPrim()).GetBlendShapesAttr().Get() or []) for t in meshes)
            detail = " · ".join([count(joints, "关节")] + ([count(len(meshes), "网格")] if meshes else []) + ([count(shapes, "形变")] if shapes else []))
            entries.append(Entry("character" if meshes else "skeleton", where, _frames(stage, _character_times(prim)), detail))
            it.PruneChildren()
        elif prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            times = _xform_times(prim) | _attr_times(cam.GetFocalLengthAttr(), cam.GetHorizontalApertureAttr(), cam.GetVerticalApertureAttr(),
                                                     cam.GetHorizontalApertureOffsetAttr(), cam.GetVerticalApertureOffsetAttr())
            frames = _frames(stage, times)
            t = Usd.TimeCode(frames[0])
            vertical = cam.GetVerticalApertureAttr().HasAuthoredValue()
            aspect = cam.GetVerticalApertureAttr().Get(t) / cam.GetHorizontalApertureAttr().Get(t) if vertical else 9 / 16
            entries.append(Entry("camera", where, frames, f"{cam.GetFocalLengthAttr().Get(t):.4g} mm", size=_resolution(stage, prim),
                                 aspect=aspect))
        elif prim.IsA(UsdGeom.PointInstancer):  # one 模型: expanded into real meshes when read (_expand)
            inst = UsdGeom.PointInstancer(prim)
            times = _xform_times(prim) | _attr_times(inst.GetPositionsAttr(), inst.GetOrientationsAttr(), inst.GetScalesAttr(),
                                                     inst.GetProtoIndicesAttr())
            frames = _frames(stage, times)
            shown = len(inst.GetProtoIndicesAttr().Get(Usd.TimeCode(frames[0])) or [])
            entries.append(Entry("model", where, frames, f"{count(len(inst.GetPrototypesRel().GetTargets()), '原型')} · {count(shown, '实例')}"))
            it.PruneChildren()  # its prototypes are what it places, not models of their own
        elif prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr()
            frames = _frames(stage, _xform_times(prim) | _attr_times(points))
            entries.append(Entry("model", where, frames, count(len(points.Get(Usd.TimeCode(frames[0])) or []), "顶点"),
                                 points.ValueMightBeTimeVarying()))
        elif prim.IsA(UsdGeom.Points):
            points = UsdGeom.Points(prim).GetPointsAttr()
            most = max([len(points.Get(t) or []) for t in points.GetTimeSamples()] or [len(points.Get() or [])])
            entries.append(Entry("points", where, _frames(stage, _xform_times(prim) | _attr_times(points)), count(most, "点")))
        elif prim.IsA(UsdGeom.BasisCurves):
            curves = UsdGeom.BasisCurves(prim)
            points, counts = curves.GetPointsAttr(), curves.GetCurveVertexCountsAttr()
            times = _attr_times(points, counts)
            most = max([len(points.Get(t) or []) for t in points.GetTimeSamples()] or [len(points.Get() or [])])
            strands = max([len(counts.Get(t) or []) for t in counts.GetTimeSamples()] or [len(counts.Get() or [])])
            entries.append(Entry("curves", where, _frames(stage, _xform_times(prim) | times),
                                 f"{count(strands, '条')} · {count(most, '点')}"))
    return Listing(tuple(entries))


# ------------------------------------------------------------------ reading the selected entries


def camera(path: Path, entry: Entry, size: tuple[int, int]) -> CameraSamples:
    """A camera as its samples: its world matrix at every frame (every parent, animated ones too), made rigid, in
    cm and Y up; its lens per frame; its picture `size` (ImportNode.picture_size)."""
    from ...data.camera import CameraSamples
    from ...data.scene_arrays import rigid

    stage = open_stage(path)
    prim = stage.GetPrimAtPath(entry.path)
    raw = CameraSamples.from_prim(prim, entry.frames, width=size[0], height=size[1],
                                  info={"imported_from": path.name, "object": entry.path})
    return replace(raw, cam_to_world=rigid(axes_of(stage).matrices(raw.cam_to_world)))


def copied(path: Path, entries: list[Entry], kind: str, out: Path, group: str, owner: str) -> Packet:
    """The entries' prims copied into a new scene where their file had them, under their import's folder
    (/shot/<group>/<path in the file>, io/usd.py import_path; the groups between are identity), each subtree as it
    is and placed where it was: one op before its own transform with its parents' world transform, the file's up axis
    turned into Lab2Shot's; then every length in cm. An entry inside another chosen one is copied with it."""
    from ...data.payloads import SCENE_FILE, scene_packet

    src = open_stage(path)
    axes = axes_of(src)
    flat = src.Flatten()
    frames = sorted({f for e in entries for f in e.frames})
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    UsdGeom.Xform.Define(stage, usd.ROOT_PATH)
    materials: dict[Sdf.Path, Sdf.Path] = {}
    turn = np.eye(4)
    turn[:3, :3] = axes.turn
    chosen = [Sdf.Path(e.path) for e in entries]
    for e in entries:
        own = Sdf.Path(e.path)
        if any(own != c and own.HasPrefix(c) for c in chosen):
            continue  # inside another chosen entry: copied with it, placed by it
        dst = Sdf.Path(usd.import_path(group, e.path))
        usd.place(stage, dst.pathString)  # its groups, so the copy has a parent
        original = src.GetPrimAtPath(e.path)
        outside = lambda t: _material(src, flat, layer, stage, t, materials, group)  # noqa: E731
        if original.IsA(UsdGeom.PointInstancer):
            _expand(original, flat, layer, stage, dst, e.frames, outside)
        else:
            spec = _spec_path(flat, own)  # an instance's mesh: its prototype's spec, placed where the instance is
            Sdf.CopySpec(flat, spec, layer, dst)
            _retarget(stage.GetPrimAtPath(dst), spec, dst, outside)
        _place(original, stage.GetPrimAtPath(dst), turn)
        stage.GetPrimAtPath(dst).SetCustomDataByKey("lab2shot:object", e.path)
        usd.place(stage, dst.pathString, e.path)  # renamed segments keep their names
    usd.mark_group(stage, group, owner)  # the folder of this very import cook (`owner`: its fingerprint)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(src))
    usd.rescale_stage(stage, axes.to_cm)
    usd.apply_conventions(stage, frames)
    stage.GetRootLayer().customLayerData = {"lab2shot": usd.layer_data({"imported_from": path.name})}
    out.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out / SCENE_FILE))
    return scene_packet(out, frames, f"scene.{kind}", **({"scale": "metric"} if kind == "points" else {}))


# the most faces a PointInstancer may expand into on import (what you see is the data: an instancer is never thinned,
# one past this is refused saying how many faces it would make, E-USD-INSTANCERSIZE)
EXPANDED_FACES_MAX = 20_000_000


def _expand(instancer: Usd.Prim, flat: Sdf.Layer, layer: Sdf.Layer, stage: Usd.Stage, dst: Sdf.Path, frames, outside) -> None:
    """A PointInstancer as real meshes: under `dst` an Xform per shown instance
    (instance_<i>) keyed at every frame with that instance's transform and the instancer's own (USD's own
    ComputeInstanceTransformsAtTime), its prototype's subtree copied under it. Hidden instances are left out, as USD
    shows none of them."""
    inst = UsdGeom.PointInstancer(instancer)
    protos = inst.GetPrototypesRel().GetTargets()
    first = Usd.TimeCode(frames[0])
    indices = [int(i) for i in (inst.GetProtoIndicesAttr().Get(first) or [])]
    mask = list(inst.ComputeMaskAtTime(first))
    shown = [i for i in range(len(indices)) if not mask or mask[i]]
    src = instancer.GetStage()
    faces = [sum(len(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get(first) or []) for p in Usd.PrimRange(src.GetPrimAtPath(q))
                 if p.IsA(UsdGeom.Mesh)) for q in protos]
    total = sum(faces[indices[i]] for i in shown)
    if total > EXPANDED_FACES_MAX:
        raise Invalid(Msg("E-USD-INSTANCERSIZE", path=str(instancer.GetPath()), faces=total, instances=len(shown), limit=EXPANDED_FACES_MAX))
    xf = UsdGeom.Xformable(instancer)
    moving = bool(_attr_times(inst.GetPositionsAttr(), inst.GetOrientationsAttr(), inst.GetScalesAttr())) or xf.TransformMightBeTimeVarying()
    times = list(frames) if moving else list(frames[:1])
    samples = [(Usd.TimeCode(f), inst.ComputeInstanceTransformsAtTime(Usd.TimeCode(f), Usd.TimeCode(f)), xf.GetLocalTransformation(Usd.TimeCode(f)))
               for f in times]
    UsdGeom.Xform.Define(stage, dst)
    for i in shown:
        holder = dst.AppendChild(f"instance_{i}")
        op = UsdGeom.Xform.Define(stage, holder).AddTransformOp()
        for t, mats, local in samples:
            op.Set(mats[i] * local, t if moving else Usd.TimeCode.Default())  # row vectors: the instance's, then the instancer's
        proto = protos[indices[i]]
        target = holder.AppendChild(proto.name)
        spec = _spec_path(flat, proto)
        Sdf.CopySpec(flat, spec, layer, target)
        _retarget(stage.GetPrimAtPath(target), spec, target, outside)


def _spec_path(flat: Sdf.Layer, path: Sdf.Path) -> Sdf.Path:
    """Where the flattened layer holds what a listed prim reads: its own path, or for a prim inside an instance (an
    instance proxy, which has no spec of its own) the path under the prototype its instance references in the same
    file, followed prefix by prefix so an instance inside a prototype is resolved too."""
    for p in path.GetPrefixes():
        spec = flat.GetPrimAtPath(p)
        if spec is None:
            break
        if p == path:
            return path
        arcs = [*spec.referenceList.prependedItems, *spec.referenceList.explicitItems, *spec.referenceList.addedItems]
        inner = next((r.primPath for r in arcs if not r.assetPath and not r.primPath.isEmpty), None)
        if inner is not None:
            return _spec_path(flat, inner.AppendPath(path.MakeRelativePath(p)))
    raise Invalid(Msg("E-USD-UNREADABLE", file=str(path)))


def _retarget(root: Usd.Prim, old: Sdf.Path, new: Sdf.Path, outside=lambda t: t) -> None:
    """Relationships and connections inside a copied subtree that pointed into it point into the copy (a skinned mesh
    to its skeleton, a skeleton to its animation, a mesh to its blend shapes); one to something outside goes where
    `outside` says (a bound material: copied along)."""
    def moved(t: Sdf.Path) -> Sdf.Path:
        return t.ReplacePrefix(old, new) if t.HasPrefix(old) else outside(t)

    for prim in Usd.PrimRange(root):
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if targets and [moved(t) for t in targets] != list(targets):
                rel.SetTargets([moved(t) for t in targets])
        for attr in prim.GetAttributes():
            sources = attr.GetConnections()
            if sources and [moved(t) for t in sources] != list(sources):
                attr.SetConnections([moved(t) for t in sources])


def _material(src: Usd.Stage, flat: Sdf.Layer, layer: Sdf.Layer, stage: Usd.Stage, target: Sdf.Path, done: dict,
              group: str) -> Sdf.Path:
    """A material an entry is bound to, outside it: copied once under its import's folder, /shot/<group>/Looks (its
    shaders with it); anything else outside stays as it is."""
    from pxr import UsdShade

    material = target.GetPrimPath()
    if material in done:
        return target.ReplacePrefix(material, done[material])
    prim = src.GetPrimAtPath(material)
    if not prim or not prim.IsA(UsdShade.Material):
        return target
    looks = f"{usd.ROOT_PATH}/{group}/Looks"
    name = usd.valid_name(material.name)
    while stage.GetPrimAtPath(f"{looks}/{name}"):
        name += "_"
    UsdGeom.Scope.Define(stage, looks)
    done[material] = Sdf.Path(f"{looks}/{name}")
    Sdf.CopySpec(flat, material, layer, done[material])
    _retarget(stage.GetPrimAtPath(done[material]), material, done[material])
    return target.ReplacePrefix(material, done[material])


def _place(original: Usd.Prim, prim: Usd.Prim, turn: np.ndarray) -> None:
    """The copy placed where the original was: its parents' world transform (a sample at each of theirs, animated
    parents too) as the first op, the file's up axis turned into Lab2Shot's on the left. A prim that resets the transform
    stack ignores its parents: only the turn."""
    xf = UsdGeom.Xformable(prim)
    own = xf.GetOrderedXformOps()
    reset = xf.GetResetXformStack()
    parent = original.GetParent()
    times = sorted({t for p in [parent] for t in _xform_times(p)}) if parent and not parent.IsPseudoRoot() and not reset else []
    op = usd.free_transform_op(xf, "placed")  # a file Lab2Shot wrote was placed so already: then it is :placed1

    def world(t):
        if reset or not parent or parent.IsPseudoRoot():
            return turn
        return turn @ np.array(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(t)).T

    if times:
        for t in times:
            op.Set(Gf.Matrix4d(world(Usd.TimeCode(t)).T.tolist()), Usd.TimeCode(t))
    else:
        op.Set(Gf.Matrix4d(world(Usd.TimeCode.Default()).T.tolist()))
    xf.SetXformOpOrder([op, *own], reset)
