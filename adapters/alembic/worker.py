"""Alembic worker: runs inside third_party/alembic/.venv (PyAlembic + PyImath, compiled by build.py); never imports
Lab2Shot core.

    python worker.py <job.json>

job["node"]:
    alembic.import   inputs.file (.abc), params fps -> raw/scene.npz: every item of the file in the scene-arrays
                     layout (lab2shot_worker/scene_arrays.py): cameras, models (poly meshes), point clouds, curves. The import
                     node lists the file's hierarchy from it and makes packets of the items chosen; listing and reading
                     share this one job and its cached result.
    alembic.output   inputs.scene (scene arrays: lengths in the unit the node chose, Y up), params file (.abc) -> the
                     archive (Ogawa): the node's delivery

What the file records and what it doesn't: Alembic keeps transforms, shapes and time in seconds, but neither a unit nor
an up axis (the import node's 单位 and 上轴 say them; scene.npz has unit_cm 1 and no axes) nor skeletons (no characters
either way: the core refuses them for Alembic). An archive may carry a DCC FPS hint in its info — write_archive sets it
and the import reads it back as the rate to convert seconds with.

Items (both directions):
    - where an item is: the path DCCs show, its transform (/rig/cam1 for the shape /rig/cam1/cam1Shape) when it is
      that transform's only shape, else the shape itself; its name is the path's last part;
    - its frames: the frames of every sample of it and of its parent transforms (fps converts seconds; of sub-frame
      samples the one nearest the frame is kept); nothing animated: its one sample's frame;
    - world: every parent transform composed at each of those frames. Written back, each item is a top-level
      transform holding its world per sample and one shape "<name>Shape" under it.

Conventions at the npz boundary (both directions):
    - matrices are column-vector (translation in [:3, 3]); Alembic stores row-vector matrices (translation in the last
      row), so every matrix is transposed here;
    - focal length in mm, apertures in mm (Alembic stores apertures in cm);
    - faces wind counter-clockwise seen from the front (USD / right-handed default); Alembic's convention is clockwise
      (what Maya's and Houdini's exporters write and their importers expect), so each face's corner order is reversed
      both ways, with its face-varying UVs and normals;
    - cameras look down -Z with +Y up (USD / Maya / Houdini / Alembic all agree);
    - lengths are written as they come (the node scaled them to its 单位), no rescale.
"""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import json

import numpy as np

import imath
import imathnumpy
from alembic import Abc, AbcCoreAbstract, AbcGeom, Util
from lab2shot_worker import fail, progress, say, serve, shown
from lab2shot_worker.run import Run
from lab2shot_shared.scene_arrays import SceneArrays, load, text

CM_PER_MM = 0.1
FALLBACK_FPS = 24.0  # lab2shot.data.units.DEFAULT_FPS: neither the node nor the archive says a rate
WRAP = Abc.WrapExistingFlag.kWrapExisting
VARYING = AbcGeom.GeometryScope.kVaryingScope  # one value per point
FACEVARYING = AbcGeom.GeometryScope.kFacevaryingScope
COLOR = "Cd"  # the point colour: Houdini's name, what it reads and writes in an Alembic's arbitrary geometry parameters


# ------------------------------------------------------------------ conversions


def m44_to_np(m) -> np.ndarray:
    """imath.M44d -> [4, 4] with the same element layout (Alembic: row vectors)."""
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)


def np_to_m44(a: np.ndarray):
    return imath.M44d(*(float(v) for v in np.asarray(a, np.float64).reshape(16)))


def filled(array_type, data: np.ndarray, dtype) -> object:
    """A PyImath array (e.g. imath.V3fArray) holding a copy of `data`."""
    data = np.ascontiguousarray(data, dtype=dtype)
    out = array_type(len(data))
    if len(data):
        imathnumpy.arrayToNumpy(out)[...] = data
    return out


def as_numpy(values, dtype=np.float64) -> np.ndarray:
    return np.array(imathnumpy.arrayToNumpy(values), dtype)


def reverse_faces(counts: np.ndarray, per_corner: np.ndarray) -> np.ndarray:
    """Reverse the corner order inside every face (CCW <-> CW); works for any per-corner array (indices, UVs, normals)."""
    counts = np.asarray(counts, np.int64)
    if int(counts.sum()) != len(per_corner):
        fail("E-ALEMBIC-FACES", corners=int(counts.sum()), indices=len(per_corner))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    face = np.repeat(np.arange(len(counts)), counts)
    offset = np.arange(len(per_corner)) - starts[face]
    return per_corner[starts[face] + counts[face] - 1 - offset]


# ------------------------------------------------------------------ the hierarchy


def children(obj) -> list:
    return [obj.getChild(i) for i in range(obj.getNumChildren())]


def is_xform(obj) -> bool:
    return AbcGeom.IXform.matches(obj.getMetaData())


def is_camera(obj) -> bool:
    return AbcGeom.ICamera.matches(obj.getMetaData())


def is_mesh(obj) -> bool:
    return AbcGeom.IPolyMesh.matches(obj.getMetaData())


def is_cloud(obj) -> bool:
    return AbcGeom.IPoints.matches(obj.getMetaData())


def is_curves(obj) -> bool:
    return AbcGeom.ICurves.matches(obj.getMetaData())


def all_objects(obj):
    for child in children(obj):
        yield child
        yield from all_objects(child)


def item_path(obj) -> str:
    """Where a shape is in the hierarchy, as DCCs show it: its transform (/rig/cam1 for /rig/cam1/cam1Shape) when it is
    that transform's only shape, else the shape itself."""
    parent = obj.getParent()
    if parent.valid() and parent.getFullName() != "/" and is_xform(parent) and sum(not is_xform(c) for c in children(parent)) == 1:
        return parent.getFullName()
    return obj.getFullName()


def parent_xforms(obj) -> list:
    """Xform schemas from the nearest parent up to the root."""
    chain = []
    parent = obj.getParent()
    while parent.valid() and parent.getFullName() != "/":
        if is_xform(parent):
            chain.append(AbcGeom.IXform(parent, WRAP).getSchema())
        parent = parent.getParent()
    return chain


def sample_times(schema) -> list[float]:
    ts = schema.getTimeSampling()
    return [ts.getSampleTime(i) for i in range(schema.getNumSamples())]


def world_matrix(xforms: list, sel) -> np.ndarray:
    """Local-to-world, column vectors: the parents' local transforms composed (row vectors: local(parent) *
    local(grandparent) * ...), up to one that does not inherit its parents'."""
    acc = np.eye(4)
    for xs in xforms:
        if xs.getNumSamples() == 0:
            continue
        sample = xs.getValue(sel)
        acc = acc @ m44_to_np(sample.getMatrix())
        if not sample.getInheritsXforms():
            break
    return acc.T


def item_frames(schema, xforms: list, fps: float, where: str) -> dict[int, float]:
    """Frame -> the time to read it at: every sample of the shape and its parents when any is animated, else its one
    sample. Of sub-frame samples (motion blur) the one nearest the frame is kept."""
    times = sorted({t for s in [schema, *xforms] if s.getNumSamples() > 1 for t in sample_times(s)})
    times = times or sample_times(schema)[:1] or [0.0]
    by_frame: dict[int, float] = {}
    for t in times:
        f = int(round(t * fps))
        if f not in by_frame or abs(t * fps - f) < abs(by_frame[f] * fps - f):
            by_frame[f] = t
    if len(by_frame) < len(times):
        say("N-ALEMBIC-SUBFRAMES", path=where, count=len(times) - len(by_frame))
    return dict(sorted(by_frame.items()))


# ------------------------------------------------------------------ read


def open_archive(abc: Path):
    if not abc.is_file():
        fail("E-ALEMBIC-NOFILE", path=shown(abc))
    try:
        return Abc.IArchive(str(abc))
    except RuntimeError:  # "Unknown core type" and the like
        fail("E-ALEMBIC-NOTABC", name=abc.name)


class Item:
    """A shape of the file: where it is, the frames it has, its world at each and the selectors to read them."""

    def __init__(self, obj, schema, fps: float):
        self.obj, self.schema = obj, schema
        self.path = item_path(obj)
        self.name = self.path.rsplit("/", 1)[1]
        xforms = parent_xforms(obj)
        by_frame = item_frames(schema, xforms, fps, self.path)
        self.frames = np.array(list(by_frame), np.int64)
        self.selectors = [Abc.ISampleSelector(float(t)) for t in by_frame.values()]
        self.world = np.stack([world_matrix(xforms, sel) for sel in self.selectors])

    def shape_samples(self) -> list:
        """The selectors to read the shape at: one per frame when the shape itself changes, else its one sample."""
        return self.selectors if self.schema.getNumSamples() > 1 else self.selectors[:1]


def read_camera(obj, fps: float, out: SceneArrays) -> None:
    """A camera's lens at every frame of it: its focal length, apertures and film offset (Alembic's cm -> mm), its
    squeeze (the pixel aspect), its overscan (Alembic's parts of half the film back -> parts of the picture) and the
    lab2shot: user properties the lens's distortion is kept under."""
    it = Item(obj, AbcGeom.ICamera(obj, WRAP).getSchema(), fps)
    lens = {k: [] for k in ("focal", "hap", "vap", "hoff", "voff", "squeeze", "over")}
    has_ops = False
    for sel in it.selectors:
        cam = it.schema.getValue(sel)
        lens["focal"].append(cam.getFocalLength())
        lens["hap"].append(cam.getHorizontalAperture() / CM_PER_MM)
        lens["vap"].append(cam.getVerticalAperture() / CM_PER_MM)
        lens["hoff"].append(cam.getHorizontalFilmOffset() / CM_PER_MM)
        lens["voff"].append(cam.getVerticalFilmOffset() / CM_PER_MM)
        lens["squeeze"].append(cam.getLensSqueezeRatio())
        lens["over"].append([cam.getOverScanLeft(), cam.getOverScanTop(), cam.getOverScanRight(), cam.getOverScanBottom()])
        has_ops = has_ops or cam.getNumOps() > 0
    if has_ops:
        say("W-ALEMBIC-FILMBACKOPS", path=it.path)
    props = user_properties(it.schema, fps)
    size = json.loads(props.pop(RESOLUTION, "null"))
    out.add("camera", it.name, it.path, it.frames, it.world, focal_mm=np.array(lens["focal"], np.float64),
            h_aperture_mm=np.array(lens["hap"], np.float64), v_aperture_mm=np.array(lens["vap"], np.float64),
            center_mm=np.array([lens["hoff"], lens["voff"]], np.float64).T, pixel_aspect=np.float64(lens["squeeze"][0]),
            overscan=np.array(lens["over"][0], np.float64) / 2.0,
            properties=np.array(json.dumps(props, ensure_ascii=False)),
            resolution=np.array(size, np.int64) if size else None)


RESOLUTION = "lab2shot:resolution"  # a camera's picture size [width, height] as JSON: a user property (Alembic has none)


def user_properties(schema, fps: float) -> dict:
    """A schema's lab2shot: user properties: texts, numbers, and a number over frames as {frames, values}."""
    props = schema.getUserProperties()
    out: dict = {}
    if not props.valid():
        return out
    for i in range(props.getNumProperties()):
        header = props.getPropertyHeader(i)
        name = header.getName()
        if not name.startswith("lab2shot:") or not header.isScalar():
            continue
        pod = header.getDataType().getPod()
        if pod == Util.POD.kStringPOD:
            out[name] = Abc.IStringProperty(props, name).getValue()
        elif pod == Util.POD.kFloat64POD:
            prop = Abc.IDoubleProperty(props, name)
            n, sampling = prop.getNumSamples(), prop.getTimeSampling()
            values = [float(prop.getValue(Abc.ISampleSelector(float(sampling.getSampleTime(k))))) for k in range(n)]  # by time: an int selector is taken as seconds
            out[name] = values[0] if n == 1 else {"frames": [int(round(sampling.getSampleTime(k) * fps)) for k in range(n)], "values": values}
    return out


def write_user_properties(schema, properties: dict, samplings) -> None:
    """Named properties as the schema's user properties: a text, a number, or a number over its own frames."""
    user = schema.getUserProperties()
    for name, value in properties.items():
        if isinstance(value, str):
            Abc.OStringProperty(user, name).setValue(value)
        elif isinstance(value, dict):
            prop = Abc.ODoubleProperty(user, name, samplings(value["frames"]))
            for v in value["values"]:
                prop.setValue(float(v))
        else:
            Abc.ODoubleProperty(user, name).setValue(float(value))


def read_model(obj, fps: float, out: SceneArrays) -> None:
    """A poly mesh: topology, UVs and normals of its first sample (faces turned counter-clockwise), its local points
    per frame when they change, else once."""
    it = Item(obj, AbcGeom.IPolyMesh(obj, WRAP).getSchema(), fps)
    sels = it.shape_samples()
    first = it.schema.getValue(sels[0])
    counts = as_numpy(first.getFaceCounts(), np.int32)
    indices = reverse_faces(counts, as_numpy(first.getFaceIndices(), np.int32))  # CW -> CCW
    points = [as_numpy(first.getPositions(), np.float32).reshape(-1, 3)]
    for sel in sels[1:]:
        p = as_numpy(it.schema.getValue(sel).getPositions(), np.float32).reshape(-1, 3)
        if len(p) != len(points[0]):
            say("W-ALEMBIC-VARYINGPOINTS", path=it.path)
            points = points[:1]
            break
        points.append(p)
    extra: dict[str, np.ndarray] = {}
    uvs = it.schema.getUVsParam()
    if uvs.valid():
        sample = uvs.getIndexedValue(sels[0])
        vals = as_numpy(sample.getVals(), np.float32).reshape(-1, 2)
        idx = as_numpy(sample.getIndices(), np.int32)
        if uvs.getScope() == FACEVARYING and len(idx) == len(indices):
            extra.update(uv=vals, uv_indices=reverse_faces(counts, idx))
        elif len(idx) == len(points[0]):  # one per point
            extra.update(uv=vals[idx], uv_indices=indices)
    normals = it.schema.getNormalsParam()
    if normals.valid() and normals.getScope() == FACEVARYING:
        n = as_numpy(normals.getExpandedValue(sels[0]).getVals(), np.float32).reshape(-1, 3)
        if len(n) == len(indices):
            extra["normals"] = reverse_faces(counts, n)
    extra.update(read_face_sets(obj, len(counts)))
    out.add("model", it.name, it.path, it.frames, it.world, counts=counts, indices=indices, points=np.stack(points), **extra)


def read_face_sets(obj, faces: int) -> dict:
    """A mesh's face sets as the arrays' 分区 (subset_names / subset_counts / subset_faces): Alembic's FaceSet is the
    same thing as USD's GeomSubset — a named set of this mesh's faces (Maya writes its shading groups and face sets
    this way, Houdini reads them as primitive groups). Face order is the same in both, so no index is turned here;
    an index outside the mesh (a file written against another topology) is dropped and said once."""
    names, counts, picked = [], [], []
    for i in range(obj.getNumChildren()):
        child = obj.getChild(i)
        if not AbcGeom.IFaceSet.matches(child.getMetaData()):
            continue
        schema = AbcGeom.IFaceSet(child, WRAP).getSchema()
        chosen = np.unique(as_numpy(schema.getValue(Abc.ISampleSelector(0)).getFaces(), np.int32))
        inside = chosen[(chosen >= 0) & (chosen < faces)]
        if len(inside) != len(chosen):
            say("W-ALEMBIC-FACESETRANGE", path=child.getFullName(), dropped=int(len(chosen) - len(inside)))
        names.append(child.getName())
        counts.append(len(inside))
        picked.append(inside)
    if not names:
        return {}
    return {"subset_names": np.array(names), "subset_counts": np.array(counts, np.int64),
            "subset_faces": np.concatenate(picked).astype(np.int32) if picked else np.zeros(0, np.int32)}


def point_colors(schema):
    """A point cloud's or a curve set's colour parameter (Cd, rgb or float triples), or None."""
    params = schema.getArbGeomParams()
    header = params.getPropertyHeader(COLOR) if params.valid() else None
    for kind in (AbcGeom.IC3fGeomParam, AbcGeom.IV3fGeomParam):
        if header is not None and kind.matches(header):
            return kind(params, COLOR)
    return None


def per_point(schema, sels, per: list[np.ndarray]) -> dict[str, np.ndarray]:
    """The widths and Cd colours a point cloud or a set of curves carries one per point, over its samples `sels`
    (`per`: each sample's points). A constant width is not one per point and is left out: the core writes the one
    width itself."""
    out: dict[str, np.ndarray] = {}
    widths = schema.getWidthsParam()
    for name, param in {"widths": widths if widths.valid() else None, "colors": point_colors(schema)}.items():
        if param is None:
            continue
        size = 3 if name == "colors" else 1
        vals = [as_numpy(param.getExpandedValue(sel).getVals(), np.float32).reshape(-1) for sel in sels]
        if all(v.size == size * len(p) for v, p in zip(vals, per)):
            joined = np.concatenate(vals)
            out[name] = joined if size == 1 else joined.reshape(-1, 3)
    return out


def tracked(schema, sels, per: list[np.ndarray]) -> dict[str, np.ndarray]:
    """A tracked cloud's ids (when they say more than counting the points of each sample, each once per sample),
    velocities (v) and visible, each only when every sample has one per point."""
    out: dict[str, np.ndarray] = {}
    samples = [schema.getValue(sel) for sel in sels]
    ids = [as_numpy(s.getIds(), np.int64).reshape(-1) for s in samples]
    if all(len(i) == len(p) and len(np.unique(i)) == len(i) for i, p in zip(ids, per)) and \
            any(not np.array_equal(i, np.arange(len(i))) for i in ids):
        out["ids"] = np.concatenate(ids)
    vel = [s.getVelocities() for s in samples]
    if all(v is not None and len(v) == len(p) for v, p in zip(vel, per)) and any(len(p) for p in per):
        out["velocities"] = np.concatenate([as_numpy(v, np.float32).reshape(-1, 3) for v in vel])
    params = schema.getArbGeomParams()
    header = params.getPropertyHeader("visible") if params.valid() else None
    if header is not None and AbcGeom.IInt32GeomParam.matches(header):
        seen = [as_numpy(AbcGeom.IInt32GeomParam(params, "visible").getExpandedValue(sel).getVals(), np.int32).reshape(-1) for sel in sels]
        if all(len(s) == len(p) for s, p in zip(seen, per)):
            out["visible"] = np.concatenate(seen)
    return out


def read_cloud(obj, fps: float, out: SceneArrays) -> None:
    """A point cloud: its local points per frame when they change, else once; Cd colours and widths when there is one
    per point."""
    it = Item(obj, AbcGeom.IPoints(obj, WRAP).getSchema(), fps)
    sels = it.shape_samples()
    per = [as_numpy(it.schema.getValue(sel).getPositions(), np.float32).reshape(-1, 3) for sel in sels]
    extra = per_point(it.schema, sels, per)
    extra.update(tracked(it.schema, sels, per))
    out.add("points", it.name, it.path, it.frames, it.world, counts=np.array([len(p) for p in per], np.int64),
            points=np.concatenate(per).reshape(-1, 3), **extra)


def read_curves(obj, fps: float, out: SceneArrays) -> None:
    """A set of 3D curves: how many points each curve has and its local points per frame when they change, else once;
    Cd colours and widths when there is one per point."""
    it = Item(obj, AbcGeom.ICurves(obj, WRAP).getSchema(), fps)
    sels = it.shape_samples()
    samples = [it.schema.getValue(sel) for sel in sels]
    per = [as_numpy(s.getPositions(), np.float32).reshape(-1, 3) for s in samples]
    strands = [as_numpy(s.getCurvesNumVertices(), np.int32).reshape(-1) for s in samples]
    for n, p in zip(strands, per):
        if int(n.sum()) != len(p):
            fail("E-ALEMBIC-CURVEPOINTS", name=it.name, counted=int(n.sum()), points=len(p))
    extra = per_point(it.schema, sels, per)
    out.add("curves", it.name, it.path, it.frames, it.world, counts=np.array([len(p) for p in per], np.int64),
            curve_counts=np.array([len(n) for n in strands], np.int64), curve_vertex_counts=np.concatenate(strands),
            points=np.concatenate(per).reshape(-1, 3), **extra)


READERS = ((is_camera, read_camera), (is_mesh, read_model), (is_cloud, read_cloud), (is_curves, read_curves))


def import_file(run: Run) -> None:
    """Every camera, model, point cloud and set of curves of the file (Alembic records no unit and no up axis:
    unit_cm 1, no axes).
    The rate is the archive's own DCC FPS hint (write_archive and DCC exporters record it); a file without one reads
    24. It only converts Alembic's seconds into frame numbers; the import node has no frame-rate parameter of its own,
    so the frames found here are the data."""
    job = run.job
    run.stage("读取 Alembic")
    archive = open_archive(job.inputs["file"])
    recorded = float(Abc.GetArchiveInfo(archive).get("dccFPS") or 0.0)
    fps = recorded or FALLBACK_FPS
    out = SceneArrays(fps, unit_cm=1.0)
    shapes = [(o, read) for o in all_objects(archive.getTop()) for match, read in READERS if match(o)]
    for i, (obj, read) in enumerate(shapes):
        read(obj, fps, out)
        progress(i + 1, len(shapes), obj.getFullName())
    frames = sorted({int(f) for items in out.items.values() for item in items for f in item["frames"]})
    out.top["frames"] = np.array(frames, np.int64)
    out.save(job.raw_dir / "scene.npz")
    # `frames` is the count here (it overrides the standard [first, last])
    run.finish(frames, kind="scene", **{kind: len(items) for kind, items in out.items.items()}, frames=len(frames))


# ------------------------------------------------------------------ write


_NAME_BAD = re.compile(r"[^A-Za-z0-9_]")


class Names:
    """Maya-safe, sibling-unique object names."""

    def __init__(self) -> None:
        self.used: dict[object, set[str]] = {}

    def __call__(self, parent_key: object, name: str) -> str:
        base = _NAME_BAD.sub("_", name.strip()) or "node"
        if base[0].isdigit():
            base = "_" + base
        used = self.used.setdefault(parent_key, set())
        out, n = base, 1
        while out in used:
            n += 1
            out = f"{base}{n}"
        used.add(out)
        return out


def matrix_sample(column_vector: np.ndarray):
    s = AbcGeom.XformSample()
    s.setMatrix(np_to_m44(np.asarray(column_vector).T))
    return s


class Samplings:
    """The archive's time samplings, one per list of frames the items have: uniform when the frames follow on one
    another, else at each frame's time. An item's samples are its frames' (an item with one sample is written once, at
    its frame)."""

    def __init__(self, archive, fps: float):
        self.archive, self.fps, self.made = archive, fps, {}

    def __call__(self, frames) -> int:
        key = tuple(int(f) for f in frames)
        if key not in self.made:
            if len(key) > 1 and any(b - a != 1 for a, b in zip(key, key[1:])):
                T = AbcCoreAbstract.TimeSamplingType
                times = AbcCoreAbstract.TimeVector()
                times[:] = [f / self.fps for f in key]
                ts = AbcCoreAbstract.TimeSampling(T(T.AcyclicNumSamples(), T.AcyclicTimePerCycle()), times)
            else:
                ts = AbcCoreAbstract.TimeSampling(1.0 / self.fps, key[0] / self.fps)
            self.made[key] = self.archive.addTimeSampling(ts)
        return self.made[key]


def placed(top, names: Names, item: dict, tsidx: int, groups: dict):
    """The item's transform, its world per sample, under its path's groups below the scene's /shot (an import's folder and the
    hierarchy its file had: transforms with one identity sample, made once each, so the item's world is its local one
    too); returns (transform, its name)."""
    parts = [p for p in text(item["path"]).split("/") if p]
    if parts and parts[0] == "shot":
        parts = parts[1:]
    parent, key = top, "/"
    for segment in parts[:-1]:
        inner = f"{key}{segment}/"
        if inner not in groups:
            groups[inner] = AbcGeom.OXform(parent, names(key, segment), 0)
            groups[inner].getSchema().set(matrix_sample(np.eye(4)))
        parent, key = groups[inner], inner
    name = names(key, parts[-1] if parts else text(item["name"]))
    world = np.asarray(item["world"], np.float64).reshape(-1, 4, 4)
    if len(world) not in (1, len(item["frames"])):
        fail("E-ALEMBIC-TRANSFORMS", name=name, transforms=len(world), frames=len(item["frames"]))
    xf = AbcGeom.OXform(parent, name, tsidx)
    for m in world:
        xf.getSchema().set(matrix_sample(m))
    return xf, name


def write_model(parent, name: str, item: dict, tsidx: int, report, samplings) -> None:
    counts = np.asarray(item["counts"], np.int32)
    indices = np.asarray(item["indices"], np.int32)
    points = np.asarray(item["points"], np.float32)
    points = points[None] if points.ndim == 2 else points
    if len(points) not in (1, len(item["frames"])):
        fail("E-ALEMBIC-MESHFRAMES", name=name, samples=len(points), frames=len(item["frames"]))
    extra = {}
    if "uv" in item:
        extra["iUVs"] = AbcGeom.OV2fGeomParamSample(
            filled(imath.V2fArray, np.asarray(item["uv"]).reshape(-1, 2), np.float32),
            filled(imath.UnsignedIntArray, reverse_faces(counts, np.asarray(item["uv_indices"])), np.uint32), FACEVARYING)
    if "normals" in item:
        extra["iNormals"] = AbcGeom.ON3fGeomParamSample(
            filled(imath.V3fArray, reverse_faces(counts, np.asarray(item["normals"]).reshape(-1, 3)), np.float32), FACEVARYING)
    shape = AbcGeom.OPolyMesh(parent, f"{name}Shape", tsidx)
    mesh = shape.getSchema()
    topo_i = filled(imath.IntArray, reverse_faces(counts, indices), np.int32)  # CCW -> CW
    topo_c = filled(imath.IntArray, counts, np.int32)
    for f, p in enumerate(points):
        P = filled(imath.V3fArray, p, np.float32)
        mesh.set(AbcGeom.OPolyMeshSchemaSample(P, topo_i, topo_c, **extra) if f == 0 else AbcGeom.OPolyMeshSchemaSample(P))
        report(f"网格 {name}")
    write_face_sets(shape, item)


def write_face_sets(shape, item: dict) -> None:
    """The mesh's 分区 as Alembic face sets, one child FaceSet each: Maya and Houdini read them as face sets and
    primitive groups (USD's GeomSubset on the other side, data/subsets.py). Faces keep their order between the two,
    so nothing is turned here; a part may overlap another, so they are written non-exclusive."""
    if "subset_names" not in item:
        return
    names = [text(n) for n in np.asarray(item["subset_names"]).reshape(-1)]
    cuts = np.cumsum(np.asarray(item["subset_counts"], np.int64).reshape(-1))[:-1]
    for name, faces in zip(names, np.split(np.asarray(item["subset_faces"], np.int64).reshape(-1), cuts)):
        schema = AbcGeom.OFaceSet(shape, name).getSchema()
        schema.setFaceExclusivity(AbcGeom.FaceSetExclusivity.kFaceSetNonExclusive)
        sample = AbcGeom.OFaceSetSchemaSample()
        sample.setFaces(filled(imath.IntArray, faces, np.int32))
        schema.set(sample)


def write_cloud(parent, name: str, item: dict, tsidx: int, report, samplings) -> None:
    """Points per sample with their ids (a tracked cloud's own, else counting them); widths and colours (Cd) one per
    point; a tracked cloud's velocities (v) and visible."""
    per = np.asarray(item["counts"], np.int64).reshape(-1)
    points = np.asarray(item["points"], np.float32).reshape(-1, 3)
    if len(per) not in (1, len(item["frames"])) or int(per.sum()) != len(points):
        fail("E-ALEMBIC-CLOUDPOINTS", name=name, groups=len(per), counted=int(per.sum()), points=len(points), frames=len(item["frames"]))
    widths = np.asarray(item["widths"], np.float32).reshape(-1) if "widths" in item else None
    colors = np.asarray(item["colors"], np.float32).reshape(-1, 3) if "colors" in item else None
    ids = np.asarray(item["ids"]).reshape(-1) if "ids" in item else None
    vel = np.asarray(item["velocities"], np.float32).reshape(-1, 3) if "velocities" in item else None
    seen = np.asarray(item["visible"], np.int32).reshape(-1) if "visible" in item else None
    schema = AbcGeom.OPoints(parent, f"{name}Shape", tsidx).getSchema()
    cd = AbcGeom.OC3fGeomParam(schema.getArbGeomParams(), COLOR, False, VARYING, 1, Abc.Argument(tsidx)) if colors is not None else None
    vis = AbcGeom.OInt32GeomParam(schema.getArbGeomParams(), "visible", False, VARYING, 1, Abc.Argument(tsidx)) if seen is not None else None
    starts = np.concatenate([[0], np.cumsum(per)])
    for f in range(len(per)):
        part = slice(int(starts[f]), int(starts[f + 1]))
        sample = AbcGeom.OPointsSchemaSample()
        sample.setPositions(filled(imath.V3fArray, points[part], np.float32))
        sample.setIds(filled(imath.IntArray, ids[part] if ids is not None else np.arange(int(per[f])), np.int32))
        if vel is not None:
            sample.setVelocities(filled(imath.V3fArray, vel[part], np.float32))
        # setWidths copies the widths without keeping their array alive: hold it until the sample is written
        w = AbcGeom.OFloatGeomParamSample(filled(imath.FloatArray, widths[part], np.float32), VARYING) if widths is not None else None
        if w is not None:
            sample.setWidths(w)
        schema.set(sample)
        del w
        if cd is not None:
            cd.set(AbcGeom.OC3fGeomParamSample(filled(imath.C3fArray, colors[part], np.float32), VARYING))
        if vis is not None:
            vis.set(AbcGeom.OInt32GeomParamSample(filled(imath.IntArray, seen[part], np.int32), VARYING))
        report(f"点云 {name}")


def write_camera(parent, name: str, item: dict, tsidx: int, report, samplings) -> None:
    """The lens per sample: apertures and film offset in cm as Alembic keeps them, the pixel aspect as its squeeze, the
    overscan as parts of half the film back; the distortion as user properties (read_camera reads them all back)."""
    lens = {k: np.asarray(item[f"{k}_mm"], np.float64).reshape(-1) for k in ("focal", "h_aperture", "v_aperture")}
    center = np.asarray(item["center_mm"], np.float64).reshape(-1, 2)
    over = np.asarray(item["overscan"], np.float64).reshape(4) * 2.0  # left top right bottom
    n = max(max(len(v) for v in lens.values()), len(center))
    schema = AbcGeom.OCamera(parent, f"{name}Shape", tsidx).getSchema()
    props = json.loads(text(item["properties"]))
    if "resolution" in item:  # Alembic has no picture size of its own
        props[RESOLUTION] = json.dumps([int(v) for v in np.asarray(item["resolution"]).reshape(-1)[:2]])
    write_user_properties(schema, props, samplings)
    for f in range(n):
        cam = AbcGeom.CameraSample()
        cam.setFocalLength(float(lens["focal"][min(f, len(lens["focal"]) - 1)]))
        cam.setHorizontalAperture(float(lens["h_aperture"][min(f, len(lens["h_aperture"]) - 1)]) * CM_PER_MM)
        cam.setVerticalAperture(float(lens["v_aperture"][min(f, len(lens["v_aperture"]) - 1)]) * CM_PER_MM)
        cam.setHorizontalFilmOffset(float(center[min(f, len(center) - 1), 0]) * CM_PER_MM)
        cam.setVerticalFilmOffset(float(center[min(f, len(center) - 1), 1]) * CM_PER_MM)
        cam.setLensSqueezeRatio(float(item["pixel_aspect"]))
        cam.setOverScanLeft(float(over[0]))
        cam.setOverScanTop(float(over[1]))
        cam.setOverScanRight(float(over[2]))
        cam.setOverScanBottom(float(over[3]))
        schema.set(cam)
        report(f"相机 {name}")


def write_curves(parent, name: str, item: dict, tsidx: int, report, samplings) -> None:
    """Curves per sample: how many points each has, its points, and widths and colours (Cd) one per point. Linear,
    non-periodic: what hair, guide curves and motion trails are, and what Houdini and Maya read back as curves."""
    per = np.asarray(item["counts"], np.int64).reshape(-1)
    per_curve = np.asarray(item["curve_counts"], np.int64).reshape(-1)
    points = np.asarray(item["points"], np.float32).reshape(-1, 3)
    strands = np.asarray(item["curve_vertex_counts"], np.int32).reshape(-1)
    if len(per) not in (1, len(item["frames"])) or int(per.sum()) != len(points) or int(strands.sum()) != len(points):
        fail("E-ALEMBIC-CURVEPOINTS", name=name, counted=int(strands.sum()), points=len(points))
    widths = np.asarray(item["widths"], np.float32).reshape(-1) if "widths" in item else None
    colors = np.asarray(item["colors"], np.float32).reshape(-1, 3) if "colors" in item else None
    schema = AbcGeom.OCurves(parent, f"{name}Shape", tsidx).getSchema()
    cd = AbcGeom.OC3fGeomParam(schema.getArbGeomParams(), COLOR, False, VARYING, 1, Abc.Argument(tsidx)) if colors is not None else None
    starts = np.concatenate([[0], np.cumsum(per)])
    curve_starts = np.concatenate([[0], np.cumsum(per_curve)])
    for f in range(len(per)):
        part = slice(int(starts[f]), int(starts[f + 1]))
        sample = AbcGeom.OCurvesSchemaSample()
        sample.setPositions(filled(imath.V3fArray, points[part], np.float32))
        sample.setCurvesNumVertices(filled(imath.IntArray, strands[int(curve_starts[f]):int(curve_starts[f + 1])], np.int32))
        sample.setType(AbcGeom.CurveType.kLinear)
        sample.setWrap(AbcGeom.CurvePeriodicity.kNonPeriodic)
        sample.setBasis(AbcGeom.BasisType.kNoBasis)
        # setWidths copies the widths without keeping their array alive: hold it until the sample is written
        w = AbcGeom.OFloatGeomParamSample(filled(imath.FloatArray, widths[part], np.float32), VARYING) if widths is not None else None
        if w is not None:
            sample.setWidths(w)
        schema.set(sample)
        del w
        if cd is not None:
            cd.set(AbcGeom.OC3fGeomParamSample(filled(imath.C3fArray, colors[part], np.float32), VARYING))
        report(f"三维曲线 {name}")


WRITERS = {"model": write_model, "points": write_cloud, "curves": write_curves, "camera": write_camera}


def write_archive(items: dict[str, list[dict]], tmp: Path, fps: float, report) -> dict:
    """Everything Alembic lives in this function: the archive is finished when it returns."""
    archive = Abc.CreateArchiveWithInfo(str(tmp), fps, "Lab2Shot", "Lab2Shot alembic.output")
    top = archive.getTop()
    samplings = Samplings(archive, fps)
    names = Names()
    groups: dict = {}  # the transforms an item's path makes above it, once each
    for kind, write in WRITERS.items():
        for item in items[kind]:
            tsidx = samplings(item["frames"])
            xf, name = placed(top, names, item, tsidx, groups)
            write(xf, name, item, tsidx, report, samplings)
    return {kind: len(items[kind]) for kind in WRITERS}


def write_abc(run: Run) -> None:
    job = run.job
    out = Path(job.params["file"])
    if out.suffix.lower() != ".abc":
        fail("E-ALEMBIC-SUFFIX", path=shown(out))
    top, items = load(job.inputs["scene"])
    if items["character"]:
        fail("E-ALEMBIC-NOSKELETON", count=len(items["character"]))
    fps = float(top["fps"])
    if fps <= 0:
        fail("E-ALEMBIC-FPS", fps=fps)
    total = max(1, sum(max(len(np.asarray(i["frames"]).reshape(-1)), 1) for kind in WRITERS for i in items[kind]))
    done = [0]

    def report(message: str) -> None:
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == total:
            progress(min(done[0], total), total, message)

    run.stage("写出 Alembic")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    try:
        counts = write_archive(items, tmp, fps, report)
        gc.collect()  # every Alembic object is gone: the archive is closed
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    run.finish([], kind="abc", path=str(out), **counts)


# ------------------------------------------------------------------ main


NODES = {"alembic.import": import_file, "alembic.output": write_abc}


def main(job_path: str) -> None:
    run = Run.start(job_path, tuple(NODES), "Alembic", gpu=False)  # PyAlembic: no GPU
    NODES[run.job.node or next(iter(NODES))](run)  # Run.start already validated the node name


if __name__ == "__main__":
    serve(main)
