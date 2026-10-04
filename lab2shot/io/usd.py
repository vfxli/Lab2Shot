"""USD output shared by every adapter.

Conventions (Houdini / Maya friendly):
  * upAxis = Y, metersPerUnit = 0.01 (centimeters)
  * time codes are the source frame numbers (a 1001-1100 plate stays 1001-1100)
  * camera lens values in millimeters: USD stores focal length and aperture in
    tenths of a scene unit, which is exactly mm in a centimeter stage

Matrices passed to this module use the column-vector convention
(p' = M @ p, translation in the last column); they are transposed to USD's
row-vector layout here, so adapters never have to think about it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdSkel, Vt

from lab2shot_shared import names
from lab2shot_shared.motion import continuous, local_from_world, matrix_to_quat
from lab2shot_shared.names import ORIGINAL
from lab2shot_shared.units import DEFAULT_FPS

from .. import __version__
from ..messages import Msg
from . import FileProblem

METERS_PER_UNIT = 0.01
ROOT_PATH = "/shot"
STAGE_FPS = DEFAULT_FPS  # stage time base: one time code per frame (apply_conventions)


# --------------------------------------------------------------------------- stage


def apply_conventions(stage: Usd.Stage, frames) -> None:
    """Lab2Shot's stage conventions: Y up, centimetres, the shot's frame range, /shot as default prim.

    The time base is fixed (`STAGE_FPS`): one frame of the shot is one time code, so frame numbers and time codes are
    always equal. Frame rate is not a property of the data in Lab2Shot (it is a project-level setting and there is no
    project concept here), so every intermediate USD uses the same time base and needs no conversion. The delivery
    frame rate is a parameter of the output settings node and is written only to the delivered file (data/scene.py
    export and the format modules' workers)."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    stage.SetTimeCodesPerSecond(STAGE_FPS)
    stage.SetFramesPerSecond(STAGE_FPS)
    if len(frames):
        stage.SetStartTimeCode(min(frames))
        stage.SetEndTimeCode(max(frames))
    stage.SetDefaultPrim(stage.GetPrimAtPath(ROOT_PATH) or UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim())


def create_stage(frames: list[int] | tuple[int, ...], info: dict | None = None) -> Usd.Stage:
    """In-memory stage with Lab2Shot conventions; save with `save_stage`."""
    stage = Usd.Stage.CreateInMemory()
    apply_conventions(stage, frames)
    stage.GetRootLayer().customLayerData = {"lab2shot": layer_data({"lab2shot_version": __version__, **(info or {})})}
    return stage


def layer_data(value):
    """A value as USD layer metadata can hold it: lists become typed arrays (a list of numbers, of strings), a list
    mixing kinds becomes its JSON text, nested dicts are converted throughout, numpy scalars become Python ones."""
    if isinstance(value, dict):  # a None has no USD type: the key is left out (a worker's "not measured")
        return {str(k): layer_data(v) for k, v in value.items() if v is not None}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        items = [x.item() if isinstance(x, np.generic) else x for x in value]
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
            return Vt.DoubleArray([float(x) for x in items])
        if all(isinstance(x, str) for x in items):
            return Vt.StringArray(items)
        return json.dumps(items, ensure_ascii=False, default=str)
    return value


def save_stage(stage: Usd.Stage, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not stage.GetRootLayer().Export(str(path)):
        raise FileProblem(Msg("E-USD-WRITE", path=str(path)))
    return path


# Every prim and property name written here comes from lab2shot_shared/names.py (identifier; unique among siblings),
# with the original kept where the two differ (customData ORIGINAL, jointNames, blend-shape tokens): one rule for
# joints, blend shapes, subsets, meshes, groups, cameras and properties.


def _children(stage: Usd.Stage, parent: str) -> set[str]:
    prim = stage.GetPrimAtPath(parent)
    return {c.GetName() for c in prim.GetAllChildren()} if prim else set()


def child_name(stage: Usd.Stage, parent: str, name: str) -> str:
    """A new child of the prim at `parent`: `name` as an identifier, unique among the children already there."""
    return names.unique(names.identifier(name), _children(stage, parent))


def free_path(stage: Usd.Stage, path: str) -> str:
    """`path` (its last part already an identifier), or its last part made unique among its siblings: two things a file
    held at one path (some formats let siblings share a name) land side by side, table and table_2."""
    parent, leaf = path.rsplit("/", 1)
    return f"{parent}/{names.unique(leaf, _children(stage, parent))}"


def keep_original(prim: Usd.Prim, name: str) -> None:
    """Record `name` on a prim whose own name is not it (an identifier made of it): name_of then gives `name`."""
    if prim.GetName() != name:
        prim.SetCustomDataByKey(ORIGINAL, name)


def name_of(prim: Usd.Prim) -> str:
    """The name a prim stands for: the kept original, else its own name as it is (never decoded: a name from another
    tool that looks like a spelling, Bone_u0041, is itself)."""
    return str(prim.GetCustomDataByKey(ORIGINAL) or prim.GetName())


def shown_path(prim: Usd.Prim) -> str:
    """The prim's place in the hierarchy by the names it stands for (name_of of it and each ancestor): what a format
    that takes any text as a name (FBX) writes, where the prim path holds identifiers."""
    parts = []
    while prim and not prim.IsPseudoRoot():
        parts.append(name_of(prim))
        prim = prim.GetParent()
    return names.join_path(reversed(parts))


# --------------------------------------------------------------------------- where imported things are placed


IMPORT_GROUP = "lab2shot:import_group"  # customData on the folder an import node's entries are placed under


def import_group(node_id: str, type_id: str, file: str) -> str:
    """The folder every entry of one import node is placed under, /shot/<this>/... (every import gets one, so the
    hierarchy keeps its shape when a second import is added), as a name: the imported file's stem, whatever the node
    is called (in a DCC the asset's own name, `zhanshi`, says more than a node's, `char_fbx`).
    import_path and mark_group make it an identifier and keep this name on the folder (keep_original), so a DCC and an
    FBX delivery show it as given. Two imports landing on the same folder are refused when packed (data/scene.py pack,
    B-NAME-SAME)."""
    return Path(file).stem


def import_path(group: str, source: str) -> str:
    """/shot/<group>/<the entry's path in its file>: an imported entry keeps the hierarchy it had, the group and each
    segment an identifier (names.identifier: a USD file's own names stay as they are; place() keeps the original of a
    segment that changed). A file Lab2Shot wrote holds everything under its own /shot: that root is not repeated."""
    segments = names.split_path(source)
    if segments and segments[0] == ROOT_PATH.strip("/"):
        segments = segments[1:]
    return "/".join([ROOT_PATH, names.identifier(group), *(names.identifier(s) for s in segments)])


def mark_group(stage: Usd.Stage, group: str, owner: str) -> None:
    """Mark /shot/<group> as the folder of the import cook `owner` (its fingerprint): packed, the folders of one cook
    are one folder (its camera and its models together), another cook's of the same name a sibling (data/scene.py
    pack)."""
    prim = stage.GetPrimAtPath(f"{ROOT_PATH}/{names.identifier(group)}")
    if prim:
        prim.SetCustomDataByKey(IMPORT_GROUP, owner)
        keep_original(prim, group)


def free_transform_op(xf: UsdGeom.Xformable, base: str) -> UsdGeom.XformOp:
    """Add a transform op named `base` to `xf`, or base1, base2, ... when that name is taken; return the new op.

    `AddTransformOp(opSuffix=base)` alone is not enough: USD allows one op per suffix and
    `UsdGeomXformable::AddXformOp` raises "The xformOp 'xformOp:transform:<base>' already exists in xformOpOrder"
    on a duplicate. Scenes written by Lab2Shot are USD (data/scene.py SCENE_FILE), so the same data can pass through
    twice: the first write already bakes the /shot transform onto each child prim as `:shot`, and placing and writing
    it again would collide with that name, failing the whole output with `E-COOK-FAILED`. `formats/usd/reader.py`
    uses this function as well when placing imported prims (`placed`, `placed1`, ...)."""
    names = {o.GetOpName() for o in xf.GetOrderedXformOps()}
    suffix = next(f"{base}{n or ''}" for n in range(len(names) + 1) if f"xformOp:transform:{base}{n or ''}" not in names)
    return xf.AddTransformOp(opSuffix=suffix)


def place(stage: Usd.Stage, path: str, source: str = "") -> None:
    """Make `path`'s missing ancestors identity Xform groups (a group carries no transform: whatever is placed at `path`
    carries its whole world transform), and record on every prim of `path` below its group whose name had to change the
    name it had in its file (lab2shot:name). The prim at `path` itself is its writer's to define."""
    at = Sdf.Path(path)
    prefixes = at.GetPrefixes()  # /shot, /shot/<group>, ..., the path itself
    for p in prefixes[:-1]:
        if not stage.GetPrimAtPath(p):
            UsdGeom.Xform.Define(stage, p)
    if source:
        wanted = names.split_path(source)
        if wanted and wanted[0] == ROOT_PATH.strip("/"):
            wanted = wanted[1:]
        ours = at.pathString.split("/")[-len(wanted):] if wanted else []
        prefix = Sdf.Path("/".join(at.pathString.split("/")[:-len(wanted)])) if wanted else at
        for i, (was, now) in enumerate(zip(wanted, ours)):
            if was != now:
                prim = stage.GetPrimAtPath(prefix.AppendPath("/".join(ours[: i + 1])))
                if prim:
                    keep_original(prim, was)


def _usd_matrices(m: np.ndarray) -> Vt.Matrix4dArray:
    m = np.asarray(m, dtype=np.float64).reshape(-1, 4, 4)
    return Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(np.transpose(m, (0, 2, 1))))


# --------------------------------------------------------------------------- camera

# Research code speaks OpenCV cameras (+X right, +Y down, +Z forward, metres); USD cameras look down -Z with +Y up,
# and Lab2Shot works in centimetres. The conversion itself (CV_TO_GL, opencv_poses_to_usd) is data/units.py: io/ is
# the base layer (no data type, no unit belongs to it), data/ the one home for that axis-and-scale conversion.


@dataclass
class CameraData:
    width: int  # image resolution in pixels
    height: int
    focal_mm: float | np.ndarray  # constant, or one value per frame
    filmback_mm: float  # horizontal aperture; io/ is the base layer, so it owns no default (data.units.FILMBACK_MM does)
    camera_to_world: np.ndarray | None = None  # [4,4] or [F,4,4]; None = at origin looking down -Z
    near_far_cm: tuple[float, float] = (1.0, 1.0e6)
    center_mm: np.ndarray | None = None  # [F,2] or [1,2]: the lens centre off the picture's centre (aperture offsets), mm
    pixel_aspect: float = 1.0
    overscan: tuple[int, int, int, int] = (0, 0, 0, 0)  # pixels past the picture: left, top, right, bottom
    properties: dict = field(default_factory=dict)  # the lens's named properties (data/camera.py lens_properties)


def _set_values(attr: Usd.Attribute, values, frames: list[int]) -> None:
    """One value when it never changes, else a time sample at each frame."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 1 or np.allclose(values, values[0]):
        attr.Set(float(values[0]))
    else:
        for f, value in zip(frames, values):
            attr.Set(float(value), Usd.TimeCode(f))


def write_camera(stage: Usd.Stage, name: str, cam: CameraData, frames: list[int], path: str = "", source: str = "",
                 lens: str = "") -> UsdGeom.Camera:
    """/shot/<name>, or at `path` (an imported camera where its file had it, `source`: import_path) under identity
    groups, its import's folder marked and every renamed segment's original kept (place). Its lens beyond the focal
    length and film back: the aperture offsets, and under the namespace `lens` the pixel aspect, overscan and the
    named properties (OpenUSD has no lens distortion schema)."""
    named = not path
    path = path or f"{ROOT_PATH}/{names.identifier(name)}"
    place(stage, path)
    usd_cam = UsdGeom.Camera.Define(stage, path)
    place(stage, path, source)
    if named:
        keep_original(usd_cam.GetPrim(), name)
    usd_cam.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    usd_cam.CreateHorizontalApertureAttr(float(cam.filmback_mm))
    # the film back is as high as the picture and the pixels make it: F_h = F_w H / (W a), so an
    # anamorphic squeeze (a = 2) is kept by the aperture pair, not by a squeeze factor USD has no place for
    usd_cam.CreateVerticalApertureAttr(float(cam.filmback_mm * cam.height / (cam.width * (cam.pixel_aspect or 1.0))))
    usd_cam.CreateClippingRangeAttr(Gf.Vec2f(*cam.near_far_cm))
    _set_values(usd_cam.CreateFocalLengthAttr(), cam.focal_mm, frames)

    # center_mm is the lens centre (the principal point) off the picture's centre, +x right +y up (data/camera.py). USD's
    # aperture offsets say the opposite thing: where the aperture window (the picture) sits off the lens axis. A lens
    # centre 2 mm right of the picture's centre is a picture 2 mm left of the axis: the offsets are the negated centre.
    # camera_lens reads them back the same way. Checked against ground-truth face landmarks (a head fitted in a crop
    # of the plate projects onto the face only with this sign).
    center = np.zeros((1, 2)) if cam.center_mm is None else np.asarray(cam.center_mm, np.float64).reshape(-1, 2)
    if center.any():
        _set_values(usd_cam.CreateHorizontalApertureOffsetAttr(), -center[:, 0], frames)
        _set_values(usd_cam.CreateVerticalApertureOffsetAttr(), -center[:, 1], frames)
    prim = usd_cam.GetPrim()
    if cam.pixel_aspect != 1.0:
        named_attr(prim, lens + "pixelAspect", Sdf.ValueTypeNames.Double).Set(float(cam.pixel_aspect))
    if any(cam.overscan):
        named_attr(prim, lens + "overscan", Sdf.ValueTypeNames.Int4).Set(Gf.Vec4i(*[int(v) for v in cam.overscan]))
    write_properties(prim, cam.properties)

    if cam.camera_to_world is not None:
        xf = usd_cam.AddTransformOp()
        mats = np.asarray(cam.camera_to_world, dtype=np.float64)
        if mats.ndim == 2:
            xf.Set(Gf.Matrix4d(mats.T.tolist()))
        else:
            for f, m in zip(frames, mats):
                xf.Set(Gf.Matrix4d(m.T.tolist()), Usd.TimeCode(f))

    # plate resolution travels with the camera (no render-settings prims in the scene)
    usd_cam.GetPrim().SetCustomDataByKey("lab2shot:resolution", Gf.Vec2i(cam.width, cam.height))
    return usd_cam


def named_attr(prim: Usd.Prim, name: str, type_name) -> Usd.Attribute:
    """A new custom attribute for `name`: each namespace segment an identifier, unique on the prim (names.unique), and
    `name` itself kept in its custom data when that changed it ("Distortion - Degree 2"; read_properties reads it)."""
    safe = names.unique(":".join(names.identifier(part) for part in name.split(":")), set(prim.GetPropertyNames()))
    attr = prim.CreateAttribute(safe, type_name, custom=True)
    if safe != name:
        attr.SetCustomDataByKey(ORIGINAL, name)
    return attr


def write_properties(prim: Usd.Prim, properties: dict) -> None:
    """Named properties as custom attributes: a text, a number, or a number over frames ({frames, values})."""
    for name, value in properties.items():
        if isinstance(value, str):
            named_attr(prim, name, Sdf.ValueTypeNames.String).Set(value)
        elif isinstance(value, dict):
            attr = named_attr(prim, name, Sdf.ValueTypeNames.Double)
            for f, v in zip(value["frames"], value["values"]):
                attr.Set(float(v), Usd.TimeCode(float(f)))
        else:
            named_attr(prim, name, Sdf.ValueTypeNames.Double).Set(float(value))


def read_properties(prim: Usd.Prim, namespace: str) -> dict:
    """write_properties back: every authored attribute whose name (as given) starts with `namespace`."""
    out: dict = {}
    for attr in prim.GetAuthoredAttributes():
        name = attr.GetCustomDataByKey(ORIGINAL) or attr.GetName()
        if not name.startswith(namespace):
            continue
        times = attr.GetTimeSamples()
        if len(times) > 1:
            out[name] = {"frames": [int(round(t)) for t in times], "values": [float(attr.Get(t)) for t in times]}
        else:
            value = attr.Get(times[0]) if times else attr.Get()
            out[name] = value if isinstance(value, str) else tuple(value) if hasattr(value, "__len__") else float(value)
    return out


def camera_lens(prim: Usd.Prim, times: list, lens: str) -> dict:
    """What write_camera keeps of a lens besides its focal length and film back, at `times`: the centre offset per
    time (mm), the pixel aspect, the overscan (pixels: left, top, right, bottom) and the named properties."""
    cam = UsdGeom.Camera(prim)
    h, v = cam.GetHorizontalApertureOffsetAttr(), cam.GetVerticalApertureOffsetAttr()
    # aperture offsets are where the picture sits off the lens axis; center_mm is the lens centre off the picture (write_camera)
    center = -np.array([[h.Get(Usd.TimeCode(t)) or 0.0, v.Get(Usd.TimeCode(t)) or 0.0] for t in times], np.float64)
    properties = read_properties(prim, lens)
    aspect = properties.pop(lens + "pixelAspect", 1.0)
    overscan = properties.pop(lens + "overscan", (0, 0, 0, 0))
    return {"center_mm": center, "pixel_aspect": float(aspect), "overscan": tuple(int(x) for x in overscan), "properties": properties}


def camera_resolution(prim: Usd.Prim) -> tuple[int, int] | None:
    res = prim.GetCustomDataByKey("lab2shot:resolution")
    return (int(res[0]), int(res[1])) if res is not None else None


# The camera's plate. In a DCC the background plate is a camera parameter, set by hand or computed by a node. This key
# records the fingerprint of the picture packet the camera was solved from (or was assigned). It travels with the prim
# and survives scene merges, camera-space conversions and pass-through. When the 3D view looks through the camera, this
# picture is its plate (undistorted with the camera's own lens); a camera without it (an imported USD camera) has no
# plate. The plate is not inferred from the node graph, since the nearest upstream picture need not be this camera's.
PLATE = "lab2shot:plate"


def camera_plate(prim: Usd.Prim) -> str:
    """The picture this camera belongs to (a packet fingerprint), "" when none is recorded."""
    plate = prim.GetCustomDataByKey(PLATE)
    return str(plate) if plate else ""


# --------------------------------------------------------------------------- units


# the lengths a gprim holds besides its transform (UsdGeom schema attributes): scaled with the stage (rescale_stage)
GPRIM_LENGTHS = {"Sphere": ("radius",), "Cube": ("size",), "Cylinder": ("radius", "height"), "Capsule": ("radius", "height"),
                 "Cone": ("radius", "height")}


def rescale_stage(stage: Usd.Stage, factor: float) -> None:
    """Multiply every length in a (flattened) stage by `factor`, e.g. 0.01 for cm -> m.

    Lens values (focal length, apertures) stay in millimeters: DCCs read them
    as mm whatever the scene unit, and only their ratio sets the field of view. A camera's focusDistance is a scene
    length and is scaled. Gprims' own sizes (Sphere / Cylinder / Capsule / Cone radius and height, Cube size) and a
    PointInstancer's positions are scaled too; its scales and protoIndices are not lengths.
    """
    def scaled_matrix(m) -> Gf.Matrix4d:
        m = Gf.Matrix4d(m)
        m.SetTranslateOnly(m.ExtractTranslation() * factor)
        return m

    def scaled_matrices(values):
        return Vt.Matrix4dArray([scaled_matrix(m) for m in values])

    def each_sample(attr, fn):
        if not attr or not attr.HasAuthoredValue():
            return
        times = attr.GetTimeSamples()
        if times:
            for t in times:
                attr.Set(fn(attr.Get(t)), Usd.TimeCode(t))
        else:
            attr.Set(fn(attr.Get()))

    vec_scale = lambda v: type(v)([x * factor for x in v])  # noqa: E731

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.PointBased):
            geom = UsdGeom.PointBased(prim)
            each_sample(geom.GetPointsAttr(), vec_scale)
            each_sample(geom.GetExtentAttr(), vec_scale)
            each_sample(geom.GetVelocitiesAttr(), vec_scale)  # lengths per second
        if prim.IsA(UsdGeom.Points):  # a point's size and a curve's thickness are lengths too
            each_sample(UsdGeom.Points(prim).GetWidthsAttr(), vec_scale)
            each_sample(prim.GetAttribute("primvars:gaussian:scales"), vec_scale)
        elif prim.IsA(UsdGeom.Curves):
            each_sample(UsdGeom.Curves(prim).GetWidthsAttr(), vec_scale)
        if prim.IsA(UsdSkel.Skeleton):
            skel = UsdSkel.Skeleton(prim)
            each_sample(skel.GetBindTransformsAttr(), scaled_matrices)
            each_sample(skel.GetRestTransformsAttr(), scaled_matrices)
        if prim.IsA(UsdSkel.Animation):
            each_sample(UsdSkel.Animation(prim).GetTranslationsAttr(), vec_scale)
        if prim.IsA(UsdSkel.BlendShape):
            each_sample(UsdSkel.BlendShape(prim).GetOffsetsAttr(), vec_scale)
        each_sample(prim.GetAttribute("primvars:skel:geomBindTransform"), scaled_matrix)
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            each_sample(cam.GetClippingRangeAttr(), lambda r: Gf.Vec2f(r[0] * factor, r[1] * factor))
            each_sample(cam.GetFocusDistanceAttr(), lambda d: d * factor)
        for name in GPRIM_LENGTHS.get(prim.GetTypeName(), ()):
            each_sample(prim.GetAttribute(name), lambda d: d * factor)
        if prim.IsA(UsdGeom.PointInstancer):
            each_sample(UsdGeom.PointInstancer(prim).GetPositionsAttr(), vec_scale)
        if prim.IsA(UsdGeom.Xformable):
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    each_sample(op.GetAttr(), lambda v: v * factor)
                elif op.GetOpType() == UsdGeom.XformOp.TypeTransform:
                    each_sample(op.GetAttr(), scaled_matrix)
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT / factor)


# --------------------------------------------------------------------------- mesh


# Subsets (parts of a mesh) are stored as USD GeomSubsets, the standard primitive for a set of a mesh's faces; Houdini
# reads them as primitive groups and Maya as face sets, so they are delivered with the data. Family name of subsets:
# subsets within a family do not overlap and need not cover the whole mesh (UsdGeom.Tokens.unrestricted). Material
# binding subsets (familyName = "materialBind") that come with a file are parts of the mesh as well and are read alike.
SUBSET_FAMILY = "part"


def write_mesh(
    stage: Usd.Stage,
    path: str,
    points: np.ndarray,
    faces: np.ndarray,
    frames: list[int] | None = None,
    uv: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
    counts: np.ndarray | None = None,
    subsets: dict[str, np.ndarray] | None = None,
) -> UsdGeom.Mesh:
    """Triangle / polygon mesh. `points` [V,3] static or [F,V,3] point cache (cm). `faces` [T,k] (k corners each),
    or with `counts` (corners per face) every face's corners one after another.

    `uv` [N,2] with `uv_faces` (same shape as `faces`) becomes the face-varying "st" primvar.

    `subsets` {subset name: the faces it holds [n] int}: each one a UsdGeom.Subset of face elements (SUBSET_FAMILY),
    such as FLAME's scalp / face / neck, a mesh's shading groups, or any other part of the mesh. A face index outside
    the mesh is a caller error and is rejected rather than written out.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    faces = np.asarray(faces, dtype=np.int32)
    counts = np.full(len(faces), faces.shape[1], np.int32) if counts is None else np.asarray(counts, np.int32)
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    if uv is not None and uv_faces is not None:
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(uv, dtype=np.float32)))
        st.SetIndices(Vt.IntArray.FromNumpy(np.asarray(uv_faces, dtype=np.int32).reshape(-1)))

    pts = np.asarray(points, dtype=np.float32)
    points_attr, extent_attr = mesh.CreatePointsAttr(), mesh.CreateExtentAttr()
    if pts.ndim == 2:
        points_attr.Set(Vt.Vec3fArray.FromNumpy(pts))
        extent_attr.Set(_extent(pts))
    else:
        for f, p in zip(frames, pts):
            points_attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p)), Usd.TimeCode(f))
            extent_attr.Set(_extent(p), Usd.TimeCode(f))
    for name, indices in (subsets or {}).items():
        write_subset(mesh, name, indices)
    return mesh


def write_subset(mesh: UsdGeom.Mesh, name: str, faces) -> UsdGeom.Subset:
    """One subset of a mesh: the faces `faces` (indices into its faces) as a UsdGeom.Subset of face elements."""
    indices = np.unique(np.asarray(faces, np.int64).reshape(-1))
    total = len(mesh.GetFaceVertexCountsAttr().Get() or [])
    if len(indices) and (indices[0] < 0 or indices[-1] >= total):
        raise ValueError(f"subset {name!r} of {mesh.GetPath()}: face indices must be within 0..{total - 1}")
    subset = UsdGeom.Subset.CreateGeomSubset(mesh, child_name(mesh.GetPrim().GetStage(), str(mesh.GetPath()), name),
                                             UsdGeom.Tokens.face, Vt.IntArray.FromNumpy(indices.astype(np.int32)),
                                             SUBSET_FAMILY, UsdGeom.Tokens.unrestricted)
    keep_original(subset.GetPrim(), name)
    return subset


def mesh_subsets(mesh: UsdGeom.Mesh) -> dict[str, np.ndarray]:
    """The subsets of a mesh: {its name: the faces it holds [n] int}, in the order the file has them. Every face subset
    counts, whichever family it belongs to (a file's material-bind subsets are parts of the mesh too); a subset of
    something other than faces (USD also allows points and edges) is not one of them."""
    out: dict[str, np.ndarray] = {}
    for prim in mesh.GetPrim().GetChildren():
        if not prim.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(prim)
        if subset.GetElementTypeAttr().Get() != UsdGeom.Tokens.face:
            continue
        out[name_of(prim)] = np.asarray(subset.GetIndicesAttr().Get() or [], np.int64).reshape(-1)
    return out


def _all_same(samples) -> bool:
    """Whether all samples are identical (compared by shape, then bytes, without copying). Empty or single-sample
    input counts as identical."""
    if not samples or len(samples) == 1:
        return True
    first = np.asarray(samples[0])
    for s in samples[1:]:
        a = np.asarray(s)
        if a.shape != first.shape or a.dtype != first.dtype or not np.array_equal(a, first):
            return False
    return True


def _write_point_samples(geom, frames: list[int], points: list[np.ndarray], colors: list[np.ndarray] | None,
                         primvars: dict[str, list[np.ndarray]] | None, own: dict) -> None:
    """One point-based prim's per-sample arrays, written once for every kind of it (a point cloud, a set of 3D curves):
    its points and their extent, its display colours, any per-point primvars, and whatever else its own kind keeps per
    sample (`own`: attribute -> (the per-sample arrays, the Vt array maker, the numpy dtype)). A single sample for
    several frames is static: written once, without time samples.

    Samples that are identical on every frame are also written as static. The check is made here, at write time, so
    every point cloud, curve set and point cache benefits without per-node handling. It is lossless: the data is
    unchanged and only stored once.

    Static and per-frame data take different paths (server/view_data.py sends static data once and per-frame data in
    chunks). A node that emits the same point cloud on every frame would otherwise be treated as per-frame and resent
    for every frame, which at a typical 30-40 Mbps uplink costs tens of seconds. Counting samples
    (`len(points) == 1`) does not catch this case; the values themselves are compared."""
    if _all_same(points) and _all_same(colors) and all(_all_same(v) for v in (primvars or {}).values()) \
            and all(_all_same(v) for v, _, _ in own.values()):
        frames, points = frames[:1], points[:1]
        colors = colors[:1] if colors else colors
        primvars = {n: v[:1] for n, v in (primvars or {}).items()} or None
        own = {a: (v[:1], make, dtype) for a, (v, make, dtype) in own.items()}
    pos_attr, ext_attr = geom.CreatePointsAttr(), geom.CreateExtentAttr()
    api = UsdGeom.PrimvarsAPI(geom)
    color_pv = api.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex) if colors else None
    extra = {n: (api.CreatePrimvar(n, _PRIMVAR_TYPES[k], UsdGeom.Tokens.vertex), k)
             for n, k in ((n, _primvar_kind(v[0])) for n, v in (primvars or {}).items())}
    times = [Usd.TimeCode.Default()] if len(points) == 1 else [Usd.TimeCode(f) for f in frames]
    for i, t in enumerate(times):
        p = np.ascontiguousarray(points[i], dtype=np.float32)
        pos_attr.Set(Vt.Vec3fArray.FromNumpy(p), t)
        if len(p):
            ext_attr.Set(_extent(p), t)
        if color_pv is not None:
            color_pv.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(colors[i], dtype=np.float32)), t)
        for attr, (values, make, dtype) in own.items():
            attr.Set(make(np.ascontiguousarray(values[i], dtype=dtype)), t)
        for n, (pv, kind) in extra.items():
            pv.Set(_PRIMVAR_ARRAYS[kind](np.ascontiguousarray(primvars[n][i], dtype=_PRIMVAR_DTYPES[kind])), t)


def write_points(
    stage: Usd.Stage,
    path: str,
    frames: list[int],
    points: list[np.ndarray],
    colors: list[np.ndarray] | None = None,
    primvars: dict[str, list[np.ndarray]] | None = None,
    width_cm: float = 0.2,
    ids: list[np.ndarray] | None = None,
    velocities: list[np.ndarray] | None = None,
) -> UsdGeom.Points:
    """Per-frame point clouds (the point count may change every frame).

    `colors`: per-frame [N,3] display colors (0..1). `primvars`: extra per-point
    data per frame, e.g. {"canonical": [...], "visible": [...]}: [N,3] floats as
    float3, [N] floats as float, [N] ints or bools as int. `ids`: per-frame [N]
    the point's identity (the same point keeps its id from frame to frame: a
    tracked point, Houdini's `id`); `velocities`: per-frame [N,3] in cm per second
    (Houdini's `v`, motion blur). A single cloud for several frames is static:
    written once, without time samples.
    """
    pts = UsdGeom.Points.Define(stage, path)
    pts.CreateWidthsAttr(Vt.FloatArray([width_cm]))
    pts.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    own = {}
    if ids is not None:
        own[pts.CreateIdsAttr()] = (ids, Vt.Int64Array.FromNumpy, np.int64)
    if velocities is not None:
        own[pts.CreateVelocitiesAttr()] = (velocities, Vt.Vec3fArray.FromNumpy, np.float32)
    _write_point_samples(pts, frames, points, colors, primvars, own)
    return pts


def write_curves(
    stage: Usd.Stage,
    path: str,
    frames: list[int],
    vertex_counts: list[np.ndarray],
    points: list[np.ndarray],
    widths: list[np.ndarray] | None = None,
    colors: list[np.ndarray] | None = None,
    normals: list[np.ndarray] | None = None,
    primvars: dict[str, list[np.ndarray]] | None = None,
    width_cm: float = 0.2,
) -> UsdGeom.BasisCurves:
    """Per-frame 3D curves (UsdGeom.BasisCurves): `vertex_counts` per frame [C] how many points each curve has,
    `points` per frame [N,3] their points one curve after another (N = the counts' sum).

    Linear, non-periodic curves: what hair, guide curves and motion trails are, and what every DCC reads back
    without guessing a basis. `widths`: one per point (hair is thick at the root and thin at the tip); a set of
    curves with one width throughout gets the constant `width_cm` instead, as a point cloud does. `colors`: per-point
    display colours (0..1); `normals`: per-point orientation; `primvars`: any other per-point data.
    A single sample for several frames is static: written once, without time samples.
    """
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    own = {curves.CreateCurveVertexCountsAttr(): (vertex_counts, Vt.IntArray.FromNumpy, np.int32)}
    width_attr = curves.CreateWidthsAttr()
    if widths is None:  # one thickness throughout, as a point cloud's size is
        width_attr.Set(Vt.FloatArray([width_cm]))
        curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    else:
        curves.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        own[width_attr] = (widths, Vt.FloatArray.FromNumpy, np.float32)
    if normals is not None:
        curves.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        own[curves.CreateNormalsAttr()] = (normals, Vt.Vec3fArray.FromNumpy, np.float32)
    _write_point_samples(curves, frames, points, colors, primvars, own)
    return curves


# per-point primvar kinds _write_point_samples writes: USD type, array maker, numpy dtype
_PRIMVAR_TYPES = {"float3": Sdf.ValueTypeNames.Float3Array, "float": Sdf.ValueTypeNames.FloatArray, "int": Sdf.ValueTypeNames.IntArray}
_PRIMVAR_ARRAYS = {"float3": Vt.Vec3fArray.FromNumpy, "float": Vt.FloatArray.FromNumpy, "int": Vt.IntArray.FromNumpy}
_PRIMVAR_DTYPES = {"float3": np.float32, "float": np.float32, "int": np.int32}


def _primvar_kind(values) -> str:
    a = np.asarray(values)
    if a.ndim == 2:
        return "float3"
    return "int" if a.dtype.kind in "biu" else "float"


def _extent(p: np.ndarray) -> Vt.Vec3fArray:
    return Vt.Vec3fArray([Gf.Vec3f(*map(float, p.min(0))), Gf.Vec3f(*map(float, p.max(0)))])


# --------------------------------------------------------------------------- skeleton


@dataclass
class SkinnedCharacter:
    """A skeleton with animation and an optional skinned mesh.

    All transforms are joint-to-world, column-vector convention, centimeters.

    `joint_names` are the bones' CG names (data/joints.py cg_names: Mixamo's set, Hips / LeftArm / LeftUpLeg …). A
    model's own names are renamed on the way in (data/skeleton.py character_of_model): ML body models name the
    joint, CG names the bone, and the two are a whole bone apart: SMPL's left_shoulder is the upper arm, its left_hip
    the thigh. What comes out of Lab2Shot carries the CG name, so an animator reads it right and HumanIK matches it.
    """

    joint_names: list[str]
    parents: np.ndarray  # [J], -1 for roots; parents precede children
    bind_world: np.ndarray  # [J,4,4] rest pose the mesh is bound in
    anim_world: np.ndarray  # [F,J,4,4] per frame
    rest_points: np.ndarray | None = None  # [V,3] mesh in bind pose
    faces: np.ndarray | None = None  # [T,3]
    joint_indices: np.ndarray | None = None  # [V,K] skin influences
    joint_weights: np.ndarray | None = None  # [V,K]
    uv: np.ndarray | None = None  # [N,2]
    uv_faces: np.ndarray | None = None  # [T,3] indices into uv
    blendshape_names: list[str] = field(default_factory=list)  # shapes added to the rest mesh before skinning
    blendshape_offsets: np.ndarray | None = None  # [K,V,3]
    blendshape_weights: np.ndarray | None = None  # [F,K] per frame
    custom_data: dict = field(default_factory=dict)



PERSON_ID = "lab2shot:person_id"  # customData on a character's SkelRoot: which person box it was solved from
# (the one place the key is written; a solver sets it, server/view_data.py sends it so the 3D view colours the
# character like that person's box in 2D)


def joint_names(skel: UsdSkel.Skeleton) -> list[str]:
    """A skeleton's joint names, one per joint in its joint order: its jointNames when there is one for every joint,
    else each joint path's last part as it is (another tool's skeleton; a jointNames of another length belongs to no
    joint order and is not used). The one reader: the editor's lists, the handles and the cook all get the same count."""
    joints = [str(j) for j in skel.GetJointsAttr().Get() or []]
    given = [str(n) for n in skel.GetJointNamesAttr().Get() or []]
    return given if len(given) == len(joints) else [j.rsplit("/", 1)[-1] for j in joints]


def joint_paths(joint_names: list[str], parents: np.ndarray) -> list[str]:
    """Each joint's UsdSkel path (its parent's path / names.identifier(name)). A path is a joint's identity: UsdSkel
    rebuilds the hierarchy from the paths, so two siblings that come out as one path (names that differ only in
    punctuation, a:b and a_b) would hang one's children under the other and Validate() would not notice: such a later
    sibling goes through names.unique. Its name in jointNames is untouched."""
    paths: list[str] = []
    taken: set[str] = set()
    for i, (name, parent) in enumerate(zip(joint_names, parents)):
        if parent >= i:
            raise ValueError("Joint parents must precede their children")
        path = names.unique(names.identifier(name) if parent < 0 else f"{paths[parent]}/{names.identifier(name)}", taken)
        taken.add(path)
        paths.append(path)
    return paths


def write_joint_samples(anim: UsdSkel.Animation, joints, local: np.ndarray, frames: list[int]) -> None:
    """A skeleton animation's joints and their local transforms [F,J,4,4] (joint-to-parent) at `frames`: translations,
    rotations (sign-continuous quaternions) and scales, one time sample per frame."""
    anim.CreateJointsAttr(Vt.TokenArray(list(joints)))
    t_attr, r_attr, s_attr = anim.CreateTranslationsAttr(), anim.CreateRotationsAttr(), anim.CreateScalesAttr()
    n_joints = local.shape[1]
    from lab2shot_shared.motion import decompose

    t, q, s = decompose(np.asarray(local, np.float64).reshape(-1, 4, 4))
    t, q, s = t.reshape(-1, n_joints, 3), continuous(q.reshape(-1, n_joints, 4)), s.reshape(-1, n_joints, 3)
    for i, f in enumerate(frames):
        time = Usd.TimeCode(f)
        t_attr.Set(Vt.Vec3fArray.FromNumpy(t[i].astype(np.float32)), time)
        r_attr.Set(Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for w, x, y, z in q[i]]), time)
        s_attr.Set(Vt.Vec3hArray.FromNumpy(s[i].astype(np.float16)), time)


@dataclass
class SkinnedMesh:
    """A mesh skinned to a skeleton (write_rig): points in the bind pose in the skeleton's space, polygons (corners per
    face), UVs, the skin, blend shapes (added to the bind points before skinning) and their weights per frame."""

    name: str
    points: np.ndarray  # [V,3]
    counts: np.ndarray  # [P] corners per face
    indices: np.ndarray  # [N] every face's corners one after another
    joint_indices: np.ndarray  # [V,K]
    joint_weights: np.ndarray  # [V,K]
    uv: np.ndarray | None = None  # [U,2]
    uv_indices: np.ndarray | None = None  # [N]
    shapes: list[str] = field(default_factory=list)
    shape_offsets: np.ndarray | None = None  # [B,V,3]
    shape_weights: np.ndarray | None = None  # [F,B] at the rig's frames
    # The model's own subsets (FLAME / SMPL scalp, face, neck, lips, nose, ears, ...): {name: its faces}. A person is
    # always output as a skeleton with a skinned mesh, never a point cache (so it can be corrected in a DCC), so the
    # subsets travel with the skinned mesh.
    subsets: dict[str, np.ndarray] | None = None


# a character prim's customData: its path is the root joint's own, not a group above it (said by whoever read it,
# never guessed from the names later)
ROOT_AT_PATH = "lab2shot:rootAtPath"


def write_rig(stage: Usd.Stage, name: str, joint_names: list[str], parents, bind_world: np.ndarray, anim_world: np.ndarray,
              frames: list[int], meshes: list[SkinnedMesh] = (), custom_data: dict | None = None, path: str = "",
              source: str = "") -> UsdSkel.Root:
    """/shot/<name> (or `path`: an imported rig where its file had it, `source`, under identity groups): a skeleton root
    with its skeleton (bind pose: joint-to-world [J,4,4]), its animation (joint-to-world [F,4,4] at `frames`, a sample
    at each) and the meshes skinned to it with their blend shapes (the animation's weights: every mesh's shapes, each
    once)."""
    root_path = path or f"{ROOT_PATH}/{names.identifier(name)}"
    place(stage, root_path)
    skel_root = UsdSkel.Root.Define(stage, root_path)
    place(stage, root_path, source)
    if not path:
        keep_original(skel_root.GetPrim(), name)
    for key, value in (custom_data or {}).items():  # key by key: the whole-dict setter would drop the original name kept above
        skel_root.GetPrim().SetCustomDataByKey(key, value)
    parents = np.asarray(parents, dtype=np.int64)
    joints = Vt.TokenArray(joint_paths(joint_names, parents))
    skel = UsdSkel.Skeleton.Define(stage, f"{root_path}/skeleton")
    skel.CreateJointsAttr(joints)
    # jointNames holds each joint's original name; USD tokens accept characters such as colons, and only the joints
    # paths need legal names. Mixamo's mixamorig:Hips keeps its colon so that HumanIK and retargeting in Maya recognize
    # the namespace.
    skel.CreateJointNamesAttr(Vt.TokenArray(names.sibling_unique(list(joint_names), parents)))
    skel.CreateBindTransformsAttr(_usd_matrices(bind_world))
    skel.CreateRestTransformsAttr(_usd_matrices(local_from_world(np.asarray(bind_world, np.float64), parents)))
    anim = UsdSkel.Animation.Define(stage, f"{root_path}/anim")
    write_joint_samples(anim, joints, local_from_world(np.asarray(anim_world, dtype=np.float64), parents), frames)
    UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    # blend shapes: the tokens (skel:blendShapes, the animation's blendShapes) are the shapes' own names, as jointNames
    # are the joints' — what 「表情重定向（ARKit52）」 reads and matches; only the BlendShape prims need identifiers. A name is one
    # animation channel: the same name on two meshes with the same weights (a face and its teeth) shares it, a name
    # taken with other weights, or twice on one mesh, goes through names.unique.
    shapes: dict[str, np.ndarray] = {}  # the animation's shape -> its weight per frame
    for m in meshes:
        mesh = write_mesh(stage, f"{root_path}/{child_name(stage, root_path, m.name)}", m.points, m.indices, uv=m.uv,
                          uv_faces=m.uv_indices, counts=m.counts, subsets=m.subsets)
        keep_original(mesh.GetPrim(), m.name)
        binding = UsdSkel.BindingAPI.Apply(mesh.GetPrim())
        binding.CreateSkeletonRel().SetTargets([skel.GetPath()])
        k = m.joint_indices.shape[1]
        binding.CreateJointIndicesPrimvar(False, k).Set(Vt.IntArray.FromNumpy(m.joint_indices.astype(np.int32).reshape(-1)))
        binding.CreateJointWeightsPrimvar(False, k).Set(Vt.FloatArray.FromNumpy(m.joint_weights.astype(np.float32).reshape(-1)))
        binding.CreateGeomBindTransformAttr(Gf.Matrix4d(1.0))
        if not m.shapes:
            continue
        tokens, targets = [], []
        for b, shape_name in enumerate(m.shapes):
            weights = np.asarray(m.shape_weights[:, b], np.float32)
            token = names.unique(str(shape_name), {*tokens, *(t for t, w in shapes.items() if not np.array_equal(w, weights))})
            shapes[token] = weights
            shape = UsdSkel.BlendShape.Define(stage, f"{mesh.GetPath()}/{child_name(stage, str(mesh.GetPath()), token)}")
            keep_original(shape.GetPrim(), token)
            shape.CreateOffsetsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.shape_offsets[b], np.float32)))
            tokens.append(token)
            targets.append(shape.GetPath())
        binding.CreateBlendShapesAttr(Vt.TokenArray(tokens))
        binding.CreateBlendShapeTargetsRel().SetTargets(targets)
    if shapes:
        anim.CreateBlendShapesAttr(Vt.TokenArray(list(shapes)))
        w_attr = anim.CreateBlendShapeWeightsAttr()
        weights = np.stack(list(shapes.values()), -1)
        for i, f in enumerate(frames):
            w_attr.Set(Vt.FloatArray.FromNumpy(weights[i]), Usd.TimeCode(f))
    return skel_root


def write_character(stage: Usd.Stage, name: str, ch: SkinnedCharacter, frames: list[int],
                    subsets: dict[str, np.ndarray] | None = None, shot: list[int] | None = None) -> UsdSkel.Root:
    """A solver's character: its skeleton and animation, and its one mesh skinned to it.

    `subsets`: the model's own subsets (FLAME / SMPL scalp, face, neck, ...), carried by the skinned character.
    `shot`: the frames of the whole shot; when given, the person is invisible on frames outside `frames` (the solved
    frames). Otherwise USD would interpolate between two solved ranges and an undetected person would drift across
    the gap. This is implemented only here and every worker that writes people relies on it."""
    faces = None if ch.faces is None else np.asarray(ch.faces)
    skinned = [] if faces is None or ch.rest_points is None else [SkinnedMesh(
        "body", ch.rest_points, np.full(len(faces), faces.shape[1]), faces.reshape(-1), ch.joint_indices, ch.joint_weights,
        ch.uv, None if ch.uv_faces is None else np.asarray(ch.uv_faces).reshape(-1), list(ch.blendshape_names),
        ch.blendshape_offsets, None if ch.blendshape_weights is None else np.asarray(ch.blendshape_weights),
        subsets=subsets)]
    root = write_rig(stage, name, ch.joint_names, ch.parents, ch.bind_world, ch.anim_world, frames, skinned, ch.custom_data)
    if shot is not None and set(frames) != set(shot):
        own = set(frames)
        set_visible(root.GetPrim(), list(shot), [f in own for f in shot])
    return root


def set_visible(prim: Usd.Prim, frames: list[int], visible) -> None:
    """The prim hidden on the frames where `visible` is false, a sample on each frame (held: USD does not blend a
    token). The one writer of visibility: a solver's person (write_character) and a file's item read from the scene
    arrays' `visible` (data/scene_arrays.py)."""
    vis = UsdGeom.Imageable(prim).CreateVisibilityAttr()
    for f, shown in zip(frames, visible):
        vis.Set("inherited" if shown else "invisible", Usd.TimeCode(int(f)))


def visible_at(prim: Usd.Prim, frames: list[int]) -> np.ndarray:
    """Whether the prim is shown on each frame (its own visibility and its ancestors'): what the scene arrays carry as
    `visible` for writers of other formats."""
    img = UsdGeom.Imageable(prim)
    return np.array([img.ComputeVisibility(Usd.TimeCode(int(f))) != UsdGeom.Tokens.invisible for f in frames])


def deformed_points(target, skel_query, time: Usd.TimeCode):
    """A skinning target's points at `time` the way a DCC deforms them: blend shapes first (weights from the
    skeleton's animation), then linear blend skinning. Skeleton space; None when skinning fails."""
    points = UsdGeom.Mesh(target.GetPrim()).GetPointsAttr().Get(time)
    if target.HasBlendShapes():
        weights = skel_query.GetAnimQuery().ComputeBlendShapeWeights(time)
        mapped = target.GetBlendShapeMapper().Remap(weights) if target.GetBlendShapeMapper() else weights
        shapes = UsdSkel.BlendShapeQuery(UsdSkel.BindingAPI(target.GetPrim()))
        sub_weights, shape_idx, sub_idx = shapes.ComputeSubShapeWeights(mapped)
        shapes.ComputeDeformedPoints(sub_weights, shape_idx, sub_idx, shapes.ComputeBlendShapePointIndices(),
                                     shapes.ComputeSubShapePointOffsets(), points)
    if not target.ComputeSkinnedPoints(skel_query.ComputeSkinningTransforms(time), points, time):
        return None
    return points
