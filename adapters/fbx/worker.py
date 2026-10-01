"""FBX worker: runs inside third_party/fbx/.venv (fbxio, the pybind11 module build.py compiles against the Autodesk
FBX SDK); never imports Lab2Shot core.

    python worker.py <job.json>

job["node"]:
    fbx.import   inputs.file (.fbx) -> raw/scene.npz: every camera, model and character of the file as scene arrays
                 (lab2shot_shared/scene_arrays.py), in the file's unit and axes (top level unit_cm and axes say them)
    fbx.output   inputs.scene (scene arrays: cm, Y up) -> params.file (.fbx: cm, Maya Y-up axes written explicitly)

What an FBX file holds, as the scene arrays name it:
    camera      a node with a camera. FBX cameras aim down their node's +X axis with +Y up (the SDK's convention; one
                with a look-at target is turned so that axis points at it);
                ours look down -Z with +Y up, so our camera matrix is the node's global matrix turned -90 degrees about
                its Y axis (and the writer turns ours +90 degrees). The lens is the focal length at each frame (or
                the one a field of view gives through the film back), the film back in mm.
    model       a node with a mesh that is not skinned. Its world is the node's global matrix times its geometric
                transform (which moves the mesh, not the node's children); its shape changes only through blend
                shapes whose weights are animated (then it has points per frame).
    character   a skeleton: an FBX skeleton node whose parent is not one, with every skeleton node under it, and the
                meshes skinned to its joints. Joints bind where their clusters say (TransformLinkMatrix), else where
                the bind pose has them, else where their parent's bind and their own rest place put them; the bound
                meshes' points are placed by their cluster's TransformMatrix (and geometric transform), their blend
                shapes turned with them, as the SDK's own skinning does it.
    points      FBX has no point clouds: the reader finds none, the writer refuses them.

Item paths are the node names from the top of the scene, as DCC outliners show them (/World/set/table); a character's
is its root joint's. An item's frames are every frame from the first key to the last over the node, its parents and
what drives its attribute (a camera's lens, blend-shape weights, a character's joints); without keys, one frame: the
start of the file's time span.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

import fbxio
from lab2shot_worker import fail, progress, say, serve, shown
from lab2shot_worker.run import Run
from lab2shot_shared.motion import orthonormal
from lab2shot_shared.names import join_path, split_path
from lab2shot_shared.scene_arrays import SceneArrays, load, shown_span, text, texts
from lab2shot_shared.units import standard_fps

API = 8  # the fbxio binding this worker is written for (fbxio.cpp m.attr("API")): takes(), a camera's lens centre, squeeze and user properties, an upload opened as FBX only, writing nothing, the file's texts with bytes that are not UTF-8 replaced, visibility keys, any frame rate written as itself, and Visibility keys among a node's key frames
OVERSCAN = "lab2shot:lens:overscan"  # FBX has no overscan: a user property, parts of the picture (left, top, right, bottom) as JSON


def check_build() -> None:
    """A fbxio built from an older fbxio.cpp (the extension's code updated, its environment not rebuilt) says so plainly
    before anything is read or written, never an AttributeError halfway through a job."""
    built = getattr(fbxio, "API", 1)
    if built < API:
        fail("E-FBX-REBUILD", built=built, needed=API)


# our camera (looks down -Z, up +Y) = FBX camera node (aims down +X, up +Y) turned -90 degrees about Y
FBX_TO_CAMERA = np.array([[0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
CAMERA_TO_FBX = np.linalg.inv(FBX_TO_CAMERA)
AXES = "xyz"
FRONT = {("y", "even"): "x", ("y", "odd"): "z", ("z", "even"): "x", ("z", "odd"): "y", ("x", "even"): "y", ("x", "odd"): "z"}


# ------------------------------------------------------------------ reading


def axes_of(info: dict) -> np.ndarray:
    """The 3x3 turning the file's axes into Y up, right-handed, front +Z (Maya's): rows are the file's right, up and
    front directions. A left-handed file can't be turned (it would need a mirror)."""
    if not info["right_handed"]:
        fail("E-FBX-LEFTHANDED")
    e = np.eye(3)
    up = info["up_sign"] * e[AXES.index(info["up"])]
    front = info["front_sign"] * e[AXES.index(FRONT[(info["up"], info["front_parity"])])]
    return np.stack([np.cross(up, front), up, front])


def open_file(path: Path):
    if not path.is_file():
        fail("E-FBX-NOFILE", path=shown(path))
    try:
        return fbxio.open(str(path))
    except (RuntimeError, UnicodeDecodeError):  # the SDK's message may quote the file's bytes (fbxio.cpp utf8() makes them UTF-8)
        fail("E-FBX-NOTFBX", name=path.name)


class Reader:
    """The nodes of an open FBX scene, with what the items need of them."""

    def __init__(self, scene):
        self.scene = scene
        self.info = scene.info()
        self.nodes = scene.nodes()
        self.start = round(self.info["start"])
        self.keys = {n["id"]: scene.key_frames(n["id"]) for n in self.nodes}
        self.names: list[str] = [n["name"] for n in self.nodes]  # every name of the file read (replaced_names)

    def path(self, i: int) -> str:
        names = []
        while i >= 0:
            names.append(self.nodes[i]["name"])
            i = self.nodes[i]["parent"]
        return join_path(reversed(names))  # a name with "/" in it stays one name (lab2shot_shared/names.py)

    def ancestors(self, i: int) -> list[int]:
        out = []
        while i >= 0:
            out.append(i)
            i = self.nodes[i]["parent"]
        return out

    def frames(self, ids) -> np.ndarray:
        """Every frame from the first key to the last over these nodes and their parents; none: the start."""
        keys = [k for i in {a for n in ids for a in self.ancestors(n)} for k in self.keys[i]]
        if not keys:
            return np.array([self.start], np.int64)
        first, last = math.ceil(min(keys) - 1e-6), math.floor(max(keys) + 1e-6)
        return np.arange(first, max(first, last) + 1, dtype=np.int64)

    def shown(self, i: int, frames) -> np.ndarray:
        """Whether node `i` is shown on each frame: its Visibility and every node's above it."""
        return np.all([self.scene.visibility(a, np.asarray(frames, np.float64)) > 0.5 for a in self.ancestors(i)], axis=0)

    def visible(self, i: int, frames) -> dict:
        """The item's `visible` on its frames when it is hidden on some, else nothing: one read for every kind, as
        write_scene's hide is one write."""
        shown = self.shown(i, frames)
        return {} if shown.all() else {"visible": shown.astype(np.int32)}

    def world(self, i: int, frames, aim: bool = False) -> np.ndarray:
        return self.scene.global_matrices(i, np.asarray(frames, np.float64), aim)


def mesh_arrays(m: dict) -> dict:
    out = {"counts": np.asarray(m["counts"], np.int32), "indices": np.asarray(m["indices"], np.int32)}
    if m["uv"] is not None:
        out.update(uv=np.asarray(m["uv"], np.float32), uv_indices=np.asarray(m["uv_indices"], np.int32))
    if m["normals"] is not None:
        out["normals"] = np.asarray(m["normals"], np.float32)
    return out


def shaped(base: np.ndarray, shapes: list[dict], weights: np.ndarray) -> np.ndarray:
    """Points per frame [F,V,3]: the base plus every blend shape's full target offset times its weight."""
    offsets = np.stack([np.asarray(s["points"])[:, :3] - base for s in shapes])
    return base[None] + np.einsum("fb,bvk->fvk", weights, offsets)


def read_cameras(r: Reader, out: SceneArrays) -> int:
    ids = [n["id"] for n in r.nodes if n["kind"] == "camera"]
    for i in ids:
        frames = r.frames([i])
        lens = r.scene.camera(i, frames.astype(np.float64))
        n = len(frames)
        extra = {}
        if lens["aspect_mode"] == "fixed_resolution" and lens["aspect_width"] > 0 and lens["aspect_height"] > 0:
            extra["resolution"] = np.array([round(lens["aspect_width"]), round(lens["aspect_height"])], np.int64)
        props = dict(lens["properties"])
        r.names += props
        overscan = json.loads(props.pop(OVERSCAN, "[0, 0, 0, 0]"))
        out.add("camera", r.nodes[i]["name"], r.path(i), frames, r.world(i, frames, aim=True) @ FBX_TO_CAMERA,
                focal_mm=np.asarray(lens["focal_mm"], np.float64), h_aperture_mm=np.full(n, lens["film_width_mm"]),
                v_aperture_mm=np.full(n, lens["film_height_mm"]), center_mm=np.asarray(lens["center_mm"], np.float64),
                pixel_aspect=np.float64(lens["pixel_aspect"]), overscan=np.asarray(overscan, np.float64),
                properties=np.array(json.dumps(props, ensure_ascii=False)), **extra, **r.visible(i, frames))
    return len(ids)


def read_models(r: Reader, out: SceneArrays, skinned: set[int]) -> int:
    ids = [n["id"] for n in r.nodes if n["kind"] == "mesh" and n["id"] not in skinned]
    for k, i in enumerate(ids):
        frames = r.frames([i])
        m = r.scene.mesh(i)
        base = np.asarray(m["points"], np.float64)
        shapes = r.scene.blend_shapes(i)
        r.names += [s["name"] for s in shapes]
        points = base[None]
        if shapes:  # a mesh blend shapes move without a skeleton: its points per frame
            weights = np.asarray(r.scene.blend_weights(i, frames.astype(np.float64)))
            if len(frames) > 1 and np.ptp(weights, axis=0).max(initial=0.0) > 0:
                points = shaped(base, shapes, weights)
            elif weights.size:
                points = shaped(base, shapes, weights[:1])
        world = r.world(i, frames) @ r.scene.geometric(i)
        out.add("model", r.nodes[i]["name"], r.path(i), frames, world, points=points.astype(np.float32), **mesh_arrays(m),
                **r.visible(i, frames))
        progress(k + 1, len(ids), "读取模型")
    return len(ids)


def read_characters(r: Reader, out: SceneArrays) -> tuple[int, set[int]]:
    """Every skeleton with the meshes skinned to it. Returns (how many, the ids of the meshes they took)."""
    joint = {n["id"] for n in r.nodes if n["kind"] == "skeleton"}
    # a joint's parent joint is the nearest joint above it, through any null between (a group between Hips and Spine
    # keeps one skeleton: its transform is in the joints' world matrices already); a root has no joint above it
    above = {i: next((a for a in r.ancestors(i)[1:] if a in joint), -1) for i in joint}
    roots = [i for i in sorted(joint) if above[i] < 0]
    member = {}  # joint id -> (character, index in it)
    rigs = []
    for c, root in enumerate(roots):
        ids = [i for i in sorted(joint) if root in r.ancestors(i)]  # depth-first order: parents first
        member.update({i: (c, k) for k, i in enumerate(ids)})
        rigs.append({"root": root, "ids": ids, "meshes": []})
    skins = {}
    for n in r.nodes:
        if n["kind"] != "mesh":
            continue
        skin = r.scene.skin(n["id"])
        links = [cl["link"] for cl in skin["clusters"]] if skin else []
        owners = [member[j][0] for j in links if j in member]
        if links and not owners:  # skinned to nothing but nulls (3ds Max's Dummy / Point as bones): never a still model
            fail("E-FBX-NOSKELETON", path=r.path(n["id"]), names=[r.nodes[j]["name"] for j in links][:5])
        if not owners:
            continue
        if len(set(owners)) > 1:
            say("W-FBX-SEVERALRIGS", path=r.path(n["id"]))
        if skin["type"] not in ("linear", "rigid"):
            say("W-FBX-SKINTYPE", path=r.path(n["id"]), skin=skin["type"])
        skins[n["id"]] = skin
        rigs[max(set(owners), key=owners.count)]["meshes"].append(n["id"])  # the rig most of its clusters bind to
    pose = r.scene.bind_pose()
    for c, rig in enumerate(rigs):
        ids = rig["ids"]
        index = {i: k for k, i in enumerate(ids)}
        parents = np.array([index.get(above[i], -1) for i in ids], np.int64)
        frames = r.frames(ids + rig["meshes"])
        anim = np.stack([r.world(i, frames) for i in ids], 1)  # [T,J,4,4]
        bind = [None] * len(ids)
        for mesh in rig["meshes"]:
            for cl in skins[mesh]["clusters"]:
                k = index.get(cl["link"])
                if k is None:
                    continue
                link = np.asarray(cl["transform_link"])
                if bind[k] is not None and np.abs(bind[k] - link).max() > 1e-4:
                    say("W-FBX-BINDPOSE", path=r.path(ids[k]))
                elif bind[k] is None:
                    bind[k] = link
        for k, i in enumerate(ids):
            if bind[k] is None and i in pose:
                bind[k] = np.asarray(pose[i])
            if bind[k] is None:  # no cluster, no bind pose: its parent's bind and its own rest place (at the start)
                g = r.world(i, [r.start])[0]
                if parents[k] >= 0:
                    parent_now = r.world(ids[parents[k]], [r.start])[0]
                    g = bind[parents[k]] @ np.linalg.inv(parent_now) @ g
                bind[k] = g
        bind = np.stack(bind)
        # an FBX character's path is its root joint's, not a group of its own: said here, never guessed later.
        # Shown where its root joint (and what is above it) is and any of its meshes is: one mesh hidden (a LOD, a
        # proxy) hides that mesh alone, its own `visible`; all of them hidden (a delivered solve's gaps) hides it
        meshes_shown = {m: r.shown(m, frames) for m in rig["meshes"]}
        shown = r.shown(rig["root"], frames) & (np.any(list(meshes_shown.values()), axis=0) if meshes_shown else True)
        item = out.add("character", r.nodes[rig["root"]]["name"], r.path(rig["root"]), frames, joints=np.array([r.nodes[i]["name"] for i in ids]),
                       parents=parents, bind=bind, anim=anim, root_at_path=True,
                       **({} if shown.all() else {"visible": shown.astype(np.int32)}))
        for mesh in rig["meshes"]:
            m = r.scene.mesh(mesh)
            clusters = [cl for cl in skins[mesh]["clusters"] if cl["link"] in index]
            place = np.asarray(clusters[0]["transform"]) @ r.scene.geometric(mesh)
            base = np.asarray(m["points"], np.float64)
            points = base @ place[:3, :3].T + place[:3, 3]
            influences = [[] for _ in range(len(base))]
            for cl in clusters:
                for v, w in zip(np.asarray(cl["indices"]), np.asarray(cl["weights"])):
                    if w > 0:
                        influences[int(v)].append((index[cl["link"]], float(w)))
            # what the clusters bound to no joint of this rig held is shared out among the rest; a vertex bound to
            # nothing else follows the joint nearest it in the bind pose (never skinned to the origin)
            orphans = 0
            for v, inf in enumerate(influences):
                total = sum(w for _, w in inf)
                if total > 0:
                    inf[:] = [(j, w / total) for j, w in inf]
                else:
                    inf[:] = [(int(np.argmin(np.linalg.norm(bind[:, :3, 3] - points[v], axis=1))), 1.0)]
                    orphans += 1
            if loose := [r.nodes[cl["link"]]["name"] for cl in skins[mesh]["clusters"] if cl["link"] not in member]:
                say("W-FBX-CLUSTER", path=r.path(mesh), count=len(loose), names=loose[:5], orphans=orphans)
            width = max(1, max(len(x) for x in influences))
            joint_idx = np.zeros((len(base), width), np.int32)
            joint_w = np.zeros((len(base), width), np.float32)
            for v, inf in enumerate(influences):
                for s, (j, w) in enumerate(inf):
                    joint_idx[v, s], joint_w[v, s] = j, w
            arrays = {**mesh_arrays(m), "points": points.astype(np.float32), "joint_indices": joint_idx, "joint_weights": joint_w}
            shapes = r.scene.blend_shapes(mesh)
            r.names += [s["name"] for s in shapes]
            if shapes:
                offsets = np.stack([np.asarray(s["points"])[:, :3] - base for s in shapes]) @ place[:3, :3].T
                arrays.update(shapes=np.array([s["name"] for s in shapes]), shape_offsets=offsets.astype(np.float32),
                              shape_weights=np.asarray(r.scene.blend_weights(mesh, frames.astype(np.float64)), np.float32))
            if "normals" in arrays:  # face-varying normals turned into world with the bind placement
                n3 = arrays["normals"].astype(np.float64) @ np.linalg.inv(place[:3, :3])
                arrays["normals"] = (n3 / np.maximum(np.linalg.norm(n3, axis=1, keepdims=True), 1e-12)).astype(np.float32)
            if not (meshes_shown[mesh] == shown).all():  # hidden apart from the character (a LOD, a proxy)
                arrays["visible"] = meshes_shown[mesh].astype(np.int32)
            out.add_mesh(item, r.nodes[mesh]["name"], **arrays)
        progress(c + 1, len(rigs), "读取骨架动画")
    return len(rigs), set(skins)


def choose_take(scene, wanted: str) -> tuple[list[str], str, str]:
    """The take the file is read at: the one the node names, else the longest take that has keys (a
    Mixamo download's first take, 「Take 001」, is empty); a file whose takes hold no keys is read at its first. Several
    animated takes and none named: said once, naming the one read. The scene reads the chosen take from then on
    (fbxio set_take). Returns (every take's name, the one an empty 「动画段」 reads, the one read)."""
    takes = scene.takes()
    names = [t["name"] for t in takes]
    if not takes:
        return names, "", ""
    animated = [t for t in takes if t["keys"] > 0]
    default = max(animated, key=lambda t: t["stop"] - t["start"])["name"] if animated else names[0]
    if wanted and wanted not in names:
        fail("E-FBX-NOTAKE", take=wanted, takes=names)
    chosen = wanted or default
    if not wanted and len(animated) > 1:
        say("N-FBX-TAKES", take=chosen, takes=[t["name"] for t in animated])
    scene.set_take(chosen)
    return names, default, chosen


SHOWN_NAMES = 5  # replaced names listed in W-FBX-NAMEENCODING, then how many


def replaced_names(file: str, names: list[str]) -> None:
    """Said once (W-FBX-NAMEENCODING): the file's node, take, blend-shape and user-property names that were not UTF-8
    and reached here with U+FFFD in place of their bytes (fbxio.cpp text()). They are the items' paths, the 「动画段」
    choices, the shapes' and properties' names, so the replacement is not left unsaid."""
    bad = list(dict.fromkeys(n for n in names if "\ufffd" in n))
    if bad:
        say("W-FBX-NAMEENCODING", name=file, count=len(bad), names=bad[:SHOWN_NAMES])


def read_fbx(run: Run) -> None:
    job = run.job
    run.stage("读取 FBX")
    scene = open_file(job.inputs["file"])
    names, default, _ = choose_take(scene, job.params.get("take") or "")
    r = Reader(scene)
    out = SceneArrays(r.info["fps"], r.info["unit_cm"], axes_of(r.info))
    if names:  # what 「动画段」 may be, and what it reads left empty (the core lists them: nodes/formats.py Listing)
        out.option("take", names, default)
    characters, skinned = read_characters(r, out)
    models = read_models(r, out, skinned)
    cameras = read_cameras(r, out)
    replaced_names(job.inputs["file"].name, r.names + names)  # once, every name read
    out.save(job.raw_dir / "scene.npz")
    frames = sorted({int(f) for items in out.items.values() for item in items for f in item["frames"]})
    run.finish(frames, kind="scene", cameras=cameras, models=models, characters=characters, fps=r.info["fps"],
               unit_cm=r.info["unit_cm"], creator=r.info["creator"])


# ------------------------------------------------------------------ writing


def node_name(name: str) -> str:
    """A node's name as it was: FBX takes any text and repeats among siblings (Mixamo's mixamorig:Hips, whose colon Maya's
    HumanIK reads as a namespace; 左足; two props both named table), so the original is the name itself — what USD and
    Alembic keep beside the identifier they need (customData / the user property lab2shot:name). Reading it back,
    repeated siblings are told apart by their keys (lab2shot_shared/scene_arrays.py load: table, table_2), their names
    kept. Only an empty name gets one."""
    return name or "node"


def write_scene(arrays: Path, out: Path) -> dict:
    top, items = load(arrays)
    if items["points"]:
        fail("E-FBX-NOPOINTS")
    if abs(float(top.get("unit_cm", 1.0)) - 1.0) > 1e-9 or ("axes" in top and not np.allclose(top["axes"], np.eye(3))):
        fail("E-FBX-NOTCMYUP", unit_cm=float(top.get("unit_cm", 1.0)))
    fps = standard_fps(float(top["fps"]))  # a typed 23.976 is NTSC's 24000/1001, which has a time mode of its own
    frames = [int(f) for kind in items.values() for it in kind for f in np.asarray(it["frames"]).reshape(-1)]
    start, stop = (min(frames), max(frames)) if frames else (1, 1)
    scene = fbxio.create(fps, float(start), float(stop))
    counts = {"models": 0, "cameras": 0, "characters": 0}
    groups: dict[str, int] = {}

    def under(path: str) -> tuple[int, str]:
        """The node an item goes under and its own name: its path's groups below our /shot (an import's folder and
        the hierarchy its file had: plain nodes with no transform, made once each), then its last name. Its matrix is
        its world one, so under groups that carry nothing it is its local one too."""
        parts = split_path(path)
        if parts and parts[0] == "shot":
            parts = parts[1:]
        parent, key = -1, ""
        for segment in parts[:-1]:
            key = f"{key}/{segment}"
            if key not in groups:
                groups[key] = scene.add_node(parent, node_name(segment))
                scene.set_transform(groups[key], np.array([float(start)]), np.eye(4)[None])
            parent = groups[key]
        return parent, (parts[-1] if parts else "node")

    def hide(nodes: list[int], it: dict) -> None:
        """Visibility keys on `nodes` where the item is hidden (shown_span: its `visible` and its gaps; stepped, FBX
        never blends them), the one write for every kind."""
        span, shown = shown_span(it)
        if not shown.all():
            for node in nodes:
                scene.set_visibility(node, span.astype(np.float64), shown.astype(np.float64))

    for it in items["model"]:
        pts = np.asarray(it["points"], np.float64)
        if pts.ndim == 3 and len(pts) > 1:
            fail("E-FBX-POINTCACHE", path=text(it["path"]))
        parent, leaf = under(text(it["shown"]))
        node = scene.add_node(parent, node_name(leaf))
        scene.set_transform(node, np.asarray(it["frames"], np.float64), np.asarray(it["world"], np.float64))
        scene.set_mesh(node, pts.reshape(-1, 3), np.asarray(it["counts"], np.int64), np.asarray(it["indices"], np.int64),
                       it.get("uv"), it.get("uv_indices"), it.get("normals"))
        hide([node], it)
        counts["models"] += 1

    for it in items["camera"]:
        f = np.asarray(it["frames"], np.float64)
        parent, leaf = under(text(it["shown"]))
        node = scene.add_node(parent, node_name(leaf))
        world = np.asarray(it["world"], np.float64).reshape(-1, 4, 4)
        world[:, :3, :3] = orthonormal(world[:, :3, :3])  # a camera sees no scale
        scene.set_transform(node, f, world @ CAMERA_TO_FBX)
        focal = np.asarray(it["focal_mm"], np.float64).reshape(-1)
        res = np.asarray(it["resolution"]).reshape(-1) if "resolution" in it else np.zeros(2)
        props = json.loads(text(it["properties"]))
        overscan = np.asarray(it["overscan"], np.float64).reshape(4)
        if overscan.any():
            props[OVERSCAN] = json.dumps(overscan.tolist())
        scene.set_camera(node, f, focal if np.ptp(focal) > 0 else focal[:1], float(np.median(it["h_aperture_mm"])),
                         float(np.median(it["v_aperture_mm"])), int(res[0]), int(res[1]),
                         np.asarray(it["center_mm"], np.float64).reshape(-1, 2), float(it["pixel_aspect"]), props)
        hide([node], it)
        counts["cameras"] += 1

    for it in items["character"]:
        f = np.asarray(it["frames"], np.float64)
        holder, leaf = under(text(it["shown"]))
        joints, parents = texts(it["joints"]), np.asarray(it["parents"], np.int64).reshape(-1)
        anim, bind = np.asarray(it["anim"], np.float64).reshape(len(f), -1, 4, 4), np.asarray(it["bind"], np.float64)
        # the reader said whether this path ends at the root joint or at a group of its own (root_at_path): the root
        # joint takes that place itself when the path is its, so nothing of its name is made twice. Guessing it from
        # the names would get a group named like its root joint wrong (a file read then written would come back
        # .../hips/hips)
        root_here = bool(joints) and bool(np.asarray(it.get("root_at_path", False)).reshape(-1)[0])
        group = holder if root_here else scene.add_node(holder, node_name(leaf))
        if not root_here:
            scene.set_transform(group, f[:1], np.eye(4)[None])
        ids = []
        for j, (name, p) in enumerate(zip(joints, parents)):
            if p >= j:
                fail("E-FBX-JOINTORDER", path=text(it["path"]), joint=name)
            at = (holder if root_here else group) if p < 0 else ids[p]
            ids.append(scene.add_node(at, node_name(name)))
            local = anim[:, j] if p < 0 else np.linalg.inv(anim[:, p]) @ anim[:, j]
            moving = len(f) > 1 and np.abs(local - local[:1]).max() > 1e-9
            scene.set_transform(ids[-1], f if moving else f[:1], local if moving else local[:1])
            scene.set_skeleton(ids[-1], p < 0, 1.0)
        pose_ids, pose_mats = list(ids), list(bind)
        for mesh in it["meshes"]:
            node = scene.add_node(group, node_name(text(mesh["name"])))
            scene.set_transform(node, f[:1], np.eye(4)[None])
            points = np.asarray(mesh["points"], np.float64)
            scene.set_mesh(node, points, np.asarray(mesh["counts"], np.int64), np.asarray(mesh["indices"], np.int64),
                           mesh.get("uv"), mesh.get("uv_indices"), mesh.get("normals"))
            idx, wts = np.asarray(mesh["joint_indices"]), np.asarray(mesh["joint_weights"])
            clusters = []
            for j in range(len(joints)):
                hit = (idx == j) & (wts > 0)
                if hit.any():
                    v, s = np.nonzero(hit)
                    clusters.append((ids[j], v.astype(np.int64), wts[v, s].astype(np.float64), np.eye(4), bind[j]))
            scene.add_skin(node, clusters)
            for b, name in enumerate(texts(mesh["shapes"]) if "shapes" in mesh else []):
                w = np.asarray(mesh["shape_weights"], np.float64).reshape(len(f), -1)[:, b]
                keyed = len(f) > 1 and np.ptp(w) > 0
                scene.add_blend_shape(node, name, points + np.asarray(mesh["shape_offsets"])[b], f if keyed else f[:1], w if keyed else w[:1])
            pose_ids.append(node)
            pose_mats.append(np.eye(4))
        scene.add_bind_pose(pose_ids, np.stack(pose_mats))
        # keyed on its own group, its root joints and its meshes, so a DCC hides the skin as well as the bones
        hide(([] if root_here else [group]) + [ids[j] for j, p in enumerate(parents) if p < 0], it)
        for node, mesh in zip(pose_ids[len(ids):], it["meshes"]):  # a mesh hidden on its own (a LOD) and with it
            own = np.asarray(mesh.get("visible", np.ones(len(f))), np.int32).reshape(-1)
            hide([node], {"frames": it["frames"], "visible": own * np.asarray(it.get("visible", np.ones(len(f))), np.int32).reshape(-1)})
        counts["characters"] += 1

    tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp.fbx")
    try:
        scene.save(str(tmp))
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    return counts


def write_fbx(run: Run) -> None:
    job = run.job
    out = Path(job.params["file"])
    if out.suffix.lower() != ".fbx":
        fail("E-FBX-SUFFIX", path=shown(out))
    run.stage("写出 FBX")
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = write_scene(job.inputs["scene"], out)
    run.finish([], kind="fbx", path=str(out), **counts)


# ------------------------------------------------------------------ main


NODES = {"fbx.import": read_fbx, "fbx.output": write_fbx}


def main(job_path: str) -> None:
    run = Run.start(job_path, tuple(NODES), "FBX", gpu=False)  # the FBX SDK: no GPU
    check_build()  # an fbxio older than this worker says so before anything of the job is read or written
    NODES[run.job.node or next(iter(NODES))](run)  # check_node's own default (Run.start checked it)


if __name__ == "__main__":
    serve(main)
