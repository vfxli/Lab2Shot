"""Reading a bound object out of the user's scene — only reading: OpenMaya function sets and getAttr, nothing that
writes, connects, renames, re-parents, selects or evaluates into anything (getAttr with a time evaluates in a context
of its own and leaves the scene as it was). What it reads is plain data (JSON); export_worker.py rebuilds the object
from it in a mayapy of its own and writes the file there.

Why not export the user's object directly: every way Maya has of writing a file from the scene writes to the scene
on the way — the FBX exporter takes a skeleton to its bind pose and back and adds blend-shape target folders, and
file -exportSelected rewrites an animation curve's tangentType (with its channels) or breaks and remakes the
channels' connections (without them). tests/test_maya.py watches for exactly that.

The data (one dict per bound object):
  camera    {"lens": {attribute: value}, "samples": [[frame, worldMatrix(16), focalLength], ...]}
  skeleton  {"joints": [joint], "samples": {joint path: [[frame, translate, rotate, scale] | [frame, worldMatrix]]}}
  character {"joints": [joint], "bind": {joint path: worldMatrix at bind}, "meshes": [mesh]}
  model     {"meshes": [mesh]}
  joint     {path, name, parent (path or ""), ro, jo, ra, ssc, local: [t, r, s], world}
  mesh      {name, world, points, counts, connects, u, v, uv_counts, uv_ids, skin?: {influences, method, normalize,
             weights: [[vertex, influence, weight], ...]}, blendshapes: [{name, targets: [{index, name, weight,
             items: [{item, points, components}]}]}]}
"""

from __future__ import annotations

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as cmds


def _dag(name: str) -> om.MDagPath:
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDagPath(0)


def _obj(name: str) -> om.MObject:
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def short(name: str) -> str:
    return name.split("|")[-1].split(":")[-1]


def _matrix(name: str, time=None) -> list[float]:
    """The world matrix now (the DAG path's, from the API), or at a time: the product of each level's own .matrix at
    that time. Never worldMatrix[0] by plug: asking an element of that output array that was never asked for makes
    Maya add it to the user's node (an array element added: a write, tests/test_maya.py sees it)."""
    if time is None:
        return list(_dag(name).inclusiveMatrix())
    m = om.MMatrix()
    path = cmds.ls(name, long=True)[0]
    while path:
        m = m * om.MMatrix(cmds.getAttr(path + ".matrix", time=time))
        path = (cmds.listRelatives(path, parent=True, fullPath=True) or [""])[0]
    return list(m)


def _v3(node: str, attr: str, time=None) -> list[float]:
    got = cmds.getAttr(f"{node}.{attr}", time=time) if time is not None else cmds.getAttr(f"{node}.{attr}")
    return list(got[0])


def joints(root: str) -> list[dict]:
    """The joints from `root` down, parents first."""
    found = [cmds.ls(root, long=True)[0]] + list(reversed(cmds.listRelatives(root, allDescendents=True, fullPath=True, type="joint") or []))
    inside = set(found)
    out = []
    for j in found:
        parent = (cmds.listRelatives(j, parent=True, fullPath=True) or [""])[0]
        out.append({"path": j, "name": short(j), "parent": parent if parent in inside else "",
                    "ro": cmds.getAttr(j + ".rotateOrder"), "jo": _v3(j, "jointOrient"), "ra": _v3(j, "rotateAxis"),
                    "ssc": bool(cmds.getAttr(j + ".segmentScaleCompensate")),
                    "local": [_v3(j, "translate"), _v3(j, "rotate"), _v3(j, "scale")], "world": _matrix(j)})
    return out


def _orig_shape(transform: str) -> tuple[str, str]:
    """(the shape the user sees, the undeformed one the deformer chain starts from)."""
    shapes = cmds.listRelatives(transform, shapes=True, fullPath=True, type="mesh") or []
    shown = [s for s in shapes if not cmds.getAttr(s + ".intermediateObject")]
    base = [s for s in cmds.ls(cmds.listHistory(shown[0]) or [], type="mesh", long=True) or []
            if cmds.getAttr(s + ".intermediateObject") and not cmds.listConnections(s + ".inMesh", source=True, destination=False)]
    return shown[0], (base[0] if base else shown[0])


def _geometry(shape: str) -> dict:
    fn = om.MFnMesh(_dag(shape))
    points = fn.getPoints(om.MSpace.kObject)
    counts, connects = fn.getVertices()
    u, v = fn.getUVs()
    uv_counts, uv_ids = fn.getAssignedUVs() if len(u) else ([], [])
    return {"points": [[p.x, p.y, p.z] for p in points], "counts": list(counts), "connects": list(connects),
            "u": list(u), "v": list(v), "uv_counts": list(uv_counts), "uv_ids": list(uv_ids)}


def _skin(shown: str) -> dict | None:
    clusters = cmds.ls(cmds.listHistory(shown, pruneDagObjects=True) or [], type="skinCluster") or []
    if not clusters:
        return None
    sc = clusters[0]
    fn = oma.MFnSkinCluster(_obj(sc))
    influences = fn.influenceObjects()
    comp = om.MFnSingleIndexedComponent()
    vertices = comp.create(om.MFn.kMeshVertComponent)
    comp.setCompleteData(om.MFnMesh(_dag(shown)).numVertices)
    weights, count = fn.getWeights(_dag(shown), vertices)
    sparse = [[i // count, i % count, w] for i, w in enumerate(weights) if w > 1e-7]
    bind = {}
    for k in range(len(influences)):
        path = influences[k].fullPathName()
        index = fn.indexForInfluenceObject(influences[k])
        bind[path] = list(om.MMatrix(cmds.getAttr(f"{sc}.bindPreMatrix[{index}]")).inverse())
    return {"influences": [influences[k].fullPathName() for k in range(len(influences))], "weights": sparse,
            "method": cmds.getAttr(sc + ".skinningMethod"), "normalize": cmds.getAttr(sc + ".normalizeWeights"),
            "bind": bind}


def _blendshapes(shown: str, base_points: list) -> list[dict]:
    out = []
    for bs in cmds.ls(cmds.listHistory(shown, pruneDagObjects=True) or [], type="blendShape") or []:
        geos = [short(g) for g in cmds.blendShape(bs, q=True, geometry=True) or []]
        indices = cmds.blendShape(bs, q=True, geometryIndices=True) or []
        g = indices[geos.index(short(shown))] if short(shown) in geos else (indices[0] if indices else 0)
        alias = cmds.aliasAttr(bs, q=True) or []
        names = {alias[i + 1]: alias[i] for i in range(0, len(alias), 2)}
        targets = []
        for idx in cmds.getAttr(bs + ".weight", multiIndices=True) or []:
            items = []
            group = f"{bs}.inputTarget[{g}].inputTargetGroup[{idx}]"
            for item in cmds.getAttr(group + ".inputTargetItem", multiIndices=True) or []:
                plug = f"{group}.inputTargetItem[{item}]"
                live = cmds.listConnections(plug + ".inputGeomTarget", source=True, destination=False, shapes=True)
                if live:  # a target mesh still connected: its points against the base
                    pts = om.MFnMesh(_dag(cmds.ls(live[0], long=True)[0])).getPoints(om.MSpace.kObject)
                    deltas = [(i, [p.x - b[0], p.y - b[1], p.z - b[2]]) for i, (p, b) in enumerate(zip(pts, base_points))]
                    deltas = [(i, d) for i, d in deltas if any(abs(x) > 1e-9 for x in d)]
                    items.append({"item": item, "points": [d + [1.0] for _i, d in deltas],
                                  "components": [f"vtx[{i}]" for i, _d in deltas]})
                    continue
                points = cmds.getAttr(plug + ".inputPointsTarget") or []
                comps = cmds.getAttr(plug + ".inputComponentsTarget") or []
                items.append({"item": item, "points": [list(p) for p in points], "components": list(comps)})
            targets.append({"index": idx, "name": names.get(f"weight[{idx}]", f"target{idx}"),
                            "weight": cmds.getAttr(f"{bs}.weight[{idx}]"), "items": items})
        out.append({"name": short(bs), "targets": targets})
    return out


def mesh(transform: str, deformed: bool = True) -> dict:
    """A mesh: undeformed with its skin and blend shapes (deformed=True: a character's), or as it is seen now."""
    shown, base = _orig_shape(transform)
    data = {"name": short(transform), "world": _matrix(transform), **_geometry(base if deformed else shown)}
    if deformed:
        data["skin"] = _skin(shown)
        data["blendshapes"] = _blendshapes(shown, data["points"])
    return data


def camera(node: str, shape: str) -> dict:
    return {"name": short(node), "lens": {a: cmds.getAttr(f"{shape}.{a}") for a in (
        "horizontalFilmAperture", "verticalFilmAperture", "horizontalFilmOffset", "verticalFilmOffset", "filmFit",
        "lensSqueezeRatio", "nearClipPlane", "farClipPlane")}}


def sample(kind: str, paths: list[str], frames: list[int], root: str = "") -> dict:
    """Animated values at these frames (getAttr with a time: evaluated in a context of its own, nothing written)."""
    out: dict = {}
    for n in paths:
        rows = out.setdefault(n, [])
        if kind == "scene.camera":
            shape = cmds.listRelatives(n, shapes=True, fullPath=True, type="camera")[0]
            for f in frames:
                rows.append([f, _matrix(n, f), cmds.getAttr(shape + ".focalLength", time=f)])
        elif n == root:  # the root joint in world space (whatever its parents do)
            for f in frames:
                rows.append([f, _matrix(n, f)])
        else:
            for f in frames:
                rows.append([f, _v3(n, "translate", f), _v3(n, "rotate", f), _v3(n, "scale", f)])
    return out
