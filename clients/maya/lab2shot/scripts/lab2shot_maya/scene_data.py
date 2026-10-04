"""A delivered scene as plain data, and built again from it — so that a result comes into the user's scene through
nothing but creating new nodes and setting their own attributes.

Why: every way Maya has of bringing a file in writes to nodes that were there before — the FBX importer sets time1,
defaultRenderGlobals and standardSurface1 and connects to initialShadingGroup; `file -import` of any file, even one
holding a single transform, sets defaultRenderGlobals. (An image plane, however made, gets four connections from
defaultColorMgtGlobals, its values left as they are: the one link a backplate makes, host.hang_picture.)
Building with createNode, the API's function sets, setAttr / connectAttr on the new nodes, sets -forceElement into a
shading group of our own, skinCluster and blendShape on our own meshes writes to nothing else (tests/test_maya.py
watches every node that was there).

`dump(tops)` runs in import_worker.py's mayapy, on what the importer made there; `build(data, parent)` in the user's
scene, inside the version's namespace (the current one) and undo step.

data: {"nodes": [transform / joint, parents first], "meshes": [...], "cameras": [...], "curves": [...]}
  node   {path, name, parent, type, attrs: {name: value}, original}
  mesh   {transform (node path), name, points, counts, connects, u, v, uv_counts, uv_ids, skin?, blendshapes}
         skin: {influences: [joint path], weights, method, normalize, bind_pre: {joint path: matrix}, geom: matrix}
  camera {transform, name, attrs}
  curve  {type: animCurveTL / TA / TU, node (path), attr, times, values (internal units), in, out (tangent types),
          angles: [[in, out] radians] | None, weighted}
"""

from __future__ import annotations

import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as cmds

TRANSFORM_ATTRS = ("translate", "rotate", "scale", "rotateOrder", "rotateAxis", "shear", "visibility",
                   "inheritsTransform")
JOINT_ATTRS = ("jointOrient", "segmentScaleCompensate", "radius")
CAMERA_ATTRS = ("horizontalFilmAperture", "verticalFilmAperture", "horizontalFilmOffset", "verticalFilmOffset",
                "filmFit", "lensSqueezeRatio", "nearClipPlane", "farClipPlane", "focalLength", "orthographic",
                "orthographicWidth", "fStop", "focusDistance")


def _short(path: str) -> str:
    return path.split("|")[-1].split(":")[-1]


def _value(node: str, attr: str):
    v = cmds.getAttr(f"{node}.{attr}")
    return list(v[0]) if isinstance(v, list) and v and isinstance(v[0], tuple) else v


def _obj(name: str):
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def dump(tops: list[str]) -> dict:
    from . import reader

    nodes, meshes, cameras = [], [], []
    stack = [(t, "") for t in reversed(tops)]
    while stack:
        path, parent = stack.pop()
        kind = cmds.nodeType(path)
        if kind not in ("transform", "joint"):
            continue
        attrs = {a: _value(path, a) for a in TRANSFORM_ATTRS + (JOINT_ATTRS if kind == "joint" else ())
                 if cmds.attributeQuery(a, node=path, exists=True)}
        original = cmds.getAttr(path + ".l2s_name") if cmds.attributeQuery("l2s_name", node=path, exists=True) else ""
        nodes.append({"path": path, "name": _short(path), "parent": parent, "type": kind, "attrs": attrs,
                      "original": original})
        for shape in cmds.listRelatives(path, shapes=True, fullPath=True, noIntermediate=True) or []:
            st = cmds.nodeType(shape)
            if st == "mesh" and not any(m["transform"] == path for m in meshes):
                m = reader.mesh(path, deformed=True)
                skin = m.get("skin")
                if skin:
                    sc = cmds.ls(cmds.listHistory(shape, pruneDagObjects=True) or [], type="skinCluster")[0]
                    fn = oma.MFnSkinCluster(_obj(sc))
                    infl = fn.influenceObjects()
                    skin["bind_pre"] = {infl[k].fullPathName(): list(cmds.getAttr(f"{sc}.bindPreMatrix[{fn.indexForInfluenceObject(infl[k])}]"))
                                        for k in range(len(infl))}
                    skin["geom"] = list(cmds.getAttr(sc + ".geomMatrix"))
                meshes.append({**m, "transform": path, "name": _short(shape)})
            elif st == "camera":
                cameras.append({"transform": path, "shape": shape, "name": _short(shape),
                                "attrs": {a: cmds.getAttr(f"{shape}.{a}") for a in CAMERA_ATTRS}})
        for child in reversed(cmds.listRelatives(path, children=True, fullPath=True, type="transform") or []):
            stack.append((child, path))
    return {"nodes": nodes, "meshes": meshes, "cameras": cameras, "curves": _curves(nodes, meshes, cameras)}


def _curves(nodes, meshes, cameras) -> list[dict]:
    out = []
    owners = [n["path"] for n in nodes] + [cmds.listRelatives(c["transform"], shapes=True, fullPath=True, type="camera")[0]
                                           for c in cameras]
    for m in meshes:
        shown = cmds.listRelatives(m["transform"], shapes=True, fullPath=True, noIntermediate=True, type="mesh")[0]
        owners += cmds.ls(cmds.listHistory(shown, pruneDagObjects=True) or [], type="blendShape") or []
    seen = set()
    for owner in owners:
        for curve in cmds.listConnections(owner, source=True, destination=False, type="animCurve") or []:
            if curve in seen:
                continue
            seen.add(curve)
            plugs = cmds.listConnections(curve + ".output", source=False, destination=True, plugs=True) or []
            if not plugs:
                continue
            dest_node, _, attr = plugs[0].partition(".")
            node_path = (cmds.ls(dest_node, long=True) or [dest_node])[0]
            fn = oma.MFnAnimCurve(_obj(curve))
            n = fn.numKeys
            unit = om.MTime.uiUnit()
            times = [fn.input(i).asUnits(unit) for i in range(n)]
            values = [fn.value(i) for i in range(n)]
            ins = [fn.inTangentType(i) for i in range(n)]
            outs = [fn.outTangentType(i) for i in range(n)]
            angles = [[fn.getTangentAngleWeight(i, True)[0].asRadians(), fn.getTangentAngleWeight(i, False)[0].asRadians()]
                      for i in range(n)]
            out.append({"type": cmds.nodeType(curve), "node": node_path, "owner_type": cmds.nodeType(node_path),
                        "attr": attr, "times": times, "values": values, "in": ins, "out": outs, "angles": angles,
                        "weighted": fn.isWeighted})
    return out


# ---------------------------------------------------------------- building (the user's scene: new nodes only)


def build(data: dict, parent: str, offset: int = 0) -> list[str]:
    """The data's nodes as new nodes in the current namespace, top ones under `parent` (a long name); keys moved
    `-offset` frames. Returns the new top nodes (long names). Writes to nothing but the nodes it makes."""
    made: dict[str, str] = {}  # data path -> new long name
    tops = []
    for n in data["nodes"]:
        under = made.get(n["parent"], parent)
        node = cmds.createNode(n["type"], name=n["name"], parent=under, skipSelect=True)
        node = cmds.ls(node, long=True)[0]
        for a, v in n["attrs"].items():
            try:
                if isinstance(v, list):
                    cmds.setAttr(f"{node}.{a}", *v)
                else:
                    cmds.setAttr(f"{node}.{a}", v)
            except RuntimeError:
                continue
        if n.get("original"):
            cmds.addAttr(node, longName="l2s_name", dataType="string")
            cmds.setAttr(node + ".l2s_name", n["original"], type="string")
        made[n["path"]] = node
        if not n["parent"]:
            tops.append(node)
    shapes: dict[str, str] = {}
    group = None
    for m in data["meshes"]:
        transform = made.get(m["transform"])
        if transform is None:
            continue
        shape = _mesh(m, transform)
        shapes[m["transform"]] = shape
        if group is None:  # one shading group of our own per version (never the scene's initialShadingGroup)
            material = cmds.createNode("lambert", name="l2s_material", skipSelect=True)
            group = cmds.createNode("shadingEngine", name="l2s_materialSG", skipSelect=True)
            cmds.connectAttr(material + ".outColor", group + ".surfaceShader")
        cmds.sets(shape, edit=True, forceElement=group)
        if m.get("blendshapes"):
            made.update(_blendshapes(transform, m["blendshapes"]))
        if m.get("skin"):
            _skin(transform, m["skin"], made)
    for c in data["cameras"]:
        transform = made.get(c["transform"])
        if transform is None:
            continue
        shape = cmds.createNode("camera", name=c["name"], parent=transform, skipSelect=True)
        for a, v in c["attrs"].items():
            try:
                cmds.setAttr(f"{shape}.{a}", v)
            except RuntimeError:
                continue
        made[c["shape"]] = cmds.ls(shape, long=True)[0]
    _keys(data["curves"], made, offset)
    return tops


def _mesh(m: dict, transform: str) -> str:
    fn = om.MFnMesh()
    points = om.MPointArray([om.MPoint(*p) for p in m["points"]])
    parent = _obj(transform)
    if m["u"]:
        fn.create(points, m["counts"], m["connects"], om.MFloatArray(m["u"]), om.MFloatArray(m["v"]), parent=parent)
        fn.assignUVs(m["uv_counts"], m["uv_ids"])
    else:
        fn.create(points, m["counts"], m["connects"], parent=parent)
    return cmds.rename(fn.fullPathName(), m["name"])


def _blendshapes(transform: str, shapes: list[dict]) -> dict:
    made = {}
    for b in shapes:
        bs = cmds.deformer(transform, type="blendShape", name=b["name"], frontOfChain=True)[0]
        made[b["name"]] = bs
        for t in b["targets"]:
            i = t["index"]
            for item in t["items"]:
                plug = f"{bs}.inputTarget[0].inputTargetGroup[{i}].inputTargetItem[{item['item']}]"
                pts = [tuple(p) + ((1.0,) if len(p) == 3 else ()) for p in item["points"]]
                cmds.setAttr(plug + ".inputPointsTarget", len(pts), *pts, type="pointArray")
                cmds.setAttr(plug + ".inputComponentsTarget", len(item["components"]), *item["components"], type="componentList")
            cmds.setAttr(f"{bs}.weight[{i}]", t["weight"])
            try:
                cmds.aliasAttr(t["name"], f"{bs}.weight[{i}]")
            except RuntimeError:
                pass
    return made


def _skin(transform: str, skin: dict, made: dict) -> None:
    influences = [made[p] for p in skin["influences"] if p in made]
    sc = cmds.skinCluster(*influences, transform, toSelectedBones=True, bindMethod=0, skinMethod=skin["method"],
                          normalizeWeights=skin["normalize"], obeyMaxInfluences=False,
                          maximumInfluences=max(1, len(influences)))[0]
    fn = oma.MFnSkinCluster(_obj(sc))
    infl = fn.influenceObjects()
    now = [infl[k].fullPathName() for k in range(len(infl))]
    back = {v: k for k, v in made.items()}
    for k in range(len(infl)):  # the bind pose as the delivered skin had it, whatever the joints' pose now
        source = back.get(now[k])
        if source in (skin.get("bind_pre") or {}):
            cmds.setAttr(f"{sc}.bindPreMatrix[{fn.indexForInfluenceObject(infl[k])}]", *skin["bind_pre"][source], type="matrix")
    if skin.get("geom"):
        cmds.setAttr(sc + ".geomMatrix", *skin["geom"], type="matrix")
    order = {p: now.index(made[p]) for p in skin["influences"] if p in made and made[p] in now}
    shape = cmds.listRelatives(transform, shapes=True, fullPath=True, noIntermediate=True)[0]
    sel = om.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    count = om.MFnMesh(dag).numVertices
    weights = om.MDoubleArray(count * len(now), 0.0)
    for v, i, w in skin["weights"]:
        path = skin["influences"][i]
        if path in order:
            weights[v * len(now) + order[path]] = w
    comp = om.MFnSingleIndexedComponent()
    vertices = comp.create(om.MFn.kMeshVertComponent)
    comp.setCompleteData(count)
    fn.setWeights(dag, vertices, om.MIntArray(list(range(len(now)))), weights, False)


def _keys(curves: list[dict], made: dict, offset: int) -> None:
    unit = om.MTime.uiUnit()
    for c in curves:
        target = made.get(c["node"])
        if target is None:
            continue
        sel = om.MSelectionList()
        try:
            sel.add(f"{target}.{c['attr']}")
        except RuntimeError:
            continue
        plug = sel.getPlug(0)
        fn = oma.MFnAnimCurve()
        fn.create(plug, {"animCurveTL": oma.MFnAnimCurve.kAnimCurveTL, "animCurveTA": oma.MFnAnimCurve.kAnimCurveTA,
                         "animCurveTU": oma.MFnAnimCurve.kAnimCurveTU}.get(c["type"], oma.MFnAnimCurve.kAnimCurveTU))
        fn.setIsWeighted(bool(c["weighted"]))
        times = [om.MTime(t - offset, unit) for t in c["times"]]
        fn.addKeys(times, c["values"], oma.MFnAnimCurve.kTangentGlobal, oma.MFnAnimCurve.kTangentGlobal)
        for i in range(fn.numKeys):
            fn.setInTangentType(i, c["in"][i])
            fn.setOutTangentType(i, c["out"][i])
