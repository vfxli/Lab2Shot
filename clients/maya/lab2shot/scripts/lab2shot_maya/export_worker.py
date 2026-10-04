"""Builds a bound object again from what reader.py read out of the user's scene, and writes the file a Lab2Shot tool
takes — in a mayapy of its own, in an empty scene: nothing here runs in the user's Maya, so nothing here can touch
the user's scene.

Run as: mayapy -c "<sys.argv = [this file, scripts folder, export_job.json]; exec this file>" (host.py finish_export:
the paths travel base64-encoded, a Windows command line mangles what is not ASCII). Writes <export_job.json>.result
({"error": ""} or the reason). Leaves through maya.standalone.uninitialize (never os._exit: Maya crashes on the way
out otherwise).

job: {kind, out, offset (the picture's frame = Maya's frame + offset), data: reader.py's, samples: reader.sample's}
"""

import json
import math
import sys
import traceback

DEG = 180.0 / math.pi


def _local_values(om, local, joint):
    """translate, rotate (degrees, in the joint's rotate order, its joint orient and rotate axis taken out) and scale
    of a joint whose local matrix is `local` (Maya: S · RA · R · JO · T)."""
    tm = om.MTransformationMatrix(local)
    t = tm.translation(om.MSpace.kTransform)
    s = tm.scale(om.MSpace.kTransform)
    jo = om.MEulerRotation([x / DEG for x in joint["jo"]], om.MEulerRotation.kXYZ).asMatrix()
    ra = om.MEulerRotation([x / DEG for x in joint["ra"]], om.MEulerRotation.kXYZ).asMatrix()
    r = ra.inverse() * tm.asRotateMatrix() * jo.inverse()
    e = om.MTransformationMatrix(r).rotation().reorder(joint["ro"])
    return [t.x, t.y, t.z], [e.x * DEG, e.y * DEG, e.z * DEG], list(s)


def _joints(cmds, om, joints, world_of):
    """The joints rebuilt, parents first, each at the world matrix `world_of` gives; {path: new node}."""
    made = {}
    for j in joints:
        parent = made.get(j["parent"])
        node = cmds.createNode("joint", name=j["name"], parent=parent) if parent else cmds.createNode("joint", name=j["name"])
        node = cmds.ls(node, long=True)[0]
        cmds.setAttr(node + ".rotateOrder", j["ro"])
        cmds.setAttr(node + ".jointOrient", *j["jo"])
        cmds.setAttr(node + ".rotateAxis", *j["ra"])
        cmds.setAttr(node + ".segmentScaleCompensate", j["ssc"])
        world = world_of(j)
        parent_world = world_of(next(p for p in joints if p["path"] == j["parent"])) if j["parent"] else om.MMatrix()
        t, r, s = _local_values(om, world * parent_world.inverse(), j)
        cmds.setAttr(node + ".translate", *t)
        cmds.setAttr(node + ".rotate", *r)
        cmds.setAttr(node + ".scale", *s)
        made[j["path"]] = node
    return made


def _mesh(cmds, om, m):
    transform = cmds.createNode("transform", name=m["name"])
    sel = om.MSelectionList()
    sel.add(transform)
    fn = om.MFnMesh()
    points = om.MPointArray([om.MPoint(*p) for p in m["points"]])
    if m["u"]:
        fn.create(points, m["counts"], m["connects"], om.MFloatArray(m["u"]), om.MFloatArray(m["v"]), parent=sel.getDependNode(0))
        fn.assignUVs(m["uv_counts"], m["uv_ids"])
    else:
        fn.create(points, m["counts"], m["connects"], parent=sel.getDependNode(0))
    shape = cmds.rename(fn.fullPathName(), m["name"] + "Shape")
    cmds.sets(shape, edit=True, forceElement="initialShadingGroup")
    cmds.xform(transform, worldSpace=True, matrix=m["world"])
    return cmds.ls(transform, long=True)[0]


def _blendshapes(cmds, transform, shapes):
    for b in shapes:
        bs = cmds.deformer(transform, type="blendShape", name=b["name"])[0]
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
            except RuntimeError:  # a name Maya does not take as an alias: the target keeps weight[i]
                pass


def _skin(cmds, om, oma, transform, skin, made):
    influences = [made[p] for p in skin["influences"] if p in made]
    sc = cmds.skinCluster(*influences, transform, toSelectedBones=True, bindMethod=0, skinMethod=skin["method"],
                          normalizeWeights=skin["normalize"], obeyMaxInfluences=False,
                          maximumInfluences=max(1, len(influences)))[0]
    sel = om.MSelectionList()
    sel.add(sc)
    fn = oma.MFnSkinCluster(sel.getDependNode(0))
    now = [fn.influenceObjects()[k].fullPathName() for k in range(len(fn.influenceObjects()))]
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


def _key(cmds, node, attrs, values, frame):
    for attr, value in zip(attrs, values):
        cmds.setKeyframe(f"{node}.{attr}", time=frame, value=value)


def build(cmds, mel, om, oma, job):
    kind, data, offset = job["kind"], job["data"], int(job.get("offset") or 0)
    samples = job.get("samples") or {}
    if kind == "scene.camera":
        cam = cmds.rename(cmds.camera()[0], data["name"])
        shape = cmds.listRelatives(cam, shapes=True, fullPath=True)[0]
        for a, v in data["lens"].items():
            cmds.setAttr(f"{shape}.{a}", v)
        for frame, matrix, focal in next(iter(samples.values())):
            tm = om.MTransformationMatrix(om.MMatrix(matrix))
            t = tm.translation(om.MSpace.kWorld)
            r = [x * DEG for x in tm.rotation(asQuaternion=False)]
            _key(cmds, cam, ("tx", "ty", "tz", "rx", "ry", "rz"), (t.x, t.y, t.z, *r), frame + offset)
            cmds.setKeyframe(f"{shape}.focalLength", time=frame + offset, value=focal)
        cmds.filterCurve(cmds.listConnections(cam, type="animCurve") or [], filter="euler")
        cmds.select(cam, replace=True)
        mel.eval("FBXExportCameras -v true;")
        return
    if kind == "scene.skeleton":
        joints = data["joints"]
        made = _joints(cmds, om, joints, lambda j: om.MMatrix(j["world"]))
        root = joints[0]
        for path, rows in samples.items():
            node = made.get(path)
            if node is None:
                continue
            j = next(x for x in joints if x["path"] == path)
            for row in rows:
                if path == root["path"]:
                    t, r, s = _local_values(om, om.MMatrix(row[1]), j)
                else:
                    t, r, s = row[1], row[2], row[3]
                _key(cmds, node, ("tx", "ty", "tz"), t, row[0] + offset)
                _key(cmds, node, ("rx", "ry", "rz"), r, row[0] + offset)
                _key(cmds, node, ("sx", "sy", "sz"), s, row[0] + offset)
        cmds.select(made[root["path"]], replace=True)
        mel.eval("FBXExportSkins -v false;")
        return
    meshes = data["meshes"]
    if kind == "scene.character":
        joints = data["joints"]
        bind = {p: om.MMatrix(m) for mesh in meshes for p, m in ((mesh.get("skin") or {}).get("bind") or {}).items()}
        by_path = {j["path"]: j for j in joints}
        known: dict = {}

        def world_of(j):  # at the bind pose: the skin's own record, else the current pose under the parent's bind
            if j["path"] in known:
                return known[j["path"]]
            if j["path"] in bind:
                w = bind[j["path"]]
            elif j["parent"]:
                parent = by_path[j["parent"]]
                w = om.MMatrix(j["world"]) * om.MMatrix(parent["world"]).inverse() * world_of(parent)
            else:
                w = om.MMatrix(j["world"])
            known[j["path"]] = w
            return w

        made = _joints(cmds, om, joints, world_of)
        built = []
        for m in meshes:
            transform = _mesh(cmds, om, m)
            _blendshapes(cmds, transform, m.get("blendshapes") or [])
            if m.get("skin"):
                _skin(cmds, om, oma, transform, m["skin"], made)
            built.append(transform)
        cmds.select([made[joints[0]["path"]], *built], replace=True)
        mel.eval("FBXExportSkins -v true;")
        mel.eval("FBXExportShapes -v true;")
        return
    cmds.select([_mesh(cmds, om, m) for m in meshes], replace=True)  # scene.model


def main():
    scripts, job_file = sys.argv[1], sys.argv[2]  # sys.argv[0] is this file
    sys.path.insert(0, scripts)
    result = {"error": ""}
    import maya.standalone

    maya.standalone.initialize(name="python")
    try:
        import maya.api.OpenMaya as om
        import maya.api.OpenMayaAnim as oma
        import maya.cmds as cmds
        import maya.mel as mel

        with open(job_file, encoding="utf-8") as f:
            job = json.load(f)
        cmds.loadPlugin("fbxmaya", quiet=True)
        mel.eval('source "currentTimeUnitToDisplayFPSString.mel";')
        cmds.currentUnit(linear=job["unit"], time=job["time"])
        if cmds.upAxis(q=True, axis=True) != job["up"]:
            cmds.upAxis(axis=job["up"], rotateView=False)
        mel.eval("FBXResetExport;")
        mel.eval("FBXExportInAscii -v false;")
        mel.eval("FBXExportConstraints -v false;")
        mel.eval("FBXExportInputConnections -v false;")
        mel.eval("FBXExportBakeComplexAnimation -v false;")
        build(cmds, mel, om, oma, job)
        out = job["out"].replace("\\", "/").replace('"', '\\"')
        mel.eval(f'FBXExport -f "{out}" -s;')
    except Exception as exc:  # noqa: BLE001 - said to the plugin through the result file
        result["error"] = str(exc) or type(exc).__name__  # the plugin says it (dcc.maya.worker_failed)
        result["trace"] = traceback.format_exc()
    with open(job_file + ".result", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    try:
        import maya.cmds as cmds

        cmds.file(new=True, force=True)
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    main()
    sys.exit(0)
