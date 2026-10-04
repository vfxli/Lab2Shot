"""Reads a delivered file in a mayapy of its own and writes it out as plain data (scene_data.dump) — so that bringing
the result into the user's scene is nothing but creating new nodes (scene_data.build): Maya's importers, run in the
user's scene, write to nodes that were there (the FBX importer sets time1, defaultRenderGlobals, standardSurface1 and
connects to initialShadingGroup; any file -import sets defaultRenderGlobals).

Here, in an empty scene with the user's units, time unit and up axis: the file is imported; its namespaces folded;
names the importer spelled with FBXASC codes cleaned (the original kept as l2s_name); then dumped.

Run as: mayapy -c "<sys.argv = [this file, scripts folder, import_job.json]; exec this file>" (host.py). Writes
<import_job.json>.result ({"error": ""} or the reason). Leaves through maya.standalone.uninitialize.

job: {source, out, unit, time, up}
"""

import json
import re
import sys
import traceback


def _decode(name):
    raw = bytearray()
    for part in re.split(r"(FBXASC\d{3})", name):
        if part.startswith("FBXASC") and len(part) == 9:
            raw.append(int(part[6:]) & 0xFF)
        else:
            raw += part.encode("utf-8")
    return raw.decode("utf-8", errors="replace")


def _clean(text, most=60):
    word = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", text)).strip("_")
    if not word or len(word) < len(text.strip()) / 3:
        import hashlib

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:6]
        word = f"{word}_{digest}" if word else f"node_{digest}"
    if not word[0].isalpha():
        word = "n_" + word
    return word[:most]


def main():
    scripts, job_file = sys.argv[1], sys.argv[2]
    sys.path.insert(0, scripts)
    result = {"error": ""}
    import maya.standalone

    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as cmds
        import maya.mel as mel

        with open(job_file, encoding="utf-8") as f:
            job = json.load(f)
        cmds.currentUnit(linear=job["unit"], time=job["time"])
        if cmds.upAxis(q=True, axis=True) != job["up"]:
            cmds.upAxis(axis=job["up"], rotateView=False)
        before = set(cmds.ls(cmds.ls() or [], uuid=True) or [])
        source = job["source"]
        if source.lower().endswith(".fbx"):
            cmds.loadPlugin("fbxmaya", quiet=True)
            mel.eval('source "currentTimeUnitToDisplayFPSString.mel";')
            mel.eval('FBXProperty "Import|IncludeGrp|Animation" -v true;')
            mel.eval("FBXImportMode -v add;")
            mel.eval("FBXImportSetMayaFrameRate -v false;")
            mel.eval("FBXImportFillTimeline -v false;")
            mel.eval("FBXImportCameras -v true;")
            mel.eval("FBXImportSkins -v true;")
            mel.eval("FBXImportShapes -v true;")
            cmds.file(source, i=True, type="FBX", ignoreVersion=True)
        elif source.lower().endswith(".abc"):
            cmds.loadPlugin("AbcImport", quiet=True)
            cmds.AbcImport(source, mode="import")
        else:
            cmds.loadPlugin("mayaUsdPlugin", quiet=True)
            cmds.mayaUSDImport(file=source, primPath="/")
        # namespaces the file brought: folded into the root (the version gets its own in the user's scene)
        for ns in sorted(cmds.namespaceInfo(":", listOnlyNamespaces=True, recurse=True) or [], key=lambda n: -n.count(":")):
            if ns not in ("UI", "shared"):
                cmds.namespace(removeNamespace=":" + ns, mergeNamespaceWithRoot=True)
        new = [u for u in cmds.ls(cmds.ls() or [], uuid=True) or [] if u not in before]
        for ref in new:  # clean names, the original kept
            node = (cmds.ls(ref) or [None])[0]
            if node is None or "FBXASC" not in node.split("|")[-1]:
                continue
            original = _decode(node.split("|")[-1])
            renamed = cmds.rename(node, _clean(original))
            if not cmds.attributeQuery("l2s_name", node=renamed, exists=True):
                cmds.addAttr(renamed, longName="l2s_name", dataType="string")
            cmds.setAttr(renamed + ".l2s_name", original, type="string")
        from lab2shot_maya import scene_data

        new = [u for u in cmds.ls(cmds.ls() or [], uuid=True) or [] if u not in before]
        tops = [n for n in cmds.ls(new, long=True, transforms=True) or [] if not cmds.listRelatives(n, parent=True)]
        with open(job["out"], "w", encoding="utf-8") as f:
            json.dump(scene_data.dump(tops), f, ensure_ascii=False)
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
