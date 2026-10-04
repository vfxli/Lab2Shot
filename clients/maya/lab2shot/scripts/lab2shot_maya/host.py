"""Maya as a Lab2Shot host (lab2shot_dcc.host.Host): Maya objects ↔ files of Lab2Shot's data types, and the Lab2Shot
node. Nothing about tools, jobs, names or the panel is here (lab2shot_dcc does those for every DCC).

What it never does: write to anything that was in the scene before — the user's nodes and Maya's own default nodes
alike: no attribute (not even set to the value it has), connection, name or parent.
- Export only reads (reader.py: OpenMaya and getAttr); a separate mayapy (export_worker.py) rebuilds the object from
  that data and writes the file. No exporter runs on the user's scene (the FBX exporter takes the user's skeleton to
  its bind pose and back and adds blend-shape target folders; file -exportSelected rewrites animation curves or
  breaks and remakes their connections).
- Import: a separate mayapy (import_worker.py) turns the delivered file into a .ma of new nodes only, with nothing
  about the scene's own nodes in it (Maya's FBX importer, run here, would set time1, defaultRenderGlobals,
  standardSurface1 and connect to initialShadingGroup); that .ma comes in under |Lab2Shot|<node>_vNNN in a namespace
  of its own, the scene's linear unit, up axis and frame rate checked afterwards.
- Binding remembers an object by its UUID and the long name it had (no connection to it).
tests/test_maya.py watches every node that was in the scene through binding, export and import with OpenMaya
callbacks (attribute set, connection made / broken, array element added / removed, lock, rename, re-parent, removal).

A Lab2Shot node is a plain `network` node carrying the state as JSON (l2s_state), with no connection to anything
(its bindings and result groups are UUIDs in the state: a network node left with no connections after its last
neighbour is deleted is deleted by Maya too — an undone import must never take the node with it); writing it never
enters the user's undo queue.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import re

import maya.cmds as cmds
import maya.mel as mel

from lab2shot_dcc import safety
from lab2shot_dcc.paths import text
from lab2shot_dcc.host import Host

STATE = "l2s_state"
ROOT_MARK = "l2s_root"
INFO = "l2s_info"
ROOT_NAME = "Lab2Shot"
TEMP_NS = "l2s_tmp_export"
VIDEO = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mxf")
FPS = {"game": 15.0, "film": 24.0, "pal": 25.0, "ntsc": 30.0, "show": 48.0, "palf": 50.0, "ntscf": 60.0}
UNITS = {"cm": "cm", "m": "m"}  # the linear units the tools' `unit` takes (others: not filled)
_FORMATS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "formats.json")


def _formats() -> dict:
    """Maya's format table (formats.json beside this file): data, not code."""
    try:
        with open(_FORMATS_FILE, encoding="utf-8") as f:
            got = json.load(f)
        return {str(k): str(v) for k, v in got.items() if not str(k).startswith("_")}
    except (OSError, ValueError):
        return {}


def _all_uuids() -> list[str]:
    """Every node's UUID (ls -uuid alone lists nothing: it needs the nodes)."""
    return cmds.ls(cmds.ls() or [], uuid=True) or []


def _uuid(node: str) -> str:
    return (cmds.ls(node, uuid=True) or [""])[0]


def _node(ref: str) -> str | None:
    """The node of a UUID now (its long name), None when it is gone."""
    try:
        got = cmds.ls(ref, long=True) if ref else []
    except RuntimeError:  # not a name or UUID Maya can read
        got = []
    return got[0] if got else None


def _bound(binding: dict) -> str | None:
    """A bound object now: by its UUID (it follows renames and re-parenting), else by the long name it had when it was
    bound (a scene re-imported gives new UUIDs); None when neither finds it."""
    return _node(binding.get("ref", "")) or ((cmds.ls(binding.get("path") or "", long=True) or [None])[0]
                                              if binding.get("path") and cmds.objExists(binding["path"]) else None)


def _settings() -> tuple:
    return (cmds.currentUnit(q=True, linear=True), cmds.upAxis(q=True, axis=True), cmds.currentUnit(q=True, time=True))


@contextlib.contextmanager
def _kept_state():
    """The selection and the current frame as they were, whatever happens inside."""
    import maya.api.OpenMaya as om

    selection = om.MGlobal.getActiveSelectionList()
    names = cmds.ls(selection=True, long=True) or []
    frame = cmds.currentTime(q=True)
    try:
        yield
    finally:
        # put back only what changed (setting the time again writes time1 even when it is the same frame); the
        # selection through the API: no undo entry
        if cmds.currentTime(q=True) != frame:
            cmds.currentTime(frame, update=False)
        if (cmds.ls(selection=True, long=True) or []) != names:
            om.MGlobal.setActiveSelectionList(selection)


def _shape_of(node: str, kind: str) -> str | None:
    if cmds.nodeType(node) == kind:
        return node
    shapes = cmds.listRelatives(node, shapes=True, fullPath=True, type=kind) or []
    return shapes[0] if shapes else None


def _image_planes(camera_shape: str) -> list[str]:
    planes = cmds.listConnections(camera_shape + ".imagePlane", source=True, destination=False, shapes=True) or []
    return [cmds.ls(p, long=True)[0] for p in planes if cmds.nodeType(p) == "imagePlane"]


def _picture_of(plane: str) -> dict:
    """What an image plane shows, read only: its file, whether it steps through a sequence, its frame offset (the
    picture's frame = Maya's frame + it) and its colour space (when colour management is on)."""
    colorspace = ""
    try:
        if cmds.colorManagementPrefs(q=True, cmEnabled=True):
            colorspace = cmds.getAttr(plane + ".colorSpace") or ""
    except (RuntimeError, ValueError):
        colorspace = ""
    return {"file": cmds.getAttr(plane + ".imageName") or "", "sequence": bool(cmds.getAttr(plane + ".useFrameExtension")),
            "frame_offset": int(cmds.getAttr(plane + ".frameOffset") or 0), "colorspace": colorspace}


def _root_joint(joint: str) -> str:
    while True:
        parent = (cmds.listRelatives(joint, parent=True, fullPath=True) or [None])[0]
        if parent is None or cmds.nodeType(parent) != "joint":
            return joint
        joint = parent


def _skinned_meshes(joints: list[str]) -> list[str]:
    """Mesh transforms skinned to any of these joints."""
    meshes = set()
    clusters = set(cmds.listConnections(joints, type="skinCluster", source=False, destination=True) or [])
    for sc in clusters:
        for geo in cmds.skinCluster(sc, q=True, geometry=True) or []:
            t = cmds.listRelatives(geo, parent=True, fullPath=True) or [geo]
            meshes.add(cmds.ls(t[0], long=True)[0])
    return sorted(meshes)


def _joints_under(node: str) -> list[str]:
    found = cmds.listRelatives(node, allDescendents=True, fullPath=True, type="joint") or []
    return ([cmds.ls(node, long=True)[0]] if cmds.nodeType(node) == "joint" else []) + found


class MayaHost(Host):
    name = "maya"
    label = "Maya"
    prefers = ("scene", "image", "video")  # what Maya most wants back: 3D data first (lab2shot_dcc.catalog.preferred)
    # what export writes each kind of object as (export_worker.py: FBX): into the inputs that read FBX
    exports = {"scene.camera": ".fbx", "scene.character": ".fbx", "scene.skeleton": ".fbx", "scene.model": ".fbx"}

    def __init__(self):
        self.formats = _formats()
        self.batch = bool(cmds.about(batch=True))
        self._queue: "queue.Queue" = queue.Queue()  # batch (mayapy): the caller pumps it on the main thread
        self.exiting = False
        self._exit_callbacks: list = []
        self._pending: dict = {}  # export_job.json path -> what export read, until finish_export writes it
        import maya.api.OpenMaya as om

        self._exit_id = om.MSceneMessage.addCallback(om.MSceneMessage.kMayaExiting, self._exiting)

    # ---- threads and windows

    def ui_language(self) -> str:
        return str(cmds.about(uiLanguage=True) or "")  # "zh_CN" in Maya's Chinese interface, "en_US" …

    def run_on_main(self, fn) -> None:
        if self.exiting:  # Maya is quitting: nothing more is queued into it (the job goes on on the server)
            return
        if self.batch:
            self._queue.put(fn)
            return
        import maya.utils

        maya.utils.executeDeferred(fn)

    def is_exiting(self) -> bool:
        return self.exiting

    def on_exit(self, callback) -> None:
        """`callback()` once when Maya starts quitting (before it tears anything down)."""
        self._exit_callbacks.append(callback)

    def _exiting(self, *_):
        self.exiting = True
        for cb in self._exit_callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001 - quitting goes on whatever happens
                pass
        try:
            import maya.api.OpenMaya as om

            om.MMessage.removeCallback(self._exit_id)
        except Exception:  # noqa: BLE001
            pass

    def pump(self, timeout: float = 0.05) -> None:
        """mayapy only: run what was queued for the main thread (the GUI's idle loop does this by itself)."""
        try:
            fn = self._queue.get(timeout=timeout)
        except queue.Empty:
            return
        fn()

    def main_window(self):
        if self.batch:
            return None
        from lab2shot_dcc.qt import QtWidgets, wrap

        import maya.OpenMayaUI as omui

        ptr = omui.MQtUtil.mainWindow()
        return wrap(ptr, QtWidgets.QWidget) if ptr else None

    def show_window(self, widget, title: str, key: str) -> None:
        """A framework window in a dockable workspaceControl named `key` (floating at first; the user may dock it).
        Closing the control closes the window (its timers stop with it)."""
        import maya.OpenMayaUI as omui
        from lab2shot_dcc.qt import unwrap

        if cmds.workspaceControl(key, exists=True):
            cmds.deleteUI(key)
        cmds.workspaceControl(key, label=title, floating=True, retain=False, initialWidth=460 if "Panel" in key else 1100,
                              initialHeight=760, closeCommand=lambda *_: _close_quietly(widget))
        omui.MQtUtil.addWidgetToMayaLayout(unwrap(widget), int(omui.MQtUtil.findControl(key)))
        widget.setProperty("lab2shot_control", key)
        widget.show()

    def close_window(self, widget) -> None:
        key = widget.property("lab2shot_control")
        if key and cmds.workspaceControl(key, exists=True):
            cmds.deleteUI(key)  # its closeCommand closes the widget
        else:
            _close_quietly(widget)

    def on_selection_changed(self, callback):
        if self.batch:
            return None
        return cmds.scriptJob(event=["SelectionChanged", callback])

    def off_selection_changed(self, handle) -> None:
        if handle is not None and cmds.scriptJob(exists=handle):
            cmds.scriptJob(kill=handle, force=True)

    def confirm(self, question: str) -> bool:
        if self.batch:
            return True
        yes, no = text("dcc.maya.import"), text("dcc.maya.not_now")
        return cmds.confirmDialog(title="Lab2Shot", message=question, button=[yes, no], defaultButton=yes,
                                  cancelButton=no, dismissString=no) == yes

    # ---- what things are

    def _types_of(self, node: str) -> list[str]:
        kind = cmds.nodeType(node)
        out = []
        if kind == "imagePlane" or _shape_of(node, "imagePlane"):
            plane = _shape_of(node, "imagePlane")
            path = cmds.getAttr(plane + ".imageName") or ""
            return ["video" if path.lower().endswith(VIDEO) else "image"]
        cam = _shape_of(node, "camera")
        if cam:
            out.append("scene.camera")
            for plane in _image_planes(cam):
                path = cmds.getAttr(plane + ".imageName") or ""
                out.append("video" if path.lower().endswith(VIDEO) else "image")
            return out
        joints = _joints_under(node)
        if joints:
            root = _root_joint(joints[0])
            out.append("scene.skeleton")
            if _skinned_meshes(_joints_under(root)):
                out.append("scene.character")
            return out
        meshes = cmds.listRelatives(node, allDescendents=True, fullPath=True, type="mesh") or []
        if meshes:
            skinned = [m for m in meshes if cmds.listConnections(m, type="skinCluster")]
            out.append("scene.character" if skinned else "scene.model")
        return out

    def selection_types(self) -> list[str]:
        out = []
        for node in cmds.ls(selection=True, long=True) or []:
            for t in self._types_of(node):
                if t not in out:
                    out.append(t)
        return out

    def bind(self, data_type: str, kinds: list[str]):
        from lab2shot_dcc.catalog import fits

        for node in cmds.ls(selection=True, long=True) or []:
            have = self._types_of(node)
            for t in have:
                if fits(data_type, t):
                    return self._binding(node, t)
        return None

    def why_not(self, data_type: str) -> str:
        words = {"image": "dcc.maya.want.image", "video": "dcc.maya.want.video", "scene.camera": "dcc.maya.want.camera",
                 "scene.character": "dcc.maya.want.character", "scene.skeleton": "dcc.maya.want.skeleton",
                 "scene.model": "dcc.maya.want.model"}  # a data type -> its word's key
        key = next((w for t, w in words.items() if any(a.startswith(t) for a in data_type.split("|"))), None)
        return text("dcc.maya.select_first", what=text(key) if key else data_type)

    def _binding(self, node: str, data_type: str) -> dict:
        short = cmds.ls(node)[0]
        if data_type in ("image", "video"):
            plane = _shape_of(node, "imagePlane")
            camera = None
            if plane is None:
                camera = _shape_of(node, "camera")
                plane = _image_planes(camera)[0]
            picture = _picture_of(plane)
            return {**picture, "ref": _uuid(plane), "path": cmds.ls(plane, long=True)[0],
                    "camera": _uuid(camera) if camera else "", "type": data_type,
                    "label": text("dcc.maya.plane_label", node=cmds.ls(plane)[0], file=os.path.basename(picture["file"]))}
        if data_type in ("scene.character", "scene.skeleton"):
            joints = _joints_under(node)
            if cmds.nodeType(node) == "joint" or data_type == "scene.skeleton":
                node = _root_joint(joints[0]) if joints else node
                short = cmds.ls(node)[0]
        return {"ref": _uuid(node), "path": cmds.ls(node, long=True)[0], "type": data_type, "label": short}

    def describe_binding(self, binding: dict) -> str:
        if binding.get("file") and not binding.get("ref"):
            return binding["file"]
        node = _bound(binding)
        if node is None:
            return text("dcc.maya.binding_gone", label=binding.get("label", ""))
        if binding.get("file"):
            return text("dcc.maya.plane_label", node=cmds.ls(node)[0], file=os.path.basename(binding["file"]))
        return cmds.ls(node)[0]

    def binding_alive(self, binding: dict) -> bool:
        if binding.get("ref"):
            return _bound(binding) is not None
        return os.path.exists(binding.get("file", ""))

    # ---- export (reading only: not one write to the user's nodes)

    def export(self, binding: dict, folder: str) -> str:
        """Main thread, one step: the bound object read out of the scene (reader.py: OpenMaya and getAttr only, not
        one write to the user's nodes, no selection change, nothing in the undo queue). Its animation is read in
        short steps afterwards (finish_export), the file is written by a separate mayapy (export_worker.py)."""
        from . import reader

        node = _bound(binding)
        if node is None:
            raise RuntimeError(text("dcc.maya.bound_gone", label=binding.get("label", "")))
        kind = binding.get("type", "")
        sampled, root = [], ""
        if kind == "scene.camera":
            shape = _shape_of(node, "camera")
            data, sampled = reader.camera(node, shape), [node]
        elif kind in ("scene.character", "scene.skeleton"):
            joints = _joints_under(node)
            if not joints:
                raise RuntimeError(text("dcc.maya.no_joints", node=cmds.ls(node)[0]))
            root = _root_joint(joints[0])
            data = {"joints": reader.joints(root)}
            if kind == "scene.character":
                meshes = _skinned_meshes(_joints_under(root))
                if not meshes:
                    raise RuntimeError(text("dcc.maya.no_skinned_mesh", node=cmds.ls(root)[0]))
                data["meshes"] = [reader.mesh(m) for m in meshes]
            else:
                sampled = [j["path"] for j in data["joints"]]
        elif kind == "scene.model":
            data = {"meshes": [reader.mesh(node, deformed=False)]}
        else:
            raise RuntimeError(text("dcc.maya.cannot_export", kind=kind))
        first, last = self.frame_range()
        unit, up, time_unit = _settings()
        job = {"kind": kind, "data": data, "sampled": sampled, "root": root, "range": [first, last],
               "offset": int(binding.get("plate_offset") or 0), "unit": unit, "up": up, "time": time_unit,
               "out": safety.free_path(folder, safety.clean(reader.short(node), 40, "object") + ".fbx")}
        path = os.path.join(folder, "export_job.json")
        self._pending[path] = job  # written out on the background thread (finish_export)
        return path

    def finish_export(self, binding: dict, prepared: str, cancelled) -> str:
        """Background thread: the animation read in short main-thread steps (lab2shot_dcc.mainthread.sample), then a
        separate mayapy turns the copy into the file (export_worker.py); stopped when cancelled or past
        EXPORT_TIMEOUT."""
        from lab2shot_dcc import mainthread

        from . import reader

        job = self._pending.pop(prepared)
        samples: dict = {}
        if job["sampled"]:
            first, last = job["range"]
            samples = mainthread.sample(self, first, last, lambda frames: reader.sample(job["kind"], job["sampled"], frames,
                                                                                      job["root"]), cancelled) or {}
        job["samples"] = samples
        with open(prepared, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
        return self._run_worker("export_worker.py", prepared, cancelled, job["out"])

    def _run_worker(self, worker_name: str, job_path: str, cancelled, out: str) -> str:
        """Background thread: a separate mayapy runs one of our workers on a job file; stopped when cancelled or past
        EXPORT_TIMEOUT. Returns `out` once the worker wrote it."""
        import base64
        import subprocess
        import sys
        import time

        folder = os.path.dirname(job_path)
        env = {**os.environ, "MAYA_SKIP_USERSETUP_PY": "1", "MAYA_DISABLE_CER": "1", "MAYA_DISABLE_CIP": "1"}
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), worker_name)
        scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # paths go in as base64 (a Windows command line mangles what is not ASCII: a Chinese user or project name)
        args = base64.b64encode(json.dumps([worker, scripts, job_path]).encode("utf-8")).decode("ascii")
        code = ("import base64,json,sys;sys.argv=json.loads(base64.b64decode('%s').decode('utf-8'));"
                "exec(compile(open(sys.argv[0],encoding='utf-8').read(),sys.argv[0],'exec'))" % args)
        log_path = job_path + ".log"
        with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
            proc = subprocess.Popen([_mayapy(), "-c", code], stdout=log_file, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, cwd=os.path.expanduser("~"),
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
            end = time.time() + EXPORT_TIMEOUT
            while proc.poll() is None:
                if cancelled() or time.time() > end:
                    proc.kill()
                    proc.wait()
                    if cancelled():
                        from lab2shot_dcc.connection import Cancelled

                        raise Cancelled()
                    raise RuntimeError(text("dcc.maya.worker_timeout", minutes=EXPORT_TIMEOUT // 60))
                time.sleep(0.2)
        try:
            with open(job_path + ".result", encoding="utf-8") as f:
                result = json.load(f)
        except (OSError, ValueError):
            raise RuntimeError(text("dcc.maya.worker_died", code=proc.returncode, log=log_path)) from None
        if result.get("error"):
            raise RuntimeError(text("dcc.maya.export_failed" if worker_name == "export_worker.py" else "dcc.maya.import_failed",
                                    error=result["error"]))
        if not os.path.isfile(out):
            raise RuntimeError(text("dcc.maya.nothing_written", file=out))
        return out

    def prepare_import(self, item: dict, path: str, cancelled) -> str:
        """Background thread: the delivered file read in a mayapy of its own into plain data (import_worker.py), with
        the scene's units, time unit and up axis (asked on the main thread). A file Maya does not bring in as scene data
        (pictures: depth, mattes …) is left as it is: it stays in the version folder, can_import says why it is not
        imported."""
        if os.path.splitext(path)[1].lower() not in SCENE_FILES:
            return path
        unit, up, time_unit = _main_call(self, _settings, cancelled)
        out = os.path.splitext(path)[0] + "_scene.json"
        job = {"source": path, "out": out, "unit": unit, "up": up, "time": time_unit}
        job_path = os.path.splitext(path)[0] + "_import.json"
        with open(job_path, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False)
        return self._run_worker("import_worker.py", job_path, cancelled, out)

    # ---- what the scene's context is made of (lab2shot_dcc.context.scene_context puts it together)

    def frame_range(self) -> tuple[int, int]:
        return int(cmds.playbackOptions(q=True, minTime=True)), int(cmds.playbackOptions(q=True, maxTime=True))

    def fps(self) -> float:
        unit = cmds.currentUnit(q=True, time=True)
        if unit in FPS:
            return FPS[unit]
        m = re.match(r"^([\d.]+)fps$", unit)
        return float(m.group(1)) if m else 24.0

    def linear_unit(self) -> str:
        return UNITS.get(cmds.currentUnit(q=True, linear=True), "")

    def camera_lens(self, binding: dict) -> tuple[float, float] | None:
        """The camera by its UUID: focal length (mm) and horizontal film aperture (Maya's inches, in mm)."""
        node = _node(binding.get("ref", ""))
        shape = _shape_of(node, "camera") if node else None
        if not shape:
            return None
        return float(cmds.getAttr(shape + ".focalLength")), round(float(cmds.getAttr(shape + ".horizontalFilmAperture")) * 25.4, 4)

    def camera_picture(self, binding: dict) -> dict | None:
        """The bound camera's own picture: its first image plane, read only."""
        node = _bound(binding)
        shape = _shape_of(node, "camera") if node else None
        planes = _image_planes(shape) if shape else []
        return _picture_of(planes[0]) if planes else None

    def can_import(self, data_type: str, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        plugin = {".fbx": "fbxmaya", ".abc": "AbcImport", ".usd": "mayaUsdPlugin", ".usda": "mayaUsdPlugin",
                  ".usdc": "mayaUsdPlugin", ".usdz": "mayaUsdPlugin"}.get(ext)
        if path.lower().endswith("_scene.json"):  # made by prepare_import: nothing to load here
            return ""
        if plugin is None:
            return text("dcc.maya.cannot_import", ext=ext)
        if not cmds.pluginInfo(plugin, q=True, loaded=True):
            try:
                cmds.loadPlugin(plugin, quiet=True)
            except RuntimeError:
                return text("dcc.maya.plugin_failed", plugin=plugin)
        return ""

    # ---- adding to the scene

    def undo_chunk(self, label: str):
        @contextlib.contextmanager
        def chunk():
            cmds.undoInfo(openChunk=True, chunkName=label)
            try:
                yield
            finally:
                cmds.undoInfo(closeChunk=True)
        return chunk()

    def one_undo_step(self, label: str, fn):
        """Maya's importers cannot be undone: the whole version runs in lab2shotUndoStep (undo.py)."""
        from . import undo

        return undo.run(label, fn)

    def name_taken(self, name: str) -> bool:
        return bool(cmds.objExists(name) or cmds.ls(name) or cmds.namespace(exists=":" + name))

    def _root(self) -> str:
        for node in cmds.ls(assemblies=True, long=True) or []:
            if cmds.attributeQuery(ROOT_MARK, node=node, exists=True):
                return node
        name = safety.numbered(ROOT_NAME, lambda n: bool(cmds.ls(n)), safety.NODE_MOST)
        root = cmds.group(empty=True, name=name, world=True)
        cmds.addAttr(root, longName=ROOT_MARK, attributeType="bool")
        return cmds.ls(root, long=True)[0]

    def make_group(self, name: str, info: dict) -> str:
        with _kept_state():
            group = cmds.group(empty=True, name=name, parent=self._root())
            cmds.addAttr(group, longName=INFO, dataType="string")
            cmds.setAttr(f"{group}.{INFO}", json.dumps(info, ensure_ascii=False), type="string")
            return _uuid(group)

    def import_result(self, item: dict, path: str, group: str, namespace: str) -> list[str]:
        """The data prepare_import read (scene_data) built as new nodes in the version's namespace, under its group;
        the scene's unit, up axis and frame rate checked after. Nothing that was in the scene is written to."""
        from . import scene_data

        if not path.lower().endswith("_scene.json"):
            raise RuntimeError(text("dcc.maya.not_scene_data", path=path))
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        group_node = _node(group)
        before = _settings()
        offset = int((item.get("plate") or {}).get("frame_offset") or 0)
        with _kept_state():
            made = self._import_with_namespace(namespace, lambda: scene_data.build(data, group_node, offset))
        after = _settings()
        if after != before:
            raise RuntimeError(text("dcc.maya.settings_changed", before=before, after=after))
        return [_uuid(m) for m in made]

    def _import_with_namespace(self, namespace: str, run) -> list[str]:
        if not cmds.namespace(exists=":" + namespace):
            cmds.namespace(add=namespace)
        current = cmds.namespaceInfo(currentNamespace=True)
        cmds.namespace(set=":" + namespace)
        try:
            got = run()
        finally:
            cmds.namespace(set=":" + current.lstrip(":") if current != ":" else ":")
        return got if isinstance(got, list) else [got]

    def cameras_in(self, refs: list[str]) -> list[str]:
        out = []
        for node in (n for n in (_node(r) for r in refs) if n):
            shapes = ([node] if cmds.nodeType(node) == "camera" else []) + \
                     (cmds.listRelatives(node, allDescendents=True, fullPath=True, type="camera") or [])
            out += [u for u in (_uuid(s) for s in shapes) if u and u not in out]
        return out

    def hang_picture(self, camera: str, picture: dict, namespace: str) -> list[str]:
        """A new image plane on one of our new cameras (in its underworld, as Maya hangs every plane), showing
        `picture` as the user's plane would: the same file, stepping through the sequence at the picture's frame =
        Maya's frame + frame_offset (frameExtension driven by a linear curve of our own: Maya's own way, and an
        expression, connect time1), in its colour space. Only our own nodes are written. Maya connects every colour-managed node it
        creates (an image plane, a file texture, the user's alike) to defaultColorMgtGlobals' cmEnabled /
        configFileEnabled / configFilePath / workingSpaceName: those four connections leave that node's values as
        they are, and no way of creating a plane avoids them (tests/test_backplate.py)."""
        import maya.api.OpenMaya as om
        import maya.api.OpenMayaAnim as oma

        shape = _node(camera)
        if shape is None or cmds.nodeType(shape) != "camera":
            return []

        def make() -> list[str]:
            got = cmds.imagePlane(camera=shape, fileName=picture["file"], name="backplate", showInAllViews=False)
            plane = cmds.ls(got[-1], long=True)[0]
            made = [cmds.ls(n, long=True)[0] for n in got]
            if bool(picture.get("sequence")) or picture["file"].lower().endswith(VIDEO):
                # the curve first: turning useFrameExtension on with frameExtension free makes Maya connect time1 to
                # it (time1 is not ours); a curve needs no connection to time
                curve = cmds.createNode("animCurveTU", name="backplate_frame", skipSelect=True)
                sel = om.MSelectionList()
                sel.add(curve)
                fn = oma.MFnAnimCurve(sel.getDependNode(0))
                unit = om.MTime.uiUnit()
                # smooth, not linear: a linear tangent on an end key has no next key to aim at and goes flat, and
                # linear infinity carries on along the end tangents (frameExtension is whole: it rounds)
                fn.addKeys([om.MTime(0, unit), om.MTime(1, unit)], [0.0, 1.0], oma.MFnAnimCurve.kTangentSmooth,
                           oma.MFnAnimCurve.kTangentSmooth)
                fn.setPreInfinityType(oma.MFnAnimCurve.kLinear)
                fn.setPostInfinityType(oma.MFnAnimCurve.kLinear)
                cmds.connectAttr(curve + ".output", plane + ".frameExtension")
                made.append(cmds.ls(curve, long=True)[0])
                cmds.setAttr(plane + ".useFrameExtension", True)
                cmds.setAttr(plane + ".frameOffset", int(picture.get("frame_offset") or 0))
            space = picture.get("colorspace") or ""
            if space and space in (cmds.colorManagementPrefs(q=True, inputSpaceNames=True) or []):
                cmds.setAttr(plane + ".ignoreColorSpaceFileRules", True)
                cmds.setAttr(plane + ".colorSpace", space, type="string")
            return made

        with _kept_state():
            made = self._import_with_namespace(namespace, make)
        return [u for u in (_uuid(n) for n in made) if u]

    def select(self, refs: list[str]) -> None:
        nodes = [n for n in (_node(r) for r in refs) if n]
        if nodes:
            cmds.select(nodes, replace=True)

    def unsaved(self) -> bool:
        """The scene has never been saved (untitled): nothing is computed for it (Plugin.why_not_saved)."""
        return not cmds.file(q=True, sceneName=True)

    def project_dir(self) -> str:
        return cmds.workspace(q=True, rootDirectory=True)

    # ---- the Lab2Shot node

    def nodes(self) -> list[str]:
        return [_uuid(n) for n in cmds.ls(type="network", long=True) or []
                if cmds.attributeQuery(STATE, node=n, exists=True)]

    def node_name(self, node: str) -> str:
        found = _node(node)
        return cmds.ls(found)[0] if found else node

    def create_node(self, name: str) -> str:
        node = cmds.createNode("network", name=name, skipSelect=True)
        cmds.addAttr(node, longName=STATE, dataType="string")
        return _uuid(node)

    def store(self, node: str, state: dict) -> None:
        found = _node(node)
        if found is None:
            raise RuntimeError(text("dcc.maya.node_gone"))
        # the plugin's own bookkeeping never enters the user's undo queue: Ctrl+Z undoes the user's work and a whole
        # imported version (lab2shotUndoStep, which puts the node's state back itself), not "job queued"
        was = cmds.undoInfo(q=True, stateWithoutFlush=True)
        cmds.undoInfo(stateWithoutFlush=False)
        try:
            cmds.setAttr(f"{found}.{STATE}", json.dumps(state, ensure_ascii=False), type="string")
        finally:
            cmds.undoInfo(stateWithoutFlush=was)

    def load(self, node: str) -> dict:
        found = _node(node)
        if found is None:
            return {}
        try:
            got = json.loads(cmds.getAttr(f"{found}.{STATE}") or "{}")
            return got if isinstance(got, dict) else {}
        except ValueError:
            return {}

    def selected_node(self) -> str | None:
        for n in cmds.ls(selection=True, long=True) or []:
            if cmds.nodeType(n) == "network" and cmds.attributeQuery(STATE, node=n, exists=True):
                return _uuid(n)
        return None


EXPORT_TIMEOUT = 900  # seconds the export process may take
SCENE_FILES = (".fbx", ".abc", ".usd", ".usda", ".usdc", ".usdz")  # what import_worker.py reads into scene data


def _mayapy() -> str:
    """This Maya's own mayapy."""
    import sys

    name = "mayapy.exe" if sys.platform == "win32" else "mayapy"
    for folder in (os.path.join(os.environ.get("MAYA_LOCATION", ""), "bin"), os.path.dirname(sys.executable)):
        path = os.path.join(folder, name)
        if folder and os.path.isfile(path):
            return path
    raise RuntimeError(text("dcc.maya.no_mayapy"))


def _main_call(host, fn, cancelled):
    """The framework's one main-thread call (lab2shot_dcc.mainthread): the host only provides run_on_main."""
    from lab2shot_dcc import mainthread

    return mainthread.call(host, fn, cancelled)


def _close_quietly(widget) -> None:
    try:
        widget.close()
    except RuntimeError:  # already gone with its control
        pass


def _mel_path(path: str) -> str:
    return path.replace("\\", "/").replace('"', '\\"')
