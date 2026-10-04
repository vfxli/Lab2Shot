"""Nuke as a Lab2Shot host (lab2shot_dcc.host.Host): Nuke nodes <-> files of Lab2Shot's data types, and the Lab2Shot
node. Nothing about tools, jobs, names or the panel is here (lab2shot_dcc does those for every DCC).

What it never does: write to a node that was in the script before — no knob (not even `selected`), no input, no name,
no position; the Root's settings stay as they are (checked around every import). Everything is made with
nuke.nodes.<Class>() (nuke.createNode would wire the new node to the selection, select it and open its panel), at the
root of the script, and our nodes are told apart by a hidden knob of their own (`l2s_ref`). The one link to a user's
node is an input of OUR node: the STMap of a delivered ST map takes the bound picture's Read as its source (the link is
kept on the STMap; the user's Read is unchanged).

References: a user's node is remembered by its full name and a fingerprint (class and file: reader.fingerprint), so a
rename is followed when the fingerprint is unique; our own nodes by the uuid in their `l2s_ref` knob ("l2s:<uuid>").

One space (the Nuke plugin design, V-unit; measured in Nuke 17.0v2): GeoImport takes a USD file's numbers as they are
(metersPerUnit is only metadata there), and a delivered .nk camera is in the same centimetre numbers, so a camera and
the points it saw meet without any factor on Nuke's side.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import queue
import threading
import uuid

import nuke

from lab2shot_dcc import log, safety
from lab2shot_dcc.host import Host

from . import exr, nk, reader

STATE = "l2s_state"  # the Lab2Shot node's state: JSON as base64 (a String knob is TCL: [ ] { } $ \ would be read)
REF = "l2s_ref"  # our nodes' own id
INFO = "l2s_info"  # what our result nodes came from (the original name), JSON as base64
SUMMARY = "l2s_summary"
OPEN = "l2s_open"
TAB = "l2s_tab"
NODE_CLASS = "NoOp"
PANEL_KEY = "Lab2ShotPanelControl"  # Host.show_panel's key: the docked pane
_HERE = os.path.dirname(os.path.abspath(__file__))
_FORMATS_FILE = os.path.join(_HERE, "formats.json")
_COLORSPACES_FILE = os.path.join(_HERE, "colorspaces.json")
ROOT_KNOBS = ("format", "proxy_format", "fps", "first_frame", "last_frame", "colorManagement", "OCIO_config",
              "customOCIOConfigPath", "workingSpaceLUT", "monitorLut", "int8Lut", "int16Lut", "logLut", "floatLut")
IMAGES = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dpx", ".hdr", ".tga", ".webp") + reader.VIDEO
GEO = (".usd", ".usda", ".usdc", ".usdz", ".abc")
POINT_FILES = (".ply", ".splat", ".spz")
DATA_TYPES = ("image.1", "image.2")  # data maps: read raw (no colour conversion, as the server wrote them)
GAP = 120  # node graph units between the nodes a version lays out


def _table(path: str) -> dict:
    """One of the host's data tables beside this file (formats.json, colorspaces.json): data, not code."""
    try:
        with open(path, encoding="utf-8") as f:
            got = json.load(f)
        return {str(k): v for k, v in got.items() if not str(k).startswith("_")}
    except (OSError, ValueError):
        return {}


def _say(key: str, **params) -> str:
    """One of the plugin's words (lab2shot_dcc.paths.text). A dcc.nuke.* key the language catalogues do not have yet
    is logged and said as its key (the catalogue entries are added by whoever owns lab2shot/i18n)."""
    from lab2shot_dcc.paths import text

    try:
        return text(key, **params)
    except KeyError:
        log.get().warning("no word for %s yet", key)
        return key + (" " + json.dumps(params, ensure_ascii=False, default=str) if params else "")


def _b64(data) -> str:
    return base64.b64encode(json.dumps(data, ensure_ascii=False).encode("utf-8")).decode("ascii")


def _unb64(text: str):
    return json.loads(base64.b64decode(text.encode("ascii")).decode("utf-8")) if text else None


def _all():
    return nuke.allNodes(recurseGroups=True)


def _ours(node) -> str:
    knob = node.knob(REF)
    return f"l2s:{knob.value()}" if knob is not None and knob.value() else ""


def _find_ours(ref: str):
    if not ref.startswith("l2s:"):
        return None
    want = ref[4:]
    for n in nuke.allNodes():
        knob = n.knob(REF)
        if knob is not None and knob.value() == want:
            return n
    return None


def _find_user(binding: dict):
    """A user's bound node now: by its full name when it still has the same fingerprint, else the one node with that
    fingerprint (renamed); None when neither finds it."""
    path, mark = binding.get("path", ""), binding.get("fingerprint", "")
    node = nuke.toNode(path) if path else None
    if node is not None and (not mark or reader.fingerprint(node) == mark):
        return node
    if mark:
        found = [n for n in _all() if not _ours(n) and n.Class() == mark.split("|", 1)[0] and reader.fingerprint(n) == mark]
        if len(found) == 1:
            return found[0]
    return None


def _root_settings() -> dict:
    root = nuke.root()
    return {k: root[k].toScript() for k in ROOT_KNOBS if root.knob(k) is not None}


def _hidden_string(name: str, value: str = ""):
    knob = nuke.String_Knob(name, name)
    knob.setFlag(nuke.INVISIBLE)
    knob.setValue(value)
    return knob


def _width(node) -> int:
    try:
        return int(node.screenWidth()) or 80
    except Exception:  # noqa: BLE001
        return 80


def _height(node) -> int:
    try:
        return int(node.screenHeight()) or 20
    except Exception:  # noqa: BLE001
        return 20


class NukeHost(Host):
    name = "nuke"
    label = "Nuke"
    # tools that give pictures first, then a camera, 2D tracks, then the rest of 3D (catalog.preferred)
    prefers = ("image", "video", "scene.camera", "tracks2d", "scene")
    # a camera goes up as the USD camera export writes (lab2shot_dcc.usd_camera): into the inputs that read USD
    exports = {"scene.camera": ".usda"}

    def __init__(self):
        self.formats = {k: str(v) for k, v in _table(_FORMATS_FILE).items()}
        self.colorspaces = {k: list(v) for k, v in _table(_COLORSPACES_FILE).items() if isinstance(v, list)}
        self.gui = bool(nuke.GUI)
        self._queue: "queue.Queue" = queue.Queue()  # nuke -t: the caller pumps it on the main thread
        self.exiting = False
        self._exit_callbacks: list = []
        self._pending: dict = {}  # export's file -> what finish_export samples and writes
        self._made: list | None = None  # the nodes one undo step made (removed again when it fails)
        self._layout: dict = {}  # a version's backdrop ref -> how many nodes are in it
        self._in_step = False
        if self.gui:
            try:
                from lab2shot_dcc.qt import QtWidgets

                app = QtWidgets.QApplication.instance()
                if app is not None:
                    app.aboutToQuit.connect(self._exiting)
            except Exception:  # noqa: BLE001 - quitting is then only noticed by the jobs' own time limits
                log.get().warning("no quit signal from Nuke's Qt")

    # ---- threads and windows

    def run_on_main(self, fn) -> None:
        if self.exiting:
            return
        if not self.gui:
            self._queue.put(fn)
            return
        if threading.current_thread() is threading.main_thread():
            from lab2shot_dcc.qt import QtCore

            QtCore.QTimer.singleShot(0, fn)  # executeInMainThread from the main thread would hang (Nuke's docs)
            return
        nuke.executeInMainThread(fn)

    def pump(self, timeout: float = 0.05) -> None:
        """nuke -t only: run what was queued for the main thread (Nuke's own event loop does this in the GUI)."""
        try:
            fn = self._queue.get(timeout=timeout)
        except queue.Empty:
            return
        fn()

    def is_exiting(self) -> bool:
        return self.exiting

    def on_exit(self, callback) -> None:
        self._exit_callbacks.append(callback)

    def _exiting(self, *_):
        self.exiting = True
        for cb in self._exit_callbacks:
            try:
                cb()
            except Exception:  # noqa: BLE001 - quitting goes on whatever happens
                pass

    def main_window(self):
        if not self.gui:
            return None
        from lab2shot_dcc.qt import QtWidgets

        for w in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(w, QtWidgets.QMainWindow):
                return w
        return None

    def show_window(self, widget, title: str, key: str) -> None:
        """The panel docked in a pane (panel.py: registered in the Pane menu, kept with saved workspaces); any other
        framework window (the embedded web page) a tool window over Nuke's main window, one per key."""
        from lab2shot_dcc.qt import QtCore

        if key == PANEL_KEY:
            from . import panel

            panel.dock(widget)
            return
        old = getattr(self, "_windows", {}).get(key)
        if old is not None and old is not widget:
            try:
                old.close()
            except RuntimeError:
                pass
        self._windows = {**getattr(self, "_windows", {}), key: widget}
        parent = self.main_window()
        if parent is not None:
            widget.setParent(parent, QtCore.Qt.Window | QtCore.Qt.Tool)
        widget.setWindowTitle(title)
        widget.show()
        widget.raise_()

    def on_selection_changed(self, callback):
        """Nuke has no selection callback (knobChanged only runs while a node's panel is open): the selection's names
        are compared every 400 ms while the panel listens, and `callback()` runs when they changed."""
        if not self.gui:
            return None
        from lab2shot_dcc.guard import guarded
        from lab2shot_dcc.qt import QtCore

        timer = QtCore.QTimer()
        timer.setInterval(400)
        seen = {"names": self._selected_names()}

        def tick():
            names = self._selected_names()
            if names != seen["names"]:
                seen["names"] = names
                callback()

        timer.timeout.connect(guarded(tick, what="selection poll"))
        timer.start()
        return timer

    def off_selection_changed(self, handle) -> None:
        if handle is not None:
            handle.stop()

    @staticmethod
    def _selected_names() -> tuple:
        return tuple(n.fullName() for n in nuke.selectedNodes())

    def confirm(self, question: str) -> bool:
        return bool(nuke.ask(question)) if self.gui else True

    # ---- reading the script (never changing it)

    def _selected_user_nodes(self) -> list:
        return [n for n in nuke.selectedNodes() if not _ours(n) and n.knob(STATE) is None]

    def selection_types(self) -> list[str]:
        out = []
        for n in self._selected_user_nodes():
            for t in reader.types_of(n):
                if t not in out:
                    out.append(t)
        return out

    def bind(self, data_type: str, kinds: list[str]):
        from lab2shot_dcc.catalog import fits

        for n in self._selected_user_nodes():
            for t in reader.types_of(n):
                if fits(data_type, t):
                    return self._binding(n, t)
        return None

    def why_not(self, data_type: str) -> str:
        for n in self._selected_user_nodes():
            if n.Class() == "Read" and reader.read_offset(n) is None and data_type.split("|")[0].startswith(("image", "video")):
                return _say("dcc.nuke.frame_expression", node=n.fullName())
        words = {"image": "dcc.nuke.want.image", "video": "dcc.nuke.want.video", "scene.camera": "dcc.nuke.want.camera",
                 "scene.points": "dcc.nuke.want.points", "scene.gaussian": "dcc.nuke.want.points",
                 "scene": "dcc.nuke.want.scene"}
        key = next((w for t, w in words.items() if any(a.startswith(t) for a in data_type.split("|"))), None)
        return _say("dcc.nuke.select_first", what=_say(key) if key else data_type)

    def _binding(self, node, data_type: str):
        base = {"ref": "node:" + node.fullName(), "path": node.fullName(), "fingerprint": reader.fingerprint(node),
                "type": data_type, "label": node.fullName()}
        if data_type in ("image", "video"):
            pic = reader.picture(node)
            if pic["frame_offset"] is None:
                return None  # why_not says why
            label = _say("dcc.nuke.picture_label", node=node.fullName(), file=os.path.basename(pic["pattern"]))
            return {**base, "file": pic["file"], "sequence": pic["sequence"], "frame_offset": pic["frame_offset"],
                    "colorspace": pic["colorspace"], "label": label}
        if node.Class() in reader.GEO_FILES:
            # a 3D file node: its file goes up as it is (nothing to export)
            return {**base, "file": reader.file_of(node), "label": _say("dcc.nuke.picture_label", node=node.fullName(),
                                                                        file=os.path.basename(reader.file_of(node)))}
        return base

    def describe_binding(self, binding: dict) -> str:
        if binding.get("file") and not binding.get("ref"):
            return binding["file"]
        node = _find_user(binding)
        if node is None:
            return _say("dcc.nuke.binding_gone", label=binding.get("label", ""))
        if binding.get("file"):
            return _say("dcc.nuke.picture_label", node=node.fullName(), file=os.path.basename(binding["file"]))
        return node.fullName()

    def binding_alive(self, binding: dict) -> bool:
        if binding.get("ref"):
            return _find_user(binding) is not None
        return os.path.exists(binding.get("file", ""))

    # ---- export (reading only)

    def export(self, binding: dict, folder: str) -> str:
        """Main thread: what the bound node is, read (nothing written); the camera's frames are read in short steps
        by finish_export and written as a USD camera there (lab2shot_dcc.usd_camera: no Nuke needed)."""
        node = _find_user(binding)
        if node is None:
            raise RuntimeError(_say("dcc.nuke.bound_gone", label=binding.get("label", "")))
        if binding.get("type") != "scene.camera" or node.Class() not in reader.CAMERAS:
            raise RuntimeError(_say("dcc.nuke.cannot_export", kind=binding.get("type", "")))
        first, last = self.frame_range()
        out = safety.free_path(folder, safety.clean(node.name(), 40, "camera") + ".usda")
        self._pending[out] = {"path": node.fullName(), "binding": binding, "range": [first, last],
                              "offset": int(binding.get("plate_offset") or 0), "fps": self.fps(),
                              "near_far": reader.near_far(node), "name": node.name()}
        return out

    def finish_export(self, binding: dict, prepared: str, cancelled) -> str:
        """Background thread: the camera sampled in short main-thread steps (lab2shot_dcc.mainthread.sample), then
        written."""
        from lab2shot_dcc import mainthread, usd_camera

        job = self._pending.pop(prepared)
        first, last = job["range"]

        def read_frames(frames):
            node = _find_user(job["binding"])
            if node is None:
                raise RuntimeError(_say("dcc.nuke.bound_gone", label=job["binding"].get("label", "")))
            return reader.camera_samples(node, frames, job["offset"])

        samples = mainthread.sample(self, first, last, read_frames, cancelled) or []
        return usd_camera.write(prepared, job["name"], samples, job["fps"], job["near_far"])

    # ---- what the scene's context is made of (lab2shot_dcc.context.scene_context puts it together)

    def frame_range(self) -> tuple[int, int]:
        root = nuke.root()
        return int(root["first_frame"].value()), int(root["last_frame"].value())

    def fps(self) -> float:
        return float(nuke.root()["fps"].value())

    def linear_unit(self) -> str:
        """Centimetres: Nuke's 3D has no unit of its own, and what it brings in (GeoImport's USD numbers, a delivered
        .nk camera) is in centimetre numbers (this module's docstring: one space)."""
        return "cm"

    def camera_lens(self, binding: dict) -> tuple[float, float] | None:
        """The bound Camera's focal length and horizontal aperture (both mm in Nuke)."""
        cam = _find_user(binding)
        if cam is None or cam.Class() not in reader.CAMERAS:
            return None
        return float(cam["focal"].value()), float(cam["haperture"].value())

    def can_import(self, data_type: str, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in IMAGES + GEO + POINT_FILES + (".nk",):
            return ""
        return _say("dcc.nuke.kept_as_file", ext=ext or os.path.basename(path))

    # ---- adding to the script

    def one_undo_step(self, label: str, fn):
        """Everything one version adds, and the node's record of it, as one undo step (nuke.Undo); when it fails
        half way, what it made is removed again before the error goes on."""
        self._made, self._in_step = [], True
        nuke.Undo.begin(label)
        try:
            return fn()
        except Exception:
            for n in reversed(self._made or []):
                try:
                    nuke.delete(n)
                except Exception:  # noqa: BLE001 - gone already
                    pass
            raise
        finally:
            nuke.Undo.end()
            self._made, self._in_step = None, False

    def name_taken(self, name: str) -> bool:
        return nuke.toNode(name) is not None

    def _new(self, cls: str, name: str, info: dict | None = None):
        """One node of ours at the root of the script: named, marked, wired to nothing, nothing selected."""
        with nuke.root():
            node = getattr(nuke.nodes, cls)()
        if self._made is not None:
            self._made.append(node)
        node.setName(safety.numbered(name, self.name_taken, safety.NODE_MOST), uncollide=True)
        node.addKnob(_hidden_string(REF, uuid.uuid4().hex))
        if info:
            node.addKnob(_hidden_string(INFO, _b64(info)))
        return node

    def _free_x(self) -> int:
        nodes = nuke.allNodes()
        return max((n.xpos() + _width(n) for n in nodes), default=0) + GAP

    def make_group(self, name: str, info: dict) -> str:
        """A Backdrop for one version, to the right of everything in the script, at the height of its Lab2Shot node."""
        lab = nuke.toNode(str(info.get("node") or ""))
        y = lab.ypos() if lab is not None else min((n.ypos() for n in nuke.allNodes()), default=0)
        x = self._free_x()
        bd = self._new("BackdropNode", name, info)
        bd["label"].setValue(f"{info.get('tool', '')} v{int(info.get('version') or 0):03d}".strip())
        bd.setXYpos(x, y)
        bd["bdwidth"].setValue(GAP + 40)
        bd["bdheight"].setValue(160)
        ref = _ours(bd)
        self._layout[ref] = 0
        return ref

    def _place(self, group: str, node) -> None:
        bd = _find_ours(group)
        if bd is None:
            return
        i = self._layout.get(group, 0)
        self._layout[group] = i + 1
        node.setXYpos(bd.xpos() + 30 + i * GAP, bd.ypos() + 60)
        bd["bdwidth"].setValue(max(bd["bdwidth"].value(), 60 + (i + 1) * GAP))

    def import_result(self, item: dict, path: str, group: str, namespace: str) -> list[str]:
        """The delivered file as new nodes in the version's Backdrop, named `<namespace>_<name>`; the Root's settings
        checked afterwards (an import that changed them fails, and what it made is removed: one_undo_step)."""
        before = _root_settings()
        offset = int((item.get("plate") or {}).get("frame_offset") or 0)
        ext = os.path.splitext(path)[1].lower()
        base = f"{namespace}_{safety.clean(item.get('name') or os.path.splitext(os.path.basename(path))[0], 40, 'item')}"
        info = {"name": item.get("name", ""), "item": item.get("item", ""), "type": item.get("type", ""),
                "file": path.replace("\\", "/")}
        if ext == ".nk":
            made = self._import_nk(path, base, info, offset)
        elif ext in GEO:
            made = [self._geo(path, base, info, offset)]
        elif ext in POINT_FILES:
            node = self._new("GeoReference", base, info)
            node["file_path"].setValue(path.replace("\\", "/"))
            made = [node]
        elif ext in IMAGES:
            made = self._pictures(item, path, base, info, offset)
        else:
            raise RuntimeError(self.can_import(item.get("type", ""), path))
        for n in made:
            self._place(group, n)
        after = _root_settings()
        if after != before:
            changed = sorted(k for k in after if after.get(k) != before.get(k))
            raise RuntimeError(_say("dcc.nuke.settings_changed", knobs=", ".join(changed)))
        return [_ours(n) for n in made]

    def _import_nk(self, path: str, base: str, info: dict, offset: int) -> list:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        made = []
        for b in nk.blocks(text):
            if not hasattr(nuke.nodes, b["class"]):
                raise RuntimeError(_say("dcc.nuke.unknown_class", node_class=b["delivered"]))
            node = self._new(b["class"], base if not made else f"{base}_{len(made) + 1}", {**info, "node": b["name"]})
            node.readKnobs(nk.shift_frames(b["knobs"], -offset))
            made.append(node)
        return made

    def _geo(self, path: str, base: str, info: dict, offset: int):
        node = self._new("GeoImport", base, info)
        node["file"].setValue(path.replace("\\", "/"))
        if offset:
            # GeoImport's time_offset T: Nuke frame F reads the file's time F - T; the file is keyed at the picture's
            # frames (F + offset)
            node["time_offset"].setValue(-offset)
        return node

    def _pictures(self, item: dict, path: str, base: str, info: dict, offset: int) -> list:
        files = sorted(f for f in item.get("files") or [path] if os.path.splitext(f)[1].lower() == os.path.splitext(path)[1].lower())
        read = self._new("Read", base, info)
        sequence = bool(item.get("sequence")) and len(files) > 1
        if sequence:
            stem, ext = os.path.splitext(os.path.basename(path))
            digits = stem.rsplit(".", 1)[-1] if "." in stem else ""
            head = stem[: len(stem) - len(digits)]
            numbers = sorted(int(os.path.splitext(os.path.basename(f))[0].rsplit(".", 1)[-1]) for f in files
                             if os.path.basename(f).startswith(head))
            pattern = os.path.join(os.path.dirname(path), f"{head}{'#' * len(digits)}{ext}").replace("\\", "/")
            # setValue, never fromUserText: typing a file into a Read stretches the Root's frame range to it (and may add
            # a format to the script) — the user's settings (measured: 1-40 became 1-1040)
            read["file"].setValue(pattern)
            read["first"].setValue(numbers[0])
            read["last"].setValue(numbers[-1])
            read["origfirst"].setValue(numbers[0])
            read["origlast"].setValue(numbers[-1])
            if offset:
                read["frame_mode"].setValue("offset")
                read["frame"].setValue(str(offset))
        else:
            read["file"].setValue(path.replace("\\", "/"))
            if path.lower().endswith(reader.VIDEO):
                # a movie's frames as the Read finds them in the file (its knobs stay 1-1 with setValue; fromUserText
                # would stretch the Root's range), its first frame at the job's first frame: the Root's first frame
                # when it was cooked (the job's range is the Root's: scene_context)
                first, last = int(read.firstFrame()), int(read.lastFrame())
                for knob, value in (("first", first), ("last", last), ("origfirst", first), ("origlast", last)):
                    read[knob].setValue(value)
                read["frame_mode"].setValue("start at")
                read["frame"].setValue(str(int(nuke.root()["first_frame"].value())))
        head = exr.header(path) if path.lower().endswith(".exr") else {}
        layers = exr.layer_names(head.get("channels") or []) if head else ["rgba"]
        meta = head.get("layers") or {}
        # data (depth, ST maps, masks: no colour conversion, as the server wrote them) when it is a data type or has no
        # picture in it; else the pictures' colour space, when this script's colour management knows it by that name
        if item.get("type") in DATA_TYPES or "rgba" not in layers:
            read["raw"].setValue(True)
        else:
            from lab2shot_dcc.results import colorspace_for

            wanted = str((meta.get("rgba") or {}).get("colorspace") or item.get("colorspace") or "")
            # the knob lists "name<TAB>menu path"; its value is the name
            match = colorspace_for(wanted, [str(v).split("\t", 1)[0] for v in read["colorspace"].values()],
                                   self.colorspaces)
            if match:
                read["colorspace"].setValue(match)
        made = [read]
        # ST maps: the layers the server marks with a direction (lab2shot:layers), or a whole two-channel picture
        stmaps = [(name, str(m.get("direction") or "")) for name, m in meta.items() if isinstance(m, dict) and m.get("direction")]
        if not stmaps and item.get("type") == "image.2":
            stmaps = [("rgb", "")]
        plate = self._user_read_of((item.get("plate") or {}).get("file", ""))
        if stmaps:
            read.channels()  # the file's header read: its layers known to Nuke before a knob names one
        for layer, direction in stmaps:
            st = self._new("STMap", f"{base}_{safety.clean(layer, 30, 'stmap')}", {**info, "layer": layer,
                                                                                 "direction": direction})
            st.setInput(1, read)  # 1: stmap (0: src; measured)
            st["uv"].setValue(layer)
            if plate is not None and direction != "distort":
                st.setInput(0, plate)  # our node's input; the user's Read stays as it is
            made.append(st)
        return made

    def _user_read_of(self, file: str):
        if not file:
            return None
        want = os.path.normcase(os.path.abspath(file))
        for n in nuke.allNodes("Read"):
            if _ours(n):
                continue
            pic = reader.picture(n)
            if pic["file"] and os.path.normcase(os.path.abspath(pic["file"])) == want:
                return n
        return None

    def locate(self, refs: list[str]) -> None:
        """Frame a version's nodes in the node graph and open its Backdrop's panel; nothing is selected (selecting
        writes `selected` on the user's nodes, and the user's selection is theirs)."""
        nodes = [n for n in (_find_ours(r) for r in refs) if n is not None]
        if not nodes or not self.gui:
            return
        bd = next((n for n in nodes if n.Class() == "BackdropNode"), None)
        box = [bd] if bd is not None else nodes
        x0 = min(n.xpos() for n in box)
        y0 = min(n.ypos() for n in box)
        x1 = max(n.xpos() + (int(n["bdwidth"].value()) if n.Class() == "BackdropNode" else _width(n)) for n in box)
        y1 = max(n.ypos() + (int(n["bdheight"].value()) if n.Class() == "BackdropNode" else _height(n)) for n in box)
        nuke.zoom(1.0, [(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        if bd is not None:
            nuke.show(bd)

    def unsaved(self) -> bool:
        """The script has never been saved (Untitled): nothing is computed for it (Plugin.why_not_saved)."""
        name = nuke.root().name()
        return not name or name == "Root" or not os.path.isabs(name)

    def project_dir(self) -> str:
        """The project directory (Project Settings) when it is set and there, else the saved script's folder."""
        root = nuke.root()
        try:
            project = root["project_directory"].evaluate() if root["project_directory"].value() else ""
        except Exception:  # noqa: BLE001
            project = ""
        if project and os.path.isdir(project):
            return project
        if self.unsaved():
            raise RuntimeError(_say("dcc.plugin.save_first"))
        return os.path.dirname(root.name())

    # ---- the Lab2Shot node

    def nodes(self) -> list[str]:
        return [_ours(n) for n in nuke.allNodes(NODE_CLASS) if n.knob(STATE) is not None and _ours(n)]

    def node_name(self, node: str) -> str:
        found = _find_ours(node)
        return found.name() if found is not None else node

    def create_node(self, name: str) -> str:
        nodes = nuke.allNodes()
        x = max((n.xpos() + _width(n) for n in nodes), default=0) + GAP
        y = min((n.ypos() for n in nodes), default=0)
        with self._quiet():
            node = self._new(NODE_CLASS, name)
            node.setXYpos(x, y)
            node.addKnob(nuke.Tab_Knob(TAB, "Lab2Shot"))
            node.addKnob(nuke.Text_Knob(SUMMARY, "", ""))
            button = nuke.PyScript_Knob(OPEN, _say("dcc.menu.open_panel"),
                                        "try:\n    import lab2shot_nuke.menu\n    lab2shot_nuke.menu.open_panel()\n"
                                        "except Exception:\n    pass\n")
            node.addKnob(button)
            node.addKnob(_hidden_string(STATE, _b64({})))
            node["tile_color"].setValue(0x3E8EF7FF)
        return _ours(node)

    @contextlib.contextmanager
    def _quiet(self):
        """The plugin's own bookkeeping stays out of the user's undo queue (except inside one version's undo step,
        whose record of the version must go with it)."""
        if self._in_step:
            yield
            return
        nuke.Undo.disable()
        try:
            yield
        finally:
            nuke.Undo.enable()

    def store(self, node: str, state: dict) -> None:
        found = _find_ours(node)
        if found is None or found.knob(STATE) is None:
            raise RuntimeError(_say("dcc.nuke.node_gone"))
        with self._quiet():
            found[STATE].setValue(_b64(state))
            found[SUMMARY].setValue(self._summary(state))

    @staticmethod
    def _summary(state: dict) -> str:
        """The node's one line of text in the node's panel: the tool, the last job and its state (as the queue says
        a state)."""
        tool = str((state.get("tool") or {}).get("name") or "")
        job = state.get("job") or {}
        said = ""
        if job.get("state"):
            key = f"dcc.queue.state.{job['state']}"
            said = _say(key) if job["state"] in ("queued", "running", "done", "failed", "cancelled") else str(job["state"])
        versions = state.get("versions") or []
        last = f"v{int(versions[-1].get('version') or 0):03d}" if versions else ""
        return "  ".join(x for x in (tool, str(job.get("id") or ""), said, last) if x)

    def load(self, node: str) -> dict:
        found = _find_ours(node)
        if found is None or found.knob(STATE) is None:
            return {}
        try:
            got = _unb64(found[STATE].value())
            return got if isinstance(got, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def selected_node(self) -> str | None:
        for n in nuke.selectedNodes():
            if n.knob(STATE) is not None and _ours(n):
                return _ours(n)
        return None
