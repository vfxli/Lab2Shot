"""A host with no DCC behind it: a dict for a scene, files for objects, a queue for the main thread.

It proves the framework (lab2shot_dcc) needs nothing of any DCC: the whole job — bind, export, upload, submit, follow,
fetch, import — runs through it in plain Python. `pump()` is the "main loop": the test calls it until the job ends.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import threading
import uuid

from lab2shot_dcc.host import Host


class FakeHost(Host):
    name = "fakehost"
    label = "Fake"
    formats = {"camera": "fbx", "skeleton": "fbx", "character": "fbx"}

    def __init__(self, project: str):
        self.project = project
        self.main_thread = threading.current_thread()
        self.calls: "queue.Queue" = queue.Queue()
        self.scene: dict[str, dict] = {}  # ref -> {"name", "type", "file"?, "group"?, "namespace"?, "state"?}
        self.selected: list[str] = []
        self.undo_steps: list[list[str]] = []
        self._open_chunk: list[str] | None = None
        self.context = {"fps": 24.0, "unit": "cm", "range": (1, 10)}  # what the scene's context is made of
        self.off_main_calls = 0

    # ---- main thread

    def run_on_main(self, fn) -> None:
        self.calls.put(fn)

    def pump(self, seconds: float = 0.05) -> None:
        try:
            fn = self.calls.get(timeout=seconds)
        except queue.Empty:
            return
        fn()

    def _main_only(self) -> None:
        if threading.current_thread() is not self.main_thread:
            self.off_main_calls += 1

    # ---- the scene

    def add_object(self, name: str, data_type: str, file: str = "") -> str:
        ref = uuid.uuid4().hex
        self.scene[ref] = {"name": name, "type": data_type, "file": file}
        return ref

    def selection_types(self):
        self._main_only()
        return [self.scene[r]["type"] for r in self.selected if r in self.scene]

    def bind(self, data_type, kinds):
        self._main_only()
        for r in self.selected:
            obj = self.scene.get(r)
            if obj and any(obj["type"] == t or obj["type"].startswith(t + ".") or t.startswith(obj["type"]) for t in data_type.split("|")):
                return {"ref": r, "label": obj["name"], "type": obj["type"]}
        return None

    def why_not(self, data_type):
        return f"选中的不是 {data_type}"

    def describe_binding(self, binding):
        obj = self.scene.get(binding.get("ref", ""))
        return obj["name"] if obj else str(binding.get("file") or "")

    def export(self, binding, folder):
        self._main_only()
        obj = self.scene[binding["ref"]]
        target = os.path.join(folder, os.path.basename(obj["file"]))
        shutil.copyfile(obj["file"], target)
        return target

    def frame_range(self):
        self._main_only()
        return self.context.get("range")

    def fps(self):
        self._main_only()
        return self.context.get("fps")

    def linear_unit(self):
        self._main_only()
        return self.context.get("unit", "")

    def camera_lens(self, binding):
        """A camera object's "lens": (focal mm, filmback mm), set by a test."""
        self._main_only()
        return (self.scene.get(binding.get("ref", "")) or {}).get("lens")

    @contextlib.contextmanager
    def undo_chunk(self, label):
        self._main_only()
        self._open_chunk = []
        try:
            yield
        finally:
            self.undo_steps.append(self._open_chunk)
            self._open_chunk = None

    def _made(self, ref):
        if self._open_chunk is not None:
            self._open_chunk.append(ref)

    def name_taken(self, name):
        return any(o.get("name") == name or o.get("namespace") == name for o in self.scene.values())

    def make_group(self, name, info):
        self._main_only()
        ref = self.add_object(name, "group")
        self.scene[ref]["info"] = info
        self._made(ref)
        return ref

    def import_result(self, item, path, group, namespace):
        self._main_only()
        assert os.path.isfile(path), path
        ref = self.add_object(f"{namespace}:{os.path.basename(path)}", item.get("type") or "file", path)
        self.scene[ref].update(group=group, namespace=namespace, kinds=item.get("kinds"))
        self._made(ref)
        return [ref]

    def camera_picture(self, binding):
        self._main_only()
        return (self.scene.get(binding.get("ref", "")) or {}).get("picture")

    def cameras_in(self, refs):
        self._main_only()
        return [r for r in refs if r in self.scene and ("camera" in (self.scene[r].get("kinds") or [])
                                                        or self.scene[r]["type"] == "scene.camera")]

    def hang_picture(self, camera, picture, namespace):
        self._main_only()
        ref = self.add_object(f"{namespace}:backplate", "backplate")
        self.scene[ref].update(camera=camera, picture=dict(picture), namespace=namespace)
        self._made(ref)
        return [ref]

    def undo(self):
        for ref in self.undo_steps.pop():
            self.scene.pop(ref, None)

    unsaved_scene = False  # a test sets it: the scene never saved

    def unsaved(self):
        return self.unsaved_scene

    def project_dir(self):
        self._main_only()
        return self.project

    # ---- nodes

    def nodes(self):
        return [r for r, o in self.scene.items() if o["type"] == "lab2shot"]

    def node_name(self, node):
        return self.scene[node]["name"]

    def create_node(self, name):
        ref = self.add_object(name, "lab2shot")
        self.scene[ref]["state"] = {}
        return ref

    def store(self, node, state):
        self._main_only()
        import json

        self.scene[node]["state"] = json.loads(json.dumps(state))

    def load(self, node):
        import json

        return json.loads(json.dumps(self.scene[node].get("state") or {}))

    def confirm(self, text):
        return True
