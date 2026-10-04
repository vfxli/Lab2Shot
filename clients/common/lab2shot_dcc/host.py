"""The host interface: the only thing a DCC implements (clients/<dcc>/), as thin as it can be.

The framework works in data types (lab2shot's: image, video, scene.camera, scene.character, scene.skeleton,
scene.model …) and in the fixed outside names of templates/_conventions.md; a host turns its own objects into files of
those types and files of those types into new objects of its own. It never knows a tool.

Threads: the framework calls a host only on the DCC's main thread (through `run_on_main`), except `run_on_main` itself,
which any thread may call and which must return at once (it queues; it never waits).

References: a host names its objects by `ref` strings of its own making that survive a rename and a re-parent (Maya:
the node's UUID); the framework stores them and hands them back.
"""

from __future__ import annotations

import contextlib


class Binding(dict):
    """What an input is bound to: {"ref": a host object (exported when the job is submitted), or "file": a file on
    this machine (uploaded as it is); "label": what the panel shows; "type": the data type; and whatever the host
    keeps about it (a picture's "frame_offset", "sequence")}."""


class Host:
    """Implemented once per DCC. Every method may raise; the framework catches and reports (lab2shot_dcc.guard)."""

    name = "host"  # short, for the log file and the job's application
    label = "Host"
    # delivery format per kind of 3D data (data/types.py SCENE_KINDS ids → a format name of GET /api/tools "formats"):
    # data, one table per DCC, never a branch in code
    formats: dict = {}
    # what this DCC most wants back, data type families first to last (lab2shot_dcc.catalog.preferred: the tool order
    # after what takes the selection): data, one table per DCC, never a branch in code
    prefers: tuple = ("scene",)
    # the file an object of each data type becomes when export writes it (its suffix: Maya ".fbx", Nuke's camera
    # ".usda"): an input reads only the suffixes its signature `accept`s, so a selected camera goes into the input
    # that reads what this DCC writes (cam_usd_path, not cam_fbx_path). Data, one table per DCC; a type not listed is
    # uploaded as the file it already is (the binding's "file")
    exports: dict = {}
    # a colour space as the server names it (its OCIO configuration's: the delivery manifest's `colorspace`) -> the
    # names the same space has in this DCC's configurations, tried in order (lab2shot_dcc.results.colorspace_for);
    # one not found is left unset and logged, never guessed. Data, one table per DCC
    colorspaces: dict = {}

    # ---- language

    def ui_language(self) -> str:
        """The DCC's own interface language, as it names it (Maya: about -uiLanguage, e.g. "zh_CN", "en_US"); "" when
        it does not say. The plugin speaks Chinese when this starts with zh, else English, unless its user chose one
        (lab2shot_dcc.connection.language)."""
        return ""

    # ---- threads and windows

    def run_on_main(self, fn) -> None:
        """Queue `fn()` to run on the main thread when the DCC is idle; return at once."""
        raise NotImplementedError

    def main_window(self):
        """The DCC's main Qt window (dialogs and the panel hang under it), or None."""
        return None

    def show_window(self, widget, title: str, key: str) -> None:
        """Show one of the framework's windows (a QWidget: the panel, the embedded web page) docked in the DCC, one
        per `key` (showing another with the same key replaces it). Default: a tool window."""
        widget.setWindowTitle(title)
        widget.show()

    def close_window(self, widget) -> None:
        """Close a window show_window showed."""
        widget.close()

    def show_panel(self, widget, title: str) -> None:
        self.show_window(widget, title, "Lab2ShotPanelControl")

    def is_exiting(self) -> bool:
        """The DCC has started quitting (nothing more is queued into it). Default: never."""
        return False

    def on_exit(self, callback) -> None:
        """`callback()` once when the DCC starts quitting (the plugin lets its background threads go; jobs go on on
        the server). Default: never called."""

    def on_selection_changed(self, callback):
        """Call `callback()` (guarded, throttled by the panel) when the selection changes; returns a handle for
        `off_selection_changed`. Default: never."""
        return None

    def off_selection_changed(self, handle) -> None:
        pass

    # ---- reading the scene (never changing it)

    def selection_types(self) -> list[str]:
        """The data types of what is selected now (a camera with an image plane: scene.camera and image)."""
        return []

    def bind(self, data_type: str, kinds: list[str]) -> Binding | None:
        """The selection as an input of this type, or None when nothing selected is of it (`why_not` says why)."""
        return None

    def why_not(self, data_type: str) -> str:
        return ""

    def describe_binding(self, binding: dict) -> str:
        """What the panel shows for a binding (the object's current name: it may have been renamed)."""
        return str(binding.get("label") or binding.get("file") or binding.get("ref") or "")

    def binding_alive(self, binding: dict) -> bool:
        """The bound object is still in the scene, found by its ref or, failing that, the path it had when bound (a
        file binding: the file exists). Nothing is ever connected to the user's object to keep track of it."""
        return True

    def export(self, binding: dict, folder: str) -> str:
        """Write the bound object to a file in `folder` (only reading the user's objects; anything temporary made
        and removed whatever happens); returns the file. `binding["plate_offset"]`: animation is written at the bound
        picture's frame numbers (the DCC's frame + plate_offset), the frames the server works in; a result comes back
        in them and is put back at the DCC's frames (import_result: item["plate"]["frame_offset"])."""
        raise NotImplementedError

    def finish_export(self, binding: dict, prepared: str, cancelled) -> str:
        """Background thread, after export: whatever turns what export wrote on the main thread into the file the
        tool takes, outside the DCC's scene (a DCC that can only make the file in a separate process does it here;
        `cancelled()` says to stop). Default: export already wrote the file."""
        return prepared

    def prepare_import(self, item: dict, path: str, cancelled) -> str:
        """Background thread, before import_result: whatever makes a delivered file ready to come into the scene
        without touching anything already there (Maya converts it in a process of its own); returns the file
        import_result takes. Default: the file itself."""
        return path

    # what the scene's context is made of (lab2shot_dcc.context.scene_context puts it together, one rule for every
    # DCC: the picture's frame offset, its colour space, which camera): the host only reads these

    def frame_range(self) -> tuple[int, int] | None:
        """The DCC's frame range now (its own frame numbers: first, last), None when it has none."""
        return None

    def fps(self) -> float | None:
        """The scene's frame rate, None when it does not say."""
        return None

    def linear_unit(self) -> str:
        """The scene's linear unit as the tools' `unit` names it ("cm", "m"), "" when it is none of those."""
        return ""

    def camera_lens(self, binding: dict) -> tuple[float, float] | None:
        """A bound camera's (`binding`: its ref, or what the host keeps about it) focal length and horizontal
        filmback, both in mm, read now; None when it is not found or is no camera."""
        return None

    def camera_picture(self, binding: dict) -> dict | None:
        """The picture a bound camera carries itself (Maya: its image plane), read only: {file, sequence, frame_offset
        (the picture's frame = the DCC's frame + it), colorspace}, or None when it has none. The framework decides
        whether it is used (jobs.backplate). Default: a camera carries no picture."""
        return None

    def can_import(self, data_type: str, path: str) -> str:
        """"" when this file of this type can be brought in; else why not (a missing importer)."""
        return ""

    # ---- changing the scene (only adding)

    def undo_chunk(self, label: str):
        """A context manager: everything inside is one undo step (for a DCC whose every change can be undone)."""
        return contextlib.nullcontext()

    def one_undo_step(self, label: str, fn):
        """Run `fn()` (everything one result version adds to the scene, and the node's record of it) as ONE undo step;
        return what it returns. Default: inside undo_chunk. A DCC whose importers cannot be undone runs it in a
        command of its own that removes what it made on undo and may call `fn` again on redo (Maya)."""
        with self.undo_chunk(label):
            return fn()

    def make_group(self, name: str, info: dict) -> str:
        """A new, empty group for one result version under the plugin's own top group; returns its ref. `name` is
        clean and free (the framework numbered it); `info` the original names to keep on it."""
        raise NotImplementedError

    def name_taken(self, name: str) -> bool:
        """A scene object or namespace of this name exists already."""
        return False

    def import_result(self, item: dict, path: str, group: str, namespace: str) -> list[str]:
        """Bring the file of one delivered item in as new objects under `group`, inside `namespace`, scene units / up
        axis / frame rate left as they are (checked afterwards); returns the new objects' refs. `item`: its type,
        kinds, original name, the picture's frame offset …"""
        raise NotImplementedError

    def cameras_in(self, refs: list[str]) -> list[str]:
        """The cameras among (or under) objects import_result just made, as refs hang_picture takes. Default: none."""
        return []

    def hang_picture(self, camera: str, picture: dict, namespace: str) -> list[str]:
        """Hang `picture` (jobs.backplate: file, sequence, frame_offset, colorspace) behind one of OUR new cameras
        (cameras_in), as a new object of our own inside `namespace`; returns the new objects' refs. Which picture, and
        whether any, is the framework's rule, never the host's. Default: a DCC without backplates makes nothing."""
        return []

    def select(self, refs: list[str]) -> None:
        pass

    def locate(self, refs: list[str]) -> None:
        """Show the user these objects of a result version (the panel's version card): default, select them. A DCC
        where selecting would write to the user's own objects shows them another way (Nuke: frames them in the node
        graph)."""
        self.select(refs)

    def project_dir(self) -> str:
        """The DCC project folder results go under (data/lab2shot/…)."""
        raise NotImplementedError

    def unsaved(self) -> bool:
        """The scene (Nuke's script, Maya's scene file) has never been saved. Nothing is computed for an unsaved
        scene: its results go next to the saved scene, and there is none yet (Plugin.compute refuses, the panel
        greys 「计算」 and says to save first). Default: saved."""
        return False

    # ---- the Lab2Shot node: the plugin's state saved in the scene

    def nodes(self) -> list[str]:
        """Every Lab2Shot node of the scene (refs)."""
        return []

    def node_name(self, node: str) -> str:
        return node

    def create_node(self, name: str) -> str:
        raise NotImplementedError

    def store(self, node: str, state: dict) -> None:
        """Keep the node's state in the scene (saved with it)."""
        raise NotImplementedError

    def load(self, node: str) -> dict:
        return {}

    def selected_node(self) -> str | None:
        """The Lab2Shot node selected now, if any."""
        return None

    def confirm(self, text: str) -> bool:
        """Ask the user yes or no (the main thread): default no dialog, yes."""
        return True
