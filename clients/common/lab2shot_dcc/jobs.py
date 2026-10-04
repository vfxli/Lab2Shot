"""A job's life, the same for every tool and every DCC: export → upload → submit → follow → fetch → import.

Threads (安全守则「不卡前台」): one background (daemon) thread per job does everything that talks to the server or
touches files; whatever changes the scene — exporting the bound objects (temporary copies made and removed there),
importing the result, saving the node's state — is queued onto the DCC's main thread through the host's
`run_on_main`, one small step at a time. The main thread never waits for anything; the background thread waits for
its main-thread steps (and stops waiting when cancelled). A DCC that quits does not wait for the thread; the job goes
on on the server and is fetched next time from the job id the node keeps.

Limits: the server must answer a connection within 10 s (connection.CONNECT_S) before every step that talks to it.
Every step that talks to the server — connect, read the tool, upload, submit, follow, download — goes through the one
retry policy (`_again`, judged by `_again_worth`): a dropped line or a server restarting (no answer, 408 / 429 / 502
/ 503 / 504; never a 500, which is an answer) is tried again RECONNECT_TRIES times, waiting RECONNECT_WAIT in turn
and saying which try it is; then the run stops with a sentence that says what to press (the job id kept: 「取回结果」
fetches it later). Cancelling stops the upload or download
in the middle (the transport's cancel_event), cancels the job on the server, and removes what was half written (the staging
folder, the exported files): nothing half done is left.

Progress is read by the panel (`snapshot`), never pushed into it: the panel refreshes on its own timer, throttled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid

from . import connection, contract, log, mainthread, paths, results, safety
from .connection import Cancelled
from .context import scene_context
from .paths import text

POLL_S = 1.0
# every other step that talks to the server (connect, read the tool, upload, submit, download): a dropped line or a
# server restarting (no answer, 502/503/504, "服务器正在重启") is tried again this often, waiting RECONNECT_WAIT
# seconds in turn, the panel saying which try it is; past the last, the run stops and says so (nothing waits forever)
RECONNECT_TRIES = 5
RECONNECT_WAIT = (2, 4, 8, 15, 30)


class Unreachable(RuntimeError):
    status = 0  # no answer: worth trying again


def _again_worth(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    import http.client
    import socket

    return isinstance(exc, (ConnectionError, socket.timeout, TimeoutError, http.client.HTTPException)) \
        or status in (0, 408, 429, 502, 503, 504)


ASK_ABOVE = 500 << 20  # bytes of results: ask before importing more (安全守则)
TERMINAL = ("done", "failed", "cancelled")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Run:
    """One job of one Lab2Shot node. `start()` it from the main thread; `cancel()` any time."""

    def __init__(self, host, conn, node: str, tool: dict | None, job: str = "", ask=None, colorspaces=None,
                 version: int = 0):
        self.host, self.conn, self.node, self.tool = host, conn, node, tool
        self.colorspaces = list(colorspaces or [])
        self.job = job  # given: fetch an earlier job's results (B5); else submitted by this run
        self.version = version  # given: import that version's files, already here (Plugin.import_version), nothing fetched
        self.ask = ask or host.confirm
        self._lock = threading.Lock()
        self._status = {"phase": "starting", "text": text("dcc.job.starting"), "fraction": None, "error": "", "done": False,
                        "version": None}
        self._cancel = threading.Event()
        self._lab = None
        self._thread: threading.Thread | None = None
        self._exports = ""
        self._stage = ""

    # ---- what the panel reads

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._status, job=self.job)

    def _set(self, **changes) -> None:
        with self._lock:
            self._status.update(changes)
        if "text" in changes or "error" in changes:
            log.get().info("[%s] %s %s", self.host.node_name(self.node), changes.get("text", ""), changes.get("error", ""))

    @property
    def finished(self) -> bool:
        return self.snapshot()["done"]

    # ---- control

    def start(self) -> None:
        self._thread = threading.Thread(target=self._main, name=f"lab2shot-job-{self.node}", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        """Stop: the transfer in progress, the job on the server, what was half written. Any thread; returns at once."""
        if self._cancel.is_set() or self.finished:
            return
        self._cancel.set()
        self._set(text=text("dcc.job.cancelling"))
        if self._lab is not None:
            self._lab.cancel_event.set()
        if self.job:
            job = self.job

            def stop_on_server():
                try:
                    self.conn.session().cancel(job)
                except Exception:  # noqa: BLE001 - finished meanwhile, or the server away: the job ends on its own
                    log.get().info("cancelling job %s: the server did not take it: %s", job, traceback.format_exc(limit=1))

            threading.Thread(target=stop_on_server, name="lab2shot-cancel", daemon=True).start()

    def abandon(self) -> None:
        """The DCC is quitting: stop here (transfers stop at the next chunk, nothing more is asked of the DCC), leave
        the job on the server — fetched next time from the id the node keeps."""
        self._cancel.set()
        if self._lab is not None:
            self._lab.cancel_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        """For tests and scripts only (never the main thread of a DCC)."""
        if self._thread is not None:
            self._thread.join(timeout)
        return self.finished

    # ---- the background thread

    def _on_main(self, fn):
        """Run `fn` on the main thread and wait here (the background thread) for its answer (mainthread.call)."""
        return mainthread.call(self.host, fn, self._cancel.is_set)

    def _check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _main(self) -> None:
        try:
            self._run()
        except Cancelled:
            self._set(phase="cancelled", text=text("dcc.job.cancelled"), done=True)
            self._save_job(state="cancelled")
        except Exception as exc:  # noqa: BLE001 - said in the panel and the log, never raised into the DCC
            log.get().error("job failed: %s\n%s", exc, traceback.format_exc())
            self._set(phase="failed", text=text("dcc.job.failed"), error=str(exc) or type(exc).__name__, done=True)
            self._save_job(state="failed", error=str(exc))
        finally:
            results.discard(self._stage)
            if self._exports:
                shutil.rmtree(self._exports, ignore_errors=True)

    def _save_job(self, **fields) -> None:
        """Keep the job's id and state on the node (main thread); never raises. A run that ended before it submitted
        anything leaves the node's last job as it was."""
        if not self.job:
            return

        def save():
            state = self.host.load(self.node)
            state["job"] = {**(state.get("job") or {}), "id": self.job, "server": self.conn.server, **fields}
            self.host.store(self.node, state)
        try:
            self.host.run_on_main(lambda: _quiet(save))
        except Exception:  # noqa: BLE001
            log.get().warning("saving the job state failed: %s", traceback.format_exc())

    def _reach(self) -> None:
        why = connection.reachable(self.conn.server)
        if why:
            raise Unreachable(why)

    def _again(self, what: str, fn):
        """`fn()`, tried again RECONNECT_TRIES times when the server does not answer or is restarting (the module
        docstring); the panel says each try; past the last, a RuntimeError that says what did not happen."""
        for attempt in range(RECONNECT_TRIES + 1):
            self._check()
            try:
                self._reach()
                return fn()
            except Cancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - judged right here
                if not _again_worth(exc):
                    raise
                if attempt == RECONNECT_TRIES:
                    raise RuntimeError(text("dcc.job.gave_up_fetch", error=exc, step=text(what), job=self.job) if self.job
                                       else text("dcc.job.gave_up_cook", error=exc, step=text(what)))
                wait = RECONNECT_WAIT[min(attempt, len(RECONNECT_WAIT) - 1)]
                self._set(text=text("dcc.job.reconnect", wait=wait, attempt=attempt + 1, tries=RECONNECT_TRIES))
                log.get().info("%s: %s", what, exc)
                self._sleep(wait)

    def _run(self) -> None:
        if self.version:
            self._import_kept()
            return
        self._again("dcc.job.step.connect", lambda: None)
        self._lab = self.conn.session()
        if self._cancel.is_set():
            self._lab.cancel_event.set()
        if not self.job:
            self._submit()
        outputs = self._follow()
        self._fetch_and_import(outputs)

    # ---- submit

    def _submit(self) -> None:
        tool = self.tool
        state = self._on_main(lambda: self.host.load(self.node))
        bindings = state.get("bindings") or {}
        self._exports = tempfile.mkdtemp(prefix="export_", dir=_tmp_dir())
        files = {}
        for param, binding in bindings.items():
            if param not in contract.input_keys(tool):
                continue
            self._check()
            if binding.get("file"):
                for f in binding.get("files") or [binding["file"]]:
                    if not os.path.exists(f):
                        raise RuntimeError(text("dcc.job.file_gone", file=f))
                files[param] = binding.get("files") or binding["file"]
                continue
            self._set(phase="exporting", text=text("dcc.job.exporting", what=self.host.describe_binding(binding)))
            folder = os.path.join(self._exports, safety.clean(param, 40, "input"))
            os.makedirs(folder, exist_ok=True)
            # what is exported keeps time with the bound picture: the host writes keys at the picture's own frame
            # numbers (its frame = the DCC's frame + plate_offset), as the server reads the picture
            exported = {**binding, "plate_offset": int(_plate({"bindings": bindings}).get("frame_offset") or 0)}
            prepared = self._on_main(lambda b=exported, f=folder: self.host.export(b, f))
            files[param] = self.host.finish_export(exported, prepared, self._cancel.is_set)
        context = self._on_main(lambda: scene_context(self.host, bindings))
        values = contract.submission(tool, {**state, "scene_values": {**(state.get("scene_values") or {}),
                                                                      **contract.scene_fill(tool, context, self.colorspaces)},
                                            "files": files})
        lab = self._lab
        graph = tool.get("graph") or self._again("dcc.job.step.read_tool", lambda: lab.tool(tool["id"])["graph"])
        wanted = self._formats(tool)

        def told(event):
            if event.get("type") == "upload":
                self._set(phase="uploading", text=text("dcc.job.uploading", done=event.get("done"), total=event.get("total")),
                          fraction=float(event.get("done", 0)) / max(1, event.get("total", 1)))

        self._set(phase="uploading", text=text("dcc.job.step.upload"), fraction=None)
        prepared = self._again("dcc.job.step.upload", lambda: lab.prepare(graph, values, told))
        self._set(phase="uploading", text=text("dcc.job.step.submit"), fraction=None)
        # one key for this press of 「计算」: a submission sent again after a dropped line or a restart gets the job the
        # first one made, never a second job (POST /api/jobs `submit`)
        key = "l2s-" + uuid.uuid4().hex
        self.job = self._again("dcc.job.step.submit", lambda: lab.submit(graph, prepared, formats=wanted, key=key))
        self._set(phase="queued", text=text("dcc.job.submitted"), fraction=None)
        self._save_job(state="queued", tool=tool.get("id"), submitted=_now())

    def _formats(self, tool: dict) -> dict:
        """The host's format table, for the kinds the tool delivers."""
        kinds = {k for d in tool.get("delivers") or [] for k in d.get("kinds") or []}
        return {k: f for k, f in (self.host.formats or {}).items() if k in kinds}

    # ---- follow

    def _follow(self) -> list:
        since = 0
        while True:
            self._check()
            state = self._again("dcc.job.step.poll", lambda: self._lab.poll(self.job, since))
            since = state.get("next", since)
            for event in state.get("events") or []:
                self._told(event)
            if state.get("done"):
                if state.get("state") == "cancelled":
                    raise Cancelled()
                if state.get("error"):
                    raise RuntimeError(state["error"])
                return list(state.get("outputs") or [])
            self._sleep(POLL_S)

    def _sleep(self, seconds: float) -> None:
        if self._cancel.wait(seconds):
            raise Cancelled()

    def _told(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "queued":
            self._set(phase="queued", text=text("dcc.job.queued", position=event.get("position")))
        elif kind == "node_start":
            self._set(phase="running", text=text("dcc.job.running", node=event.get("label", "")))
        elif kind in ("message", "error") and event.get("text"):
            self._set(text=str(event["text"])[:300])
        elif kind == "progress" and event.get("fraction") is not None:
            self._set(fraction=float(event["fraction"]))

    # ---- fetch and import

    def _fetch_and_import(self, outputs: list) -> None:
        if not outputs:
            raise RuntimeError(text("dcc.job.no_outputs"))
        state = self._on_main(lambda: self.host.load(self.node))
        project = self._on_main(self.host.project_dir)
        node_name = self._on_main(lambda: self.host.node_name(self.node))
        base = results.node_folder(project, node_name)
        total = sum(int(o.get("bytes") or 0) for o in outputs)
        results.check_room(base, total)
        self._stage = results.staging(base)
        mapping: dict[str, str] = {}
        for i, o in enumerate(outputs):
            self._check()
            self._set(phase="fetching", text=text("dcc.job.downloading", index=i + 1, count=len(outputs), mb=int(o.get("bytes") or 0) / 1e6),
                      fraction=None)
            into = self._stage if len(outputs) == 1 else os.path.join(self._stage, safety.clean(o.get("node") or f"out{i}", 40, "out"))
            os.makedirs(into, exist_ok=True)
            archive = os.path.join(self._stage, "_l2s_download.zip")
            self._again("dcc.job.step.download", lambda o=o, a=archive: self._lab.fetch(o, a))
            self._check()
            self._set(text=text("dcc.job.unpacking"))
            got = results.unpack(archive, into, self._cancel, text("dcc.job.project", host=self.host.label))
            os.remove(archive)
            mapping.update({(f"{o.get('node')}/{k}" if len(outputs) > 1 else k): v for k, v in got.items()})
        self._check()
        known = [int(v.get("version") or 0) for v in state.get("versions") or []]
        number = results.next_version(base, known)
        version_dir = os.path.join(base, f"v{number:03d}")
        mapping = results.settle(self._stage, version_dir, mapping)
        self._stage = ""
        items = []
        for i, o in enumerate(outputs):
            folder = version_dir if len(outputs) == 1 else os.path.join(version_dir, safety.clean(o.get("node") or f"out{i}", 40, "out"))
            prefix = "" if len(outputs) == 1 else f"{o.get('node')}/"
            local = {k[len(prefix):]: v for k, v in mapping.items() if k.startswith(prefix)}
            delivers = (self.tool or {}).get("delivers") if self.tool else None
            items += results.items(results.read_manifest(folder), local, delivers)
        size = sum(os.path.getsize(f) for it in items for f in it["files"] if os.path.isfile(f))
        self._set(phase="importing", text=text("dcc.job.importing", mb=size / 1e6), fraction=None)
        if size > ASK_ABOVE:
            yes = self._on_main(lambda: self.ask(text("dcc.job.ask_import", mb=size / 1e6, folder=version_dir)))
            if not yes:
                self._record(number, version_dir, "", "", [], imported=False)
                self._set(phase="done", text=text("dcc.job.not_imported", folder=version_dir), done=True, version=number)
                return
        plate = _plate(state)
        for it in items:  # made ready for the scene off the main thread (a DCC may convert it in a process of its own)
            if it.get("main"):
                it["plate"] = plate
                it["source"] = it["main"]
                it["main"] = self.host.prepare_import(it, it["main"], self._cancel.is_set)
        entry = self._on_main(lambda: self._import(number, version_dir, node_name, items, state))
        done = text("dcc.job.imported", version=f"v{number:03d}", group=entry["group_name"])
        if entry.get("note"):
            done = text("dcc.job.imported_note", done=done, note=entry["note"])
        self._set(phase="done", text=done, done=True,
                  version=number, fraction=1.0)

    def _import(self, number: int, version_dir: str, node_name: str, items: list, state: dict) -> dict:
        """Main thread: the whole version — its group, its namespace, every item, the node's record of it — as one
        undo step (host.one_undo_step: one Ctrl+Z takes all of it away; the host may run it again on redo)."""
        host = self.host
        clean_node = safety.clean(node_name, 40, "lab2shot")
        plate = _plate(state)
        for it in items:
            it["plate"] = plate

        def work() -> dict:
            group_name = safety.numbered(f"{clean_node}_v{number:03d}", host.name_taken, safety.NODE_MOST)
            namespace = safety.numbered(f"l2s_{clean_node}_v{number:03d}", host.name_taken, safety.NODE_MOST)
            group = host.make_group(group_name, {"node": node_name, "version": number, "job": self.job,
                                                 "tool": (self.tool or {}).get("name", ""), "folder": version_dir})
            objects = []
            for it in items:
                if not it.get("main"):
                    continue
                why = host.can_import(it.get("type", ""), it["main"])
                if why:
                    log.get().warning("skipped %s: %s", it["main"], why)
                    continue
                objects += host.import_result(it, it["main"], group, namespace)
            made, plate, note = self._backplates(objects, state, namespace)
            entry = self._record(number, version_dir, group, namespace, objects + made, group_name=group_name,
                                 on_main=True, backplate=plate)
            return {**entry, "note": note}

        return host.one_undo_step(text("dcc.job.undo_import", name=f"{clean_node}_v{number:03d}"), work)

    def _backplates(self, objects: list, state: dict, namespace: str) -> tuple[list, dict, str]:
        """Main thread, inside the version's undo step: a backplate behind every camera this version brought in, by
        the one rule of every DCC (backplate below); the host only makes the plate, on its own new camera. Returns
        (the plates' refs, which picture: {"from", "file"}, a sentence for the panel when none was hung). A plate that
        cannot be made is said, never a failed import."""
        host = self.host
        cameras = host.cameras_in(objects)
        if not cameras:
            return [], {}, ""
        picture, why = backplate(state, host.camera_picture)
        if not picture:
            log.get().info("no image plane on the result camera: %s", why)
            return [], {}, why
        made = []
        try:
            for camera in cameras:
                made += host.hang_picture(camera, picture, namespace)
        except Exception as exc:  # noqa: BLE001 - the result is in; only its backplate is missing
            log.get().warning("hanging the image plane failed: %s\n%s", exc, traceback.format_exc())
            return made, {}, text("dcc.job.plane_failed", error=exc)
        return made, {"from": picture["from"], "file": picture["file"]}, ""

    def _record(self, number, version_dir, group, namespace, objects, imported=True, group_name="", on_main=False,
                backplate=None) -> dict:
        entry = {"version": number, "job": self.job, "folder": version_dir, "group": group, "group_name": group_name,
                 "namespace": namespace, "objects": objects, "imported": imported, "made": _now(),
                 "tool": (self.tool or {}).get("name", ""), "backplate": backplate or {}}

        def save():
            state = self.host.load(self.node)
            kept = [v for v in state.get("versions") or [] if int(v.get("version") or 0) != number]  # imported later
            was = next((v for v in state.get("versions") or [] if int(v.get("version") or 0) == number), None)
            if was is not None:  # a version imported after it was fetched keeps its job, when it was made and by what
                entry.update(made=was.get("made") or entry["made"], tool=was.get("tool") or entry["tool"],
                             job=was.get("job") or entry["job"])
            state["versions"] = sorted([*kept, entry], key=lambda v: int(v.get("version") or 0))
            if not self.version:  # importing a kept version is no news about the node's last job
                state["job"] = {**(state.get("job") or {}), "id": self.job, "state": "fetched"}
            self.host.store(self.node, state)

        if on_main:
            save()
        else:
            self._on_main(save)
        return entry


    def _import_kept(self) -> None:
        """A version fetched before but not imported (the user said 「先不导入」): its files read where they were kept
        (the version folder, its manifest and the file map results.settle wrote), then the same import as a fetch."""
        number = self.version
        state = self._on_main(lambda: self.host.load(self.node))
        entry = next((v for v in state.get("versions") or [] if int(v.get("version") or 0) == number), None)
        version_dir = str((entry or {}).get("folder") or "")
        if not version_dir or not os.path.isdir(version_dir):
            raise RuntimeError(text("dcc.job.version_gone", version=f"v{number:03d}", folder=version_dir))
        node_name = self._on_main(lambda: self.host.node_name(self.node))
        with open(os.path.join(version_dir, results.MAP_FILE), encoding="utf-8") as f:
            mapping = json.load(f)
        delivers = (self.tool or {}).get("delivers") if self.tool else None
        items = []
        if os.path.isfile(os.path.join(version_dir, results.MANIFEST)):  # one output: the version folder is it
            items = results.items(results.read_manifest(version_dir), mapping, delivers)
        else:  # several: one sub-folder each, the map's keys led by the output's node
            for name in sorted(os.listdir(version_dir)):
                folder = os.path.join(version_dir, name)
                if not os.path.isfile(os.path.join(folder, results.MANIFEST)):
                    continue
                manifest = results.read_manifest(folder)
                prefix = f"{manifest.get('node')}/"
                local = {k[len(prefix):]: v for k, v in mapping.items() if k.startswith(prefix)}
                items += results.items(manifest, local, delivers)
        self._set(phase="importing", text=text("dcc.job.importing", mb=sum(
            os.path.getsize(f) for it in items for f in it["files"] if os.path.isfile(f)) / 1e6), fraction=None)
        plate = _plate(state)
        for it in items:
            if it.get("main"):
                it["plate"] = plate
                it["source"] = it["main"]
                it["main"] = self.host.prepare_import(it, it["main"], self._cancel.is_set)
        done = self._on_main(lambda: self._import(number, version_dir, node_name, items, state))
        said = text("dcc.job.imported", version=f"v{number:03d}", group=done["group_name"])
        if done.get("note"):
            said = text("dcc.job.imported_note", done=said, note=done["note"])
        self._set(phase="done", text=said, done=True, version=number, fraction=1.0)


def _plate(state: dict) -> dict:
    """The picture the job ran on: the first bound picture (its frame offset puts results back at the DCC's frames)."""
    for b in (state.get("bindings") or {}).values():
        if str(b.get("type", "")).startswith(("image", "video")) and b.get("file"):
            return {"file": b["file"], "frame_offset": b.get("frame_offset", 0),
                    "sequence": b.get("sequence", _numbered(b["file"])), "colorspace": b.get("colorspace", "")}
    return {}


_FRAME = re.compile(r"\d+\.[A-Za-z0-9]+$")
_DIGITS = re.compile(r"\d+(?=\.[A-Za-z0-9]+$)")  # the frame number before the extension (files_label)
VIDEO = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mxf")
NO_BACKPLATE = "dcc.job.no_plane"  # why a result camera has no image plane: its word's key


def files_label(picked: str | list[str], sequence: bool) -> str:
    """What the panel shows for a bound file: the frames picked together, or for one frame of a sequence the frames
    found next to it (the ones sent: lab2shot_client.local_files) — first … last and how many, so the user sees the
    whole sequence was taken, not the one frame clicked."""
    frames = sorted(picked) if isinstance(picked, list) else []
    if not frames and sequence and _numbered(picked) and not str(picked).lower().endswith(VIDEO):
        folder, name = os.path.split(str(picked))
        shape = _DIGITS.sub("#", name)
        try:
            frames = sorted(os.path.join(folder, n) for n in os.listdir(folder or ".") if _DIGITS.sub("#", n) == shape)
        except OSError:
            frames = []
    if len(frames) < 2:
        return str(picked if not frames else frames[0])
    return text("dcc.job.frames", first=frames[0], last=os.path.basename(frames[-1]), count=len(frames))


def _numbered(path: str) -> bool:
    """A picked file is one frame of a sequence when its name ends in a frame number (plate.1001.exr, img_0259.jpg);
    a video plays through its frames."""
    name = os.path.basename(str(path))
    return name.lower().endswith(VIDEO) or bool(_FRAME.search(name))


def backplate(state: dict, camera_picture) -> tuple[dict, str]:
    """The picture behind a delivered camera — one rule for every DCC: the picture and the camera
    are separate inputs; a bound picture is used; else the bound camera's own picture (`camera_picture(binding)`,
    the host reading it, never changing it); neither: no backplate, and why (输入不全). Returns ({file, sequence,
    frame_offset, colorspace, from: "input" | "camera"} or {}, the sentence when {})."""
    picture = _plate(state)
    if picture:
        return {**picture, "from": "input"}, ""
    for b in (state.get("bindings") or {}).values():
        if b.get("type") == "scene.camera" and b.get("ref"):
            own = camera_picture(b) or {}
            if own.get("file"):
                return {"file": own["file"], "frame_offset": int(own.get("frame_offset") or 0),
                        "sequence": bool(own.get("sequence", _numbered(own["file"]))),
                        "colorspace": own.get("colorspace", ""), "from": "camera"}, ""
    return {}, text(NO_BACKPLATE)


def _quiet(fn):
    try:
        fn()
    except Exception:  # noqa: BLE001
        log.get().warning("main-thread step failed: %s", traceback.format_exc())


def _tmp_dir() -> str:
    folder = os.path.join(paths.user_dir(), "tmp")
    os.makedirs(folder, exist_ok=True)
    return folder
