"""What an output node (「输出」) collects for its task: the one place that knows how an output looks on the disk.

「输出」's own work is 整理 + 压缩 (collect, then pack; its download is there only after it): it collects the files the
output-settings nodes wired into it wrote (their results in the account's cache) into one folder of its task, then
packs that folder into one zip beside it. Both stay on the server as long as the task (任务保留天数, transfer/tasks.py):

    <task folder>/output/<pkg>/                 one 「输出」's files, unpacked: DCC plugins and `lab2shot cook` fetch
        lab2shot.json                           single files from here (what is inside: they read it to import)
        <名字>/<名字>.usd ...                    one sub-folder per output-settings node, its files named after it
        <条目>/<名字>/...                        inside a 逐项处理 block: one sub-folder per item, the same inside
    <task folder>/<pkg>.zip                     the same folder packed, for the browser's own download (resumable)

`pkg` is made by the server from the account's id, the task's id and the node's id (u12_1a2b3c4d5e6f_deliver):
nothing a user typed is ever part of a path on the server. The zip holds exactly one top folder, named like the file
the browser saves it as (`name`: the graph's name made safe, the ML mark, the account's id and the task's id —
深度_ML_Lab2Shot_ViPE_u12_1a2b3c4d5e6f.zip holds 深度_ML_Lab2Shot_ViPE_u12_1a2b3c4d5e6f/): unpacked, it is one folder
of that name, never loose files. Files that are compressed already (EXR, PNG, JPG, videos …) are stored as they are;
everything else is deflated at the highest level (more server time, less to download).

The files are hard links to the output-settings nodes' results in the account's cache (no second copy; a copy where
the file system can't link: io/files.py link_or_copy); like everything in a task folder they are written once and never
changed in place. An output is finished once its zip is there (written aside and renamed into place): only then is it
listed, fetched or downloaded. Nothing else is kept about it: the folder is the record, and it goes with its task.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import zipfile
from functools import lru_cache
from pathlib import Path

from ..data.packet import Packet
from ..errors import CookCancelled, CookError, Invalid, NotFound
from ..io.atomic import put_in_place
from ..io.digest import sha256
from ..io.files import inside, link_or_copy
from ..messages import Msg
from ..nodes.output import marked, name_key
from ..text import file_part
from . import relative_name, tasks

MANIFEST = "lab2shot.json"  # at the root of every output: what is inside
# already compressed: stored in the zip as they are (deflating them again costs time and gains next to nothing)
STORED = frozenset({
    ".exr", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".avif", ".heic", ".jp2",
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mxf", ".mp3", ".aac", ".m4a", ".ogg", ".flac",
    ".zip", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".7z", ".rar", ".usdz", ".usdc", ".npz",
})
LEVEL = 9  # deflate's highest
CHUNK = 1 << 22  # bytes read at a time while packing: a stop is noticed between chunks
READABLE_MOST = 40  # characters of the readable part of a download's name
_NODE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def _gone() -> NotFound:
    return NotFound(Msg("E-OUTPUT-GONE", days=tasks.keep_days()))


def pkg_of(user_id: int, task_id: str, node_id: str) -> str:
    """The server's name of one 「输出」's output in a task (its folder under output/ and its zip's file name): the
    account's id, the task's id and the node's id (a node id that is not plain letters and digits by its hash)."""
    tasks.task_dir(task_id)
    part = node_id if _NODE.match(node_id or "") else sha256(str(node_id))[:12]
    return f"u{int(user_id)}_{task_id}_{part}"


def folder(task_id: str, pkg: str) -> Path:
    """One output's unpacked folder (a name that is not one the server makes is not there)."""
    tasks.zip_path(task_id, pkg)  # the same rule for both names
    return tasks.output_dir(task_id) / pkg


def is_stored(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in STORED)


class Collector:
    """Where the 「输出」 nodes of one task collect and pack (nodes/services.py OutputSink): the farm builds one per
    task and hands it to the task's cook (engine/cook.py CookContext.collector). The node's cook only collects
    (each of its instances: one outside every block, one per item inside one, perhaps several at once); the engine
    packs, once, when all of them have (Engine.cook). `owner`: user (the account's
    id), title (the graph's name), graph (its file's meta.id), several (the graph has more than one 「输出」: each
    download's name then says which). Each finished output is told to `emit` as the task's "output" event."""

    def __init__(self, task_id: str, owner: dict, emit) -> None:
        tasks.task_dir(task_id)
        self.task, self.owner, self.emit = task_id, owner, emit
        self._lock = threading.Lock()
        self._listed: dict[str, list[dict]] = {}  # node -> what its instances collected (the manifest's outputs)
        self._items: dict[str, dict[tuple, str]] = {}  # node -> item path -> its sub-folder (each item its own)
        self._learned: dict[str, list[str]] = {}  # node -> the models behind what it collected (the ML mark)
        self._packed: set[str] = set()

    def pkg(self, node_id: str) -> str:
        return pkg_of(self.owner["user"], self.task, node_id)

    def _item_dir(self, node_id: str, path: tuple, names: tuple[str, ...]) -> str:
        """The sub-folder of one item of a 逐项处理 block (under the lock): its names, outer block first, made safe. An
        item's names are unique in its list, so only names that making safe changed can come out the same (「a/b」 and
        「a_b」): those carry a suffix of their own item keys, the same whichever instance collects first (several collect
        at once: a number by arrival would differ from one run to the next). Names alike on some file system (name_key:
        「A」 and 「a」, é written two ways) are told apart the same way: the later one carries its suffix (which is later
        can differ between runs; only such a pair, never a name on its own)."""
        taken = self._items.setdefault(node_id, {})
        if path in taken:
            return taken[path]
        raw = list(names or path)
        parts = [file_part(n, 60) or f"item{i + 1}" for i, n in enumerate(raw)]
        base = "/".join(parts)
        alike = any(name_key(t) == name_key(base) for t in taken.values())
        taken[path] = base if parts == raw and not alike else f"{base}_{sha256('/'.join(path))[:6]}"
        return taken[path]

    def collect(self, node_id: str, label: str, outputs: list[Packet], stop: threading.Event, path: tuple = (),
                names: tuple[str, ...] = ()) -> dict:
        """Collect the files of `outputs` (files packets: meta name, main, files) into the node's folder: each in a
        sub-folder of its 名字, inside a 逐项处理 block under the item's own sub-folder (`path`: the instance's item
        path, `names`: its items' names), until `stop` (the node's). Returns {"commercial", "projects": the
        non-commercial ones}."""
        for p in outputs:  # a packet without .complete is not a packet: an incomplete one is never collected
            # checked outright, never by an assert (python -O would drop it); complete is what is asked here: the cook
            # took it as an input only when valid (engine/presence.py says which test is for what)
            if not Packet.exists(p.dir):
                raise CookError(node_id, Msg("E-OUTPUT-INCOMPLETE", node=label))
        base = folder(self.task, self.pkg(node_id))
        with self._lock:
            if node_id in self._packed:  # packed already: a late instance never changes a finished zip
                raise CookError(node_id, Msg("E-OUTPUT-PACKED", node=label))
            under = self._item_dir(node_id, tuple(path), tuple(names)) if path else ""
        listed = []
        for p in outputs:
            name = str(relative_name(p.meta["name"]))
            at = f"{under}/{name}" if under else name
            for rel in p.meta["files"]:
                if stop.is_set():
                    raise CookCancelled
                dst = base / at / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    link_or_copy(p.path(rel), dst)
                elif not _same_file(p.path(rel), dst):  # the same place holds another file: never kept on the quiet
                    raise CookError(node_id, Msg("E-OUTPUT-CLASH", node=label, file=f"{at}/{rel}"))
            listed.append({"name": name, "item": under, "type": p.meta.get("made_from", ""), "main": f"{at}/{p.meta['main']}",
                           "files": [f"{at}/{rel}" for rel in p.meta["files"]], "commercial": p.meta["commercial"]})
        with self._lock:
            self._listed.setdefault(node_id, []).extend(listed)
            learned = self._learned.setdefault(node_id, [])
            learned += [x for p in outputs for x in p.meta.get("learned", []) if x not in learned]
        return {"commercial": commercial(listed)}

    def download_name(self, node_id: str, label: str) -> str:
        """What the browser saves the zip as, without .zip, and the one folder inside it: the graph's name (and the
        node's, when the graph has several 「输出」) made safe, marked when a model made what is in it, then the
        account's id and the task's id."""
        readable = file_part(self.owner.get("title", ""), READABLE_MOST) or "Lab2Shot"
        if self.owner.get("several"):
            readable = f"{readable}_{file_part(label, 20) or node_id[:20]}"
        return f"{marked(readable, self._learned.get(node_id, []))}_u{int(self.owner['user'])}_{self.task}"

    def pack(self, node_id: str, label: str, stop: threading.Event) -> dict:
        """The node's collection is complete: write its manifest and pack its folder into its zip (once), until `stop`
        (the packing's own: its cook stopped, or it ran past its time limit). Returns the output's record, told as the
        "output" event."""
        with self._lock:
            if node_id in self._packed:
                return record(self.task, self.pkg(node_id))
            listed = list(self._listed.get(node_id, []))
            # every output setting wired in gave nothing (or a block had no items): never a zip of a manifest alone
            if not any(o["files"] for o in listed):
                raise CookError(node_id, Msg("E-DELIVER-EMPTY", node=label))
            self._packed.add(node_id)
        pkg, name = self.pkg(node_id), self.download_name(node_id, label)
        base = folder(self.task, pkg)
        base.mkdir(parents=True, exist_ok=True)
        manifest = {"schema": "lab2shot.output/1", "task": self.task, "node": node_id, "label": label, "pkg": pkg,
                    "name": f"{name}.zip", "title": self.owner.get("title", ""), "graph": self.owner.get("graph", ""),
                    "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "outputs": listed,
                    "commercial": commercial(listed)}
        (base / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        _pack(base, tasks.zip_path(self.task, pkg), name, stop)
        found = record(self.task, pkg)
        self.emit({"type": "output", **summary(found)})
        return found


def _pack(base: Path, target: Path, top: str, stop: threading.Event) -> None:
    """Pack every file under `base` into the zip `target`, each under `top`/ (the entry names are the server's:
    `top` and the names the folder holds, which the server wrote). Written aside and renamed into place durably, so a
    zip that is there is whole, after a crash too; a stop leaves none."""
    part = target.with_name(target.name + ".part")
    files = sorted((f for f in base.rglob("*") if f.is_file()), key=lambda f: (f.relative_to(base).as_posix() != MANIFEST,
                                                                           f.relative_to(base).as_posix()))
    try:
        with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
            for f in files:
                rel = f.relative_to(base).as_posix()
                info = zipfile.ZipInfo.from_file(f, f"{top}/{rel}", strict_timestamps=False)
                if is_stored(rel):
                    info.compress_type = zipfile.ZIP_STORED
                else:
                    info.compress_type, info.compress_level = zipfile.ZIP_DEFLATED, LEVEL
                with f.open("rb") as src, zf.open(info, "w") as dst:  # its size is known: zip64 when it needs it
                    while chunk := src.read(CHUNK):
                        if stop.is_set():
                            raise CookCancelled
                        dst.write(chunk)
        os.chmod(part, 0o444)  # written once: never changed in place (a task folder's rule)
        put_in_place(part, target)
    finally:
        part.unlink(missing_ok=True)


# ------------------------------------------------------------------ reading them back


@lru_cache(maxsize=256)
def _read_manifest(path: str, _stamp: int) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _manifest(base: Path) -> dict | None:
    """The output's manifest, read once per version of it (the queue asks about every output of every job it shows)."""
    try:
        stamp = (base / MANIFEST).stat().st_mtime_ns
    except OSError:
        return None
    return _read_manifest(str(base / MANIFEST), stamp)


def commercial(listed: list[dict]) -> bool:
    """What was collected may be used commercially: some file was, and every piece of it may (nothing collected is not
    「可商用」: all([]) would say it is)."""
    return any(o["files"] for o in listed) and all(o["commercial"] for o in listed)


def _same_file(a: Path, b: Path) -> bool:
    """The file collected there already is this one (linked, or the same bytes: an instance collected again)."""
    import filecmp

    try:
        return os.path.samefile(a, b) or filecmp.cmp(a, b, shallow=False)
    except OSError:
        return False


def record(task_id: str, pkg: str, files: bool = False) -> dict:
    """One finished output of a task: task, node, label, pkg, name (what the browser saves the zip as), bytes (the
    zip's), count (its files), commercial, title, graph, made; with `files`, every file of its folder (relative, the
    manifest first) and its manifest's outputs. NotFound: not there, or not finished (no zip yet)."""
    try:
        base, z = folder(task_id, pkg), tasks.zip_path(task_id, pkg)
    except NotFound:
        raise _gone() from None
    m = _manifest(base) if z.is_file() else None
    if m is None:
        raise _gone()
    st = z.stat()
    out = {k: m.get(k, "") for k in ("node", "label", "name", "title", "graph", "made")}
    out.update({"task": task_id, "pkg": pkg, "bytes": st.st_size, "count": 1 + sum(len(o.get("files", [])) for o in m.get("outputs", [])),
                "commercial": bool(m.get("commercial")), "finished": st.st_mtime})
    if files:
        out.update({"files": _files(base), "outputs": m.get("outputs", [])})
    return out


def summary(r: dict) -> dict:
    """A record without its file list: what events, the queue and the job's record carry."""
    return {k: v for k, v in r.items() if k not in ("files", "outputs")}


def _files(base: Path) -> list[str]:
    names = sorted(f.relative_to(base).as_posix() for f in base.rglob("*") if f.is_file())
    return sorted(names, key=lambda n: n != MANIFEST)


def of_task(task_id: str) -> list[dict]:
    """Every finished output of a task (oldest first)."""
    try:
        out_dir = tasks.output_dir(task_id)
    except NotFound:
        return []
    found = []
    for d in sorted(out_dir.iterdir()) if out_dir.is_dir() else []:
        try:
            found.append(record(task_id, d.name))
        except NotFound:
            continue
    return sorted(found, key=lambda r: r["finished"])


def of_account(user_id: int | None = None) -> list[dict]:
    """Every finished output of the account's live tasks (of everyone's, `user_id` None), newest first, each with
    `user` (the account) and `expires` (when its task goes: 任务保留天数 after it ended; None while it runs)."""
    from ..database import db

    rows = (db().rows("SELECT id, user_id, ended FROM tasks ORDER BY created DESC") if user_id is None
            else db().rows("SELECT id, user_id, ended FROM tasks WHERE user_id = ? ORDER BY created DESC", (user_id,)))
    out = []
    for r in rows:
        expires = r["ended"] + tasks.keep_days() * tasks.DAY if r["ended"] is not None else None
        out += [{**o, "user": r["user_id"], "expires": expires} for o in reversed(of_task(r["id"]))]
    return out


def present(output: dict) -> dict:
    """An output as a job's record or the queue shows it now: still there (`gone` False) or gone with its task."""
    try:
        return {**output, **summary(record(output["task"], output["pkg"])), "gone": False}
    except (KeyError, NotFound):
        return {**output, "gone": True}


def zip_file(task_id: str, pkg: str) -> Path:
    """A finished output's zip (NotFound: not there)."""
    record(task_id, pkg)
    return tasks.zip_path(task_id, pkg)


def file(task_id: str, pkg: str, name: str) -> Path:
    """One file of a finished output's folder, by the name its record lists. Where a name may lead is decided in one
    place for the whole project (io.files.inside: resolved first, so a link out of the folder is out); a name that
    leads anywhere else is answered exactly like one that is not there, so an outsider learns nothing either way."""
    record(task_id, pkg)
    try:
        f = inside(folder(task_id, pkg), name)
    except Invalid:
        raise _gone() from None
    if not f.is_file():
        raise _gone()
    return f


def discard_unfinished(task_id: str) -> None:
    """A task that ended without finishing (failed, cancelled): the folders its outputs were collecting into and never
    packed are not outputs (never listed, fetched or downloaded): they go. Finished ones (their zip is there) stay."""
    import shutil

    try:
        out_dir = tasks.output_dir(task_id)
    except NotFound:
        return
    for d in list(out_dir.iterdir()) if out_dir.is_dir() else []:
        try:
            finished = tasks.zip_path(task_id, d.name).is_file()
        except NotFound:
            finished = False
        if not finished:
            shutil.rmtree(d, ignore_errors=True)
    tasks.changed_folder(task_id)  # it has ended: its folder's size is looked at again (tasks.account_bytes)


def remove_account(user_id: int) -> int:
    """An account is deleted: every output of its tasks goes now (folders and zips; the tasks themselves go by the
    usual cleaning, farm/disk.py). Returns how many finished outputs went."""
    import shutil

    from ..database import db

    gone = 0
    for r in db().rows("SELECT id FROM tasks WHERE user_id = ?", (user_id,)):
        gone += len(of_task(r["id"]))
        for z in tasks.task_dir(r["id"]).glob("*.zip"):
            z.unlink(missing_ok=True)
        shutil.rmtree(tasks.output_dir(r["id"]), ignore_errors=True)
        tasks.changed_folder(r["id"])
    return gone
