"""Node names and descriptions are data, not code: they live in nodes.json in the folder of the node's code.

    adapters/<package>/nodes.json  {"label": name, "description": text} for each node of that adapter, by node type id
    lab2shot/nodes/nodes.json      the core nodes (read, image, mask, camera, geometry, scene, building blocks,
                                   output, and each format module's reader and writer nodes)

The file sits with the node code: a node's text is in the nodes.json of the folder its code is in (found from the
class's module path, not from a registry). When the catalogue is built, the text is applied to every node class
(`apply`); from then on everything that reads `cls.label` / `cls.description` (error messages, cards, the queue, the
catalogue) reads the file's text. An account that manages node categories edits a node's name or description in the
node menu (server/categories.py menu_text); the change is written back to its file and applied to the class at once.
When a file is edited or restored outside the server, `refresh` compares each file's modification time (`stamp`) and
applies it again; it is called before the catalogue route answers, before a restricted account's hidden names are
computed and before a save, so reloading the page shows the new text. Applying and saving happen under one lock, and
"applied" is recorded only after the last class. Readers (the catalogue, the hidden-name list) read the classes' text
outside the lock: a request that coincides with a save may read partly new and partly old text, but that answer is
cached only under the old stamp and the next request recomputes under the new key; it is served again only if a file
is restored to an old version with its original modification time. Administrators edit one node at a time, and this
boundary is accepted.

No names or descriptions exist in code: a node without an entry (newly merged, or whose file cannot be read) shows its
type id as its name and an empty description; the catalogue is still built, `lab2shot check` lists nodes missing
entries, and `problem()` names the broken file. Writes use atomic replacement (io/atomic.py) and keep a .bak copy
first (as the category files do, lab2shot/categories.py).
"""

from __future__ import annotations

import json
import shutil
import threading
from functools import lru_cache
from pathlib import Path

from ..config import ROOT
from ..errors import Invalid, NotFound
from ..io.atomic import write_text
from ..messages import Msg

FILE = "nodes.json"
LABEL_CHARS = 30  # fits on one line of the node menu (longest existing: 26)
TEXT_CHARS = 520  # the description is shown in full in the menu (longest existing: 502)


def file_for(cls) -> Path:
    """The text file of a node class: its folder's nodes.json (an adapter's folder, or lab2shot/nodes/ for the core's)."""
    parts = cls.__module__.split(".")
    if parts[0] == "adapters" and len(parts) >= 2:
        return ROOT / "adapters" / parts[1] / FILE
    return ROOT / "lab2shot" / "nodes" / FILE


def files() -> list[Path]:
    """Every text file there is: the core's and one per adapter folder that has one."""
    return [ROOT / "lab2shot" / "nodes" / FILE, *sorted((ROOT / "adapters").glob(f"*/{FILE}"))]


def stamp() -> tuple:
    """Every text file with its modification time: what `refresh` compares and the catalogue route's cache key
    follows (per file, so a file restored with an older time still counts as changed)."""
    out = []
    for f in files():
        try:
            out.append((str(f), f.stat().st_mtime_ns))
        except OSError:
            out.append((str(f), 0))
    return tuple(out)


def _read(path: Path) -> tuple[dict[str, dict], str, str]:
    """(entries, problem, why): the file's entries, or none, the sentence to show and the bare reason."""
    if not path.is_file():
        return {}, "", ""
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(got, dict) or not all(isinstance(v, dict) for v in got.values()):
            raise ValueError("不是一张 节点 → {label, description} 的表")
        for k, v in got.items():
            if not isinstance(v.get("label", ""), str) or not isinstance(v.get("description", ""), str):
                raise ValueError(f"{k} 的 label / description 不是文字")
    except (OSError, ValueError) as exc:
        why = str(exc)[:120]
        return {}, Msg("W-NODETEXT-BROKEN", file=str(path), why=why).text, why
    return {str(k): v for k, v in got.items()}, "", ""


@lru_cache(maxsize=256)
def _entries(path: Path, _stamp: float) -> tuple[dict[str, dict], str, str]:
    return _read(path)


def _stamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def entries(path: Path) -> dict[str, dict]:
    return _entries(path, _stamp(path))[0]


def problem(path: Path) -> str:
    return _entries(path, _stamp(path))[1]


def problems() -> list[str]:
    return [p for f in files() if (p := problem(f))]


_applied_at: tuple = ()  # stamp() of the files the classes last took their words from
_lock = threading.RLock()  # one writer of the classes' words at a time (requests run in a thread pool)


def apply(types) -> tuple:
    """The files' words onto the node classes (`types`: id -> class). A class with no entry, or whose file cannot be
    read, is named by its type id and has no description: the catalogue never fails for a missing word. Returns the
    files' stamp the words came from."""
    global _applied_at
    with _lock:
        at = stamp()
        for cls in types.values():
            got = entries(file_for(cls)).get(cls.id) or {}
            cls.label = str(got.get("label") or "").strip() or cls.id
            cls.description = str(got.get("description") or "").strip()
        _applied_at = at  # after the last class: a reader never sees the new stamp over half-new words
        return at


def refresh(types) -> tuple:
    """Apply again when a file changed since (one restored by hand, one edited outside the server). Called before the
    catalogue is answered, before a restricted account's hidden names are worked out, and before a save. Returns the
    stamp the classes' words now match (the caller's cache key)."""
    with _lock:
        at = stamp()
        return at if at == _applied_at else apply(types)


def _check(label: str, description: str) -> tuple[str, str]:
    label, description = label.strip(), description.strip()
    if not label or len(label) > LABEL_CHARS:
        raise Invalid(Msg("E-NODETEXT-LABEL", most=LABEL_CHARS))
    if len(description) > TEXT_CHARS:
        raise Invalid(Msg("E-NODETEXT-TEXT", most=TEXT_CHARS))
    return label, description


def save(type_id: str, label: str, description: str) -> tuple[dict, bool]:
    """A node's words, written to its file and put on its class now. Returns the entry and whether anything changed."""
    from .registry import node_types

    types = node_types()
    cls = types.get(type_id)
    if cls is None:
        raise NotFound(Msg("E-NODE-NOSUCH", type=type_id))
    label, description = _check(label, description)
    path = file_for(cls)
    with _lock:  # read, change, write and re-apply as one step: two administrators never overwrite each other's words
        refresh(types)  # anything changed outside the server is on the classes first
        rows, problem, why = _read(path)
        if problem:
            bak = path.with_name(path.name + ".bak")
            if bak.is_file():
                raise Invalid(Msg("E-NODETEXT-BROKEN", file=str(path), why=why, bak=str(bak)))
            raise Invalid(Msg("E-NODETEXT-BROKENNOBAK", file=str(path), why=why))
        entry = {"label": label, "description": description}
        if rows.get(type_id) == entry:
            cls.label, cls.description = label, description  # the file already says so; the class says so now too
            return entry, False
        rows[type_id] = entry
        try:
            if path.is_file():
                shutil.copyfile(path, path.with_name(path.name + ".bak"))
            write_text(path, json.dumps(dict(sorted(rows.items())), ensure_ascii=False, indent=1))
        except OSError as exc:  # the folder is not writable here (a packaged install): said, not a 500
            raise Invalid(Msg("E-NODETEXT-UNWRITABLE", file=str(path), why=str(exc)[:80])) from exc
        apply(types)  # every class from the files as they are now (this write included): nothing is marked applied that is not
        return entry, True
