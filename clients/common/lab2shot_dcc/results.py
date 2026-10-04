"""Result versions on disk, and what one delivery holds.

Folders: <project>/data/lab2shot/<node>/v001, v002 … — one per fetch, never reused, never removed by the plugin. A
fetch first lands in a staging folder beside them (.staging_<n>): the zip is downloaded there (the transport writes it
aside and renames it when complete), checked whole (every member's CRC: zipfile.testzip), unpacked with every name
cleaned (lab2shot_dcc.safety.under: the server's Chinese names, spaces and long names never become paths here), the
manifest's files checked present; only then is the staging folder renamed to its version folder. A cancelled or failed
fetch removes its staging folder: half a result is never there to import. Before downloading: the folder must be
writable and the disk must have room for the zip and what it unpacks to (ROOM_FACTOR × its size).

The manifest (lab2shot.json at the root of every delivery; lab2shot/transfer/outputs.py) becomes `items`: one per
output-settings node — its original name, the data type it was made from, its main file and files (as they are now on
this machine), its pictures' colour space ("" when it holds none), and the 3D kinds the tool's signature says that node
delivers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile

from . import log, safety
from .connection import Cancelled
from .paths import text

ROOM_FACTOR = 2.5
MANIFEST = "lab2shot.json"
MAP_FILE = "lab2shot_files.json"  # our own: server's relative path -> the cleaned one written here
_VERSION = re.compile(r"^v(\d{3,})$")


def node_folder(project: str, node_name: str) -> str:
    # one spelling of every path from here on (a DCC may give its project with forward slashes)
    return os.path.abspath(os.path.join(project, "data", "lab2shot", safety.clean(node_name, safety.FILE_MOST, "node")))


def next_version(folder: str, known: list[int] = ()) -> int:
    """The first version number above every one there (folders, and versions the node recorded)."""
    used = set(known)
    if os.path.isdir(folder):
        for n in os.listdir(folder):
            m = _VERSION.match(n)
            if m:
                used.add(int(m.group(1)))
    return max(used, default=0) + 1


def staging(folder: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = safety.numbered(os.path.join(os.path.abspath(folder), ".staging"), os.path.exists, 4000)
    os.makedirs(path)
    return path


def check_room(folder: str, size: int) -> None:
    why = safety.writable(folder)
    if why:
        raise RuntimeError(why)
    need = int(size * ROOM_FACTOR) + (64 << 20)
    have = safety.free_bytes(folder)
    if have < need:
        raise RuntimeError(text("dcc.results.no_room", size=size / 1e6, need=need / 1e6, folder=folder,
                                have=have / 1e6))


def unpack(archive: str, into: str, cancel=None, project_word: str = "") -> dict[str, str]:
    """Every member of the delivery's zip (its one top folder dropped) into `into`, names cleaned; returns
    {server's relative path: local path}. The zip is checked whole first."""
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(text("dcc.results.corrupt", member=bad))
        mapping: dict[str, str] = {}
        for member in zf.infolist():
            if member.is_dir():
                continue
            parts = member.filename.replace("\\", "/").split("/")
            rel = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
            target = safety.under(into, rel)
            if target is None:
                continue
            if target in mapping.values():  # two names cleaned alike: numbered
                stem, ext = os.path.splitext(target)
                target = safety.numbered(stem, lambda s: os.path.exists(s + ext) or (s + ext) in mapping.values(), 4000) + ext
            if not safety.path_fits(target):
                raise RuntimeError(text("dcc.results.path_too_long", most=safety.PATH_MOST, path=target,
                                        project=project_word or text("dcc.results.project")))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    dst.write(chunk)
            mapping[rel] = target
    with open(os.path.join(into, MAP_FILE), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=1)
    return mapping


def items(manifest: dict, mapping: dict[str, str], delivers: list[dict] | None = None) -> list[dict]:
    """The delivery's items (the module docstring); raises when a file the manifest lists did not arrive."""
    kinds_of = {}
    for d in delivers or []:
        for s in d.get("settings") or []:
            kinds_of[str(s.get("name") or "")] = {"kinds": s.get("kinds") or [], "format": s.get("format") or "",
                                                  "types": s.get("types") or []}
    out = []
    for o in manifest.get("outputs") or []:
        files = [mapping.get(f) for f in o.get("files") or []]
        if any(f is None or not os.path.isfile(f) for f in files):
            raise RuntimeError(text("dcc.results.missing_file", name=o.get("name")))
        rel = str(o.get("main") or "")
        if "#" in rel:  # a sequence's main is its #### pattern: its first frame
            stem = os.path.basename(rel).split("#", 1)[0]
            frames = sorted(f for f, r in zip(files, o.get("files") or []) if os.path.basename(r).startswith(stem)
                            and not r.endswith(".lab2shot.json"))
            main = frames[0] if frames else ""
        else:
            main = mapping.get(rel, "")
        about = kinds_of.get(str(o.get("name") or ""), {})
        out.append({"name": str(o.get("name") or ""), "item": str(o.get("item") or ""), "type": str(o.get("type") or ""),
                    "main": main, "files": files, "kinds": about.get("kinds", []), "format": about.get("format", ""),
                    "types": about.get("types", []), "sequence": "#" in rel, "colorspace": str(o.get("colorspace") or "")})
    return out


def read_manifest(folder: str) -> dict:
    with open(os.path.join(folder, MANIFEST), encoding="utf-8") as f:
        return json.load(f)


def settle(stage: str, version_dir: str, mapping: dict[str, str]) -> dict[str, str]:
    """The staging folder becomes the version folder (it must not exist yet); returns the mapping with the files where
    they are now."""
    stage, version_dir = os.path.abspath(stage), os.path.abspath(version_dir)
    if os.path.exists(version_dir):
        raise RuntimeError(text("dcc.results.version_exists", folder=version_dir))
    os.replace(stage, version_dir)
    moved = {}
    for k, v in mapping.items():
        v = os.path.abspath(v)
        rel = os.path.relpath(v, stage)
        moved[k] = os.path.join(version_dir, rel) if not rel.startswith("..") else v
    with open(os.path.join(version_dir, MAP_FILE), "w", encoding="utf-8") as f:
        json.dump(moved, f, ensure_ascii=False, indent=1)
    return moved


def discard(path: str) -> None:
    """Remove a staging folder (only ever one of ours: it is named .staging…)."""
    if path and os.path.basename(path).startswith(".staging") and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def colorspace_for(name: str, available, table: dict | None = None) -> str:
    """The DCC's own name for a colour space the server named (the manifest's `colorspace`), among the names the
    DCC's configuration has now (`available`; a family path "Input/ARRI/x" matches by its last part): the server's
    name itself, then the host's table of other names for it (Host.colorspaces), compared without case. "" when none
    is there: the picture is then left in the DCC's default and the log says so — a colour space is never guessed."""
    if not name:
        return ""
    have: dict[str, str] = {}
    for full in available or []:
        full = str(full)
        for key in (full, full.rsplit("/", 1)[-1]):
            have.setdefault(key.strip().lower(), full)
    table = {str(k).lower(): v for k, v in (table or {}).items()}
    for candidate in [name, *(table.get(str(name).lower()) or [])]:
        found = have.get(str(candidate).strip().lower())
        if found:
            return found
    log.get().warning("colour space %r: none of its names is in this configuration; left as it is", name)
    return ""
