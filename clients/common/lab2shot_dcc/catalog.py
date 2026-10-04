"""The tool list: four sources, one shape, every tool listed (only ordered, never filtered: design §5).

Sources (server/tools.py): preset (the templates), node (single nodes the server wraps), mine (the account's saved
graphs) — all from GET /api/tools — and local (node graph files on this machine the user added: each read here and
described by POST /api/tools/describe). Every entry has the same fields: id, name, intro, source, category {id, path,
ids}, exposed, inputs, delivers; a local one also its graph and file. Beside them the server lists its `categories`
(the tabs of the panel's home page).

Order (`ranked`), inside each source's group:
1. tools that take what is selected now (the host's selection_types against each input's type) first: the more of
   the selected kinds it takes the higher, then the more of them go into inputs it needs (not optional ones);
2. then by what the host prefers to get back (Host.prefers, data, one table per DCC: Maya 3D data first, Nuke pictures
   first; a host that says nothing: 3D first);
3. then in the server's own order (its categories' order: camera tracking first).
"""

from __future__ import annotations

import json
import os

from . import connection
from .paths import text

SOURCES = (("preset", "dcc.source.preset"), ("node", "dcc.source.node"), ("mine", "dcc.source.mine"),
           ("local", "dcc.source.local"))  # (source, its word's key)
LOCAL_MOST = 2 << 20  # bytes a local graph file may have


def fetch(lab) -> dict:
    """GET /api/tools and /api/ocio: {"tools", "formats", "categories", "colorspaces"} (a background thread: it talks
    to the server)."""
    got = lab._request("GET", "/api/tools", timeout=connection.TOOLS_READ_S)
    try:  # the colour spaces the server knows: a scene's colour space is filled in only when it is one of them
        colorspaces = list(lab._request("GET", "/api/ocio").get("colorspaces") or [])
    except Exception:  # noqa: BLE001 - an older server: nothing filled in
        colorspaces = []
    return {"tools": list(got.get("tools") or []), "formats": list(got.get("formats") or []),
            "categories": list(got.get("categories") or []), "colorspaces": colorspaces}


def local_files() -> list[str]:
    return [p for p in connection.read_settings().get("local_graphs", []) if isinstance(p, str)]


def remember_local(path: str) -> None:
    files = [p for p in local_files() if os.path.normcase(p) != os.path.normcase(path)]
    connection.write_settings(local_graphs=[path, *files][:50])


def forget_local(path: str) -> None:
    connection.write_settings(local_graphs=[p for p in local_files() if os.path.normcase(p) != os.path.normcase(path)])


def read_local(path: str) -> dict:
    """A node graph file of this machine (lab2shot.graph/1)."""
    if os.path.getsize(path) > LOCAL_MOST:
        raise ValueError(text("dcc.catalog.too_big", path=path))
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("schema") != "lab2shot.graph/1":
        raise ValueError(text("dcc.catalog.not_graph", path=path))
    return data


def describe_local(lab, path: str) -> dict:
    """One local graph as a tool entry (POST /api/tools/describe)."""
    graph = read_local(path)
    got = lab._request("POST", "/api/tools/describe", {"graph": graph})
    name = got.get("name") or os.path.splitext(os.path.basename(path))[0]
    return {"id": "local:" + os.path.abspath(path), "name": name, "intro": got.get("intro", ""), "source": "local",
            "category": got.get("category") or {"id": "", "path": [], "ids": []}, "exposed": got.get("exposed", []),
            "inputs": got.get("inputs", []), "delivers": got.get("delivers", []), "graph": graph, "file": path}


def fits(input_type: str, selected: str) -> bool:
    """A selected object of type `selected` can go into an input of `input_type` (alternatives "a|b"; a family root
    takes its members: image takes image.3)."""
    for want in str(input_type or "").split("|"):
        if not want:
            continue
        if want == selected or selected.startswith(want + ".") or want.startswith(selected + "."):
            return True
        if want == "scene" and selected.startswith("scene."):
            return True
    return False


def suffix(selected: str, exports=None) -> str:
    """The file a selected object of this type becomes (Host.exports), "" when the host does not say."""
    return str((exports or {}).get(selected) or "").lower()


def reads(item: dict, file_suffix: str) -> bool:
    """The input reads a file of this suffix (its signature's `accept`: [] or an unknown suffix, anything)."""
    accept = [str(a).lower() for a in item.get("accept") or []]
    return not accept or not file_suffix or file_suffix.lower() in accept


def takes(item: dict, selected: str, exports=None) -> bool:
    """An input takes a selected object: its type fits, and the file the host exports it as is one the input reads
    (Nuke's camera, a .usda, goes into cam_usd_path and not cam_fbx_path; Maya's, an .fbx, the other way)."""
    return fits(item.get("type", ""), selected) and reads(item, suffix(selected, exports))


PREFERS = ("scene",)  # Host.prefers when a host says nothing: tools delivering 3D data first (the rule before hosts had one)


def preferred(tool: dict, prefers=None) -> int:
    """Where the tool's deliveries come in the host's preference (Host.prefers: data type families, most wanted
    first): the best place any delivered type takes; one past the end when none is listed."""
    prefers = tuple(PREFERS if prefers is None else prefers)
    types = {t for d in tool.get("delivers") or [] for t in d.get("types") or []}
    if any(d.get("kinds") for d in tool.get("delivers") or []):
        types.add("scene")  # 3D kinds delivered (a scene of a camera …): the scene family
    places = [i for i, family in enumerate(prefers) for t in types if t == family or t.startswith(family + ".")]
    return min(places, default=len(prefers))


def score(tool: dict, selected: list[str], index: int = 0, prefers=None, exports=None) -> tuple:
    """(how many of the selected kinds it takes, how many of them an input it needs takes, the host's preference, the
    server's order): a tool with five optional picture slots is not five times as fitting as one that needs the
    picture selected. An input takes a selected kind as `takes` says (Host.exports)."""
    inputs = tool.get("inputs") or []
    took = {s for s in selected if any(takes(i, s, exports) for i in inputs)}
    needed = {s for s in took if any(takes(i, s, exports) for i in inputs if not i.get("optional"))}
    return (-len(took), -len(needed), preferred(tool, prefers), index)


def ranked(tools: list[dict], selected: list[str], prefers=None, exports=None) -> list[tuple[str, str, list[dict]]]:
    """[(source id, its label, its tools in order)] for every source with tools (the module docstring)."""
    out = []
    for source, label in SOURCES:
        mine = [(i, t) for i, t in enumerate(tools) if t.get("source") == source]
        if mine:
            out.append((source, text(label), [t for _i, t in sorted(mine, key=lambda it: score(
                it[1], selected, it[0], prefers, exports))]))
    return out
