"""What the panel shows, worked out without Qt and without a DCC (设计_DCC新面板.md §1.1): the home page's cards, tabs
and suggestions, a card's 「需要：画面、摄影机」, the search, the tool page's inputs and its common / advanced
parameters, the result versions, the scene's tasks. The Qt layer (lab2shot_dcc.ui) only draws what these return, so
every rule here is tested in plain Python (clients/common/tests/test_view_model.py).

Words: the plugin's own (dcc.ui.*) and the data types' names (type.<id>.label, type.or: the client's word table,
tools/messages_client.py), in the plugin's language now; what the server says (a tool's name, intro, a category's
label) comes in its language with the tool list.
"""

from __future__ import annotations

import os
import re

from . import catalog, contract, jobs
from .paths import text

ALL = "all"  # the tabs that are not a category: every tool, the account's own templates, local graphs, recently used
RECENT = "recent"
NODES = "source:node"
MINE = "source:mine"
LOCAL = "source:local"
SUGGEST_MOST = 6  # cards in 「适合当前选中」
RECENT_MOST = 12  # tools kept in 「最近用过」, per server
UNPLACED_COLOUR = "#8E8E93"  # a tool in no category (the page's default category colour: lab2shot/categories.py)

_CJK_RANGES = ((0x3000, 0x303F), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF))
_CJK = re.compile("[%s]" % "".join(f"{chr(a)}-{chr(b)}" for a, b in _CJK_RANGES))


def has_cjk(s: str) -> bool:
    return bool(_CJK.search(str(s or "")))


# ---- words of the data types


def type_label(data_type: str) -> str:
    """A data type's name (type.<id>.label), its id when the table has no name for it."""
    try:
        return text(f"type.{data_type}.label")
    except KeyError:
        return data_type


def type_names(data_type: str) -> str:
    """An input's type as said: alternatives "a|b" joined by type.or (「图像或视频」, "Image or Video")."""
    parts = [type_label(t) for t in dict.fromkeys(p for p in str(data_type or "").split("|") if p)]
    return text("type.or").join(parts)


def _first_line(s: str) -> str:
    for line in str(s or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


# ---- a tool card


def category_of(tool: dict) -> str:
    """The tool's first-level category id ("" when it sits in none)."""
    cat = tool.get("category") or {}
    ids = cat.get("ids") or []
    return str(ids[0]) if ids else str(cat.get("id") or "")


def needs(tool: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """([(type, its name)] the tool must be given, [(type, name)] it may be given): each input's type once, an input's
    alternatives as one entry (「图像或视频」), a type needed never listed again as optional."""
    must: dict[str, str] = {}
    may: dict[str, str] = {}
    for i in tool.get("inputs") or []:
        t = str(i.get("type") or "")
        if not t:
            continue
        (may if i.get("optional") else must).setdefault(t, type_names(t))
    return list(must.items()), [(t, n) for t, n in may.items() if t not in must]


def card(tool: dict, colours: dict | None = None) -> dict:
    """{id, name, intro, needs, optional, category, colour, source}: what a tool card shows (名字、一行说明、「需要：」)."""
    must, may = needs(tool)
    cat = category_of(tool)
    return {"id": tool.get("id"), "name": str(tool.get("name") or tool.get("id") or ""),
            "intro": _first_line(tool.get("intro")), "needs": must, "optional": may, "category": cat,
            "colour": (colours or {}).get(cat, UNPLACED_COLOUR), "source": tool.get("source", "")}


def needs_line(c: dict) -> str:
    """「需要：画面、摄影机」 as one sentence (a card's tooltip; the card itself draws chips)."""
    out = []
    if c["needs"]:
        out.append(text("dcc.ui.card.needs", types=text("list.sep").join(n for _t, n in c["needs"])))
    if c["optional"]:
        out.append(text("dcc.ui.card.optional", types=text("list.sep").join(n for _t, n in c["optional"])))
    return "\n".join(out) if out else text("dcc.ui.card.no_inputs")


# ---- search


def words_of(query: str) -> list[str]:
    return [w for w in str(query or "").lower().split() if w]


def matches(tool: dict, words: list[str]) -> bool:
    """Every word found (any case) in the tool's name, intro, category path or its inputs' type names."""
    if not words:
        return True
    hay = " ".join([str(tool.get("name") or ""), str(tool.get("intro") or ""),
                    " ".join((tool.get("category") or {}).get("path") or []),
                    " ".join(type_names(i.get("type", "")) + " " + str(i.get("type", "")) for i in tool.get("inputs") or [])]
                   ).lower()
    return all(w in hay for w in words)


# ---- the home page


def shelves(categories: list[dict]) -> set[str]:
    """The categories that are tabs: the first level of the templates tree (a node placed there — an algorithm's node —
    sits in its tab beside the templates); the node menu's own band (Lab2Shot's own nodes) is one tab, 「节点」."""
    return {str(c.get("id")) for c in categories or [] if not c.get("parent") and c.get("section", "templates") == "templates"}


def tabs(tools: list[dict], categories: list[dict], recent: list[str]) -> list[dict]:
    """[{id, label, colour}]: 全部 first; then every first-level category of the templates tree some tool sits in, in
    the server's order (工作流 first); the single nodes placed elsewhere (节点), the account's own templates and local
    graphs, each when there is one; 最近用过 last when there is one."""
    shelved = shelves(categories)
    used = {category_of(t) for t in tools if t.get("source") in ("preset", "node")}
    out = [{"id": ALL, "label": text("dcc.ui.tab.all"), "colour": ""}]
    for c in categories or []:
        if c.get("id") in shelved and c.get("id") in used:
            out.append({"id": c["id"], "label": str(c.get("label") or c["id"]), "colour": str(c.get("color") or "")})
    for source, key in catalog.SOURCES:
        if source != "preset" and any(in_tab(t, f"source:{source}", shelved) for t in tools):
            out.append({"id": f"source:{source}", "label": text(key), "colour": ""})
    known = {t.get("id") for t in tools}
    if any(r in known for r in recent or []):
        out.append({"id": RECENT, "label": text("dcc.ui.tab.recent"), "colour": ""})
    return out


def in_tab(tool: dict, tab: str, shelved: set[str]) -> bool:
    if tab in ("", ALL):
        return True
    if tab == NODES:
        return tool.get("source") == "node" and category_of(tool) not in shelved
    if tab.startswith("source:"):
        return tool.get("source") == tab.split(":", 1)[1]
    return tool.get("source") in ("preset", "node") and category_of(tool) == tab


def home(tools: list[dict], categories: list[dict], selected: list[str], search: str = "", tab: str = ALL,
         recent: list[str] | None = None, prefers=None, exports=None) -> dict:
    """{tabs, tab, suggested: [card], cards: [card]} (the module docstring). Ordered, never filtered by the DCC
    (catalog.score: what takes the selection first, then what the host prefers); `search` and `tab` are the user's own
    filters. 「最近用过」: the tools in the order they were last used."""
    colours = {c["id"]: c.get("color") or UNPLACED_COLOUR for c in categories or []}
    recent = list(recent or [])
    all_tabs = tabs(tools, categories, recent)
    if tab not in {t["id"] for t in all_tabs}:
        tab = ALL
    sources = [s for s, _key in catalog.SOURCES]

    def key(i, t):  # what takes the selection first; among equals the sources in their order (templates first)
        s = catalog.score(t, selected, i, prefers, exports)
        return (s[0], sources.index(t.get("source")) if t.get("source") in sources else len(sources), *s[1:])

    order = {id(t): key(i, t) for i, t in enumerate(tools)}
    words = words_of(search)
    found = [t for t in tools if matches(t, words)]
    if tab == RECENT:
        by_id = {t.get("id"): t for t in found}
        shown = [by_id[r] for r in recent if r in by_id]
    else:
        shelved = shelves(categories)
        shown = sorted((t for t in found if in_tab(t, tab, shelved)), key=lambda t: order[id(t)])
    suggested = []
    if selected and not words:
        fitting = [t for t in tools if order[id(t)][0] < 0]
        suggested = [card(t, colours) for t in sorted(fitting, key=lambda t: order[id(t)])[:SUGGEST_MOST]]
    return {"tabs": all_tabs, "tab": tab, "suggested": suggested, "cards": [card(t, colours) for t in shown]}


def auto_bind(tool: dict, selected: list[str], exports=None) -> list[dict]:
    """The inputs a 「适合当前选中」 card binds what is selected to: every required input that takes something selected;
    then an optional input only for a selected type no input took yet (the first that takes it), so one camera
    selected fills the camera slot, never also a reference picture or an HDRI slot as a side effect. An input takes
    what it reads (catalog.takes: the host's camera file goes into the input that reads that format)."""
    out, used = [], set()
    for i in tool.get("inputs") or []:
        if not i.get("optional"):
            took = [t for t in selected if catalog.takes(i, t, exports)]
            if took:
                out.append(i)
                used.update(took)
    for i in tool.get("inputs") or []:
        if i.get("optional"):
            free = [t for t in selected if t not in used and catalog.takes(i, t, exports)]
            if free:
                out.append(i)
                used.add(free[0])
    return out


REUSE, NEW = "reuse", "new"


def choose_node(node: str | None, scene_nodes: list[str], state: dict | None) -> str:
    """Which node a card chosen on the home page gives its tool to (one task, one node: a node's versions never end up
    under another tool): the node shown (`node`, `state` its state) when it is still in the scene and has no tool yet
    — REUSE; else a new node — NEW."""
    if not node or node not in (scene_nodes or []):
        return NEW
    return NEW if ((state or {}).get("tool") or {}).get("id") else REUSE


# ---- the top bar


def connection_state(conn, plugin) -> tuple[str, str]:
    """The connection's dot (a theme colour token) and its words: not connected (no server) / not logged in / the
    tool list failed / the tool list loading / ready (the server and who is logged in). `conn`: server, has_token(),
    user; `plugin`: tools, tools_error, loading."""
    if not conn.server:
        return "text-3", text("dcc.ui.top.offline")
    address = conn.server.split("://", 1)[-1]
    if not conn.has_token():
        return "orange", address
    if plugin.tools_error:
        return "error", address
    if plugin.loading or not plugin.tools:
        return "orange", address
    who = (conn.user or {}).get("username", "")
    return "green", address + (f" · {who}" if who else "")


# ---- an input card's 「用选中的」


def input_use(row: dict, selected: list[str], exports=None, host_label: str = "", why_not=None) -> dict:
    """Whether 「用选中的」 of an input card (`row`: one of `inputs`) can be pressed with this selection (its data
    types), and when not, why — {"enabled", "why"}: nothing fits; or the right kind in a file the input does not read
    (the host exports its camera as .usda, the input reads .fbx); else the host's own words (`why_not(type)`: what
    to select) or the general ones."""
    item = row["item"]
    if any(catalog.takes(item, t, exports) for t in selected):
        return {"enabled": True, "why": ""}
    fitting = [t for t in selected if catalog.fits(row["type"], t)]
    if fitting:
        return {"enabled": False, "why": text("dcc.plugin.wrong_format", host=host_label, suffix=catalog.suffix(fitting[0], exports),
                                              accept=" ".join(item.get("accept") or []))}
    return {"enabled": False, "why": (why_not(row["type"]) if why_not else "") or text("dcc.ui.input.nothing_fits")}


def recent_after(recent: list[str], tool_id: str) -> list[str]:
    """The list with this tool first (once), at most RECENT_MOST."""
    return [tool_id, *(r for r in recent if r != tool_id)][:RECENT_MOST]


# ---- the scene's tasks (「本场景的任务」)


def task_state(state: dict, snapshot: dict | None) -> str:
    """One word for a node's task: running / queued / failed / cancelled / done / imported / new."""
    if snapshot and not snapshot.get("done"):
        return "queued" if snapshot.get("phase") in ("starting", "queued") else "running"
    if snapshot and snapshot.get("phase") in ("failed", "cancelled"):
        return snapshot["phase"]
    job = (state or {}).get("job") or {}
    if job.get("state") in ("failed", "cancelled"):
        return job["state"]
    if (state or {}).get("versions"):
        return "imported" if all(v.get("imported", True) for v in state["versions"]) else "done"
    return "new"


STATE_WORDS = {"running": "dcc.ui.state.running", "queued": "dcc.ui.state.queued", "failed": "dcc.ui.state.failed",
               "cancelled": "dcc.ui.state.cancelled", "done": "dcc.ui.state.done", "imported": "dcc.ui.state.imported",
               "new": "dcc.ui.state.new"}  # a task's state -> its word's key


def task(node: str, name: str, state: dict, snapshot: dict | None = None, tool: dict | None = None) -> dict:
    """One of 「本场景的任务」; `tool` the node's tool as listed now (its name in the language now), else the name the
    node kept."""
    s = task_state(state, snapshot)
    named = (tool or {}).get("name") or ((state or {}).get("tool") or {}).get("name") or ""
    return {"node": node, "name": name, "tool": str(named),
            "state": s, "state_label": text(STATE_WORDS[s])}


# ---- the tool page


def _sequence_text(b: dict) -> str:
    """A bound picture sequence: 「首帧 … 尾帧 · N 帧」 (the frames sent: jobs.files_label's rule)."""
    return frames_text(b.get("files") or [])


def sequence_frames(path: str) -> list[str]:
    """The frames beside one frame of a numbered sequence (plate.1001.exr: every plate.####.exr in its folder), sorted;
    [] for a file that is not one (a video, a still). Reads the folder: called off the main thread."""
    name = os.path.basename(str(path or ""))
    if not jobs._FRAME.search(name) or name.lower().endswith(jobs.VIDEO):
        return []
    folder = os.path.dirname(str(path))
    shape = jobs._DIGITS.sub("#", name)
    try:
        return sorted(n for n in os.listdir(folder or ".") if jobs._DIGITS.sub("#", n) == shape)
    except OSError:
        return []


def frames_text(names: list[str]) -> str:
    """「首帧 … 尾帧 · N 帧」 of a sequence's file names ("" for fewer than two)."""
    if len(names) < 2:
        return ""
    return text("dcc.ui.input.frames", first=os.path.basename(names[0]), last=os.path.basename(names[-1]),
                count=len(names))


def inputs(tool: dict, state: dict, describe=None, frames=None) -> list[dict]:
    """[{param, label, type, type_label, required, bound, text, detail, sequence}] — one per input slot. `describe(binding)`
    is the host's own words for what is bound (the object's current name); `frames(binding)`: 「首帧 … 尾帧 · N 帧」
    for a picture the host carries (a cached background scan), or ""."""
    bindings = (state or {}).get("bindings") or {}
    out = []
    for i in tool.get("inputs") or []:
        b = bindings.get(i.get("param"))
        if b:
            said = describe(b) if describe else str(b.get("label") or b.get("file") or b.get("ref") or "")
            detail = _sequence_text(b) or (frames(b) if frames else "")
        else:
            said = text("dcc.ui.input.optional" if i.get("optional") else "dcc.ui.input.required")
            detail = ""
        out.append({"param": i.get("param"), "label": str(i.get("label") or i.get("param") or ""),
                     "type": str(i.get("type") or ""), "type_label": type_names(i.get("type", "")),
                     "required": not i.get("optional"), "bound": bool(b), "text": said, "detail": detail,
                     "sequence": i.get("widget") == "sequence", "item": i})
    return out


def parameters(tool: dict, state: dict) -> dict:
    """{"common": [row], "advanced": [row]}: the parameters the panel shows (contract.parameters, hidden ones left out by
    the interface's own conditions), split by the template's folded groups (`folded`: exposed_params). A row:
    {x, group (its groups' names), from_scene, disabled, note (the template's note, in the server's language; "" none)}."""
    values = contract.current(tool, state)
    scene = (state or {}).get("scene_values") or {}
    mine = (state or {}).get("values") or {}
    out: dict[str, list] = {"common": [], "advanced": []}
    for x in contract.parameters(tool):
        if contract.hidden(x, values):
            continue
        out["advanced" if x.get("folded") else "common"].append(
            {"x": x, "group": list(x.get("group") or []), "from_scene": x["name"] in scene and x["name"] not in mine,
             "disabled": contract.disabled(x, values), "value": values.get(x["name"]), "values": values,
             "note": str(x.get("note") or "")})
    return out


def missing_inputs(tool: dict, state: dict) -> list[str]:
    bindings = (state or {}).get("bindings") or {}
    return [str(i.get("label") or i["param"]) for i in tool.get("inputs") or []
            if not i.get("optional") and i["param"] not in bindings]


def versions(state: dict) -> list[dict]:
    """The node's result versions, newest first: {version, label (v003), made, tool, imported, status, objects, folder}."""
    out = []
    for v in reversed((state or {}).get("versions") or []):
        imported = bool(v.get("imported", True))
        out.append({"version": int(v.get("version") or 0), "label": f"v{int(v.get('version') or 0):03d}",
                    "made": str(v.get("made") or ""), "tool": str(v.get("tool") or ""), "imported": imported,
                    "status": text("dcc.ui.version.imported" if imported else "dcc.ui.version.not_imported"),
                    "objects": list(v.get("objects") or []), "folder": str(v.get("folder") or "")})
    return out


def run_line(snapshot: dict | None) -> str:
    """The status under 「计算」: the run's own words, the server's words when it failed, and the job id."""
    if not snapshot:
        return ""
    line = text("dcc.ui.run.error", status=snapshot["text"], error=snapshot["error"]) if snapshot.get("error") \
        else str(snapshot.get("text") or "")
    return line


def files_label(picked, sequence: bool) -> str:
    return jobs.files_label(picked, sequence)
