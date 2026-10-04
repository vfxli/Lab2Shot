"""`lab2shot check dcc`: the DCC plugins' zero-write rule, held without a DCC (clients/).

The plugins never write to anything that was in the user's scene (clients/maya/.../host.py, clients/nuke/.../host.py:
no attribute, knob, connection, input, name, parent, position, selection or current frame). Two guards here, both
plain Python:
- static: every function of the Maya and Nuke readers (reader.py) and Hosts (host.py) is read (AST) for the DCC's write
  calls (MAYA_WRITES / NUKE_WRITES …); only the functions that make or change the plugin's OWN nodes may have them
  (ALLOWED: a data table, each with its reason). Nested functions and lambdas count as the function they are in.
- the clients' unit tests that need no DCC and no server (UNIT_TESTS) are run.
The end-to-end tests (clients/maya/tests/test_maya.py in mayapy, clients/nuke/tests/test_nuke.py in nuke -t: every node
of the scene watched through binding, export and import) need the DCC and a running server: run once per release
(the release procedure)."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from .. import i18n

MAYA = "clients/maya/lab2shot/scripts/lab2shot_maya"
NUKE = "clients/nuke/lab2shot/lab2shot_nuke"
SCANNED = (("maya", f"{MAYA}/reader.py"), ("maya", f"{MAYA}/host.py"), ("nuke", f"{NUKE}/reader.py"), ("nuke", f"{NUKE}/host.py"))

# maya.cmds commands that write the scene; currentTime only when it is not a query (q / query=True)
MAYA_WRITES = frozenset({"setAttr", "connectAttr", "disconnectAttr", "addAttr", "deleteAttr", "rename", "parent", "delete",
                         "currentTime", "select", "lockNode", "createNode", "duplicate"})
MAYA_QUERIED = frozenset({"currentTime"})
# OpenMaya names that write (a modifier does what it is told; the active selection list)
MAYA_API_WRITES = frozenset({"MDGModifier", "MDagModifier", "setActiveSelectionList"})
# Nuke: methods of a node or knob that write, nuke.<function>s that write (nuke.frame only with a frame: setting it),
# and nuke.nodes.<Class>() (a new node)
NUKE_WRITES = frozenset({"setValue", "setInput", "addKnob", "removeKnob", "setSelected", "setName", "setXYpos",
                         "fromUserText", "readKnobs"})
NUKE_MODULE_WRITES = frozenset({"delete", "createNode"})
NUKE_SET_WITH_ARGS = frozenset({"frame"})

# The functions allowed to write: they make or change the plugin's own nodes and namespaces only (file -> function ->
# why). A name here that the file no longer has is a problem too (the table follows the code).
ALLOWED: dict[str, dict[str, str]] = {
    f"{MAYA}/host.py": {
        "_kept_state": "puts the user's current frame and selection back as they were, only when they changed",
        "MayaHost._root": "the plugin's own |Lab2Shot top group",
        "MayaHost.make_group": "a result version's own group",
        "MayaHost.hang_picture": "an image plane and frame curve of our own, on a camera the result brought in",
        "MayaHost.select": "selects a result version's own nodes (Host.locate)",
        "MayaHost.create_node": "the Lab2Shot node",
        "MayaHost.store": "the Lab2Shot node's state",
    },
    f"{NUKE}/host.py": {
        "_hidden_string": "a new knob for one of our nodes",
        "NukeHost.one_undo_step": "removes what a failed version step made",
        "NukeHost._new": "a new node of ours",
        "NukeHost.make_group": "a result version's own Backdrop",
        "NukeHost._place": "places a version's node in its Backdrop",
        "NukeHost._import_nk": "a delivered .nk's nodes, made new",
        "NukeHost._geo": "a GeoImport of ours",
        "NukeHost._pictures": "a Read and STMaps of ours (an STMap's input may be the user's Read: our node's input)",
        "NukeHost.import_result": "a point file's GeoReference of ours",
        "NukeHost.create_node": "the Lab2Shot node",
        "NukeHost.store": "the Lab2Shot node's state",
    },
}

# the clients' tests that need no DCC and no server (test_flow.py and test_reconnect.py need a running server)
UNIT_TESTS = ("clients/common/tests/test_units.py", "clients/common/tests/test_view_model.py", "clients/nuke/tests/test_units.py")
TEST_TIMEOUT_S = 300


def _queried(call: ast.Call) -> bool:
    return any(k.arg in ("q", "query") and isinstance(k.value, ast.Constant) and k.value.value is True for k in call.keywords)


def _maya_write(node: ast.AST) -> str:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
            and node.func.value.id == "cmds" and node.func.attr in MAYA_WRITES:
        if node.func.attr in MAYA_QUERIED and _queried(node):
            return ""
        return f"cmds.{node.func.attr}"
    if isinstance(node, ast.Attribute) and node.attr in MAYA_API_WRITES:
        return node.attr
    if isinstance(node, ast.Name) and node.id in MAYA_API_WRITES:
        return node.id
    return ""


def _nuke_write(node: ast.AST) -> str:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        f = node.func
        if isinstance(f.value, ast.Name) and f.value.id == "nuke":
            if f.attr in NUKE_MODULE_WRITES or (f.attr in NUKE_SET_WITH_ARGS and (node.args or node.keywords)):
                return f"nuke.{f.attr}"
            return ""
        if f.attr in NUKE_WRITES:
            return f".{f.attr}"
    if isinstance(node, ast.Attribute) and node.attr == "nodes" and isinstance(node.value, ast.Name) and node.value.id == "nuke":
        return "nuke.nodes"
    return ""


def _functions(tree: ast.Module):
    """(qualified name, node) of every top-level function and method; what is nested in one counts as it."""
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield item.name, item
        elif isinstance(item, ast.ClassDef):
            for sub in item.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield f"{item.name}.{sub.name}", sub


def writes(dcc: str, source: str) -> list[tuple[str, int, str]]:
    """(function, line, the write call) for every write API in a reader's or Host's source; "" as the function: at the
    module's top level."""
    tree = ast.parse(source)
    found = _maya_write if dcc == "maya" else _nuke_write
    out: list[tuple[str, int, str]] = []
    inside: set[int] = set()
    for name, fn in _functions(tree):
        for node in ast.walk(fn):
            inside.add(id(node))
            if said := found(node):
                out.append((name, getattr(node, "lineno", fn.lineno), said))
    for node in ast.walk(tree):
        if id(node) not in inside and (said := found(node)):
            out.append(("", getattr(node, "lineno", 0), said))
    return out


def _check_writes(r, root: Path) -> None:
    bad, scanned, functions = 0, 0, 0
    for dcc, rel in SCANNED:
        path = root / rel
        if not path.is_file():
            r.bad(i18n.t("cli.check.dcc.missing", file=rel))
            bad += 1
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        names = {name for name, _ in _functions(tree)}
        allowed = ALLOWED.get(rel, {})
        for gone in sorted(set(allowed) - names):
            r.bad(i18n.t("cli.check.dcc.allowed_gone", file=rel, function=gone))
            bad += 1
        scanned += 1
        functions += len(names - set(allowed))
        for name, line, call in writes(dcc, source):
            if name in allowed:
                continue
            r.bad(i18n.t("cli.check.dcc.write", file=rel, line=line, function=name or "<module>", call=call))
            bad += 1
    if not bad:
        r.ok(i18n.t("cli.check.dcc.ok", files=scanned, count=functions,
                    allowed=sum(len(v) for v in ALLOWED.values())))


def _check_tests(r, root: Path) -> None:
    bad = 0
    for rel in UNIT_TESTS:
        path = root / rel
        if not path.is_file():
            r.bad(i18n.t("cli.check.dcc.missing", file=rel))
            bad += 1
            continue
        try:
            done = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", cwd=root, timeout=TEST_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            r.bad(i18n.t("cli.check.dcc.test_timeout", file=rel, seconds=TEST_TIMEOUT_S))
            bad += 1
            continue
        if done.returncode:
            lines = (done.stdout + done.stderr).splitlines()
            failed = [x for x in lines if x.startswith("FAIL")] or lines[-3:]
            r.bad(i18n.t("cli.check.dcc.test_failed", file=rel, said=i18n.t("cli.check.common.sep").join(failed[:6])))
            bad += 1
    if not bad:
        r.ok(i18n.t("cli.check.dcc.tests_ok", count=len(UNIT_TESTS)))


def check_dcc(r, root: Path) -> None:
    """No write API in the DCC plugins' read side (only ALLOWED functions, which make the plugin's own nodes), and the
    clients' unit tests that need no DCC pass."""
    _check_writes(r, root)
    _check_tests(r, root)
