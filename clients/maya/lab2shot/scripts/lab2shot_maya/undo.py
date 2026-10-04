"""One result version as ONE undo step in Maya.

Maya's file import cannot be undone (`file -import` leaves its nodes behind on Ctrl+Z), and the steps after it (the
group, re-parenting, renames) cannot be redone once the imported nodes are gone — Maya crashes redoing a re-parent of
deleted nodes. So the whole version is done inside one command of our own, `lab2shotUndoStep` (registered by the
plug-in lab2shot/plug-ins/lab2shot_undo.py, plain Python, nothing compiled), with Maya's own undo recording off inside
it: nothing in it is journaled piece by piece. The command remembers what was there before (every node, every
namespace, every Lab2Shot node's state); undo deletes exactly the nodes and namespaces that came with it and puts the
Lab2Shot nodes' states back; redo runs the same work again from the files on disk (same names, freed by the undo).
A failure inside removes whatever the work made before it failed: never half an import.
"""

from __future__ import annotations

import os
import uuid

import maya.api.OpenMaya as om
import maya.cmds as cmds

COMMAND = "lab2shotUndoStep"
PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "plug-ins",
                      "lab2shot_undo.py")
ACTIONS: dict = {}
RESULTS: dict = {}


def _all_uuids() -> list[str]:
    """Every node's UUID (ls -uuid alone lists nothing: it needs the nodes)."""
    return cmds.ls(cmds.ls() or [], uuid=True) or []


def _namespaces() -> set[str]:
    return set(cmds.namespaceInfo(":", listOnlyNamespaces=True, recurse=True) or [])


def _states() -> dict:
    from .host import STATE

    out = {}
    for n in cmds.ls(type="network", long=True) or []:
        if cmds.attributeQuery(STATE, node=n, exists=True):
            out[cmds.ls(n, uuid=True)[0]] = cmds.getAttr(f"{n}.{STATE}") or ""
    return out


class _Recording:
    def __enter__(self):
        self.was = cmds.undoInfo(q=True, stateWithoutFlush=True)
        cmds.undoInfo(stateWithoutFlush=False)

    def __exit__(self, *exc):
        cmds.undoInfo(stateWithoutFlush=self.was)
        return False


def _remove(nodes: list[str], namespaces: list[str]) -> None:
    alive = cmds.ls(nodes, long=True) or []
    if alive:
        cmds.lockNode(alive, lock=False)
        cmds.delete(alive)
    for ns in sorted(namespaces, key=lambda n: -n.count(":")):
        if cmds.namespace(exists=":" + ns):
            cmds.namespace(removeNamespace=":" + ns, deleteNamespaceContent=True)


class Step(om.MPxCommand):
    def __init__(self):
        om.MPxCommand.__init__(self)
        self.key = ""
        self.created: list[str] = []
        self.namespaces: list[str] = []
        self.states: dict = {}

    @staticmethod
    def create():
        return Step()

    def isUndoable(self):  # noqa: N802 - Maya's name
        return True

    def doIt(self, args):  # noqa: N802
        self.key = args.asString(0)
        self.redoIt()

    def redoIt(self):  # noqa: N802
        nodes, spaces = set(_all_uuids()), _namespaces()
        self.states = _states()
        with _Recording():
            try:
                RESULTS[self.key] = (ACTIONS[self.key](), None)
            except Exception as exc:  # noqa: BLE001 - handed to `run`, after what it made is gone
                import traceback

                from lab2shot_dcc import log

                log.get().error("an import (undo step) failed: %s\n%s", exc, traceback.format_exc())
                RESULTS[self.key] = (None, exc)
            self.created = [u for u in _all_uuids() if u not in nodes]
            self.namespaces = sorted(_namespaces() - spaces)
            if RESULTS[self.key][1] is not None:
                _remove(self.created, self.namespaces)
                self._put_back_states()

    def undoIt(self):  # noqa: N802
        with _Recording():
            _remove(self.created, self.namespaces)
            self._put_back_states()

    def _put_back_states(self):
        from .host import STATE

        for ref, text in self.states.items():
            node = cmds.ls(ref, long=True)
            if node and (cmds.getAttr(f"{node[0]}.{STATE}") or "") != text:
                cmds.setAttr(f"{node[0]}.{STATE}", text, type="string")


def run(label: str, fn):
    """`fn()` as one undo step; returns what it returns (raises what it raised, after removing what it made)."""
    if not cmds.pluginInfo("lab2shot_undo", q=True, loaded=True):
        cmds.loadPlugin(PLUGIN, quiet=True)
    key = uuid.uuid4().hex
    ACTIONS[key] = fn
    getattr(cmds, COMMAND)(key)  # ACTIONS[key] stays: redo runs it again
    result, error = RESULTS.pop(key, (None, RuntimeError(f"{label}: did not run")))
    if error is not None:
        ACTIONS.pop(key, None)
        raise error
    return result
