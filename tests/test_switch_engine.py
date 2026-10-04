"""Two engine regressions around 「切换」 (switch), each on the smallest graph that showed it:
- a 「有没有」 (has_data) driving the switch one of whose ways the node it tells of feeds: planning went round in a
  circle (present -> the source's outcome -> failure -> cached -> the outputs the switch takes -> the 「有没有」 again)
  and /api/status died of a RecursionError;
- a switch inside a switch, every way of the inner one from an import port that is not there yet (no file picked):
  the outer one's type stayed open (the viewer drew it as a value)."""

from __future__ import annotations

import dataclasses
import unittest
from unittest import mock

from lab2shot.site import catalog
from lab2shot.engine.graph import Graph
from lab2shot.messages import Msg


def setUpModule() -> None:
    catalog.install()


def _graph(nodes: list[dict], edges: list[tuple[str, str, str, str]]) -> Graph:
    return Graph.from_json({"schema": "lab2shot.graph/1", "nodes": nodes,
                            "edges": [{"from": [a, b], "to": [c, d]} for a, b, c, d in edges]})


class PresenceDrivesSwitch(unittest.TestCase):
    """gen feeds both the 「有没有」 and the switch's first way; the 「有没有」 picks the way (on: the first)."""

    GRAPH = (
        [{"id": "gen", "type": "diffusers.generate", "params": {}},
         {"id": "gen2", "type": "diffusers.generate", "params": {}},
         {"id": "has", "type": "has_data", "params": {}},
         {"id": "pick", "type": "switch", "params": {}},
         {"id": "image_out", "type": "image.output", "params": {"name": "result"}},
         {"id": "deliver", "type": "output", "params": {}}],
        [("gen", "image", "has", "data"), ("gen", "image", "pick", "a"), ("gen2", "image", "pick", "b"),
         ("has", "value", "pick", "param:which"), ("pick", "out", "image_out", "image"),
         ("image_out", "files", "deliver", "files")],
    )

    def _check(self, gen_state: str) -> None:
        from lab2shot.engine.evaluation import Evaluation
        from lab2shot.engine.templates import _empty_cache
        from lab2shot.serving import Account

        with _empty_cache():
            ev = Evaluation(_graph(*self.GRAPH), Account(1))
            nodes = ev.status()["nodes"]  # a RecursionError here was the bug
            for nid, entry in nodes.items():
                self.assertNotEqual((entry.get("error") or {}).get("code"), "E-FARM-INTERNAL", nid)
            self.assertEqual(nodes["gen"]["state"], gen_state)
            self.assertTrue(ev.params("has")["present"])
            self.assertEqual(ev.taken_ports("pick"), frozenset({"param:which", "a"}))

    def test_status_plans_and_takes_the_way_present_picks(self):
        self._check("todo")

    def test_extension_not_usable_still_plans(self):
        # an extension that can't be used now asks whether the node has its result (failure): that must not ask
        # the switch's route, which hangs on the 「有没有」 being worked out
        from adapters.diffusers.nodes import Generate as gen_type  # noqa: N813 (the node class)

        broken = dataclasses.replace(gen_type.project, available=lambda: Msg("E-COOK-UNAVAILABLE", node="x", reason="x"))
        with mock.patch.object(gen_type, "project", broken):
            self._check("failed")


class NestedSwitchTypes(unittest.TestCase):
    """inner: an FBX's and a USD's 角色 (no file picked: neither port is there yet); outer: inner, or a BVH's 骨架."""

    def test_outer_switch_takes_the_ways_declared_types(self):
        g = _graph(
            [{"id": "fbx", "type": "fbx.import", "params": {}},
             {"id": "usd", "type": "usd.import", "params": {}},
             {"id": "bvh", "type": "bvh.import", "params": {}},
             {"id": "inner", "type": "switch", "params": {}},
             {"id": "outer", "type": "switch", "params": {}}],
            [("fbx", "characters", "inner", "a"), ("usd", "characters", "inner", "b"),
             ("inner", "out", "outer", "a"), ("bvh", "skeletons", "outer", "b")],
        )
        self.assertEqual(g.output_type("inner", "out"), "scene.character")
        self.assertEqual(set(g.output_type("outer", "out").split("|")), {"scene.character", "scene.skeleton"})
        self.assertEqual(g.ways("outer", "out"), ["scene.character", "scene.skeleton"])

    def test_a_way_not_known_yet_is_left_out(self):
        # the inner switch has nothing wired at all: its open type says nothing, the outer takes its other way's
        g = _graph(
            [{"id": "bvh", "type": "bvh.import", "params": {}},
             {"id": "inner", "type": "switch", "params": {}},
             {"id": "outer", "type": "switch", "params": {}}],
            [("inner", "out", "outer", "a"), ("bvh", "skeletons", "outer", "b")],
        )
        self.assertEqual(g.output_type("outer", "out"), "scene.skeleton")


if __name__ == "__main__":
    unittest.main()


class PresenceOfAnAbsentPort(unittest.TestCase):
    """A skeleton-only FBX leaves 角色 unselected (its port absent). The wire from it into a 「有没有」 is the
    answer 「没有」 (Evaluation.comes), not a wire that waits; into anything else it still waits."""

    def test_presence_takes_an_absent_port_as_nothing(self):
        g = _graph(
            [{"id": "fbx", "type": "fbx.import", "params": {"path": "x.fbx", "skeletons": ["/a"], "characters": []}},
             {"id": "has", "type": "has_data", "params": {}},
             {"id": "pick", "type": "switch", "params": {}}],
            [("fbx", "characters", "has", "data"), ("fbx", "characters", "pick", "a")])
        self.assertNotIn("has", g.wiring)
        self.assertEqual([m.code for _, m in g.wiring["pick"]], ["B-WIRE-WAITS"])
