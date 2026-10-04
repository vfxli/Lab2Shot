"""A node of an extension that runs no model (BiRefNet's 前景色: a CPU blur fusion, NodeDef.learned False) is listed
among a delivery's sources but not as learned: the file name and the ML mark (nodes/output.py learned_projects) name
only the methods whose model really ran."""

from __future__ import annotations

import unittest

from lab2shot.engine.graph import Graph
from lab2shot.site import catalog


def setUpModule() -> None:
    catalog.install()


def _graph(nodes: list[dict], edges: list[tuple[str, str, str, str]]) -> Graph:
    return Graph.from_json({"schema": "lab2shot.graph/1", "nodes": nodes,
                            "edges": [{"from": [a, b], "to": [c, d]} for a, b, c, d in edges]})


def _sources(mask_node: dict, mask_port: str) -> list[dict]:
    from lab2shot.engine.evaluation import Evaluation
    from lab2shot.engine.templates import _empty_cache
    from lab2shot.serving import Account

    graph = _graph(
        [{"id": "read", "type": "file", "params": {}}, mask_node,
         {"id": "fg", "type": "birefnet.foreground_color", "params": {}},
         {"id": "out", "type": "image.output", "params": {"name": "fg"}},
         {"id": "deliver", "type": "output", "params": {}}],
        [("read", "image", mask_node["id"], "image"), ("read", "image", "fg", "image"),
         (mask_node["id"], mask_port, "fg", "mask"), ("fg", "foreground", "out", "image"),
         ("out", "files", "deliver", "files")])
    with _empty_cache():
        return Evaluation(graph, Account(1)).provenance("out")["sources"]


class ForegroundIsNotLearned(unittest.TestCase):
    def test_flags(self):
        from lab2shot.nodes import node_types

        types = node_types()
        self.assertFalse(types["birefnet.foreground_color"].learned)
        self.assertTrue(types["birefnet.matte"].learned)

    def test_hand_drawn_mask_names_no_model(self):
        from lab2shot.nodes.output import learned_projects

        sources = _sources({"id": "roto", "type": "draw_mask", "params": {}}, "mask")
        self.assertEqual([(s["node"], s["learned"]) for s in sources], [("birefnet.foreground_color", False)])
        self.assertEqual(learned_projects({"sources": sources}), [])

    def test_birefnet_matte_still_named(self):
        from lab2shot.nodes.output import learned_projects

        sources = _sources({"id": "matte", "type": "birefnet.matte", "params": {}}, "alpha")
        learned = {s["node"]: s["learned"] for s in sources}
        self.assertEqual(learned, {"birefnet.matte": True, "birefnet.foreground_color": False})
        self.assertEqual(len(learned_projects({"sources": sources})), 1)


if __name__ == "__main__":
    unittest.main()
