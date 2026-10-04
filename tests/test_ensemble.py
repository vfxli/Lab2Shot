"""The ensemble kernels (lab2shot/nodes/kit/ensemble.py) and the ensemble nodes' bookkeeping (nodes/core/ensemble.py):
robust statistics, the object-level checks, the names results go by, and who each candidate is."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from lab2shot.nodes.kit import ensemble as E
from lab2shot.site import catalog


def setUpModule() -> None:
    catalog.install()


class Statistics(unittest.TestCase):
    def test_weighted_median(self):
        x = np.array([1.0, 2.0, 10.0])[:, None]
        self.assertAlmostEqual(float(E.wmedian(x)[0]), 2.0)
        # a heavy weight pulls the median to its value
        self.assertAlmostEqual(float(E.wmedian(x, np.array([1.0, 1.0, 5.0])[:, None])[0]), 10.0)
        # an even count splits exactly: the two middle values averaged
        self.assertAlmostEqual(float(E.wmedian(np.array([1.0, 2.0, 3.0, 4.0])[:, None])[0]), 2.5)
        # NaN is no value; all NaN gives NaN
        self.assertAlmostEqual(float(E.wmedian(np.array([np.nan, 3.0, 5.0])[:, None])[0]), 4.0)
        self.assertTrue(np.isnan(E.wmedian(np.array([np.nan, np.nan])[:, None])[0]))

    def test_box_sum(self):
        a = np.ones((5, 7))
        self.assertEqual(E.box_sum(a, 1)[2, 3], 9.0)
        self.assertEqual(E.box_sum(a, 1)[0, 0], 4.0)  # clipped at the corner


class Matte(unittest.TestCase):
    def test_identity_vote(self):
        person = np.zeros((20, 20), np.float32)
        person[5:15, 5:10] = 1
        other = np.zeros((20, 20), np.float32)
        other[5:15, 12:18] = 1
        # three groups, two agree: the majority wins
        self.assertEqual(E.identity_vote(np.stack([person, person, other]), None).tolist(), [True, True, False])
        # two that disagree: what is known of the object settles it
        self.assertEqual(E.identity_vote(np.stack([other, person]), person).tolist(), [False, True])
        # without a prior a tie keeps both
        self.assertEqual(E.identity_vote(np.stack([other, person]), None).tolist(), [True, True])

    def test_group_means_and_fuse(self):
        A = np.stack([np.full((4, 4), v, np.float32) for v in (0.0, 0.2, 1.0)])
        order, reps = E.group_means(A, [0, 0, 1])
        self.assertEqual(order, [0, 1])
        self.assertAlmostEqual(float(reps[0, 0, 0]), 0.1)
        out, near = E.matte_fuse(A, np.array([True, True, False]), np.ones(3))
        self.assertAlmostEqual(float(out[0, 0]), 0.1, places=6)
        self.assertTrue(set(np.unique(near)) <= {0, 1})

    def test_band_from_the_source(self):
        A = np.zeros((3, 10, 10), np.float32)
        A[:, :, 5:] = 1.0
        A[0, :, 4] = 0.5  # the candidates disagree on column 4 only
        A[2, :, 4] = 0.8
        out, near = E.matte_fuse(A, np.ones(3, bool), np.ones(3), source=2)
        self.assertAlmostEqual(float(out[0, 4]), 0.8, places=6)
        self.assertAlmostEqual(float(out[0, 0]), 0.0)
        self.assertEqual(int(near[0, 4]), 2)


class Trusted(unittest.TestCase):
    """Leaning on the most trusted candidate where the candidates disagree (集一b)."""

    def test_share_grows_with_the_disagreement(self):
        t = E.trusted_share(np.array([0.0, 0.02, 0.2, np.nan]), 0.02)
        self.assertEqual(float(t[0]), 0.0)  # they agree: the combination stands
        self.assertAlmostEqual(float(t[1]), 0.5)  # at the floor: half and half
        self.assertGreater(float(t[2]), 0.9)  # wide disagreement: mostly the trusted one
        self.assertEqual(float(t[3]), 0.0)  # no measure: no lean

    def test_lean_keeps_holes_and_takes_what_there_is(self):
        value = np.array([1.0, 1.0, np.nan, 1.0])
        trusted = np.array([3.0, 3.0, 3.0, np.nan])
        u = np.array([0.0, 0.02, 1.0, 1.0])
        out, share = E.lean_on_trusted(value, trusted, u, 0.02)
        self.assertAlmostEqual(float(out[0]), 1.0)
        self.assertAlmostEqual(float(out[1]), 2.0)
        self.assertTrue(np.isnan(out[2]))  # the combination has no value (most candidates say none): stays none
        self.assertAlmostEqual(float(out[3]), 1.0)  # the trusted one has none: the combination as is
        self.assertEqual(float(share[3]), 0.0)

    def test_matte_deviation(self):
        A = np.stack([np.full((2, 2), v, np.float32) for v in (0.2, 0.4, 1.0)])
        d = E.matte_deviation(A, np.ones(3, bool), 2)
        self.assertTrue(np.allclose(d, 0.7))
        self.assertTrue(np.allclose(E.matte_deviation(A, np.array([False, False, True]), 2), 0.0))

    def test_edges_objective_leans_on_the_trusted_alpha(self):
        A = np.zeros((3, 6, 6), np.float32)
        A[:, :, 3:] = 1.0
        A[2, :, 2] = 0.9  # the trusted one alone sees a soft edge on column 2
        mean, _ = E.matte_fuse(A, np.ones(3, bool), np.array([1.0, 1.0, 2.0]))
        out, share = E.lean_on_trusted(mean.astype(np.float64), A[2].astype(np.float64),
                                       E.matte_deviation(A, np.ones(3, bool), 2), E.TRUSTED_MATTE_DEVIATION)
        self.assertAlmostEqual(float(out[0, 0]), 0.0)  # agreement: the mean stands
        self.assertGreater(float(out[0, 2]), float(mean[0, 2]))  # disagreement: towards the trusted one
        self.assertGreater(float(share[0, 2]), 0.5)


class Names(unittest.TestCase):
    def test_one_rule_for_every_result(self):
        self.assertEqual(E.ensemble_name("depth", "accuracy", ["unik3d", "moge"]), "depth_ensemble_accuracy_moge_unik3d")
        # the order of the wires does not matter, nor does case or punctuation
        self.assertEqual(E.ensemble_name("depth", "accuracy", ["MoGe", "unik3d"]),
                         E.ensemble_name("depth", "accuracy", ["unik3d", "moge"]))

    def test_long_names_are_abbreviated_the_same_way(self):
        models = ["depthanything3", "moge", "unidepth", "unik3d", "videodepthanything"]
        name = E.ensemble_name("depth", "temporal", models)
        self.assertLessEqual(len(name), E.NAME_MOST)
        self.assertEqual(name, "depth_ensemble_temporal_dept_moge_unid_unik_vide")
        many = [f"model{i}" for i in range(12)]
        long = E.ensemble_name("matte", "accuracy", many)
        self.assertTrue(long.startswith("matte_ensemble_accuracy_12models_"))
        self.assertEqual(long, E.ensemble_name("matte", "accuracy", list(reversed(many))))
        self.assertRegex(long, r"^[a-z0-9_]+$")

    def test_licence(self):
        self.assertTrue(E.strictest_licence([True, True]))
        self.assertFalse(E.strictest_licence([True, False]))


class Candidates(unittest.TestCase):
    """Who each candidate is, from the graph's lineage of its wire (nodes/core/ensemble.py)."""

    @staticmethod
    def cand(name, sources, source=""):
        """`sources`: (node type, node id, project, commercial[, learned]); the wire comes from `source` (a node id)."""
        got = [{"id": s[1], "node": s[0], "label": s[1], "project": s[2], "commercial": s[3],
                "learned": s[4] if len(s) > 4 else True} for s in sources]
        return {"row": {"name": name, "label": ""}, "wire": {"id": source or name, "label": source or name},
                "sources": got, "learned": [s for s in got if s["learned"]],
                "commercial": all(s["commercial"] for s in got)}

    def test_the_shared_reference_is_nobodys_model(self):
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        vipe = [("vipe.camera_solve", "vipe", "ViPE", True), ("vipe.depth", "vipe_depth", "ViPE", True)]
        cands = [self.cand("candidate1", [("unik3d.depth", "unik3d1", "UniK3D", True), *vipe]),
                 self.cand("candidate2", [("moge.depth", "moge1", "MoGe", True), *vipe]),
                 self.cand("candidate3", vipe)]
        EnsembleMatte.identify(cands)
        self.assertEqual([c["token"] for c in cands], ["unik3d", "moge", "vipe"])
        self.assertEqual(cands[0]["title"], "UniK3D")

    def test_guide_groups_and_licence(self):
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        rough = ("birefnet.matte", "rough", "BiRefNet", True)
        cands = [self.cand("candidate1", [rough]),
                 self.cand("candidate2", [rough, ("videomama.matte", "vm", "VideoMaMa", False)], "vm"),
                 self.cand("candidate3", [("sam3.segment", "sam", "SAM 3", True), ("matanyone.matte", "ma", "MatAnyone 2", False)], "ma")]
        self.assertEqual(EnsembleMatte.guide_groups(cands), [0, 0, 1])
        EnsembleMatte.identify(cands)
        self.assertEqual([c["commercial"] for c in cands], [True, False, False])
        self.assertEqual(cands[1]["token"], "videomama")

    def test_a_classical_solver_is_a_candidate_too(self):
        """Who it is does not hang on `learned` (COLMAP runs no model): every third-party project above it counts."""
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        cands = [self.cand("candidate1", [("vipe.camera_solve", "vipe", "ViPE", True)], "vipe"),
                 self.cand("candidate2", [("colmap.camera_solve", "colmap", "COLMAP", True, False)], "colmap")]
        EnsembleMatte.identify(cands)
        self.assertEqual([c["token"] for c in cands], ["vipe", "colmap"])
        self.assertEqual(EnsembleMatte.guide_groups(cands), [0, 1])

    def test_named_by_node_id_not_by_label(self):
        """Two source nodes of one label in a user's graph: the one the wire comes from is found by its id."""
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        a = self.cand("candidate1", [("birefnet.matte", "m1", "BiRefNet", True), ("sdmatte.matte", "m2", "SDMatte", True)], "m2")
        b = self.cand("candidate2", [("matanyone.matte", "m3", "MatAnyone 2", True)], "m3")
        for s in (*a["sources"], *b["sources"]):
            s["label"] = "matte"  # the same label everywhere
        EnsembleMatte.identify([a, b])
        self.assertEqual([a["token"], b["token"]], ["sdmatte", "matanyone"])

    def test_derived_candidates_take_their_nearest_own_model(self):
        """Normals computed from a depth (derived) share the camera every depth took: nobody's own."""
        from lab2shot.nodes.core.ensemble import _Ensemble

        cam = ("vipe.camera_solve", "vipe", "ViPE", True)
        cands = [self.cand("candidate1", [("moge.normal", "moge_n", "MoGe", True)], "moge_n"),
                 self.cand("candidate2", [cam, ("unik3d.depth", "u", "UniK3D", True)], "n_from_u"),
                 self.cand("candidate3", [cam, ("moge.depth", "moge_d", "MoGe", True)], "n_from_m")]
        _Ensemble.identify(cands, derived=True)
        self.assertEqual([c["token"] for c in cands], ["moge", "unik3d", "moge"])
        self.assertEqual(_Ensemble.model_groups(cands), [0, 1, 2])  # MoGe's normals and MoGe's depth: two node types

    def test_the_licence_warning_names_every_project(self):
        """A project above a candidate that runs no model still counts for the licence and is named."""
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        said = []
        ctx = SimpleNamespace(say=lambda code, **kw: said.append((code, kw)))
        cands = [self.cand("candidate1", [("birefnet.matte", "m", "BiRefNet", True)], "m"),
                 self.cand("candidate2", [("x.prep", "p", "PrepTool", False, False), ("sdmatte.matte", "s", "SDMatte", True)], "s")]
        self.assertFalse(EnsembleMatte.say_licence(ctx, cands))
        self.assertEqual(said, [("W-ENSEMBLE-NONCOMMERCIAL", {"projects": ["PrepTool"]})])

    def test_the_candidates_licence_is_the_wires(self):
        from lab2shot.nodes.core.ensemble import EnsembleMatte

        lineage = {"candidate1": [{"id": "m", "label": "m", "commercial": True,
                                   "sources": [{"id": "m", "node": "birefnet.matte", "label": "m", "project": "BiRefNet",
                                                "commercial": True, "learned": True}]}],
                   "candidate2": [{"id": "s", "label": "s", "commercial": False,
                                   "sources": [{"id": "p", "node": "x.prep", "label": "p", "project": "PrepTool",
                                                "commercial": False, "learned": False}]}]}
        ctx = SimpleNamespace(params={"candidates": [{"name": "candidate1"}, {"name": "candidate2"}]},
                              input=lambda name: object(), lineage=lineage)
        cands = EnsembleMatte.candidates(ctx)
        self.assertEqual([c["commercial"] for c in cands], [True, False])
        self.assertEqual([len(c["sources"]) for c in cands], [1, 1])
        self.assertEqual([len(c["learned"]) for c in cands], [1, 0])


class Lineage(unittest.TestCase):
    """engine/evaluation.py lineage: per wire, the source node's id and label and what made it as delivered (the same
    account a delivery's sidecar gives: delivered_provenance)."""

    @staticmethod
    def evaluation():
        from lab2shot.engine.evaluation import Evaluation
        from lab2shot.engine.graph import Graph
        from lab2shot.serving import Account

        nodes = [{"id": "read", "type": "file", "params": {}},
                 {"id": "rough", "type": "birefnet.matte", "params": {}},
                 {"id": "vm", "type": "videomama.matte", "params": {}},
                 {"id": "ens", "type": "ensemble_matte", "params": {}}]
        edges = [("read", "image", "rough", "image"), ("read", "image", "vm", "image"), ("rough", "alpha", "vm", "mask"),
                 ("rough", "alpha", "ens", "candidate1"), ("vm", "alpha", "ens", "candidate2")]
        graph = Graph.from_json({"schema": "lab2shot.graph/1", "nodes": nodes,
                                 "edges": [{"from": [a, b], "to": [c, d]} for a, b, c, d in edges]})
        return Evaluation(graph, Account(1))

    # nothing is cooked here (the plate has no file): every node above is taken as having given something, but the one
    # a test leaves out
    @staticmethod
    def gave(*nothing):
        from lab2shot.engine.evaluation import Evaluation

        return mock.patch.object(Evaluation, "_gave_something", lambda self, nid, at: nid not in nothing)

    def test_ids_sources_and_licence(self):
        from lab2shot.engine.templates import _empty_cache

        with _empty_cache(), self.gave():
            got = self.evaluation().lineage("ens")
        one, two = got["candidate1"][0], got["candidate2"][0]
        self.assertEqual((one["id"], two["id"]), ("rough", "vm"))
        self.assertEqual([s["id"] for s in two["sources"]], ["rough", "vm"])
        self.assertTrue(one["commercial"])
        self.assertFalse(two["commercial"])  # VideoMaMa: non-commercial

    def test_a_source_that_gave_nothing_is_left_out(self):
        """The same rule as a delivery's: a node that gave nothing this time is not among what made it, and the
        licence is that of the rest."""
        from lab2shot.engine.templates import _empty_cache

        with _empty_cache(), self.gave("vm"):
            ev = self.evaluation()
            two = ev.lineage("ens")["candidate2"][0]
            delivered = ev.delivered_provenance("vm")
        self.assertEqual([s["id"] for s in two["sources"]], ["rough"])
        self.assertTrue(two["commercial"])
        self.assertEqual(two["sources"], delivered["sources"])
        self.assertEqual(len(ev.provenance("vm")["sources"]), 2)  # the whole graph's account keeps it


if __name__ == "__main__":
    unittest.main()
