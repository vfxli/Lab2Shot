"""The normal ensemble: kernels on the unit sphere (lab2shot/nodes/kit/ensemble_normal.py) and who each candidate is
(nodes/core/ensemble_normal.py)."""

from __future__ import annotations

import unittest

import numpy as np

from lab2shot.nodes.kit import ensemble_normal as EN


def field(v, shape=(4, 5)):
    v = np.asarray(v, np.float64)
    return np.broadcast_to(v / np.linalg.norm(v), shape + (3,)).copy()


class Kernels(unittest.TestCase):
    def test_unit_normals_and_holes(self):
        n = np.array([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.0], [np.nan, 0.0, 1.0]]])
        u = EN.unit_normals(n)
        self.assertTrue(np.allclose(u[0, 0], [0, 0, 1]))
        self.assertTrue(np.isnan(u[0, 1]).all() and np.isnan(u[0, 2]).all())

    def test_vote_weights_one_model_one_vote(self):
        self.assertEqual(EN.vote_weights([3.0, 3.0, 1.0], [0, 0, 1]).tolist(), [1.5, 1.5, 1.0])

    def test_outlier_left_out(self):
        good = field([0, 0, 1])
        near = field([0.05, 0, 1])
        bad = field([1, 0, 0.2])
        N = np.stack([good, near, good, bad])
        W = np.ones(N.shape[:3])
        acc = EN.normal_consensus(N, W, "accuracy")
        self.assertLess(float(np.nanmax(EN.angle_deg(acc.value, good))), 2.0)
        self.assertFalse(acc.kept[3].any())  # the far one is out everywhere
        rob = EN.normal_consensus(N, W, "robust")
        self.assertLess(float(np.nanmax(EN.angle_deg(rob.value, good))), 2.0)
        self.assertTrue(set(np.unique(acc.nearest)) <= {0, 1, 2})

    def test_weights_and_regions(self):
        a, b = field([0, 0, 1]), field([0, 1, 1])
        N = np.stack([a, b])
        W = np.ones(N.shape[:3])
        W[1, :, :2] = 0.0  # the second candidate only counts on the right (a region)
        got = EN.normal_consensus(N, W, "accuracy")
        self.assertLess(float(np.nanmax(EN.angle_deg(got.value[:, :2], a[:, :2]))), 1e-6)
        self.assertGreater(float(np.nanmin(EN.angle_deg(got.value[:, 2:], a[:, 2:]))), 5.0)
        none = EN.normal_consensus(N, np.zeros_like(W), "accuracy")
        self.assertTrue(np.isnan(none.value).all() and (none.nearest == -1).all())

    def test_confidence(self):
        c = EN.normal_confidence(np.array([0.0, EN.NORMAL_FLOOR_DEG, np.nan]))
        self.assertEqual(c.tolist(), [1.0, 0.5, 0.0])


class Candidates(unittest.TestCase):
    @staticmethod
    def cand(name, sources, source=""):
        got = [{"id": l, "node": n, "label": l, "project": p, "commercial": c, "learned": True} for n, l, p, c in sources]
        return {"row": {"name": name, "label": ""}, "wire": {"id": source or name, "label": source or name},
                "sources": got, "learned": got, "commercial": all(s["commercial"] for s in got)}

    def test_the_reference_camera_is_nobodys_model(self):
        from lab2shot.nodes.core.ensemble_normal import EnsembleNormal

        vipe = ("vipe.camera_solve", "vipe", "ViPE", True)
        cands = [self.cand("candidate1", [vipe, ("moge.depth", "moge", "MoGe", True)], "moge"),
                 self.cand("candidate2", [("sapiens2.normal", "sap", "Sapiens2", False)], "sap"),
                 self.cand("candidate3", [vipe, ("depthanything3.depth", "da3", "Depth Anything 3", True)], "da3_normal"),
                 self.cand("candidate4", [vipe, ("unik3d.depth", "unik3d", "UniK3D", True)], "unik3d_normal"),
                 self.cand("candidate5", [vipe, ("moge.depth", "moge", "MoGe", True)], "moge_normal")]
        EnsembleNormal.identify(cands, derived=True)
        self.assertEqual([c["token"] for c in cands], ["moge", "sapiens2", "depthanything3", "unik3d", "moge"])
        self.assertEqual(EnsembleNormal.model_groups(cands), [0, 1, 2, 3, 0])  # MoGe's normals and MoGe's depth: one vote
        self.assertEqual([c["commercial"] for c in cands], [True, False, True, True, True])

    def test_node_registered(self):
        from lab2shot.nodes.core import CORE_NODES

        self.assertIn("ensemble_normal", {n.id for n in CORE_NODES})


if __name__ == "__main__":
    unittest.main()
