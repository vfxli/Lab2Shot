"""The depth ensemble: kernels in log depth (lab2shot/nodes/kit/ensemble_depth.py) and the node's registration
(nodes/core/ensemble_depth.py)."""

from __future__ import annotations

import unittest

import numpy as np

from lab2shot.nodes.kit import ensemble_depth as E


class Bands(unittest.TestCase):
    def test_blur_skips_holes(self):
        b = np.full((5, 5), 2.0)
        b[2, 2] = np.nan
        self.assertAlmostEqual(float(E.blur(b, 1)[2, 2]), 2.0)  # NaN is not counted

    def test_guided_lowpass_keeps_a_step_on_the_guide(self):
        guide = np.zeros((20, 20))
        guide[:, 10:] = 1.0
        src = guide * 2.0 + 1.0
        out = E.guided_lowpass(src, guide, 3, 1e-4)
        self.assertLess(abs(out[10, 8] - 1.0), 0.05)
        self.assertLess(abs(out[10, 11] - 3.0), 0.05)


class Depth(unittest.TestCase):
    def test_consensus_drops_the_odd_one_per_pixel(self):
        L = np.log(np.stack([np.full((4, 4), v) for v in (100.0, 101.0, 99.0, 300.0)]))
        c = E.consensus(L)
        self.assertTrue(np.allclose(np.exp(c.value), 100.0, rtol=0.01))
        self.assertFalse(c.kept[3].any())
        self.assertTrue((c.nearest != 3).all())

    def test_whole_candidate_rejected_only_with_three_independent(self):
        rng = np.random.default_rng(1)
        base = rng.uniform(4, 6, (30, 30))
        good = [base + rng.normal(0, 0.01, base.shape) for _ in range(3)]
        bad = base[::-1] * 1.0  # another shape altogether
        dis = E.disagreement(np.stack(good + [bad]))
        out = E.outliers(dis)
        self.assertEqual(out.tolist(), [False, False, False, True])
        # two candidates (or two groups) cannot outvote each other
        self.assertFalse(E.outliers(dis[:2]).any())
        self.assertFalse(E.outliers(dis, groups=[0, 0, 0, 1]).any())

    def test_reprojection_error_is_zero_for_a_consistent_depth(self):
        z = np.full((40, 60), 500.0)
        K = np.array([[50.0, 0, 30], [0, 50.0, 20], [0, 0, 1]])
        e = E.reprojection_error(z, z, K, K, np.eye(4), 2)
        self.assertLess(np.nanmax(e), 1e-9)
        # a wall that moved 10 % further away in the other view is 10 % off there
        e = E.reprojection_error(z, z * 1.1, K, K, np.eye(4), 2)
        self.assertAlmostEqual(float(np.nanmedian(e)), 1 / 11, places=6)

    def test_scale_consensus(self):
        # candidates were each multiplied by ~0.5 to match the reference: the reference thinks the scene half as big
        self.assertAlmostEqual(E.scale_consensus([0.5, 0.5, 0.52]), 2.0, places=6)
        self.assertIsNone(E.scale_consensus([0.5]))
        self.assertIsNone(E.scale_consensus([None, -1.0]))

    def test_split_bands(self):
        low = np.full((16, 16), 2.0)
        high = np.full((16, 16), 5.0)
        high[8, 8] = 6.0
        out = E.split_bands(low, high, 3)
        self.assertAlmostEqual(float(out[0, 0]), 2.0, places=6)
        self.assertGreater(out[8, 8], 2.5)  # the high band's detail is kept

    def test_confidence(self):
        c = E.depth_confidence(np.array([0.0, 0.01, 1.0]))
        self.assertAlmostEqual(float(c[0]), 1.0)
        self.assertAlmostEqual(float(c[1]), 0.5)
        self.assertLess(c[2], 0.02)




class Node(unittest.TestCase):
    def test_registered_with_its_objectives(self):
        from lab2shot.nodes.core import CORE_NODES
        from lab2shot.nodes.core.ensemble_depth import EnsembleDepth

        self.assertIn(EnsembleDepth, CORE_NODES)
        objective = EnsembleDepth.Params.model_fields["objective"]
        self.assertEqual(objective.default, "accuracy")
        self.assertEqual(set(E.DEPTH_OBJECTIVES), {"accuracy", "scale", "temporal"})

    def test_sampled_frames_spread_evenly(self):
        from lab2shot.nodes.core.ensemble_depth import PRE_FRAMES, EnsembleDepth

        got = EnsembleDepth.sampled(list(range(100)))
        self.assertEqual(len(got), PRE_FRAMES)
        self.assertEqual((got[0], got[-1]), (0, 99))
        self.assertEqual(EnsembleDepth.sampled([5, 6]), [5, 6])


if __name__ == "__main__":
    unittest.main()
