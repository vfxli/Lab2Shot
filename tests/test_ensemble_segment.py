"""The object segmentation ensemble: kernels (lab2shot/nodes/kit/ensemble_segment.py)."""

from __future__ import annotations

import unittest

import numpy as np

from lab2shot.nodes.kit import ensemble_segment as ES


def box(y0, y1, x0, x1, shape=(20, 20)):
    m = np.zeros(shape, np.float32)
    m[y0:y1, x0:x1] = 1
    return m


class Kernels(unittest.TestCase):
    def test_pick_objects(self):
        labels = np.zeros((20, 20))
        labels[2:10, 2:10] = 3
        labels[12:18, 12:18] = 5
        sel = box(2, 10, 2, 11) > 0
        self.assertEqual(ES.pick_objects(labels, sel), [3])
        # nothing mostly inside: the best overlap
        self.assertEqual(ES.pick_objects(labels, box(9, 14, 9, 14) > 0), [5])
        self.assertEqual(ES.pick_objects(np.zeros((4, 4)), np.ones((4, 4), bool)), [])

    def test_one_guide_one_vote(self):
        a = box(0, 10, 0, 10)
        b = box(0, 10, 10, 20)
        A = np.stack([a, a, b, b, b])
        kept = np.ones(5, bool)
        # groups: the two a's are one vote, the three b's three votes
        share = ES.segment_vote(A, kept, [0, 0, 1, 2, 3])
        self.assertAlmostEqual(float(share[5, 5]), 0.25)
        self.assertAlmostEqual(float(share[5, 15]), 0.75)
        share = ES.segment_vote(A, kept, [0, 0, 1, 1, 1])
        self.assertAlmostEqual(float(share[5, 5]), 0.5)

    def test_decide_tie_and_hysteresis(self):
        share = np.array([0.5, 0.4, 0.2, 0.7])
        prior = np.array([True, True, True, False])
        self.assertEqual(ES.segment_decide(share).tolist(), [False, False, False, True])
        self.assertEqual(ES.segment_decide(share, prior).tolist(), [True, False, False, True])
        self.assertEqual(ES.segment_decide(share, prior, ES.STABLE_LOW).tolist(), [True, True, False, True])

    def test_staple_trusts_the_reliable(self):
        truth = box(5, 15, 5, 15) > 0
        noisy = truth.copy()
        noisy[0:3] = True  # one voter adds a stripe
        V = np.stack([truth, truth, noisy])
        W, p, q = ES.staple(V)
        self.assertTrue(((W > 0.5) == truth).all())
        self.assertLess(q[2], q[0])

    def test_share_by_objective_and_confidence(self):
        a = box(0, 10, 0, 10)
        A = np.stack([a, a, 1 - a])
        kept = np.ones(3, bool)
        s = ES.segment_share(A, kept, [0, 1, 2], np.ones(3), "reliability")
        self.assertTrue(((s > 0.5) == (a > 0)).all())
        c = ES.segment_confidence(A, kept, a > 0)
        self.assertAlmostEqual(float(c[0, 0]), 2 / 3, places=5)
        self.assertEqual(ES.segment_confidence(A, np.zeros(3, bool), a > 0).max(), 0.0)

    def test_node_registered(self):
        from lab2shot.nodes.core import CORE_NODES

        self.assertIn("ensemble_segment", {n.id for n in CORE_NODES})


if __name__ == "__main__":
    unittest.main()
