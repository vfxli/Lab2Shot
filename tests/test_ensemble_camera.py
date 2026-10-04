"""The camera ensemble (nodes/core/ensemble_camera.py, nodes/kit/ensemble_camera.py)."""

from __future__ import annotations

import unittest

import numpy as np

from lab2shot.nodes.kit import ensemble as E
from lab2shot.nodes.kit import ensemble_camera as EC


class Cameras(unittest.TestCase):
    """The camera judge (held-out reprojection of independent tracks) and the pose medians."""

    @staticmethod
    def shot(frames=60, points=400, seed=0):
        from lab2shot_shared.motion import rotvec_to_matrix

        rng = np.random.default_rng(seed)
        c2w = np.repeat(np.eye(4)[None], frames, 0)
        for f in range(frames):
            c2w[f, :3, :3] = rotvec_to_matrix(np.array([0.0, 0.2 * f / frames, 0.0]))
            c2w[f, :3, 3] = [f * 2.0, 0.0, 0.0]  # cm
        K = np.repeat(np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])[None], frames, 0)
        X = rng.uniform([-300, -200, 400], [500, 200, 800], (points, 3))
        w2c = np.linalg.inv(c2w)
        cam = np.einsum("fij,nj->nfi", w2c[:, :3, :3], X) + w2c[None, :, :3, 3]
        uv = np.einsum("fij,nfj->nfi", K, cam / cam[..., 2:3])[..., :2] + rng.normal(0, 0.3, (points, frames, 2))
        vis = (uv[..., 0] > 0) & (uv[..., 0] < 640) & (uv[..., 1] > 0) & (uv[..., 1] < 480)
        return c2w, K, uv, vis

    def test_judge_sees_a_jump_and_not_the_scale(self):
        c2w, K, uv, vis = self.shot()
        good = np.nanmedian(EC.frame_errors(EC.held_out_errors(c2w, K, uv, vis)))
        self.assertLess(good, 1.0)
        bigger = c2w.copy()
        bigger[:, :3, 3] *= 3.0  # a whole world 3x as big fits the picture as well
        self.assertAlmostEqual(np.nanmedian(EC.frame_errors(EC.held_out_errors(bigger, K, uv, vis))), good, places=4)
        jumped = c2w.copy()
        jumped[30:, :3, 3] += [0.0, 10.0, 0.0]
        self.assertGreater(np.nanmedian(EC.frame_errors(EC.held_out_errors(jumped, K, uv, vis))[30:]), 2 * good)

    def test_medians_ignore_one_wild_candidate(self):
        from lab2shot_shared.motion import matrix_to_rotvec, rotvec_to_matrix

        R = np.stack([rotvec_to_matrix(np.array([0.0, 0.01 * k, 0.0])) for k in range(4)] + [rotvec_to_matrix(np.array([1.5, 0, 0]))])
        m = E.rotation_median(R)
        self.assertLess(np.degrees(np.linalg.norm(matrix_to_rotvec(m))), 2.0)
        x = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [100, 100, 100]])
        self.assertLess(np.linalg.norm(E.geometric_median(x) - [0.5, 0.5, 0]), 0.6)

    def test_segment_choice_and_stitch(self):
        scores = np.array([[1.0] * 10 + [5.0] * 10, [2.0] * 20])
        ch = EC.segment_choice(scores, np.array([True, True]), keep=1.2)
        self.assertEqual(ch.tolist(), [0] * 10 + [1] * 10)
        self.assertEqual(EC.segment_choice(scores, np.array([True, True]), keep=10.0).tolist(), [0] * 20)
        c2w = np.repeat(np.eye(4)[None, None], 2, 0).repeat(20, 1).copy()
        c2w[1, :, 0, 3] = 10.0
        out = EC.stitch(c2w, ch, 2)
        self.assertEqual(out[0, 0, 3], 0.0)
        self.assertEqual(out[-1, 0, 3], 10.0)
        self.assertTrue(0.0 < out[9, 0, 3] < 10.0)  # cross-faded at the switch

    def test_whole_error_and_confidence(self):
        j = np.array([[1.0, 1.0, np.nan], [np.nan, np.nan, np.nan], [2.0, 2.0, 2.0]])
        e = EC.whole_error(j, np.array([1.0, 1.0, 2.0]))
        self.assertEqual(e[0], 1.0)
        self.assertTrue(np.isinf(e[1]))
        self.assertEqual(e[2], 1.0)
        c = EC.camera_confidence(np.array([0.0, EC.CAMERA_CONF_PX, np.nan]))
        self.assertEqual(c.tolist(), [1.0, 0.5, 0.0])


    def test_combine_objectives(self):
        c2w, K, uv, vis = self.shot()
        wild = c2w.copy()
        wild[:, :3, 3] += np.linspace(0, 1, len(c2w))[:, None] * [0.0, 40.0, 0.0]  # drifts away from the picture
        cams = np.stack([c2w, wild, c2w + 0.0])
        judge = np.stack([EC.frame_errors(EC.held_out_errors(c, K, uv, vis)) for c in cams])
        for objective in EC.CAMERA_OBJECTIVES:
            got, choice, used = EC.combine(cams, judge, np.ones(3), objective)
            self.assertEqual(got.shape, c2w.shape)
            self.assertFalse(used[1], objective)  # the drifting one is never used
            self.assertNotIn(1, choice.tolist())
            self.assertLess(np.abs(got[:, :3, 3] - c2w[:, :3, 3]).max(), 1e-3, objective)

    def test_combine_without_a_judge_takes_the_median(self):
        c2w = np.repeat(np.eye(4)[None, None], 3, 0).repeat(5, 1).copy()
        c2w[0, :, 0, 3], c2w[1, :, 0, 3], c2w[2, :, 0, 3] = 0.0, 1.0, 100.0
        got, choice, used = EC.combine(c2w, np.full((3, 5), np.nan), np.ones(3), "robust")
        self.assertTrue(used.all())
        self.assertLess(abs(got[0, 0, 3] - 1.0), 1.0)
        self.assertEqual(choice.tolist(), [1] * 5)

    def test_still_base_is_kept(self):
        """The base camera barely moving: camera_space read no scale for the others, so no objective switches away from
        it, even when another candidate fits the tracks better."""
        c2w, K, uv, vis = self.shot()
        base = c2w.copy()
        base[:, :3, 3] = c2w[:, :3, 3] * 0.05  # spread about 5 cm: under camera_space's MOVED_CM
        base[30:, :3, 3] += [0.0, 0.5, 0.0]  # and a jump the tracks see
        cams = np.stack([c2w, base])  # the better-fitting one first; the base is the second ("same")
        judge = np.stack([EC.frame_errors(EC.held_out_errors(c, K, uv, vis)) for c in cams])
        self.assertLess(2 * np.nanmedian(judge[0]), np.nanmedian(judge[1]))
        fits = ["whole", "same"]
        self.assertEqual(EC.still_base(cams, fits, 10.0), 1)
        self.assertIsNone(EC.still_base(np.stack([c2w, c2w]), ["whole", "same"], 10.0))  # the base moved
        self.assertEqual(EC.still_base(np.stack([base, c2w]), [None, None], 10.0), 0)  # no "same": the first
        self.assertIsNone(EC.still_base(base[None], ["same"], 10.0))  # one candidate: nothing to switch to
        for objective in EC.CAMERA_OBJECTIVES:
            got, choice, used = EC.combine(cams, judge, np.ones(2), objective, only=EC.still_base(cams, fits, 10.0))
            self.assertTrue(np.array_equal(got, base), objective)
            self.assertEqual(set(choice.tolist()), {1})
            self.assertEqual(used.tolist(), [False, True])
            moved, _, _ = EC.combine(cams, judge, np.ones(2), objective, only=EC.still_base(np.stack([c2w, c2w]), fits, 10.0))
            self.assertFalse(np.array_equal(moved, base), objective)

    def test_still_threshold_is_camera_spaces(self):
        from lab2shot.nodes.core.scene import CameraSpaceConvert

        self.assertEqual(CameraSpaceConvert.MOVED_CM, 10.0)
        m = np.repeat(np.eye(4)[None], 2, 0)
        m[1, 0, 3] = 2 * 9.99
        self.assertLess(EC.spread_cm(m), CameraSpaceConvert.MOVED_CM)

    def test_allowed(self):
        self.assertEqual(EC.allowed(np.array([1.0, 1.9, 2.1, np.inf])).tolist(), [True, True, False, False])
        self.assertTrue(EC.allowed(np.array([np.inf, np.inf])).all())


class Node(unittest.TestCase):
    def test_registered_with_its_outputs(self):
        from lab2shot.nodes.core import CORE_NODES
        from lab2shot.nodes.core.ensemble_camera import EnsembleCamera

        self.assertIn(EnsembleCamera, CORE_NODES)
        self.assertEqual([p.name for p in EnsembleCamera.outputs], ["camera", "confidence", "choice", "scores"])
        ports = EnsembleCamera.made_ports({"candidates": [{"name": "candidate1", "weight": 1.0, "label": "ViPE"},
                                                          {"name": "candidate2", "weight": 1.0, "label": ""}]})
        self.assertEqual([p.type for p in ports], ["scene.camera"] * 2)

    def test_lens_in_the_first_candidates_filmback(self):
        from lab2shot.data.camera import CameraSamples
        from lab2shot.nodes.core.ensemble_camera import EnsembleCamera

        base = CameraSamples(frames=(1, 2), cam_to_world=np.repeat(np.eye(4)[None], 2, 0), focal_mm=np.array([35.0, 35.0]),
                             h_aperture_mm=np.array([36.0]), v_aperture_mm=np.array([20.25]), width=1920, height=1080)
        K = np.repeat(np.array([[1000.0, 0, 960], [0, 1000.0, 540], [0, 0, 1]])[None], 2, 0)
        K[1, 0, 0] = K[1, 1, 1] = 2000.0
        K[1, 0, 2] = 970.0
        got = EnsembleCamera._with_lens(base, [1, 2], np.repeat(np.eye(4)[None], 2, 0), K, 1920)
        self.assertTrue(np.allclose(got.focal_xy_px(1920)[:, 0], [1000.0, 2000.0]))
        self.assertTrue(np.allclose(got.principal_px(1920), [[960.0, 540.0], [970.0, 540.0]]))


if __name__ == "__main__":
    unittest.main()
