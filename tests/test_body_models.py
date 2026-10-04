"""「标准人」's bodies as a registry (lab2shot/data/standard_bodies.py, P(options_from=), Extension.standard_bodies):
an extension's body registers, becomes a 「骨架」 option with its own name, is greyed with why while its data is not
installed, and the core names none but SMPL-X.

    uv run python -m unittest tests.test_body_models
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lab2shot import i18n
from lab2shot.site import catalog
from lab2shot.data import standard_bodies as sb
from lab2shot.errors import Invalid


def _fake_make(height_cm=None):
    return "character", {"model": "TEST", "joints": 1, "vertices": 0, "faces": 0, "height_cm": height_cm or 0}


class BodyRegistry(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        catalog.install()
        from lab2shot.nodes.registry import node_types

        cls.node = node_types()["standard_human"]

    def tearDown(self) -> None:
        sb.forget("testext")

    def options(self) -> dict:
        spec = next(p for p in self.node.param_specs() if p["name"] == "skeleton")
        return {"options": spec["options"], "labels": spec["option_labels"]}

    def test_core_names_only_its_own(self) -> None:
        bodies = sb.standard_bodies()
        self.assertEqual(next(iter(bodies)), sb.CORE_BODY)
        self.assertEqual({b.owner for b in bodies.values() if not b.owner}, {""})
        self.assertEqual([b.id for b in bodies.values() if not b.owner], [sb.CORE_BODY])

    def test_kimodo_bodies_registered_by_its_extension(self) -> None:
        bodies = sb.standard_bodies()
        for body_id in ("soma", "g1"):
            self.assertIn(body_id, bodies)
            self.assertEqual(bodies[body_id].owner, "kimodo")
            self.assertEqual(bodies[body_id].owner_title, "Kimodo")

    def test_registered_body_is_an_option_with_its_words(self) -> None:
        sb.register(sb.StandardBody("testbody", "TEST", _fake_make, word="node.standard_human.param.skeleton.label"),
                    owner="testext", owner_title="TestExt")
        got = self.options()
        self.assertIn("testbody", got["options"])
        with i18n.using("en"):
            self.assertEqual(self.options()["labels"]["testbody"], i18n.t("node.standard_human.param.skeleton.label"))

    def test_core_body_cannot_be_taken(self) -> None:
        sb.register(sb.StandardBody(sb.CORE_BODY, "X", _fake_make), owner="testext")
        self.assertEqual(sb.standard_bodies()[sb.CORE_BODY].owner, "")

    def test_licence_of_a_choice_comes_from_the_registry(self) -> None:
        from lab2shot.nodes.applies import licensed_choices

        sb.register(sb.StandardBody("testnc", "TEST", _fake_make, licence="research"), owner="testext")
        got = licensed_choices(self.node)["skeleton"]
        self.assertIn("noncommercial", got[sb.CORE_BODY])
        self.assertIn("registration", got[sb.CORE_BODY])
        self.assertEqual(got["testnc"], {"research"})

    def test_missing_data_greys_the_option_and_refuses_the_cook(self) -> None:
        from lab2shot.nodes.applies import NodeFacts

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("lab2shot_shared.body_models.ROOT", Path(tmp)):
                sb.register(sb.StandardBody("testbody", "TEST", _fake_make, files=("skin.npz",)),
                            owner="testext", owner_title="TestExt")
                body = sb.standard_bodies()["testbody"]
                self.assertIsNotNone(body.lacking())
                answer = self._availability(NodeFacts(params={"skeleton": sb.CORE_BODY, "height_cm": None}))
                self.assertIn("skeleton=testbody", answer.inactive)
                self.assertEqual(answer.inactive["skeleton=testbody"].code, "I-BODY-NOTINSTALLED")
                with self.assertRaises(Invalid) as said:
                    sb.standard_body("testbody")
                self.assertEqual(said.exception.code, "E-BODY-NOEXTDATA")
                (Path(tmp) / "testbody").mkdir()
                (Path(tmp) / "testbody" / "skin.npz").write_bytes(b"")
                self.assertIsNone(body.lacking())
                self.assertEqual(sb.standard_body("testbody", 170)[1]["height_cm"], 170)
                answer = self._availability(NodeFacts(params={"skeleton": sb.CORE_BODY, "height_cm": None}))
                self.assertNotIn("skeleton=testbody", answer.inactive)

    def _availability(self, facts):
        from lab2shot.nodes.applies import resolve

        return resolve(self.node, facts).params

    def test_unknown_skeleton_is_refused(self) -> None:
        with self.assertRaises(Invalid) as said:
            sb.standard_body("no_such_body")
        self.assertEqual(said.exception.code, "E-BODY-UNKNOWN")


if __name__ == "__main__":
    unittest.main()
