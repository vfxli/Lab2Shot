"""The top bar's 「DCC 插件」 and its downloads follow the setting plugins.download (server/plugins.py): off by default,
the administrators' only while it is off, everyone's once it is on."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from lab2shot.errors import Forbidden


def _session(caps: frozenset[str]) -> SimpleNamespace:
    """A browser session whose role holds `caps` (none: a 普通用户), its rights fresh."""
    return SimpleNamespace(user=SimpleNamespace(id=7, capabilities=caps), capabilities=caps, lapsed=False)


def _listed(request: object) -> dict:
    """GET /api/plugins as its JSON answer."""
    from lab2shot.server import plugins

    got = plugins.plugins(request)
    return json.loads(got.body) if hasattr(got, "body") else got


def _shown(s: SimpleNamespace) -> bool:
    """The top bar offers 「DCC 插件」 to this session (subject plugins.download, as the login state resolves it)."""
    from lab2shot.availability import resolve
    from lab2shot.roles import SessionFacts
    from lab2shot.server import available

    return "plugins.download" in resolve({"plugins.download": available.ACTIONS["plugins.download"]}, SessionFacts.of(s)).available


USER = _session(frozenset())
ADMIN = _session(frozenset({"settings.edit"}))


class PluginDownloads(unittest.TestCase):
    def setUp(self) -> None:
        from lab2shot.server import app  # noqa: F401  (the routes declared, as the server has them)

    def _with(self, on: bool, s: SimpleNamespace):
        return (mock.patch("lab2shot.config.settings", return_value={"plugins.download": on, "server.https": False}),
                mock.patch("lab2shot.server.auth.session", return_value=s))

    def test_off_by_default(self):
        from lab2shot.config import SCHEMA

        self.assertIs(SCHEMA["plugins.download"].default, False)
        self.assertEqual(SCHEMA["plugins.download"].kind, "bool")

    def test_off_only_administrators_see_and_download(self):
        from lab2shot.server import plugins

        for s, allowed in ((USER, False), (ADMIN, True)):
            a, b = self._with(False, s)
            with a, b:
                self.assertEqual(_shown(s), allowed)
                if allowed:
                    got = _listed(mock.Mock())
                    self.assertFalse(got["open"])
                    self.assertTrue(got["plugins"])
                    self.assertEqual(plugins.download("maya", mock.Mock()).media_type, "application/zip")
                else:
                    for call in (lambda: plugins.plugins(mock.Mock()), lambda: plugins.download("maya", mock.Mock())):
                        with self.assertRaises(Forbidden) as got:
                            call()
                        self.assertEqual(got.exception.message.code, "E-PLUGIN-CLOSED")

    def test_on_every_user_downloads(self):
        from lab2shot.server import plugins

        a, b = self._with(True, USER)
        with a, b:
            self.assertTrue(_shown(USER))
            self.assertTrue(_listed(mock.Mock())["open"])
            self.assertEqual(plugins.download("maya", mock.Mock()).media_type, "application/zip")


if __name__ == "__main__":
    unittest.main()
