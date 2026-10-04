"""The core's mechanisms that nothing else runs before a user meets them: words (i18n), the database's migrations, the
cache's one deletion path, delivering one content once, the architecture checks, and the small contracts the first
architecture review fixed (errors answered field by field, reserved usernames, a lens group's name, the 「尺度」 of a
metric method, the licences kept on disk)."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from lab2shot import i18n
from lab2shot.site import catalog

ROOT = Path(__file__).resolve().parents[1]


def setUpModule() -> None:
    catalog.install()  # the node registry as the server and the command line have it


# ------------------------------------------------------------------ words


class Words(unittest.TestCase):
    def test_lookup_in_each_language(self):
        self.assertEqual(i18n.t("param.step.label", in_lang="zh"), "隔帧")
        self.assertEqual(i18n.t("param.step.label", in_lang="en"), "Increment")
        with i18n.using("en"):
            self.assertEqual(i18n.t("param.step.label"), "Increment")

    def test_node_keys_fall_back_to_the_shared_ones(self):
        self.assertEqual(i18n.fallbacks("node.x.y.param.step.label"), ["node.x.y.param.step.label", "param.step.label"])
        self.assertEqual(i18n.node_text("no_such.node", "param", "step", "label", in_lang="en"), "Increment")

    def test_a_key_in_no_catalogue_is_an_error(self):
        with self.assertRaises(i18n.TextError):
            i18n.t("no.such.key.anywhere")

    def test_lang_is_a_placeholder_like_any_other(self):
        # A-H2: t()'s own keywords are in_lang / in_scope, so {lang} is filled, not taken as the language
        said = i18n.t("cli.check.templates.too_wide", name="x", what="intro", lang="en", width=9, most=8, in_lang="en")
        self.assertIn("(en)", said)
        said = i18n.t("cli.check.messages.short_wide", code="E-X", lang="zh", width=9, most=8, in_lang="zh")
        self.assertIn("（zh）", said)

    def test_no_placeholder_is_named_like_a_keyword_of_t(self):
        from lab2shot.i18n.lint import Catalogues, reserved_problems

        self.assertEqual(reserved_problems(Catalogues.read()), [])
        fake = Catalogues({lang: {"x.y": ("{in_lang} {n}", Path("x.toml"))} for lang in i18n.LANGS}, {})
        self.assertEqual(len(reserved_problems(fake)), len(i18n.LANGS))

    def test_plural_forms(self):
        self.assertEqual(i18n.plural_form("en", 1), "one")
        self.assertEqual(i18n.plural_form("en", 2), "other")
        self.assertEqual(i18n.plural_form("zh", 1), "other")

    def test_a_message_says_itself_to_a_card_user(self):
        # a message's 「.app」 words (no node names, no wires) travel beside its text, nested
        # messages said in theirs too; a code without them has none
        from lab2shot.messages import Msg, localized

        root = Msg("E-COOK-FAILED", node="read (file)", reason="no image sequence chosen")
        m = Msg("E-OUTPUT-CHAINFAILED", node="deliver (output)", count=1, chains=[Msg("I-OUTPUT-CHAIN", chain="a", root="read")],
                reasons=[root])
        with i18n.using("en"):
            said = m.json()
            self.assertIn("deliver (output)", said["text"])
            self.assertNotIn("deliver", said["app"])
            self.assertTrue(said["app"].endswith("no image sequence chosen"))
            # a code with no app words of its own whose words name a node or a wire is said to a card's user
            # as its level's general sentence (no node name reaches a card); one that names none has no app words
            self.assertEqual(Msg("I-OUTPUT-CHAIN", chain="a", root="read").json()["app"], Msg("I-APP-INSIDE").text)
            self.assertNotIn("read", Msg("B-SWITCH-NOWIRE", node="read", way="b").app)
            self.assertNotIn("app", Msg("W-COOK-NONCOMMERCIAL", projects=["ViPE"]).json())
            waits = Msg("B-WIRE-WAITS", source="motion_fbx (fbx.import)", what="skeleton animation", waits="one is chosen",
                        node="motion_pick (switch)", input="Skeleton")
            self.assertNotIn("motion_", waits.app)
            self.assertNotIn("wire", waits.app)
        with i18n.using("zh"):
            again = localized(said)
            self.assertNotEqual(again["app"], said["app"])  # said again in the reader's language
            self.assertTrue(again["app"].startswith("还不能打包"))

    def test_a_tracker_with_nothing_to_track_is_held_before_the_cook(self):
        # no grid and no point clicked is a B- note (the plan stops there, the button greys with it)
        from lab2shot.nodes.families import PointTracker

        self.assertEqual([m.code for m, _ in PointTracker.wiring_notes({"grid": 0, "picks": []}, {})], ["B-TRACKS-NOQUERY"])
        self.assertEqual(PointTracker.wiring_notes({"grid": 5, "picks": []}, {}), [])

    def test_retarget_ignore_rules_have_their_names(self):
        # the 「忽略规则」 menu of the AI retarget cards (no 500, no 「找不到了」)
        from lab2shot.nodes.core.retarget import RetargetPrepare

        with i18n.using("en"):
            got = RetargetPrepare.choices({}, {})["ignore_rule"]
        self.assertIn("star", got["options"])
        self.assertEqual(set(got["labels"]), set(got["options"]))
        self.assertTrue(all(isinstance(v, str) and v and v != k for k, v in got["labels"].items()))


# ------------------------------------------------------------------ the database


def _schema(conn: sqlite3.Connection) -> list[tuple]:
    return sorted(conn.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"))


class Migrations(unittest.TestCase):
    def _made(self, upto: int) -> sqlite3.Connection:
        from lab2shot.database import schema

        conn = sqlite3.connect(":memory:", autocommit=True)
        conn.execute("PRAGMA foreign_keys=ON")
        for to, _what, sql in schema.pending(0):
            if to > upto:
                break
            conn.execute("BEGIN IMMEDIATE")
            conn.executescript(sql)
            conn.execute(f"PRAGMA user_version = {to}")
            conn.execute("COMMIT")
        return conn

    def test_every_older_version_migrates_to_the_same_schema(self):
        from lab2shot.database import schema

        made = self._made(schema.VERSION)
        want = _schema(made)
        made.close()
        for start in range(schema.FIRST, schema.VERSION):
            with self.subTest(start=start):
                conn = self._made(start)
                self.addCleanup(conn.close)
                for to, _what, sql in schema.pending(start):
                    conn.execute("BEGIN IMMEDIATE")
                    conn.executescript(sql)
                    conn.execute(f"PRAGMA user_version = {to}")
                    conn.execute("COMMIT")
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], schema.VERSION)
                self.assertEqual(_schema(conn), want)
                self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_a_new_database_and_an_older_one(self):
        from lab2shot import workdir
        from lab2shot.database import Database, DatabaseError, schema

        with tempfile.TemporaryDirectory() as work:
            new = Database(Path(work))
            try:
                self.assertEqual(new.version, schema.VERSION)
                self.assertTrue(new.row("SELECT id FROM users WHERE id = 1"))
            finally:
                new.close()
        with tempfile.TemporaryDirectory() as work:
            workdir.claim(Path(work))
            (Path(work) / "db").mkdir()
            conn = self._made(schema.VERSION - 1)
            disk = sqlite3.connect(Path(work) / "db" / "lab2shot.db")
            conn.backup(disk)
            disk.close()
            conn.close()
            with self.assertRaises(DatabaseError) as refused:
                Database(Path(work))
            self.assertEqual(refused.exception.code, "E-DB-OLDER")
            upgraded = Database(Path(work), upgrade=True)
            try:
                self.assertEqual(upgraded.version, schema.VERSION)
                self.assertTrue(any((Path(work) / "db").rglob(f"*before-v{schema.VERSION}*")))
            finally:
                upgraded.close()


# ------------------------------------------------------------------ the cache


class InstalledMigrations(unittest.TestCase):
    """installer/migrate.py: a record rewritten for the new fingerprint rule, journalled and undone."""

    def test_a_record_is_renamed_and_the_journal_undoes_it(self):
        from lab2shot.extensions.build_state import _hash
        from lab2shot.installer.migrate import Journal, _renamed, undo

        state = {"env": {"fingerprint": "old"}, "plan": "old", "selfcheck": {"fingerprint": "old", "ok": True},
                 "steps": {"repo": {"fingerprint": "r"}, "switch": {"fingerprint": _hash(["old", "r"])}}}
        new = _renamed(state, "old", "new")
        self.assertEqual((new["env"]["fingerprint"], new["plan"], new["selfcheck"]["fingerprint"]), ("new",) * 3)
        self.assertEqual(new["steps"]["switch"]["fingerprint"], _hash(["new", "r"]))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a").mkdir()
            record = d / "a" / "install_state.json"
            record.write_text(json.dumps(state))
            (d / "a" / ".venv").mkdir()
            j = Journal(d / "journal")
            j.write(record, new)
            j.move(d / "a" / ".venv", d / "b" / ".venv")
            j.write(d / "b" / "install_state.json", {"x": 1})
            self.assertEqual(undo(d / "journal"), [])
            self.assertEqual(json.loads(record.read_text()), state)
            self.assertTrue((d / "a" / ".venv").is_dir())
            self.assertFalse((d / "b" / ".venv").exists() or (d / "b" / "install_state.json").exists())
            self.assertEqual(undo(d / "journal"), [])  # undone once: nothing more


class CacheRemoval(unittest.TestCase):
    def test_remove_takes_the_entry_and_what_hangs_off_it(self):
        from lab2shot.data import packet

        with tempfile.TemporaryDirectory() as cache, mock.patch.object(packet, "cache_root", lambda: Path(cache)):
            for name in ("abc", "abc_display", "abc_work", "abd"):
                (Path(cache) / name).mkdir()
                (Path(cache) / name / packet.COMPLETE).write_text("")
            heard = []
            packet.on_removed(lambda fp, why: heard.append((fp, why)))
            try:
                self.assertTrue(packet.remove("abc", "clean"))
                self.assertEqual(sorted(p.name for p in Path(cache).iterdir()), ["abd"])
                self.assertEqual(heard, [("abc", "clean")])
                self.assertFalse(packet.remove("abc", "clean"))  # gone: nothing to say again
                self.assertTrue(packet.remove("abd_display", "x") is False)
            finally:
                packet._on_removed.remove(packet._on_removed[-1])

    def test_a_fingerprint_names_a_folder_and_nothing_else(self):
        from lab2shot.data.packet import packet_dir
        from lab2shot.errors import Invalid

        for bad in ("../x", "a/b", "a.b", "名字"):
            with self.subTest(bad=bad), self.assertRaises(Invalid):
                packet_dir(bad)


# ------------------------------------------------------------------ deliveries


class OneDeliveryPerContent(unittest.TestCase):
    def _outputs(self, data: dict) -> list[str]:
        return sorted(n["type"] for n in data["nodes"] if n["type"].endswith(".output"))

    def test_a_kind_moved_where_it_is_already_delivered_is_left_out(self):
        from lab2shot.engine.deliver_formats import with_formats

        card = json.loads((ROOT / "templates" / "camera_pi3.json").read_text(encoding="utf-8"))
        before = self._outputs(card)
        self.assertIn("nuke.output", before)
        # the camera asked in USD: the card's USD already writes it, so the Nuke node is left with nothing and goes
        self.assertNotIn("nuke.output", self._outputs(with_formats(card, {"camera": "usd"})))
        # asked as a Nuke camera: the card's Nuke node already writes it; no second Nuke node is made
        self.assertEqual(self._outputs(with_formats(card, {"camera": "nuke_camera"})).count("nuke.output"), 1)
        self.assertIs(with_formats(card, {}), card)


# ------------------------------------------------------------------ the architecture checks


class Architecture(unittest.TestCase):
    def test_every_route_naming_an_account_s_data_declares_its_owner(self):
        from lab2shot.cli.check_arch import route_problems
        from lab2shot.server.app import app
        from lab2shot.server.routes import DECLARED, Access

        n, problems = route_problems(app, DECLARED)
        self.assertGreater(n, 100)
        self.assertEqual(problems, [])
        unowned = {k: (Access.user("x") if " /api/jobs/{job_id}" in k and k.startswith("GET") else a)
                   for k, a in DECLARED.items()}
        self.assertTrue(route_problems(app, unowned)[1])
        # a request body naming jobs: undeclared, or declared without an owner, is found
        from dataclasses import replace

        key = "POST /api/feedback/jobs"
        self.assertEqual(DECLARED[key].body_ids, ("jobs",))
        for wrong in (replace(DECLARED[key], body_ids=()), replace(DECLARED[key], owned=None),
                      replace(DECLARED[key], body_ids=("jobs", "nothing"))):
            self.assertTrue(route_problems(app, {**DECLARED, key: wrong})[1])

    def test_imports_go_down_the_layers(self):
        from lab2shot.cli.check_arch import layer_of, layer_problems

        self.assertEqual(layer_problems(ROOT)[1], [])
        self.assertLess(layer_of("", "data"), layer_of("", "nodes"))
        self.assertLess(layer_of("", "accounts"), layer_of("", "farm"))
        self.assertEqual(layer_of("lab2shot/i18n/lint.py", "i18n"), layer_of("", "cli"))

    def test_an_upward_import_is_found(self):
        from lab2shot.cli.check_arch import layer_problems

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "lab2shot" / "data").mkdir(parents=True)
            (Path(root) / "lab2shot" / "data" / "x.py").write_text("def f():\n    from ..farm import queue\n")
            (Path(root) / "adapters" / "demo").mkdir(parents=True)
            (Path(root) / "adapters" / "demo" / "nodes.py").write_text("from lab2shot.messages import Msg\n")
            problems = layer_problems(Path(root))[1]
        self.assertEqual(len(problems), 2)

    def test_keys_are_made_by_io_digest(self):
        from lab2shot.cli.check_arch import digest_problems

        self.assertEqual(digest_problems(ROOT)[1], [])


# ------------------------------------------------------------------ accounts


class AccountRemoval(unittest.TestCase):
    """site/account_removal.py: deleting and purging go through every part above the accounts (the test's own database,
    tests/__init__.py; the queue left out: no job of the account is there)."""

    def test_delete_then_purge(self):
        import time

        from lab2shot import accounts, roles
        from lab2shot.site import account_removal
        from lab2shot.errors import Invalid

        dep = accounts.departments()[0]
        u = accounts.create("removal_case", accounts.new_password("A-long-pass-9x"), "RemovalCase",
                            dep, time.time() + 86400, [], role=roles.DEFAULT)
        accounts.set_quota_gb(u.id, 7)
        self.assertEqual(accounts.quota_gb(u.id), 7.0)
        idle = mock.Mock(stop_user=mock.Mock(return_value=0), live_of=mock.Mock(return_value=[]))
        with mock.patch("lab2shot.farm.farm", return_value=idle):
            with self.assertRaises(Invalid):  # not deleted yet: purge refuses
                account_removal.purge(u.id)
            self.assertEqual(account_removal.delete(u.id)["jobs_stopped"], 0)
            self.assertTrue(accounts.get(u.id).deleted)
            done = account_removal.purge(u.id)
        self.assertEqual(set(done), {"jobs", "feedback", "graphs"})
        self.assertIsNone(accounts.quota_gb(u.id))  # the row is gone


# ------------------------------------------------------------------ small contracts


class Contracts(unittest.TestCase):
    def test_field_errors_answer_field_by_field_whatever_the_status(self):
        from lab2shot.config import InvalidSettings
        from lab2shot.messages import Msg

        problems = {"server.port": Msg("E-SETTINGS-NEEDS", setting="x", label="y", roles="z")}
        for status in (None, 403):
            got = InvalidSettings(problems, status)
            answer = got.answer()
            self.assertEqual(got.status, status or 400)
            self.assertEqual(answer["code"], "E-SETTINGS-INVALID")
            self.assertEqual(set(answer["errors"]), {"server.port"})
            self.assertEqual(answer["codes"], {"server.port": "E-SETTINGS-NEEDS"})

    def test_reserved_usernames_are_the_template_owners(self):
        from lab2shot import accounts
        from lab2shot.site import library

        self.assertEqual(accounts.RESERVED_USERNAMES, {library.OWNER_ADMIN, library.OWNER_ADAPTER})

    def test_a_lens_value_is_said_with_its_group_s_name(self):
        from lab2shot.data.values import LENS, _one
        from lab2shot.nodes.registry import node_types

        node_types()  # the node layer names the groups when it loads (nodes/lens.py)
        said = _one(LENS, {"group": "3de4", "model": "m", "params": {"k1": 0.5, "center_x_mm": 1.0, "pixel_aspect": 1.0}})
        self.assertIn("3DE4 · m", said)
        self.assertIn("k1 0.5", said)
        self.assertNotIn("center_x_mm", said)

    def test_scale_applies_only_to_a_method_of_relative_scale(self):
        from lab2shot.availability import resolve
        from lab2shot.nodes.applies import facts_for, param_conditions
        from lab2shot.nodes.params import param_defaults
        from lab2shot.nodes.registry import node_types

        types = node_types()
        if "pi3.reconstruct" not in types:
            self.skipTest("the Pi3 extension is not loaded here")
        t = types["pi3.reconstruct"]
        cond = {"unit_cm": param_conditions(t)["unit_cm"]}

        def scale(model: str):
            got = resolve(cond, facts_for(t, {**param_defaults(t.Params), "model": model}), t)
            return "on" if "unit_cm" in got.available else got.inactive.get("unit_cm") or got.pending.get("unit_cm")

        self.assertTrue(t.is_metric({"model": "pi3x"}))
        self.assertFalse(t.is_metric({"model": "pi3"}))
        self.assertEqual(scale("pi3"), "on")
        self.assertEqual(scale("pi3x").code, "I-SCALE-METRIC")  # greyed with its reason, not hidden

    def test_every_depth_node_declares_metric_and_a_worker_cannot_say_otherwise(self):
        from lab2shot.availability import Cond
        from lab2shot.errors import Invalid
        from lab2shot.nodes.families.depth_camera import DepthCamera
        from lab2shot.nodes.registry import node_types

        depth = {tid: t for tid, t in node_types().items() if issubclass(t, DepthCamera)}
        self.assertTrue(depth)
        self.assertIsNone(DepthCamera.metric)  # no default to forget it by
        self.assertEqual([tid for tid, t in depth.items() if not isinstance(t.metric, (bool, Cond))], [])

        t = next(t for t in depth.values() if t.metric is True)
        ctx = mock.Mock(params={}, label="n")

        def raw(said):
            r = mock.Mock()
            r.path.return_value.exists.return_value = said is not ...
            r.result.return_value = {} if said is None else {"metric": said}
            return r

        self.assertTrue(t.agrees_metric(ctx, raw(True)))
        self.assertTrue(t.agrees_metric(ctx, raw(None)))  # a worker that says nothing: the declaration
        self.assertTrue(t.agrees_metric(ctx, raw(...)))  # no result.json
        with self.assertRaises(Invalid):
            t.agrees_metric(ctx, raw(False))

    def test_licences_are_kept_on_disk_by_the_card_and_the_registry(self):
        from lab2shot.site import library

        card = ROOT / "templates" / "camera_pi3.json"
        st = card.stat()
        library._licences.cache_clear()
        library._kept.cache_clear()
        first = library._licences(str(card), st.st_mtime_ns, st.st_size, library.registry_print())
        kept = json.loads((Path(library.settings().work_dir) / library.LICENCES_FILE).read_text(encoding="utf-8"))
        self.assertEqual(kept["registry"], library.registry_print())
        self.assertIn(first, kept["cards"].values())
        library._licences.cache_clear()
        with mock.patch("lab2shot.engine.templates.route_tags", side_effect=AssertionError("judged again")):
            self.assertEqual(library._licences(str(card), st.st_mtime_ns, st.st_size, library.registry_print()), first)
            library._licences.cache_clear()
            with self.assertRaises(AssertionError):  # another registry: worked out again
                library._licences(str(card), st.st_mtime_ns, st.st_size, "another registry")


if __name__ == "__main__":
    unittest.main()


class CardVisibility(unittest.TestCase):
    """What touches anything not given is not seen. A card with one node the account may not use is
    not listed at all (no route judged, nothing greyed); one whose every node it may use is; a login that manages the
    templates sees them all."""

    def test_one_node_not_given_hides_the_card(self):
        from types import SimpleNamespace

        from lab2shot.nodes import node_types
        from lab2shot.nodes import tags
        from lab2shot.server import access

        types = node_types()
        plain = next(k for k, t in types.items() if tags.node_tags(t) <= tags.IMPLIED | {tags.COMMERCIAL})
        gen = next(k for k, t in types.items() if tags.GENERATIVE in tags.node_tags(t))
        given_gen = frozenset({tags.COMMERCIAL, *tags.node_tags(types[gen])})
        strict = next(k for k, t in types.items() if not tags.may(tags.node_tags(t), given_gen))

        def card(cid, *kinds):
            return {"id": cid, "enabled": True, "graph": {"nodes": [{"id": f"n{i}", "type": k} for i, k in enumerate(kinds)]}}

        cards = [card("ok", plain), card("research", plain, strict), card("generative", plain, gen),
                 card("unknown", plain, "no_such.node"), {**card("off", plain), "enabled": False}]
        user = SimpleNamespace(allowed=frozenset({tags.COMMERCIAL}), capabilities=frozenset())
        manager = SimpleNamespace(allowed=frozenset({tags.COMMERCIAL}), capabilities=frozenset({"templates.create"}))
        with mock.patch("lab2shot.site.library.presets", lambda: cards):
            self.assertEqual([c["id"] for c in access.templates_for(user)], ["ok"])
            given = SimpleNamespace(allowed=given_gen, capabilities=frozenset())
            self.assertEqual([c["id"] for c in access.templates_for(given)], ["ok", "generative"])
            self.assertEqual([c["id"] for c in access.templates_for(manager)], [c["id"] for c in cards])
