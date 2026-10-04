"""队列优先 (Queue priority, farm/queue.py in_order): an account's waiting tasks go before every other account's waiting
task, among themselves as they came in, never ahead of what runs or of the administrator's 插队; the setting is the
administrator's alone (accounts.listing, server/users.py) and nothing an account reads of itself or of its jobs carries
it. Beside it, two things the queue tells: a running task's node that waits for a card nobody authorized says so
(N-QUEUE-NOCARD, Farm._announce_positions), and a node that steps down on a smaller card asks for its full tier where a
card holds it (placement.holds)."""

from __future__ import annotations

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from lab2shot.site import catalog


def setUpModule() -> None:
    catalog.install()


def job(name, order, state="queued", first=False, jumped=False):
    return SimpleNamespace(name=name, order=order, state=state, queue_first=first, jumped=jumped)


def names(jobs) -> list[str]:
    return [j.name for j in jobs]


class Order(unittest.TestCase):
    def test_without_priority_it_is_first_come_first_served(self):
        from lab2shot.farm.queue import in_order

        self.assertEqual(names(in_order([job("b", 2), job("a", 1), job("c", 3)])), ["a", "b", "c"])

    def test_priority_tasks_go_before_every_waiting_one_in_their_own_order(self):
        from lab2shot.farm.queue import in_order

        line = [job("a", 1), job("b", 2), job("p1", 3, first=True), job("c", 4), job("p2", 5, first=True)]
        self.assertEqual(names(in_order(line)), ["p1", "p2", "a", "b", "c"])

    def test_nothing_running_is_overtaken(self):
        from lab2shot.farm.queue import in_order

        # r1 and r2 run; a waits for a card behind r1; the priority task goes before a, never before r1
        line = [job("r1", 1, "running"), job("a", 2), job("r2", 3, "running"), job("p", 4, first=True)]
        self.assertEqual(names(in_order(line)), ["r1", "p", "a", "r2"])
        # a priority task that already runs keeps its place like any other running task
        line = [job("r", 1, "running"), job("p", 2, "running", first=True), job("a", 3)]
        self.assertEqual(names(in_order(line)), ["r", "p", "a"])

    def test_the_administrators_jump_stays_first(self):
        from lab2shot.farm.queue import in_order

        line = [job("j", -1, jumped=True), job("a", 1), job("p", 2, first=True)]
        self.assertEqual(names(in_order(line)), ["j", "p", "a"])


class Graph:
    nodes: dict = {}
    frames = (1, 2)


def real_job(user: int, first: bool = False):
    from lab2shot.farm.clients import Client
    from lab2shot.farm.queue import Job

    return Job("t", Graph(), [], Client(user, f"u{user}"), queue_first=first)


class NotShown(unittest.TestCase):
    """Only the administrator's 用户 page has it."""

    def test_the_account_and_its_jobs_never_carry_it(self):
        from lab2shot import accounts, roles

        u = accounts.create("qprio_case", accounts.new_password("A-long-pass-9x"), "Qprio",
                            accounts.departments()[0], time.time() + 86400, [], role=roles.DEFAULT)
        self.assertFalse(accounts.queue_first(u.id))  # off by default (the migration's)
        accounts.update(u.id, queue_first=True)
        self.assertTrue(accounts.queue_first(u.id))
        row = next(r for r in accounts.listing() if r["id"] == u.id)
        self.assertIs(row["queue_first"], True)  # the 用户 page's listing has it
        own = accounts.get(u.id)
        for said in (own.public(), own.full()):
            self.assertNotIn("queue_first", json.dumps(said, default=str))
        j = real_job(u.id, first=True)
        with mock.patch.object(type(j.client), "who", property(lambda self: "x")), \
                mock.patch.object(type(j.client), "full", lambda self: {"who": "x"}):
            for said in (j.view(u.id, False), j.view(u.id + 1, False), j.view(None, True), j.record(),
                         j.waiting_json(), j.progress_json(time.time())):
                text = json.dumps(said, default=str)
                self.assertNotIn("queue_first", text)
                self.assertNotIn("队列优先", text)
                self.assertNotIn("Queue priority", text)

    def test_a_user_cannot_set_it(self):
        from lab2shot.server.available import ACCOUNT

        self.assertIn("queue.manage", repr(ACCOUNT["account.queue_first"]))


class RunningNoCard(unittest.TestCase):
    """A task already running whose next node waits for a card nobody authorized is told so (no endless silence)."""

    def test_running_task_is_told(self):
        from lab2shot.engine.resources import GPU, Need
        from lab2shot.farm.queue import Farm
        from lab2shot.farm.scheduler.pools import Ticket
        from lab2shot.messages import Msg

        j = real_job(7)
        j.state = "running"
        t = Ticket(j.id, Need(GPU, "ext", 8.0, 1.0, "Pi3", "pi3"), threading.Event(), lambda: None)
        t.reason = Msg("N-GPU-NOCARD")
        fake = SimpleNamespace(pools=SimpleNamespace(now=lambda: [t]), jobs={j.id: j}, _waiting=lambda: [])
        Farm._announce_positions(fake)
        said = [e for e in j.events if e.get("type") == "message"]
        self.assertEqual([(e["code"], e["node"]) for e in said], [("N-QUEUE-NOCARD", "pi3")])
        self.assertEqual(j.waiting_json()["waiting"]["code"], "N-QUEUE-NOCARD")
        self.assertEqual(j.waiting_json()["waiting_detail"]["code"], "N-GPU-NOCARD")  # the administrator's detail
        Farm._announce_positions(fake)  # said once, not on every pass
        self.assertEqual(len([e for e in j.events if e.get("type") == "message"]), 1)
        t.reason = Msg("N-GPU-WAITBUSY", gpus=["RTX"])  # a wait that clears by itself: nothing said, nothing shown
        Farm._announce_positions(fake)
        self.assertIsNone(j.waiting_json()["waiting"])


class FullTier(unittest.TestCase):
    def test_a_card_holds_the_full_tier_or_not(self):
        from lab2shot.farm import policy
        from lab2shot.farm.scheduler import compat, placement
        from lab2shot.farm.scheduler.inventory import GpuState, Snapshot

        def card(uuid, gb, authorized=True):
            return GpuState(uuid, 0, "RTX", "RTX", "8.9", int(gb * 1024), 0, 0, 40, authorized)

        with mock.patch.object(policy, "margin_gb", lambda: 1.0), \
                mock.patch.object(compat, "fit", lambda runtime, gpu: compat.Fit(True)):
            both = Snapshot((card("a", 24), card("b", 32)), time.time())
            self.assertTrue(placement.holds("ext", 28.0, both))
            small = Snapshot((card("a", 24), card("b", 32, authorized=False)), time.time())
            self.assertFalse(placement.holds("ext", 28.0, small))

    def test_the_node_asks_for_its_full_tier_where_a_card_holds_it(self):
        from lab2shot.engine.evaluation import Evaluation
        from lab2shot.engine.graph import Graph
        from lab2shot.engine.templates import _empty_cache
        from lab2shot.nodes import node_types
        from lab2shot.nodes.applies import Cost
        from lab2shot.nodes.services import services
        from lab2shot.serving import Account

        t = node_types()["birefnet.matte"]

        def graph():  # a new one each time: a graph keeps what it resolved
            return Graph.from_json({"schema": "lab2shot.graph/1", "nodes": [{"id": "m", "type": "birefnet.matte", "params": {}}],
                                    "edges": []})
        plan = services().plan
        with _empty_cache(), mock.patch.object(t, "cost", Cost(gpu=True, vram_gb=20.0, vram_full_gb=28.0)):
            for holds, full, need in ((True, True, 28.0), (False, False, 20.0)):
                with mock.patch.object(type(plan), "holds", lambda self, runtime, gb, h=holds: h, create=True):
                    ev = Evaluation(graph(), Account(1))
                    self.assertEqual((ev.full_tier("m"), ev.vram_need("m")), (full, need))
        with _empty_cache():
            self.assertIsNone(Evaluation(graph(), Account(1)).full_tier("m"))  # a node that never steps down


if __name__ == "__main__":
    unittest.main()
