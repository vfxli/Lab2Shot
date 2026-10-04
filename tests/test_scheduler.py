"""farm/scheduler: one dispatcher pass (Pools._decide) over fake requests and a fake inventory, and the pure card
choices of placement.py (no thread runs, no nvidia-smi, no settings file)."""

from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from lab2shot.engine.resources import CPU, GPU, Need
from lab2shot.farm import policy
from lab2shot.farm.scheduler import compat, placement, pools
from lab2shot.farm.scheduler.inventory import GpuState, Snapshot
from lab2shot.farm.scheduler.pools import Pools, Ticket

POLICY = {"task_cpus": 2, "task_gpus": 1, "cpu_nodes": 4, "margin_gb": 1.0, "gpu_order": [], "node_timeout_s": 900.0,
          "release_below_gb": 2.0}


def card(uuid, index=0, memory_gb=24, used_gb=0.0, ours_gb=0.0, authorized=True) -> GpuState:
    return GpuState(uuid, index, "NVIDIA GeForce RTX", "RTX", "8.9", int(memory_gb * 1024), int(used_gb * 1024), 0, 40,
                    authorized, int(ours_gb * 1024))


class FakeHost:
    def __init__(self, *gpus):
        self.snap = Snapshot(tuple(gpus), time.time())

    def snapshot(self):
        return self.snap


def gpu_need(vram=8.0, ram=1.0, runtime="ext"):
    return Need(GPU, runtime, vram, ram, "node")


def cpu_need(ram=1.0):
    return Need(CPU, "core", 0.0, ram, "node")


class Base(unittest.TestCase):
    ram_free = 100.0
    policy_now = POLICY

    def setUp(self):
        values = dict(self.policy_now)
        patches = [mock.patch.object(policy, name, (lambda v: lambda *a: v)(value)) for name, value in values.items()]
        patches += [mock.patch.object(pools, "available_gb", lambda: self.ram_free),
                    mock.patch.object(pools, "machine_memory_gb", lambda: 256.0),
                    mock.patch.object(pools, "keep_free_gb", lambda ram=0.0: max(8.0, ram)),
                    mock.patch.object(pools.Pools, "_paused", staticmethod(lambda kind: None)),
                    mock.patch.object(compat, "fit", lambda runtime, gpu: compat.Fit(True)),
                    mock.patch.object(compat, "wait_reason", lambda *a: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def pools(self, *gpus) -> Pools:
        return Pools(FakeHost(*gpus), lambda: [], lambda: False, lambda task: None, lambda work: None)

    def ask(self, p: Pools, task, need) -> Ticket:
        t = Ticket(task, need, threading.Event(), lambda: None)
        p.tickets.append(t)
        return t

    def decide(self, p: Pools, line, held=False):
        return p._decide(line, held)


class Decide(Base):
    def test_queue_order_first(self):
        p = self.pools(card("G0"))
        late = self.ask(p, "t2", gpu_need())
        early = self.ask(p, "t1", gpu_need())
        out = self.decide(p, [("t1", False), ("t2", False)])
        self.assertEqual(out.granted, [early])
        self.assertTrue(early.granted and not late.granted)
        self.assertEqual(late.reason.code, "N-GPU-WAITBUSY")
        self.assertEqual(out.started, ["t1"])

    def test_one_task_holds_at_most_task_gpus_cards(self):
        p = self.pools(card("G0"), card("G1", 1))
        a, b = self.ask(p, "t1", gpu_need()), self.ask(p, "t1", gpu_need())
        self.decide(p, [("t1", False)])
        self.assertTrue(a.granted)
        self.assertEqual(b.reason.code, "N-QUEUE-TASKGPUS")

    def test_second_task_gets_the_other_card(self):
        p = self.pools(card("G0"), card("G1", 1))
        a, b = self.ask(p, "t1", gpu_need()), self.ask(p, "t2", gpu_need())
        self.decide(p, [("t1", False), ("t2", False)])
        self.assertTrue(a.granted and b.granted)
        self.assertNotEqual(a.gpu, b.gpu)

    def test_task_cpu_limit(self):
        p = self.pools()
        ts = [self.ask(p, "t1", cpu_need()) for _ in range(3)]
        self.decide(p, [("t1", False)])
        self.assertEqual([t.granted for t in ts], [True, True, False])
        self.assertEqual(ts[2].reason.code, "N-QUEUE-TASKCPUS")

    def test_machine_cpu_limit(self):
        p = self.pools()
        ts = [self.ask(p, f"t{i}", cpu_need()) for i in range(5)]
        self.decide(p, [(f"t{i}", False) for i in range(5)])
        self.assertEqual(sum(t.granted for t in ts), 4)
        self.assertEqual(ts[4].reason.code, "N-QUEUE-CPUBUSY")

    def test_memory_guard_holds_the_rest_and_asks_for_room(self):
        self.ram_free = 8.5  # keep_free 8: the first (1 GB) starts, after it 7.5 is left: the second waits
        p = self.pools()
        a, b, c = (self.ask(p, f"t{i}", cpu_need()) for i in range(3))
        out = self.decide(p, [("t0", False), ("t1", False), ("t2", False)])
        self.assertTrue(a.granted)
        self.assertEqual(b.reason.code, "N-QUEUE-WAITRAM")
        self.assertIs(c.reason, b.reason)  # everything after it waits too
        self.assertEqual(out.free_ram, (8.0, {"core"}))

    def test_never_fits_says_so_and_holds_up_nobody(self):
        p = self.pools()
        huge = self.ask(p, "t0", cpu_need(ram=1000))
        small = self.ask(p, "t1", cpu_need())
        self.decide(p, [("t0", False), ("t1", False)])
        self.assertEqual(huge.reason.code, "W-QUEUE-NEVERRAM")
        self.assertTrue(small.granted)

    def test_vram_wait_asks_to_reclaim(self):
        p = self.pools(card("G0", used_gb=20))  # 4 GB free, the node needs 8 + 1
        t = self.ask(p, "t0", gpu_need())
        out = self.decide(p, [("t0", False)])
        self.assertFalse(t.granted)
        self.assertIn(t.reason.code, pools.VRAM_WAITS)
        self.assertEqual(out.reclaim[0], t.need)

    def test_our_idle_models_count_as_free(self):
        p = self.pools(card("G0", used_gb=20, ours_gb=18))
        t = self.ask(p, "t0", gpu_need())
        self.decide(p, [("t0", False)])
        self.assertTrue(t.granted)

    def test_held_queue_starts_nothing_new(self):
        p = self.pools(card("G0"))
        new, going = self.ask(p, "t0", gpu_need()), self.ask(p, "t1", cpu_need())
        self.decide(p, [("t0", False), ("t1", True)], held=True)
        self.assertFalse(new.granted)
        self.assertTrue(going.granted)

    def test_withheld_runtime_waits(self):
        p = self.pools(card("G0"))
        p._withheld.add("ext")
        t = self.ask(p, "t0", gpu_need())
        self.decide(p, [("t0", False)])
        self.assertFalse(t.granted)
        self.assertIsNone(t.reason)

    def test_busy_cards_reported(self):
        p = self.pools(card("G0"), card("G1", 1))
        self.ask(p, "t0", gpu_need())
        out = self.decide(p, [("t0", False)])
        self.assertEqual(len(out.busy), 1)


class Placement(Base):
    def test_reclaimable_only_when_contexts_cover_the_shortfall(self):
        snap = FakeHost(card("G0", used_gb=15.5), card("G1", 1, used_gb=20)).snapshot()
        need = gpu_need(vram=8.0)  # wants 9 GB: G0 is 0.5 GB short, G1 5 GB short
        got = placement.reclaimable_cards(need, snap, set(), {"G0": 800, "G1": 800})
        self.assertEqual(got, {"G0": 512})
        self.assertEqual(placement.reclaimable_cards(need, snap, {"G0"}, {"G0": 800}), {})
        self.assertEqual(placement.reclaimable_cards(need, snap, set(), {"G0": 400}), {})

    def test_pressed_cards(self):
        snap = FakeHost(card("G0", used_gb=23), card("G1", 1, used_gb=10), card("G2", 2, used_gb=23.5),
                        card("G3", 3, used_gb=23.5, authorized=False)).snapshot()
        self.assertEqual(placement.pressed_cards(snap, {"G2"}, 2.0), {"G0": 1024})
        self.assertEqual(placement.pressed_cards(snap, set(), 0.0), {})


if __name__ == "__main__":
    unittest.main()
