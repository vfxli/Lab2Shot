"""engine/resident.py: who gives up a card's memory and how (release_plan, a pure function), and the pool acting on it
with fake processes (no worker, no GPU)."""

from __future__ import annotations

import time
import unittest
from unittest import mock

from lab2shot.engine import resident
from lab2shot.engine.resident import END, TO_RAM, Held, Pool, release_plan


def h(key, last_used, on_gpu=True, vram=1000, context=500) -> Held:
    return Held(key, last_used, on_gpu, vram, context)


class ReleasePlan(unittest.TestCase):
    CASES = [
        # (name, held, kwargs, expected plan)
        ("a job takes the card: every one on the GPU moves to RAM, least recently used first",
         [h("b", 2), h("a", 1), h("c", 3, on_gpu=False)], {}, [("a", TO_RAM), ("b", TO_RAM)]),
        ("beyond keep_most the least recently used end, the rest move",
         [h("a", 1), h("b", 2), h("c", 3)], {"keep_most": 1}, [("a", END), ("b", END), ("c", TO_RAM)]),
        ("keep_most 0 ends all", [h("a", 1), h("b", 2)], {"keep_most": 0}, [("a", END), ("b", END)]),
        ("to_ram off: they end", [h("a", 1)], {"to_ram": False}, [("a", END)]),
        ("RAM short after the first move: the next ends",
         [h("a", 1, vram=4096), h("b", 2, vram=4096)], {"ram_free_gb": 13.0, "keep_free_gb": 8.0},
         [("a", TO_RAM), ("b", END)]),
        ("need met by the first move: the rest stay", [h("a", 1), h("b", 2)], {"need_mb": 900}, [("a", TO_RAM)]),
        ("need beyond every move: they end, least recently used first, until it is met",
         [h("a", 1), h("b", 2), h("c", 3)], {"need_mb": 3600},  # moves free 3000, each end adds its 500
         [("a", END), ("b", END), ("c", TO_RAM)]),
        ("contexts only (vram counted already, no RAM): as few end as cover it",
         [h("a", 1, vram=0, context=800), h("b", 2, vram=0, context=800), h("c", 3, vram=0, context=800)],
         {"need_mb": 1000, "to_ram": False}, [("a", END), ("b", END)]),
        ("models in RAM only: pressure ends them for their contexts",
         [h("a", 1, on_gpu=False, vram=0), h("b", 2, on_gpu=False, vram=0)], {"need_mb": 400}, [("a", END)]),
        ("nothing needed: nothing moves", [h("a", 1)], {"need_mb": 0}, []),
        ("nobody idle: nothing", [], {"need_mb": 5000}, []),
    ]

    def test_cases(self):
        for name, held, kwargs, want in self.CASES:
            with self.subTest(name):
                got = release_plan(held, **kwargs)
                self.assertEqual(sorted(got, key=lambda kv: kv[0]), sorted(want, key=lambda kv: kv[0]))

    def test_order_is_least_recently_used_first(self):
        got = release_plan([h("c", 3), h("a", 1), h("b", 2)])
        self.assertEqual([k for k, _ in got], ["a", "b", "c"])


class FakeExt:
    def __init__(self, name):
        self.name = self.title = name


class FakeProcess:
    """What the pool reads and calls of a Process; ending and offloading are recorded, nothing runs."""

    def __init__(self, name, gpu, last_used, on_gpu=True, vram_mb=1000, context_mb=None, in_ram_since=None, ram_mb=1000):
        self.ext, self.gpu, self.last_used, self._ram_mb = FakeExt(name), gpu, last_used, ram_mb
        self.models = [{"on_gpu": on_gpu}]
        self.vram_mb, self.context_mb, self.in_ram_since = vram_mb, context_mb, in_ram_since
        self.state, self.ended, self.offloaded, self.alive = "idle", False, False, True

    on_gpu = property(lambda self: any(m["on_gpu"] for m in self.models))
    held = resident.Process.held
    ram_mb = lambda self: self._ram_mb  # noqa: E731

    def end(self, grace=0):
        self.ended, self.state = True, "ending"


def fake_offload(pool: Pool, p, keep_free):
    p.offloaded, p.models, p.vram_mb, p.state = True, [{"on_gpu": False}], 0, "idle"


SETTINGS = {"resident.keep": True, "resident.per_gpu": 3, "resident.to_ram": True, "resident.idle_minutes": 30,
            "resident.ram_idle_minutes": 5, "resident.ram_pct": 30, "memory.keep_free_gb": 8.0}


class PoolActs(unittest.TestCase):
    def setUp(self):
        self.pool = Pool()
        patches = [mock.patch.object(resident, "settings", lambda: SETTINGS),
                   mock.patch.object(resident, "available_gb", lambda: 100.0),
                   mock.patch.object(resident, "machine_memory_gb", lambda: 64.0),
                   mock.patch.object(Pool, "_offload", fake_offload),
                   mock.patch.object(resident.logs, "say", lambda *a, **k: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_idle_context_mb_uses_measured_else_guess(self):
        self.pool.procs = [FakeProcess("a", "G0", 1, context_mb=450), FakeProcess("b", "G0", 2),
                           FakeProcess("c", "G1", 3, context_mb=600)]
        self.assertEqual(self.pool.idle_context_mb(), {"G0": 450 + resident.CONTEXT_MB_GUESS, "G1": 600})

    def test_clear_idle_ends_only_what_covers_the_shortfall(self):
        a, b, c = (FakeProcess(n, "G0", i, context_mb=500) for i, n in enumerate("abc"))
        self.pool.procs = [a, b, c]
        self.assertEqual(self.pool.clear_idle({"G0": 900}), 2)
        self.assertEqual((a.ended, b.ended, c.ended), (True, True, False))
        self.assertEqual(self.pool.procs, [c])

    def test_relieve_moves_to_ram_first(self):
        a, b = FakeProcess("a", "G0", 1, vram_mb=3000), FakeProcess("b", "G0", 2, vram_mb=3000)
        other = FakeProcess("x", "G1", 0)
        self.pool.procs = [a, b, other]
        self.assertEqual(self.pool.relieve({"G0": 2000}), 1)
        self.assertTrue(a.offloaded and not a.ended)
        self.assertFalse(b.offloaded or b.ended or other.offloaded)

    def test_relieve_ends_when_moving_is_not_enough(self):
        a = FakeProcess("a", "G0", 1, on_gpu=False, vram_mb=0, context_mb=500)
        self.pool.procs = [a]
        self.assertEqual(self.pool.relieve({"G0": 300}), 1)
        self.assertTrue(a.ended)

    def test_relieve_to_ram_off_ends(self):
        a = FakeProcess("a", "G0", 1)
        self.pool.procs = [a]
        with mock.patch.object(resident, "settings", lambda: {**SETTINGS, "resident.to_ram": False}):
            self.pool.relieve({"G0": 100})
        self.assertTrue(a.ended and not a.offloaded)

    def test_tidy_ends_models_in_ram_sooner(self):
        now = time.time()
        in_ram = FakeProcess("a", "G0", now - 400, on_gpu=False, in_ram_since=now - 400)  # 6.7 min > 5
        on_gpu = FakeProcess("b", "G0", now - 400)  # < 30 min
        fresh_in_ram = FakeProcess("c", "G0", now - 60, on_gpu=False, in_ram_since=now - 60)
        self.pool.procs = [in_ram, on_gpu, fresh_in_ram]
        self.pool.tidy()
        self.assertEqual((in_ram.ended, on_gpu.ended, fresh_in_ram.ended), (True, False, False))

    def test_idle_processes_keep_to_their_share_of_ram(self):
        # 30 % of 64 GB is 19.2 GB: three idle processes of 8 GB each are 24 GB, so the least recently used ends
        a, b, c = (FakeProcess(n, "G0", t, ram_mb=8 * 1024) for t, n in ((time.time() - 30, "a"), (time.time() - 20, "b"),
                                                                          (time.time() - 10, "c")))
        busy = FakeProcess("busy", "G1", time.time() - 99, ram_mb=40 * 1024)
        busy.state = "busy"
        self.pool.procs = [c, busy, a, b]
        self.pool.tidy()
        self.assertEqual((a.ended, b.ended, c.ended, busy.ended), (True, False, False, False))

    def test_idle_processes_end_when_the_machine_runs_short(self):
        # under their share, but the machine has 5 GB available of the 8 kept free: the oldest ends, freeing 4 GB
        a, b = FakeProcess("a", "G0", time.time() - 20, ram_mb=4096), FakeProcess("b", "G0", time.time() - 10, ram_mb=4096)
        self.pool.procs = [a, b]
        with mock.patch.object(resident, "available_gb", lambda: 5.0):
            self.pool.tidy()
        self.assertEqual((a.ended, b.ended), (True, False))

    def test_tidy_idle_limit(self):
        old = FakeProcess("a", "G0", time.time() - 31 * 60)
        self.pool.procs = [old]
        self.pool.tidy()
        self.assertTrue(old.ended)

    def test_make_room_keeps_per_gpu_and_moves_the_rest(self):
        job = FakeProcess("job", "G0", 10)
        job.state = "busy"
        others = [FakeProcess(n, "G0", i) for i, n in enumerate("abcd")]
        far = FakeProcess("far", "G1", 0)
        self.pool.procs = [job, *others, far]
        stages = []
        self.pool._make_room(job, 8.0, lambda stage, **kw: stages.append(stage))
        self.assertEqual([p.ended for p in others], [True, False, False, False])
        self.assertEqual([p.offloaded for p in others], [False, True, True, True])
        self.assertEqual(stages, ["resident_unload", "resident_to_ram", "resident_to_ram", "resident_to_ram"])
        self.assertFalse(far.offloaded or far.ended)

    def test_report_tracks_context_and_ram_time(self):
        p = FakeProcess("a", "G0", 1)
        resident.Process.report(p, {"models": [{"on_gpu": False}], "vram_mb": 0, "context_mb": 512})
        self.assertEqual(p.context_mb, 512)
        self.assertIsNotNone(p.in_ram_since)
        resident.Process.report(p, {"models": [{"on_gpu": True}], "vram_mb": 900, "context_mb": None})
        self.assertEqual(p.context_mb, 512)  # measured once, kept
        self.assertIsNone(p.in_ram_since)


if __name__ == "__main__":
    unittest.main()
