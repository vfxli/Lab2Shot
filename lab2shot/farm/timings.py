"""How long nodes take: every node that really computes in the queue is timed; a submitted job's nodes are estimated
from those records for its progress bar's denominator only (farm/queue.py Job.estimate / fix_budget, lab2shot/progress.py).
No estimated time is shown anywhere (no time left, no time to start, no 预计用时 before submitting): a time predicted
from earlier cooks is not reliable. The look before submitting (`look`) says what a cook takes, not how long.

A record (the database's timings table) keeps the node type, the GPU model it ran on ("CPU" for a node
that uses none), its setting (a hash of the parameters that change the work, per-shot files and picks left out), the
frames and picture size it processed and the seconds it spent computing (waiting in the queue, for memory or for
another cook is left out). A cached result, or a worker's raw results taken from an earlier run, says nothing about
the work and is not recorded.

An estimate takes the newest records of the same node type on the GPUs that take jobs (else on any GPU; "CPU" for a
node that uses none), with the same setting when there are some. It scales them by frames, and by pixels when the
records show the time depends on them: seconds = fixed part (loading a model) + a part per frame, fitted when the
records have different lengths, else in proportion to one of them.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict, dataclass

from ..io.digest import key
from ..database import Database, db

KEEP = 20  # newest matching records an estimate uses
KEPT_PER_NODE = 200  # records kept per node type, the newest (record): enough for KEEP on each card and setting in use
SHOT_WIDGETS = {"file", "sequence", "picks"}  # parameters that differ per shot, not per setting
CPU = "CPU"


def setting(node_type, params: dict) -> str:
    """The parameters that change the node's work, as one short hash (files and picks differ per shot: left out)."""
    keep = sorted(p["name"] for p in node_type.param_specs() if p["affects_result"] and p["widget"] not in SHOT_WIDGETS)
    return key({k: params.get(k) for k in keep}, 12)  # 12 characters, like every record's setting


def device(gpu: bool, gpu_name: str) -> str:
    """What a node computes on: the GPU model, or "CPU" for a node that uses none (`gpu`: its resolved cost says it
    runs on one)."""
    return gpu_name if gpu and gpu_name else CPU


FIELDS = ("t", "node", "gpu", "setting", "frames", "width", "height", "seconds")


def record(node_type, params: dict, gpu: bool, gpu_name: str, done: dict, database: Database | None = None) -> None:
    """Keep the time of a node that computed (`done`: its node_done event: seconds, frames, width, height), in
    `database` (its job's; default: the current work folder's). `gpu`: it ran on one (its resolved cost)."""
    entry = {"t": round(time.time()), "node": node_type.id, "gpu": device(gpu, gpu_name),
             "setting": setting(node_type, params), **{k: done[k] for k in ("frames", "width", "height", "seconds")}}
    with (database or db()).write() as c:
        c.execute(f"INSERT INTO timings ({', '.join(FIELDS)}) VALUES ({', '.join('?' * len(FIELDS))})", [entry[k] for k in FIELDS])
        c.execute("DELETE FROM timings WHERE node = ? AND id <= (SELECT id FROM timings WHERE node = ? ORDER BY id DESC "
                  "LIMIT 1 OFFSET ?)", (entry["node"], entry["node"], KEPT_PER_NODE))  # the table never grows past it


def records(node_types: set[str], database: Database | None = None) -> dict[str, list[dict]]:
    """The records of these node types, oldest first, per type (of `database`; default: the current work folder's):
    what estimating a job's nodes needs, never the whole table (index timings_node)."""
    if not node_types:
        return {}
    found: dict[str, list[dict]] = {}
    names = sorted(node_types)
    for r in (database or db()).rows(f"SELECT {', '.join(FIELDS)} FROM timings WHERE node IN ({', '.join('?' * len(names))}) "
                                     "ORDER BY id", names):
        found.setdefault(r["node"], []).append(dict(r))
    return found


@dataclass(frozen=True)
class Work:
    """A node a cook will compute, as its estimate needs it."""

    node: str  # id in the graph
    label: str
    type: str
    gpu: bool  # computes on a GPU
    setting: str
    frames: int
    width: int
    height: int


def predict(work: Work, gpus: list[str], same: list[dict]) -> dict:
    """{"seconds": estimate or None without records, "records": how many it rests on, "device": the GPU model (or
    "CPU") they are from}. `gpus`: the GPU models the cook may run on (the ones that take jobs, or the one it got);
    `same`: the records of its node type, oldest first (`records`)."""
    if work.gpu:
        pool = [r for r in same if r["gpu"] in gpus] or [r for r in same if r["gpu"] != CPU]
    else:
        pool = [r for r in same if r["gpu"] == CPU]
    pool = ([r for r in pool if r["setting"] == work.setting] or pool)[-KEEP:]
    if not pool:
        return {"seconds": None, "records": 0, "device": ""}
    seconds = fit([(r["frames"], r["width"] * r["height"], r["seconds"]) for r in pool], work.frames, work.width * work.height)
    return {"seconds": round(seconds, 1), "records": len(pool), "device": Counter(r["gpu"] for r in pool).most_common(1)[0][0]}


def fit(rows: list[tuple[int, int, float]], frames: int, pixels: int) -> float:
    """Seconds for `frames` at `pixels` from (frames, pixels, seconds) records. A record's work counts as its frames x
    (its pixels / target pixels) ** a frames of the target size: a = 0 (the model works at its own size, the usual
    case) unless records of several sizes fit better with a = 1."""
    target = max(pixels, 1)
    best = None
    for a in (0.0, 1.0) if len({p for _, p, _ in rows}) > 1 else (0.0,):
        work = [max(f, 1) * (max(p, 1) / target) ** a for f, p, _ in rows]
        seconds = [s for _, _, s in rows]
        fixed, per = _line(work, seconds)
        err = sum(((fixed + per * w) - s) ** 2 / max(s, 0.1) ** 2 for w, s in zip(work, seconds))
        if best is None or err < best[0] - 1e-9:
            best = (err, fixed, per)
    _, fixed, per = best
    return fixed + per * max(frames, 1)


def _line(work: list[float], seconds: list[float]) -> tuple[float, float]:
    """seconds = fixed + per * work by least squares, neither below zero; records of one length: in proportion."""
    n = len(work)
    mw, ms = sum(work) / n, sum(seconds) / n
    var = sum((w - mw) ** 2 for w in work)
    if var > 1e-9 * mw * mw:
        per = sum((w - mw) * (s - ms) for w, s in zip(work, seconds)) / var
        fixed = ms - per * mw
        if per < 0:  # longer runs were not slower: the time is all fixed
            return ms, 0.0
        if fixed >= 0:
            return fixed, per
    return 0.0, sum(w * s for w, s in zip(work, seconds)) / sum(w * w for w in work)


def planned(ev, targets: list[str], force: bool = False, ready=None) -> list[Work]:
    """What cooking `targets` computes (Evaluation.readiness: its `computing`), each with what its estimate needs: one
    Work per instance; a node in a block whose items are not known yet (a prefix behind what is pending) once, from the
    node's own parameters, its cost the graph's (no instance exists yet to plan, so no frames or size)."""
    from ..engine.evaluation import PLAN_ERRORS

    out, seen = [], set()

    def of_node(nid: str) -> None:  # the node from its own parameters, once: no instance to plan (yet)
        if nid not in seen:
            seen.add(nid)
            node = ev.graph.nodes[nid]
            out.append(Work(nid, node.label, node.type.id, ev.graph.resolved(nid).cost.gpu, setting(node.type, node.params), 0, 0, 0))

    for inst in (ready or ev.readiness(targets, force)).computing:
        nid, path = inst
        node = ev.graph.nodes[nid]
        if len(path) != ev.graph.scopes.depth(nid):  # not an instance yet (a block's items not known)
            of_node(nid)
            continue
        try:  # the instance's cost, as the engine places it (Evaluation.resolved: without the wires it goes without)
            params, size, gpu = ev.params(nid, path), ev.work(nid, path), ev.resolved(nid, path).cost.gpu
        except PLAN_ERRORS:  # it can't be planned (its file is not there): it still runs, and fails at itself
            of_node(nid)
            continue
        out.append(Work(nid, node.label, node.type.id, gpu, setting(node.type, params), len(size.frames), size.width, size.height))
    return out


def estimate(ev, targets: list[str], force: bool, ready=None) -> dict:
    """Before a cook: the frame range its inputs cover and the one it takes, the instances it computes (in cook order)
    and how many it needs are cached, both counted from the one answer (Evaluation.readiness). No time: see the
    module's docstring."""
    ready = ready or ev.readiness(targets, force)
    full = ev.frame_range(targets)
    return {
        "range": list(full) if full else None,
        "frames": list(ev.graph.frames or full) if full else None,
        "nodes": [asdict(w) for w in planned(ev, targets, force, ready)],
        "cached": len(ready.cached),
    }


def look(ev, target: str, force: bool, show=None) -> dict:
    """A look at cooking `target` before submitting it: the estimate of what submitting it would queue, from the same
    Readiness submitting reads (farm/queue.py _submit), so what the look accepts is what is accepted; `show` the
    outputs of it the viewer shows, as submitting it would send them. It never
    fails the page: a cook refused as a whole (Evaluation.readiness: a B- error of the graph) says why ("error": its
    message, {code, level, text, params}), with the frame range its inputs cover when that is known; a fault of the
    program is said as E-FARM-INTERNAL the same way (Evaluation.guarded), never a 500. The shown node's look comes with
    every status reply (server/packets.py); /api/plan gives it to scripts and DCC clients."""
    from ..engine.evaluation import PLAN_ERRORS
    from ..errors import message_of

    def refused(said: dict) -> dict:
        full = ev.guarded("look range", lambda: ev.frame_range([target]), None) if target in ev.graph.nodes else None
        return {"range": list(full) if full else None, "frames": None, "nodes": [], "cached": 0, "node": target, "error": said}

    def answer() -> dict:
        try:
            targets = [target]
            ready = ev.readiness(targets, force, frozenset(show or ()))
            if ready.refused is not None:
                return refused(ready.refused.json())
            # the instances it will fail at (their own error; the rest is cooked): said beside the plan, never instead
            failing = [{"node": i.node, "path": list(i.path), "error": o.message}
                       for i in sorted(ready.failing, key=repr) if (o := ev.outcome(*i)) is not None]
            return {**estimate(ev, targets, force, ready), "node": target, "error": None, "failing": failing}
        except PLAN_ERRORS as exc:  # something on the way can't be planned at all: said, as the status says it
            return refused(message_of(exc).json())

    return ev.guarded("look", answer, lambda said: refused(said.json()))
