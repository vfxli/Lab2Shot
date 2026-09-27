"""How long nodes take: every node that really computes in the queue is timed, and cooks are estimated from those
records before they are submitted and while they wait or run.

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
SHOT_WIDGETS = {"file", "sequence", "deliver", "picks"}  # parameters that differ per shot, not per setting
CPU = "CPU"


def setting(node_type, params: dict) -> str:
    """The parameters that change the node's work, as one short hash (files and picks differ per shot: left out)."""
    keep = sorted(p["name"] for p in node_type.param_specs() if p["affects_result"] and p["widget"] not in SHOT_WIDGETS)
    return key({k: params.get(k) for k in keep}, 12)  # the same 12 characters the records have always been made of


def device(gpu: bool, gpu_name: str) -> str:
    """What a node computes on: the GPU model, or "CPU" for a node that uses none (`gpu`: its resolved cost says it
    runs on one)."""
    return gpu_name if gpu and gpu_name else CPU


FIELDS = ("t", "node", "gpu", "setting", "frames", "width", "height", "seconds")
_read: dict = {}  # database file -> (its newest record id, the records): read again only when there are new ones


def record(node_type, params: dict, gpu: bool, gpu_name: str, done: dict, database: Database | None = None) -> None:
    """Keep the time of a node that computed (`done`: its node_done event: seconds, frames, width, height), in
    `database` (its job's; default: the current work folder's). `gpu`: it ran on one (its resolved cost)."""
    entry = {"t": round(time.time()), "node": node_type.id, "gpu": device(gpu, gpu_name),
             "setting": setting(node_type, params), **{k: done[k] for k in ("frames", "width", "height", "seconds")}}
    with (database or db()).write() as c:
        c.execute(f"INSERT INTO timings ({', '.join(FIELDS)}) VALUES ({', '.join('?' * len(FIELDS))})", [entry[k] for k in FIELDS])


def records(database: Database | None = None) -> list[dict]:
    """Every record, oldest first (of `database`; default: the current work folder's)."""
    d = database or db()
    newest = d.row("SELECT MAX(id) AS id FROM timings")["id"]
    seen, kept = _read.get(d.path, (None, []))
    if seen != newest:
        kept = [dict(r) for r in d.rows(f"SELECT {', '.join(FIELDS)} FROM timings ORDER BY id")]
        _read[d.path] = (newest, kept)
    return kept


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


def predict(work: Work, gpus: list[str], history: list[dict]) -> dict:
    """{"seconds": estimate or None without records, "records": how many it rests on, "device": the GPU model (or
    "CPU") they are from}. `gpus`: the GPU models the cook may run on (the ones that take jobs, or the one it got)."""
    same = [r for r in history if r["node"] == work.type]
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


def planned(engine, targets: list[str], force: bool = False) -> list[Work]:
    """The nodes cooking `targets` will compute (Engine.computes), with what their estimates need."""
    from ..engine.evaluation import PLAN_ERRORS

    out = []
    for nid in engine.computes(targets, force):
        node = engine.graph.nodes[nid]
        for path in _instances(engine, nid):  # a node inside a 逐项处理 block computes once per item: each is its own work
            try:
                params, size = engine.params(nid, path), engine.work(nid, path)
            except PLAN_ERRORS:  # a node that can't be planned (its file is not there) still runs, and fails at itself:
                continue  # there is nothing to estimate for it, and the cook's estimate is the other nodes'
            out.append(Work(nid, node.label, node.type.id, engine.graph.resolved(nid).cost.gpu, setting(node.type, params),
                            len(size.frames), size.width, size.height))
    return out


def _instances(engine, node_id: str) -> list:
    """The item paths of a node in this graph (engine/scopes.py): the one empty path outside every block. An item list
    that is not known yet leaves the node with none: it is estimated once its items are."""
    from ..engine.evaluation import PLAN_ERRORS

    evaluation = getattr(engine, "eval", engine)
    try:
        paths = evaluation.instances(node_id)[0]
    except PLAN_ERRORS:
        return []
    # a block whose item list is not known yet (the node above it has not cooked): the node is still counted once, so
    # what the job needs (a card, its VRAM: farm/scheduler/requirements.py) is known before anything of it has run
    return paths or [()]


def estimate(engine, targets: list[str], force: bool, gpus: list[str]) -> dict:
    """Before a cook: the frame range its inputs cover and the one it takes, the nodes it computes with their
    estimates (in cook order), how many are cached, and the sum of what can be estimated."""
    full = engine.frame_range(targets)
    works = planned(engine, targets, force)
    history = records()
    nodes = [{**asdict(w), **predict(w, gpus, history)} for w in works]
    upstream = engine.graph.needed(targets)
    return {
        "range": list(full) if full else None,
        "frames": list(engine.graph.frames or full) if full else None,
        "nodes": nodes,
        "cached": len(upstream) - len(works),
        "seconds": round(sum(n["seconds"] for n in nodes if n["seconds"] is not None), 1),
        "unknown": sum(n["seconds"] is None for n in nodes),
    }


def unplannable(engine, targets: list[str]):
    """The error of the first target that cannot be planned at all (its file is not there, a precondition it can't get
    past), None when at least one of them can. The queue refuses a cook only when NOTHING of it can be planned — one
    broken branch among others is queued, fails at its own node and lets the rest finish — and the look
    before submitting shows that same error on the page, instead of an estimate of nothing."""
    from ..engine.evaluation import PLAN_ERRORS

    first = None
    for target in targets:
        try:
            for path in _instances(engine, target) or [()]:
                engine.plan(target, path)
            return None
        except PLAN_ERRORS as exc:
            first = first or exc
    return first


def look(engine, target: str, force: bool, gpus: list[str]) -> dict:
    """A look at cooking `target` before submitting it (estimate, for what it computes: Graph.computed_by). It never
    fails the page: a graph that can't be planned yet says why ("error": its message, {code, level, text, params}), with the frame range its inputs cover when
    that is known. The shown node's look comes with every status reply (server/packets.py); /api/plan gives it to
    scripts and DCC clients."""
    from ..errors import CookError, message_of

    try:
        targets = engine.graph.computed_by(target)
        if (problem := unplannable(engine, targets)) is not None:
            raise problem  # said as it is, at its node: never an estimate of a cook that cannot even be planned
        return {**estimate(engine, targets, force, gpus), "node": target, "error": None}
    except (ValueError, CookError, OSError) as exc:  # GraphError is a ValueError
        full = engine.frame_range([target]) if target in engine.graph.nodes else None
        return {"range": list(full) if full else None, "frames": None, "nodes": [], "cached": 0, "seconds": 0,
                "unknown": 0, "node": target, "error": message_of(exc).json()}
