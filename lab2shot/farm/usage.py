"""Usage statistics: how much each third-party project, and each of its nodes, is used over a time range, and by which
department and which account — for the administrator to see which projects earn their place and who uses what (the
admin page's 使用统计: 按项目, 按环节, 按人).

Nothing is logged apart for them. When a job ends, the database keeps per node type how its nodes served it
(job_usage): runs (they computed: the seconds, the part of them on a GPU, the frames) and reuses (answered without
computing: a cached result, or a worker's raw results from an earlier run — a reuse is still a use). Who asked is the
job's account and the account's department (lab2shot/accounts.py; a deleted account, or one whose row is gone, counts as
「已删除的用户」), when is the job's end. A job the server was stopped in the middle of (a
restart) has no end and is not counted.

Every installed project is listed with every one of its node types, used or not: the ones never used in the range show
zero. The core's nodes are listed apart ("core").

Every finished job counts, the small ones too (reading a file, a value, delivering what is cached).
An account without a department counts under 未分环节.

A reset starts the statistics again from now; the starts it replaced are kept (undo_reset puts the last one back).
Nothing is deleted.
"""

from __future__ import annotations

import time

from .. import i18n
from ..database import db
from ..errors import Invalid
from ..messages import Msg
from ..periods import Periods, local_day

DAY_S = 86400
MAX_DAYS = 3660  # a per-day series is at most ten years long
COUNTS = ("runs", "reuses", "seconds", "gpu_seconds", "frames")


def start() -> float | None:
    """Since when the statistics count (the last reset); None: since the first job."""
    return db().meta("usage.since")


def reset() -> float:
    """Start the statistics again from now; the start it replaces is kept for undo_reset."""
    d, now = db(), round(time.time(), 3)
    with d.write():
        d.set_meta("usage.history", [*d.meta("usage.history", []), start()])
        d.set_meta("usage.since", now)
    return now


def undo_reset() -> float | None:
    """Put back the start the last reset replaced."""
    d = db()
    history = d.meta("usage.history", [])
    if not history:
        raise Invalid(Msg("E-USAGE-NOUNDO"))
    with d.write():
        d.set_meta("usage.since", history[-1])
        d.set_meta("usage.history", history[:-1])
    return history[-1]


ENDS = ("done", "partial", "failed", "cancelled")  # how a job ended (farm/queue.py Job.state)


def jobs_in(p: Periods) -> dict:
    """The admin overview's 任务: the jobs submitted in 今日, 近 7 天 and 本月 (every account's, every kind: a render and
    a file read alike), all of them and by how each ended; the ones still queued or running count in `all` only. A job
    deleted since (by its account or an administrator) no longer counts; one whose task folder expired still does."""
    rows = db().rows(f"SELECT {local_day('submitted')} AS day, state, COUNT(*) AS n FROM jobs WHERE submitted >= ? "
                     "GROUP BY 1, 2", (p.since,))
    shown = ("today", "days7", "month")
    total = p.sums(((r["day"], r["n"]) for r in rows), shown)
    ended = {s: p.sums(((r["day"], r["n"]) for r in rows if r["state"] == s), shown) for s in ENDS}
    return {k: {"all": total[k], **{s: ended[s][k] for s in ENDS}} for k in shown}


def _counts() -> dict:
    return {**dict.fromkeys(COUNTS, 0), "users": set(), "last": None}


def _add(row: dict, use: dict, who: str) -> None:
    for k in COUNTS:
        row[k] += use[k]
    if who:
        row["users"].add(who)


def _done(row: dict) -> dict:
    return {**row, "seconds": round(row["seconds"], 1), "gpu_seconds": round(row["gpu_seconds"], 1),
            "users": sorted(row["users"])}


def _uses() -> list[tuple[float, str, str, str, dict]]:
    """(when the job ended, account, department, node type, its usage) of every finished job; an account as
    「张三（zhangsan）」, a deleted one as 「已删除的用户」."""
    from ..accounts import deleted_label, label_of

    gone = deleted_label()

    rows = db().rows(f"""
        SELECT j.finished, a.name, a.username, a.deleted, COALESCE(a.department, '') AS department, u.node_type,
               {', '.join('u.' + k for k in COUNTS)}
        FROM job_usage u JOIN jobs j ON j.id = u.job_id LEFT JOIN users a ON a.id = j.user_id
        WHERE j.finished IS NOT NULL ORDER BY j.finished""")
    return [(r[0], gone if r["deleted"] or r["username"] is None else label_of(r["name"], r["username"]), r["department"],
             r["node_type"], {k: r[k] for k in COUNTS}) for r in rows]


def stats(since: float | None = None, until: float | None = None, tz_minutes: int = 0) -> dict:
    """Usage from `since` to `until` (seconds since the epoch; None: since the last reset / until now), per project
    and per node type, per department (and its people, and their projects) and per person (and their projects): runs,
    reuses, compute seconds (and on a GPU), frames, who asked, when last used (since the last reset, in the range or
    before it), and per-day series of runs, reuses and seconds. Days
    are counted in the viewer's time zone (`tz_minutes` east of UTC)."""
    NOBODY = i18n.t("farm.no_department")  # an account without a department (the administrator until they choose one)
    from ..extensions import extensions
    from ..extensions.status import extension_status
    from ..accounts import department_label, departments
    from ..nodes import node_types

    begun = start()
    if since is not None and until is not None and until <= since:
        raise Invalid(Msg("E-USAGE-RANGE"))
    lo = max((t for t in (since, begun) if t is not None), default=None)
    uses = [u for u in _uses() if begun is None or u[0] >= begun]
    ranged = [u for u in uses if (lo is None or u[0] >= lo) and (until is None or u[0] < until)]

    now = time.time()

    def day(t: float) -> int:  # the day it falls on in the viewer's time zone
        return int((t + tz_minutes * 60) // DAY_S)

    first = day(lo if lo is not None else min((u[0] for u in ranged), default=now))
    last = day(until - 0.001 if until is not None else now)
    if last - first >= MAX_DAYS:
        raise Invalid(Msg("E-USAGE-TOOLONG", days=MAX_DAYS))
    days = [time.strftime("%Y-%m-%d", time.gmtime(d * DAY_S)) for d in range(first, last + 1)]

    def daily() -> dict:
        return {k: [0] * len(days) for k in ("runs", "reuses", "seconds")}

    types = node_types()
    exts = extensions()
    projects: dict[str, dict] = {}

    def title(runtime: str) -> str:
        return i18n.t("farm.core_nodes") if runtime == "core" else exts[runtime].title if runtime in exts else runtime

    def project(runtime: str, installed: bool = False) -> dict:
        if runtime not in projects:
            core = runtime == "core"
            projects[runtime] = {"name": runtime, "title": title(runtime), "core": core, "installed": core or installed,
                                 **_counts(), "daily": daily(), "nodes": {}}
        return projects[runtime]

    def runtime_of(type_id: str) -> str:
        t = types.get(type_id)
        return t.runtime if t else type_id.split(".")[0]  # a type no longer there: by the namespace of its id

    def rows(type_id: str) -> tuple[dict, dict]:
        """The project's row and the node type's."""
        t = types.get(type_id)
        row = project(runtime_of(type_id))
        if type_id not in row["nodes"]:
            row["nodes"][type_id] = {"id": type_id, "subtitle": t.subtitle if t else type_id, **_counts()}
        return row, row["nodes"][type_id]

    runtimes = {t.runtime for t in types.values()}
    for name, ext in exts.items():
        if name in runtimes and extension_status(ext)["installed"]:
            project(name, installed=True)
    project("core")
    for type_id, t in types.items():
        if t.runtime in projects:
            rows(type_id)

    # departments (their people, their projects) and people (their projects)
    listed = departments()
    # keyed by the department's value (an id of the factory list, or as an administrator wrote it); `name` is how it shows
    depts: dict[str, dict] = {d: {"value": d, "name": department_label(d), "listed": True, **_counts(), "daily": daily(), "people": {}} for d in listed}
    people: dict[str, dict] = {}

    def sub(parent: dict, key: str, name: str, **extra) -> dict:
        if name not in parent[key]:
            parent[key][name] = {"name": name, **extra, **_counts()}
        return parent[key][name]

    for t, _, _, type_id, _ in uses:
        for row in rows(type_id):
            row["last"] = max(row["last"] or t, t)
    for t, person, dept, type_id, use in ranged:
        i = day(t) - first
        row, node = rows(type_id)
        _add(row, use, person)
        _add(node, use, person)
        who, where = person or NOBODY, dept or NOBODY
        d = depts.setdefault(where, {"value": where, "name": department_label(where) if dept else NOBODY, "listed": False, **_counts(), "daily": daily(), "people": {}})
        p = people.setdefault(who, {"name": who, "departments": set(), **_counts(), "daily": daily(), "projects": {}})
        dp = sub(d, "people", who, projects={})
        runtime = runtime_of(type_id)
        for r in (d, p, dp, sub(dp, "projects", runtime, title=title(runtime)), sub(p, "projects", runtime, title=title(runtime))):
            _add(r, use, person)
        p["departments"].add(department_label(where) if dept else NOBODY)
        for series in (row["daily"], d["daily"], p["daily"]):
            for k in ("runs", "reuses", "seconds"):
                series[k][i] += use[k]
    for t, person, dept, _, _ in uses:  # everyone's last use, in the range or before it
        if (p := people.get(person or NOBODY)) is not None:
            p["last"] = max(p["last"] or t, t)
        if (d := depts.get(dept or NOBODY)) is not None:
            d["last"] = max(d["last"] or t, t)

    def order(items) -> list[dict]:
        return sorted(items, key=lambda r: (-r["seconds"], -r["runs"] - r["reuses"], r["name"]))

    def finish_series(r: dict) -> dict:
        r["daily"]["seconds"] = [round(s, 1) for s in r["daily"]["seconds"]]
        return r

    out = []
    for row in projects.values():
        nodes = sorted((_done(n) for n in row["nodes"].values()), key=lambda n: (-n["runs"] - n["reuses"], n["subtitle"]))
        out.append({**_done(finish_series(row)), "nodes": nodes})
    out.sort(key=lambda p: (p["core"], -p["runs"] - p["reuses"], -p["seconds"], p["title"].lower()))
    dept_out = [{**_done(finish_series(d)), "people": order(
        {**_done(dp), "projects": order(_done(x) for x in dp["projects"].values())} for dp in d["people"].values())}
        for d in depts.values()]
    dept_out.sort(key=lambda d: (not d["listed"], d["value"] == NOBODY, -d["seconds"], -d["runs"] - d["reuses"],
                                 listed.index(d["value"]) if d["listed"] else 0))
    people_out = order({**_done(finish_series(p)), "departments": sorted(p["departments"]),
                        "projects": order(_done(x) for x in p["projects"].values())} for p in people.values())
    return {"since": lo, "until": until, "start": begun, "undo": bool(db().meta("usage.history")), "days": days,
            "projects": out, "departments": dept_out, "people": people_out, "nobody": NOBODY}
