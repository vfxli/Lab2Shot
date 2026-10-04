"""Cooking over HTTP: every cook is a job in the farm's queue (lab2shot/farm). Before submitting, the editor looks at
the cook (/api/plan: the frame range its inputs cover and what it computes; no time estimate). Users follow
their jobs by Server-Sent Events (web UI) or by polling (DCC plugins, the command line), see their own jobs and how
busy the queue is, and load a job's graph again with whether its results are still cached; the admin page (/admin,
no link from the user pages) sees every job and who started it, chooses the GPUs that take jobs, manages the models
kept loaded between jobs and sees how much each project is used."""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from typing import Any
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from .routes import MAX_STREAMS, Access, Body, Days, Router

HISTORY_MOST = 5000  # the most job records answered at once (the page limit of 后台「任务记录」; the same as the 5000 lines of logs.py /log)
from .. import i18n, logs
from ..database import json_of
from ..data.packet import Packet, packet_dir
from ..engine import CookError, Graph, GraphError
from ..engine.resident import pool as resident
from ..engine.templates import apply_values, pick_targets
from ..farm import Client, Job, farm, forget_job, history, job_row, timings
from ..farm.queue import cache_mark, finished_of, listed_version, spoken_record
from ..errors import Invalid, NotFound, TooManyTries
from ..io.digest import sha256
from ..messages import Msg, localized
from ..text import decimal
from . import auth, graphs, owners, quota, restart
from .wire import FRESH, NEVER, WAITS, generations
from .access import admit, audit, particulars
from .graphs import GraphRequest
from .packets import evaluation_of

router = Router(prefix="/api", tags=["Job Queue"])
admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # this module's admin routes (app.py includes each module's)


def client_of(request: Request, declared: dict | None = None) -> Client:
    """Who is asking: the account, and what the request shows about where it comes from."""
    return Client.of(auth.me(request), auth.details(request, declared))


def submit(request: Request, data: dict, graph: Graph, targets: list[str], declared: dict, force: bool,
           version: int | None = None, show: list[str] | None = None, follows: str = "") -> Job:
    """Queue a graph admitted for the account (server/access.py admit). It cooks as its submitter's own account
    (farm/queue.py Farm.submit), in its own cache (data/store.py), where its status (server/packets.py) finds the
    results: a node whose upload is not this account's fails at itself, the rest cooks as usual."""
    # a full quota stops here for everything 「计算」 or 「提交」 would write into results. The page greys the buttons by
    # the same rule; this check blocks submissions that bypass the page.
    me = auth.me(request)
    quota.refuse_if_full(me.id)
    return farm().submit(data, graph, targets, client_of(request, declared), force, version=version, show=show, follows=follows)


class PlanRequest(GraphRequest):
    target: str
    force: bool = False
    show: list[str] | None = None  # as a JobRequest's: the look is of the cook it would submit


class JobRequest(GraphRequest):
    """POST /api/jobs, the one way anything queues a graph: the page, DCC plugins, scripts, the command line. The
    graph whole or by version (GraphRequest), or `template` (id or name) instead; `values` set on it
    (exposed names, or node.param); what to cook, exactly one of
      cook     a node and its upstream (a 「输出」 collects and packs what is wired into it: transfer/outputs.py)
      deliver  these 「输出」; [] every 「输出」 of the graph, like Nuke's Render All (DCC plugins, `lab2shot cook`)
    `frames` [first, last] within what the inputs have; `force` cook again what is cached; `client` what the client
    says of itself (auth.details: application, platform ...); `version` the page's cook-inputs version it submits at,
    carried back on every event of the job (farm/queue.py Job.version); `show` the outputs of the node it shows that the
    viewer displays (none: every output of it), so the rest is not computed for nothing (engine/cook.py `show`);
    `formats` {kind of 3D data: format} the deliveries are wanted in (engine/deliver_formats.py: the graph is rewritten
    before it is queued, refused when a kind can't be written so). `template` is any tool's id (server/tools.py find).
    `follows`: an earlier job of the same account this one follows (the page's focus mode computing a job a DCC plugin
    opened in it: the plugin takes the newest finished job that follows its own as the new version). Only one's own
    job: another account's, or none, is refused alike (owners.job_followed)."""

    template: str | None = None
    values: dict[str, Any] = {}
    cook: str | None = None
    deliver: list[str] | None = None
    frames: list[int] | None = None
    force: bool = False
    client: dict = {}
    version: int | None = None
    show: list[str] | None = None
    formats: dict[str, str] = {}
    follows: str | None = Field(None, max_length=64)
    # the submitter's key for this one submission (a click of 「计算」, one call of a script): sent again after an
    # answer that never arrived (the line dropped, the server restarted), it gets the job the first one made instead of
    # a second (submitted_before). Kept in the record's client details (`submit`)
    submit: str | None = Field(None, max_length=64)


SUBMIT_KEY = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SUBMIT_KEPT_S = 86400  # a submit key finds the job it made within this long
# (account, key) -> [the lock its submissions take in turn, how many are in or waiting]: the entry goes only when the
# last one leaves, so a submission arriving meanwhile always takes the same lock (never a second one beside it)
_submitting: dict[tuple[int, str], list] = {}
_submitting_lock = threading.Lock()


@contextlib.contextmanager
def _one_at_a_time(user_id: int, key: str):
    """Submissions of one account with the same key, one after another: the second sees the job the first made."""
    with _submitting_lock:
        entry = _submitting.setdefault((user_id, key), [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _submitting_lock:
            entry[1] -= 1
            if entry[1] == 0:
                _submitting.pop((user_id, key), None)


def submitted_before(user_id: int, key: str) -> str | None:
    """The job this account's earlier submission with `key` made (not one the server cancelled as it closed: that one
    never ran, and the same click may queue it now), within SUBMIT_KEPT_S; None: there is none. The jobs_user index
    (user_id, submitted) narrows it to this account's jobs of the last day before the record is looked into."""
    row = farm().db.row("SELECT id FROM jobs WHERE user_id = ? AND submitted > ? AND state != 'cancelled' "
                        "AND json_extract(record, '$.client.details.submit') = ? ORDER BY submitted DESC LIMIT 1",
                        (user_id, time.time() - SUBMIT_KEPT_S, key))
    return row["id"] if row else None


@router.post("/jobs", access=Access.user("Submit a cook: a graph or a template, cooking one node or packaging outputs",
                                          owned=owners.job_followed, body_ids=("follows",)),
             summary="Submit a cook (used by the web page, DCC plugins, scripts and the command line): a graph (whole or by version) "
                     "or a template, values setting parameters; either cook one node (with its upstream) or deliver these outputs "
                     "([] for all); the job is queued and each node runs on a free GPU or CPU slot when its turn comes; returns the "
                     "job id")
def queue_job(req: JobRequest, request: Request) -> dict:
    """One task for what the request asks (JobRequest), queued like every other (farm/queue.py); with a submit key, at
    most one per key (submitted_before)."""
    if not req.submit:
        return _queue_job(req, request)
    if not SUBMIT_KEY.match(req.submit):
        raise Invalid(Msg("B-JOB-SUBMITKEY"))
    me = auth.me(request).id
    with _one_at_a_time(me, req.submit):
        if (found := submitted_before(me, req.submit)) is not None:
            return {"job": found, "again": True}
        return _queue_job(req.model_copy(update={"client": {**req.client, "submit": req.submit}}), request)


def _queue_job(req: JobRequest, request: Request) -> dict:
    if [req.cook, req.deliver].count(None) != 1:
        raise Invalid(Msg("B-JOB-WHAT"))
    follows = request.state.owned  # the job it follows, the account's own (owners.job_followed)
    if req.template:
        from .tools import find

        data = find(req.template, request)[0]
    else:
        data, _ = graphs.resolve(req, request)
    if req.values:
        data = apply_values(data, req.values)
    if req.formats:
        from ..engine.deliver_formats import with_formats

        admit(request, data)  # what the account may not use is refused as admit says it, before the graph is read here
        data = with_formats(data, req.formats)
    if req.frames is not None:
        data = {**data, "frames": req.frames}
    graph = admit(request, data)
    if req.deliver is not None:
        targets = pick_targets(graph, req.deliver) if req.deliver else graph.deliveries()
        if not targets:
            raise Invalid(Msg("B-COOK-NOOUTPUT"))
    else:
        if req.cook not in graph.nodes:
            raise Invalid(Msg("B-COOK-NOSUCHNODE", node=req.cook))
        targets = [req.cook]
    return {"job": submit(request, data, graph, targets, req.client, req.force, req.version, req.show, follows).id}


@router.post("/plan", access=Access.user("Before submitting: frame range and the nodes to cook"), summary="Before submitting: the inputs' frame range, which frames this run cooks, which nodes need cooking (the rest "
                                                                                             "are cached)")
def plan(req: PlanRequest, request: Request) -> dict:
    """For scripts and DCC clients (the editor gets the shown node's with every status reply): timings.look. Reads its
    Evaluation from the same cache /api/status uses (engine/evaluations.py)."""
    return timings.look(evaluation_of(req, request), req.target, req.force, req.show)


GONE = Msg("E-JOB-GONE")


KEEPALIVE_S = 15  # an event stream with nothing to say sends a named `ping` event this often: a proxy keeps the line
# open, and the page (platform/events.ts) counts it as a sign of life (a comment line never reaches its script)


# what a job's event says about the cards (engine/cook.py, farm/queue.py): only for whoever sees them (farm.cards), in
# the events one poll gets and in the stream alike
CARD_KEYS = ("gpus", "gpu", "gpu_name", "waiting_detail")
# the same in what Farm.view and the history answer, to the page's queue and the admin page's alike (a 二级管理员 with
# queue.manage need not have farm.cards)
QUEUE_CARDS = ("gpus", "jobs[].cards", "jobs[].waiting_detail")
HISTORY_CARDS = ("history[].cards",)


@router.get("/jobs/{job_id}/state", access=Access.user("Query your own job", owned=owners.job, hides={"farm.cards": tuple(f"events[].{k}" for k in CARD_KEYS)}), summary="Query your own job: events since `since`, state (queued / cooking / done / failed / cancelled), files written")
def job_state(job_id: str, request: Request, since: int = 0) -> dict:
    return request.state.owned.poll(since)


@router.get("/jobs/{job_id}/events", access=Access.user("Live progress of your own job", owned=owners.job, hides={"farm.cards": CARD_KEYS}), summary="Live push of your own job's events (Server-Sent Events); the browser reconnects by itself and continues where "
                                                                                                                                                   "it broke off (from the start after a server restart); closed at once when the account is disabled or expires")
def events(job_id: str, request: Request, last_event_id: str = "") -> StreamingResponse:
    """Every event carries the id <server run>.<its number> (events are numbered permanently: farm/queue.py Job.after):
    a browser reconnecting (the connection dropped, the server restarted) sends the last one back (Last-Event-ID; a page
    that opens the stream anew, which a browser sends no such header for, puts it in `last_event_id`) and gets what came
    after it; from another run of the server (a job queued again after a restart) it gets the job's
    events from the start. The stream ends when the job does,
    and when the server is about to restart (the browser then retries until it is back), and when the account cannot
    be used any more (disabled, expired, logged out).

    The stream waits on the event loop (Job.changed), not on a thread: open streams cost the server no thread, so any
    number of pages following jobs never hold up another request."""
    job = request.state.owned
    credential = auth.credential(request)
    me = auth.me(request)
    run, _, last = (request.headers.get("last-event-id") or last_event_id).partition(".")
    sent = (decimal(last) + 1) if run == restart.BOOT and decimal(last) is not None else 0
    # its slot taken last, right before the stream that gives it back (its `finally`) is handed over: nothing between
    if not auth.guards().streams.take(me.id, MAX_STREAMS):  # the ones already open are never cut: only this one waits
        raise TooManyTries(Msg("E-ACCESS-TOOMANYSTREAMS", most=MAX_STREAMS))

    def over() -> bool:  # the job's last event is out, the server is about to restart, or the account is out
        from .. import accounts

        # a fresh look each time (not auth.session's per-request cache): this connection outlives one request
        return (bool(job.events) and job.events[-1]["type"] == "finished") or farm().closed or \
            accounts.session(credential) is None

    async def stream():
        nonlocal sent
        try:
            yield "retry: 2000\n\n"
            while True:
                batch = job.after(sent)  # the job's lock only while the list is copied, never across a wait
                for e in batch:
                    if e.get("type") == "node_done" and e.get("outputs") and "gens" not in e:
                        # the fresh packets' generations, as the status reply gives them (wire.py generations)
                        e = {**e, "gens": await run_in_threadpool(generations, e["outputs"], job.client.user)}
                    # made on the farm's thread: its messages said again in this reader's language (server/lang.py)
                    yield f"id: {restart.BOOT}.{e['n']}\ndata: {json.dumps(localized(e), ensure_ascii=False)}\n\n"
                    sent = e["n"] + 1
                if batch:
                    continue
                if await run_in_threadpool(over):  # the account check reads the database: on a thread
                    return
                if not await job.changed(sent, KEEPALIVE_S):
                    yield "event: ping\ndata: {}\n\n"  # a data line is required: an event without one is never dispatched
        finally:  # however it ends (the job finished, the line dropped, the server is closing): the slot goes back
            auth.guards().streams.give_back(me.id)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": FRESH})


@router.post("/jobs/{job_id}/cancel", access=Access.user("Cancel your own job", owned=owners.job), summary="Cancel your own job: a queued one is removed, a cooking one stops")
def cancel(job_id: str, request: Request) -> dict:
    farm().cancel(job_id, None)  # whose it is: the route's owner said
    return {"ok": True}


@router.delete("/jobs/{job_id}", access=Access.user("Delete your own finished job", owned=owners.job_record),
               summary="Delete one **finished** job: its whole job folder (graph, media, output folders and zip, logs) is deleted and "
                       "cache only it referenced is cleaned up afterwards (anything other jobs still reference stays); a running one "
                       "is cancelled first. The answer says how much was freed and the usage afterwards")
def forget(job_id: str, request: Request) -> dict:
    """Deleting a task removes it whole (transfer/tasks.py: its folder with its outputs, its row and references); the
    cache entries only it referenced go with the next cleaning. Measured before deletion (`drop_for_job`), the usage
    after it."""
    dropped = quota.drop_for_job(job_id, request.state.owned)
    forget_job(job_id, None)  # whose it is: the route's owner said
    return {"ok": True, **dropped, **quota.usage(int(request.state.owned["user"]))}


@router.delete("/jobs", access=Access.user("Delete all your own finished jobs"),
               summary="Delete **all** your finished jobs at once with the space they take (job folders; cache only they referenced is "
                       "cleaned up afterwards); queued and cooking ones stay, and nobody else's are touched. The answer says how many "
                       "were deleted and how much was freed")
def forget_all(request: Request) -> dict:
    me = auth.me(request).id
    # Usage without traffic: traffic is visible only to the admin side, and no user-facing route sends it
    # (server/quota.py my_storage)
    return {**forget_finished(me), **quota.usage(me)}


def forget_finished(user_id: int) -> dict:
    """Free space in one step: delete all of account `user_id`'s finished jobs, each task whole (the account's own
    「删除全部」, and the administrator's for an account: `lab2shot admin jobs --clear` through the running server).

    Each job goes through exactly the same path as a single deletion (`quota.drop_for_job` + `forget_job`). Queued or
    computing jobs are skipped (a job counts as finished only after cancellation), and the answer states how
    many were skipped."""
    going = finished_of(user_id)
    jobs, freed, skipped = 0, 0, len(farm().live_of(user_id))  # queued or computing: only a cancelled job counts as finished
    for jid in going:
        try:
            row = job_row(jid)
        except (NotFound, KeyError, ValueError):
            continue
        try:
            freed += int(quota.drop_for_job(jid, row).get("bytes") or 0)
            forget_job(jid, user_id)
        except Invalid:  # one of them started running again meanwhile: skip it and remove the rest
            skipped += 1
            continue
        jobs += 1
    return {"ok": True, "jobs": jobs, "skipped": skipped, "bytes": freed}


def forget_group(user_id: int, key: str) -> dict:
    """Delete a group of account `user_id`'s tasks (transfer/groups.py): every task in it, each exactly as a single
    deletion does (`quota.drop_for_job` + `forget_job`: its folder with its outputs, its row and references). The group
    is looked up with the account in the query, so another account's group is not there (NotFound, the same answer as
    none). Tasks still queued or computing are left as they are and counted (cancel them first)."""
    from ..transfer import groups

    ids = groups.tasks_of(user_id, key)
    if not ids:
        raise NotFound(Msg("E-TASKGROUP-GONE"))
    live = farm().active_ids()
    jobs, freed, skipped = 0, 0, 0
    for jid in ids:
        if jid in live:
            skipped += 1
            continue
        try:
            row = job_row(jid)
        except NotFound:
            row = None
        if row is not None and int(row["user"]) != int(user_id):  # a second check: the query already named the account
            continue
        try:
            if row is not None:
                freed += int(quota.drop_for_job(jid, row).get("bytes") or 0)
            forget_job(jid, user_id)
        except Invalid:  # it started running again meanwhile: left, like the ones already running
            skipped += 1
            continue
        except NotFound:  # gone meanwhile (expired, deleted by another request)
            continue
        jobs += 1
    return {"ok": True, "jobs": jobs, "skipped": skipped, "bytes": freed}


@router.delete("/task-groups/{key}", access=Access.user("Delete one of your job groups"),
               summary="Delete one of your job groups: every **finished** job in it is deleted whole (job folder with output folders "
                       "and zip, as for one job); queued and cooking ones stay (cancel them first). Only your own groups are found. "
                       "The answer says how many were deleted, how many skipped, how much was freed and the usage afterwards")
def forget_task_group(key: str, request: Request) -> dict:
    me = auth.me(request).id
    return {**forget_group(me, key), **quota.usage(me)}


class GroupName(Body):
    name: str = Field("", max_length=1000)  # cleaned and cut to 64 characters (groups.clean_name); "" = the automatic name


@router.put("/task-groups/{key}/name", access=Access.user("Rename one of your job groups"),
            summary="Rename one of your job groups (only the shown name; the grouping is unchanged and later jobs of the group use "
                    "the name too); an empty name goes back to the automatic one. Only your own groups are found. The answer is the "
                    "group's shown name now")
def rename_task_group(key: str, req: GroupName, request: Request) -> dict:
    from ..transfer import groups

    return {"ok": True, "name": groups.rename(auth.me(request).id, key, req.name)}


# The account's own tasks the 队列 window lists: all of them that have not expired (任务保留天数 bounds how many there
# are; they are shown by group, collapsed, so no group is ever cut in half). This is only the safety net under that,
# the most one answer carries: the same as the admin 任务记录's page (HISTORY_MOST).
HISTORY = HISTORY_MOST


@router.get("/queue", access=Access.user("Queue: your own jobs; others' show only whether queued or cooking and their place", hides={
    "farm.cards": (*QUEUE_CARDS, "switches.gpu")}, snapshot=True), summary="Queue: the machine and the GPUs taking jobs, cooking and queued jobs (all of yours; of others only whether "
                                                                           "queued or cooking and their place); the version of your job records history_version (fetch /api/queue/history "
                                                                           "only when it changes); whether you can still add anything (used, limit, full; the parts' details are asked "
                                                                           "once from /api/my/storage by the queue window's My Usage); load=0: without the machine load and GPU use that "
                                                                           "change every second (304 when nothing changed while the editor's queue window is closed)")
def queue(request: Request, load: bool = True) -> dict:
    u = auth.me(request)
    view = farm().view(u.id, auth.can(request, "queue.manage"))
    if not load:
        view["machine"] = None
        view["gpus"] = [{**g, "used_mb": 0, "utilization": 0, "temperature": 0} for g in view["gpus"]]
    # The account's finished tasks are not in this answer, which is polled every 1.5-30 seconds: only the version of
    # their list (taken after the view: a task the view no longer shows going has its end in it already). The page
    # asks /api/queue/history when the version is not the one it has. The answer carries only the three "full or not"
    # figures (quota.gate), not the breakdown by part with its explanations: that is requested once by 「我的占用」
    # through /api/my/storage (when the window opens and after cleaning), which is not a poll.
    return {**view, "history_version": listed_version(u.id), "storage": quota.gate(u.id)}


def _history_key(request: Request) -> str:
    from . import revisions

    return sha256(repr((auth.me(request).id, revisions.account(request), listed_version(auth.me(request).id))))[:24]


@router.get("/queue/history", access=Access.user("Queue: your own job records", hides={"farm.cards": HISTORY_CARDS},
                                                 keyed=_history_key),
            summary="All your jobs not yet expired (submitted with Cook, at most 5000), newest first, each with its group and "
                    "whether its results are still in the cache (all / partly / cleaned up); when version equals /api/queue's "
                    "history_version the records have not changed (it is the ETag; 304 when unchanged)")
def queue_history(request: Request) -> dict:
    u = auth.me(request)
    version = listed_version(u.id)
    mine_ = history(HISTORY, u.id)
    # the records of the account's tasks, found by the account (a bound parameter)
    rows = {r["id"]: r for r in farm().db.rows("SELECT j.id, j.record FROM jobs j JOIN tasks t ON t.id = j.id WHERE j.user_id = ?",
                                               (u.id,))} if mine_ else {}
    for e in mine_:
        r = rows.get(e["id"])
        # the task's graph is read only when the mark is not cached (cache_mark)
        e["cache"] = cache_mark(e["id"], None, json_of(r["record"]), u.id) \
            if r and e["event"] == "finished" else None
    return {"version": version, "history": mine_}


@router.get("/load", access=Access.user("How busy the server is now", snapshot=True), summary="How busy the server is now (one line every page can show): how many jobs wait in the queue and how many are "
                                                                                            "cooking, CPU and GPU slots used and in all, CPU and memory use in percent, and each GPU's load and memory use "
                                                                                            "in percent")
def load(request: Request) -> dict:
    """The one small answer a page can ask for often: it reads the samples the queue already has (farm/load.py, once a
    second however many ask; the cards' from the inventory thread), never a new one. Every account sees it whole:
    whether each card is busy and how full its memory is, by number, never which job is on it (that stays with
    farm.cards, in the queue's answers).

    While the editor is idle it polls only this route: the two switches, the frame limit and the account's usage are
    folded into it instead of a separate queue poll. Most bytes of a poll are request headers and the session cookie
    rather than data, so slowing the interval saves little; merging the polls is what saves."""
    # The switches and frame limit are answered by the farm (Queue.load); the account's usage is per account and only
    # this route knows who is asking. The service itself (this run's id, the UI version, whether a restart is pending)
    # is included as well, so a logged-in page has only this one poll.
    from .settings import server_now

    return {**farm().load(), "storage": quota.gate(auth.me(request).id), "server": server_now(request)}


@router.get("/jobs/{job_id}", access=Access.user("Load one of your jobs: the graph as submitted and whether its results are still cached", owned=owners.job_record), summary="Load one of your jobs: the whole graph as submitted (nodes, parameters, wires, views) and whether its results "
                                                                                                                                                                        "are still in the cache (all / partly / cleaned up)")
def job_load(job_id: str, request: Request) -> dict:
    row = request.state.owned
    return {"id": job_id, "graph": row["graph"], "title": spoken_record(dict(row["record"]))["title"], "submitted": row["record"].get("submitted"),
            "cache": cache_mark(job_id, row["graph"], row["record"], row["user"]),
            # what it computed (node ids) and the job it followed: the page's focus mode computes the same again
            "nodes": row["record"].get("nodes") or [], "follows": row["record"].get("follows") or ""}


# ------------------------------------------------------------------ streaming partial results

def _streaming_packet(job: Job, node: str, port: str) -> tuple[Packet, list[int], list[int]]:
    """The incomplete packet a streaming node is writing now (its viewer meta, the frames that have appeared so far,
    and the node's whole frame range). NotFound when there is no partial result: the node is not there, the packet
    is already done (the content-addressed address serves it), or the cook has not reached it / it was cancelled (its
    folder is gone)."""
    from ..data.payloads import partial_frames
    from ..data.types import channels_of

    if node not in job.graph.nodes:
        raise NotFound(Msg("B-COOK-NOSUCHNODE", node=node))
    # the plan the cook runs it with (Engine.ran, via Job.ran): after a pending switch or a wired value, and in a
    # block (its latest item), the fingerprints are those the cook found, never the plan from before it
    ran = next((r for inst, r in reversed(list(job.ran.items())) if inst.node == node), None)
    if ran is None:
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    plan = ran.plan
    if port not in plan.outputs:
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    d = packet_dir(plan.outputs[port])
    if not d.exists() or Packet.exists(d):  # not started, or already finished
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    info = ran.info
    type_ = job.graph.output_type(node, port)
    # which frames are there so far is the data layer's answer, never a file-name pattern written here:
    # partial_frames reads the folder a streaming node is filling
    frames = partial_frames(d)
    # names relative to the packet dir: file_path reads a relative name
    meta = {"files": dict(frames), "width": info.width, "height": info.height}
    if channels_of(type_) <= 2:  # still computing: give the viewer a 0..1 range for now; the finished packet carries its own measured range
        meta["values"], meta["range"] = True, [0.0, 1.0]
    else:
        # Three or four channels are viewed as a picture (the payloads.is_data heuristic): display_frame reads
        # colorspace and alpha (shown_as_is, has_alpha), and a missing key raises KeyError. As with a written picture
        # packet, every picture packet is in the working space (io/color.py), and four channels carry alpha.
        from ..io.color import working_space

        meta["colorspace"], meta["alpha"] = working_space(), channels_of(type_) == 4
    return Packet(d, type_, meta), sorted(frames), [int(f) for f in info.frames]


@router.get("/jobs/{job_id}/partial/{node}/{port}", access=Access.user("View partial results while cooking", owned=owners.job), summary="Streaming while cooking: the frames a node still cooking has written (frames done, frames in all, size and "
                                                                                                                                    "type); only for the job's owner, not cached by the browser")
def partial(job_id: str, node: str, port: str, request: Request) -> Response:
    p, frames, whole = _streaming_packet(request.state.owned, node, port)
    out = {"frames_done": frames, "total": len(whole), "width": p.meta["width"], "height": p.meta["height"], "type": p.type}
    return Response(json.dumps(out), media_type="application/json", headers={"Cache-Control": NEVER})


@router.get("/jobs/{job_id}/partial/{node}/{port}/frame/{frame}.png", access=Access.user("View one frame while cooking", owned=owners.job), summary="Streaming while cooking: the **proxy image** of one frame of a node still cooking, for the viewer (the same "
                                                                                                                                                   "proxy as the final address once written, the same bytes); only for the job's owner, not cached by the browser, "
                                                                                                                                                   "with X-Content-Sha256 in the headers")
def partial_frame(job_id: str, node: str, port: str, frame: int, request: Request) -> Response:
    import mimetypes

    from ..data.payloads import is_data, is_labels
    from ..data.types import channels_of
    from ..view.frames import display_frame, map_view_frame
    from ..view.proxy import form_suffix, picture_of, tier

    p, frames, _ = _streaming_packet(request.state.owned, node, port)
    if frame not in frames:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame))
    if not channels_of(p.type):
        raise Invalid(Msg("E-VIEW-NOTIMAGE"))
    # a provisional view, never into the packet's own _view
    picture = map_view_frame if is_data(p) else display_frame
    path = picture(p, frame, cache=p.dir / "_partial")
    if path is None:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame))
    # Streaming previews are sent as proxies as well: the viewer receives only proxies, and this is another path that
    # sends 2D pixels; sending full original frames here would bypass the proxy. The proxy is made in `_partial`, like
    # the provisional display image above, and not in the packet's own `_view` (an unfinished packet's range still
    # changes).
    px = tier()
    # an id map (is_labels) is scaled as its finished proxies are: nearest (view/proxy.py shrink)
    small = picture_of(path, p.dir / "_partial" / f"proxy.{frame}.{px}{form_suffix(p)}", px, labels=is_labels(p))
    data = small.read_bytes()
    kind = mimetypes.guess_type(small.name)[0] or "image/png"
    return Response(data, media_type=kind, headers={"Cache-Control": NEVER, "X-Content-Sha256": sha256(data)})


def _partial_points(job: Job, node: str, port: str, camera: str | None) -> tuple[tuple, list[int], tuple[int, ...]]:
    """A depth or position map still being computed, viewed as a point cloud: the `how` the view worker needs, and the
    frames already written.

    The view worker is a separate process and knows neither jobs nor graphs (server/view_worker.py), so this route
    works out which folder, type, width, coordinate system, frame count and camera, and passes them on. `how` does not
    include the number of frames written: otherwise every written frame would make a new view, every chunk address
    would change and everything the browser holds would be invalidated. The worker scans the folder each time for
    which frames have files."""
    p, frames, whole = _streaming_packet(job, node, port)
    if not frames:
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    how = ("points_partial", str(p.dir), p.type, int(p.meta["width"]), str(p.meta.get("space") or ""),
           tuple(whole), camera or None)
    # `whole` (the frames of the whole range) is returned separately rather than taken from `how` by index: if the item
    # count of `how` changed, an index would silently point to another item
    return how, frames, tuple(whole)


@router.get("/jobs/{job_id}/partial/{node}/{port}/points", access=Access.user("View the point cloud description while cooking", owned=owners.job_seen_through, lane=WAITS), summary="Streaming while cooking: the description of the frames a node still cooking has written, seen as a point cloud "
                                                                                                                                                                    "(one part per frame; each part's address never changes afterwards); only for the job's owner")
def partial_points(job_id: str, node: str, port: str, request: Request, camera: str | None = None) -> Response:
    from .view_data import respond_description
    from .view_worker import describe

    how, _, _whole = _partial_points(request.state.owned, node, port, camera)
    url = f"/api/jobs/{job_id}/partial/{quote(node, safe='')}/{quote(port, safe='')}/points/{{part}}" + (f"?camera={quote(camera, safe='')}" if camera else "")
    return respond_description(describe(how, url), request.headers.get("accept-encoding", ""))


@router.get("/jobs/{job_id}/partial/{node}/{port}/points/{part}", access=Access.user("View the point cloud data while cooking (one frame)", owned=owners.job_seen_through, lane=WAITS), summary="Streaming while cooking: the points of one frame of the node still cooking (binary, gzip; one part per frame). "
                                                                                                                                                                                       "A frame not written yet is 404, so a browser never keeps an empty frame")
def partial_points_part(job_id: str, node: str, port: str, part: str, request: Request, camera: str | None = None) -> Response:
    """One frame per chunk (view_data.chunk_plan one_frame_chunks): chunk `c{k}` is frame k of this node.

    A frame not yet written answers 404, not an empty frame: a written frame's bytes never change afterwards (that is
    what `streams=True` means for this node family), so the browser is right to cache a frame it received; if it
    cached an empty one, the frame would never become visible once computed."""
    from .view_data import respond
    from .view_worker import part as view_part

    how, frames, whole = _partial_points(request.state.owned, node, port, camera)
    k = decimal(part[1:]) if part.startswith("c") and decimal(part[1:]) is not None else -1
    if not 0 <= k < len(whole) or whole[k] not in frames:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=whole[k] if 0 <= k < len(whole) else -1))
    return respond(view_part(how, part), request.headers.get("accept-encoding", ""), kept=False)


# ------------------------------------------------------------------ admin


def _particulars(request: Request, entry: dict) -> dict:
    """A job (or its record) as the admin page lists it to this login: from where it came (its client's particulars:
    the request's address, browser, computer) only when it manages that account (access.particulars)."""
    client = entry.get("client") or {}
    return {**entry, "client": particulars(auth.session(request), client.get("user"), client)} if client else entry


def _admin_view(request: Request, view: dict) -> dict:
    return {**view, "jobs": [_particulars(request, j) for j in view.get("jobs", [])]}


@admin.get("/queue", access=Access.admin("queue.manage", hides={"farm.cards": QUEUE_CARDS}), summary="Queue: everyone's jobs, with each job's account and everything about its request (IP, browser, computer name "
                                                                                                     "and system user reported by the DCC plugin...); history_version: the version of the job records (fetch "
                                                                                                     "/api/admin/history only when it changes)")
def admin_queue(request: Request) -> dict:
    view = _admin_view(request, farm().view(admin=True))
    return {**view, "history_version": listed_version(None)}  # taken after the view, as for the account's own


class GpuRequest(Body):
    authorized: list[str]  # UUIDs of the GPUs that take jobs


class Switches(Body):
    gpu: bool | None = None  # GPU jobs (queue.gpu_jobs)
    compute: bool | None = None  # compute jobs (queue.compute_jobs)


@admin.put("/queue/switches", access=Access.admin("queue.manage", hides={"farm.cards": QUEUE_CARDS}), summary="The GPU job and compute job switches (settings queue.gpu_jobs, queue.compute_jobs): when off no new jobs are "
                                                                                                              "taken and running ones finish; administrators and deputy administrators may switch them")
def queue_switches(req: Switches, request: Request) -> dict:
    from ..config import settings

    settings().save({k: v for k, v in (("queue.gpu_jobs", req.gpu), ("queue.compute_jobs", req.compute)) if v is not None})
    farm().wake()  # jobs waiting for the switch look again
    return _admin_view(request, farm().view(admin=True))


@admin.put("/gpus", access=Access.admin("gpu.authorize"), summary="Authorize which GPUs take jobs (by UUID); a GPU no longer authorized finishes the jobs it has and takes no new "
                                                                  "ones")
def set_gpus(req: GpuRequest, request: Request) -> dict:
    farm().authorize(req.authorized)
    return _admin_view(request, farm().view(admin=True))


@admin.get("/cards", access=Access.admin("farm.cards"), summary="GPUs: each card's model, memory, architecture, whether it takes jobs, what it runs and its load, which "
                                                                "extensions can / cannot / may run on it and why, which parameter tiers it lets run; every tier and the cards "
                                                                "that can run it; measured GPU memory of each node that uses a GPU; why queued GPU jobs wait and whether an "
                                                                "authorized card can run them")
def admin_cards() -> dict:
    from ..farm import cards

    return cards.view(farm())


class CardsChange(Body):
    authorized: list[str]  # the UUIDs that would take jobs


@admin.post("/cards/consequences", access=Access.admin("farm.cards"), summary="The consequences of an authorization change before making it (nothing is changed): which tiers lose or gain a "
                                                                              "card that can run them, which queued jobs would wait forever, which extensions would have no authorized card, "
                                                                              "each said as one numbered sentence")
def admin_cards_consequences(req: CardsChange) -> dict:
    from ..farm import cards

    return cards.consequences(farm(), req.authorized)


@admin.post("/jobs/{job_id}/first", access=Access.admin("queue.manage", owned=owners.job), summary="Move to front: put a queued or cooking job at the head of the queue so that GPU and CPU slots freed from now "
                                                                                                   "on go to it first; nodes already cooking are not affected. Each move is written to the admin action log (who, "
                                                                                                   "which job, its place before)")
def admin_first(job_id: str, request: Request) -> dict:
    job = farm().jobs.get(job_id)
    title, owner = (job.shown_title, job.client.who) if job is not None else (job_id, "")
    was = farm().first(job_id)
    audit(Msg("I-AUDIT-QUEUEFIRST", who=auth.actor(request).label, title=title, owner=owner, was=was),
          about=request.state.owned.client.user, session=auth.session(request), method="POST", path=str(request.url.path))
    return _admin_view(request, farm().view(admin=True))


@admin.post("/jobs/{job_id}/cancel", access=Access.admin("queue.manage", owned=owners.job), summary="Cancel anyone's job (of accounts you manage)")
def admin_cancel(job_id: str, request: Request) -> dict:
    farm().cancel(job_id, None, Msg("N-JOB-ADMINCANCELLED"))
    return {"ok": True}


# Deleting another account's tasks and renaming its groups change that account's own data: data.others (lab2shot/roles.py).
# queue.manage only looks at the queue and steers it (cancel, 插队, the switches); it never removes or rewrites what is someone's.
@admin.delete("/jobs/{job_id}", access=Access.admin("data.others"),
              summary="Delete anyone's finished job with its job folder (graph, media, output folders and zip, logs); cache only it "
                      "referenced is cleaned up afterwards")
def admin_forget(job_id: str) -> dict:
    """Takes the same path as a user's own deletion (`quota.drop_for_job`); otherwise an administrator deleting a job
    for someone would remove only the record while the space stays used. What is cleaned belongs to the job's account,
    not the administrator's."""
    from ..errors import NotFound

    try:
        row = job_row(job_id)
    except NotFound:
        row = None
    if row is not None:
        quota.drop_for_job(job_id, row)
    forget_job(job_id, None)
    return {"ok": True}


@admin.delete("/task-groups/{user_id}/{key}", access=Access.admin("data.others"),
              summary="Delete anyone's job group: every finished job in it is deleted whole (the same path as deleting one job); "
                      "queued and cooking ones stay")
def admin_forget_task_group(user_id: int, key: str) -> dict:
    return forget_group(user_id, key)


@admin.delete("/users/{user_id}/jobs", access=Access.admin("data.others"),
              summary="Delete all **finished** jobs of one account (the same path as its own Delete All); queued and cooking ones "
                      "stay (cancel them first). The answer says how many were deleted, how many skipped and how much was freed. "
                      "`lab2shot admin jobs --clear` comes here with the machine token")
def admin_forget_finished(user_id: int) -> dict:
    return forget_finished(user_id)


@admin.put("/task-groups/{user_id}/{key}/name", access=Access.admin("data.others"),
           summary="Rename anyone's job group (only the shown name; the grouping is unchanged); an empty name goes back to the "
                   "automatic one. Each rename is written to the admin action log")
def admin_rename_task_group(user_id: int, key: str, req: GroupName, request: Request) -> dict:
    from .. import accounts
    from ..transfer import groups

    name = groups.rename(user_id, key, req.name)
    audit(Msg("I-AUDIT-GROUPRENAMED", who=auth.actor(request).label, owner=accounts.get(user_id).username, name=name),
          about=user_id, session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"ok": True, "name": name}


@admin.get("/outputs", access=Access.admin("data.others"), summary="Everyone's packaged outputs (newest first): whose, which job, how big, when they go with the job; an "
                                                                   "administrator may download them for others")
def admin_outputs() -> list[dict]:
    from ..transfer import outputs

    return outputs.of_account(None)


@admin.get("/disk", access=Access.admin("data.others"), summary="Disk: how much job folders, cache and uploaded media take, how much has not been used for 7 and 30 days "
                                                                "(measured in the background: the answer is the latest measurement and when it was made; fresh=1 measures "
                                                                "again); free space on the data disk and whether it is below the free space that pauses new cooks (below it "
                                                                "everyone's new cooks pause)")
def admin_disk(fresh: bool = False) -> dict:
    from ..config import settings
    from ..farm import policy

    if fresh:
        policy.forget_space()
    # the data disk now, and whether every account's new computing pauses for it (暂停新计算的剩余空间)
    return {**farm().disk(fresh), "space": {**policy.space(), "path": str(settings().data_dir)}}


class CleanRequest(Body):
    area: str  # tasks / cache / uploads
    days: Days = 30  # remove what has not been used for longer


@admin.post("/disk/clean", access=Access.admin("data.others"), summary="Delete what has not been used for a number of days in one kind (an account with a job queued or cooking is "
                                                                       "skipped this time)")
def admin_clean(req: CleanRequest) -> dict:
    from ..farm import disk

    # the queue is asked again before each removal (Farm.removing): an account whose job comes in meanwhile keeps what
    # it has; the figures are measured again afterwards, in the background
    done = disk.clean(req.area, req.days, guard=farm().cleaner())
    logs.say(logs.get("admin"), Msg("I-DISK-ADMINCLEANED", area=req.area, days=req.days, count=done["removed"], mb=done["bytes"] / 1e6))
    farm().disk(fresh=True)
    return done


def _admin_history_key(request: Request) -> str:
    """What the admin history answer reads: whose login asks (revisions.account, its id among it: what of another
    account's it may see is this login's, access.manages) and every account's role (accounts.revision), the records."""
    from .. import accounts
    from . import revisions

    q = request.query_params
    user = decimal(q.get("user", ""))
    return sha256(repr((revisions.account(request), accounts.revision(), q.get("limit"), q.get("user"),
                        listed_version(user))))[:24]


@admin.get("/history", access=Access.admin("queue.manage", keyed=_admin_history_key, hides={"farm.cards": HISTORY_CARDS}),
           summary="Job records (newest first): every job not yet expired (at most 5000), which account, from where, when, what it "
                   "cooked and how it ended; `user`: only this account's (the admin queue's By Person). Matches /api/admin/queue's "
                   "history_version: 304 when unchanged")
def admin_history(request: Request, limit: int = HISTORY_MOST, user: int | None = None) -> list[dict]:
    # every unexpired task, as the account's own 队列 window lists them (a group is never cut in half); the most is fixed,
    # as for the paging of /log and /status: a query parameter must not be able to exhaust memory. `user` is filtered in
    # the query (a bound parameter), so one account's older jobs are not cut off by everyone's newer ones
    return [_particulars(request, e) for e in history(min(max(limit, 1), HISTORY_MOST), user_id=user)]


@admin.get("/jobs/{job_id}/graph", access=Access.admin("data.others"), summary="A job's graph as submitted")
def admin_job_graph(job_id: str) -> dict:
    return job_row(job_id)["graph"]


@admin.get("/usage", access=Access.admin("stats.view"), summary="Usage statistics: over a period, each third-party project and each of its nodes, each department and its "
                                                                "people, how many cooks each person ran, how many were reused, how long they cooked (how long on GPUs), how "
                                                                "many frames, when last used, plus the numbers per day; projects installed but never used are listed too")
def admin_usage(since: float | None = None, until: float | None = None, tz: int = 0) -> dict:
    """`since` / `until`: seconds since the epoch (none: since the last reset / until now); `tz`: the viewer's time
    zone, minutes east of UTC, for counting days."""
    from ..farm import usage

    return usage.stats(since, until, tz)  # a bad range says so itself (Invalid, E-USAGE-*)


@admin.post("/usage/reset", access=Access.admin("usage.reset"), summary="Reset usage statistics: count again from now (no job record is deleted; the old starting point is kept and can "
                                                                        "be restored)")
def admin_usage_reset() -> dict:
    from ..farm import usage

    since = usage.reset()
    logs.say(logs.get("admin"), Msg("I-USAGE-RESETBYADMIN"))
    return {"start": since}


@admin.post("/usage/reset/undo", access=Access.admin("usage.reset"), summary="Undo the last usage statistics reset: back to the old starting point")
def admin_usage_reset_undo() -> dict:
    from ..farm import usage

    since = usage.undo_reset()
    logs.say(logs.get("admin"), Msg("I-USAGE-RESETUNDONE"))
    return {"start": since}


# ------------------------------------------------------------------ models kept loaded


def _resident_view() -> dict:
    names = {g.uuid: f"GPU {g.index} · {g.short_name}" for g in farm().host.snapshot().gpus}
    view = resident().view()
    for p in view["processes"]:
        p["gpu_name"] = i18n.t("server.no_gpu") if p["gpu"] == "" else names.get(p["gpu"], p["gpu"] or "")
    return view


@admin.get("/resident", access=Access.admin("models.manage"), summary="Resident models: which projects' models are still loaded, on which GPU, how much GPU memory and RAM they take, "
                                                                      "how long idle, in GPU memory or in RAM")
def admin_resident() -> dict:
    return _resident_view()


@admin.post("/resident/{pid}/offload", access=Access.admin("models.manage"), summary="Move a resident process's model from GPU memory to RAM (moved back to the GPU quickly when next used)")
def admin_resident_offload(pid: str) -> dict:
    resident().offload(pid)
    return _resident_view()


@admin.post("/resident/{pid}/unload", access=Access.admin("models.manage"), summary="Unload a resident process completely: GPU memory and RAM are freed and it loads again when next used")
def admin_resident_unload(pid: str) -> dict:
    resident().unload(pid)
    return _resident_view()
