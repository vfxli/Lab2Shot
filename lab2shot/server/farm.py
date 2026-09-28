"""Cooking over HTTP: every cook is a job in the farm's queue (lab2shot/farm). Before submitting, the editor looks at
the cook (/api/plan: the frame range its inputs cover, what it computes, how long that should take). Users follow
their jobs by Server-Sent Events (web UI) or by polling (DCC plugins, the command line), see their own jobs and how
busy the queue is, and load a job's graph again with whether its results are still cached; the admin page (/admin,
no link from the user pages) sees every job and who started it, chooses the GPUs that take jobs, manages the models
kept loaded between jobs and sees how much each project is used."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .routes import MAX_STREAMS, Access, Router

HISTORY_MOST = 5000  # the most job records answered at once (the page limit of 后台「任务记录」; the same as the 5000 lines of logs.py /log)
from .. import library, logs
from ..database import json_of
from ..data.packet import Packet, packet_dir
from ..engine import CookError, Graph, GraphError
from ..engine.resident import pool as resident
from ..engine.templates import apply_values, pick_targets, template
from ..farm import Client, Job, farm, forget_job, history, job_row, timings
from ..farm.queue import cache_mark, listed_version
from ..errors import Conflict, Invalid, NotFound, TooManyTries
from ..io.digest import sha256
from ..messages import Msg
from . import auth, graphs, owners, quota, restart
from .wire import FRESH, NEVER
from .access import account_of, admit, audit, templates_for
from .graphs import GraphRequest
from .packets import evaluation_of

router = Router(prefix="/api", tags=["任务队列"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # this module's admin routes (app.py includes each module's)


def client_of(request: Request, declared: dict | None = None) -> Client:
    """Who is asking: the account, and what the request shows about where it comes from."""
    return Client.of(auth.me(request), auth.details(request, declared))


def submit(request: Request, data: dict, graph: Graph, targets: list[str], declared: dict, force: bool,
           version: int | None = None, show: list[str] | None = None) -> Job:
    """Queue a graph admitted for the account (server/access.py admit). It cooks in the account's own cache
    (data/store.py), where its status (server/packets.py) finds the results. The job reads the files as its submitter
    (access.account_of): a node whose upload is not this account's fails at itself, the rest cooks as usual."""
    # a full quota stops here for everything 「计算」 or 「提交」 would write into results. The page greys the buttons by
    # the same rule; this check blocks submissions that bypass the page.
    me = auth.me(request)
    quota.refuse_if_full(me.id)
    # the template it was opened from, as the usage statistics count it (按模板): read here, where the account is known
    return farm().submit(data, graph, targets, client_of(request, declared), force,
                         account=account_of(auth.session(request)), version=version, show=show,
                         template=library.opened_from(data, me.username))


class PlanRequest(GraphRequest):
    target: str
    force: bool = False


class JobRequest(GraphRequest):
    """POST /api/jobs, the one way anything queues a graph: the page, DCC plugins, scripts, the command line. The
    graph whole or by version (GraphRequest), or `template` (id or name) instead; `values` set on it
    (exposed names, or node.param); what to cook, exactly one of
      cook     a node and its upstream (a 「输出」 collects and packs what is wired into it: transfer/outputs.py)
      deliver  these 「输出」; [] every 「输出」 of the graph, like Nuke's Render All (DCC plugins, `lab2shot cook`)
    `frames` [first, last] within what the inputs have; `force` cook again what is cached; `client` what the client
    says of itself (auth.details: application, platform ...); `version` the page's cook-inputs version it submits at,
    carried back on every event of the job (farm/queue.py Job.version); `show` the outputs of the node it shows that the
    viewer displays (none: every output of it), so the rest is not computed for nothing (engine/cook.py `show`)."""

    template: str | None = None
    values: dict[str, Any] = {}
    cook: str | None = None
    deliver: list[str] | None = None
    frames: list[int] | None = None
    force: bool = False
    client: dict = {}
    version: int | None = None
    show: list[str] | None = None


@router.post("/jobs", access=Access.user("提交计算：节点图或模板，算一个节点或整理打包「输出」"),
             summary="提交计算（网页、DCC 插件、脚本、命令行都用它）：节点图（整个或按版本）或模板，values 设参数；cook 算一个节点（连同上游），"
                     "deliver 整理打包这些「输出」（[] 是全部）二选一；进队列排队，每个节点轮到了就在空着的显卡或 CPU 名额上算；返回任务 id")
def queue_job(req: JobRequest, request: Request) -> dict:
    """One task for what the request asks (JobRequest), queued like every other (farm/queue.py)."""
    if [req.cook, req.deliver].count(None) != 1:
        raise Invalid(Msg("B-JOB-WHAT"))
    if req.template:  # a template's graph says which template it is, as one the editor opens does (its task's group)
        card = template(req.template, templates_for(auth.me(request)))
        data = {**card["graph"], "meta": {**(card["graph"].get("meta") or {}), "template": card["id"]}}
    else:
        data = graphs.resolve(req, request)
    if req.values:
        data = apply_values(data, req.values)
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
        targets = graph.cook_targets(req.cook)
    return {"job": submit(request, data, graph, targets, req.client, req.force, req.version, req.show).id}


@router.post("/plan", access=Access.user("提交前：帧范围和预计用时"), summary="提交前：输入的帧范围、这次算哪些帧、哪些节点要算（其余有缓存）、按以往的用时预计要多久")
def plan(req: PlanRequest, request: Request) -> dict:
    """For scripts and DCC clients (the editor gets the shown node's with every status reply): timings.look. Reads its
    Evaluation from the same cache /api/status uses (engine/evaluations.py)."""
    return timings.look(evaluation_of(req, request), req.target, req.force, farm().models())


GONE = Msg("E-JOB-GONE")


KEEPALIVE_S = 15  # an event stream with nothing to say sends a comment this often (a proxy keeps the line open)


@router.get("/jobs/{job_id}/state", access=Access.user("查自己的任务", owned=owners.job, hides={"farm.cards": ("events[].gpus", "events[].gpu", "events[].waiting_detail")}), summary="查询自己的任务：since 之后的事件、状态（排队/计算中/完成/出错/已取消）、写出的文件")
def job_state(job_id: str, request: Request, since: int = 0) -> dict:
    return request.state.owned.poll(since)


@router.get("/jobs/{job_id}/events", access=Access.user("自己任务的实时进度", owned=owners.job, hides={"farm.cards": ("gpus", "gpu", "waiting_detail")}), summary="自己任务的事件的实时推送（Server-Sent Events）；断了浏览器自己重连，从断开的地方接着（服务重启过就从头）；账号停用或过期时马上断开")
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
    if not auth.guards().streams.take(me.id, MAX_STREAMS):  # the ones already open are never cut: only this one waits
        raise TooManyTries(Msg("E-ACCESS-TOOMANYSTREAMS", most=MAX_STREAMS))
    run, _, last = (request.headers.get("last-event-id") or last_event_id).partition(".")
    sent = int(last) + 1 if run == restart.BOOT and last.isdigit() else 0

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
                    yield f"id: {restart.BOOT}.{e['n']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                    sent = e["n"] + 1
                if batch:
                    continue
                if await run_in_threadpool(over):  # the account check reads the database: on a thread
                    return
                if not await job.changed(sent, KEEPALIVE_S):
                    yield ": keep-alive\n\n"
        finally:  # however it ends (the job finished, the line dropped, the server is closing): the slot goes back
            auth.guards().streams.give_back(me.id)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": FRESH})


@router.post("/jobs/{job_id}/cancel", access=Access.user("取消自己的任务", owned=owners.job), summary="取消自己的任务：排队的直接移出，计算中的停下")
def cancel(job_id: str, request: Request) -> dict:
    farm().cancel(job_id, None)  # whose it is: the route's owner said
    return {"ok": True}


@router.delete("/jobs/{job_id}", access=Access.user("删除自己已经结束的任务", owned=owners.job_record),
               summary="把一条**已经结束**的任务删掉：它的任务文件夹（节点图、素材、输出的文件夹和 zip、日志）整个删除，"
                       "只被它引用的缓存随后清理（别的任务还引用的一律留着）；正在算的先取消。答里带腾出多少和删完后的占用")
def forget(job_id: str, request: Request) -> dict:
    """Deleting a task removes it whole (transfer/tasks.py: its folder with its outputs, its row and references); the
    cache entries only it referenced go with the next cleaning. Measured before deletion (`drop_for_job`), the usage
    after it."""
    dropped = quota.drop_for_job(job_id, request.state.owned)
    forget_job(job_id, None)  # whose it is: the route's owner said
    return {"ok": True, **dropped, **quota.usage(int(request.state.owned["user"]))}


@router.delete("/jobs", access=Access.user("删除自己全部已经结束的任务"),
               summary="一次删掉自己**全部已经结束**的任务，连同它们占的空间（任务文件夹；只被它们引用的缓存随后清理）；"
                       "正在排队和计算的不动，也不碰别人的东西。答里带删了几条、腾出多少")
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
    live = farm().active_ids()
    ids = [str(e.get("id") or "") for e in history(limit=1000, user_id=user_id)]
    going = {jid for jid in ids if jid and jid not in live}
    jobs, freed, skipped = 0, 0, 0
    for jid in ids:
        if not jid:
            continue  # a record without a job id can be neither counted nor removed
        if jid not in going:
            skipped += 1  # queued or computing: only a cancelled job counts as finished
            continue
        try:
            row = job_row(jid)
        except (NotFound, KeyError, ValueError):
            continue
        if int(row["user"]) != user_id:  # only the account's own (history is already filtered by account; a second check)
            continue
        try:
            freed += int(quota.drop_for_job(jid, row, going).get("bytes") or 0)
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


@router.delete("/task-groups/{key}", access=Access.user("删除自己的一组任务"),
               summary="删除自己的一组任务：组里每个**已经结束**的任务都整个删掉（任务文件夹连同输出的文件夹和 zip，和删一条任务一样）；"
                       "正在排队和计算的不动（先取消）。只找得到自己的组。答里带删了几条、跳过几条、腾出多少和删完后的占用")
def forget_task_group(key: str, request: Request) -> dict:
    me = auth.me(request).id
    return {**forget_group(me, key), **quota.usage(me)}


class GroupName(BaseModel):
    name: str = Field("", max_length=1000)  # cleaned and cut to 64 characters (groups.clean_name); "" = the automatic name


@router.put("/task-groups/{key}/name", access=Access.user("给自己的一组任务改名"),
            summary="给自己的一组任务改名（只改显示的名字，分组不变；之后进这一组的任务也用这个名字）；name 为空就回到自动起的名字。"
                    "只找得到自己的组。答里是这一组现在显示的名字")
def rename_task_group(key: str, req: GroupName, request: Request) -> dict:
    from ..transfer import groups

    return {"ok": True, "name": groups.rename(auth.me(request).id, key, req.name)}


# The account's own tasks the 队列 window lists: all of them that have not expired (任务保留天数 bounds how many there
# are; they are shown by group, collapsed, so no group is ever cut in half). This is only the safety net under that,
# the most one answer carries: the same as the admin 任务记录's page (HISTORY_MOST).
HISTORY = HISTORY_MOST


@router.get("/queue", access=Access.user("队列：自己的任务；别人的只显示排在前面的有几个、大概多久", hides={
    "farm.cards": ("gpus", "switches.gpu", "jobs[].cards", "jobs[].waiting_detail")}), summary="队列：机器和接任务的显卡、计算中和排队的任务（自己的全部，别人的只有排在哪、大概多久）；"
                              "自己的任务记录的版本 history_version（变了才去 /api/queue/history 取一次）；"
                              "自己还能不能往里放东西（占了多少、上限多少、满没满；各部分的明细由队列窗口里的「我的占用」自己问一次 /api/my/storage）；"
                              "load=0：不带每秒都在变的机器负载和显卡用量（编辑器关着队列窗口时，没变化就是 304）")
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


@router.get("/queue/history", access=Access.user("队列：自己的任务记录", hides={"farm.cards": ("history[].cards",)},
                                                 keyed=_history_key),
            summary="自己所有还没过期的任务（点「计算」提交的，最多 5000 条），最新的在前，每个带它的组和结果还在不在缓存里（全在 / 部分 / 已清理）；"
                    "version 和 /api/queue 的 history_version 相同时，这份记录没变（ETag 就是它，没变是 304）")
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
        e["cache"] = cache_mark(e["id"], None, json_of(r["record"]), account_of(auth.session(request))) \
            if r and e["event"] == "finished" else None
    return {"version": version, "history": mine_}


@router.get("/load", access=Access.user("服务器现在忙不忙"), summary="服务器现在忙不忙（每个页面都能显示的一条）：队列里等着几个、正在算几个，CPU 和 GPU 计算位各用了几个、一共几个，CPU 和内存用了百分之几，以及每张显卡忙不忙、显存用了百分之几")
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


@router.get("/jobs/{job_id}", access=Access.user("加载自己的一个任务：提交时的节点图和结果还在不在缓存里", owned=owners.job_record), summary="加载自己的一个任务：提交时的整张节点图（节点、参数、连线、视图），和它的结果还在不在缓存里（全在 / 部分 / 已清理，重新算大概多久）")
def job_load(job_id: str, request: Request) -> dict:
    row = request.state.owned
    return {"id": job_id, "graph": row["graph"], "title": row["record"].get("title", ""), "submitted": row["record"].get("submitted"),
            "cache": cache_mark(job_id, row["graph"], row["record"], account_of(auth.session(request)))}


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
    ev = job.eval  # the job's own evaluation: never a second one built here
    try:
        plan = ev.plan(node)
    except (GraphError, CookError, OSError, ValueError):
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    if port not in plan.outputs:
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    d = packet_dir(plan.outputs[port])
    if not d.exists() or Packet.exists(d):  # not started, or already finished
        raise NotFound(Msg("E-JOB-NOPARTIAL", node=node, out=port))
    info = ev.info(node)
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


@router.get("/jobs/{job_id}/partial/{node}/{port}", access=Access.user("看边算边传的临时结果", owned=owners.job), summary="边算边传：一个节点还在算时已经写好的帧（算到第几帧、共几帧、尺寸和类型）；只给任务的主人，浏览器不缓存")
def partial(job_id: str, node: str, port: str, request: Request) -> Response:
    p, frames, whole = _streaming_packet(request.state.owned, node, port)
    out = {"frames_done": frames, "total": len(whole), "width": p.meta["width"], "height": p.meta["height"], "type": p.type}
    return Response(json.dumps(out), media_type="application/json", headers={"Cache-Control": NEVER})


@router.get("/jobs/{job_id}/partial/{node}/{port}/frame/{frame}.png", access=Access.user("看边算边传的某一帧", owned=owners.job), summary="边算边传：一个节点还在算时某一帧给视图看的**代理图**（和写完后的正式地址同一份代理，字节相同）；只给任务的主人，浏览器不缓存，头里带 X-Content-Sha256")
def partial_frame(job_id: str, node: str, port: str, frame: int, request: Request) -> Response:
    import mimetypes

    from ..data.payloads import is_data
    from ..data.types import channels_of
    from ..view.frames import display_frame, map_view_frame
    from ..view.proxy import picture_of, tier

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
    small = picture_of(path, p.dir / "_partial" / f"proxy.{frame}.{px}", px)
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


@router.get("/jobs/{job_id}/partial/{node}/{port}/points", access=Access.user("看边算边传的点云描述", owned=owners.job), summary="边算边传：一个节点还在算时，它已经写好的那几帧当点云看的描述（一帧一块，每块的地址此后不变）；只给任务的主人")
def partial_points(job_id: str, node: str, port: str, request: Request, camera: str | None = None) -> Response:
    from .view_data import respond_description
    from .view_worker import describe

    how, _, _whole = _partial_points(request.state.owned, node, port, camera)
    url = f"/api/jobs/{job_id}/partial/{quote(node, safe='')}/{quote(port, safe='')}/points/{{part}}" + (f"?camera={camera}" if camera else "")
    return respond_description(describe(how, url), request.headers.get("accept-encoding", ""))


@router.get("/jobs/{job_id}/partial/{node}/{port}/points/{part}", access=Access.user("看边算边传的点云数据（一帧）", owned=owners.job), summary="边算边传：还在算的那个节点某一帧的点（二进制，gzip；一帧一块）。这一帧还没写出来就是 404——浏览器绝不会把空的一帧记下来")
def partial_points_part(job_id: str, node: str, port: str, part: str, request: Request, camera: str | None = None) -> Response:
    """One frame per chunk (view_data.chunk_plan one_frame_chunks): chunk `c{k}` is frame k of this node.

    A frame not yet written answers 404, not an empty frame: a written frame's bytes never change afterwards (that is
    what `streams=True` means for this node family), so the browser is right to cache a frame it received; if it
    cached an empty one, the frame would never become visible once computed."""
    from .view_data import respond
    from .view_worker import part as view_part

    how, frames, whole = _partial_points(request.state.owned, node, port, camera)
    k = int(part[1:]) if part.startswith("c") and part[1:].isdigit() else -1
    if not 0 <= k < len(whole) or whole[k] not in frames:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=whole[k] if 0 <= k < len(whole) else -1))
    return respond(view_part(how, part), request.headers.get("accept-encoding", ""), kept=False)


# ------------------------------------------------------------------ admin


@admin.get("/queue", access=Access.admin("queue.manage"), summary="队列：所有人的任务，附带每个任务的账号和请求的全部信息（IP、浏览器、DCC 插件报告的计算机名和系统用户……）；"
                                                                  "history_version：任务记录的版本（变了才去 /api/admin/history 取一次）")
def admin_queue() -> dict:
    view = farm().view(admin=True)
    return {**view, "history_version": listed_version(None)}  # taken after the view, as for the account's own


class GpuRequest(BaseModel):
    authorized: list[str]  # UUIDs of the GPUs that take jobs


class Switches(BaseModel):
    gpu: bool | None = None  # GPU jobs (queue.gpu_jobs)
    compute: bool | None = None  # compute jobs (queue.compute_jobs)


@admin.put("/queue/switches", access=Access.admin("queue.manage"), summary="显卡任务、计算任务两个开关（设置里的 queue.gpu_jobs、queue.compute_jobs）：关掉后不再接新任务，正在算的算完；管理员和二级管理员都能开关")
def queue_switches(req: Switches) -> dict:
    from ..config import settings

    settings().save({k: v for k, v in (("queue.gpu_jobs", req.gpu), ("queue.compute_jobs", req.compute)) if v is not None})
    farm().wake()  # jobs waiting for the switch look again
    return farm().view(admin=True)


@admin.put("/gpus", access=Access.admin("gpu.authorize"), summary="授权哪些显卡接任务（按 UUID）；取消授权的显卡算完手上的任务后不再接新的")
def set_gpus(req: GpuRequest) -> dict:
    farm().authorize(req.authorized)
    return farm().view(admin=True)


@admin.get("/cards", access=Access.admin("farm.cards"), summary="显卡：每张卡的型号、显存、架构、接不接任务、在跑什么和负载，扩展包在它上面能跑 / 不能跑 / 未知和原因，它让哪些参数档位能跑；所有档位和能跑它的卡；每个用显卡的节点实测的显存；排队中的显卡任务为什么等、有没有授权的卡能跑")
def admin_cards() -> dict:
    from ..farm import cards

    return cards.view(farm())


class CardsChange(BaseModel):
    authorized: list[str]  # the UUIDs that would take jobs


@admin.post("/cards/consequences", access=Access.admin("farm.cards"), summary="改授权之前先看后果（什么都不改）：哪些档位没有卡能跑了或能跑了、哪些排队的任务会一直等、哪些扩展包没有授权的卡能跑，每条一句带编号的话")
def admin_cards_consequences(req: CardsChange) -> dict:
    from ..farm import cards

    return cards.consequences(farm(), req.authorized)


@admin.post("/jobs/{job_id}/first", access=Access.admin("queue.manage"), summary="插队：把一个排队中或计算中的任务挪到队首，之后空出来的显卡和 CPU 名额先给它；正在算的节点不受影响。每次插队记进管理操作日志（谁、哪个任务、原来第几位）")
def admin_first(job_id: str, request: Request) -> dict:
    job = farm().jobs.get(job_id)
    title, owner = (job.title, job.client.who) if job is not None else (job_id, "")
    was = farm().first(job_id)
    audit(Msg("I-AUDIT-QUEUEFIRST", who=auth.label(request), title=title, owner=owner, was=was),
          session=auth.session(request), method="POST", path=str(request.url.path))
    return farm().view(admin=True)


@admin.post("/jobs/{job_id}/cancel", access=Access.admin("queue.manage"), summary="取消任何人的任务")
def admin_cancel(job_id: str) -> dict:
    farm().cancel(job_id, None, Msg("N-JOB-ADMINCANCELLED").text)
    return {"ok": True}


# Deleting another account's tasks and renaming its groups change that account's own data: data.others (lab2shot/roles.py).
# queue.manage only looks at the queue and steers it (cancel, 插队, the switches); it never removes or rewrites what is someone's.
@admin.delete("/jobs/{job_id}", access=Access.admin("data.others"),
              summary="删除任何人已经结束的任务，连同它的任务文件夹（节点图、素材、输出的文件夹和 zip、日志）；只被它引用的缓存随后清理")
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
              summary="删除任何人的一组任务：组里每个已经结束的任务都整个删掉（和删一条任务同一条路）；正在排队和计算的不动")
def admin_forget_task_group(user_id: int, key: str) -> dict:
    return forget_group(user_id, key)


@admin.delete("/users/{user_id}/jobs", access=Access.admin("data.others"),
              summary="删除一个账号全部**已经结束**的任务（和它自己点「删除全部」同一条路）；正在排队和计算的不动（先取消）。"
                      "答里带删了几条、跳过几条、腾出多少。`lab2shot admin jobs --clear` 用本机令牌走这里")
def admin_forget_finished(user_id: int) -> dict:
    return forget_finished(user_id)


@admin.put("/task-groups/{user_id}/{key}/name", access=Access.admin("data.others"),
           summary="给任何人的一组任务改名（只改显示的名字，分组不变）；name 为空就回到自动起的名字。每次改名记进管理操作日志")
def admin_rename_task_group(user_id: int, key: str, req: GroupName, request: Request) -> dict:
    from .. import accounts
    from ..transfer import groups

    name = groups.rename(user_id, key, req.name)
    audit(Msg("I-AUDIT-GROUPRENAMED", who=auth.label(request), owner=accounts.get(user_id).username, name=name),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"ok": True, "name": name}


@admin.get("/outputs", access=Access.admin("data.others"), summary="所有人的「输出」打包好的结果（最新的在前）：谁的、哪个任务、多大、什么时候随任务删除；管理员可以替别人下载")
def admin_outputs() -> list[dict]:
    from ..transfer import outputs

    return outputs.of_account(None)


@admin.get("/disk", access=Access.admin("data.others"), summary="硬盘：任务文件夹、缓存、上传的素材各占多少，多少已经 7 天、30 天没用过")
def admin_disk() -> list[dict]:
    from ..farm import disk

    return disk.usage()


class CleanRequest(BaseModel):
    area: str  # tasks / cache / uploads
    days: float = 30  # remove what has not been used for longer


@admin.post("/disk/clean", access=Access.admin("data.others"), summary="删掉一类里多少天没用过的东西（队列里有任务时不清理）")
def admin_clean(req: CleanRequest) -> dict:
    from ..farm import disk

    if farm().busy():
        raise Conflict(Msg("E-DISK-BUSY"))
    # Measuring the disk takes minutes and jobs submitted meanwhile must not lose sources: the queue is checked again
    # before each removal (Farm.removing), and removal stops when a job arrives
    done = disk.clean(req.area, req.days, guard=farm().removing)
    logs.say(logs.get("admin"), Msg("I-DISK-ADMINCLEANED", area=req.area, days=req.days, count=done["removed"], mb=done["bytes"] / 1e6))
    return done


def _admin_history_key(request: Request) -> str:
    from . import revisions

    q = request.query_params
    user = int(q["user"]) if q.get("user", "").isdigit() else None
    return sha256(repr((revisions.account(request), q.get("limit"), q.get("user"), listed_version(user))))[:24]


@admin.get("/history", access=Access.admin("queue.manage", keyed=_admin_history_key),
           summary="任务记录（最新的在前）：每个还没过期的任务（最多 5000 条），哪个账号、从哪里、什么时候、算了什么、结果如何；"
                   "`user`：只看这个账号的（后台队列的「按人」）。和 /api/admin/queue 的 history_version 对应：没变是 304")
def admin_history(request: Request, limit: int = HISTORY_MOST, user: int | None = None) -> list[dict]:
    # every unexpired task, as the account's own 队列 window lists them (a group is never cut in half); the most is fixed,
    # as for the paging of /log and /status: a query parameter must not be able to exhaust memory. `user` is filtered in
    # the query (a bound parameter), so one account's older jobs are not cut off by everyone's newer ones
    return history(min(max(limit, 1), HISTORY_MOST), user_id=user)


@admin.get("/jobs/{job_id}/graph", access=Access.admin("data.others"), summary="一个任务提交时的节点图")
def admin_job_graph(job_id: str) -> dict:
    return job_row(job_id)["graph"]


@admin.get("/usage", access=Access.admin("stats.view"), summary="使用统计：一段时间里每个三方项目和它的每个节点、每个环节和它的人、每个人算了几次、复用几次、算了多久（显卡上多久）、多少帧、最近什么时候用，外加每天的数字；装了没用过的项目也列出")
def admin_usage(since: float | None = None, until: float | None = None, tz: int = 0) -> dict:
    """`since` / `until`: seconds since the epoch (none: since the last reset / until now); `tz`: the viewer's time
    zone, minutes east of UTC, for counting days."""
    from ..farm import usage

    return usage.stats(since, until, tz)  # a bad range says so itself (Invalid, E-USAGE-*)


@admin.post("/usage/reset", access=Access.admin("usage.reset"), summary="使用统计清零：从现在重新算起（任务记录一个不删；原来的起点留着，可以撤销）")
def admin_usage_reset() -> dict:
    from ..farm import usage

    since = usage.reset()
    logs.say(logs.get("admin"), Msg("I-USAGE-RESETBYADMIN"))
    return {"start": since}


@admin.post("/usage/reset/undo", access=Access.admin("usage.reset"), summary="撤销上一次使用统计清零：回到原来的起点")
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
        p["gpu_name"] = "不用显卡" if p["gpu"] == "" else names.get(p["gpu"], p["gpu"] or "")
    return view


@admin.get("/resident", access=Access.admin("models.manage"), summary="常驻模型：哪些项目的模型还加载着、在哪张显卡、占多少显存和内存、空闲多久、在显存里还是内存里")
def admin_resident() -> dict:
    return _resident_view()


@admin.post("/resident/{pid}/offload", access=Access.admin("models.manage"), summary="把一个常驻进程的模型从显存移到内存（下次用时很快移回显卡）")
def admin_resident_offload(pid: str) -> dict:
    resident().offload(pid)
    return _resident_view()


@admin.post("/resident/{pid}/unload", access=Access.admin("models.manage"), summary="完全卸载一个常驻进程：显存和内存都释放，下次用时重新加载")
def admin_resident_unload(pid: str) -> dict:
    resident().unload(pid)
    return _resident_view()
