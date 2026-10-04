"""HTTP routes for 提交反馈 (storage is in lab2shot/site/feedback.py).

Each page submits the user's text together with the diagnostics it collected; the server appends its own
(environment, the user's recent jobs, logs). The 用户反馈 section of the admin page lists, shows, downloads, marks
and deletes submissions."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from functools import cache

from fastapi import Request
from fastapi.responses import FileResponse, Response

from .routes import Access, Body, Limit, Router
from .. import __version__, accounts, logs, roles
from ..site import feedback
from ..config import ROOT, machine_memory_gb
from ..database import db
from ..engine.resident import available_gb
from ..engine.resident import pool as resident
from ..errors import Forbidden, NotFound
from ..messages import Msg
from ..farm import farm
from . import auth, owners, restart
from .access import audit, manages
from .available import hidden_of, strip
from .farm import client_of

admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # admin routes of this module, included by app.py
router = Router(prefix="/api", tags=["Feedback"])
log = logs.get("admin")

# Upper bound for the whole request: diagnostics, screenshots (base64 adds one third) and headroom for other fields.
MAX_REQUEST = feedback.MAX_BUNDLE + feedback.MAX_IMAGES * feedback.MAX_IMAGE * 4 // 3 + (256 << 10)


@cache
def revision() -> str:
    """Return the git revision of the running code (with a dirty marker), or 未知 when not running from a clone."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment() -> dict:
    """Return the current server environment: versions, host, GPUs, memory, queue states and resident models."""
    jobs = farm().jobs_now()
    states: dict[str, int] = {}
    for j in jobs:
        states[j.state] = states.get(j.state, 0) + 1
    return {
        "version": __version__, "revision": revision(), "python": platform.python_version(),
        "platform": platform.platform(), "hostname": platform.node(), "pid": os.getpid(), "boot": restart.BOOT,
        "started": restart.STARTED, "uptime_s": round(time.time() - restart.STARTED), "executable": sys.executable,
        "database_version": db().version, "time": time.time(),
        "memory": {"total_gb": round(machine_memory_gb(), 1), "available_gb": round(available_gb(), 1)},
        "gpus": [{**g.describe(), "takes_jobs": g.uuid in farm().authorized} for g in farm().host.snapshot().gpus],
        "queue": states,
        "resident": [{k: p.get(k) for k in ("extension", "gpu", "state", "models", "vram_mb", "ram_mb", "idle_s")}
                     for p in resident().view()["processes"]],
    }


# ------------------------------------------------------------------ users


class FeedbackIn(Body):
    text: str = ""
    category: str = ""
    client: dict = {}  # what the client says of itself (auth.details)
    diagnostics: dict = {}  # what the page collected
    images: list[dict] = []


@router.post("/feedback", access=Access.user("Submit feedback: write only, no read", limit=Limit(body=MAX_REQUEST)), summary="Submit feedback: the problem written, category, screenshots (at most 3, each under 5 MB) and diagnostics the "
                                                                                                                             "page collected; the server adds its own environment and the user's recent jobs and logs")
def submit(req: FeedbackIn, request: Request) -> dict:
    client = client_of(request, req.client)
    row = feedback.submit(req.text, req.category, client.user, client.full(), req.diagnostics, req.images, environment())
    return {"id": row["id"], "at": row["at"]}


class JobsRequest(Body):
    jobs: list[str] = []


@router.post("/feedback/jobs", access=Access.user("Before submitting feedback: your recent jobs", owned=owners.own_jobs, body_ids=("jobs",)), summary="Before submitting feedback: your recent jobs the server will attach (state, errors), without the full logs")
def preview_jobs(req: JobsRequest, request: Request) -> dict:
    return {"jobs": [{k: j[k] for k in ("id", "title", "state", "submitted", "error")}
                     | {"error_log": bool(j["error_log"]), "server_log": len(j["server_log"])}
                     for j in feedback.jobs_of(auth.me(request).id, request.state.owned)],
            "environment": {k: v for k, v in environment().items() if k in ("version", "revision", "python", "platform")}}


@router.get("/feedback/mine", access=Access.user("My feedback: only your own, with administrator replies"), summary="My feedback: the feedback you submitted (state, administrator replies) and how many new replies or state "
                                                                                                                    "changes are unread; never diagnostics, internal notes or anyone else's")
def mine(request: Request) -> dict:
    return feedback.mine(auth.me(request).id)


@router.post("/feedback/mine/read", access=Access.user("My feedback: read"), summary="My feedback read: replies and state changes no longer count as unread")
def mine_read(request: Request) -> dict:
    return feedback.mark_read(auth.me(request).id)


# ------------------------------------------------------------------ admin


@admin.get("/feedback", access=Access.admin("feedback.reply"), summary="User feedback, newest first, filtered by state (new/seen/solved), rating (unrated / valid / invalid), period, "
                                                                       "account name; with the count of each state and each rating")
def listing(request: Request, status: str = "", since: float | None = None, until: float | None = None, person: str = "",
            rating: str = "") -> dict:
    s = auth.session(request)  # only the feedback of accounts this login manages, counted as listed
    return feedback.listing(status or None, since, until, person, seen=lambda owner: manages(s, owner),
                            rating=None if not rating else "" if rating == "unrated" else rating)


# what a feedback's diagnostics say of the whole server, not of its sender (lab2shot/site/feedback.py submit): each only for
# whoever may see it there (available.py FIELDS), in its detail and in its download alike
DIAGNOSTICS = {"logs.view": ("bundle.server.server_log",), "farm.cards": ("bundle.server.gpus",),
               "models.manage": ("bundle.server.resident",)}


@admin.get("/feedback/{fid}", access=Access.admin("feedback.reply", owned=owners.feedback, hides=DIAGNOSTICS), summary="All of one feedback: the problem written, screenshots, diagnostics (graph, logs, errors, jobs and their error "
                                                                                                                       "logs, environment)")
def detail(fid: str, request: Request) -> dict:
    return feedback.detail(fid)


@admin.get("/feedback/{fid}/files/{name}", access=Access.admin("feedback.reply", owned=owners.feedback), summary="A screenshot attached to feedback")
def image(fid: str, name: str, request: Request) -> FileResponse:
    if name not in request.state.owned["images"]:
        raise NotFound(Msg("E-FEEDBACK-NOSHOT"))
    return FileResponse(feedback.folder(fid) / name)


@admin.get("/feedback/{fid}/download", access=Access.admin("feedback.reply", owned=owners.feedback), summary="Download one feedback whole as a zip: feedback.json (problem, who, state, diagnostics) and screenshots")
def download(fid: str, request: Request) -> Response:
    row = request.state.owned
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(row["at"]))
    paths = hidden_of(auth.session(request), DIAGNOSTICS)

    def trim(row: dict) -> dict:
        for path in paths:
            strip(row, path)
        return row

    return Response(feedback.archive(fid, trim), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="feedback-{stamp}-{fid}.zip"'})


class AnswerRequest(Body):
    status: str
    reply: str = ""  # visible to the submitting user
    note: str = ""  # visible to administrators only


@admin.put("/feedback/{fid}", access=Access.admin("feedback.reply", owned=owners.feedback), summary="Answer one feedback: state (new / seen / solved), the reply to the user (seen in My Feedback) and internal "
                                                                                                    "notes only administrators see")
def answer(fid: str, req: AnswerRequest, request: Request) -> dict:
    return feedback.answer(fid, req.status, req.reply, req.note, auth.actor(request))


class RateRequest(Body):
    rating: str  # "" 未评定 / valid 有效 / invalid 无效


@admin.put("/feedback/{fid}/rating", access=Access.admin("feedback.reply", owned=owners.feedback), summary="Rate one feedback: unrated / valid / invalid (apart from its state). Valid counts toward the sending account's "
                                                                                                           "valid feedback reward (every set number of them extends the account's expiry by the set number of days); "
                                                                                                           "changing valid to something else takes that reward back by the ledger; written to the admin action log")
def rate(fid: str, req: RateRequest, request: Request) -> dict:
    s = auth.session(request)
    done = feedback.rate(fid, req.rating, auth.actor(request))
    row = done["feedback"]
    audit(Msg("I-AUDIT-FEEDBACKRATED", who=s.user.label, role=roles.word(s.user.role), username=row["username"] or row["person"],
              id=fid, rating=row["rating_label"], said=done["said"]),
          about=row["user"], session=s, method="PUT", path=str(request.url.path))
    return done


@admin.get("/users/{user_id}/rewards", access=Access.admin("feedback.reply"), summary="One account's valid feedback reward: how many valid feedbacks, how many days the standing rewards added in "
                                                                                      "all, how many more to the next one, and the ledger of each reward and revocation")
def rewards(user_id: int, request: Request) -> dict:
    u = accounts.get(user_id)
    s = auth.session(request)
    if not manages(s, u.id, u.role):
        raise Forbidden(Msg("E-ROLES-NOTYOURS", role=roles.label(s.user.role), username=u.username, target=roles.label(u.role)))
    return feedback.rewards_of(user_id)


@admin.delete("/feedback/{fid}", access=Access.admin("feedback.delete", owned=owners.feedback), summary="Delete one feedback (with its screenshots and diagnostics; copies in database backups stay until the backups "
                                                                                                        "rotate out)")
def delete(fid: str, request: Request) -> dict:
    feedback.delete(fid, auth.actor(request))
    return {"ok": True}
