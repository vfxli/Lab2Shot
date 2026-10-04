"""User feedback (提交反馈): what a user writes about a problem, with what it takes to find it out — the diagnostics
the page collects (version, browser, the graph, the page's log, its errors and failed requests), what the server adds
for that user's recent jobs (their state, the failed node's log, the server's log lines about them) and the machine
(GPUs, memory, extensions), and up to MAX_IMAGES screenshots.

    database, table feedback            one row per feedback: when, who, category, text, status, the admin's reply
                                        (the user sees it) and note (the user does not)
    work/feedback/<id>/bundle.json      the diagnostics, as the admin page shows and downloads them
    work/feedback/<id>/shot-1.png ...   the screenshots

Users see their own feedback (mine): what they wrote, its status and the administrator's reply, never the
diagnostics (the server's log has other people's lines) nor the note, nor anyone else's feedback. Theirs is what their
account sent. A change of status or a new reply is unread until they look (changed, read_at). A deleted account's
feedback stays for the administrator.

Each feedback also has a rating (评定: 未评定 / 有效 / 无效), apart from its status: an administrator who may answer
feedback rates it (rate). Valid feedback earns its account time (有效反馈奖励, settings feedback.reward_count N and
feedback.reward_days M): every N valid ones, counted once each, move the account's expiry M days on (from now when
it has passed; an account that never expires is only recorded). The ledger (table feedback_rewards) has a row per grant
(the feedback it is for, its days, the expiry before and after) and per revoke: a valid feedback rated otherwise again
takes back the grant it was in, by that grant's own days (never to before now), and the other feedback of that grant
counts again. A grant keeps the N and M it was made with: changing the settings never changes what was given.

Everything is bounded (MAX_* below) and scrubbed of secrets before it is kept: tokens (Hugging Face, API keys,
bearer headers, passwords), and the value of any environment variable of the server whose name says it is one. Kept
until the administrator deletes it; the database's backups keep the rows and the files (database/FILED).
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from .. import i18n, logs
from ..config import settings
from ..database import LIKE_ESCAPE, contains, db, json_of, json_text
from ..errors import Invalid, NotFound, TooMany
from ..messages import Msg
from ..periods import Periods, local_day

if TYPE_CHECKING:
    from ..accounts import Actor

log = logs.get("feedback")

MAX_TEXT = 10_000  # characters the user writes
MAX_BUNDLE = 2 << 20  # bytes of the page's diagnostics (as JSON)
MAX_IMAGES = 3
MAX_IMAGE = 5 << 20  # bytes of one screenshot
LOG_LINES = 200  # of a node's error log, of the server's log per job, of the server's log at the end
JOBS = 20  # recent jobs of the user attached
# One feedback can take ~20 MB of disk (screenshots, the diagnostics, the logs attached): one account may send DAY_MAX
# within a day, and have OPEN_MAX not yet solved at once; the administrator solving or deleting them makes room.
DAY_MAX = 20
OPEN_MAX = 50

# the vocabularies (their words: feedback.category.<id>, feedback.status.<id>, feedback.rating.<id>; "" is "none")
CATEGORIES = ("", "error", "usage", "idea")
STATUS = ("new", "seen", "solved")
RATINGS = ("", "valid", "invalid")  # the rating, apart from the status (rate)


def _word(kind: str, value: str) -> str:
    return i18n.t(f"feedback.{kind}.{value or 'none'}")
_IMAGES = {b"\x89PNG\r\n\x1a\n": "png", b"\xff\xd8\xff": "jpg", b"RIFF": "webp", b"GIF8": "gif"}
_ID = re.compile(r"^[0-9a-f]{12}$")


# ------------------------------------------------------------------ secrets out


_SECRETS = [
    (re.compile(r"hf_[A-Za-z0-9]{16,}"), "hf_***"),
    (re.compile(r"\b(sk|pk|rk)-[A-Za-z0-9_-]{16,}"), r"\1-***"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "gh*_***"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), "xox*-***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"), "Bearer ***"),
    (re.compile(r"(?i)\b(authorization|password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)"
                r"(\"?\s*[:=]\s*\"?|\s+)([^\s\"',;&]{4,})"), r"\1\2***"),
    (re.compile(r"(?i)([?&](?:token|key|secret|password|sig|signature)=)[^&\s\"']+"), r"\1***"),
]
_SECRET_ENV = re.compile(r"(?i)(token|secret|password|passwd|api_?key|access_?key|private_?key|credential)")


def _env_secrets() -> list[str]:
    """This server's secret values (by the names of the environment variables they are in): never in feedback."""
    return sorted({v for k, v in os.environ.items() if _SECRET_ENV.search(k) and len(v) >= 6}, key=len, reverse=True)


def scrub(value, secrets: list[str] | None = None):
    """`value` (text, or lists and dicts of it) with every secret replaced by ***."""
    secrets = _env_secrets() if secrets is None else secrets
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, "***")
        for pattern, repl in _SECRETS:
            value = pattern.sub(repl, value)
        return value
    if isinstance(value, dict):
        return {k: "***" if isinstance(k, str) and isinstance(v, str) and v and _SECRET_ENV.search(k) else scrub(v, secrets)
                for k, v in value.items()}  # {"password": ...}: whatever it is
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    return value


# ------------------------------------------------------------------ what the server adds


def _tail(path: Path, lines: int = LOG_LINES) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - (256 << 10)))  # the end is what matters
            return f.read().decode("utf-8", "replace").splitlines()[-lines:]
    except OSError:
        return []


def _inside_work(path: str | None) -> Path | None:
    """A log path from a job's record, when it is inside the work folder (never anything else of the machine)."""
    if not path:
        return None
    p = Path(path).resolve()
    work = settings().work_dir.resolve()
    return p if work in p.parents and p.is_file() else None


def _log_about(job_ids: list[str], lines: list[str]) -> dict[str, list[str]]:
    """The server log's lines about each job: those naming it, with the traceback lines that follow them."""
    out: dict[str, list[str]] = {j: [] for j in job_ids}
    current: list[str] = []
    for line in lines:
        if re.match(r"^\d{4}-\d\d-\d\d ", line):
            current = [j for j in job_ids if j in line]
        for j in current:
            out[j].append(line)
    return {j: found[-LOG_LINES:] for j, found in out.items()}


def jobs_of(user_id: int, asked: list[str]) -> list[dict]:
    """The jobs a feedback is about, this account's only: the ones the page names (up to JOBS) and its latest, newest
    first, with what became of them, the failed node's log and the server's log lines about them."""
    from ..farm import farm

    asked = asked[:JOBS]
    found = db().rows(f"SELECT id, record FROM jobs WHERE user_id = ? AND id IN ({','.join('?' * len(asked))})",
                      (user_id, *asked)) if asked else []
    found += db().rows("SELECT id, record FROM jobs WHERE user_id = ? ORDER BY submitted DESC LIMIT ?", (user_id, JOBS))
    live = {j.id: j for j in list(farm().jobs.values())}  # a job still in the queue: as it is now
    rows = {r["id"]: live[r["id"]].record() if r["id"] in live else json_of(r["record"]) for r in found}
    ordered = sorted(rows.values(), key=lambda r: -(r.get("submitted") or 0))[:JOBS]
    about = _log_about([r["id"] for r in ordered], logs.tail(20_000))
    out = []
    for r in ordered:
        error_log = _inside_work(r.get("error_log"))
        out.append({k: r.get(k) for k in ("id", "title", "targets", "state", "frames", "submitted", "started", "finished",
                                          "cards", "error", "reason")}
                   | {"error_log": {"file": str(error_log), "lines": _tail(error_log)} if error_log else None,
                      "server_log": about[r["id"]]})
    return out


def extension_states(graph: dict, named: list = ()) -> list[dict]:
    """Each node type of the graph, and each extension the page names (the help page of a project): whether its
    project is installed and ready (and why not)."""
    from ..extensions import extensions
    from ..extensions.status import extension_status
    from ..nodes import node_types

    types, exts, out, seen = node_types(), extensions(), [], set()
    for name in named:
        ext = exts.get(name) if isinstance(name, str) else None
        if ext:
            st = extension_status(ext)
            out.append({"type": "", "extension": name, "installed": st["installed"], "ready": st["ready"], "reason": st["reason"]})
    for n in graph.get("nodes") or []:
        tid = n.get("type") if isinstance(n, dict) else None
        if not isinstance(tid, str) or tid in seen:
            continue
        seen.add(tid)
        t = types.get(tid)
        runtime = t.runtime if t else tid.split(".")[0]
        if runtime == "core" and t:
            out.append({"type": tid, "extension": "core", "ready": True, "reason": ""})
            continue
        ext = exts.get(runtime)
        st = extension_status(ext) if ext else {"installed": False, "ready": False, "reason": Msg("E-FEEDBACK-NOEXTENSION").text}
        out.append({"type": tid, "extension": runtime, "installed": st.get("installed"), "ready": st.get("ready"),
                    "reason": st.get("reason", "")})
    return out


# ------------------------------------------------------------------ keeping it


def folder(fid: str) -> Path:
    if not _ID.match(fid):
        raise NotFound(Msg("E-FEEDBACK-NOTFOUND"))
    return settings().work_dir / "feedback" / fid


def forget_in_bundles(u) -> None:
    """Permanent deletion of an account (account_removal.purge): no feedback's bundle names it any more. A bundle holds
    what the server's log said when it was sent (submit), whoever sent it, and a line of that log naming this account
    goes from all of them (accounts.marked: its username, its label, where it logged in from); the account's own
    feedback no longer says it sent it."""
    from ..accounts import kept_deleted, marked, unmarked
    from ..io.atomic import write_text

    pattern = marked(u)
    for row in db().rows("SELECT id, user_id FROM feedback"):
        bundle = folder(row["id"]) / "bundle.json"
        try:
            kept = json.loads(bundle.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        cleaned = unmarked(kept, pattern)
        if row["user_id"] == u.id and isinstance(cleaned.get("server"), dict):
            cleaned["server"]["client"] = {"who": kept_deleted()}
        if cleaned != kept:
            write_text(bundle, json.dumps(cleaned, ensure_ascii=False, indent=1))


def _image(item: dict, i: int) -> tuple[str, bytes]:
    data = str(item.get("data") or "")
    data = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise Invalid(Msg("E-FEEDBACK-IMAGEUNREADABLE", index=i)) from None
    if len(raw) > MAX_IMAGE:
        raise Invalid(Msg("E-FEEDBACK-IMAGEBIG", index=i, size=len(raw) / 2**20, max=MAX_IMAGE >> 20))
    kind = next((k for magic, k in _IMAGES.items() if raw.startswith(magic)), None)
    if kind is None or (kind == "webp" and raw[8:12] != b"WEBP"):
        raise Invalid(Msg("E-FEEDBACK-NOTIMAGE", index=i))
    return f"shot-{i}.{kind}", raw


def submit(text: str, category: str, user_id: int, client: dict, page: dict, images: list[dict], server: dict) -> dict:
    """Keep a feedback from an account. `client`: the request's account and details (farm/clients.py); `page`: what
    the page collected; `server`: what the server knows now (environment); the jobs and logs are added here. Raises
    Invalid (a message for the user) when something does not fit."""
    text = (text or "").strip()
    if not text:
        raise Invalid(Msg("E-FEEDBACK-NOTEXT"))
    if len(text) > MAX_TEXT:
        raise Invalid(Msg("E-FEEDBACK-TEXTLONG", count=len(text), max=MAX_TEXT))
    if category not in CATEGORIES:
        raise Invalid(Msg("E-FEEDBACK-CATEGORY", category=category))
    if len(images) > MAX_IMAGES:
        raise Invalid(Msg("E-FEEDBACK-IMAGES", max=MAX_IMAGES))
    if db().row("SELECT COUNT(*) AS n FROM feedback WHERE user_id = ? AND at >= ?", (user_id, time.time() - 86400))["n"] >= DAY_MAX:
        raise TooMany(Msg("E-FEEDBACK-TOOOFTEN", max=DAY_MAX))
    if db().row("SELECT COUNT(*) AS n FROM feedback WHERE user_id = ? AND status != 'solved'", (user_id,))["n"] >= OPEN_MAX:
        raise TooMany(Msg("E-FEEDBACK-TOOMANYOPEN", max=OPEN_MAX))
    shots = [_image(item, i) for i, item in enumerate(images, 1)]
    if len(json.dumps(page, ensure_ascii=False).encode("utf-8")) > MAX_BUNDLE:
        raise Invalid(Msg("E-FEEDBACK-BUNDLEBIG", max=MAX_BUNDLE >> 20))
    asked = [str(j.get("id")) for j in page.get("jobs") or [] if isinstance(j, dict) and j.get("id")]
    bundle = scrub({"page": page, "server": {**server, "client": client, "jobs": jobs_of(user_id, asked),
                                             "extensions": extension_states(page.get("graph") or {}, page.get("extensions") or []),
                                             "server_log": logs.tail(LOG_LINES)}})
    text = scrub(text)

    fid = uuid.uuid4().hex[:12]
    base = folder(fid)
    base.mkdir(parents=True)
    files = []
    for name, raw in [("bundle.json", json.dumps(bundle, ensure_ascii=False, indent=1).encode("utf-8")), *shots]:
        (base / name).write_bytes(raw)
        files.append({"name": name, "bytes": len(raw), "kind": "bundle" if name == "bundle.json" else "image"})
    now = time.time()
    with db().write() as c:  # the row last: a feedback is there once its files are
        c.execute("INSERT INTO feedback (id, at, user_id, category, text, status, note, updated, "
                  "updated_by, files, bytes, read_at) VALUES (?, ?, ?, ?, ?, 'new', '', NULL, '', ?, ?, ?)",
                  (fid, now, user_id, category, text, json_text(files), sum(f["bytes"] for f in files), now))
    logs.say(log, Msg("I-FEEDBACK-RECEIVED", id=fid, who=client.get("who", ""), first=text.splitlines()[0][:60]))
    return get(fid)


ROWS = ("SELECT f.*, u.username, u.name AS person, u.department, u.deleted AS gone FROM feedback f "
        "LEFT JOIN users u ON u.id = f.user_id")


def _row(r) -> dict:
    from ..accounts import deleted_label, department_label

    files = json_of(r["files"])
    return {"id": r["id"], "at": r["at"], "user": r["user_id"], "username": r["username"] or "",
            "person": deleted_label() if r["gone"] else r["person"] or "", "department": department_label(r["department"] or ""),  # its words (display only)
            "category": r["category"], "category_label": _word("category", r["category"]) if r["category"] in CATEGORIES else r["category"], "text": r["text"],
            "title": r["text"].strip().splitlines()[0][:120] if r["text"].strip() else "", "status": r["status"],
            "status_label": _word("status", r["status"]) if r["status"] in STATUS else r["status"], "reply": r["reply"], "replied": r["replied"], "replied_by": r["replied_by"],
            "note": r["note"], "updated": r["updated"], "updated_by": r["updated_by"], "files": files,
            "images": [f["name"] for f in files if f["kind"] == "image"], "bytes": r["bytes"], "changed": r["changed"],
            "unread": _unread(r), "rating": r["rating"], "rating_label": _word("rating", r["rating"]) if r["rating"] in RATINGS else r["rating"],
            "rated": r["rated"], "rated_by": r["rated_by"], "reward": r["reward"]}


def _unread(r) -> bool:
    """The user has not seen its latest status or reply yet."""
    return r["changed"] is not None and r["changed"] > (r["read_at"] or 0)


def get(fid: str) -> dict:
    folder(fid)
    r = db().row(f"{ROWS} WHERE f.id = ?", (fid,))
    if r is None:
        raise NotFound(Msg("E-FEEDBACK-GONE"))
    return _row(r)


def listing(status: str | None = None, since: float | None = None, until: float | None = None, person: str = "",
            seen=lambda owner: True, rating: str | None = None) -> dict:
    """Feedback, newest first, by status, rating, time and part of an account's name or username; and how many there
    are of each status and of each rating. `seen(owner)`: whether whoever asks may see a sender's feedback at all
    (server/access.py manages): what it may not see is neither listed nor counted."""
    where, args = [], []
    if status:
        if status not in STATUS:
            raise Invalid(Msg("E-FEEDBACK-STATUS", status=status))
        where.append("status = ?")
        args.append(status)
    if rating is not None:
        if rating not in RATINGS:
            raise Invalid(Msg("E-FEEDBACK-RATING", rating=rating))
        where.append("f.rating = ?")
        args.append(rating)
    if since is not None:
        where.append("at >= ?")
        args.append(since)
    if until is not None:
        where.append("at < ?")
        args.append(until)
    if person.strip():
        where.append(f"(u.name LIKE ? ESCAPE '{LIKE_ESCAPE}' OR u.username LIKE ? ESCAPE '{LIKE_ESCAPE}')")
        args += [contains(person)] * 2
    rows = [r for r in db().rows(f"{ROWS} {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY f.at DESC", tuple(args))
            if seen(r["user_id"])]
    counts, ratings = {s: 0 for s in STATUS}, {k: 0 for k in RATINGS}
    for r in db().rows("SELECT status, rating, user_id FROM feedback"):
        if seen(r["user_id"]):
            counts[r["status"]] += 1
            ratings[r["rating"]] = ratings.get(r["rating"], 0) + 1
    return {"items": [_row(r) for r in rows], "counts": counts, "ratings": ratings}


def new_count() -> int:
    return db().row("SELECT COUNT(*) AS n FROM feedback WHERE status = 'new'")["n"]


def counts_in(p: Periods) -> dict:
    """The admin overview's 反馈: how many are not solved yet (新 and 已看) and of them not yet looked at (新), and how
    many came in 今日, 近 7 天 and 本月 (every feedback sent then, whatever its status now; a deleted one no more)."""
    rows = db().rows(f"SELECT {local_day('at')} AS day, COUNT(*) AS n FROM feedback WHERE at >= ? GROUP BY 1", (p.since,))
    open_ = db().row("SELECT COUNT(*) AS n, COALESCE(SUM(status = 'new'), 0) AS new FROM feedback WHERE status != 'solved'")
    came = p.sums(((r["day"], r["n"]) for r in rows), ("today", "days7", "month"))
    return {"open": open_["n"], "new": open_["new"], **{k: {"came": n} for k, n in came.items()}}


def detail(fid: str) -> dict:
    """A feedback with its diagnostics."""
    row = get(fid)
    try:
        bundle = json.loads((folder(fid) / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bundle = None
    return {**row, "bundle": bundle}


def answer(fid: str, status: str, reply: str, note: str, by: Actor) -> dict:
    """The administrator's answer: the status and the reply (the user sees both: unread until they look), the note
    (only administrators see it)."""
    old = get(fid)
    if status not in STATUS:
        raise Invalid(Msg("E-FEEDBACK-STATUS", status=status))
    reply, note = reply.strip(), note.strip()
    for label, text in ((i18n.t("feedback.reply"), reply), (i18n.t("feedback.note"), note)):
        if len(text) > MAX_TEXT:
            raise Invalid(Msg("E-FEEDBACK-ANSWERLONG", what=label, count=len(text), max=MAX_TEXT))
    now = time.time()
    replied = ((now, by.label, by.id) if reply else (None, "", None)) if reply != old["reply"] else (
        old["replied"], old["replied_by"], old["replied_by_id"])
    changed = now if status != old["status"] or reply != old["reply"] else old["changed"]
    with db().write() as c:
        c.execute("UPDATE feedback SET status = ?, reply = ?, replied = ?, replied_by = ?, replied_by_id = ?, note = ?, "
                  "changed = ?, updated = ?, updated_by = ?, updated_by_id = ? WHERE id = ?",
                  (status, reply, *replied, note, changed, now, by.label, by.id, fid))
    return get(fid)


# ------------------------------------------------------------------ rating, and the reward of time


def reward_rule() -> tuple[int, int]:
    """The setting now: every N valid feedback earn M days (feedback.reward_count, feedback.reward_days)."""
    return int(settings()["feedback.reward_count"]), int(settings()["feedback.reward_days"])


def _day(t: float | None) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(t)) if t else ""


def _pending(user_id: int) -> list:
    """An account's valid feedback not yet in a grant, the earliest rated first."""
    return db().rows("SELECT id FROM feedback WHERE user_id = ? AND rating = 'valid' AND reward IS NULL "
                     "ORDER BY rated, at, id", (user_id,))


def _grant(c, user_id: int, ids: list[str], per: int, days: int, by: Actor, t: float) -> dict:
    from ..accounts import shift_expiry

    moved = shift_expiry(c, user_id, days)
    gid = c.execute("INSERT INTO feedback_rewards (at, user_id, kind, feedback, per, days, old_expires, new_expires, "
                    "applied, by, by_id) VALUES (?, ?, 'grant', ?, ?, ?, ?, ?, ?, ?, ?)",
                    (t, user_id, json_text(ids), per, days, moved.old, moved.new, int(moved.applied), by.label, by.id)).lastrowid
    # what the user sees changed: 我的反馈 marks them
    c.execute(f"UPDATE feedback SET reward = ?, changed = ? WHERE id IN ({','.join('?' * len(ids))})", (gid, t, *ids))
    return {"kind": "grant", "id": gid, "feedback": ids, "days": days, "old": moved.old, "new": moved.new, "why": moved.why}


def _revoke(c, gid: int, by: Actor, t: float) -> dict:
    """Take back a grant by its own days (never to before now); its other feedback count again."""
    from ..accounts import shift_expiry

    g = db().row("SELECT * FROM feedback_rewards WHERE id = ?", (gid,))
    moved = shift_expiry(c, g["user_id"], -g["days"] if g["applied"] else 0)
    taken = round((moved.old - moved.new) / 86400, 4) if moved.applied else 0
    rid = c.execute("INSERT INTO feedback_rewards (at, user_id, kind, feedback, per, days, old_expires, new_expires, "
                    "applied, undoes, by, by_id) VALUES (?, ?, 'revoke', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (t, g["user_id"], g["feedback"], g["per"], taken, moved.old, moved.new, int(moved.applied), gid,
                     by.label, by.id)).lastrowid
    c.execute("UPDATE feedback_rewards SET undone = ? WHERE id = ?", (rid, gid))
    c.execute("UPDATE feedback SET reward = NULL WHERE reward = ?", (gid,))
    return {"kind": "revoke", "id": rid, "feedback": json_of(g["feedback"]), "days": taken, "old": moved.old,
            "new": moved.new, "why": moved.why, "granted": g["days"]}


def _settle(c, user_id: int, by: Actor, t: float) -> list[dict]:
    """Grant what an account's valid feedback not yet granted earn now: every N of them, M days (the setting now)."""
    per, days = reward_rule()
    pending = [r["id"] for r in _pending(user_id)]
    done = []
    while len(pending) >= per:
        done.append(_grant(c, user_id, pending[:per], per, days, by, t))
        pending = pending[per:]
    return done


def rate(fid: str, rating: str, by: Actor) -> dict:
    """Rate a feedback (未评定 / 有效 / 无效), and give or take back the time it earns (module doc): the feedback as it
    is now, what happened in the ledger (`events`) and what that is in words for the administrator (`said`). Rating it
    as it already is changes nothing."""
    if rating not in RATINGS:
        raise Invalid(Msg("E-FEEDBACK-RATING", rating=rating))
    folder(fid)
    t = time.time()
    events: list[dict] = []
    with db().write() as c:
        r = db().row("SELECT rating, reward, user_id, changed FROM feedback WHERE id = ?", (fid,))
        if r is None:
            raise NotFound(Msg("E-FEEDBACK-GONE"))
        was = r["rating"]
        if was != rating:
            if r["reward"] is not None:
                events.append(_revoke(c, r["reward"], by, t))
            c.execute("UPDATE feedback SET rating = ?, rated = ?, rated_by = ?, rated_by_id = ?, reward = NULL, "
                      "changed = ? WHERE id = ?",
                      (rating, t, by.label, by.id, t if rating == "valid" else r["changed"], fid))
            if r["user_id"] is not None and (rating == "valid" or events):
                events += _settle(c, r["user_id"], by, t)
    row = get(fid)
    return {"feedback": row, "events": events, "said": _said(row, was, events)}


def _said(row: dict, was: str, events: list[dict]) -> str:
    """What a rating did, in words for the administrator (and the audit log)."""
    from ..accounts import deleted_label

    person = row["person"] or row["username"] or deleted_label()
    if was == row["rating"]:
        return Msg("I-FEEDBACK-RATEDSAME", rating=row["rating_label"]).text
    parts = [Msg("I-FEEDBACK-RATED", rating=row["rating_label"]).text]
    for e in events:
        if e["kind"] == "revoke":
            parts.append(Msg("I-FEEDBACK-REVOKED", person=person, days=_days(e["days"]), until=_day(e["new"])).text
                         if e["why"] == "" else Msg("I-FEEDBACK-REVOKEDNOTE", granted=_days(e["granted"])).text)
        else:
            if e["why"] == "":
                parts.append(Msg("I-FEEDBACK-GRANTED", person=person, count=len(e["feedback"]), days=_days(e["days"]),
                                 until=_day(e["new"])).text)
            elif e["why"] == "never":
                parts.append(Msg("I-FEEDBACK-GRANTEDNEVER", person=person, count=len(e["feedback"]), days=_days(e["days"])).text)
            elif e["why"] == "none":
                parts.append(Msg("I-FEEDBACK-GRANTEDNONE", count=len(e["feedback"])).text)
            else:
                parts.append(Msg("I-FEEDBACK-GRANTEDGONE", count=len(e["feedback"]), days=_days(e["days"])).text)
    if row["rating"] == "valid" and row["reward"] is None and row["user"] is not None:
        per, days = reward_rule()
        have = len(_pending(row["user"]))
        parts.append(Msg("I-FEEDBACK-COUNTED", person=person, have=have, per=per, days=days).text if days
                     else Msg("I-FEEDBACK-NOREWARD").text)
    elif was == "valid" and not events:
        parts.append(Msg("I-FEEDBACK-UNCOUNTED").text)
    return i18n.t("feedback.joiner").join(parts)


def _days(d: float) -> str:
    return f"{d:g}"


def _reward_texts(rows: list[dict]) -> dict[str, str]:
    """What the user sees of a valid feedback's reward (我的反馈): the grant it is in, or how many more it waits for.
    An invalid or unrated one says nothing (rating it invalid never troubles the user)."""
    valid = [f for f in rows if f["rating"] == "valid"]
    if not valid:
        return {}
    grants = {g["id"]: g for g in db().rows(
        f"SELECT * FROM feedback_rewards WHERE id IN ({','.join('?' * len(valid))})", tuple(f["reward"] for f in valid))}
    per, days = reward_rule()
    pending: dict[int, int] = {}
    out = {}
    for f in valid:
        g = grants.get(f["reward"])
        if g is not None:
            others = len(json_of(g["feedback"])) - 1
            if g["applied"]:
                said = dict(days=_days(g["days"]), until=_day(g["new_expires"]))
                out[f["id"]] = (Msg("I-FEEDBACK-MINEGRANTEDWITH", others=others, **said) if others
                                else Msg("I-FEEDBACK-MINEGRANTED", **said)).text
            elif g["old_expires"] is None and g["days"]:
                out[f["id"]] = Msg("I-FEEDBACK-MINENEVER").text
            else:
                out[f["id"]] = Msg("I-FEEDBACK-MINEVALID").text
        elif f["user"] is not None:
            if f["user"] not in pending:
                pending[f["user"]] = len(_pending(f["user"]))
            left = per - pending[f["user"]]
            out[f["id"]] = (Msg("I-FEEDBACK-MINEPENDING", left=left, days=days).text if days and left > 0
                            else Msg("I-FEEDBACK-MINEVALID").text)
    return out


def rewards_of(user_id: int) -> dict:
    """An account's valid feedback and the time it earned (后台「用户」详情): how many are valid, the days its grants
    still standing gave, how many wait for the next grant, the rule now, and the ledger, newest first."""
    valid = db().row("SELECT COUNT(*) AS n FROM feedback WHERE user_id = ? AND rating = 'valid'", (user_id,))["n"]
    ledger = db().rows("SELECT * FROM feedback_rewards WHERE user_id = ? ORDER BY at DESC, id DESC", (user_id,))
    per, days = reward_rule()
    standing = [g for g in ledger if g["kind"] == "grant" and g["undone"] is None]
    return {"valid": valid, "days": sum(g["days"] for g in standing if g["applied"]),
            "recorded": sum(g["days"] for g in standing if not g["applied"]), "pending": len(_pending(user_id)),
            "per": per, "per_days": days,
            "ledger": [{"id": g["id"], "at": g["at"], "kind": g["kind"], "kind_label": i18n.t("feedback.ledger.grant" if g["kind"] == "grant" else "feedback.ledger.revoke"),
                        "feedback": json_of(g["feedback"]), "per": g["per"], "days": g["days"], "old": g["old_expires"],
                        "new": g["new_expires"], "applied": bool(g["applied"]), "undone": g["undone"], "by": g["by"]}
                       for g in ledger]}


# ------------------------------------------------------------------ a user's own


PUBLIC = ("id", "at", "category", "category_label", "text", "status", "status_label", "reply", "replied", "changed",
          "unread")


def mine(user_id: int) -> dict:
    """An account's own feedback, newest first, as the user sees it (no diagnostics, no note, no rating but what a
    valid one earned: `reward`); how many are unread; and the reward's rule now in words (`rule`, "" when none)."""
    rows = [_row(r) for r in db().rows(f"{ROWS} WHERE f.user_id = ? ORDER BY f.at DESC", (user_id,))]
    rewards = _reward_texts(rows)
    items = [{k: f[k] for k in PUBLIC} | {"images": len(f["images"]), "reward": rewards.get(f["id"], "")} for f in rows]
    per, days = reward_rule()
    rule = Msg("I-FEEDBACK-RULE", per=per, days=days).text if days else ""
    return {"items": items, "unread": sum(f["unread"] for f in items), "rule": rule}


def mark_read(user_id: int) -> dict:
    """The account has looked at its feedback: nothing of it is unread any more."""
    with db().write() as c:
        c.execute("UPDATE feedback SET read_at = ? WHERE user_id = ? AND (read_at IS NULL OR read_at < changed)",
                  (time.time(), user_id))
    return mine(user_id)


def delete(fid: str, by: Actor) -> None:
    row = get(fid)
    with db().write() as c:
        c.execute("DELETE FROM feedback WHERE id = ?", (fid,))
    shutil.rmtree(folder(fid), ignore_errors=True)  # the backups keep theirs until they rotate out
    logs.say(log, Msg("I-FEEDBACK-DELETED", by=by.label, id=fid, person=row["person"]))


def archive(fid: str, trim=lambda row: row) -> bytes:
    """The whole feedback as one zip: feedback.json (what was written, who, status, the diagnostics) and the
    screenshots. `trim(row)`: the feedback as whoever downloads it may see it (server/feedback.py DIAGNOSTICS)."""
    row = trim(detail(fid))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("feedback.json", json.dumps(row, ensure_ascii=False, indent=1))
        for name in row["images"]:
            z.write(folder(fid) / name, name)
    return out.getvalue()
