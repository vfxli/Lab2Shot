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

from . import logs
from .config import settings
from .database import LIKE_ESCAPE, contains, db, json_of, json_text
from .errors import Invalid, NotFound
from .messages import Msg

log = logs.get("feedback")

MAX_TEXT = 10_000  # characters the user writes
MAX_BUNDLE = 2 << 20  # bytes of the page's diagnostics (as JSON)
MAX_IMAGES = 3
MAX_IMAGE = 5 << 20  # bytes of one screenshot
LOG_LINES = 200  # of a node's error log, of the server's log per job, of the server's log at the end
JOBS = 20  # recent jobs of the user attached

CATEGORIES = {"": "", **dict(error="出错了", usage="用法不明白", idea="建议")}  # labels of the categories (vocabulary, not messages)
STATUS = {"new": "新", "seen": "已看", "solved": "已解决"}
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
    from .farm import farm

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
                                          "gpu_name", "error", "reason")}
                   | {"error_log": {"file": str(error_log), "lines": _tail(error_log)} if error_log else None,
                      "server_log": about[r["id"]]})
    return out


def extension_states(graph: dict, named: list = ()) -> list[dict]:
    """Each node type of the graph, and each extension the page names (the help page of a project): whether its
    project is installed and ready (and why not)."""
    from .extensions import extensions
    from .extensions.status import extension_status
    from .nodes import node_types

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
    from .accounts import DELETED

    files = json_of(r["files"])
    return {"id": r["id"], "at": r["at"], "user": r["user_id"], "username": r["username"] or "",
            "person": DELETED if r["gone"] else r["person"] or "", "department": r["department"] or "",
            "category": r["category"], "category_label": CATEGORIES[r["category"]], "text": r["text"],
            "title": r["text"].strip().splitlines()[0][:120] if r["text"].strip() else "", "status": r["status"],
            "status_label": STATUS[r["status"]], "reply": r["reply"], "replied": r["replied"], "replied_by": r["replied_by"],
            "note": r["note"], "updated": r["updated"], "updated_by": r["updated_by"], "files": files,
            "images": [f["name"] for f in files if f["kind"] == "image"], "bytes": r["bytes"], "changed": r["changed"],
            "unread": _unread(r)}


def _unread(r) -> bool:
    """The user has not seen its latest status or reply yet."""
    return r["changed"] is not None and r["changed"] > (r["read_at"] or 0)


def get(fid: str) -> dict:
    folder(fid)
    r = db().row(f"{ROWS} WHERE f.id = ?", (fid,))
    if r is None:
        raise NotFound(Msg("E-FEEDBACK-GONE"))
    return _row(r)


def listing(status: str | None = None, since: float | None = None, until: float | None = None, person: str = "") -> dict:
    """Feedback, newest first, by status, time and part of an account's name or username; and how many there are of
    each status."""
    where, args = [], []
    if status:
        if status not in STATUS:
            raise Invalid(Msg("E-FEEDBACK-STATUS", status=status))
        where.append("status = ?")
        args.append(status)
    if since is not None:
        where.append("at >= ?")
        args.append(since)
    if until is not None:
        where.append("at < ?")
        args.append(until)
    if person.strip():
        where.append(f"(u.name LIKE ? ESCAPE '{LIKE_ESCAPE}' OR u.username LIKE ? ESCAPE '{LIKE_ESCAPE}')")
        args += [contains(person)] * 2
    rows = db().rows(f"{ROWS} {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY f.at DESC", tuple(args))
    counts = {s: 0 for s in STATUS} | {r["status"]: r["n"] for r in db().rows("SELECT status, COUNT(*) AS n FROM feedback GROUP BY status")}
    return {"items": [_row(r) for r in rows], "counts": counts}


def new_count() -> int:
    return db().row("SELECT COUNT(*) AS n FROM feedback WHERE status = 'new'")["n"]


def detail(fid: str) -> dict:
    """A feedback with its diagnostics."""
    row = get(fid)
    try:
        bundle = json.loads((folder(fid) / "bundle.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        bundle = None
    return {**row, "bundle": bundle}


def answer(fid: str, status: str, reply: str, note: str, by: str) -> dict:
    """The administrator's answer: the status and the reply (the user sees both: unread until they look), the note
    (only administrators see it)."""
    old = get(fid)
    if status not in STATUS:
        raise Invalid(Msg("E-FEEDBACK-STATUS", status=status))
    reply, note = reply.strip(), note.strip()
    for label, text in (("回复", reply), ("备注", note)):
        if len(text) > MAX_TEXT:
            raise Invalid(Msg("E-FEEDBACK-ANSWERLONG", what=label, count=len(text), max=MAX_TEXT))
    now = time.time()
    replied = (now if reply else None, by if reply else "") if reply != old["reply"] else (old["replied"], old["replied_by"])
    changed = now if status != old["status"] or reply != old["reply"] else old["changed"]
    with db().write() as c:
        c.execute("UPDATE feedback SET status = ?, reply = ?, replied = ?, replied_by = ?, note = ?, changed = ?, "
                  "updated = ?, updated_by = ? WHERE id = ?", (status, reply, *replied, note, changed, now, by, fid))
    return get(fid)


# ------------------------------------------------------------------ a user's own


PUBLIC = ("id", "at", "category", "category_label", "text", "status", "status_label", "reply", "replied", "changed",
          "unread")


def mine(user_id: int) -> dict:
    """An account's own feedback, newest first, as the user sees it (no diagnostics, no note); how many are unread."""
    rows = [_row(r) for r in db().rows(f"{ROWS} WHERE f.user_id = ? ORDER BY f.at DESC", (user_id,))]
    items = [{k: f[k] for k in PUBLIC} | {"images": len(f["images"])} for f in rows]
    return {"items": items, "unread": sum(f["unread"] for f in items)}


def mark_read(user_id: int) -> dict:
    """The account has looked at its feedback: nothing of it is unread any more."""
    with db().write() as c:
        c.execute("UPDATE feedback SET read_at = ? WHERE user_id = ? AND (read_at IS NULL OR read_at < changed)",
                  (time.time(), user_id))
    return mine(user_id)


def delete(fid: str, by: str) -> None:
    row = get(fid)
    with db().write() as c:
        c.execute("DELETE FROM feedback WHERE id = ?", (fid,))
    shutil.rmtree(folder(fid), ignore_errors=True)  # the backups keep theirs until they rotate out
    logs.say(log, Msg("I-FEEDBACK-DELETED", by=by, id=fid, person=row["person"]))


def archive(fid: str) -> bytes:
    """The whole feedback as one zip: feedback.json (what was written, who, status, the diagnostics) and the
    screenshots."""
    row = detail(fid)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("feedback.json", json.dumps(row, ensure_ascii=False, indent=1))
        for name in row["images"]:
            z.write(folder(fid) / name, name)
    return out.getvalue()
