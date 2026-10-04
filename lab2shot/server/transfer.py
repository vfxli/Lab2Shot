"""Files between the user's machine and this server (lab2shot/transfer): uploads in, outputs out (what 「输出」
packed in a task: its zip, downloaded by the browser itself, and the same files unpacked, fetched one by one by DCC
plugins and `lab2shot cook`). An output is only for the account whose task it is and a login holding data.others
(server/access.py mine); an upload only for the account that sent it (transfer/uploads.py). A file goes up in parts
that survive a dropped line (lab2shot/transfer/uploads.py); one file is at most 单任务上传上限 (tasks.upload_gb) big,
since a task never carries more than that of uploads, and none is taken while the disk would keep less than
暂停新计算的剩余空间 (farm/policy.py upload_floor, at least 10 GB)."""

from __future__ import annotations

import re
import threading
from urllib.parse import quote

from fastapi import Request
from pydantic import Base64Bytes
from starlette.requests import ClientDisconnect
from fastapi.responses import FileResponse

from .routes import BIG_BODY, Access, Body, Count, Limit, Router
from .wire import UPLOADS, WAITS, off_loop
from .. import logs
from ..config import settings
from ..errors import Conflict, TooLarge
from ..messages import Msg
from ..text import file_part
from ..farm import policy
from ..transfer import outputs, uploads
from . import auth, owners, quota

log = logs.get("uploads")

GB = 1 << 30
CHECK_EVERY = 1 << 30  # bytes between looks at the disk while an upload comes in
WRITE_BATCH = 1 << 20  # bytes a request brings that are written to the part at once, on a thread (never on the loop)

router = Router(prefix="/api", tags=["File Upload and Retrieval"])


class Shas(Body):
    shas: list[str]  # contents (sha256) this browser sent before


@router.post("/uploads/have", access=Access.user("Before uploading: which contents you have uploaded (others' do not count)", limit=Limit(body=BIG_BODY)), summary="Which contents you have uploaded that the server still keeps (by sha256, many in one question): those need not "
                                                                                                                                                            "be uploaded again; others' uploads do not count")
def have(req: Shas, request: Request) -> dict:
    if len(req.shas) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    return {"have": uploads.kept(req.shas, auth.me(request).id)}


class Folders(Body):
    folders: list[list[str]]  # the names of the files picked or dropped, per folder


@router.post("/uploads/sequences", access=Access.user("When choosing files: recognise which sequences the chosen file names hold (by name only)", limit=Limit(body=BIG_BODY)), summary="When choosing files: which sequences the chosen file names hold, which are single images, which are hidden or "
                                                                                                                                                                             "system files (by name only, apart per folder). Each sequence: name (written like plate.####.exr), frames "
                                                                                                                                                                             "(frame numbers, ascending), files (where those frames are in the folder's list of names)")
def sequences(req: Folders, request: Request) -> dict:
    auth.me(request)
    if sum(len(f) for f in req.folders) > 500_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYPICKED"))
    return {"folders": [_grouped(names) for names in req.folders]}


def _grouped(names: list[str]) -> dict:
    from pathlib import Path

    from ..io.sequence import FrameSequence, group_names

    at = {n: i for i, n in enumerate(names)}
    g = group_names(names)
    return {"sequences": [{"name": FrameSequence(Path(), h, t, p, tuple(frames)).name, "frames": list(frames), "files": [at[n] for n in frames.values()]}
                          for h, t, p, frames in g.sequences],
            "singles": [at[n] for n in g.singles], "junk": [at[n] for n in g.junk]}


def task_upload_limit() -> int:
    """单任务上传上限 in bytes: what one task may carry of uploads (farm/queue.py checks a task against it when it is
    submitted; one file above it is refused here as it comes in)."""
    return int(float(settings()["tasks.upload_gb"]) * GB)


def _upload_problem(size: int, who: int | None = None, pid: str = "") -> Msg | None:
    """Why an upload of `size` bytes (so far, or as announced) can't be taken (None it can): the one file's limit, the
    machine's free disk, and — with `who` — that account's own quota (server/quota.py, with what its other parts still
    have to send; the sentence says what it uses, what its limit is and where it can free space itself, never a bare
    「失败」). `pid`: the part being opened, when it may be one opened before (sent again)."""
    limit = task_upload_limit()
    if size > limit:  # one file bigger than a whole task may carry can never be used by a task
        return Msg("E-UPLOAD-TOOBIG", gb=limit / GB)
    uploads.root().mkdir(parents=True, exist_ok=True)
    # the data disk keeps what 暂停新计算的剩余空间 keeps (farm/policy.py upload_floor: one rule with the queue's pause,
    # never less than 10 GB): an upload that would go below it is not taken, as no new task is then
    total, free = policy.disk_now()
    if free - size < (floor := policy.upload_floor(total)):
        return Msg("E-UPLOAD-DISKFULL", free_gb=free / GB, keep_gb=floor / GB, pct=policy.space()["floor_pct"])
    if who is not None:
        try:
            quota.room_for(who, size, pid)
        except TooLarge as exc:
            return exc.message
    return None


def _cut_off(pid: str, who: int) -> None:
    logs.say(log, Msg("I-UPLOAD-CUTOFF", part=pid, offset=uploads.part_state(pid, who)["offset"]))


async def _receive(pid: str, offset: int, request: Request) -> dict:
    """The bytes a request brings, added to the part as they come (WRITE_BATCH at a time); a request cut off keeps what
    it brought. Only the account sending the part may add to it. Every file operation runs on a thread: the event loop
    only moves bytes and never blocks on a file."""
    who = auth.me(request).id
    try:
        token = await off_loop(uploads.begin, pid, offset, who, lane=UPLOADS)
    except uploads.Moved as exc:
        raise Conflict(Msg("E-UPLOAD-MOVED", reason=exc)) from exc
    looked, pending = offset, bytearray()

    async def write() -> None:
        nonlocal looked
        if not pending:
            return
        at = await off_loop(uploads.add, pid, token, bytes(pending), who, lane=UPLOADS)
        quota.wrote(who, len(pending))
        pending.clear()
        if at - looked >= CHECK_EVERY:
            looked = at
            if problem := await off_loop(_upload_problem, 0, who, lane=UPLOADS):
                raise TooLarge(problem)

    try:
        try:
            async for chunk in request.stream():
                pending += chunk
                if len(pending) >= WRITE_BATCH:
                    await write()
        except ClientDisconnect:  # the line dropped: what came is kept, the sender asks where to go on (nobody to answer)
            await write()
            await off_loop(_cut_off, pid, who, lane=UPLOADS)
        else:
            await write()
    except uploads.Moved as exc:
        raise Conflict(Msg("E-UPLOAD-MOVED", reason=exc)) from exc
    finally:  # a part grown across a restart is read back once to know its content
        state = await off_loop(uploads.end, pid, token, who, lane=UPLOADS)
    return state


# checking an account's room and opening its part are one step: two parts opened at once never both fit in what only
# one of them may take (a fixed number of locks, by account: other accounts mostly on others)
_OPENING = tuple(threading.Lock() for _ in range(64))


def _open(size: int, who: int, pid: str) -> dict:
    with _OPENING[who % len(_OPENING)]:
        if problem := _upload_problem(size, who, pid):
            raise TooLarge(problem)
        return uploads.open_part(size, who, pid)


@router.post("/uploads/parts", access=Access.user("Start uploading a file (size limited, refused when the disk is nearly full)", limit=Limit(body=None)), summary="Start uploading a file (size: its size; id: a name the sender picks, 32 hexadecimal digits, optional); the "
                                                                                                                                                            "body is its first bytes (all of a small file); returns {id, offset, size}, plus sha once complete. There is a "
                                                                                                                                                            "size limit, and nothing is taken when the disk is nearly full")
async def open_part(request: Request, size: int, id: str = "") -> dict:
    part = await off_loop(_open, size, auth.me(request).id, id, lane=UPLOADS)
    return await _receive(part["id"], 0, request)


@router.patch("/uploads/parts/{pid}", access=Access.user("Continue uploading one of your files (after a break, from the byte where it broke)", limit=Limit(body=None)), summary="Continue uploading a file: the body is the bytes from offset; a wrong offset gets 409 with the actual "
                                                                                                                                                                      "position. Bytes a broken request delivered are kept")
async def add_part(pid: str, request: Request, offset: int) -> dict:
    return await _receive(pid, offset, request)


@router.get("/uploads/parts/{pid}", access=Access.user("Where your upload in progress has got to"), summary="Where one of your uploads in progress has got to: {id, offset, size}, plus sha once complete")
def part(pid: str, request: Request) -> dict:
    return uploads.part_state(pid, auth.me(request).id)


class UploadSet(Body):
    name: str = ""  # what nodes read: a file, a sequence pattern, or "" for the whole folder
    files: dict[str, str]  # file name (sub-folders allowed) -> sha256 of an uploaded blob
    origin: dict = {}  # what it was called on the user's machine, and what the client says of itself (auth.details)


@router.post("/uploads", access=Access.user("Make your uploaded files one input", limit=Limit(body=BIG_BODY)), summary="Make your uploaded files one input (a file or a sequence); returns the reference upload:<id>/<name> used in "
                                                                                                                                  "node parameters")
def make_upload(req: UploadSet, request: Request) -> dict:
    u = auth.me(request)
    said = req.origin.get("client") if isinstance(req.origin.get("client"), dict) else {}
    ref = uploads.make_set(req.name, req.files, {"path": str(req.origin.get("path") or "")[:500], "user": u.username,
                                                 **auth.details(request, said)}, u.id)
    info = uploads.describe(ref)
    logs.say(log, Msg("I-UPLOAD-SENT", user=u.username, what=req.origin.get("path") or req.name, files=info["files"], mb=info["bytes"] / 1e6, ref=ref))
    return info


class DeclareSet(Body):
    """选完文件那一刻发上来的：这份上传里有哪些文件、各是什么内容，外加第一个文件的头几十 KB。"""

    name: str = ""  # 节点读的是哪一个：一个文件，或者序列的图案（plate.####.exr）
    files: dict[str, str]  # 文件名（可以带子文件夹）-> 这个文件内容的 sha256（网页自己算的）
    origin: dict = {}
    head_of: str = ""  # 头部字节是哪个文件的（要序列的第一帧）
    head: Base64Bytes = b""  # 那个文件开头那几十 KB，base64（读不成字节的在入口就拒）
    head_whole: Count = 0  # 那个文件整个多大（判断「已经把整份都给了」）
    sizes: dict[str, Count] = {}  # 每个文件在用户机器上多大：字节还没传时，文件参数那一行的「18 MB」只能由网页给


@router.post("/uploads/declare", access=Access.user("Declare an upload (bytes not sent yet)", limit=Limit(body=BIG_BODY)),
             summary="Declare an upload when files are chosen: which files, each one's content sha256, plus the first few dozen KB "
                     "of the first file. Returns the reference upload:<id>/<name> used in node parameters (the same as once the "
                     "bytes are uploaded), the layers read from the header, and which files' bytes are still missing. When the "
                     "header given is too short to read the layers, returns need=how many more bytes; send again")
def declare_upload(req: DeclareSet, request: Request) -> dict:
    u = auth.me(request)
    if len(req.files) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYPICKED"))
    said = req.origin.get("client") if isinstance(req.origin.get("client"), dict) else {}
    ref = uploads.declare_set(req.name, req.files, {"path": str(req.origin.get("path") or "")[:500], "user": u.username,
                                                    **auth.details(request, said)}, u.id, req.sizes)
    sid, _ = uploads.ref_parts(ref)
    out: dict = {"ref": ref, "missing": uploads.still_missing(sid, u.id)}
    known = uploads.head_described(sid)
    if known is None and req.head and req.head_of:
        raw = req.head
        # the bytes kept are the ones that came, whatever the client says the whole file is: what is checked and
        # counted (quota.wrote), in the account's one step of taking bytes in (_OPENING)
        with _OPENING[u.id % len(_OPENING)]:
            if (why := _upload_problem(max(len(raw), req.head_whole), u.id)) is not None:
                raise TooLarge(why)
            known = uploads.describe_head(sid, req.head_of, raw, req.head_whole or len(raw))
            quota.wrote(u.id, len(raw))
    if known is not None:
        out.update(known)
    return out


class PlanesHave(Body):
    shas: list[str]  # 原文件的 sha256（浏览器申报时自己算的）
    channels: list[str]  # 要的通道名（状态回复里 `channels.write` 那一份）


@router.post("/uploads/planes/have", access=Access.user("Before uploading: which channels of your files the server does not have yet", limit=Limit(body=BIG_BODY)),
             summary="Before a channel-level upload: of the channels wanted from these source files (by sha256), which the server "
                     "still lacks ({sha: [channel names]}, an empty table means all are there, as when the whole file is there). "
                     "Others' uploads do not count")
def planes_have(req: PlanesHave, request: Request) -> dict:
    if len(req.shas) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    return {"missing": uploads.planes_missing(req.shas, req.channels, auth.me(request).id)}


class Planes(Body):
    """一帧的几条通道，传完之后来这一趟：平面已经按分段协议传成了一份 blob（gzip 过，按 channels 的顺序首尾相接）。"""

    sid: str  # 这份上传申报时的 id（引用 upload:<id>/... 里的那一段）
    sha: str  # 原文件的 sha256
    blob: str  # 装平面的那份 blob 的 sha256（分段协议传完时服务器交回的）
    channels: list[dict]  # [{"take": 文件里的通道名, "write": 子集里的名字, "type": half|float|uint}]
    width: int
    height: int
    compression: str = "zips"  # 原文件的压缩方式（OpenEXR 的名字）：无损的照它，有损的服务器换成 ZIPS
    display: list[int] | None = None  # 这一帧的画幅 [x, y, w, h]（浏览器解得出来才给）
    data: list[int] | None = None  # 这一帧的数据窗口 [x, y, w, h]


@router.post("/uploads/planes", access=Access.user("Write the channels you uploaded as an EXR holding only them", lane=WAITS),
             summary="Channel-level upload: some channels of one frame (one gzipped blob) -> the server writes an EXR holding only "
                     "those channels with OpenImageIO (channel names, pixel types and compression as in the source file, no "
                     "metadata), merged with the channels this source file already has, stored by content. Returns {sha, channels, "
                     "blob}. After every frame, POST /api/uploads to make the input")
def planes(req: Planes, request: Request) -> dict:
    u = auth.me(request)
    if len(req.channels) > 1024:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    out = uploads.add_planes(req.sid, req.sha, req.blob, req.channels, req.width, req.height, req.compression, u.id,
                             req.display, req.data)
    logs.say(log, Msg("I-UPLOAD-PLANES", user=u.username, file=req.sha[:12], count=len(req.channels),
                      channels=" ".join(c.get("write", "") for c in req.channels), kb=out.get("bytes", 0) / 1e3, blob=out["blob"][:12]))
    return out


@router.get("/uploads/describe", access=Access.user("Upload information shown on the file parameter (your own only)", owned=owners.upload), summary="Name, file count, size and sequence frames of one of your uploads (shown on the file parameter)")
def describe_upload(ref: str, request: Request) -> dict:
    return uploads.describe(ref)


def attachment(name: str) -> str:
    """A download's Content-Disposition: the file name as the user's machine saves it, RFC 5987 (filename*=UTF-8''…)
    with a plain ASCII filename= beside it (RFC 6266); path separators and control characters never go out, whatever
    the name holds."""
    stem, dot, ext = str(name).rpartition(".")
    clean = (f"{file_part(stem, 200) or 'Lab2Shot'}.{file_part(ext, 10)}" if dot
             else (file_part(name, 200) or "Lab2Shot"))
    fallback = re.sub(r"[^A-Za-z0-9._-]", "_", clean)
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(clean, safe='')}"


@router.get("/outputs", access=Access.user("Your results"), summary="Packaged outputs of your jobs (newest first): which job, which node, download file name, size, when they go "
                                                                       "with the job")
def my_outputs(request: Request) -> list[dict]:
    return outputs.of_account(auth.me(request).id)


@router.get("/tasks/{task_id}/outputs", access=Access.user("Results of one of your jobs", owned=owners.task), summary="Packaged outputs of one of your jobs (first packaged first)")
def task_outputs(task_id: str, request: Request) -> list[dict]:
    return outputs.of_task(task_id)


@router.get("/tasks/{task_id}/outputs/{pkg}", access=Access.user("One of your results", owned=owners.task_output), summary="One of your results: which files the unpackaged folder holds (a DCC plugin fetches them one by one as needed), "
                                                                                                                             "download file name, size")
def output_record(task_id: str, pkg: str, request: Request) -> dict:
    return outputs.record(task_id, pkg, files=True)


@router.get("/tasks/{task_id}/outputs/{pkg}/file/{name:path}", access=Access.user("Fetch one file of your result", owned=owners.task_output), summary="Fetch one file of your result (the unpackaged copy on the server; the result's record lists every file), "
                                                                                                                                                             "resumable (Range)")
def output_file(task_id: str, pkg: str, name: str, request: Request) -> FileResponse:
    f = outputs.file(task_id, pkg, name)
    return FileResponse(f, headers={"Content-Disposition": attachment(f.name)})  # a download, never shown as a page of this site


@router.get("/tasks/{task_id}/outputs/{pkg}/zip", access=Access.user("Download your whole result", owned=owners.task_output), summary="Download your whole result: the packaged zip (the browser's own download), resumable (Range / If-Range, ETag)")
def output_zip(task_id: str, pkg: str, request: Request) -> FileResponse:
    """The zip as it is on the disk, from its first byte or from where a cut-off download got to: the browser's own
    download, the DCC client (lab2shot/client.py) and anything else resume by Range, taken only while the zip is still
    the one the download began (If-Range with its ETag or Last-Modified): Starlette's FileResponse answers 206 / 416
    and Accept-Ranges itself. The zip never changes once written (transfer/outputs.py), so a resumed download is
    always the same bytes."""
    r = request.state.owned
    return FileResponse(outputs.zip_file(task_id, pkg), media_type="application/zip",
                        headers={"Content-Disposition": attachment(r["name"])})
