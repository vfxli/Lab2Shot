"""Files between the user's machine and this server (lab2shot/transfer): uploads in, deliveries out (what 「输出」
delivers: one archive, or files into a folder, and what became of it). The web page, DCC plugins and `lab2shot cook`
all use these. A delivery is only for the account whose cook made it and the administrator (server/access.py mine);
an upload only for the account that sent it (transfer/uploads.py). A file goes up in parts that survive a dropped
line (lab2shot/transfer/uploads.py); it is at most storage.upload_gb big, and none is taken while the disk would keep
less than KEEP_FREE_GB."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import Request
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

import shutil

from .routes import Access, Limit, Router
from .. import logs
from ..config import settings
from ..errors import Conflict, Invalid, TooLarge
from ..messages import Msg
from ..transfer import deliveries, uploads
from . import auth, owners, wire

log = logs.get("uploads")

KEEP_FREE_GB = 10  # uploads stop before the disk of the work folder has less than this left
CHECK_EVERY = 1 << 30  # bytes between looks at the disk while an upload comes in
WRITE_BATCH = 1 << 20  # bytes a request brings that are written to the part at once, on a thread (never on the loop)

router = Router(prefix="/api", tags=["文件上传与取回"])


class Shas(BaseModel):
    shas: list[str]  # contents (sha256) this browser sent before


@router.post("/uploads/have", access=Access.user("上传前问：自己传过哪些内容（别人传过的不算）"), summary="自己传过哪些内容的文件、服务器还留着（按 sha256，一次问很多个）：有的就不用再传；别人传过的不算")
def have(req: Shas, request: Request) -> dict:
    if len(req.shas) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    return {"have": uploads.kept(req.shas, auth.me(request).id)}


class Folders(BaseModel):
    folders: list[list[str]]  # the names of the files picked or dropped, per folder


@router.post("/uploads/sequences", access=Access.user("选文件时：认出选中的文件名里有哪几段序列（只看名字）"), summary="选文件时：选中的文件名里有哪几段序列、哪些是单张图、哪些是隐藏或系统文件（只看名字，按文件夹分开）。"
             "每段序列：name（plate.####.exr 这样的写法）、frames（帧号，从小到大）、files（这些帧在这个文件夹的名字列表里的位置）")
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


def _upload_problem(size: int, who: int | None = None) -> Msg | None:
    """Why an upload of `size` bytes (so far, or as announced) can't be taken (None it can): the one file's limit, the
    machine's free disk, and — with `who` — that account's own quota (server/quota.py; the sentence says what it uses,
    what its limit is and where it can free space itself, never a bare 「失败」)."""
    limit = int(settings()["storage.upload_gb"]) << 30
    if size > limit:
        return Msg("E-UPLOAD-TOOBIG", gb=limit >> 30)
    uploads.root().mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(uploads.root()).free - size < KEEP_FREE_GB << 30:
        return Msg("E-UPLOAD-DISKFULL", gb=KEEP_FREE_GB)
    if who is not None:
        from . import quota

        try:
            quota.room_for(who, size)
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
        token = await run_in_threadpool(uploads.begin, pid, offset, who)
    except uploads.Moved as exc:
        raise Conflict(Msg("E-UPLOAD-MOVED", reason=exc)) from exc
    looked, pending = offset, bytearray()

    async def write() -> None:
        nonlocal looked
        if not pending:
            return
        at = await run_in_threadpool(uploads.add, pid, token, bytes(pending), who)
        pending.clear()
        if at - looked >= CHECK_EVERY:
            looked = at
            if problem := await run_in_threadpool(_upload_problem, 0, who):
                raise TooLarge(problem)

    try:
        try:
            async for chunk in request.stream():
                pending += chunk
                if len(pending) >= WRITE_BATCH:
                    await write()
        except ClientDisconnect:  # the line dropped: what came is kept, the sender asks where to go on (nobody to answer)
            await write()
            await run_in_threadpool(_cut_off, pid, who)
        else:
            await write()
    except uploads.Moved as exc:
        raise Conflict(Msg("E-UPLOAD-MOVED", reason=exc)) from exc
    finally:  # a part grown across a restart is read back once to know its content
        state = await run_in_threadpool(uploads.end, pid, token, who)
    return state


def _open(size: int, who: int, pid: str) -> dict:
    if problem := _upload_problem(size, who):
        raise TooLarge(problem)
    return uploads.open_part(size, who, pid)


@router.post("/uploads/parts", access=Access.user("开始上传一个文件（有大小上限，硬盘快满时拒绝）", limit=Limit(body=None)), summary="开始上传一个文件（size：它的大小；id：发的一方自己起的名字，32 位十六进制，可不给），请求体是它开头的字节（小文件就是全部）；返回 {id, offset, size}，传完的还有 sha。有大小上限，硬盘快满时不收")
async def open_part(request: Request, size: int, id: str = "") -> dict:
    part = await run_in_threadpool(_open, size, auth.me(request).id, id)
    return await _receive(part["id"], 0, request)


@router.patch("/uploads/parts/{pid}", access=Access.user("接着上传自己的一个文件（断了从断开的字节接着传）", limit=Limit(body=None)), summary="接着上传一个文件：请求体是从 offset 起的字节；offset 不对回 409 和实际的位置。断开的请求收到的字节都留着")
async def add_part(pid: str, request: Request, offset: int) -> dict:
    return await _receive(pid, offset, request)


@router.get("/uploads/parts/{pid}", access=Access.user("自己上传中的文件传到了哪里"), summary="自己一个上传中的文件传到了哪里：{id, offset, size}，传完的还有 sha")
def part(pid: str, request: Request) -> dict:
    return uploads.part_state(pid, auth.me(request).id)


class UploadSet(BaseModel):
    name: str = ""  # what nodes read: a file, a sequence pattern, or "" for the whole folder
    files: dict[str, str]  # file name (sub-folders allowed) -> sha256 of an uploaded blob
    origin: dict = {}  # what it was called on the user's machine, and what the client says of itself (auth.details)


@router.post("/uploads", access=Access.user("把自己上传好的文件组成一份输入"), summary="把自己上传好的文件组成一份输入（一个文件或一段序列），返回节点参数里用的引用 upload:<id>/<名字>")
def make_upload(req: UploadSet, request: Request) -> dict:
    u = auth.me(request)
    said = req.origin.get("client") if isinstance(req.origin.get("client"), dict) else {}
    ref = uploads.make_set(req.name, req.files, {"path": str(req.origin.get("path") or "")[:500], "user": u.username,
                                                 **auth.details(request, said)}, u.id)
    info = uploads.describe(ref)
    logs.say(log, Msg("I-UPLOAD-SENT", user=u.username, what=req.origin.get("path") or req.name, files=info["files"], mb=info["bytes"] / 1e6, ref=ref))
    return info


class DeclareSet(BaseModel):
    """选完文件那一刻发上来的：这份上传里有哪些文件、各是什么内容，外加第一个文件的头几十 KB。"""

    name: str = ""  # 节点读的是哪一个：一个文件，或者序列的图案（plate.####.exr）
    files: dict[str, str]  # 文件名（可以带子文件夹）-> 这个文件内容的 sha256（网页自己算的）
    origin: dict = {}
    head_of: str = ""  # 头部字节是哪个文件的（要序列的第一帧）
    head: str = ""  # 那个文件开头那几十 KB，base64
    head_whole: int = 0  # 那个文件整个多大（判断「已经把整份都给了」）
    sizes: dict[str, int] = {}  # 每个文件在用户机器上多大：字节还没传时，文件参数那一行的「18 MB」只能由网页给


@router.post("/uploads/declare", access=Access.user("申报一份上传（字节还没传）"),
             summary="选完文件时申报一份上传：里面有哪些文件、各自内容的 sha256，外加第一个文件的头几十 KB。"
                     "返回节点参数里用的引用 upload:<id>/<名字>（和字节传完之后是同一个）、从头部读出来的图层、"
                     "还缺哪几个文件的字节。头给少了读不出图层时返回 need=还要多少字节，再发一次")
def declare_upload(req: DeclareSet, request: Request) -> dict:
    u = auth.me(request)
    if len(req.files) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYPICKED"))
    said = req.origin.get("client") if isinstance(req.origin.get("client"), dict) else {}
    ref = uploads.declare_set(req.name, req.files, {"path": str(req.origin.get("path") or "")[:500], "user": u.username,
                                                    **auth.details(request, said)}, u.id, req.sizes)
    sid = ref[len("upload:"):].split("/")[0]
    out: dict = {"ref": ref, "missing": uploads.still_missing(sid, u.id)}
    known = uploads.head_described(sid)
    if known is None and req.head and req.head_of:
        import base64

        raw = base64.b64decode(req.head)
        if (why := _upload_problem(req.head_whole or len(raw), u.id)) is not None:
            raise TooLarge(why)
        known = uploads.describe_head(sid, req.head_of, raw, req.head_whole or len(raw))
    if known is not None:
        out.update(known)
    return out


class PlanesHave(BaseModel):
    shas: list[str]  # 原文件的 sha256（浏览器申报时自己算的）
    channels: list[str]  # 要的通道名（状态回复里 `channels.write` 那一份）


@router.post("/uploads/planes/have", access=Access.user("上传前问：自己这几份文件的哪几条通道服务器还没有"),
             summary="通道级上传之前问：这几份原文件（按 sha256）这次要的通道里，服务器还缺哪几条（{sha: [通道名]}，空表 = 都有了，"
                     "整份文件在也算都有）。别人传的不算")
def planes_have(req: PlanesHave, request: Request) -> dict:
    if len(req.shas) > 100_000:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    return {"missing": uploads.planes_missing(req.shas, req.channels, auth.me(request).id)}


class Planes(BaseModel):
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


@router.post("/uploads/planes", access=Access.user("把自己传上来的几条通道写成只含它们的 EXR"),
             summary="通道级上传：一帧的几条通道（一份 gzip 过的 blob）→ 服务器用 OpenImageIO 写成只含这几条通道的 EXR（通道名、像素类型、"
                     "压缩照原文件，不带任何元数据），和这份原文件已有的通道并起来，按内容寻址存下。返回 {sha, channels, blob}。"
                     "传完每一帧再 POST /api/uploads 把这份输入组起来")
def planes(req: Planes, request: Request) -> dict:
    u = auth.me(request)
    if len(req.channels) > 1024:
        raise TooLarge(Msg("E-UPLOAD-TOOMANYASKED"))
    out = uploads.add_planes(req.sid, req.sha, req.blob, req.channels, req.width, req.height, req.compression, u.id,
                             req.display, req.data)
    logs.say(log, Msg("I-UPLOAD-PLANES", user=u.username, file=req.sha[:12], count=len(req.channels),
                      channels=" ".join(c.get("write", "") for c in req.channels), kb=out.get("bytes", 0) / 1e3, blob=out["blob"][:12]))
    return out


@router.get("/uploads/describe", access=Access.user("文件参数上显示的上传信息（只有自己的）", owned=owners.upload), summary="自己的一份上传的名字、文件数、大小和序列的帧（网页显示在文件参数上）")
def describe_upload(ref: str, request: Request) -> dict:
    return uploads.describe(ref)


@router.get("/deliveries", access=Access.user("自己的结果"), summary="自己的「输出」结果（最新的在前）：状态（待取回 / 已保存 / 已下载 / 不要了 / 已过期）、什么时候过期")
def delivery_list(request: Request) -> list[dict]:
    return deliveries.listing(auth.me(request).id)


@router.get("/deliveries/{run}/{node}", access=Access.user("自己的一份结果", owned=owners.delivery), summary="自己的一份结果：怎么取回（tar / tar.gz / 文件夹）、有哪些文件、现在的状态")
def delivery_record(run: str, node: str, request: Request) -> dict:
    return request.state.owned


@router.get("/deliveries/{run}/{node}/file/{name:path}", access=Access.user("取回自己结果里的一个文件", owned=owners.delivery), summary="取回自己结果里的一个文件（结果的记录里列出了每个文件；「文件夹」方式一个一个取）")
def delivery_file(run: str, node: str, name: str, request: Request) -> FileResponse:
    f = deliveries.file(run, node, name)
    return FileResponse(f, filename=f.name)  # a download, never shown as a page of this site


@router.get("/deliveries/{run}/{node}/archive", access=Access.user("取回自己的整份结果", owned=owners.delivery), summary="自己的整份结果打成一个 tar（tar.gz 方式压缩），边打包边发；download=1 表示是浏览器下载，发完记为已下载")
def delivery_archive(run: str, node: str, request: Request, download: bool = False) -> Response:
    """The archive from its first byte, or from where a cut-off download got to: one byte range (Range), taken only
    while the archive is still the one the download began (If-Range with its ETag, deliveries.archive_tag); the page's
    browser and the DCC client (lab2shot/client.py) resume the same way. A file of a folder delivery resumes by
    FileResponse's own ranges."""
    r = request.state.owned
    gz = r["mode"] == "tar.gz"
    name = r["name"] if r["mode"] != "folder" else deliveries.delivered_name(r["name"], "tar")
    done = (lambda: _downloaded(run, node)) if download else None
    tag = f'"{deliveries.archive_tag(run, node, gz)}"'
    media = "application/gzip" if gz else "application/x-tar"
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}", "ETag": tag, "Accept-Ranges": "bytes"}
    asked = request.headers.get("range")
    if asked and request.headers.get("if-range", tag) == tag:
        size = deliveries.archive_size(run, node, gz)
        try:
            span = wire.byte_range(asked, size)
        except wire.Unsatisfiable:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
        if span is not None:
            start, end = span
            part = {**headers, "Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(end - start + 1)}
            body = deliveries.archive_stream(run, node, gz, done if end == size - 1 else None, start, end - start + 1)
            return StreamingResponse(body, status_code=206, headers=part, media_type=media)
    if not gz:
        headers["Content-Length"] = str(deliveries.archive_size(run, node))
    return StreamingResponse(deliveries.archive_stream(run, node, gz, done), headers=headers, media_type=media)


@router.get("/deliveries/{run}/{node}/batch", access=Access.user("取回自己一个「输出」的全部结果", owned=owners.delivery_batch), summary="一个「输出」在这次计算里交付的全部包（块内逐项处理每条一个包）打成一个总包：里面每条一个子文件夹，一次下载拿全（文件夹方式和 DCC 插件按 /deliveries 列表一个个取，不用这个）")
def delivery_batch(run: str, node: str, request: Request) -> Response:
    """A browser download cannot be asked N times, so the N packages of a block's items go out as one
    package holding them (a plain tar, whatever each package's own mode is: it is a download, not the user's chosen
    way of saving). The list itself — which packages there are — is /api/deliveries and the job's output events."""
    found = request.state.owned
    name = deliveries.delivered_name(f"{found[0]['label']}_全部", "tar")
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}", "Accept-Ranges": "none",
               "ETag": f'"{deliveries.archive_tag(run, node, False, group=True)}"',
               "Content-Length": str(deliveries.archive_size(run, node, False, group=True))}
    return StreamingResponse(deliveries.archive_stream(run, node, False, group=True), headers=headers,
                             media_type="application/x-tar")


def _downloaded(run: str, node: str) -> None:
    try:
        deliveries.mark(run, node, "downloaded")
    except (OSError, ValueError):  # gone meanwhile (NotFound is a ValueError): nothing to note
        pass


class DeliveryState(BaseModel):
    state: str  # saved: the client wrote its copy; dismissed: the user doesn't want it


@router.post("/deliveries/{run}/{node}/state", access=Access.user("报告自己的结果存好了或不要了", owned=owners.delivery), summary="报告自己的一份结果：已经存好（saved，写完之后才报）或者不要了（dismissed）")
def delivery_state(run: str, node: str, req: DeliveryState, request: Request) -> dict:
    if req.state not in ("saved", "dismissed"):
        raise Invalid(Msg("E-DELIVERY-STATE", state=req.state))
    return deliveries.mark(run, node, req.state)
