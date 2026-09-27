"""What 「输出」 delivers, waiting for the user's machine to fetch it: the web page saves it where the user chose (one
archive, or the files into a folder), DCC plugins and `lab2shot cook` write it where the user asked.

    work/deliveries/<run>/<node>/            one 「输出」's delivery from one run (a job's id is its run):
        lab2shot.json                        what is inside (DCC plugins read it to import the files)
        <名字>/<名字>.usd ...                 one sub-folder per output-settings node, its files named after it

An 「输出」 inside a 逐项处理 block delivers once per item: each item's own package, addressed
<node>.<hash of its item path> so that one item never writes over another, its record saying which node and which
item it is (`node`, `item`). A folder handle and the DCC clients take the N packages into one folder; a browser
download, which cannot be asked N times, takes one package holding the N (group_archive_*: the same tar machinery,
each item's package a sub-folder of it).

Its record (whose it is, how it is fetched, what became of it) is in the database (lab2shot/database, deliveries).

The files are hard links to the output-settings nodes' results in the cache (no second copy; they stay when the cache is
cleaned). An archive is streamed while it is made, never kept whole: tar blocks are written here, one chunk of a file
at a time, gzip-compressed on the way for tar.gz; it can start at any byte (a download cut off goes on from there). A delivery is "saved" when the client that submitted the cook says
its copy is written (not when a download starts), "downloaded" when a whole archive was sent as a browser download,
"dismissed" when the user said they do not want it. Deliveries are kept as many days as the administrator set
(storage.delivery_days, /admin 设置, read each time), then they are gone (the results can be cooked again from the
cache, that is quick).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import threading
import time
import zlib
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path, PurePosixPath

from ..config import settings
from ..data.packet import Packet
from ..database import db, json_of, json_text
from ..errors import CookError, Invalid, NotFound
from ..io.digest import sha256
from ..io.files import inside, link_or_copy
from ..messages import Msg
from ..nodes.output import marked
from . import relative_name

MANIFEST = "lab2shot.json"  # at the root of every delivery: what is inside
MODES = {"tar": ".tar", "tar.gz": ".tar.gz", "folder": ""}  # how the user's machine fetches it -> the name's suffix
STATES = ("pending", "saved", "downloaded", "dismissed", "expired")
_PART = re.compile(r"^(?!\.+$)[A-Za-z0-9_.-]+$")  # a job id or a node id: never "." or ".."
_ARCHIVE = re.compile(r"(\.tar\.gz|\.tgz|\.tar)$", re.IGNORECASE)


def root() -> Path:
    return settings().work_dir / "deliveries"


def keep_days() -> int:
    return int(settings()["storage.delivery_days"])


def _gone() -> NotFound:
    return NotFound(Msg("E-DELIVERY-GONE", days=keep_days()))


def folder(run: str, node: str) -> Path:
    """Where one delivery is: `node` is its address: the 「输出」 node's id, or `<node>.<hash>` for one item of a
    block (address())."""
    if not _PART.match(run) or not _PART.match(node):
        raise _gone()
    return root() / run / node


def address(node: str, item: str = "") -> str:
    """The name one delivery is kept and asked for under: the node's id, and for an item of a 逐项处理 block the node
    and that item (its path, hashed: an item is named by the user and may contain anything, an address may not)."""
    return node if not item else f"{node}.{sha256(item)[:12]}"


def delivered_name(name: str, mode: str, learned: list[str] = ()) -> str:
    """What the user gets is called: the archive's file name with its suffix (sh010 -> sh010.tar), or the folder's
    name, marked when a model made what is in it (nodes/output.py marked: sh010_ML_Lab2Shot_ViPE.tar). Only a name: a
    path the client sent keeps its last part."""
    base = PurePosixPath(name.replace("\\", "/")).name
    return marked(_ARCHIVE.sub("", base), list(learned)) + MODES[mode] if MODES[mode] else marked(base, list(learned))


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    link_or_copy(src, dst)


def make(run: str, node: str, outputs: list, info: dict, item: str = "") -> dict:
    """Deliver the files of `outputs` (files packets: meta name, main, files) under `run`/`address(node, item)`: each
    output in a sub-folder of its 名字, and a manifest. `info`: label, name (as the user named it), mode, user (the
    account whose cook it is), who, title, graph (the graph file's meta.id, "" when the submitter sent none: the page
    only ever draws a delivery on a node of the graph it is open on now, never one left behind by another graph's job
    of the same node id). `item`: the item path of the instance that
    delivered ("" outside every block), so one item never writes over another.
    Returns the record."""
    at = address(node, item)
    base = folder(run, at)
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    listed = []
    for p in outputs:
        name = str(relative_name(p.meta["name"]))
        for rel in p.meta["files"]:
            _link(p.path(rel), base / name / rel)
        listed.append({"name": name, "type": p.meta.get("made_from", ""), "main": f"{name}/{p.meta['main']}",
                       "files": [f"{name}/{rel}" for rel in p.meta["files"]], "commercial": p.meta["commercial"]})
    manifest = {"schema": "lab2shot.delivery/1", "title": info.get("title", ""), "node": info.get("label", node),
                "made": time.strftime("%Y-%m-%dT%H:%M:%S"), "outputs": listed,
                "commercial": all(o["commercial"] for o in listed)}
    (base / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    record = {"run": run, "node": node, "item": item, "address": at, "label": info.get("label", node), "mode": info["mode"],
              "name": delivered_name(info["name"], info["mode"], [x for p in outputs for x in p.meta.get("learned", [])]),
              "user": info.get("user"),
              "who": info.get("who", ""), "title": info.get("title", ""), "graph": info.get("graph", ""), "created": time.time(),
              "files": files(run, at), "bytes": sum(f.stat().st_size for f in base.rglob("*") if f.is_file()),
              "commercial": manifest["commercial"], "saved": None, "downloaded": None, "dismissed": None}
    with db().write() as c:
        _write(c, record)
    return with_state(record)


def _write(c, record: dict) -> None:
    c.execute("INSERT INTO deliveries (run, node, created, user_id, record) VALUES (?, ?, ?, ?, ?) "
              "ON CONFLICT (run, node) DO UPDATE SET created = excluded.created, user_id = excluded.user_id, record = excluded.record",
              (record["run"], record.get("address") or record["node"], record["created"], record.get("user"), json_text(record)))


class Sink:
    """Where the 「输出」 nodes of one run deliver (nodes/services.py DeliverySink): under `run` (a job's id), as `owner`'s
    (user, who, title, graph: whose deliveries they are), each delivery reported to `emit` as the run's "output" event.
    The farm builds one per job; the engine hands it to the nodes that deliver."""

    def __init__(self, run: str, owner: dict, emit) -> None:
        self.run, self.owner, self.emit = run, owner, emit

    def deliver(self, node_id: str, label: str, outputs: list, name: str, mode: str, item: str = "") -> dict:
        """`item`: the item path of the instance delivering ("" outside every 逐项处理 block); inside one, every item
        delivers its own package."""
        for p in outputs:  # a packet without .complete is not a packet: an incomplete one is never delivered
            if not Packet.exists(p.dir):  # checked outright, never by an assert (python -O would drop it)
                raise CookError(node_id, Msg("E-DELIVER-INCOMPLETE", node=label))
        record = make(self.run, node_id, outputs, {"label": label, "name": name, "mode": mode, **self.owner}, item)
        self.emit({"type": "output", "node": node_id, "label": label, "run": self.run, "name": record["name"],
                   "item": item, "address": record["address"], "mode": record["mode"], "files": record["files"],
                   "bytes": record["bytes"], "commercial": record["commercial"]})
        return record


def expires(record: dict) -> float:
    return record["created"] + keep_days() * 86400


def state(record: dict, now: float | None = None) -> str:
    """pending (待取回) / saved (已保存) / downloaded (已下载) / dismissed (不要了) / expired (已过期)."""
    for key in ("saved", "downloaded", "dismissed"):
        if record.get(key):
            return key
    return "expired" if (now or time.time()) >= expires(record) else "pending"


def with_state(record: dict) -> dict:
    return {**record, "state": state(record), "expires": expires(record)}


def record(run: str, node: str) -> dict:
    folder(run, node)  # a run or node that cannot be one
    r = db().row("SELECT record, user_id FROM deliveries WHERE run = ? AND node = ?", (run, node))
    if r is None:
        raise _gone()
    return with_state({**json_of(r["record"]), "user": r["user_id"]})


def listing(user_id: int | None = None) -> list[dict]:
    """Every delivery kept (newest first), or those of one account."""
    rows = (db().rows("SELECT record, user_id FROM deliveries ORDER BY created DESC") if user_id is None
            else db().rows("SELECT record, user_id FROM deliveries WHERE user_id = ? ORDER BY created DESC", (user_id,)))
    return [with_state({**json_of(r["record"]), "user": r["user_id"]}) for r in rows]


def remove_user(user_id: int) -> int:
    """An account is deleted: its deliveries go (records and files); returns how many."""
    rows = db().rows("SELECT run, node FROM deliveries WHERE user_id = ?", (user_id,))
    with db().write() as c:
        c.execute("DELETE FROM deliveries WHERE user_id = ?", (user_id,))
    for r in rows:
        shutil.rmtree(folder(r["run"], r["node"]), ignore_errors=True)
    return len(rows)


def forget_run(run: str) -> int:
    """A computation (one job) was deleted from the queue: its delivery packages left on the server go with it
    (records and files). Returns how many were deleted.

    A job may deliver several packages (one per 「输出」 node in the graph, and per item inside 逐项处理 blocks); `run`
    is the job id, so they are deleted together by it. Only delivery packages are deleted, never the result cache: the
    cache is shared by fingerprint and may still be used by other jobs."""
    rows = db().rows("SELECT run, node FROM deliveries WHERE run = ?", (run,))
    with db().write() as c:
        c.execute("DELETE FROM deliveries WHERE run = ?", (run,))
    shutil.rmtree(root() / run, ignore_errors=True)
    return len(rows)


def mark(run: str, node: str, what: str) -> dict:
    """The delivery was saved (the client that submitted the cook wrote its copy), downloaded (a whole archive went
    out as a download) or dismissed (the user does not want it). Whose it is the caller checked (server/access.py)."""
    if what not in ("saved", "downloaded", "dismissed"):
        raise Invalid(Msg("E-DELIVERY-MARK", state=what))
    with db().write() as c:
        r = record(run, node)
        if r["state"] == "expired":
            raise _gone()
        r[what] = r[what] or time.time()
        _write(c, {k: v for k, v in r.items() if k not in ("state", "expires")})
        return with_state(r)


def files(run: str, node: str) -> list[str]:
    """Every file of a delivery, relative to it (the manifest first)."""
    base = folder(run, node)
    if not base.is_dir():
        return []
    names = sorted(str(f.relative_to(base).as_posix()) for f in base.rglob("*") if f.is_file())
    return sorted(names, key=lambda n: n != MANIFEST)


def file(run: str, node: str, name: str) -> Path:
    """One file of a delivery, by the name the record lists. Where a name may lead is decided in one place for the
    whole project (io.files.inside: resolved first, so a link out of the folder is out); a name that leads anywhere
    else is answered exactly like one that is not there, so an outsider learns nothing either way."""
    try:
        f = inside(folder(run, node), name)
    except Invalid:
        raise _gone() from None
    if not f.is_file():
        raise _gone()
    return f


# ------------------------------------------------------------------ archives


def _header(name: str, size: int, mtime: float) -> bytes:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.mode = size, int(mtime), 0o644
    return info.tobuf(tarfile.PAX_FORMAT, "utf-8", "surrogateescape")


def _entries(run: str, node: str, group: bool = False) -> list[tuple[str, Path]]:
    """What goes into an archive: (its name inside the tar, the file). One delivery's files, or, with `group`, every
    package of that 「输出」 node in that run, each under its own name (a browser download cannot be asked N times,
    so the N packages of a block's items go out as one package holding them)."""
    found: list[tuple[str, Path]] = []
    if not group:
        base = folder(run, node)
        found = [(n, base / n) for n in files(run, node)]
    else:
        for r in batch(run, node):
            base, under = folder(run, r["address"]), _ARCHIVE.sub("", r["name"]) or r["address"]
            found += [(f"{under}/{n}", base / n) for n in files(run, r["address"])]
    if not found:
        raise _gone()
    return found


def batch(run: str, node: str) -> list[dict]:
    """Every package one 「输出」 node delivered in one run, oldest first: one outside a block, one per item inside one
    (its record says which item). What a folder handle, a DCC client and `lab2shot cook` write side by side, and what
    a browser download takes as one package."""
    rows = db().rows("SELECT record, user_id FROM deliveries WHERE run = ? ORDER BY created", (run,))
    out = [with_state({**json_of(r["record"]), "user": r["user_id"]}) for r in rows]
    return [r for r in out if r.get("node") == node]


def archive_tag(run: str, node: str, gz: bool, group: bool = False) -> str:
    """What the archive's bytes are made of (its files' names, sizes and times, and the compression): the same tag is
    the same bytes, so a download cut off goes on from its byte (If-Range) and one whose files changed starts again."""
    return sha256(repr((gz, [(n, f.stat().st_size, f.stat().st_mtime_ns) for n, f in _entries(run, node, group)])))[:24]


def archive_size(run: str, node: str, gz: bool = False, group: bool = False) -> int:
    """The bytes archive_stream() sends: added up for the plain tar (a download can show its progress); for tar.gz,
    the compressed stream made once and counted (only a resumed download asks: kept per archive_tag)."""
    if gz:
        return _gz_size(archive_tag(run, node, True, group), run, node, group)
    total = 1024  # the two empty blocks that end a tar
    for name, f in _entries(run, node, group):
        st = f.stat()
        total += len(_header(name, st.st_size, st.st_mtime)) + st.st_size + (-st.st_size) % 512
    return total


@lru_cache(maxsize=16)
def _gz_size(tag: str, run: str, node: str, group: bool = False) -> int:
    return sum(len(c) for c in _gzip(_tar(_entries(run, node, group), 0)))


def archive_stream(run: str, node: str, gz: bool, done=None, start: int = 0, length: int | None = None,
                   group: bool = False) -> Iterator[bytes]:
    """A delivery as one tar (gzip-compressed for tar.gz), made while it is sent, from byte `start` and `length` bytes
    long (None: to its end); a download cut off goes on from where it got to (server/transfer.py, Range). The plain
    tar starts there directly (inside a file: read from that byte); the compressed one is made from its start again
    and its first bytes left out (the same files make the very same bytes). `done()` runs once the last byte has gone
    out (not when the receiver stops early, nor for a part that ends before the archive does). `group`: every package
    of that node in that run in one."""
    entries = _entries(run, node, group)
    chunks = _window(_gzip(_tar(entries, 0)), start, length) if gz else _window(_tar(entries, start), 0, length)
    return _then(run, chunks, done)


_STREAMING: dict[str, int] = {}  # run -> archives of it being sent this moment: prune() leaves such a run alone
_STREAMING_LOCK = threading.Lock()


def _then(run: str, chunks: Iterator[bytes], done) -> Iterator[bytes]:
    """`chunks`, counted in _STREAMING while they go out (from the first byte asked for to the generator's close: a
    receiver that stops early closes it too). Counted from the first `next`, not from archive_stream(): a generator
    never started never runs its `finally`, and a count left behind would keep the run forever."""
    with _STREAMING_LOCK:
        _STREAMING[run] = _STREAMING.get(run, 0) + 1
    try:
        yield from chunks
        if done:
            done()
    finally:
        with _STREAMING_LOCK:
            left = _STREAMING.get(run, 0) - 1
            if left > 0:
                _STREAMING[run] = left
            else:
                _STREAMING.pop(run, None)


def _tar(entries: list[tuple[str, Path]], skip: int) -> Iterator[bytes]:
    """The plain tar's bytes from byte `skip`: whole members before it are passed over without being read."""
    at = 0
    for name, path in entries:
        st = path.stat()
        header, size, pad = _header(name, st.st_size, st.st_mtime), st.st_size, (-st.st_size) % 512
        if at + len(header) + size + pad <= skip:
            at += len(header) + size + pad
            continue
        if skip < at + len(header):
            yield header[max(0, skip - at):]
        inside = max(0, skip - at - len(header))
        if inside < size:
            with path.open("rb") as f:
                f.seek(inside)
                while chunk := f.read(1 << 20):
                    yield chunk
        if pad > (padded := max(0, skip - at - len(header) - size)):
            yield b"\0" * (pad - padded)
        at += len(header) + size + pad
    yield b"\0" * (1024 - max(0, min(1024, skip - at)))


def _gzip(chunks: Iterator[bytes]) -> Iterator[bytes]:
    z = zlib.compressobj(6, zlib.DEFLATED, 31)  # wbits 31: a gzip stream
    for chunk in chunks:
        if piece := z.compress(chunk):
            yield piece
    yield z.flush()


def _window(chunks: Iterator[bytes], drop: int, length: int | None) -> Iterator[bytes]:
    """`chunks` without their first `drop` bytes, `length` bytes long (None: all the rest)."""
    for chunk in chunks:
        if drop:
            if len(chunk) <= drop:
                drop -= len(chunk)
                continue
            chunk, drop = chunk[drop:], 0
        if length is not None:
            if len(chunk) >= length:
                if length:
                    yield chunk[:length]
                return
            length -= len(chunk)
        if chunk:
            yield chunk


def prune() -> None:
    """Remove the deliveries of runs whose newest delivery is older than the days they are kept (their records with
    them). A run with an archive going out this moment (_STREAMING) is left for the next pass: its files removed under
    the stream would cut the download short."""
    cutoff = time.time() - keep_days() * 86400
    newest = {r["run"]: r["made"] for r in db().rows("SELECT run, MAX(created) AS made FROM deliveries GROUP BY run")}
    runs = {p.name: p for p in root().iterdir() if p.is_dir()} if root().is_dir() else {}
    with _STREAMING_LOCK:
        busy = set(_STREAMING)
    gone = [run for run in {*newest, *runs} - busy
            if newest.get(run, runs[run].stat().st_mtime if run in runs else 0) < cutoff]
    if not gone:
        return
    with db().write() as c:  # the records first: a delivery never shows files that are going
        c.executemany("DELETE FROM deliveries WHERE run = ?", [(run,) for run in gone])
    for run in gone:
        if run in runs:
            shutil.rmtree(runs[run], ignore_errors=True)
