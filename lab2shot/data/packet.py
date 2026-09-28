"""Data packets: the only thing that flows between nodes.

A packet is a folder in its account's cache (data/store.py: <数据位置>/cache/<account id>/) named by its fingerprint:

    <cache>/<fingerprint>/
        manifest.json     type, producing node, metadata (frames, colorspace, ...)
        ...               payload in a standard format (PNG/EXR, JSON, USD)
        .complete         written last; a packet without it is garbage

Packets are immutable once complete, so any node (in any environment) can
read them, and a finished cook is never repeated.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import logs
from ..errors import Invalid
from ..messages import Msg
from ..io.atomic import flush_tree, mark
from ..serving import uses
from .store import current

MANIFEST = "manifest.json"
COMPLETE = ".complete"
DEPS = "deps"  # the dependency list in the manifest
log = logs.get("cache")
# In every cache key (a node's fingerprint, engine/evaluation.py; a worker job's key, engine/external.py): raised when
# what a cache entry holds changes shape, so an entry of another shape is never found (no reading of other formats).
# The shape at this version: packets name their files relative to their folder (file_ref), and EXR channel names
# follow the channel count as R G B A (plus valid; read_map reads them by those names).
CACHE_VERSION = 3
# Version of the fact fields: raised when the shape of the facts core code writes into result meta changes (a camera's
# backplate, lens, a point cloud's cloud conditions, ...). It enters every node's fingerprint (the blob in
# engine/evaluation.py _plan), so old packets are recooked rather than read as valid.
# At 0 it is left out of the fingerprint; raising it (to 1, then on) recooks everything once.
# The places that write facts (nodes/kit/cameras.py, data/camera.py write, the meta fields of data/payloads.py) must raise it
# together with any change to them.
FACTS_VERSION = 0


def cache_root() -> Path:
    """The cache of the account whose work is being done now (data/store.py): each account's own."""
    return current().cache


def base_of(name: str) -> str:
    """The entry a cache folder belongs to: a packet's fingerprint for the packet and what hangs off it (SIBLINGS:
    `<fp>_display`, `<fp>_work`, `<fp>_failed`), a worker job's folder (`<key>_job`) for itself. What a task references
    and what cleaning keeps are named this way (farm/disk.py)."""
    for suffix in SIBLINGS:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def note(name: str) -> None:
    """The job being done now computed or reused this cache entry (lab2shot/serving.py Uses; nothing outside a job):
    its task references it from now on, and it is not cleaned while the job runs. Called where an entry is made or
    used: a packet committed or used (Packet.commit, Packet.used, produce), a node's cook beginning (its `_work`,
    `_failed`), a worker job run or reused (engine/external.py)."""
    into = uses()
    if into is not None:
        into.add(base_of(name))


def _manifest_state(directory: Path) -> str:
    """"ok" (a manifest is a JSON object, so its first byte is "{"), "bad" (present but empty, or filled with what a
    crash left, such as NUL bytes) or "absent"."""
    try:
        with open(directory / MANIFEST, "rb") as f:
            return "ok" if f.read(1) == b"{" else "bad"
    except FileNotFoundError:
        return "absent"
    except OSError:
        return "bad"


WORKER_COMPLETE = ".worker_complete"  # in a worker job's raw folder, written after its result.json


def worker_done(raw: Path) -> bool:
    """A worker job's raw folder holds a finished result: the marker, and a result.json with content (a marker left by
    a system crash without its data is not a result)."""
    try:
        return (raw / WORKER_COMPLETE).exists() and (raw / "result.json").stat().st_size > 0
    except OSError:
        return False


def cooked_here(path: Path | str) -> bool:
    """Whether this file was written by a cook of this server (a file of a packet in the results store), as opposed to
    something that came in from outside. What a packet's file may then draw on follows from it (transfer/uploads.py may_draw_on)."""
    root = cache_root().resolve()
    p = Path(path).resolve()
    return root in p.parents


@dataclass
class Packet:
    dir: Path
    type: str
    meta: dict[str, Any] = field(default_factory=dict)
    node: str = ""  # the type of node that made it (known once committed)
    messages: list[dict] = field(default_factory=list)  # what that node said while making it (commit)
    # when it was committed: the page's generation of this fingerprint. Each recook of the same fingerprint gives a new
    # `created`; the page's cache key includes it, so old frames no longer match, and after the server removes or
    # recooks a packet the page learns of it from the next status reply
    created: str = ""

    @property
    def fingerprint(self) -> str:
        return self.dir.name

    def path(self, *parts: str) -> Path:
        return self.dir.joinpath(*parts)

    # ------------------------------------------------------------------ io

    @classmethod
    def load(cls, directory: Path) -> Packet:
        m = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
        return cls(directory, m["type"], m.get("meta", {}), m.get("node", ""), m.get("messages", []), m.get("created", ""))

    @staticmethod
    def exists(directory: Path) -> bool:
        """Complete, with a manifest that has content: a marker left by a system crash without its data (an empty
        manifest) is not a packet, so it is cooked again rather than read."""
        return (directory / COMPLETE).exists() and _manifest_state(directory) == "ok"

    @staticmethod
    def damaged(directory: Path) -> bool:
        """Marked complete, with a manifest that is empty or unreadable (a system crash before the data reached the
        disk): not a packet. A folder whose manifest is not there at all is being removed or rewritten, not damaged."""
        return (directory / COMPLETE).exists() and _manifest_state(directory) == "bad"

    @staticmethod
    def used(directory: Path) -> None:
        """Note that a cached packet was used now (its .complete file's time): cleaning keeps what is in use, and so
        are the packets it names (its deps' `packets`: a list's items, a camera's backplate), each in turn. Without
        that, the items of a list nobody opens on their own would be the least recently used and go first, leaving a
        list in daily use invalid."""
        _touch(directory, set())

    def commit(self, node_type: str, messages: list[dict] = ()) -> Packet:
        """Write the manifest and mark the packet complete. `messages`: what the node said while it made it
        (lab2shot/messages, as Msg.json() with their anchors), kept with the result so its marks outlive the cook.

        The dependency list is written into the manifest as well (`deps`): files outside the packet that it reads
        (uploads: path, size, modification time) and other packets it references (list items, a camera's backplate,
        files living in other packets). Writers need not declare them: they are derived from the meta (`deps_of`), so
        packets written through side paths (image_packet, items_meta, scene_packet, ...) are covered too.
        Validity is judged from this list alone (`check`)."""
        manifest = {
            "type": self.type,
            "node": node_type,
            "created": datetime.now().isoformat(timespec="seconds"),
            "meta": self.meta,
            DEPS: deps_of(self.dir, self.meta),
            **({"messages": list(messages)} if messages else {}),
        }
        (self.dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        flush_tree(self.dir)  # the payload and the manifest are on the disk before the marker (io/atomic.py)
        mark(self.dir / COMPLETE)
        # temporary display images and proxies written into the packet during progressive viewing (`_partial`,
        # server/farm.py partial_frame) are obsolete once the packet is complete; kept, they would waste cache space
        # and be copied by copy_packet
        shutil.rmtree(self.dir / "_partial", ignore_errors=True)
        self.node = node_type
        self.created = manifest["created"]
        note(self.dir.name)
        return self


def _touch(directory: Path, seen: set[str]) -> None:
    used(directory / COMPLETE)
    note(directory.name)
    seen.add(directory.name)
    try:
        deps = json.loads((directory / MANIFEST).read_text(encoding="utf-8")).get(DEPS) or {}
    except (OSError, ValueError):
        return
    for fp in deps.get("packets") or ():
        if fp not in seen and len(seen) < 10_000:
            try:
                _touch(packet_dir(fp), seen)
            except Invalid:
                pass


def file_ref(directory: Path, path: Path) -> str:
    """A file's place in a packet's meta: relative to the packet's folder (it may reach outside with `..`, e.g. an
    uploaded plate the packet reads without holding it). The whole work folder moves together, so the relative path
    stays valid."""
    return os.path.relpath(path, directory)


def file_path(p: Packet, ref: str) -> Path:
    """A file a packet's meta names (file_ref), at its place."""
    return _target(p.dir, ref)


def _target(directory: Path, ref: str) -> Path:
    """Where a meta reference points: relative to the packet's folder (file_ref). An absolute path is not a reference
    file_ref writes: ValueError, rather than a path taken as it is."""
    if os.path.isabs(ref):
        raise ValueError(f"a packet names its files relative to its folder (file_ref), not {ref!r}")
    return Path(os.path.normpath(directory / ref))


def deps_of(directory: Path, meta: dict, record: bool = True) -> dict:
    """What a packet depends on outside its own folder, from its meta:
    `files`: every file it reads that is not its own (an upload's frames, a video, a file of another packet named
    directly), each with its size and modification time when `record` (at commit; a later check compares them);
    `packets`: the other packets it cannot be used without: a list's items, the packet a named file lives in;
    `soft`: packets it only points at for the viewer's convenience (a camera's plate, its backdrop); when they are gone
    the packet is still valid (the 3D view says 「没有背板」), so `check` does not look at them. A solved camera must
    not be discarded, and re-solved on the GPU, because the plate's upload was cleaned."""
    files: list[dict] = []
    packets: set[str] = set()
    root = cache_root()

    def note(ref) -> None:
        if not isinstance(ref, str) or not ref:
            return
        target = _target(directory, ref)
        if target == directory or directory in target.parents:
            return  # its own file
        try:
            inside = target.relative_to(root)
        except ValueError:
            inside = None
        if inside is not None and inside.parts and inside.parts[0] != directory.name:
            packets.add(inside.parts[0])  # a file of another packet: that packet is the dependency
            return
        entry: dict = {"ref": ref}
        if record:
            try:
                st = target.stat()
                entry.update({"size": st.st_size, "mtime": int(st.st_mtime)})
            except OSError:
                pass  # not there even now: `check` will say so
        files.append(entry)

    # `files`: {frame: path} for picture / map packets; a list (each entry a string or {"path": ...}) for delivery
    # packets. Both forms are accepted; only the strings are used
    listed = meta.get("files") or {}
    refs = listed.values() if isinstance(listed, dict) else (listed if isinstance(listed, list) else [])
    for ref in refs:
        note(ref.get("path") if isinstance(ref, dict) else ref)
    note(meta.get("path"))
    for item in meta.get("items") or ():
        if isinstance(item, dict) and item.get("packet"):
            packets.add(str(item["packet"]))
    soft = [str(meta["plate"])] if meta.get("plate") else []
    return {"files": files, "packets": sorted(packets - {directory.name}), "soft": soft}


@dataclass(frozen=True)
class Verdict:
    """Whether a cached packet may be used. `why`: "" (ok), "missing" (not there, or not
    complete), "deps" (its manifest has no dependency list: not written by Packet.commit), "source" (a file it reads
    outside its folder is gone or changed), "packet" (a packet it names is gone, i.e. its folder does not exist; one
    present without `.complete` is being written, not gone). `what`: the reference that failed."""

    ok: bool
    why: str = ""
    what: str = ""


def check(directory: Path) -> Verdict:
    """THE one judgement of a cached packet's validity: complete, every outside file still there with the size and
    time recorded at commit, every named packet still complete."""
    if not (directory / COMPLETE).exists():
        return Verdict(False, "missing")
    try:
        manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Verdict(False, "missing")
    deps = manifest.get(DEPS) if isinstance(manifest, dict) else None
    if not isinstance(deps, dict):
        return Verdict(False, "deps")
    for f in deps.get("files") or ():
        ref = f.get("ref", "")
        try:
            st = _target(directory, ref).stat()
        except (OSError, ValueError):
            return Verdict(False, "source", ref)
        if "size" in f and (st.st_size != f["size"] or int(st.st_mtime) != f["mtime"]):
            return Verdict(False, "source", ref)
    for fp in deps.get("packets") or ():
        # a dependency counts as gone only when its whole folder is gone. A folder present without `.complete` is being
        # rewritten (fresh_dir removes and recreates it, leaving no .complete for minutes); judging that invalid would
        # make every /api/status call NodePlan.has -> discard_invalid -> remove and delete downstream packets while the
        # dependency is merely recooking. This must agree with discard_invalid ("an entry that is merely not there is
        # left alone")
        # A folder marked complete without its data (a system crash) is gone as well: it is never read again
        try:
            d = packet_dir(fp)
            if not d.exists() or Packet.damaged(d):
                return Verdict(False, "packet", fp)
        except Invalid:
            return Verdict(False, "packet", fp)
    return Verdict(True)


def valid(directory: Path) -> bool:
    """`check(directory).ok`: for callers that only want yes or no."""
    return check(directory).ok


# ------------------------------------------------------------------ removing (one entry point for every deletion)

# what hangs off a fingerprint by name and goes with it. `_display` hangs off the OUTPUT packet's fingerprint
# (view/frames.py, payloads.frames_for_worker); `_work` and `_failed` off the NODE's (engine/cook.py, engine/evaluation.py).
# A packet's folder is named by its own fingerprint, so `remove(<packet fp>)` takes its `_display`, and `remove(<node fp>)`
# takes that node's scratch and failure record.
SIBLINGS = ("_display", "_work", "_failed")
_on_removed: list[Callable[[str, str], None]] = []


def on_removed(hook: Callable[[str, str], None]) -> None:
    """Register what must happen whenever a packet is removed: hook(fingerprint, why). The data layer imports nothing
    above it, so the layers above register themselves (one hook: engine/evaluations.py bumps the evaluations'
    generation, so plans and status read before the removal are not handed out again)."""
    _on_removed.append(hook)


def remove(name: str, why: str) -> bool:
    """Remove a cache entry and, for a packet, everything that hangs off it (SIBLINGS); tell the hooks. `name` is a
    packet fingerprint, or one entry's folder name (`<fp>_display`, `<key>_job`) to remove that entry alone. `why`:
    who asked: "clean", "upload" (its upload is gone), "recook", "incomplete", "invalid:<why>".
    Every path that deletes a cache entry comes through here (farm/disk.py, transfer/uploads.py
    remove_set through remove_referring, discard_invalid, fresh_dir). A node's own scratch is not an entry:
    engine/cook.py clears its `<node fp>_work` once the cook is done, and engine/external.py its raw worker folder, directly."""
    d = packet_dir(name)
    plain = not any(name.endswith(s) for s in (*SIBLINGS, "_job"))
    folders = [d, *(d.with_name(name + s) for s in SIBLINGS)] if plain else [d]
    removed = False
    for folder in folders:
        if folder.exists():
            # the markers go first: while the rest is being deleted the folder reads as incomplete, never as a
            # complete entry with its manifest missing (check() on the packets that depend on it)
            for marker in (folder / COMPLETE, folder / "raw" / WORKER_COMPLETE):
                try:
                    marker.unlink(missing_ok=True)
                except OSError:
                    pass
            shutil.rmtree(folder, ignore_errors=True)
            removed = True
    if removed:
        logs.say(log, Msg("I-CACHE-REMOVED", fp=name, why=why))
        for hook in list(_on_removed):
            hook(name, why)
    return removed


def discard_invalid(directory: Path) -> Verdict:
    """`check`, and a packet that fails on its sources is removed on the spot (it can only be cooked again): the one
    place a read finds and clears a stale entry. An entry that is merely not there is left alone: it may be being
    written at this moment."""
    verdict = check(directory)
    if not verdict.ok and verdict.why != "missing":
        remove(directory.name, f"invalid:{verdict.why}")
    return verdict


def remove_referring(match: Callable[[str], bool], why: str) -> int:
    """Remove every cached packet whose deps name a file `match` accepts (an upload set being deleted:
    `lambda ref: "/sets/<account>/<sid>/" in ref`). One manifest read per cached packet; deletion is rare."""
    dropped = 0
    root = cache_root()
    for manifest in sorted(root.glob("*/manifest.json")) if root.is_dir() else []:
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        deps = m.get(DEPS) or {}
        if any(match(str(f.get("ref", "")).replace("\\", "/")) for f in deps.get("files") or ()):
            if remove(manifest.parent.name, why):
                dropped += 1
    return dropped


def used(marker: Path) -> None:
    """Set a marker file's time to now (the cache's "last used")."""
    try:
        os.utime(marker)
    except OSError:
        pass


def packet_dir(fingerprint: str) -> Path:
    """A fingerprint names a cache folder and nothing else: letters, digits and _ (anything else, from a page or a
    client, is refused)."""
    if not fingerprint.replace("_", "").isalnum() or not fingerprint.isascii():
        raise Invalid(Msg("E-PACKET-FINGERPRINT", value=fingerprint[:40]))
    return cache_root() / fingerprint


def fresh_dir(fingerprint: str) -> Path:
    """Empty working folder for a packet about to be produced. What was there before goes through `remove`, so a
    recook also drops the display copy made from the old pixels (`<fp>_display`): without that a recook that changes
    the pixels would leave the viewer and the workers reading the stale copy.

    Called only under the entry's lock: by a cook on its outputs (engine/cook.py Outputs) and by `produce`, nowhere
    else (enforced by lab2shot check). Two producers of one entry without the lock: the second one's `fresh_dir` removes
    the first one's half-written folder from under it. A complete packet removed here (a forced recook) and its
    `_display` are read by no cook meanwhile: a cook holds the lock of every packet it reads shared (engine/cook.py
    _compute), so the recook waits for it."""
    d = packet_dir(fingerprint)
    if d.exists():
        remove(fingerprint, "recook")
    d.mkdir(parents=True)
    return d


def produce(fingerprint: str, make: Callable[[Path], Packet]) -> Packet:
    """The single way a packet enters the cache outside a node's own cook: a list's items (data/items.py, nodes/core/flow.py),
    a value (data/values.py), the display copy a worker reads (data/payloads.py frames_for_worker), a read node's entries
    (nodes/core/input.py). Under the entry's lock (the same one a cook holds on its outputs, data/locks.py) the first
    to arrive makes it into a fresh folder and commits it; anyone else, on this GPU or another, waits and loads what is
    there. `make(folder)` writes the packet into the folder it is given and returns it committed. A valid entry already
    there is used as it is (`valid`: its sources unchanged, data/packet.py check)."""
    from .locks import exclusive

    d = packet_dir(fingerprint)
    if valid(d):
        Packet.used(d)
        return Packet.load(d)
    with exclusive(fingerprint):
        if valid(d):
            Packet.used(d)
            return Packet.load(d)
        made = make(fresh_dir(fingerprint))
        if not Packet.exists(d):
            raise RuntimeError(f"produce({fingerprint}): make() returned without committing the packet")
        return made


# ------------------------------------------------------------------ a list of data
# A list packet holds no data of its own: its manifest names its items in order, each of them an ordinary packet with
# its own fingerprint and its own place in the cache. So a list of 100 sequences costs one small file, an item is
# cached and cooked on its own (「逐项开始」 hands its packet on as it is), and two lists made of the same items share
# them. A name is part of an item: two items of one list never share one (items_meta refuses it), and the
# name is what the artist sees, what a scene's group is called and what a delivery is named after.


def items_meta(items: list[tuple[str, Packet]] | list[tuple[str, str]]) -> dict:
    """The manifest of a list: `items`, (name, its packet's fingerprint) in order. Two items of one name are refused
    here, at the node that made them (data/items.py says it the same way)."""
    from ..errors import Invalid
    from ..messages import Msg
    from .types import DATA_TYPES, element_of

    named = [(str(name), p if isinstance(p, str) else p.fingerprint) for name, p in items]
    same = sorted({n for n, _ in named if [x for x, _ in named].count(n) > 1})
    if same:
        kind = DATA_TYPES.get(element_of(_items_type(items)))
        raise Invalid(Msg("B-NAME-SAME", kind=kind.label if kind else "条目", names=same))
    return {"items": [{"name": n, "packet": fp} for n, fp in named]}


def _items_type(items) -> str:
    return next((p.type for _, p in items if not isinstance(p, str)), "")


def items_of(p: Packet) -> list[tuple[str, str]]:
    """A list packet's items: (name, its packet's fingerprint), in order."""
    return [(i["name"], i["packet"]) for i in p.meta.get("items") or ()]


def item_fingerprint(*parts: Any) -> str:
    """Where one item of a list a node makes lives: named by what it holds alone (the node type, its settings and that
    item's own source), never by the whole list, so one more sequence in the folder leaves the items already there
    exactly where they were, and nothing above them cooks again."""
    from ..io.digest import key

    return key(["item", *parts], 24)


def copy_packet(src: Packet, out: Path) -> Packet:
    """The same data under another fingerprint (「取一条」, 「命名」): the packet's own files copied, its description as
    it is. Files it only points at (an upload a sequence reads, another packet's frames) are named relative to the
    packet folder and every packet folder is a sibling of every other, so they still point at the same bytes."""
    for entry in sorted(src.dir.iterdir()):
        if entry.name in (MANIFEST, COMPLETE):
            continue
        if entry.is_dir():
            shutil.copytree(entry, out / entry.name, dirs_exist_ok=True)
        else:
            shutil.copyfile(entry, out / entry.name)
    return Packet(out, src.type, dict(src.meta))
