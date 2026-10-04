"""Cooking: fingerprint every node, reuse cached packets, run the rest in order.

fingerprint(instance) = hash(cache / facts versions, type id, version, result-affecting params, fingerprints of its
                             input packets, source identity, and where they apply: the extension's result identity, the
                             frames a frame source emits, the label (what names its output), the route a switch takes,
                             the wires it goes without, a begin's item, the items an end gathers) — Evaluation._plan
A fingerprint names the content, not one write of it: a forced recook writes the same fingerprint anew, and what was
made from the earlier one says so (data/packet.py made_from).
An output port's packet lives in its account's cache (data/store.py) at <hash(fingerprint, port)>. A parameter driven
by a wire (a promoted parameter, Graph.input_ports) is an input like any other: the value packet wired into it is among
the input packets, and the node is cooked with its value in place of the typed one (Engine.wired_values).

A graph's frame range (Graph.frames) selects the frames its sources emit (NodeDef.frame_source), each source keeping
those of its own frames that fall inside; everything downstream then computes only those. The frames are part of a
source's fingerprint, so the whole shot and a part of it are cached side by side, and a range that covers all of a
source's frames is the same cook as no range.

Several cooks may run at once (from different users), and within one cook several nodes: a node is cooked under a
lock on its fingerprint, so two cooks that need the same result compute it once and neither sees it half-written.

What is cooked is node instances (engine/scopes.py Inst: a node and its item path inside 逐项处理 blocks; the empty
path outside them). A cook works through Evaluation.order: every instance whose inputs are through is started at once,
on the place its cost needs (a GPU or a CPU slot, asked of the resources the farm hands in: engine/resources.py), and
the order is asked again whenever one is through, so what only a cook tells (a block's items, a switch's condition, a
wired value) is planned once it is known. An instance gives the outputs something wants in this cook
(CookContext.wanted): those wired into what is cooked, and the ones shown of the target; an output already there is
never written again, one no one wants is not written at all.

The read side of this (fingerprints, cached outputs, parameters, status) is engine/evaluation.py's `Evaluation`,
memoised per graph version; `Engine` here holds one and cooks on top of it (see Evaluation's docstring)."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import i18n
from ..data.items import holds_nothing
from ..data.payloads import is_data
from ..data.types import channels_of, is_list
from ..data.locks import exclusive, shared
from ..data.packet import Packet, created_of, fresh_dir, note, packet_dir
from ..errors import CookCancelled, CookError, GraphError, MessageError, NothingToCook, message_of
from ..messages import Both, Msg, msg_word, word_of
from ..nodes.applies import Fact, NodeFacts, resolve, standing_notices
from ..nodes.base import NodeDef
from ..nodes.params import param_defaults
from ..nodes.services import OutputSink
from ..serving import carried, serving
from . import external
from . import scopes as sc
from .demand import FAIL, RETRY, SKIP
from .evaluation import PLAN_ERRORS, Evaluation, NodePlan, failure_file
from .records import changed
from .graph import Graph, data_kind_word as _kind_word, insert_fix, walk
from .presence import present
from .resources import CPU, GPU, Need, Resources, Ticket
from .scopes import Inst, ItemAt, ItemPath

# frames computed in parallel by a per-frame core node (CookContext.each_done: each frame read, computed and written on
# its own thread): 8, as these kernels are bound by memory bandwidth, not compute; measured on a 32-core machine
# (1080p, an ST-map warp) they stop gaining past 8 and regress from 16, and more frames in flight hold more of a 4K
# shot in memory at once
FRAME_THREADS = max(1, min(8, os.cpu_count() or 1))

EventFn = Callable[[dict], None]
WAIT_S = 0.2  # how often a cook waiting for its nodes looks whether it was stopped


def _first(packets: list[Packet] | None) -> Packet | None:
    """The first packet of an input (its picture input's plate, the map a warp follows); None when nothing is there."""
    return (packets or [None])[0]


def _readies(node_type: type[NodeDef]) -> bool:
    """The node type has something to do before it is planned (NodeDef.ready)."""
    return getattr(node_type.ready, "__func__", None) is not NodeDef.ready.__func__


def _records_provenance(node_type: type[NodeDef]) -> bool:
    """The nodes that read CookContext.provenance: the output settings (the sidecar next to their files) and
    「输出」 (its W-COOK-NONCOMMERCIAL)."""
    from ..nodes.output import OutputSettings

    return node_type.delivers or issubclass(node_type, OutputSettings)


def _present(plan: NodePlan) -> dict[str, str]:
    """The node's outputs that really have a packet now (an output nobody wanted was never written): what a node_done
    event hands the page, the same answer as the status reply's `present` (Evaluation.status)."""
    return {port: fp for port, fp in plan.outputs.items() if present(fp)}


def _stopped(stop: threading.Event) -> None:
    """Raise once `stop` is set (a cook, or the node it is cooking, should stop as soon as possible)."""
    if stop.is_set():
        raise CookCancelled


class Outputs(dict):
    """The folders a node fills, port -> an empty folder: those this cook wants it to give (CookContext.wanted) are
    there from the start, so `"normal" in ctx.outputs` answers "is this one wanted this time"; a node that asks for
    another one anyway gets its folder made then and there (its packet is written at the same fingerprint as ever,
    so nothing downstream can tell the difference; it is merely work nobody asked for).

    An unwanted output that is already complete on disk is kept, not rewritten (`kept`): the node gets a scratch
    folder beside it (`<node fp>_work/_unwanted/<port>`) and what it writes there is thrown away with the scratch.
    This matters because some nodes write every port on every cook (the camera nodes' curves / errors): making a fresh
    folder for such a port would delete a complete packet on disk while another reader may be halfway through it."""

    def __init__(self, fingerprints: dict[str, str], wanted, scratch: Path, force: bool = False,
                 hold: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self._fingerprints = fingerprints
        self._scratch = scratch
        self._force = force
        self._hold = hold  # locks a packet's folder before it is made (the wanted ones are locked before the cook starts)
        self.kept: set[str] = set()  # ports whose complete packet stays as it is: not committed by _run_node
        for port in sorted(wanted):
            if port in fingerprints:
                self[port] = self._folder(port)

    def _folder(self, port: str) -> Path:
        """A fresh folder to write the packet into, or, when a valid packet of it is on disk and this is not a forced
        recook, a scratch folder beside it and the packet stays (`kept`). A wanted port is only ever valid on disk
        when it is wanted as another output's source (Port.made_from, _to_give): the node computes it again in the
        process to make the other one, but nobody needs the bytes written again, and rewriting them would delete a
        complete packet (a depth map, say) under whoever is reading it."""
        if not self._force and present(self._fingerprints[port]):
            self.kept.add(port)
            folder = self._scratch / "_unwanted" / port
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        return fresh_dir(self._fingerprints[port])

    def __missing__(self, port: str) -> Path:
        if port not in self._fingerprints:
            raise KeyError(port)
        if self._hold is not None:
            self._hold(self._fingerprints[port])
        self[port] = folder = self._folder(port)
        return folder


def results_by_item(results: list[tuple[str, Packet | None]]) -> tuple[dict[str, list[Packet]], set[str]]:
    """A block's end: the results it takes, each (the key of the item it came from, its packet; None when that item
    gave nothing on that wire), paired by item: (item key -> its packets in wire order, the items that gave nothing on
    some wire). An item that gave nothing on one wire is left out whole, like an item that failed on one
    (Evaluation.gathered): never a result of one item put with another's."""
    by_item: dict[str, list[Packet]] = {}
    nothing: set[str] = set()
    for key, packet in results:
        if packet is None:
            nothing.add(key)
        else:
            by_item.setdefault(key, []).append(packet)
    return {k: v for k, v in by_item.items() if k not in nothing}, nothing


def _both(make) -> dict[str, dict[str, str]]:
    """{key: {lang: text}} from `make()` ({key: text}) said in each language (messages.Both)."""
    out: dict[str, dict[str, str]] = {}
    for lang in i18n.LANGS:
        with i18n.using(lang):
            for k, v in make().items():
                out.setdefault(k, {})[lang] = str(v)
    return out


@dataclass
class CookContext:
    """What a node's cook() gets: its own values, worked out by the engine, never the graph (a node knows its inputs,
    not who made them)."""

    node_id: str
    label: str
    node_type: type[NodeDef]
    params: dict[str, Any]
    inputs: dict[str, list[Packet]]
    outputs: dict[str, Path]  # port -> empty folder to fill
    output_types: dict[str, str]  # port -> the type its packet has (resolved through the graph: type_from, alternatives)
    work: Path  # scratch folder for this cook (logs, raw results)
    emit: EventFn
    gpu: str | None = None  # the GPU workers run on (CUDA_VISIBLE_DEVICES); "" none; None: as this process sees them
    stop: threading.Event = field(default_factory=threading.Event)  # set: stop as soon as possible
    frames: tuple[int, ...] = ()  # a frame source: the frames to emit (its own, within the cook's frame range)
    waited: float = 0.0  # seconds spent waiting for another cook (not computing): kept out of the node's time
    reused: bool = False  # it took a worker's raw results computed before: its time says nothing of the work
    # parameters driven by a wire -> the value as wired (nodes/values.py Value, in the parameter's unit): a parameter
    # that takes one per frame (P(per_frame)) reads its frames here; params holds the value for the whole shot
    values: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)  # parameter -> where its value came from, as the node says it (Engine.sources)
    # parameter -> the label of the node driving it with a wire, for a node that has to decide something by it
    # (「LensDistortion」: a lens handed to it is an estimate, one typed on it is a measured lens sheet)
    wired_from: dict = field(default_factory=dict)
    # which third-party projects made its inputs, whether all allow commercial use, and where the values of the nodes
    # above came from (Evaluation.provenance): an output-settings node records it next to its files, 「输出」 warns by
    # it. Worked out for those alone (it walks everything above: once per node of a long chain it would be N²); None
    # for any other node
    provenance: dict | None = None
    # per input, per wire: where what it brings came from (Evaluation.lineage), for a node that asks (NodeDef.reads_lineage)
    lineage: dict = field(default_factory=dict)
    ram_gb: float = 0.0  # the system memory its worker takes at its peak (its resolved Cost): kept free for it
    # a node that steps down on a smaller card (Cost.vram_full_gb): the VRAM its tier asked the card for
    # (Evaluation.vram_need), sent to its worker as `vram_budget_gb` (external.job_params). The worker picks its
    # step by this, never by what the card happens to have free, so a result is the tier its fingerprint says
    # (Evaluation.full_tier). 0: the node does not step down
    vram_budget_gb: float = 0.0
    collector: OutputSink | None = None  # where 「输出」 collects and packs its files: only a node that delivers gets one
    stream_worker: Callable[..., tuple] | None = None  # runs a streaming node's worker on a farm thread (Engine passes it)
    said: list[dict] = field(default_factory=list)  # the messages it said while cooking (say), kept with its result
    facts: dict = field(default_factory=dict)  # what it found out about itself while cooking (fact): name -> value
    # what this cook is known by (NodePlan.fingerprint): every output of one cook shares it (an import's folder is marked
    # with it, so its camera and its models land in one folder when packed)
    fingerprint: str = ""
    path: ItemPath = ()  # the instance's item path (engine/scopes.py): empty outside every 逐项处理 block
    # what it gives that holds nothing is expected (every input that came empty was an expected empty, a block's end
    # whose every item gave nothing as expected): its outputs say so, and what reads them is told nothing (_context)
    expected: bool = False
    # the outputs to give this time (their folders are in `outputs`): wanted downstream in this cook or shown, and not
    # there yet. A node gives exactly these; the others it neither computes nor writes
    wanted: frozenset[str] = frozenset()
    item: ItemAt | None = None  # a block's begin: the item this instance gives
    names: tuple[str, ...] = ()  # the names of the items of its path, outer block first (() outside every block)
    items: tuple[ItemAt, ...] = ()  # a block's end: the items it gathers, in the block's order
    # a block's end: each gathered item's results, one per wire into it in wire order, by the item's key (paired by
    # the engine from where each packet came from: results_by_item)
    item_results: dict[str, tuple[Packet, ...]] = field(default_factory=dict)
    # the inputs wired in the graph that bring nothing this time on purpose (Evaluation.quiet_inputs: behind a 「阻断」
    # set to block, or from a reader with no file): a node that tells of a row it has nothing for (多层 EXR 输出设置)
    # leaves these out silently
    quiet: frozenset[str] = frozenset()

    @contextmanager
    def exclusive(self, key: str, waiting: str) -> Iterator[None]:
        """Hold the lock on a cache entry (locks.exclusive); while another cook holds it the node says `waiting`,
        and the time waited is not counted as computing."""
        start = time.time()
        with exclusive(key, lambda: self.stage(waiting), self.check_stop):
            self.waited += time.time() - start
            yield

    def input(self, port: str) -> Packet | None:
        packets = self.inputs.get(port) or []
        return packets[0] if packets else None

    def run_worker(self, image: Packet | None, *, extra: dict | None = None, inputs: dict | None = None,
                   reuse: bool = True, params: dict | None = None, record: dict | None = None) -> Path:
        """Run the node's worker in its extension's environment; returns the raw folder it wrote into (the job and
        its reuse: engine/external.py run_worker). `image`: the frames to process (None for a node that reads files
        only); `extra`: values the node computes, on top of its worker parameters; `inputs`: other files it reads;
        `params`: exactly these instead of its worker parameters, for a smaller job besides the main one, cached by
        what it needs alone (Kimodo's text encoding); `record`: where a value the node worked out came from (its
        lens), kept in the job file."""
        return external.run_worker(self, image, extra=extra, inputs=inputs, reuse=reuse, params=params, record=record)

    def run_worker_streaming(self, image: Packet | None, *, extra: dict | None = None, inputs: dict | None = None,
                             record: dict | None = None) -> tuple[Path, Any, Any]:
        """A streaming node's worker (WorkerNode.streams) on a farm thread, so its convert() can follow the frames as
        they appear: returns (raw folder, done event, error(), halt()), using the farm's runner (Engine.stream_worker,
        farm/streaming.py)."""
        return self.stream_worker(self, image, extra=extra, inputs=inputs, record=record)

    def input_files(self, *ports: str) -> dict[str, Path]:
        """Worker inputs from these connected ports, keyed by port name (engine/external.py input_files)."""
        return external.input_files(self, *ports)

    def check_stop(self) -> None:
        _stopped(self.stop)

    def progress(self, done: int, total: int, message: "str | Msg | dict" = "", word: dict | None = None) -> None:
        """Report progress; a cook that was stopped ends here (every node that reports progress can be stopped).
        `word`: what it counts kept as a word (messages.word_of), said in each follower's language (messages.localized).
        `message` may itself be a Msg or a kept word: it is then said in each follower's language too."""
        from ..messages import msg_word, said_word

        if isinstance(message, Msg):
            word, message = word or msg_word(message), message.text
        elif isinstance(message, dict):
            word, message = word or message, said_word(message) or ""
        self.check_stop()
        self.emit({"type": "progress", "node": self.node_id, "path": list(self.path), "done": done, "total": total, "message": message,
                   **({"message_word": word} if word else {})})

    def each(self, items, message: str = ""):
        """Iterate and report progress after every item."""
        items = list(items)
        for i, item in enumerate(items):
            yield item
            self.progress(i + 1, len(items), message)

    def each_done(self, items, work, message: str = "", most: int = FRAME_THREADS):
        """`work(item)` for every item, several at a time, yielding the results in `items` order. This is the engine's
        one place for per-frame parallelism: a node whose frames are independent (reading a frame, computing it,
        writing it through its ExrWriter, whose `add` is thread-safe) says what one frame is and hands the frames here;
        it never deals with threads, ordering, progress or stopping itself. The numbers are the one-at-a-time ones: the
        kernels are pure, so this only makes them come sooner (every frame is bit for bit the one-at-a-time run's).
        What depends on the frames before it (a note that some frame was empty, a list of missing frames) is gathered
        by the node from the results, which come back in order.

        Threads, not processes: the time goes into numpy's kernels and OpenImageIO's file reads and writes, all of
        which let go of the interpreter lock. At most `most` items are in flight, so what a shot holds in memory at
        once does not grow with its length (a 4K frame is tens of MB; a few hundred frames would not fit). Progress is
        reported as results come back in order; a stopped cook starts no further item (the ones running finish, then
        CookCancelled is raised), and the first item that fails raises its error once the ones running are done."""
        from concurrent.futures import ThreadPoolExecutor

        from ..io.parallel import in_flight

        items = list(items)
        if len(items) < 2 or most < 2:
            for done, item in enumerate(items, 1):
                yield work(item)
                self.progress(done, len(items), message)
            return

        @carried  # the frame's thread does the cook's account's work (its cache), noting into the same job
        def one(item):
            self.check_stop()  # stopped while it waited for a thread: not started
            return work(item)

        # bounded concurrency is implemented once (io/parallel.py in_flight); this manages the thread pool and progress
        with ThreadPoolExecutor(min(most, len(items)), thread_name_prefix="l2s-frame") as pool:
            for done, (_, got) in enumerate(in_flight(pool, ((i, one, item) for i, item in enumerate(items)), most), 1):
                yield got
                self.progress(done, len(items), message)

    def stage(self, name: "str | Msg", /, **params: Any) -> None:
        """A named stage of the cook starts: `name` is the stage's id, its words node.<type>.stage.<id>, else the
        extension's or the core's shared stage.<id> (lab2shot/i18n), in the language now; one with no words shows as
        it is. A Msg (the engine's own stages: I-STAGE-PLATE …) is said in each follower's language."""
        from ..messages import msg_word, said_word, word_of

        t = self.node_type
        if isinstance(name, Msg):
            word = msg_word(name)
            name = name.code
        else:
            word = word_of(f"node.{t.id}.stage.{name}", None if t.runtime == "core" else t.runtime, **params) if t else None
        said = said_word(word) if word else None
        # the words go with it (name_word): whoever follows the cook reads them in their own language (messages.localized)
        self.emit({"type": "stage", "node": self.node_id, "path": list(self.path), "name": said or name,
                   **({"name_word": word} if said else {})})

    def phase(self, name: str) -> None:
        """Which of the four phases this node is in now (lab2shot/progress.py: 排队中 / 加载模型 / 计算 /
        取回结果). Only the places that really know say it: engine/external.py before it starts the worker process
        (加载模型) and once the worker has written its result (取回结果). A core node never runs a model, so it
        never reports either and goes straight to 计算 at node_start, which is accurate for it."""
        self.emit({"type": "phase", "node": self.node_id, "path": list(self.path), "name": name})

    def say(self, code: str, /, *, port: str = "", param: str = "", fix: str = "", **params: Any) -> None:
        """Say a message while cooking (lab2shot/messages: its level is its code's letter). It goes to whoever follows
        the cook as a "message" event and is kept with the node's result (Engine._run_node): the node's marks, the
        message panel and a later status all read it there. `port` / `param`: the input or parameter it is about
        (clicking it in the panel goes there); `fix`: the node type that, inserted in front of `port`, puts it
        right. A `{node}` its template names and `params` lacks is this node (its label: name and type)."""
        from ..messages import names, template

        # a message pointing at the node that says it names it as every message does, 「名字（类型）」 (engine/naming.py
        # node_ref, the context's label): filled in here, never written by each node (nor as its subtitle)
        if "node" not in params and "node" in names(template(code)):
            params["node"] = self.label
        anchors = {k: v for k, v in (("port", port), ("param", param)) if v}
        if fix:  # the same one click a usage check offers (engine/lint.py): the node inserted, named on its button
            anchors["fix"] = insert_fix(fix)
        said = {**Msg(code, **params).json(), **anchors}
        if said not in self.said:
            self.said.append(said)
        self.emit({"type": "message", "node": self.node_id, "path": list(self.path), **said})

    def fact(self, name: str, value: Any) -> None:
        """Say a fact about this cook the node can only know now (NodeDef.fact_labels: how many chunks the shot was cut
        into). The engine resolves the node's declarations again with it and says which parameter turned out not to
        apply (W-APPLIES-UNUSED)."""
        self.facts[name] = value


EXPECTED = "expected"  # _context: the instance lacks only empties that were expected (it gives nothing quietly)


@dataclass(frozen=True)
class Ran:
    """How the cook ran one instance (Engine.ran): its plan (the fingerprints it wrote), its parameters (wired
    values in), whether it ran on a GPU, what its result covers."""

    plan: NodePlan
    params: dict
    gpu: bool
    info: Info  # what its result covers (frames, size)


@dataclass(frozen=True)
class Cooked:
    """What a cook came to (Engine.cook): whether anything failed (an instance's error, said at its node, a time
    limit among them, or a target left without its result), and whether something came through all the same (a target
    has its result: Demand.comes_through). The farm tells 「完成」, 「部分失败」 (failed, and a target has its result)
    and 「出错」 by it."""

    failed: bool
    through: bool


class _Place:
    """The place one node holds while it runs (engine/resources.py Ticket): given back while the node only waits for
    another cook of the same result, which leaves nothing to compute more often than not, and asked for again when it
    still has to compute. `ticket` is the one it holds now."""

    def __init__(self, resources: Resources, need: Need, ticket: Ticket, stop: threading.Event) -> None:
        self.resources, self.need, self.ticket, self.stop = resources, need, ticket, stop
        self.held = True

    def give_back(self) -> None:
        if self.held:
            self.held = False
            self.resources.done(self.ticket)

    def take_again(self) -> None:
        """Hold a place again (at once when it never gave it back); raises CookCancelled once the node is stopped
        while it waits, its request then withdrawn by whoever gives its ticket back (_Run._one)."""
        if self.held:
            return
        granted = threading.Event()
        self.ticket, self.held = self.resources.ask(self.need, self.stop, granted.set), True
        while not self.ticket.granted:
            _stopped(self.stop)
            granted.wait(WAIT_S)


class _Run:
    """One cook's instances in flight (Engine.cook): each is asked of the resources once it has to compute
    (engine/resources.py), goes on a thread of its own the moment its place is granted, and is through (`done`) when
    that thread ends, its place given back. Only the cook's own thread asks and starts; a node's thread only says it is
    through. A key is an instance, or a delivery's packing (("pack", its node))."""

    def __init__(self, resources: Resources, stop: threading.Event) -> None:
        self.resources, self.stop = resources, stop  # stop: the whole cook's (the job's)
        self.cond = threading.Condition()
        self.done: set = set()  # keys that are through: cooked, failed, skipped, taken from the cache
        self.asked: dict = {}  # key -> (ticket, the node's stop, its work, its need): waiting for its place
        self.running: dict = {}  # key -> (the node's stop, its thread)
        self.error: BaseException | None = None  # a bug on a node's thread: the cook raises it once all has stopped

    def busy(self, key) -> bool:
        with self.cond:
            return key in self.asked or key in self.running

    def idle(self) -> bool:
        with self.cond:
            return not self.asked and not self.running

    def again(self, key) -> None:
        """Through no longer: it is to be done again (what is wanted of it grew: Engine._cook_all)."""
        with self.cond:
            self.done.discard(key)

    def through(self, key) -> None:
        """Through without a thread of its own (skipped, failed before it could start, taken from the cache)."""
        with self.cond:
            self.done.add(key)

    def ask(self, key, need: Need, work: Callable) -> None:
        """Ask for the place `need` says; `work(place)` runs on its own thread once it is granted (_Place: the ticket
        it holds and the node's own stop)."""
        stop = threading.Event()
        ticket = self.resources.ask(need, stop, self._woken)
        with self.cond:
            self.asked[key] = (ticket, stop, work, need)

    def _woken(self) -> None:
        with self.cond:
            self.cond.notify_all()

    def start(self) -> None:
        """Every request whose place was granted goes on its own thread (doing the cook's account's work)."""
        with self.cond:
            granted = [(k, v) for k, v in self.asked.items() if v[0].granted]
            for key, _ in granted:
                del self.asked[key]
        for key, (ticket, stop, work, need) in granted:
            place = _Place(self.resources, need, ticket, stop)
            thread = threading.Thread(target=carried(self._one), args=(key, place, work), daemon=True, name="l2s-node")
            with self.cond:
                self.running[key] = (stop, thread)
            thread.start()

    def _one(self, key, place: _Place, work: Callable) -> None:
        try:
            work(place)
        except CookCancelled:  # stopped with its cook (a time limit is the work's own error: Engine._compute)
            pass
        except BaseException as exc:  # a bug: never lost on a thread, raised by the cook
            with self.cond:
                self.error = self.error or exc
        finally:
            self.resources.done(place.ticket)  # the place goes back the moment the node is through
            with self.cond:
                self.running.pop(key, None)
                self.done.add(key)
                self.cond.notify_all()

    def wait(self) -> None:
        """Until a node is through or a place is granted, or a moment has passed (the cook's stop is looked at that
        often); raises CookCancelled once the cook is stopped, and a node thread's bug."""
        with self.cond:
            if not any(v[0].granted for v in self.asked.values()) and self.error is None:
                self.cond.wait(WAIT_S)
            error = self.error
        _stopped(self.stop)
        if error is not None:
            raise error

    def end(self) -> None:
        """Nothing stays in flight: every running node is told to stop (a worker's process is killed, a core node
        stops at its next frame), every request not granted yet is withdrawn, and every node's thread is waited for."""
        with self.cond:
            asked, self.asked = list(self.asked.values()), {}
            running = list(self.running.values())
        for ticket, stop, _, _ in asked:
            stop.set()
            self.resources.done(ticket)
        for stop, _ in running:
            stop.set()
        for _, thread in running:
            thread.join()


class Engine:
    """Cooks a graph on top of one `Evaluation` (engine/evaluation.py), which does all the read-only work (plans,
    parameters, wired values, sources, status), memoised per node instance. `Engine`'s own evaluation is never one
    another caller shares (whoever makes an Engine gives it a fresh Evaluation: the farm per cook, farm/queue.py; the
    cache in evaluations.py only holds ones built for reading): cooking changes what its plans find on disk as
    instances finish, which only this Engine's own run should see.

    Several instances of one cook run at once, each on its own thread: everything they read of the evaluation (and of
    the graph) and everything they change there is done holding the engine's one lock (`_lock`), which is never held
    while a node computes, commits its packets or waits for a place or another cook's lock. Planning reads files (an
    import lists what its file holds, through its extension's worker when it has one): that is done first, outside the
    lock (NodeDef.ready, before each scan of the order: _to_ready), so planning under the lock finds it done. What
    is still read while planning under it: an instance whose parameters could not be told before (planning then says
    why), and a list's source planned to learn a block's items (instances), when that is an import prepared in the
    same scan."""

    def __init__(self, graph: Graph, stop: threading.Event | None = None, collector: OutputSink | None = None,
                 evaluation: Evaluation | None = None, stream_worker: Callable[..., tuple] | None = None):
        """`stop`: set it to stop the cook; `collector`: where 「输出」
        collects and packs its files (whoever submits the cook builds it: the farm, per task, lab2shot/transfer/outputs.py
        Collector); without one a node that delivers is refused; `evaluation`: reuse this one for reading instead of building
        a fresh one, never one shared with anything else that might also cook, since cooking mutates it (see the class
        docstring). `stream_worker`: who runs a streaming node's worker on a thread and returns (raw folder, done,
        error(), halt()); the farm injects its own (farm/streaming.py), so the engine never opens a thread for it;
        without one a streaming node cooks with the worker blocking. Whose the files are read as is the
        evaluation's account (Evaluation.account): an upload that is not it lets its node fail at itself, as one that
        was cleaned away does. Where each node runs (a GPU, a CPU slot) is the resources' answer (Engine.cook).

        View proxies are not made in the engine: after a packet is computed, the farm hands its proxies to the
        scheduler as background work (`lab2shot/farm/queue.py _proxies_of`), which never delays a node; the task ends
        when its nodes are done. The engine has no knowledge of the viewing layer."""
        self.graph = graph
        self.stop = stop or threading.Event()
        self.collector = collector
        self.stream_worker = stream_worker
        self.eval = evaluation or Evaluation(graph)
        self.failed: dict[Inst, CookError] = {}  # instances that failed in this engine's cooks: said once
        # what each instance it started ran with, as it found it (its plan, the parameters with their wired values,
        # whether on a GPU): only added to, read by others while it cooks (the task's partial folders and timing
        # records, farm/queue.py Job.ran) instead of an evaluation of the graph from before the cook
        self.ran: dict[Inst, Ran] = {}
        self.skipped: set[Inst] = set()  # instances said to be skipped in them
        self.chained: set[Inst] = set()  # the failed ones that are only a 「输出」 with a line into it broken
        self._prepared: set[Inst] = set()  # instances whose planning was readied (NodeDef.ready, _to_ready)
        self._writing: dict[str, threading.Event] = {}  # packets an instance of this cook is writing -> set once done
        self._touched: set[str] = set()  # packets marked used in this cook (_mark_used)
        self._touching = threading.Lock()
        self._lock = threading.RLock()  # the evaluation and the graph, read and changed by the cook's threads
        self._changes = 0  # how many times what is known changed (_changed): a scan of the order goes on while it has not

    # ------------------------------------------------------------------ cooking

    def cook(self, targets: list[str], emit: EventFn, resources: Resources, force: bool = False,
             show: frozenset[str] | None = None) -> Cooked:
        """Cook every instance of `targets` and what they need. Contract: refused before anything runs only for what
        is wrong with the graph itself (Evaluation.readiness, GraphError); otherwise every instance whose inputs are
        through starts on its own thread once `resources` grant what its cost needs, and the order is asked again as
        each finishes, so what only a cook tells (items, a condition, a wired value) is planned once known. An error is
        the failing instance's alone: said once, kept with its fingerprint; what requires it is skipped, what only may
        take it goes without, a block's end gathers the other items, every other branch cooks on (why: one bad item or
        branch never costs the rest). `show`: the outputs the viewer shows of the one target (Demand.demand), the same
        its submission's readiness was given. Raises CookCancelled once `stop` is set; returns what the cook came to
        (Cooked)."""
        shown = frozenset(show or ())
        if (refused := self.eval.readiness(targets, force, shown).refused) is not None:  # what is wrong with the graph itself
            raise GraphError(refused)
        run = _Run(resources, self.stop)
        try:
            with external.stopped_by(self.stop):  # a worker job planning runs (an import in a block) stops with the cook
                self._cook_all(targets, emit, run, shown, force)
                whole = self._deliver(targets, emit, run)
        finally:
            run.end()
        # another branch came through: a target has its result (Demand.comes_through), not a side branch that only
        # fed what was skipped
        return Cooked(bool(self.failed) or len(whole) < len(targets), bool(whole))

    def _cook_all(self, targets: list[str], emit: EventFn, run: _Run, shown: frozenset[str], force: bool) -> None:
        """Every instance the targets need, each started as soon as what it takes from is through."""
        preparing = [n for n in self.eval.needed(targets) if _readies(self.graph.nodes[n].type)]
        seen = self._changes
        with self._lock:
            asked = self.eval.demand(targets, shown)  # what is wanted of each instance, as last worked out
        while True:
            _stopped(self.stop)
            if preparing:  # what planning them needs that takes a while (NodeDef.ready): before, outside the lock
                with self._lock:
                    ahead = self._to_ready(preparing)
                for node_type, params in ahead:
                    node_type.ready(params)
            with self._lock:
                order, _ = self.eval.order(targets)
                if self._changes != seen:  # what is known changed: what is wanted may have grown
                    seen = self._changes
                    for inst in self._wanting_more(order, run, asked, targets, shown):
                        run.again(inst)
                    asked = self.eval.demand(targets, shown)
            todo = [i for i in order if i not in run.done and not run.busy(i)]
            if not todo and run.idle():
                return
            moved = False
            for inst in todo:
                forced = force and inst.node in targets
                with self._lock:
                    if not all(d in run.done for d in self.eval.deps(inst)):
                        continue
                    changes = self._changes
                    begun = self._begin(inst, emit, targets, shown, forced)
                if begun is None:  # through without computing: the scan goes on, unless that changed what is known
                    run.through(inst)
                    moved = True
                    if self._changes != changes:  # (failed, skipped behind a failure): the order is worked out again
                        break
                    continue
                plan, need = begun
                run.ask(inst, need, lambda place, inst=inst, plan=plan, forced=forced:
                        self._compute(inst, plan, emit, targets, shown, forced, place))
            run.start()
            if not moved:
                run.wait()

    def _wanting_more(self, order: list[Inst], run: _Run, asked: dict, targets: list[str], shown: frozenset[str]) -> list[Inst]:
        """(Holding the lock) the instances through in this cook that do not have every output now wanted of them: what
        is wanted grows as a cook unfolds (a block's items or a switch's condition known, and what reads them known
        with it), past what was foreseen (Routing.wanted_while_pending). They are cooked again for the missing outputs
        (_to_give writes only those). Not one that failed or was skipped, nor one that gives nothing to keep (「输出」).
        Only when the demand was worked out anew (a new table: what is known changed it), and only for the instances it
        wants more of than `asked` (the last one), each then asked of the disk (Demand.missing, as _to_give writes):
        nothing else can have grown, so a long chain is not checked again at every step."""
        wanted = self.eval.demand(targets, shown)
        if wanted is asked:
            return []
        return [i for i in order if i in run.done and i not in self.failed and i not in self.skipped and self.graph.outputs(i.node)
                and wanted.get(i, frozenset()) - asked.get(i, frozenset())
                and self.eval.missing(i.node, self.eval.plan(*i), wanted.get(i, frozenset()))]

    def _deliver(self, targets: list[str], emit: EventFn, run: _Run) -> list[str]:
        """Once every instance is through: which targets have their result, and a 「输出」 whose instances all collected
        packs its zip (on a CPU slot, like a node: its own computation, under the same time limit)."""
        whole: list[str] = []
        for target in targets:
            with self._lock:
                through = self.eval.comes_through(target)
            node = self.graph.nodes[target]
            if not through:
                emit({"type": "done", "node": target, "path": []})
                continue
            # every instance of it collected (one outside every block, one per item): the zip; none collected anything
            # (a block with no items too): the collector refuses to pack (E-DELIVER-EMPTY), it fails
            if node.type.delivers:
                if self.collector is None:
                    with self._lock:
                        self._fail(Inst(target, ()), CookError(target, Msg("E-COOK-NODELIVERY", node=node.label)), emit, keep=False)
                    continue
                key = ("pack", target)
                run.ask(key, Need(CPU, "core", 0.0, 0.0, node.label),
                        lambda place, target=target: self._pack(target, emit, place.ticket, place.stop))
                while not run.idle():
                    run.start()
                    run.wait()
                if Inst(target, ()) in self.failed:
                    emit({"type": "done", "node": target, "path": []})
                    continue
            whole.append(target)
            emit({"type": "done", "node": target, "path": []})
        return whole

    def _pack(self, target: str, emit: EventFn, ticket: Ticket, stop: threading.Event) -> None:
        label = self.graph.nodes[target].label
        try:
            self.collector.pack(target, label, stop)
        except (CookCancelled, CookError) as exc:  # the same rule as a node's (_compute)
            if ticket.expired is None and (stop.is_set() or isinstance(exc, CookCancelled)):
                raise CookCancelled from None
            with self._lock:
                self._fail(Inst(target, ()), exc if ticket.expired is None else CookError(target, ticket.expired), emit,
                           keep=False)

    def _to_ready(self, nodes: list[str]) -> list[tuple[type[NodeDef], dict]]:
        """(Holding the lock) the instances of `nodes` known now and not prepared yet, with the parameters to prepare
        their planning with (NodeDef.ready): an import in a 逐项处理 block gets its file per item, once the items are
        known. Reading an instance's parameters does not plan it (only what feeds them). One whose parameters can't be
        told yet is left to planning, which says why."""
        out = []
        for nid in nodes:
            paths, _ = self.eval.instances(nid)
            for path in paths:
                if (inst := Inst(nid, path)) in self._prepared:
                    continue
                self._prepared.add(inst)
                try:
                    out.append((self.graph.nodes[nid].type, self.eval.params(nid, path)))
                except PLAN_ERRORS:
                    pass
        return out

    def _begin(self, inst: Inst, emit: EventFn, targets: list[str], shown: frozenset[str],
               force: bool) -> tuple[NodePlan, Need] | None:
        """(Holding the lock) an instance whose inputs are through: what it needs to compute (its plan and its place),
        or None when it is through without computing (skipped behind a failure, a 「输出」 with a broken line, failed
        before it could start, or taken from the cache)."""
        if inst in self.failed:  # failed in this engine's run already: said once
            return None
        fate, outcome = self.eval.fate(inst, now=True), self.eval.outcome(*inst)  # (the one classification: Demand.fate)
        if fate == SKIP:
            if inst not in self.skipped:
                self.skipped.add(inst)
                emit({"type": "skipped", "node": inst.node, "path": list(inst.path), "root": outcome.root, **outcome.message,
                      **({"blocked": True} if outcome.blocked else {})})  # behind a 「阻断」: the page says 「已跳过（被阻断）」
            return None
        if fate == FAIL:  # known before it runs: said at it, no place asked for, never kept as a record a later cook
            # would try again (Outcome.retryable); a 「输出」 with a broken line fails as it stands
            self._fail(inst, CookError(inst.node, Msg(outcome.message["code"], **outcome.message.get("params", {}))), emit,
                       keep=False)
            if outcome.chain:
                self.chained.add(inst)
            return None
        if fate == RETRY:  # a past cook's failure: tried again, and what goes without it wants it again
            failure_file(self.eval.plan(*inst).fingerprint).unlink(missing_ok=True)
            self._changed(inst)
        nid, path = inst
        try:
            with serving(self.eval.account):  # it reads the files as the account this cook is for (uploads.resolve)
                plan = self.eval.plan(nid, path)
                wanted = self.eval.demand(targets, shown).get(inst, frozenset())
                if not force and self.graph.outputs(nid) and not self._to_give(nid, plan, wanted, force):
                    emit({"type": "node_done", "node": nid, "path": list(path), "cached": True, "seconds": 0,
                          "outputs": _present(plan)})
                    self._settled(inst)  # there already (another task, a twin of the same fingerprint): known now
                    return None
        except PLAN_ERRORS as exc:  # it can't be planned (a check an item fails): its own error, never the whole cook's
            self._fail(inst, exc if isinstance(exc, CookError) else CookError(nid, message_of(exc)), emit)
            return None
        node, cost = self.graph.nodes[nid], self.eval.resolved(nid, path).cost
        return plan, Need(GPU if cost.gpu else CPU, node.type.runtime, self.eval.vram_need(nid, path), cost.ram_gb,
                          node.label, nid)

    def _compute(self, inst: Inst, plan: NodePlan, emit: EventFn, targets: list[str], shown: frozenset[str],
                 force: bool, place: _Place) -> None:
        """An instance on its own thread, granted its place: cooked under the lock of its fingerprint (and of its
        output packets), so two cooks that need the same result compute it once and neither sees it half-written.
        While it waits for such a lock its place goes back, so a card is never held idle by a node that waits; it
        asks again once it has the locks and still has to compute. `place.stop`: this node's own, set when the cook
        stops or when the node ran past its time limit."""
        nid, path = inst
        node = self.graph.nodes[nid]
        stop = place.stop
        where = {"node": nid, "path": list(path)}
        emit({"type": "node_start", **where, "label": node.label, "label_word": word_of("engine.node_ref", name=nid, type=node.type.id)})

        def waiting(said: Msg = Msg("I-WAIT-OTHERCOOK")) -> None:
            emit({"type": "stage", **where, "name": said.text, "name_word": msg_word(said)})
            place.give_back()

        def siblings(written: list[threading.Event]) -> None:
            """Wait, its place given back, for packets another instance of this cook is writing for it (_writing)."""
            for done in written:
                if not done.is_set():
                    waiting(Msg("I-WAIT-SIBLING"))
                    while not done.wait(WAIT_S):
                        _stopped(stop)

        mine: set[str] = set()
        try:
            with serving(self.eval.account), exclusive(plan.fingerprint, waiting, lambda: _stopped(stop)):
                with self._lock:  # another cook may have produced it while this one waited
                    wanted = self.eval.demand(targets, shown).get(inst, frozenset())
                    give = set(self._to_give(nid, plan, wanted, force))
                    # a packet several instances of this cook give (a block's 「条数」, one for every item): written
                    # by the first to get here, the others wait for it without a lock each (_writing)
                    theirs = [self._writing[plan.outputs[p]] for p in give if plan.outputs[p] in self._writing]
                    give = {p for p in give if plan.outputs[p] not in self._writing}
                    mine = {plan.outputs[p] for p in give}
                    for fp in mine:
                        self._writing[fp] = threading.Event()
                    through = not force and self.graph.outputs(nid) and not give
                if through:  # nothing of its own to write: once what siblings write for it is there (never under the lock)
                    siblings(theirs)
                    emit({"type": "node_done", **where, "cached": True, "seconds": 0, "outputs": _present(plan)})
                    with self._lock:
                        self._settled(inst)  # written meanwhile by another cook of it: known now
                    return
                # the lock of every output packet folder it writes is also held until the node finishes. The two places
                # that clean incomplete packets (farm/disk.py clean_incomplete, sweep_incomplete) try the packet folder's
                # own name (the output fingerprint `_hash([fp, port])`), not the node fingerprint; holding only the node
                # lock, cancelling a not-yet-started task of the same graph or starting another process would delete the
                # folder being written here, and packet.commit would then raise FileNotFoundError. Only the outputs it
                # gives are locked up front (sorted by fingerprint, after the node's own: no deadlock); an output it
                # writes although nobody asked is locked when its folder is made (Outputs hold: its fingerprint is this
                # node's own, nobody else waits for it). Locking every output would serialise the instances of a block's
                # begin, whose 「条数」 is one packet for all of them (engine/scopes.py port_fp).
                # Then the packets it reads, shared: a forced recook in another task (which holds them exclusive while it
                # removes and rewrites them, fresh_dir) waits until this node is through, and this node waits for it,
                # so no packet is removed while a cook reads it. Readers come after writers in the lock order, and the
                # graph of fingerprints has no cycles, so there is no deadlock either.
                writes = {plan.outputs[p] for p in give}  # never the begin's item: _to_give leaves it out
                with self._lock:
                    reads = self._read_packets(inst) - writes
                with ExitStack() as held:
                    for fp in sorted(writes):
                        held.enter_context(exclusive(fp, waiting, lambda: _stopped(stop)))
                    for fp in sorted(reads):
                        held.enter_context(shared(fp, lambda: waiting(Msg("I-WAIT-RECOOK")),
                                                  lambda: _stopped(stop)))
                    place.take_again()
                    self._run_node(inst, plan, emit, frozenset(give), force, place.ticket, stop,
                                   hold=lambda fp: held.enter_context(exclusive(fp, waiting, lambda: _stopped(stop))))
                siblings(theirs)  # through only once what a sibling writes for it is there
        except (CookCancelled, CookError) as exc:
            # whatever a node that was stopped raises (a streaming node's convert says its worker wrote nothing) is
            # the stop's doing: past its time limit it failed with the limit's words, what needs it is skipped; stopped
            # with its cook, nothing of it is an error
            expired = place.ticket.expired
            if expired is None and (stop.is_set() or isinstance(exc, CookCancelled)):
                raise CookCancelled from None
            with self._lock:
                self._fail(inst, exc if expired is None else CookError(nid, expired), emit)
        finally:
            with self._lock:
                for fp in mine:
                    self._writing.pop(fp).set()

    def _made_from(self, inst: Inst) -> dict[str, str]:
        """The packets the instance read, each with the generation it read (data/packet.py created_of): what its
        results are made from (Packet.commit made_from), so a forced recook of one of them makes them stale."""
        with self._lock:
            reads = self._read_packets(inst)
        return {fp: g for fp in sorted(reads) if (g := created_of(fp))}

    def _read_packets(self, inst: Inst) -> set[str]:
        """(Holding the lock) the fingerprints of the packets the instance takes on its inputs (those _context reads)."""
        taken = self.eval.taken(*inst)
        return {self.eval.plan(src, p).outputs[sport] for port in self.graph.input_ports(inst.node)
                for src, sport, p in taken[port.name]}

    def _settled(self, inst: Inst) -> None:
        """(Holding the lock) the instance has its results now, whoever wrote them (this cook, another task, a twin of
        the same fingerprint): the one way what was waiting on it learns so — the evaluation forgets what it remembered
        of it and of its readers (Demand.forget cooked: a condition, a wired value, a list read before planning was
        Pending, and is known now)."""
        self.eval.forget(inst, True)
        self._changes += 1

    def _changed(self, inst: Inst, cooked: bool = False) -> None:
        """What is known of this instance changed (cooked, failed, tried again): the evaluation forgets it and what may
        depend on it (Evaluation.forget; `cooked`: its outputs are there now, nothing else of it changed), and the
        evaluations of the cache that planned it are stale (records.changed)."""
        self.eval.forget(inst, cooked)
        self._changes += 1
        try:  # its failure record written or cleared: the evaluations that planned it are stale (its packets tell
            changed(self.eval.account.user_id, self.eval.plan(*inst).fingerprint)  # of themselves: on_committed)
        except PLAN_ERRORS:
            pass

    def _fail(self, inst: Inst, exc: CookError, emit: EventFn, keep: bool = True) -> None:
        """The instance failed: said once, kept with its fingerprint (`keep`; not a failure that is only what stands
        above it, which the evaluation tells again by itself), and what needs it is skipped from here on."""
        said = {**exc.message.json(), **({"param": exc.param} if exc.param else {}), "log": str(exc.log) if exc.log else None}
        try:
            if keep:
                path = failure_file(self.eval.plan(*inst).fingerprint)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(said, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, CookError):  # an instance that can't even be planned: its status says why already
            pass
        self.failed[inst] = exc
        self._changed(inst)
        emit({"type": "error", "node": inst.node, "path": list(inst.path), **said})

    def _to_give(self, nid: str, plan: NodePlan, wanted: frozenset[str], force: bool) -> frozenset[str]:
        """The wanted outputs to write: those missing (Demand.missing, what `satisfied` reads), every one it has when
        forced (a block's item is its list's packet: never written again)."""
        t = self.graph.nodes[nid].type
        item = t.item_output if sc.role(t) == sc.BEGIN else None
        give = set({p for p in wanted if p in plan.outputs and p != item} if force else self.eval.missing(nid, plan, wanted))
        # an output made from another output of the same node (point cloud = depth map + camera): its source is written
        # too, even if already on disk, because the node derives it in process from the freshly computed data rather than
        # reading it back from disk (Port.made_from)
        if give:
            outs = {p.name: p for p in self.graph.outputs(nid)}
            for name in list(give):
                give.update(n for n in getattr(outs.get(name), "made_from", ()) if n in plan.outputs and n != item)
        return frozenset(give)

    def _settle_outputs(self, nid: str, produced: dict[str, Packet], inputs: dict[str, list[Packet]], params: dict) -> None:
        """Settle every output before anything downstream sees it: what was photographed is carried
        over from the node's picture input by what the output port declares (data/contracts.py Shape), its window and
        size are checked against the same declaration, and each packet against its port's type and its type's
        contract. A node that breaks one fails here, at the node that did it."""
        from ..data.contracts import settle
        from ..data.types import DATA_TYPES, accepts

        node = self.graph.nodes[nid]
        ports = {p.name: p for p in self.graph.outputs(nid)}
        for port, packet in produced.items():
            if port not in ports:
                kind = DATA_TYPES[packet.type].label if packet.type in DATA_TYPES else packet.type
                raise CookError(nid, Msg("E-COOK-CONTRACT", node=node.label, output=port, kind=kind,
                                         problems=[Msg("E-COOK-NOSUCHOUTPUT", output=port)]))
        picture = _first(inputs.get(node.type.picture)) if node.type.picture else None
        problems: dict[str, list[Msg]] = {}
        for port, packet in produced.items():
            want = self.graph.output_type(nid, port)
            if not accepts(want, packet.type):
                problems.setdefault(port, []).append(Msg("E-COOK-OUTPUTTYPE", got=packet.type, want=want))
            # the meanings this port guarantees to state (Port.means: scale for depth ports, space for position and
            # normal ports, ...) are enforced here; the type states only the channel count, and the meaning is
            # guaranteed by the writing port.
            # An empty packet needs only its type (as contracts.settle / check define): with nothing computed there is
            # no meaning to state; an empty result is not an error
            for key in () if packet.meta.get("empty") else ports[port].means:
                if key not in packet.meta:
                    problems.setdefault(port, []).append(Msg("E-CONTRACT-MISSINGKEY", key=key))
            # picture or values: what the port declares (Graph.output_data, Port.data) is what the packet is marked,
            # so whatever reads the packet later (payloads.is_data: an output setting converting colour or not, the
            # viewer) agrees with what the conditions on the parameters read before the cook. A node that wrote the
            # other kind is caught here, at the node that did it.
            declared = None if packet.meta.get("empty") or is_list(packet.type) else self.graph.output_data(nid, port)
            if declared is not None:
                if is_data(packet) != declared:
                    problems.setdefault(port, []).append(Msg("E-CONTRACT-DATAKIND", declared=_kind_word(declared),
                                                             written=_kind_word(is_data(packet))))
                packet.meta["values"] = declared
        by_port = {port: ports[port].shape for port in produced}
        # an output that follows another input (a warp lands on its map's window): settled against that input
        for port, shape in by_port.items():
            if shape.follows:
                for said in settle({port: produced[port]}, {port: shape}, _first(inputs.get(shape.follows)),
                                   node.type.said_shot(params)):
                    problems.setdefault(said[0], []).append(said[1])
        rest = {port: p for port, p in produced.items() if not by_port[port].follows}
        for port, said in settle(rest, by_port, picture, node.type.said_shot(params)):
            problems.setdefault(port, []).append(said)
        for port, found in problems.items():
            packet = produced[port]
            kind = DATA_TYPES[packet.type].label if packet.type in DATA_TYPES else packet.type
            raise CookError(nid, Msg("E-COOK-CONTRACT", node=node.label, output=ports[port].label, kind=kind, problems=found))

    def _input_packet(self, fp: str) -> Packet | None:
        """An input packet a node is about to cook with, read through the evaluation (a manifest already read while
        planning is not read a second time). Marked used after, outside the lock (_mark_used)."""
        return self.eval.packet(fp)

    def _mark_used(self, packets) -> None:
        """(Never under the engine's lock) the packets an instance cooks with were used now: the cache keeps what a
        cook actually relies on (status and plan never touch this, see Evaluation's docstring). Each packet, and each
        it names (a list's items), once per cook however many instances read it (_touched): the n instances of a
        block's begin read the one list of n items, which would otherwise be n² touches."""
        with self._touching:
            for d in packets:
                Packet.used(d, self._touched)

    def _run_node(self, inst: Inst, plan: NodePlan, emit: EventFn, wanted: frozenset[str], force: bool, ticket: Ticket,
                  stop: threading.Event, hold: Callable[[str], None] | None = None) -> None:
        nid, path = inst
        node = self.graph.nodes[nid]
        started = time.time()
        with self._lock:
            ctx, lacking = self._context(inst, plan, emit, wanted, force, ticket, stop, hold)
            self.ran[inst] = Ran(plan, dict(ctx.params), self.eval.resolved(nid, path).cost.gpu, self.eval.info(nid, path))
        self._mark_used(p.dir for got in ctx.inputs.values() for p in got)
        if lacking:  # a required input with nothing: nothing to cook, and nothing to give
            return self._give_nothing(inst, plan, ctx, emit, started, expected=lacking is EXPECTED)
        try:
            produced = node.type.cook(ctx) or {}
        except NothingToCook as found:  # it found nothing to give: not an error
            ctx.say(found.message.code, **found.message.params)
            # `expected`: nothing was asked of it (nothing to translate): what reads it is told nothing either
            return self._give_nothing(inst, plan, ctx, emit, started, expected=found.expected)
        except (CookError, CookCancelled):
            raise
        except Exception as exc:
            (ctx.work / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            # a refusal raised by core code keeps its own message, code and parameters (the worker side does the same,
            # engine/external.py); an exception with no words of its own (StopIteration, KeyError()) still says what
            # failed, never 「出错：」
            said = exc.message if isinstance(exc, MessageError) else (str(exc) or type(exc).__name__)
            raise CookError(nid, Msg("E-COOK-FAILED", node=node.label, reason=said), ctx.work / "error.log") from exc
        # an output it was to give and did not: it failed, and none of what it did give is committed (no part of a
        # result is left in the cache as if it were one)
        if missing := set(wanted) - set(produced):
            raise CookError(nid, Msg("E-COOK-MISSINGOUTPUT", node=node.label, ports=sorted(missing)))
        with self._lock:
            self._settle_outputs(nid, produced, ctx.inputs, ctx.params)
            self._say_unused(inst, ctx)
        made_from = self._made_from(inst)
        outs = {p.name: p for p in self.graph.outputs(nid)}
        # the one door every empty output passes, however the node made it (empty_packet, a helper, a packet built by
        # hand): a wanted output given empty must declare it may (Port.may_be_empty) — the engine and the page treat
        # such a port so, and an undeclared one would leave downstream a value nobody can see or set
        broken = {port for port, packet in produced.items()
                  if packet.meta.get("empty") and not getattr(outs.get(port), "may_be_empty", False)}
        if undeclared := sorted(broken & set(wanted)):
            raise CookError(nid, Msg("E-COOK-CONTRACT", node=node.label, output=i18n.Both.of(lambda: i18n.separator().join(undeclared)),
                                     kind=i18n.Word("engine.contract.output_port"),
                                     problems=i18n.Word("engine.contract.empty_undeclared")))
        for port in broken:  # the same check for an output nobody asked for: never cached under its real fingerprint
            # (asked for later, the empty packet would be taken for its result); it is left to be computed then
            shutil.rmtree(produced.pop(port).path(), ignore_errors=True)
        for port, packet in produced.items():
            if (packet.meta.get("empty") and getattr(outs.get(port), "may_be_empty", False)) or (
                    ctx.expected and (holds_nothing(packet.type, packet.meta) or (is_list(packet.type) and not packet.meta.get("items")))):  # an output that may give nothing gave
                packet.meta["expected"] = True  # nothing, or an end's list of expected empties: told to what reads it
            if port in ctx.outputs.kept:  # written beside a complete packet nobody asked for: that one stays, this goes with _work
                continue
            packet.commit(node.type.id, ctx.said, made_from)
        # after success the scratch folder (`<fp>_work`: the worker's input list, error.log) is no longer needed; on
        # failure it is kept (error.log is its log); a failure record another cook of it left is no longer true
        # (committing under the node's lock is the one place a success says so: failure_file)
        shutil.rmtree(packet_dir(plan.fingerprint + "_work"), ignore_errors=True)
        failure_file(plan.fingerprint).unlink(missing_ok=True)
        with self._lock:
            self._changed(inst, cooked=True)  # a cache generation of its own: status/plan reads made while this was cooking may now be stale
            size = self.eval.work(nid, path)  # with the seconds, what the farm's timing records keep (lab2shot/farm/timings.py)
        emit({"type": "node_done", "node": nid, "path": list(path), "cached": False, "seconds": round(time.time() - started - ctx.waited, 1),
              "frames": len(size.frames), "width": size.width, "height": size.height, "reused": ctx.reused,
              "gpu_name": ticket.gpu_name,  # the card's model it ran on ("" a CPU slot): what its timing record keeps
              "outputs": _present(plan)})  # its port fingerprints: the page writes them in without asking again

    def _context(self, inst: Inst, plan: NodePlan, emit: EventFn, wanted: frozenset[str], force: bool, ticket: Ticket,
                 stop: threading.Event, hold: Callable[[str], None] | None = None) -> tuple[CookContext, bool]:
        """(Holding the lock) what the node's cook() gets, its inputs read, and whether a required input brought
        nothing (then there is nothing to cook: `_give_nothing`); what it can tell before cooking is said on it."""
        nid, path = inst
        node = self.graph.nodes[nid]
        # every input's packets: without the wires whose source failed (taken) and without the ones that bring
        # nothing: no data this time (an empty packet), or data with nothing in it where the input does not take that
        taken = self.eval.taken(nid, path)
        begins, ends = sc.role(node.type) == sc.BEGIN, self.graph.scopes.ended.get(nid) is not None
        inputs, nothing, foreseen, foreseen_items = {}, [], set(), set()  # foreseen: (input, source) whose empty packet says it was expected
        per_item: list[tuple[str, Packet | None]] = []  # a block's end: (item key, its packet or None: nothing) per wire
        for port in self.graph.input_ports(nid):
            inputs[port.name] = []
            for src, sport, p in taken[port.name]:
                packet = self._input_packet(self.eval.plan(src, p).outputs[sport])
                # not on disk: removed while this cook ran (a clean), or an output its source was never asked to write
                # (what is wanted grows as a cook unfolds: wanted_while_pending and _wanting_more are there so it is);
                # this node's error, said as one
                if packet is None:
                    raise CookError(nid, Msg("E-COOK-MISSINGINPUT", node=node.label, source=self.graph.nodes[src].label,
                                             input=port.label))
                # nothing in it: an empty packet; and where the input does not take an empty one (Port.takes_empty), one
                # holding nothing, or a list with no items that says it was expected (an end's list of expected empties:
                # 「取一条」 has no item to take, it gives nothing as expected in turn; 「取信息」 counts its 0 items)
                empty = packet.meta.get("empty") or (not port.takes_empty and (
                    holds_nothing(packet.type, packet.meta) or (packet.meta.get("expected") and not packet.meta.get("items"))))
                if ends and len(p) > len(path):
                    per_item.append((p[len(path)], None if empty else packet))
                    if empty and packet.meta.get("expected"):
                        foreseen_items.add(p[len(path)])
                if empty:
                    nothing.append((port, src))
                    if packet.meta.get("expected"):
                        foreseen.add((port.name, src))
                else:
                    # what the wire check could not tell before the cook (a crop of a file whose layers were not
                    # known yet): the packet itself, against what the input takes (Port.data), refused the same way
                    if port.data in (True, False) and not is_list(packet.type) and channels_of(packet.type) \
                            and is_data(packet) != port.data:
                        raise CookError(nid, Msg("B-WIRE-DATAKIND", source=self.graph.nodes[src].label, output=sport,
                                                 got=_kind_word(is_data(packet)),
                                                 want=_kind_word(port.data), node=node.label, input=port.label))
                    inputs[port.name].append(packet)
        item_results, gave_nothing = results_by_item(per_item)
        note(plan.fingerprint)  # its `_work` and `_failed` hang off the node's fingerprint: the job's task references them
        work = packet_dir(plan.fingerprint + "_work")
        work.mkdir(parents=True, exist_ok=True)
        outputs = Outputs(plan.outputs, wanted, work, force, hold)  # the wanted ones, made now; another the node asks for anyway, then
        if node.type.delivers and self.collector is None:
            raise CookError(nid, Msg("E-COOK-NODELIVERY", node=node.label))
        ctx = CookContext(nid, node.label, node.type, self.eval.params(nid, path), inputs, outputs,
                          {port: self.graph.output_type(nid, port) for port in plan.outputs}, work, emit, ticket.gpu, stop,
                          fingerprint=plan.fingerprint,
                          frames=self.eval.info(nid, path).frames if node.type.frame_source else (), sources=self.eval.sources(nid, path), wired_from=self.eval.wired_from(nid, path),
                          provenance=self.eval.delivered_provenance(nid, path) if _records_provenance(node.type) else None, lineage=self.eval.lineage(nid, path) if node.type.reads_lineage else {}, ram_gb=self.eval.resolved(nid, path).cost.ram_gb,
                          vram_budget_gb=self.eval.vram_need(nid, path) if self.eval.full_tier(nid, path) is not None else 0.0,
                          collector=self.collector if node.type.delivers else None, names=tuple(self.eval.item_names(nid, path)),
                          stream_worker=self.stream_worker, path=path, wanted=wanted,
                          item=self.eval.item_at(nid, path) if begins else None,
                          items=tuple(i for i in self.eval.gathered(nid, path) if i.key not in gave_nothing) if ends else (),
                          item_results={k: tuple(v) for k, v in item_results.items()} if ends else {},
                          # with them, the inputs that came empty as expected (an output that may give nothing gave nothing)
                          quiet=self.eval.quiet_inputs(nid, path) | frozenset(port for port, _src in foreseen))
        for said, port in self.eval.unused_inputs(nid, path):
            ctx.say(said.code, port=port, **said.params)
        from ..nodes.applies import overscan_notices

        taken_packets = [(port.name, port.label, packet) for port in self.graph.input_ports(nid) for packet in inputs[port.name]]
        for port, said in overscan_notices(node.type, node.label, taken_packets):  # a pixel node given pixels past the format: what it leaves out
            ctx.say(said.code, port=port, **said.params)
        # an input it cannot go without that came empty; a switch's condition is not one: an empty condition reads as no
        # wire (Presence.wired_input) and the switch goes by its own 「走哪一路」
        def needs(port) -> bool:  # an input it cannot go without, whose coming empty leaves it nothing to cook: not a
            # switch's condition (empty reads as no wire), nor a block end's items (nothing to gather: its empty list)
            return (self.eval.requires(nid, path, port) and port.name != getattr(node.type, "condition_input", None)
                    and not sc.gathers_empty(node.type))

        lacking = next(((port, src) for port, src in nothing if needs(port) and not inputs[port.name]), None)
        # the ports of alternative wirings (NodeDef.input_choice) form one requirement: when every wired one is empty
        # (upstream computed nothing), this node has nothing to compute and gives an empty result naming the port,
        # not an error
        if lacking is None and (need := node.type.choice_inputs()):
            empties = [(port, src) for port, src in nothing if port.name in need]
            if empties and not any(inputs[name] for name in need if name in inputs):
                lacking = empties[0]
        # an empty packet that is no news: one the packet itself says was expected (an output that may give nothing,
        # Port.may_be_empty, 「创建相机」 without a focal length, marked when committed: _run_node; and whatever passed such
        # an empty on, a switch, a split, because that was all it had: _give_nothing), or one into a parameter's wire
        # (the parameter keeps its own value). Nothing is said of it; only an input that needs data and came empty from
        # what should have given some is said (the one rule, here)
        def expected(port, src) -> bool:
            return port.param or (port.name, src) in foreseen

        for port, src in nothing:
            if (port, src) != lacking and (not needs(port) or inputs[port.name]) and not expected(port, src):
                ctx.say("N-INPUT-NOTHING", port=port.name, node=node.label, input=port.label, source=self.graph.nodes[src].label)
        if ends and per_item and not ctx.items:  # every item it had gave nothing: its list is empty (scopes: gathers_empty)
            items = self.eval.item_list(self.graph.scopes.ended[nid], path)
            every = {i.key for i in items} if isinstance(items, list) else None
            if every and every <= foreseen_items:  # each of its items gave nothing, as expected: its empty list is
                # expected too, nothing said (one that failed, or gave nothing unexpected, is said as ever)
                ctx.expected = True
            else:
                ctx.say("N-EACH-NOTHING", node=node.label, count=len(gave_nothing))
        if lacking is not None:
            port, src = lacking
            if not all(expected(p, s) for p, s in nothing if needs(p) and not inputs[p.name]):
                ctx.say("N-COOK-NOTHINGIN", port=port.name, node=node.label, input=port.label, source=self.graph.nodes[src].label)
                return ctx, True
            return ctx, EXPECTED  # it gives nothing as expected as what it lacked: its empty packets say so in turn
        try:  # everything above is cooked: the values wired into parameters are known
            wired = self.eval.wired_values(nid, path)
        except ValueError as exc:
            raise CookError(nid, Msg("E-COOK-WIRED", node=node.label, reason=exc)) from None
        ctx.values = {name: v for name, (_, v) in wired.items() if v is not None}  # a table takes rows, not one value
        for w in self.eval.warnings(nid, path):  # ... and the usage checks know all they can
            emit({"type": "message", "node": nid, "path": list(path), **w})
        for said in standing_notices(node.type) + node.type.foresee(ctx.params, self.eval.info(nid, path)):  # what it could tell
            # before cooking (the size a node decides for itself), kept with its result
            ctx.say(said.code, **said.params)
        return ctx, False

    def _give_nothing(self, inst: Inst, plan: NodePlan, ctx: CookContext, emit: EventFn, started: float,
                      expected: bool = False) -> None:
        """The instance has nothing to give this time (a required input brought nothing, or it found nothing): every
        wanted output an empty packet, with what it said; downstream an empty packet counts as not connected."""
        node = self.graph.nodes[inst.node]
        for port in sorted(ctx.wanted):
            Packet(ctx.outputs[port], ctx.output_types[port], {"empty": True, **({"expected": True} if expected else {})}).commit(
                node.type.id, ctx.said, self._made_from(inst))
        with self._lock:
            self._changed(inst, cooked=True)
        emit({"type": "node_done", "node": inst.node, "path": list(ctx.path), "cached": False, "nothing": True,
              "seconds": round(time.time() - started - ctx.waited, 1),
              "frames": 0, "width": 0, "height": 0, "reused": ctx.reused, "outputs": _present(plan)})

    def _say_unused(self, inst: Inst, ctx: CookContext) -> None:  # noqa: C901
        """Parameters the cook's facts (CookContext.fact) show did nothing: W-APPLIES-UNUSED for a value the user set,
        I-APPLIES-UNUSEDDEFAULT for a default."""
        if not ctx.facts:
            return
        node = self.graph.nodes[inst.node]
        before, f = self.eval.resolved(*inst), self.eval.facts(*inst)  # the instance's, both (without the wires it goes without)
        own = {**f.own, **{k: Fact(v, node.type.fact_label(k)) for k, v in ctx.facts.items()}}
        after = resolve(node.type, NodeFacts(f.params, f.wired, own, f.incoming, f.wired_out, f.wired_data))
        defaults = param_defaults(node.type.Params)
        def labels_now():
            return {p["name"]: p["label"] for p in node.type.param_specs()}
        labels = {n: Both({lang: text for lang, text in got.items()}) for n, got in _both(labels_now).items()}
        for name, why in after.params.inactive.items():
            if name not in before.params.inactive:
                code = "W-APPLIES-UNUSED" if node.params.get(name) != defaults.get(name) else "I-APPLIES-UNUSEDDEFAULT"
                ctx.say(code, param=name, label=labels.get(name, name), reason=why)
