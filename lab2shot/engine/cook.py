"""Cooking: fingerprint every node, reuse cached packets, run the rest in order.

fingerprint(node) = hash(type id, version, result-affecting params,
                         fingerprints of its input packets, source identity,
                         the frames a frame source emits)
An output port's packet lives at work/cache/<hash(fingerprint, port)>. A parameter driven by a wire (a promoted
parameter, Graph.input_ports) is an input like any other: the value packet wired into it is among the input packets,
and the node is cooked with its value in place of the typed one (Engine.wired_values).

A graph's frame range (Graph.frames) selects the frames its sources emit (NodeDef.frame_source), each source keeping
those of its own frames that fall inside; everything downstream then computes only those. The frames are part of a
source's fingerprint, so the whole shot and a part of it are cached side by side, and a range that covers all of a
source's frames is the same cook as no range.

Several cooks may run at once (one per GPU, from different users): a node is cooked under a lock on its
fingerprint, so two cooks that need the same result compute it once and neither sees it half-written.

What is cooked is node instances (engine/scopes.py Inst: a node and its item path inside 逐项处理 blocks; the empty
path outside them). A cook works through Evaluation.order and asks it again after every instance it cooks, so what
only a cook tells (a block's items, a switch's condition, a wired value) is planned once it is known. An instance
gives the outputs something wants in this cook (CookContext.wanted): those wired into what is cooked, and the ones
shown of the target; an output already there is never written again, one no one wants is not written at all.

The read side of this (fingerprints, cached outputs, parameters, status) is engine/evaluation.py's `Evaluation`,
memoised per graph version; `Engine` here holds one and cooks on top of it (see Evaluation's docstring)."""

from __future__ import annotations

import json
import shutil
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..data.packet import Packet, fresh_dir, packet_dir, valid
from ..data.items import holds_nothing
from ..errors import CookCancelled, CookError, MessageError, NothingToCook
from ..messages import Msg
from ..nodes.applies import Fact, NodeFacts, resolve, standing_notices
from ..nodes.base import Info, NodeDef
from ..nodes.params import _defaults
from ..nodes.services import DeliverySink
from ..serving import serving

if TYPE_CHECKING:
    from ..farm.units import Units
from . import scopes as sc
from .evaluation import PLAN_ERRORS, Evaluation, NodePlan, failure_file

# Number of frames computed in parallel by core nodes whose frames are independent (CookContext.each_done). The value
# is 8: these kernels are bound by memory bandwidth, not compute, so beyond a certain thread count they stop speeding
# up and then slow down. Measured on a 32-core machine, 1080p three channels, a slightly barrel-distorted ST-map,
# 24-frame shots, minimum of 7 rounds; ms/frame, speed-up over 1 thread in brackets:
#              bicubic                     bilinear
#          full frame    banded        full frame    banded      <- banded is the current method (data/maps.py bands)
#    1    882.8 1.00x  629.0 1.00x   218.6 1.00x  163.9 1.00x
#    2    531.3 1.66x  403.0 1.56x   132.4 1.65x  104.7 1.57x
#    4    381.0 2.32x  303.3 2.07x    93.9 2.33x   70.5 2.32x
#    8    340.3 2.59x  239.9 2.62x    84.6 2.58x   63.0 2.60x   <- current
#   12    335.9 2.63x  208.4 3.02x    80.0 2.73x   58.2 2.81x
#   16    355.7 2.48x  254.9 2.47x    84.3 2.59x   61.5 2.66x
#   24    363.1 2.43x  321.9 1.95x    88.8 2.46x   73.1 2.24x
# From 4 to 8 threads bicubic gains 21% and bilinear 11%; from 16 on it regresses. 12 would be another 8-13% faster
# but is not used: more frames in flight means more memory held by a 4K shot at once, and machines often run several
# tasks concurrently, so 8 leaves headroom.
FRAME_THREADS = max(1, min(8, os.cpu_count() or 1))
from .evaluations import EVALUATIONS
from .graph import Graph
from .lint import warnings
from ..data.locks import exclusive
from .policy import CookKind
from .scopes import Inst, ItemAt, ItemPath

EventFn = Callable[[dict], None]


def _first(packets: list[Packet] | None) -> Packet | None:
    """The first packet of an input (its picture input's plate, the map a warp follows); None when nothing is there."""
    return (packets or [None])[0]


def _present(plan: NodePlan) -> dict[str, str]:
    """The node's outputs that really have a packet now (an output nobody wanted was never written): what a node_done
    event hands the page, the same answer as the status reply's `present` (Evaluation.status)."""
    return {port: fp for port, fp in plan.outputs.items() if valid(packet_dir(fp))}


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

    def __init__(self, fingerprints: dict[str, str], wanted, scratch: Path, force: bool = False) -> None:
        super().__init__()
        self._fingerprints = fingerprints
        self._scratch = scratch
        self._force = force
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
        if not self._force and valid(packet_dir(self._fingerprints[port])):
            self.kept.add(port)
            folder = self._scratch / "_unwanted" / port
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        return fresh_dir(self._fingerprints[port])

    def __missing__(self, port: str) -> Path:
        if port not in self._fingerprints:
            raise KeyError(port)
        self[port] = folder = self._folder(port)
        return folder


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
    # above came from (Evaluation.provenance): an output-settings node records it next to its files
    provenance: dict = field(default_factory=lambda: {"sources": [], "commercial": True})
    ram_gb: float = 0.0  # the system memory its worker takes at its peak (its resolved Cost): kept free for it
    delivery: DeliverySink | None = None  # where 「输出」 hands its files over: only a node that delivers gets one
    stream_worker: Callable[..., tuple] | None = None  # runs a streaming node's worker on a farm thread (Engine passes it)
    said: list[dict] = field(default_factory=list)  # the messages it said while cooking (say), kept with its result
    facts: dict = field(default_factory=dict)  # what it found out about itself while cooking (fact): name -> value
    # what this cook is known by (NodePlan.fingerprint): every output of one cook shares it (an import's folder is marked
    # with it, so its camera and its models land in one folder when packed)
    fingerprint: str = ""
    path: ItemPath = ()  # the instance's item path (engine/scopes.py): empty outside every 逐项处理 block
    # the outputs to give this time (their folders are in `outputs`): wanted downstream in this cook or shown, and not
    # there yet. A node gives exactly these; the others it neither computes nor writes
    wanted: frozenset[str] = frozenset()
    item: ItemAt | None = None  # a block's begin: the item this instance gives
    items: tuple[ItemAt, ...] = ()  # a block's end: the items it gathers, in the order its inputs' packets come

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
        from . import external

        return external.run_worker(self, image, extra=extra, inputs=inputs, reuse=reuse, params=params, record=record)

    def run_worker_streaming(self, image: Packet | None, *, extra: dict | None = None, inputs: dict | None = None,
                             record: dict | None = None) -> tuple[Path, Any, Any]:
        """A streaming node's worker (WorkerNode.streams) on a farm thread, so its convert() can follow the frames as
        they appear: returns (raw folder, done event, error()), using the farm's runner (Engine.stream_worker, farm/streaming.py)."""
        return self.stream_worker(self, image, extra=extra, inputs=inputs, record=record)

    def input_files(self, *ports: str) -> dict[str, Path]:
        """Worker inputs from these connected ports, keyed by port name (engine/external.py input_files)."""
        from . import external

        return external.input_files(self, *ports)

    def check_stop(self) -> None:
        _stopped(self.stop)

    def progress(self, done: int, total: int, message: str = "") -> None:
        """Report progress; a cook that was stopped ends here (every node that reports progress can be stopped)."""
        self.check_stop()
        self.emit({"type": "progress", "node": self.node_id, "path": list(self.path), "done": done, "total": total, "message": message})

    def each(self, items, message: str = ""):
        """Iterate and report progress after every item."""
        items = list(items)
        for i, item in enumerate(items):
            yield item
            self.progress(i + 1, len(items), message)

    def each_done(self, items, work, message: str = "", most: int = FRAME_THREADS):
        """`work(item)` for every item, several at a time, yielding the results in `items` order. For work that
        is the same whichever order it runs in and never looks at another item (「STMap」 resampling a frame, reading
        and writing that frame's pixels). The numbers are the one-at-a-time ones: the kernels are pure, so this only
        makes them come sooner (「STMap」: every frame is bit for bit the one-at-a-time run's, within its
        per-frame budget).

        Threads, not processes: the time goes into numpy's kernels and OpenImageIO's file reads, both of which let go
        of the interpreter lock. At most `most` items are in flight, so what a shot holds in memory at once does not
        grow with its length (a 4K frame is tens of MB; a few hundred frames would not fit)."""
        from concurrent.futures import ThreadPoolExecutor

        from ..io.parallel import in_flight

        items = list(items)
        if len(items) < 2 or most < 2:
            for done, item in enumerate(items, 1):
                yield work(item)
                self.progress(done, len(items), message)
            return
        # bounded concurrency is implemented once (io/parallel.py in_flight); this manages the thread pool and progress
        with ThreadPoolExecutor(most, thread_name_prefix="l2s-frame") as pool:
            for done, (_, got) in enumerate(in_flight(pool, ((i, work, item) for i, item in enumerate(items)), most), 1):
                yield got
                self.progress(done, len(items), message)

    def stage(self, name: str) -> None:
        self.emit({"type": "stage", "node": self.node_id, "path": list(self.path), "name": name})

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
        right."""
        from ..nodes import node_types

        anchors = {k: v for k, v in (("port", port), ("param", param)) if v}
        if fix:  # the same one click a usage check offers (engine/lint.py): the node inserted, named on its button
            anchors["fix"] = {"insert": fix, "label": f"插入「{node_types()[fix].label}」"}
        said = {**Msg(code, **params).json(), **anchors}
        if said not in self.said:
            self.said.append(said)
        self.emit({"type": "message", "node": self.node_id, "path": list(self.path), **said})

    def fact(self, name: str, value: Any) -> None:
        """Say a fact about this cook the node can only know now (NodeDef.fact_labels: how many chunks the shot was cut
        into). The engine resolves the node's declarations again with it and says which parameter turned out not to
        apply (W-APPLIES-UNUSED)."""
        self.facts[name] = value


class Engine:
    """Cooks a graph on top of one `Evaluation` (engine/evaluation.py), which does all the read-only work (plans,
    parameters, wired values, sources, status), memoised per node instance. `Engine`'s own evaluation is never one
    another caller shares (EvaluationCache in evaluations.py never hands out an Evaluation an Engine cooks on): cooking
    mutates a NodePlan's `.cached` in place as instances finish, which only this Engine's own run should see."""

    def __init__(self, graph: Graph, gpu: str | None = None, stop: threading.Event | None = None,
                 delivery: DeliverySink | None = None, evaluation: Evaluation | None = None,
                 stream_worker: Callable[..., tuple] | None = None):
        """`gpu`: where workers run (see CookContext.gpu); `stop`: set it to stop the cook; `delivery`: where 「输出」
        hands its files over (whoever submits the cook builds it: the farm, per job, lab2shot/transfer/deliveries.py
        Sink); without one a node that delivers is refused; `evaluation`: reuse this one for reading instead of building
        a fresh one, never one shared with anything else that might also cook, since cooking mutates it (see the class
        docstring). `stream_worker`: who runs a streaming node's worker on a thread and returns (raw folder, done,
        error()); the farm injects its own (farm/streaming.py), so the engine never opens a thread itself;
        without one a streaming node cooks with the worker blocking. Whose the files are read as is the
        evaluation's account (Evaluation.account): an upload that is not it lets its node fail at itself, as one that
        was cleaned away does.

        View proxies are not made in the engine: after a packet is computed, the farm hands it to a CPU thread pool
        after `node_done` (`lab2shot/farm/queue.py _proxies_of`), so the GPU does not wait and the node's time does not
        include it; the task itself still ends only once the proxies are done. The engine has no knowledge of the
        viewing layer."""
        self.graph = graph
        self.gpu = gpu
        self.stop = stop or threading.Event()
        self.delivery = delivery
        self.stream_worker = stream_worker
        self.eval = evaluation or Evaluation(graph)
        self.failed: dict[Inst, CookError] = {}  # instances that failed in this engine's cooks: said once
        self.skipped: set[Inst] = set()  # instances said to be skipped in them

    # ------------------------------------------------------------------ reading (Evaluation does the work)

    def params(self, node_id: str, path: ItemPath = ()) -> dict:
        return self.eval.params(node_id, path)

    def wired_values(self, node_id: str, path: ItemPath = ()) -> dict[str, tuple[Any, Any]]:
        return self.eval.wired_values(node_id, path)

    def sources(self, node_id: str, path: ItemPath = ()) -> dict[str, str]:
        return self.eval.sources(node_id, path)

    def known(self, node_id: str, port: str, path: ItemPath = ()) -> Packet | None:
        return self.eval.known(node_id, port, path)

    def info(self, node_id: str, path: ItemPath = ()) -> Info:
        return self.eval.info(node_id, path)

    def work(self, node_id: str, path: ItemPath = ()) -> Info:
        return self.eval.work(node_id, path)

    def frame_range(self, targets: list[str]) -> tuple[int, int] | None:
        return self.eval.frame_range(targets)

    def check_frames(self, targets: list[str]) -> None:
        self.eval.check_frames(targets)

    def plan(self, node_id: str, path: ItemPath = ()) -> NodePlan:
        return self.eval.plan(node_id, path)

    def computes(self, targets: list[str], force: bool = False) -> list[str]:
        return self.eval.computes(targets, force)

    def kind(self, targets: list[str], force: bool = False) -> CookKind:
        return self.eval.kind(targets, force)

    def needs_ram(self, targets: list[str], force: bool = False) -> float:
        return self.eval.needs_ram(targets, force)

    def status(self) -> dict:
        return self.eval.status()

    # ------------------------------------------------------------------ cooking

    def cook(self, target: str, emit: EventFn | None = None, force: bool = False,
             show: frozenset[str] | None = None, units: "Units | None" = None) -> dict[str, Packet]:
        """Cook every instance of `target` and what they need, and return the target's output packets (a target
        outside every block: the ones given; inside one, nothing: every instance's are in the status).
        `show`: the target's outputs shown (None: every one of them); the target gives these, every other instance
        what is wired on from it in this cook (CookContext.wanted).

        `units`: this cook is one of several running the same targets side by side, one per card (farm/units.py
        计算单元). It says which instances are this one's (`mine`, by their item path: a 计算单元 is one
        item of the innermost block) and waits for the ones another card took (`wait`, True once that card is through
        with it; False: nobody took it after all, so this cook does it). Without it a cook does every instance itself.

        An error is the failing instance's alone (like Nuke: the
        node that errors is red, the ones that need it wait): it is said once, where it happened, and kept with the
        instance's fingerprint (evaluation.failure_file); an instance whose required input comes from it is skipped
        (N-COOK-SKIPPED), one that only takes it on an optional input is cooked without it (W-INPUT-UNUSED), a block's
        end gathers the other items (W-EACH-FAILED); every other branch and item is cooked on to the end. Only then,
        when `target` itself has no result, the error at its root is raised (a script, the command line and the tests
        get it as before). An instance that failed before is tried again."""
        emit = emit or (lambda e: None)
        for nid in self.graph.upstream_order(target):
            self.graph.check_inputs(nid)
        self.check_frames([target])
        shown = frozenset(p.name for p in self.graph.outputs(target)) if show is None else frozenset(show)
        done: set[Inst] = set()
        while True:  # the order is asked again after every step: what a cook tells plans what comes after
            _stopped(self.stop)
            order, _ = self.eval.order([target])
            todo = [i for i in order if i not in done]
            if not todo:
                break
            if units is None:
                inst = todo[0]
            else:
                inst = self._take(todo, done, units)
                if inst is None:  # everything left is another card's: wait for the first of them to be through
                    blocked = next((i for i in todo if units.others(i)), todo[0])
                    if units.wait(blocked):  # what that card left is on disk: plan on from there
                        done.add(blocked)
                        self.eval.forget({blocked.node})
                        continue
                    inst = blocked  # nobody came for it after all: this card takes it
            done.add(inst)
            self._step(inst, emit, target, shown, force and inst.node == target)
        emit({"type": "done", "node": target, "path": []})
        paths, pending = self.eval.instances(target)
        outcomes = [o for p in paths if (o := self.eval.outcome(target, p)) is not None]
        outcomes += [o for w in pending if (o := self.eval.outcome(*w.on)) is not None]
        if outcomes and len(outcomes) >= len(paths):
            root = outcomes[0]
            raise next((e for i, e in self.failed.items() if i.node == root.root), None) or CookError(
                root.root, Msg(root.message["code"], **root.message.get("params", {})))
        if paths != [()]:
            return {}
        outputs = self.plan(target).outputs
        return {port: Packet.load(packet_dir(outputs[port])) for port in outputs if port in shown}

    def _take(self, todo: list[Inst], done: set[Inst], units: "Units") -> Inst | None:
        """The next instance this card may cook: the first one every instance it takes something from is through with
        (here or on another card) and that the ledger gives this card (farm/units.py). None: everything left belongs
        to another card, or waits for one."""
        for inst in todo:
            sources = [Inst(src, p) for wires in self.eval.wires(*inst).values() for src, _sport, p in wires]
            sources += [w.on for w in self.eval.waits(*inst)]
            if all(s in done or units.done(s) for s in sources) and units.mine(inst):
                return inst
        return None

    def _wanted(self, inst: Inst, target: str, shown: frozenset[str]) -> frozenset[str]:
        """The outputs of `inst` this cook wants: one answer for the whole cook, `Evaluation.demand` (`computes`,
        policy and the queue's cache marks read the same table)."""
        return self.eval.demand([target], shown).get(inst, frozenset())

    def _step(self, inst: Inst, emit: EventFn, target: str, shown: frozenset[str], force: bool) -> None:
        if inst in self.failed:  # failed in this engine's run already: said once
            return
        outcome = self.eval.outcome(*inst)
        if outcome is not None and outcome.state == "skipped":
            if inst not in self.skipped:
                self.skipped.add(inst)
                emit({"type": "skipped", "node": inst.node, "path": list(inst.path), "root": outcome.root, **outcome.message})
            return
        if outcome is not None:  # it failed before: try again, and what goes without it wants it again
            try:
                failure_file(self.plan(*inst).fingerprint).unlink(missing_ok=True)
            except PLAN_ERRORS:  # it has no plan at all (its file is not there): _cook_node says so at the node
                pass
            self._changed(inst.node)
        try:
            with serving(self.eval.account):  # it reads the files as the account this cook is for (uploads.resolve)
                self._cook_node(inst, emit, force, self._wanted(inst, target, shown))
        except CookError as exc:
            self._fail(inst, exc, emit)

    def _downstream(self, nid: str) -> set[str]:
        """`nid` and every node that takes, directly or not, what it gives."""
        seen, todo = {nid}, [nid]
        while todo:
            for dst, _ in self.graph.outputs_by_node.get(todo.pop(), ()):
                if dst not in seen:
                    seen.add(dst)
                    todo.append(dst)
        return seen

    def _changed(self, nid: str) -> None:
        """What is known of an instance of `nid` changed (cooked, failed, tried again): forget the node and what comes
        after (every instance of them: a block's items, a switch's inputs and the order follow from it)."""
        self.eval.forget(self._downstream(nid))
        EVALUATIONS.bump()

    def _fail(self, inst: Inst, exc: CookError, emit: EventFn) -> None:
        """The instance failed: said once, kept with its fingerprint, and what needs it is skipped from here on."""
        said = {**exc.message.json(), **({"param": exc.param} if exc.param else {}), "log": str(exc.log) if exc.log else None}
        try:
            path = failure_file(self.plan(*inst).fingerprint)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(said, ensure_ascii=False), encoding="utf-8")
        except (OSError, ValueError, CookError):  # an instance that can't even be planned: its status says why already
            pass
        self.failed[inst] = exc
        self._changed(inst.node)
        emit({"type": "error", "node": inst.node, "path": list(inst.path), **said})

    def _to_give(self, nid: str, plan: NodePlan, wanted: frozenset[str], force: bool) -> frozenset[str]:
        """The wanted outputs to write: those not there yet, every one when forced (a block's item is its list's
        packet: never written again)."""
        t = self.graph.nodes[nid].type
        item = t.item_output if sc.role(t) == sc.BEGIN else None
        give = {p for p in wanted if p in plan.outputs and p != item
                and (force or not valid(packet_dir(plan.outputs[p])))}
        # an output made from another output of the same node (point cloud = depth map + camera): its source is written
        # too, even if already on disk, because the node derives it in process from the freshly computed data rather than
        # reading it back from disk (Port.made_from)
        if give:
            outs = {p.name: p for p in self.graph.outputs(nid)}
            for name in list(give):
                give.update(n for n in getattr(outs.get(name), "made_from", ()) if n in plan.outputs and n != item)
        return frozenset(give)

    def _cook_node(self, inst: Inst, emit: EventFn, force: bool, wanted: frozenset[str]) -> None:
        nid, path = inst
        if (gone := self.eval.source_missing(nid, path)) is not None:  # its file is not there (cleaned, or not this
            raise CookError(nid, Msg(gone["code"], **gone.get("params", {})))  # account's): what needs it is skipped
        node = self.graph.nodes[nid]
        plan = self.plan(nid, path)
        is_sink = not self.graph.outputs(nid)
        where = {"node": nid, "path": list(path)}
        if not force and not is_sink and not self._to_give(nid, plan, wanted, force):
            emit({"type": "node_done", **where, "cached": True, "seconds": 0, "outputs": _present(plan)})
            return
        emit({"type": "node_start", **where, "label": node.label})
        waiting = lambda: emit({"type": "stage", **where, "name": "另一个任务正在算同一个节点，等它算完"})  # noqa: E731
        with exclusive(plan.fingerprint, waiting, lambda: _stopped(self.stop)):
            # another cook may have produced it while this one planned or waited
            give = self._to_give(nid, plan, wanted, force)
            if not force and not is_sink and not give:
                plan.cached = self._settled(nid, plan)
                emit({"type": "node_done", **where, "cached": True, "seconds": 0, "outputs": _present(plan)})
                return
            # the lock of every output packet folder is also held until the node finishes. The two places that clean
            # incomplete packets (farm/disk.py clean_incomplete, sweep_incomplete) try the packet folder's own name (the
            # output fingerprint `_hash([fp, port])`), not the node fingerprint; holding only the node lock, cancelling a
            # not-yet-started task of the same graph or starting another process would delete the folder being written
            # here, and packet.commit would then raise FileNotFoundError. Lock order is fixed (node first, then outputs
            # sorted by fingerprint), so there is no deadlock.
            # The item port of 「逐项开始」 is the upstream list's packet, not written by this node, and is not locked.
            item = node.type.item_output if sc.role(node.type) == sc.BEGIN else None
            with ExitStack() as held:
                for fp in sorted({fp for port, fp in plan.outputs.items() if port != item}):
                    held.enter_context(exclusive(fp, waiting, lambda: _stopped(self.stop)))
                self._run_node(inst, plan, emit, give, force)

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

    def _input_packet(self, fp: str) -> Packet:
        """An input packet a node is about to cook with: marked used now (the cache keeps what a cook actually
        relies on; status and plan never touch this, see Evaluation's docstring) and read through the evaluation
        (a manifest already read while planning is not read a second time)."""
        Packet.used(packet_dir(fp))
        return self.eval.packet(fp)

    def _settled(self, nid: str, plan: NodePlan) -> bool:
        """Whether every output something needs of the node is there now (NodePlan.cached: the one formula, `has`)."""
        return plan.has(self.eval.needed_outputs(nid))

    def _run_node(self, inst: Inst, plan: NodePlan, emit: EventFn, wanted: frozenset[str], force: bool = False) -> None:
        nid, path = inst
        node = self.graph.nodes[nid]
        if node.type.project.extension is not None:  # the status said it before the cook (Evaluation.failure)
            why = node.type.project.available()
            if why is not None:  # never cooked, said once here
                raise CookError(nid, Msg("E-COOK-UNAVAILABLE", node=node.label, reason=why))
        started = time.time()
        # every input's packets: without the wires whose source failed (taken) and without the ones that bring
        # nothing: no data this time (an empty packet), or data with nothing in it where the input does not take that
        taken = self.eval.taken(nid, path)
        begins, ends = sc.role(node.type) == sc.BEGIN, self.graph.scopes.ended.get(nid) is not None
        inputs, nothing = {}, []
        # items of the block that produced nothing this time (their result is an empty packet). They differ from failed
        # or skipped items (an empty result is not an error), but for 「逐项结束」 they likewise have no result to
        # collect. If they were not removed from `items`, the item count would exceed the packets received,
        # `EachEnd.cook` would compute 0 wires per item from packets / items, raise IndexError and lose the computed
        # items as well
        gave_nothing: set[str] = set()
        for port in self.graph.input_ports(nid):
            inputs[port.name] = []
            for src, sport, p in taken[port.name]:
                packet = self._input_packet(self.plan(src, p).outputs[sport])
                if packet.meta.get("empty") or (not port.takes_empty and holds_nothing(packet.type, packet.meta)):
                    nothing.append((port, src))
                    if ends and len(p) > len(path):
                        gave_nothing.add(p[len(path)])
                else:
                    inputs[port.name].append(packet)
        work = packet_dir(plan.fingerprint + "_work")
        work.mkdir(parents=True, exist_ok=True)
        outputs = Outputs(plan.outputs, wanted, work, force)  # the wanted ones, made now; another the node asks for anyway, then
        if node.type.delivers and self.delivery is None:
            raise CookError(nid, Msg("E-COOK-NODELIVERY", node=node.label))
        ctx = CookContext(nid, node.label, node.type, self.params(nid, path), inputs, outputs,
                          {port: self.graph.output_type(nid, port) for port in plan.outputs}, work, emit, self.gpu, self.stop,
                          fingerprint=plan.fingerprint,
                          frames=self.info(nid, path).frames if node.type.frame_source else (), sources=self.sources(nid, path), wired_from=self.eval.wired_from(nid, path),
                          provenance=self.eval.provenance(nid, path), ram_gb=self.eval.resolved(nid, path).cost.ram_gb,
                          delivery=self.delivery if node.type.delivers else None,
                          stream_worker=self.stream_worker, path=path, wanted=wanted,
                          item=self.eval.item_at(nid, path) if begins else None,
                          items=tuple(i for i in self.eval.gathered(nid, path) if i.key not in gave_nothing) if ends else ())
        for said, port in self.eval.unused_inputs(nid, path):
            ctx.say(said.code, port=port, **said.params)
        from ..nodes.applies import overscan_notices

        taken_packets = [(port.name, port.label, packet) for port in self.graph.input_ports(nid) for packet in inputs[port.name]]
        for port, said in overscan_notices(node.type, taken_packets):  # a pixel node given pixels past the format: what it leaves out
            ctx.say(said.code, port=port, **said.params)
        lacking = next(((port, src) for port, src in nothing if not port.optional and not inputs[port.name]), None)
        # the ports of alternative wirings (NodeDef.input_choice) form one requirement: when every wired one is empty
        # (upstream computed nothing), this node has nothing to compute and gives an empty result naming the port,
        # not an error
        if lacking is None and (need := node.type.choice_inputs()):
            empties = [(port, src) for port, src in nothing if port.name in need]
            if empties and not any(inputs[name] for name in need if name in inputs):
                lacking = empties[0]
        for port, src in nothing:
            if (port, src) != lacking and (port.optional or inputs[port.name]):
                ctx.say("N-INPUT-NOTHING", port=port.name, node=node.label, input=port.label, source=self.graph.nodes[src].label)
        if lacking is not None:  # a required input with nothing: nothing to cook, and nothing to give
            port, src = lacking
            ctx.say("N-COOK-NOTHINGIN", port=port.name, node=node.label, input=port.label, source=self.graph.nodes[src].label)
            return self._give_nothing(nid, plan, ctx, emit, started)
        try:  # everything above is cooked: the values wired into parameters are known
            wired = self.wired_values(nid, path)
        except ValueError as exc:
            raise CookError(nid, Msg("E-COOK-WIRED", node=node.label, reason=exc)) from None
        ctx.values = {name: v for name, (_, v) in wired.items() if v is not None}  # a table takes rows, not one value
        for w in warnings(self.eval, nid, path):  # ... and the usage checks know all they can
            emit({"type": "message", "node": nid, "path": list(path), **w})
        for said in standing_notices(node.type) + node.type.foresee(ctx.params, self.info(nid, path)):  # what it could tell
            # before cooking (a pinhole node: needs undistorted plates), kept with its result
            ctx.say(said.code, **said.params)
        try:
            produced = node.type.cook(ctx) or {}
        except NothingToCook as found:  # it found nothing to give: not an error
            ctx.say(found.message.code, **found.message.params)
            return self._give_nothing(nid, plan, ctx, emit, started)
        except (CookError, CookCancelled):
            raise
        except Exception as exc:
            (work / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            # a refusal raised by core code keeps its own message, code and parameters (the worker side does the same,
            # engine/external.py); an exception with no words of its own (StopIteration, KeyError()) still says what
            # failed, never 「出错：」
            said = exc.message if isinstance(exc, MessageError) else (str(exc) or type(exc).__name__)
            raise CookError(nid, Msg("E-COOK-FAILED", node=node.label, reason=said), work / "error.log") from exc
        self._settle_outputs(nid, produced, ctx.inputs, ctx.params)
        self._say_unused(inst, ctx)
        for port, packet in produced.items():
            if port in outputs.kept:  # written beside a complete packet nobody asked for: that one stays, this goes with _work
                continue
            packet.commit(node.type.id, ctx.said)
        missing = set(wanted) - set(produced)
        if missing:
            raise CookError(nid, Msg("E-COOK-MISSINGOUTPUT", node=node.label, ports=sorted(missing)))
        plan.cached = self._settled(nid, plan)  # a later cook() on this engine reuses it
        # after success the scratch folder (`<fp>_work`: the worker's input list, error.log) is no longer needed; on
        # failure it is kept (error.log is its log)
        shutil.rmtree(packet_dir(plan.fingerprint + "_work"), ignore_errors=True)
        self._changed(nid)  # a cache generation of its own: status/plan reads made while this was cooking may now be stale
        size = self.work(nid, path)  # with the seconds, what the farm's timing records keep (lab2shot/farm/timings.py)
        emit({"type": "node_done", "node": nid, "path": list(path), "cached": False, "seconds": round(time.time() - started - ctx.waited, 1),
              "frames": len(size.frames), "width": size.width, "height": size.height, "reused": ctx.reused,
              "outputs": _present(plan)})  # its port fingerprints: the page writes them in without asking again

    def _give_nothing(self, nid: str, plan: NodePlan, ctx: CookContext, emit: EventFn, started: float) -> None:
        """The instance has nothing to give this time (a required input brought nothing, or it found nothing): every
        wanted output an empty packet, with what it said; downstream an empty packet counts as not connected."""
        node = self.graph.nodes[nid]
        for port in sorted(ctx.wanted):
            Packet(ctx.outputs[port], ctx.output_types[port], {"empty": True}).commit(node.type.id, ctx.said)
        plan.cached = self._settled(nid, plan)
        self._changed(nid)
        emit({"type": "node_done", "node": nid, "path": list(ctx.path), "cached": False, "nothing": True,
              "seconds": round(time.time() - started - ctx.waited, 1),
              "frames": 0, "width": 0, "height": 0, "reused": ctx.reused, "outputs": _present(plan)})

    def _say_unused(self, inst: Inst, ctx: CookContext) -> None:
        """Parameters the cook's facts (CookContext.fact) show did nothing: W-APPLIES-UNUSED for a value the user set,
        I-APPLIES-UNUSEDDEFAULT for a default."""
        if not ctx.facts:
            return
        node = self.graph.nodes[inst.node]
        before, f = self.eval.resolved(*inst), self.graph.facts(inst.node)
        own = {**f.own, **{k: Fact(v, node.type.fact_labels.get(k, k)) for k, v in ctx.facts.items()}}
        after = resolve(node.type, NodeFacts(f.params, f.wired, own, f.incoming, f.wired_out))
        defaults = _defaults(node.type.Params)
        labels = {p["name"]: p["label"] for p in node.type.param_specs()}
        for name, why in after.params.inactive.items():
            if name not in before.params.inactive:
                code = "W-APPLIES-UNUSED" if node.params.get(name) != defaults.get(name) else "I-APPLIES-UNUSEDDEFAULT"
                ctx.say(code, param=name, label=labels.get(name, name), reason=why)
