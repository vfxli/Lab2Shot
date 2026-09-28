"""Run an extension node's worker in the extension's own environment.

The worker gets a job.json (schema lab2shot.job/1, defined in the worker SDK):
"node": which node to run, "params", "inputs": paths of other input payloads,
and "frames": the display-referred PNGs of the input image packet, listed relative to the job folder. The worker is sent
the job file by its absolute path and gets its install locations in its environment (lab2shot_shared.protocol: every
path a worker is given is explicit). It runs in the extension's resident process (resident.py), which keeps its models loaded for the
next job.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from lab2shot_worker import RAW, SCHEMA  # the job protocol, defined once (the worker side reads the same)

from ..config import WORKER_SDK_DIR
from ..progress import FETCHING as PROGRESS_FETCHING, LOADING as PROGRESS_LOADING
from ..io.atomic import flush_tree, mark, write_text
from ..io.digest import sha256
from ..data.locks import exclusive
from ..data.packet import CACHE_VERSION, WORKER_COMPLETE, Packet, cache_root, note, packet_dir, used, worker_done
from ..data.payloads import frames_for_worker, image_files, is_data
from ..data.units import DEFAULT_FPS
from ..messages import Msg, worker_params
from ..errors import CookCancelled, NothingToCook

STACK_LINES = 12  # number of trailing worker log lines (the stack) included in an error, unedited


def job_code(job_dir: Path) -> str:
    """The error code: shown to the user in the error message and passed to the administrator, who looks up the full
    log with `lab2shot admin joblog <code>`.

    It is the first 8 characters of the job folder name (`<key>_job`); no separate mapping is stored, the code itself
    locates the folder."""
    return job_dir.name.removesuffix("_job")[:8]

from ..nodes.applies import resolve_params
from ..nodes.params import _defaults
from .cook import CookContext, CookError
from .resident import event_of, keep_free_gb, pool

if TYPE_CHECKING:
    from ..extensions import Extension
    from ..nodes.base import NodeDef


def input_files(ctx: CookContext, *ports: str) -> dict[str, Path]:
    """Worker inputs from the node's connected ports, keyed by port name: 人物框 -> its boxes.json; per-frame
    maps (masks, ...) -> work/<port>_frames.json = {"frames": {"<frame>": file}}; pictures the same, as the display-
    referred PNGs a worker reads (like the job's own frames: another picture to match, RoMa v2). Unconnected ports are
    left out."""
    out = {}
    for port in ports:
        packet = ctx.input(port)
        if packet is None:
            continue
        if packet.type == "boxes":
            out[port] = packet.path("boxes.json")
        else:
            # pictures (image.3 / image.4, not data maps) are converted to the display PNGs solvers expect. The type is
            # `image.<channels>` and must not be tested with `== "image"`, otherwise extra wired pictures would reach the
            # solver unconverted (EXR, other colour spaces). Data maps (masks and the like) stay as the original files:
            # solvers read values, not colours
            picture = packet.type.startswith("image.") and not is_data(packet)
            files = image_files(frames_for_worker(packet) if picture else packet)
            listing = ctx.work / f"{port}_frames.json"
            # frame paths relative to the listing file's folder (the worker reads them back next to it)
            listing.write_text(json.dumps({"frames": {str(f): os.path.relpath(p, ctx.work) for f, p in files.items()}}),
                               encoding="utf-8")
            out[port] = listing
    return out


def _identity(value: Path) -> str:
    """What a worker input file holds. A file of a cooked packet: its place in the cache, which names it (the packet's
    folder is its fingerprint, and a packet never changes), without reading a byte; another file: its bytes when small,
    else size + modification time."""
    try:
        inside = Path(value).resolve().relative_to(cache_root().resolve())
    except ValueError:
        inside = None
    if inside is not None:
        # a node's own scratch file (`<node fingerprint>_work/..._frames.json`) is not a packet: its path contains the
        # node fingerprint, which changes with every parameter affecting the result, so identifying it by path would
        # rerun the model for parameters that only change the output format. It is a few lines of JSON, identified by content
        if inside.parts and inside.parts[0].endswith("_work"):
            return sha256(value)
        return "packet:" + inside.as_posix()
    st = value.stat()
    if value.is_file() and st.st_size <= 64 << 20:
        return sha256(value)
    return f"{value}:{st.st_size}:{st.st_mtime_ns}"


def _stamps(paths) -> tuple[tuple[str, int, int], ...]:
    """What tells a file changed without reading it: (path, modification time, size) each."""
    return tuple((str(p), (st := Path(p).stat()).st_mtime_ns, st.st_size) for p in paths)


@functools.lru_cache(maxsize=256)
def _sdk_walk(own: tuple, sdk: tuple, sdk_dir: str) -> tuple[str, ...]:
    """sdk_modules for these files, remembered while neither they nor any SDK module changed (the stamps are the key)."""
    return tuple(str(p) for p in sdk_modules([Path(f) for f, _, _ in own]))


@functools.lru_cache(maxsize=256)
def _code_hash(stamps: tuple) -> str:
    """The hash of these files' contents, read once while none of them changed (the stamps are the key)."""
    return sha256(*[Path(f) for f, _, _ in stamps])


def _image_identity(image: Packet | None):
    if image is None:
        return None
    if image.dir.parent == cache_root():  # a cooked packet: its folder is named by its fingerprint
        return image.fingerprint
    return {str(f): _identity(p) for f, p in image_files(image).items()}  # made by the node itself (e.g. a probe frame)


def worker_code(ext: Extension) -> list[Path]:
    """The files a worker runs: its worker.py, the adapter modules it imports (its Extension.worker_modules, and those
    of the extensions it requires) and the worker SDK modules those import, followed through the SDK. A change to an
    SDK module one worker does not use leaves that worker's results and kept processes alone."""
    own = [ext.worker_script, *(e.adapter_dir / m for e in (ext, *ext.required_extensions) for m in e.worker_modules)]
    sdk = sorted(p for package in SDK_PACKAGES for p in (WORKER_SDK_DIR / package).glob("*.py"))
    return own + [Path(p) for p in _sdk_walk(_stamps(own), _stamps(sdk), str(WORKER_SDK_DIR))]  # stat only, parsed once


SDK_PACKAGES = ("lab2shot_worker", "lab2shot_shared")  # the worker SDK: the job protocol and families, the shared maths


def sdk_modules(files: list[Path]) -> list[Path]:
    """The worker SDK modules `files` import, directly or through other SDK modules, in both of its packages
    (lab2shot_worker, lab2shot_shared: one re-exports the other's modules, and each imports itself relatively). A
    package's __init__.py comes with any of its modules (importing a module runs its package); lab2shot_worker's
    always (the worker imports it to run at all)."""
    import ast

    def package_of(path: Path) -> str | None:
        return path.parent.name if path.parent.parent == WORKER_SDK_DIR and path.parent.name in SDK_PACKAGES else None

    def imported(path: Path) -> set[str]:
        own = package_of(path)
        names: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level == 1 and own:  # relative: within the package the file is in
                    base = f"{own}.{node.module}" if node.module else own
                elif node.level == 0:
                    base = node.module or ""
                else:
                    continue
                names.add(base)
                names |= {f"{base}.{a.name}" for a in node.names}  # `from lab2shot_shared import units` names a module
        return {n for n in names if n.split(".")[0] in SDK_PACKAGES}

    def module_files(name: str) -> list[Path]:
        """The files importing `name` runs: its package's __init__.py and, for a module, its own file."""
        package, *rest = name.split(".")
        out = [WORKER_SDK_DIR / package / "__init__.py"]
        if rest:
            out.append(WORKER_SDK_DIR / package / f"{rest[0]}.py")
        return [f for f in out if f.is_file()]

    first = WORKER_SDK_DIR / "lab2shot_worker" / "__init__.py"
    seen: dict[Path, None] = {first: None}
    todo = [*files, first]
    while todo:
        path = todo.pop()
        for name in imported(path):
            for f in module_files(name):
                if f not in seen:
                    seen[f] = None
                    todo.append(f)
    return sorted(seen)


def worker_identity(ext: Extension) -> dict:
    """What a worker is: its code, the environment it runs in (as installed) and the weights it loads. A resident
    process whose worker identity changed is replaced by a new one."""
    return {
        "extension": ext.name,
        "commit": ext.source.commit,
        "code": _code_hash(_stamps(worker_code(ext))),  # every file of code it runs, in order; read again only once one changed
        "env": (ext.install_state().get("env") or {}).get("fingerprint"),
        "weights": [[w.key, w.source, w.revision, w.sha256] for w in ext.weights],
    }


def job_key(ext: Extension, node: str, image: Packet | None, params: dict, inputs: dict) -> str:
    """Identity of a worker job: everything the worker receives and the worker itself (worker_identity). Node
    parameters that only change how the node writes the results (units, thresholds, point density, ...) are not in
    it, so changing them reuses the model's raw results instead of running it again."""
    ident = {
        "cache": CACHE_VERSION,
        **worker_identity(ext),
        "node": node,
        "image": _image_identity(image),
        "params": params,
        "inputs": {k: _identity(Path(v)) for k, v in inputs.items()},
    }
    return sha256(json.dumps(ident, sort_keys=True, default=str))[:24]


def job_params(ctx: CookContext, extra: dict | None = None) -> dict:
    """What a worker gets: the node's worker parameters (NodeDef.worker_params) with `extra` on top."""
    return {**ctx.node_type.worker_params(ctx.params), **(extra or {})}


def job_record(ctx: CookContext, record: dict | None = None) -> dict:
    """What a job file records besides the job: where the node's values came from (its lens, and every parameter driven
    by a wire or overriding an input: CookContext.sources)."""
    return {**({"lens": record} if record else {}), "params": ctx.sources}


def raw_folder(key: str) -> Path:
    """A worker job's raw folder: cache/<job key>_job/raw (the job folder's RAW, as the worker names it). run_job
    writes it; farm/streaming.py names it the same way to follow a worker's frames while it is still running."""
    return packet_dir(key + "_job") / RAW


def job_folder(ctx: CookContext, image: Packet | None, *, extra: dict | None = None, inputs: dict | None = None) -> Path:
    """The raw folder the node's worker job writes into, named without running the job (the same key run_job works
    out): what streaming needs to read frames as they appear."""
    ext = ctx.node_type.project.extension
    key = job_key(ext, ctx.node_type.id, image, job_params(ctx, extra), {k: Path(v) for k, v in (inputs or {}).items()})
    return raw_folder(key)


@dataclass(frozen=True)
class WorkerJob:
    """One job for a node type's worker, whoever asks for it (a cook, or planning outside any cook): everything the
    worker receives, and what the job file records besides."""

    node_type: type[NodeDef]
    params: dict  # exactly what the worker gets
    image: Packet | None = None  # the frames to process
    inputs: dict[str, Path] = field(default_factory=dict)  # other files it reads
    reuse: bool = True  # False: always run it (a worker that writes a delivery)
    record: dict = field(default_factory=dict)  # where the values came from, outside the job key
    ram_gb: float = 0.0  # the system memory the worker takes at its peak: kept free for it


class RunnerEnv(Protocol):
    """What runs a job reports to and stops by: a cook's context (CookContext), or nothing (ask_worker)."""

    node_id: str
    label: str
    gpu: str | None
    stop: threading.Event
    reused: bool

    def stage(self, name: str) -> None: ...
    def phase(self, name: str) -> None: ...
    def progress(self, done: int, total: int, message: str = "") -> None: ...
    def say(self, code: str, /, **params: Any) -> None: ...
    def check_stop(self) -> None: ...
    def exclusive(self, key: str, waiting: str) -> Any: ...


def run_worker(ctx: CookContext, image: Packet | None, *, extra: dict | None = None, inputs: dict | None = None,
               reuse: bool = True, params: dict | None = None,
               record: dict | None = None, env: RunnerEnv | None = None) -> Path:
    """Run the cooking node's worker; returns the raw folder it wrote into. `image`: the frames to process (None for
    file-only nodes). The worker gets the node's worker parameters (NodeDef.worker_params: the node's definition is
    their only one) with `extra` on top: values the node computes (a lens from a connected camera, the chosen probe
    frame, a file to write ...). `params`: exactly these instead, for a smaller job a node runs besides its main one
    and wants cached by what it needs alone (Kimodo's text encoding, reused by every shot with the same prompt).
    `inputs`: other files it reads. reuse=False always runs it (workers that write a delivery). The job file also
    records where the node's values came from (`record`, e.g. the lens it worked out, and CookContext.sources: every
    parameter driven by a wire or overriding an input), outside the job key: the same job from values that came another
    way reuses the model's results. `env`: what the job reports to and is stopped by, when not the cook's context
    itself (a streaming node's worker: Halting)."""
    job = WorkerJob(ctx.node_type, job_params(ctx, extra) if params is None else params, image,
                    {k: Path(v) for k, v in (inputs or {}).items()}, reuse, job_record(ctx, record), ctx.ram_gb)
    return run_job(job, env or ctx)


class _EitherStop(threading.Event):
    """Set by itself or by `also`. Only is_set() looks at both, and is_set() is all a job asks of its stop."""

    def __init__(self, also: threading.Event) -> None:
        super().__init__()
        self.also = also

    def is_set(self) -> bool:
        return super().is_set() or self.also.is_set()


class Halting:
    """A cook's job run on a thread of its own (a streaming node's worker, farm/streaming.py; RunnerEnv): it reports
    through the cook's context and stops when the node stops, or when the node gives up on it (`halt`: its convert()
    failed). Setting the node's own stop instead would make that failure read as the node being stopped."""

    def __init__(self, ctx: CookContext) -> None:
        self.ctx = ctx
        self.node_id, self.label, self.gpu = ctx.node_id, ctx.label, ctx.gpu
        self.stop = _EitherStop(ctx.stop)

    def halt(self) -> None:
        self.stop.set()

    @property
    def reused(self) -> bool:
        return self.ctx.reused

    @reused.setter
    def reused(self, value: bool) -> None:
        self.ctx.reused = value

    def stage(self, name: str) -> None:
        self.ctx.stage(name)

    def phase(self, name: str) -> None:
        self.ctx.phase(name)

    def progress(self, done: int, total: int, message: str = "") -> None:
        self.check_stop()
        self.ctx.progress(done, total, message)

    def say(self, code: str, /, **params: Any) -> None:
        self.ctx.say(code, **params)

    def check_stop(self) -> None:
        if self.stop.is_set():
            raise CookCancelled()

    @contextmanager
    def exclusive(self, key: str, waiting: str) -> Iterator[None]:
        start = time.time()
        with exclusive(key, lambda: self.stage(waiting), self.check_stop):
            self.ctx.waited += time.time() - start  # waiting for another cook is not the node's computing (CookContext.exclusive)
            yield


# the stop of the cook this thread is running (Engine.cook sets it): a job run outside any node while it runs stops with it
_cook_stop: ContextVar[threading.Event | None] = ContextVar("lab2shot_cook_stop", default=None)


@contextmanager
def stopped_by(stop: threading.Event) -> Iterator[None]:
    """A job run outside any node's cook from here on (ask_worker: an import node inside a 逐项处理 block, planned only
    once the cook knows its file) stops when `stop` is set, and so does its wait for another cook of the same job."""
    token = _cook_stop.set(stop)
    try:
        yield
    finally:
        _cook_stop.reset(token)


class _Quiet:
    """A job run outside any node's cook: nothing follows its progress, no GPU; it stops with the cook it runs in
    (stopped_by), when it runs in one (RunnerEnv)."""

    node_id = "ask"
    gpu = ""

    def __init__(self, label: str) -> None:
        self.label, self.stop, self.reused = label, _cook_stop.get() or threading.Event(), False

    def stage(self, name: str) -> None:
        pass

    def phase(self, name: str) -> None:
        pass

    def progress(self, done: int, total: int, message: str = "") -> None:
        pass

    def say(self, code: str, /, **params: Any) -> None:
        pass

    def check_stop(self) -> None:
        if self.stop.is_set():
            raise CookCancelled()

    @contextmanager
    def exclusive(self, key: str, waiting: str) -> Iterator[None]:
        with exclusive(key, check=self.check_stop):
            yield


def ask_worker(node_type: type[NodeDef], params: dict, inputs: dict[str, Path]) -> Path:
    """Run `node_type`'s worker job outside a cook, with `params` as its parameters: to learn what a file holds before
    anything is cooked (the entries of a file, for an import node's lists and its check before a cook: PlanEnv.ask_worker).
    It is the same job a cook of the node sends, so either reuses the other's raw results. Quick jobs only: nothing
    reports its progress, no GPU."""
    ram = resolve_params(node_type, {**_defaults(node_type.Params), **params}).cost.ram_gb
    job = WorkerJob(node_type, node_type.worker_params(params), None, {k: Path(v) for k, v in inputs.items()},
                    record={"params": {}}, ram_gb=ram)
    return run_job(job, _Quiet(node_type.label))


def run_job(job: WorkerJob, env: RunnerEnv) -> Path:
    """Run a worker job in its extension's environment (the node type's project: nodes/services.py ProjectFacts);
    returns the raw folder. Raw results live in cache/<job key>_job/raw and are reused by any job that sends the worker
    the same (job.reuse)."""
    ext = job.node_type.project.extension
    if ext is None or not ext.paths.python.exists():
        project = job.node_type.project
        raise CookError(env.node_id, Msg("E-WORKER-NOTINSTALLED", project=project.title, extension=job.node_type.runtime))
    node, params = job.node_type.id, job.params
    state = ext.install_state().get("weights", {})
    for w in ext.weights:
        if w.option and params.get(w.option[0]) == w.option[1] and state.get(w.key) != "ok":
            weight = w.note or w.key
            raise CookError(env.node_id, Msg("E-WORKER-GATEDWEIGHTS", node=env.label, weight=weight, page=w.page, extension=ext.name)
                            if w.gated else Msg("E-WORKER-NOWEIGHTS", node=env.label, weight=weight, extension=ext.name))
    key = job_key(ext, node, job.image, params, job.inputs)
    # nodes that differ only in how they write results send the same job: run it once
    with env.exclusive(key, "另一个任务正在跑同样的计算，等它算完"):
        return _run_job(env, ext, node, key, job)


def _run_job(ctx: RunnerEnv, ext: Extension, node: str, key: str, spec: WorkerJob) -> Path:
    image, params, inputs, reuse, record = spec.image, spec.params, spec.inputs, spec.reuse, spec.record
    raw = raw_folder(key)
    job_dir = raw.parent
    note(job_dir.name)  # the job's task references the model's raw results, reused or made now (data/packet.py note)
    done_marker = raw / WORKER_COMPLETE
    if reuse and worker_done(raw):
        ctx.stage("复用已完成的计算")
        ctx.reused = True
        used(done_marker)
        job_file = job_dir / "job.json"
        try:
            job = json.loads(job_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            job = {}
        if job and job.get("record") != record:  # where the values came from this time
            write_text(job_file, json.dumps({**job, "record": record}, indent=2, ensure_ascii=False))
        return raw
    if raw.exists():  # an interrupted run: start clean
        shutil.rmtree(raw)

    frames: dict = {}
    size = {"width": 0, "height": 0}
    if image is not None:
        ctx.stage("准备画面")
        plate = frames_for_worker(image, lambda d, t: ctx.progress(d, t, "色彩转换"))
        frames = image_files(plate)
        size = {k: plate.meta[k] for k in size}
    raw.mkdir(parents=True)
    job = {
        "schema": SCHEMA,
        "id": key,
        "extension": ext.name,
        "node": node,
        "created": datetime.now().isoformat(timespec="seconds"),
        "frames": [{"frame": f, "path": os.path.relpath(p, job_dir)} for f, p in sorted(frames.items())],
        **size,
        # a time base, not the shot's frame rate: shots have no frame rate (see the DEFAULT_FPS comment in
        # data/units.py; the frame rate appears only on output-settings nodes). It is read only by upstream APIs that
        # require an fps: WHAM's `detector.track(img, fps, length)`, MediaPipe's video timestamps, ViPE's stream.
        "fps": DEFAULT_FPS,
        "params": params,
        "labels": {p["name"]: p["label"] for p in spec.node_type.param_specs()},  # a message naming a parameter
        "inputs": {k: os.path.relpath(v, job_dir) for k, v in inputs.items()},
        "record": record,
    }
    job_file = job_dir / "job.json"
    write_text(job_file, json.dumps(job, indent=2, ensure_ascii=False))

    log_path = job_dir / "worker.log"
    failed: dict = {}  # the worker's fail(code, **params), when it said one
    nothing: dict = {}  # the worker's nothing(code, **params): it found nothing to give (not an error)
    with log_path.open("w", encoding="utf-8") as log:

        def on_line(line: str) -> None:
            log.write(line)
            log.flush()
            event = event_of(line) or {}
            kind = event.get("type")
            if kind == "progress":
                ctx.progress(event["done"], event["total"], event.get("message", ""))
            elif kind == "stage":
                ctx.stage(event["name"])
            elif kind == "message":  # the worker's say(code, **params): its text from the catalogue, here
                ctx.say(event["code"], **worker_params(event.get("params", {})))
            elif kind == "failed":
                failed.update(event)
            elif kind == "nothing":
                nothing.update(event)

        identity = json.dumps(worker_identity(ext), sort_keys=True)  # the process kept for it must run this worker
        keep_free = keep_free_gb(spec.ram_gb)

        def run():
            # only this place knows the 「加载模型」 phase (the second of the four in lab2shot/progress.py):
            # starting the process, loading weights to the GPU and reading the material all happen before the first
            # progress line. When a progress line arrives, progress moves to 「计算」 by itself, without guessing from
            # stage names
            ctx.phase(PROGRESS_LOADING)
            return pool().run(ext, ctx.gpu, identity, job_file, on_line, ctx.stop, keep_free, ctx.stage)

        outcome = run()
        if outcome.oom and not outcome.clean:  # next to models kept loaded: once more, alone on the GPU
            ctx.stage("显存不够：卸载这张显卡上的常驻模型后重算")
            log.write("\n---- 显存不够：卸载这张显卡上的常驻模型，在新进程里重算 ----\n")
            pool().clear_gpu(ctx.gpu)
            shutil.rmtree(raw)
            raw.mkdir(parents=True)
            outcome = run()
    ctx.check_stop()
    if not outcome.ok:
        if failed:  # its `param` anchor (the parameter to change) stays beside the message, as ctx.say keeps it
            params = worker_params(failed.get("params", {}))
            param = params.pop("param", "")
            raise CookError(ctx.node_id, Msg(failed["code"], **params), log_path, param=param)
        # the error carries two things: the last lines of the worker log (the stack as is, without a paraphrase) and a
        # code, the first 8 characters of the job folder name (job_code), which the user passes to the administrator,
        # who looks up the full log with `lab2shot admin joblog <code>`.
        # The message must not refer to places the page does not have (the parameter panel has no node log).
        text = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = [ln.rstrip() for ln in text if ln.strip() and not ln.startswith("@@lab2shot")]
        detail = "\n".join(tail[-STACK_LINES:]) or (outcome.error or "")
        raise CookError(ctx.node_id, Msg("E-WORKER-FAILED", node=ctx.label,
                                         detail=detail, code=job_code(job_dir)), log_path)
    if nothing:  # a clean end with nothing to give: the node's outputs are empty (engine/cook.py), nothing to reuse
        raise NothingToCook(Msg(nothing["code"], **worker_params(nothing.get("params", {}))))
    flush_tree(raw)  # the worker's files are on the disk before the marker (io/atomic.py)
    mark(done_marker)
    # the worker has written result.json: what remains is the core converting raw into packets and writing them,
    # the 「取回结果」 phase
    ctx.phase(PROGRESS_FETCHING)
    return raw
