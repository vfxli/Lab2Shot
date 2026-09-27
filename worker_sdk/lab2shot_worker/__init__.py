"""Runs inside a paper's own environment, so it depends on nothing but numpy
(image libraries and torch are imported only by the functions that need them).

Protocol with the core: the worker gets the path of job.json, reads normalized
input frames listed there, writes raw results into the job's raw folder, and
reports progress as single stdout lines starting with PREFIX followed by JSON.
Any other output is treated as log text. What a worker says to the user it says by
message code and parameters only (say(), fail()): the core writes the words, from its
catalogue (lab2shot/messages, the extension's own adapters/<name>/messages.toml).

Everything a worker needs besides its model lives here, once: the job and its parameters (as the node sent them),
events, offline model loading (load_job refuses the network; local_hub, hf_dest), and in their own modules:

    serving.py        the entry point serve(main) and models that stay loaded between jobs (@resident, offload)
    run.py            one job's bookkeeping (Run): timing, GPU memory, progress and result.json's standard fields,
                      the same for every main()
    files.py          frame / mask / depth reading, sizes, EXR and atomic result files
    body_models.py    the body, hand and face models people download by hand (SMPL, SMPL-X, MANO, FLAME)
    build.py          helpers for extensions compiled from source at install time (EnvSpec.build scripts)
    recon.py          the reconstruction contract (chunked multi-view solves), pose interpolation
    feedforward.py    the feed-forward reconstruction driver (VGGT, Pi3, Depth Anything 3)
    mono_geometry.py  the per-frame geometry contract (MoGe, UniDepth, UniK3D, DA3, FaceAnything)
    world_humans.py   the world-humans contract (GVHMR, WHAM, TRAM, HaMeR, SMIRK)
    tracking.py       people-box tracking and temporal smoothing
    matte.py          the mask-guided matting contract (MatAnyone 2, VideoMaMa)
    point_tracks.py   the point-track contract: 2D (TAPNext++, CoTracker3, AllTracker's sampled points, WOFTSAM's
                      plane corners) and 3D (TAPIP3D, Track4World)
    optical_flow.py   the optical-flow contract (MEMFOF, WAFT)
    correspondence.py the dense-correspondence contract: where each plate pixel is in another picture (AllTracker,
                      RoMa v2)
    light_probe.py    the light-probe contract (DiffusionLight, LuxDiT)

What only one project and the ones built on it use stays in that project's adapter (Extension.worker_modules; an
extension that requires it imports it from there): SAM 3D Body's solve and rig (adapters/sam_3d_body/sam3dbody.py),
the DUSt3R model input of CUT3R and MonST3R (adapters/cut3r/dust3r_input.py).

What the core runs too (the protocol, poses, smoothing, motion, the EXR writer, scene arrays, body models, GPU
architectures, the light-probe conventions, the in-betweening files) is lab2shot_shared, one implementation, numpy only:
these modules import it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, TypeVar

from lab2shot_shared.protocol import (CODE, CPU_BUDGET_ENV, PREFIX, PROJECT_DIR, RAW, REPO_ENV, SCHEMA, WEIGHTS_ENV,  # noqa: E402,F401
                                      Failure, shown)

T = TypeVar("T")

class Params(dict):
    """The node's parameters: complete and validated by the node (lab2shot/nodes), which is their only definition,
    with the values the node computed (a lens from a connected camera, the chosen probe frame ...). A worker reads
    them as they are, params["name"]: no defaults, bounds or choices of its own. A name the node did not send is a
    node / worker mismatch and fails with a clear message."""

    def __missing__(self, key: str) -> NoReturn:
        fail("E-WORKER-NOPARAM", name=key)


@dataclass
class Job:
    data: dict[str, Any]
    dir: Path  # the job folder, absolute (load_job): everything the worker writes goes in it

    @property
    def frames(self) -> list[tuple[int, Path]]:
        """(frame number, normalized sRGB 8-bit PNG) in frame order."""
        return [(f["frame"], self.dir / f["path"]) for f in self.data["frames"]]

    @property
    def params(self) -> Params:
        return Params(self.data["params"])

    @property
    def width(self) -> int:
        return self.data["width"]

    @property
    def height(self) -> int:
        return self.data["height"]

    @property
    def fps(self) -> float:
        """Lab2Shot's fixed time base (one time code per frame), not the shot's frame rate: a shot has no frame rate,
        which appears only once, on the output settings node. Read only by upstream APIs that require an fps
        (WHAM's `detector.track(img, fps, length)`, MediaPipe's video timestamps, ViPE's stream)."""
        return self.data["fps"]

    @property
    def raw_dir(self) -> Path:
        """Where the raw results go (result.json and the family's files), inside the job folder."""
        return self.dir / RAW

    @property
    def repo_dir(self) -> Path:
        """The extension's original repository (read only: never written to)."""
        return _install_path(REPO_ENV)

    @property
    def weights_dir(self) -> Path:
        return _install_path(WEIGHTS_ENV)

    def scratch(self, name: str) -> Path:
        """An empty folder `name` inside the job folder, for what a worker keeps while it runs (COLMAP's database and
        images, the chunks waiting for loop closure): whatever an earlier run of the job left there is removed first."""
        import shutil

        if not name or Path(name).name != name or name in (".", "..", RAW):
            raise ValueError(f"a scratch folder is a plain name inside the job folder, not {name!r}")
        folder = self.dir / name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir()
        return folder

    @property
    def node(self) -> str:
        return self.data.get("node", "")

    def label(self, param: str) -> str:
        """How the node labels its parameter `param` (engine/external.py sends every label): for a message that names
        the parameter, in the node's own words."""
        return self.data["labels"][param]

    @property
    def inputs(self) -> dict[str, Path]:
        """Extra input files the node sent (name -> path, relative to the job folder); empty when there are none."""
        return {k: self.dir / v for k, v in (self.data.get("inputs") or {}).items() if v}

    def listing(self, name: str) -> dict[int, Path]:
        """A per-frame input (e.g. a mask sequence): the {"frames": {"<frame>": file}} JSON -> {frame: file}."""
        spec = self.inputs.get(name)
        if spec is None:
            return {}
        try:
            data = json.loads(spec.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail("E-WORKER-LISTING", name=name, path=shown(spec), detail=str(exc))
        return {int(f): (spec.parent / p) for f, p in (data.get("frames") or {}).items() if p}

    def frame_index(self) -> dict[int, int]:
        """frame number -> its index in the job's frames."""
        return {int(f): i for i, (f, _) in enumerate(self.frames)}

    def people_by_index(self) -> dict[int, dict[int, list[float]]]:
        """people() keyed by frame index in this job instead of frame number; people without a box in the job's
        frames are left out."""
        index = self.frame_index()
        people = {pid: {index[f]: box for f, box in boxes.items() if f in index} for pid, boxes in self.people().items()}
        return {pid: boxes for pid, boxes in people.items() if boxes}

    def people(self) -> dict[int, dict[int, list[float]]]:
        """The "boxes" input (people boxes): person id -> {frame: [x1, y1, x2, y2] in pixels}, in the file's
        order (most prominent first). People without a box are left out; empty without the input."""
        path = self.inputs.get("boxes")
        if path is None:
            return {}
        people = {}
        for person in json.loads(path.read_text(encoding="utf-8")).get("people", []):
            boxes = {int(f): [float(v) for v in box[:4]] for f, box in (person.get("boxes") or {}).items()
                     if box is not None and len(box) >= 4}
            if boxes:
                people[int(person["id"])] = boxes
        return people


def _install_path(name: str) -> Path:
    """An install location the core gives the worker process (lab2shot_shared.protocol REPO_ENV, WEIGHTS_ENV): an
    absolute path, or the job fails (never the current folder in its place)."""
    value = os.environ.get(name, "")
    if not value or not Path(value).is_absolute():
        fail("E-WORKER-NOINSTALLPATH", name=name, value=value)
    return Path(value)


def cpu_budget() -> int:
    """How many threads this computation may use (the cores left after the core's reserved cores).

    The process is already pinned to those cores by the core (`engine/resident.py hold_back`), and numeric libraries
    receive `OMP_NUM_THREADS`; projects with their own thread parameter must read this value and pass it on (COLMAP's
    `num_threads` defaults to -1, one thread per machine core, which after pinning means 32 threads on 24 cores). When
    the core provides no value (a worker run by hand), the machine's core count is used."""
    said = os.environ.get(CPU_BUDGET_ENV, "")
    if said.isdigit() and int(said) > 0:
        return int(said)
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def load_job(path: str | Path) -> Job:
    """The job file the core wrote for this run, by its absolute path (its folder is the job folder). Reading it makes
    this process a worker, and workers run offline (_forbid_downloads)."""
    path = Path(path)
    if not path.is_absolute():
        fail("E-WORKER-JOBPATH", path=str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported job schema {data.get('schema')!r}")
    job = Job(data, path.parent)
    _forbid_downloads(job.node or "worker")  # a job, even one whose node has no name: the next one starts clean
    return job


_CHANNEL: Any = None  # the stdout used to talk to the core, pinned when serve() starts (see _pin_channel)


def _pin_channel() -> None:
    """Pin the stdout used by the protocol, so that upstream code replacing `sys.stdout` does not affect communication
    between the worker and the core.

    `safe_state(quiet)` in the 3D Gaussian Splatting family (HairGS's `utils/general.py`, shared with GaussianHaircut)
    replaces `sys.stdout` with its own object: when quiet it writes nothing, so after the worker runs the upstream
    script through runpy, `stage`, `progress` and `job_end` are all swallowed and the core never learns that the job
    finished; when not quiet it appends a timestamp to every line, corrupting the protocol's JSON lines. The protocol
    therefore does not follow `sys.stdout`."""
    global _CHANNEL
    _CHANNEL = sys.stdout


def _emit(kind: str, **payload: Any) -> None:
    out = sys.stdout if _CHANNEL is None else _CHANNEL
    out.write(PREFIX + json.dumps({"type": kind, **payload}, ensure_ascii=False) + "\n")
    out.flush()


def stage(name: str) -> None:
    """Start a named stage, e.g. "检测人物"."""
    _emit("stage", name=name)


def progress(done: int, total: int, message: str = "") -> None:
    _emit("progress", done=done, total=total, message=message)


def say(code: str, /, **params: Any) -> None:
    """Say a message to the user: its code (W- a warning, N- a notice, I- the log only) and the parameters its
    template names, plain values (numbers, strings, lists of them). The core puts the words in (lab2shot/messages)."""
    if not CODE.match(code):
        raise ValueError(f"not a message code: {code!r}")
    _emit("message", code=code, params=params)



def write_result(job: Job, **info: Any) -> Path:
    """Finish: raw/result.json tells the core's converter what was produced.

    result.json.part is written first and then renamed, as in `files.save_npz`: the reader
    (`RawOutput.file()` in `lab2shot/nodes/families/base.py`) is a streaming node that checks every 0.05 s whether the
    file exists and treats it as complete once it does. A plain `write_text` creates the file before writing its
    content, leaving a moment in which the file exists but is empty; a reader hitting that moment gets
    `json.loads('')` -> JSONDecodeError, and the busier the machine, the wider the window. With .part + rename,
    "the file exists" means "the file is complete" for both per-frame and whole-shot files."""
    job.raw_dir.mkdir(parents=True, exist_ok=True)
    path = job.raw_dir / "result.json"
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    part.replace(path)  # rename is atomic: a reader either does not see the file or sees it complete
    _emit("done", result=str(path))
    return path


# --------------------------------------------------------------------------- failing


def fail(code: str, /, **params: Any) -> NoReturn:
    """Stop with an error for the user: its code (E-) and the parameters its template names (see say). The node shows
    it; the worker's log keeps the code."""
    if not CODE.match(code) or not code.startswith("E-"):
        raise ValueError(f"fail() takes an E- message code, not {code!r}")
    _emit("failed", code=code, params=params)
    sys.stdout.flush()
    raise SystemExit(code)


def reason(code: str, /, **params: Any) -> dict:
    """A message used as a parameter of another message ({why} in the template): the code and its parameters are
    passed, and the Chinese text is still written only in the catalogue.

    The core composes messages the same way (`Msg("E-OUTPUT-CANTHOLD", why=Msg(...))`); the worker can only pass
    JSON-serializable values, so it passes this dictionary, which the core turns back into a Msg
    (`lab2shot.messages.worker_params`).
    """
    if not CODE.match(code):
        raise ValueError(f"not a message code: {code!r}")
    return {"message": code, "params": params}


def nothing(code: str, /, **params: Any) -> NoReturn:
    """Stop without an error: the worker found nothing to give (no hands, no face, no person): its code (N-) and the
    parameters its template names. The node gives empty outputs and shows the notice; whatever comes after decides
    whether it can go on without them. Not a failure: the job ends cleanly."""
    if not CODE.match(code) or not code.startswith("N-"):
        raise ValueError(f"nothing() takes an N- message code, not {code!r}")
    _emit("nothing", code=code, params=params)
    sys.stdout.flush()
    raise SystemExit(0)


def require_weights(ext: str, *paths: Path, what: str = "权重", page: str = "") -> None:
    """The installer downloads and verifies every weight file or folder: one that is missing means
    `lab2shot ext install <ext>` has not run (or is pending). fail() naming it (`what`: a noun, 「找不到{what}」);
    `page`: where access to gated weights is requested first."""
    for path in paths:
        if not path.exists():
            if page:
                fail("E-WORKER-MISSINGGATED", what=what, path=shown(path), page=page, extension=ext)
            fail("E-WORKER-MISSINGWEIGHTS", what=what, path=shown(path), extension=ext)


def hf_dest(repo: str, filename: str) -> str:
    """Where a pinned Hugging Face file is kept below an extension's weights folder, for the installer
    (lab2shot.extensions.hf_file's default dest) and the worker alike: <repo name>/<file>."""
    return f"{repo.rsplit('/', 1)[-1]}/{filename}"


def require_cuda(project: str) -> None:
    import torch

    if not torch.cuda.is_available():
        fail("E-WORKER-NOCUDA", project=project)


_OFFLINE: dict[str, Any] = {"node": "", "refused": []}  # the node whose job runs (set: the process is offline), the hosts it was refused


def _forbid_downloads(node: str) -> None:
    """Workers run offline: every file comes from the installer (Hugging Face also sees HF_HUB_OFFLINE in the worker
    environment). A process the core started as a worker (REPO_ENV and WEIGHTS_ENV set, engine/resident.py) is made
    offline when this module is imported, before the worker script imports torch and the upstream code, so nothing
    an import does (a detector fetching its checkpoint at import time) gets out either; load_job calls this again for
    every job, to name the node. A connection to another machine is refused as if there were no network, whatever
    library makes it (torch.hub, gdown, a detector fetching its own checkpoint ...), so a model that still tries to
    download something fails with a message instead of fetching it. Only loopback stays open (localhost,
    127.0.0.0/8, ::1): this machine's LAN addresses count as another machine too (a worker has no business with any
    service on the network).
    torch.hub.load_state_dict_from_url still reads files already in TORCH_HOME."""
    import errno
    import ipaddress
    import socket

    installed = bool(_OFFLINE["node"])
    if _OFFLINE.get("job"):  # a resident process: the next job starts with no refusals of the one before
        _OFFLINE["refused"] = []  # (what was refused at import time, before the first job, stays for that job's message)
    _OFFLINE["job"] = _OFFLINE.get("job", False) or bool(node)
    _OFFLINE["node"] = node or _OFFLINE["node"] or "worker"
    if installed:  # already offline
        return

    def remote(sock: socket.socket, address) -> str | None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return None
        host = str(address[0])
        try:
            return None if host == "localhost" or ipaddress.ip_address(host.split("%")[0]).is_loopback else host
        except ValueError:  # a host name: resolved (and so leaving the machine) unless it is localhost
            return host

    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def refused_connect(sock, address):
        if host := remote(sock, address):
            _refused(host)
            raise ConnectionRefusedError(errno.ECONNREFUSED, f"{_OFFLINE['node']}: the worker runs offline, connection to {host} refused")
        return connect(sock, address)

    def refused_connect_ex(sock, address):
        if host := remote(sock, address):
            _refused(host)
            return errno.ECONNREFUSED
        return connect_ex(sock, address)

    socket.socket.connect, socket.socket.connect_ex = refused_connect, refused_connect_ex


def _refused(host: str) -> None:
    if host not in _OFFLINE["refused"]:
        _OFFLINE["refused"].append(host)


# Network access must be disabled before the imports at the top of worker.py (upstream code, torch.hub and the like);
# otherwise downloads triggered at import time cannot be blocked, and doing it only in load_job() is too late. Worker
# processes started by the core always carry these two variables (engine/resident.py worker_environment), while the
# core importing this package does not; they therefore identify a worker process, which goes offline before anything
# else is imported.
if os.environ.get(REPO_ENV) and os.environ.get(WEIGHTS_ENV):
    _forbid_downloads("")


def offline_failure() -> dict | None:
    """What a job that ended with an error of its own (not fail()) says when a connection was refused on the way
    (E-WORKER-OFFLINE): most likely a model fetching a file the installer did not provide. None: nothing was refused.
    A library that tries the network and carries on without it is no failure, so nothing is said while the job runs."""
    if not _OFFLINE["refused"]:
        return None
    return {"code": "E-WORKER-OFFLINE", "params": {"node": _OFFLINE["node"], "hosts": list(_OFFLINE["refused"])}}


def local_hub(repos: dict[str, str | Path], project: str) -> None:
    """torch.hub.load of these repositories (repository name, e.g. "dinov3" for facebookresearch/dinov3 -> the
    pinned local checkout the installer made) loads the local copy; any other repository fails instead of being
    fetched from GitHub at run time. Local folders pass through. Every call replaces the previous one's rule."""
    import torch.hub

    original = getattr(torch.hub.load, "upstream", torch.hub.load)

    def load(repo_or_dir, model, *args, **kwargs):
        name = str(repo_or_dir).split(":")[0].rstrip("/").split("/")[-1]
        if name in repos:
            for key in ("source", "force_reload", "trust_repo", "skip_validation"):
                kwargs.pop(key, None)
            return original(str(repos[name]), model, *args, source="local", **kwargs)
        if Path(str(repo_or_dir)).is_dir():
            return original(repo_or_dir, model, *args, **kwargs)
        fail("E-WORKER-HUBLOAD", project=project, repo=str(repo_or_dir))

    load.upstream = original
    torch.hub.load = load


def stub_module(name: str, **attrs: Any) -> None:
    """An empty stand-in for a module upstream imports but never uses on the path the worker runs (training logs,
    renderers, a detector the 人物框 input replaces): it need not be installed, and is not loaded. Attributes the
    importing code names (from x import y) are given as `attrs`. A module already imported is kept."""
    import types

    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules.setdefault(name, module)


_SINK = None


@contextmanager
def quiet():
    """Keep upstream's per-frame prints out of the log (the node shows the log's last line). The sink stays open:
    loggers created inside keep writing to it."""
    global _SINK
    import os

    if _SINK is None:
        _SINK = open(os.devnull, "w")
    with redirect_stdout(_SINK):
        yield


def limit_gpu_memory(margin_mb: int = 768) -> int:
    """Keep PyTorch inside the card's currently free memory plus what this process already holds (models kept
    loaded from an earlier job); returns the cap in MB.

    Windows / WSL drivers do not fail an allocation beyond the card's memory: they spill into system RAM and
    run 10x slower (and can exhaust the machine's RAM). With a cap, PyTorch raises an out-of-memory error
    instead, which a worker can catch and retry smaller."""
    import os

    import torch

    free, total = torch.cuda.mem_get_info()
    cap = max(free + torch.cuda.memory_reserved() - margin_mb * 2**20, total // 10)
    if os.environ.get(TEST_GPU_CAP_ENV):
        # Test-only (never set by the product): a smaller cap, so a real CUDA out-of-memory error can be caused on
        # purpose. Another program's memory cannot do it under WSL: mem_get_info does not count it there,
        # and the driver spills into system RAM instead of failing.
        cap = min(cap, int(os.environ[TEST_GPU_CAP_ENV]) * 2**20)
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total))
    return cap // 2**20


STEP_DOWN_ENV = "LAB2SHOT_OOM_STEP_DOWN"  # "1" only under the GPU test harness (tests/integration/conftest.py)
TEST_GPU_CAP_ENV = "LAB2SHOT_TEST_GPU_CAP_MB"  # test-only: limit_gpu_memory caps PyTorch at this many MB


def step_down_allowed() -> bool:
    """Whether a size that changes the result may be lowered automatically after running out of memory. Only in
    tests: in real use the artist decides (tests may retry one step lower automatically; in real use the step stops,
    states the reason and offers safe options for the user to choose from)."""
    import os

    return os.environ.get(STEP_DOWN_ENV) == "1"


# --------------------------------------------------------------------------- out of GPU memory: the one policy


@dataclass(frozen=True)
class MemoryBound:
    """What one GPU step's memory grows with, declared where the worker runs the step (fit_memory). Made by one of:

        MemoryBound.parameter(name, steps)  a node parameter the artist sets (每段帧数, 处理尺寸): a smaller value gives
                                            a different result, so only the GPU test harness lowers it by itself
        MemoryBound.batch(steps)            how many frames the worker sends at once, each computed on its own: a
                                            smaller batch only takes longer, lowered in real use too
        MemoryBound.shot(frames, people)    the input itself, all at once (nothing splits it): nothing to lower
        MemoryBound.one_frame(frame, people)  the people of one frame, together: nothing to lower

    `steps`: the values it may take, largest first. For a parameter, the values the node offers for it that are safe
    on a 24 GB card: the options an artist is given when real use stops (none left that the rest of the
    node's values allow: nothing smaller to offer)."""

    param: str = ""
    steps: tuple[int, ...] = ()
    frames: int = 0
    people: int = 0
    frame: int | None = None

    @classmethod
    def parameter(cls, name: str, steps: Sequence[int]) -> MemoryBound:
        return cls(param=name, steps=_largest_first(steps))

    @classmethod
    def batch(cls, steps: Sequence[int]) -> MemoryBound:
        if not steps:
            raise ValueError("a batch needs the sizes it may take")
        return cls(steps=_largest_first(steps))

    @classmethod
    def shot(cls, frames: int, people: int) -> MemoryBound:
        return cls(frames=int(frames), people=int(people))

    @classmethod
    def one_frame(cls, frame: int, people: int) -> MemoryBound:
        return cls(frame=int(frame), people=int(people))

    @property
    def changes_result(self) -> bool:
        """Whether a smaller value gives a different result. Only a batch does not."""
        return bool(self.param) or not self.steps

    @property
    def is_input(self) -> bool:
        """The input itself bounds it: no parameter, no batch."""
        return not self.param and not self.steps


def _largest_first(steps: Sequence[int]) -> tuple[int, ...]:
    out = tuple(sorted({int(s) for s in steps}, reverse=True))
    if out and out[-1] < 1:
        raise ValueError(f"memory steps must be positive values, got {list(steps)}")
    return out


def fit_memory(job: Job, bound: MemoryBound, run: Callable[[int | None], T], value: int | None = None) -> T:
    """Run one GPU step, run(value), under the one out-of-memory policy of every worker (tests may retry one step
    lower automatically; in real use the step stops, states the reason and offers safe options for the user to
    choose from). `value`: what the bound is
    now (the parameter as the node resolved it, the batch size); None for the input itself.

    `run` does the whole step from its start with the value it is given (never half a shot with one value and the rest
    with another) and returns what the worker needs from it, the value too when the worker records it. Any error but
    running out of memory (is_oom) goes through untouched. Out of memory (limit_gpu_memory makes one happen instead of
    a spill into RAM), the cached blocks are freed and then:

        a parameter   real use stops (E-WORKER-VRAMSIZE): the smaller safe values to choose on the node (the message
                      is said at the parameter; the card's total and free memory go to the log only, I-WORKER-VRAMCARD). The GPU test harness (step_down_allowed) says so
                      (W-WORKER-VRAMTESTSTEP) and runs the step again at the next smaller value.
        a batch       runs again at the next smaller size, in either mode (W-WORKER-VRAMBATCH).
        no smaller    stops in either mode: E-WORKER-VRAMFLOOR (a parameter), E-WORKER-VRAMONEBATCH (a batch).
        the input     stops at once: E-WORKER-VRAMSHOT, E-WORKER-VRAMFRAME (what to take out of the input).

    Every stop is an out-of-memory failure (is_oom): next to models kept loaded, the core runs the job once more alone
    on the card (engine/external.py). The only place a worker handles running out of memory (tests/test_boundaries.py)."""
    while True:
        try:
            return run(value)
        except RuntimeError as exc:  # torch's OutOfMemoryError and the WSL driver's error are RuntimeErrors
            if not is_oom(exc):
                raise
        free_gb, total_gb = _card_after_oom()
        # the card's memory is card information: the log only (I-), never in what the user reads
        say("I-WORKER-VRAMCARD", total_gb=total_gb, free_gb=free_gb)
        if bound.is_input:
            if bound.frame is not None:
                fail("E-WORKER-VRAMFRAME", frame=bound.frame, people=bound.people)
            fail("E-WORKER-VRAMSHOT", frames=bound.frames, people=bound.people)
        smaller = [s for s in bound.steps if s < value]
        if bound.param:
            label = job.label(bound.param)
            if not smaller:
                fail("E-WORKER-VRAMFLOOR", param=bound.param, label=label, size=value)
            if not step_down_allowed():
                fail("E-WORKER-VRAMSIZE", param=bound.param, label=label, size=value, options=smaller)
            say("W-WORKER-VRAMTESTSTEP", param=bound.param, label=label, size=value, smaller=smaller[0])
        else:
            if not smaller:
                fail("E-WORKER-VRAMONEBATCH", size=value)
            say("W-WORKER-VRAMBATCH", size=value, smaller=smaller[0])
        value = smaller[0]


def _card_after_oom() -> tuple[float, float]:
    """Free what the failed step left cached; (free, total) GB of the card as the message says them."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return round(free / 2**30, 2), round(total / 2**30, 2)


# --------------------------------------------------------------------------- GPU errors: full, or the wrong card

# torch.OutOfMemoryError exists only from 2.0, torch.cuda.OutOfMemoryError from 2.2; an environment pinned to an
# older torch raises AttributeError just from writing `except torch.OutOfMemoryError:`, and that AttributeError is
# what the user sees instead of the real error. is_oom() below never touches a version-specific attribute: every
# torch version raises a RuntimeError (OutOfMemoryError included: it subclasses RuntimeError) whose type name or text
# says what happened.
_INCOMPATIBLE_GPU_MARKERS = ("no kernel image is available", "no kernel image available")
_OOM_MARKERS = ("out of memory", "cuda driver error: device not ready")
# fit_memory's stops: every out-of-memory failure a worker says is one of the SDK's E-WORKER-VRAM... codes
_OOM_CODE = re.compile(r"^E-WORKER-VRAM[A-Z0-9]*$")


def is_incompatible_gpu(exc: BaseException) -> bool:
    """This card's architecture (compute capability) is not one this environment's torch (or a compiled CUDA
    extension in it) was built for: "CUDA error: no kernel image is available for execution on the device". Not
    fixable by a smaller batch or a shorter chunk (is_oom is always False for it): the environment needs a torch
    build that covers this card, or the job needs a different GPU."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if any(m in str(exc).lower() for m in _INCOMPATIBLE_GPU_MARKERS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def is_oom(exc: BaseException) -> bool:
    """Whether it failed because the GPU ran out of memory, on any torch version, and also when a lower-level driver
    error under WSL/Windows means the same thing ("CUDA driver error: device not ready"). Works across the
    exception chain (a wrapped RuntimeError included). Never true for is_incompatible_gpu: shrinking a batch cannot
    fix an environment that cannot run on this card's architecture at all."""
    if is_incompatible_gpu(exc):
        return False
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == "OutOfMemoryError" or any(m in str(exc).lower() for m in _OOM_MARKERS):
            return True
        if isinstance(getattr(exc, "code", None), str) and _OOM_CODE.match(exc.code):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def incompatible_gpu_failure(exc: BaseException) -> dict:
    """The message an artist can act on for is_incompatible_gpu(exc) (E-WORKER-GPUARCH): which torch, this card's
    architecture (when it can still be read: the driver reports it even though it refused to run a kernel)."""
    import torch

    cap = "?"
    try:
        major, minor = torch.cuda.get_device_capability()
        cap = f"{major}.{minor}"
    except Exception:
        pass
    return {"code": "E-WORKER-GPUARCH", "params": {"torch": torch.__version__, "capability": cap}}


def set_seed(seed: int, device: Any = None):
    """Every random generator a model may draw from, seeded with `seed`: Python's random, numpy's global generator and
    torch's (CPU and every CUDA device); returns a torch.Generator on `device` (None: the CPU) seeded with it, for a
    sampler that takes one. The one way a worker fixes its randomness: a shot gives the same
    result every time, and on either card. Call it again for a fresh generator at the same seed."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # the CPU and, lazily, every CUDA device
    return torch.Generator(device=device).manual_seed(seed)


def check_node(job: Job, *nodes: str) -> str:
    """The job's node, which must be one this worker runs."""
    node = job.node or nodes[0]
    if node not in nodes:
        fail("E-WORKER-NODE", node=node, nodes=list(nodes))
    return node


from .files import fit_size as fit_size, link_file as link_file, read_frame as read_frame  # noqa: E402  (re-exported)
from .files import read_depth as read_depth, read_mask as read_mask, save_npz as save_npz  # noqa: E402
from .files import read_channels as read_channels  # noqa: E402  (re-exported)
from .serving import offload as offload, resident as resident, serve as serve  # noqa: E402
