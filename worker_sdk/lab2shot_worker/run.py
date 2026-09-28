"""Shared bookkeeping for one job: the few dozen lines every worker's main() needs, written only here.

This module contains no algorithm. It covers only: the start time, how long loading the model took and how much VRAM
it uses, per-frame time, progress, and the set of standard fields in result.json (seconds / load_seconds /
seconds_per_frame / count / frames / gpu_model_mb / gpu_peak_mb / gpu_peak_allocated_mb / gpu_cap_mb). A third-party developer
adding a per-frame model is left with loading the model, inferring each frame and writing each frame's output in
main(); no bookkeeping code is needed and field names stay consistent.

Usage (adapters/birefnet/worker.py and adapters/sapiens2/worker.py are working examples)::

    run = Run(job, "BiRefNet")            # CUDA check, raw folder, peak VRAM reset
    with run.loading("BiRefNet 模型"):    # stage name + timing + the model's VRAM
        model = load_model(...)
    run.stage("抠像")
    for i, (frame, path) in run.each(job.frames, "抠像"):   # progress reported automatically
        with run.frame():                 # this frame's time (measured after synchronize)
            ...inference, file writing...
    run.finish([f for f, _ in job.frames], kind="matte", ...)  # standard fields + the worker's own -> result.json

The steps every main() starts with are here as well::

    run = Run.start(job_path, "moge.geometry", "MoGe")          # load_job + check_node + Run(...)
    run.weights(checkpoint, what="MoGe 权重")                    # when missing, states where to get them (require_weights)
    frames = run.frames(step=p["step"])                          # non-empty, every step-th frame with the last always included, first frame's size
    model = run.model("MoGe 模型", load_model, checkpoint, device)  # loading timing + VRAM cap + resident loading
    ...the algorithm itself...
    run.finish(frames.numbers, ...)
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, TypeVar

from pathlib import Path
from dataclasses import dataclass

from . import Job, MemoryBound, check_node, fail, fit_memory, limit_gpu_memory, load_job, progress, read_frame, require_cuda, require_weights, say, stage, write_result

T = TypeVar("T")


def node_prefix(node: str | tuple[str, ...]) -> str:
    """The extension id a node id starts with ("cotracker.track" -> "cotracker"): what `lab2shot ext install` takes."""
    first = node if isinstance(node, str) else node[0]
    return first.split(".")[0]


@dataclass(frozen=True)
class Frames:
    """The frames a job works on (Run.frames): their numbers and files in order, and the picture's size."""

    numbers: list[int]
    paths: list[Path]
    width: int
    height: int

    def __len__(self) -> int:
        return len(self.numbers)

    def __iter__(self):
        return iter(zip(self.numbers, self.paths, strict=True))

    @property
    def pairs(self) -> list[tuple[int, Path]]:
        """(frame number, path), as Job.frames gives them."""
        return list(zip(self.numbers, self.paths, strict=True))


def _torch():
    """torch, only when the worker itself has already imported it (this module does not depend on torch, so CPU
    workers can use it too)."""
    return sys.modules.get("torch")


def _cuda_mb(what: str) -> int:
    t = _torch()
    if t is None or not t.cuda.is_available() or not t.cuda.is_initialized():
        return 0
    return round(getattr(t.cuda, what)() / 2**20)


class Run:
    """One job's bookkeeping, the same for every worker (see the module docstring). `gpu=False`: a CPU worker; CUDA is
    not checked and VRAM is not recorded."""

    def __init__(self, job: Job, project: str, *, gpu: bool = True, extension: str = ""):
        self.job, self.project, self.gpu = job, project, gpu
        self.extension = extension or project.lower()  # the extension id in messages (lab2shot ext install <id>): CoTracker3 -> "cotracker"
        if gpu:
            require_cuda(project)
            _torch().cuda.reset_peak_memory_stats()
        job.raw_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.load_seconds = 0.0
        self.model_bytes = 0
        self.frame_seconds: list[float] = []
        self.gpu_cap_mb: int | None = None  # what limit_gpu_memory set (run.model), MB
        self.node: str = job.node or ""  # the node this job is for (Run.start checks it)

    @classmethod
    def start(cls, job: str | Job, node: str | tuple[str, ...], project: str, *, gpu: bool = True, extension: str = "") -> "Run":
        """The first three lines of every main(): read the job file (or take the Job already read), check it is for this
        worker's node(s), start the bookkeeping. `project`: the extension's name in messages (require_cuda);
        `extension`: its id when that is not simply the lower-cased name (CoTracker3 -> cotracker)."""
        job = job if isinstance(job, Job) else load_job(job)
        found = check_node(job, *((node,) if isinstance(node, str) else node))
        run = cls(job, project, gpu=gpu, extension=extension or node_prefix(node))
        run.node = found  # the node this job is for (a worker serving several nodes branches on it)
        return run

    @property
    def params(self):
        return self.job.params

    def weights(self, *paths: Path, what: str = "权重", page: str = "") -> None:
        """The weight files this job needs are on disk, or the job fails saying where they come from (require_weights)."""
        require_weights(self.extension, *paths, what=what, page=page)

    def frames(self, step: int = 1, least: int = 1, few: str = "E-WORKER-NOFRAMES", **few_params: Any) -> Frames:
        """The job's frames: none is a failure (E-WORKER-NOFRAMES), fewer than `least` says `few` with `few_params`
        (a tracker's E-WORKER-TOOFEWFRAMES); every `step`-th one, the last frame always included (the result covers the
        whole shot); the picture's size read from the first frame, and reported when the job file's size differs
        (W-FEEDFWD-JOBSIZE: node and worker disagree)."""
        all_frames = self.job.frames
        if not all_frames:
            fail("E-WORKER-NOFRAMES")
        used = all_frames[::max(1, int(step))]
        if used[-1][0] != all_frames[-1][0]:
            used.append(all_frames[-1])
        if len(used) < least:
            fail(few, **{"least": least, "have": len(used), **few_params})
        height, width = read_frame(used[0][1]).shape[:2]
        if self.job.width and self.job.height and (self.job.width, self.job.height) != (width, height):
            say("W-FEEDFWD-JOBSIZE", job_width=self.job.width, job_height=self.job.height, width=width, height=height)
        return Frames([f for f, _ in used], [p for _, p in used], width, height)

    def model(self, what: str, loader, *args, **kwargs):
        """Load a model under the bookkeeping (`loading`): the GPU capped first (limit_gpu_memory: an allocation beyond
        the card fails instead of spilling into RAM), then `loader(*args, **kwargs)`; a @resident loader keeps it
        between jobs (serving.py)."""
        with self.loading(what):
            if self.gpu and self.gpu_cap_mb is None:
                self.gpu_cap_mb = limit_gpu_memory()
            return loader(*args, **kwargs)

    def fit(self, bound: MemoryBound, run, value=None):
        """One GPU step under the one out-of-memory policy (fit_memory): stop and say the smaller safe values in real use,
        step down in a test run (step_down_allowed)."""
        return fit_memory(self.job, bound, run, value)

    @contextmanager
    def loading(self, what: str) -> Iterator[None]:
        """The model loading section: stage name 「加载 …」, how long it took and how much VRAM is used after loading."""
        stage(f"加载 {what}")
        t = time.time()
        yield
        self.load_seconds += time.time() - t  # a worker with several models (detector, pose, body, ...) loads several times: their sum
        if self.gpu:
            self.model_bytes = _torch().cuda.memory_allocated()

    def stage(self, name: str) -> None:
        """The next stage (the stage name on the progress bar)."""
        stage(name)

    @contextmanager
    def frame(self) -> Iterator[None]:
        """This frame's time, measured after the GPU has actually finished (synchronize)."""
        t = self.frame_started()
        yield
        self.frame_done(t)

    def frame_started(self) -> float:
        """When the measured section is not the whole loop body (file writing excluded), time it manually:
        `t = run.frame_started()` ... `run.frame_done(t)`."""
        return time.time()

    def frame_done(self, started: float) -> None:
        if self.gpu:
            _torch().cuda.synchronize()
        self.frame_seconds.append(time.time() - started)

    @contextmanager
    def timed(self, name: str) -> Iterator[list[float]]:
        """A stage with its own time: `with run.timed("跟踪") as t:` ... afterwards `t[0]` is the seconds it took (a worker
        records read_seconds / track_seconds beside the standard fields)."""
        self.stage(name)
        started = time.time()
        box = [0.0]
        yield box
        box[0] = time.time() - started

    def spread(self, seconds: float, count: int) -> None:
        """One pass over `count` frames took `seconds` (a tracker's single forward pass): shared out per frame, so
        seconds_per_frame means the same as for a worker that times each frame."""
        if count > 0:
            self.frame_seconds.extend([seconds / count] * count)

    def each(self, items: Iterable[T], message: str) -> Iterator[tuple[int, T]]:
        """Process items one by one, reporting progress after each (works directly with Job.frames' (frame number, path) list)."""
        items = list(items)
        for i, item in enumerate(items):
            yield i, item
            progress(i + 1, len(items), message)

    def finish(self, frame_numbers: list[int], /, **info: Any):
        """result.json: standard fields first, the worker's own fields on top (the worker's value wins on a name clash:
        Sapiens2's `frames` is the full frame number list rather than the standard [first, last], and is used as given)."""
        frames = frame_numbers
        elapsed = time.time() - self.t0
        count = len(frames)
        # a worker that timed its frames (run.frame / run.spread): their mean; one that did not (a whole-shot solve,
        # or several stages such as detect, track, solve): everything but loading, shared out per frame
        per_frame = (float(sum(self.frame_seconds) / len(self.frame_seconds)) if self.frame_seconds
                     else (elapsed - self.load_seconds) / count if count else None)
        # a result that is not about frames (a rig: no frames, no per-frame time) gets no `frames` and no
        # `seconds_per_frame` at all, not an empty list and a null: the converters hand the result's fields on as USD
        # customData, which takes neither (USD refuses them: "Invalid value type for customData")
        standard = {
            **({"frames": [frames[0], frames[-1]]} if frames else {}),
            "count": count,
            "seconds": round(elapsed, 2),
            "load_seconds": round(self.load_seconds, 2),
            **({"seconds_per_frame": round(per_frame, 4)} if per_frame is not None else {}),
        }
        if self.gpu:
            standard.update(gpu_model_mb=round(self.model_bytes / 2**20),
                            gpu_peak_mb=_cuda_mb("max_memory_reserved"),
                            gpu_peak_allocated_mb=_cuda_mb("max_memory_allocated"))
            if self.gpu_cap_mb is not None:
                standard["gpu_cap_mb"] = self.gpu_cap_mb
        return write_result(self.job, **{**standard, **info})
