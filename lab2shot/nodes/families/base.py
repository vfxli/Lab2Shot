"""The single template for nodes that run a worker, analogous to submitting a render: prepare() fills in the job
(Job), the worker computes, and convert() turns what it wrote (RawOutput) into packets. Families inherit from it;
extension nodes that belong to no family inherit from it directly.

    cook = prepare(ctx) -> Job -> ctx.run_worker(...) -> convert(ctx, RawOutput, Job)

cook is never overridden (enforced at class creation): nodes differ only in what they send (prepare) and how they read
the result back (convert). A subclass that extends its family's job calls super().prepare(ctx) and applies .with_() for
the rest; one that adds outputs calls super().convert(ctx, raw, job) and extends the returned dict."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ...data.packet import Packet
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef
from ..lens import NO_LENS, Lens
from ..kit.confidence import Confidence


class MissingFrames(Enum):
    """How a node handles a frame for which its worker wrote no raw result (declared on the node class)."""

    FAIL = "fail"  # stop and report the node and frame (the worker is expected to write every frame)
    SKIP = "skip"  # omit the frame from per-frame outputs (e.g. every step-th frame solved) and report the count once


class RawOutput:
    """The folder into which a worker wrote its raw results; node code reads raw results only through this class. A
    missing file raises an error naming the node and the file rather than a Python traceback; missing frames follow
    the node's MissingFrames policy."""

    def __init__(self, folder: Path, missing: MissingFrames, *, done: threading.Event | None = None,
                 error: Any = None):
        self.folder = Path(folder)
        self.policy = missing
        self.done = done  # set when the worker ends (streaming nodes); None: the worker has already ended
        self.error = error  # streaming nodes: () -> the worker's exception after it ends (None while running or on success)
        self.missing: list[int] = []  # frames skipped so far (SKIP)
        self._said: set[int] = set()
        self._result: dict | None = None

    def path(self, name: str) -> Path:
        """The path of a worker file, whether or not it was written (check optional files with `.exists()`)."""
        return self.folder / name

    def file(self, name: str) -> Path:
        """A file the worker is required to write (result.json, envmap.exr, a person's npz); if it is missing, the
        cook stops with an error naming it.

        A streaming node (WorkerNode.streams) reads while its worker is still running, so the requirement applies once
        the worker has ended, not at the moment of the call; `frames()` applies the same rule per frame. Deciding
        before the worker ends would race with it: result.json is written last but read first by convert()
        (families/depth_camera.py PerFrameDepthCamera), so the first cook would fail spuriously."""
        path = self.path(name)
        while self.done is not None and not path.exists() and not self.done.is_set():  # still being written
            self.done.wait(0.05)
        if not path.exists():
            raise Invalid(Msg("E-FAMILY-NORAW", file=name, folder=str(self.folder)))
        return path

    def result(self) -> dict:
        """raw/result.json, read once and cached."""
        if self._result is None:
            self._result = json.loads(self.file("result.json").read_text(encoding="utf-8"))
        return self._result

    def arrays(self, name: str):
        """A whole-shot npz the worker is required to write (cameras.npz, tracks.npz, a person's file, etc.)."""
        return np.load(self.file(name))

    def maps(self, array: str, pattern: str = "frame_{}.npz"):
        """Return a reader (frame) -> the named array of that frame's npz, or None when the file or array is absent.
        It provides random access to raw files already traversed by `frames()`; 点云 uses it to obtain the model's
        own point map after the family has written the depth. Progress reporting and the missing-frame policy remain
        with `frames()`; this reader only returns what exists."""
        def read(frame: int):
            path = self.folder / pattern.format(frame)
            if not path.exists():
                return None
            with np.load(path) as d:
                return d[array] if array in d.files else None

        return read

    def frames(self, ctx, frames, pattern: str = "frame_{}.npz") -> Iterator[tuple[int, Any]]:
        """Yield (frame, npz) for each of `frames`, reporting progress. For a streaming node (WorkerNode.streams) the
        worker is still writing while the node reads, so a frame is yielded as soon as its file appears (save_npz
        writes <name>.part and then renames, so an existing file is complete); a frame whose file has not appeared
        waits until the worker ends (`done`). Without `done`, all files already exist. Only after that does a missing
        file stop the cook (FAIL) or get omitted and reported once at the end of the loop (SKIP)."""
        for f in ctx.each(list(frames)):
            path = self.folder / pattern.format(f)
            while self.done is not None and not path.exists() and not self.done.is_set():  # still being written
                ctx.check_stop()
                self.done.wait(0.05)
            if path.exists():
                yield f, np.load(path)
                continue
            if self.error is not None:  # the worker ended without this frame: its own error takes precedence
                failed = self.error()
                if failed is not None:
                    raise failed
            if self.policy is MissingFrames.FAIL:
                raise Invalid(Msg("E-FAMILY-NOFRAME", node=ctx.label, frame=f))
            if f not in self.missing:
                self.missing.append(f)
        self.say_missing(ctx)

    def say_missing(self, ctx) -> None:
        """Warn about skipped frames not yet reported, so repeated reads of one folder report each frame once."""
        new = [f for f in self.missing if f not in self._said]
        if not new:
            return
        self._said.update(new)
        from ...io.sequence import format_frame_range

        ctx.say("N-FAMILY-SKIPPED", count=len(new), frames=format_frame_range(sorted(new)))


@dataclass(frozen=True)
class Job:
    """The job filled in by prepare() and read back by convert().

    `notes` holds values a family computes in prepare() for use by its own convert() only (each family's docstring
    lists its keys); an extension node reads or adds only the keys its family documents."""

    plate: Packet | None  # frames and size of the results (usually ctx.input("image")); None: no plate
    send: Packet | None = None  # frames sent to the worker when they differ from the plate (a light probe's single frame)
    extra: Mapping[str, Any] = field(default_factory=dict)  # values computed by the node for the worker (fov_x_deg, etc.)
    inputs: Mapping[str, Path] = field(default_factory=dict)  # additional files the worker reads (boxes, mask, motion.npz)
    lens: Lens = NO_LENS  # the lens used, recorded in the job file
    camera: Packet | None = None  # the connected camera used to place results
    notes: Mapping[str, Any] = field(default_factory=dict)

    def with_(self, **changes) -> Job:
        """Return a copy with the given fields changed; extra, inputs and notes are merged rather than replaced."""
        for name in ("extra", "inputs", "notes"):
            if name in changes:
                changes[name] = {**getattr(self, name), **changes[name]}
        return replace(self, **changes)


class WorkerNode(NodeDef):
    """A node that runs its worker: prepare -> Job -> worker -> convert (see the module docstring).

    `confidence`: how the model provides per-pixel confidence (Confidence), or None; a node that declares it gets the
    output 置信度 (listed in type order by nodes/applies.py all_outputs), written by kit/confidence.py ConfidenceWriter.
    `missing_frames`: how a frame without a raw result is handled (MissingFrames)."""

    confidence: ClassVar[Confidence | None] = None
    missing_frames: ClassVar[MissingFrames] = MissingFrames.FAIL
    # True: per-frame results are final as soon as a frame's raw file appears (families writing EXR per frame), so the
    # node may be displayed frame by frame while the worker is still cooking; convert() must read through
    # RawOutput.frames, which waits for each frame and decides on missing ones only after the worker ends.
    # False: the worker's final step may change earlier frames (reconstruction, whole-shot smoothing, tracking,
    # optical flow), so nothing is displayed until the whole shot is done.
    streams: ClassVar[bool] = False
    # Name of the array in the worker's per-frame npz holding the camera-space point map predicted by the model
    # (OpenCV axes, metres, on the same grid as the depth); "" means the model outputs depth only and the point cloud
    # is computed by unprojection (kit/maps.py family_points).
    # A node that declares it outputs the model's points directly on 「点云」 instead of reconstructing them from
    # depth and Focal Length. It may be declared only when the model itself predicts a point map; points the worker
    # unprojects from depth (DepthAnything3, FaceAnything's unproject_frame) are not native.
    native_points: ClassVar[str] = ""

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if "cook" in vars(cls):
            raise TypeError(f"{cls.__name__}: a WorkerNode does not override cook(); put what it sends in prepare() "
                            "and how it reads the result in convert()")
        if cls.confidence is not None and not isinstance(cls.confidence, Confidence):
            raise TypeError(f"{cls.__name__}.confidence is a Confidence(scale, help=...), not {cls.confidence!r}")
        if cls.confidence is not None and any(p.name == "confidence" for p in cls.outputs):
            raise TypeError(f"{cls.__name__}: the confidence output comes from `confidence`; do not declare it in outputs too")

    @classmethod
    def prepare(cls, ctx) -> Job:
        """Return the job; by default only the plate on 图像."""
        return Job(ctx.input("image"))

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """NodeDef.describe() plus the worker-node setting `streams` (whether per-frame results may be displayed
        while the worker is still cooking; read by the server's partial-result interface)."""
        out = super().describe()
        out["streams"] = cls.streams
        return out

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        raise NotImplementedError(f"{cls.id}: convert() does not say how to read the worker's result")

    @classmethod
    def cook(cls, ctx) -> dict[str, Packet]:
        job = cls.prepare(ctx)
        image = job.send if job.send is not None else job.plate
        if cls.streams and getattr(ctx, "stream_worker", None) is not None:
            # the farm runs the worker on a separate thread, so convert() consumes frames as they appear
            folder, done, error, halt = ctx.run_worker_streaming(image, extra=dict(job.extra), inputs=dict(job.inputs),
                                                                 record=job.lens.record())
            try:
                produced = cls.convert(ctx, RawOutput(folder, cls.missing_frames, done=done, error=error), job)
            except BaseException:
                # the node is through only once its worker is: it gives its card back when it fails, and a worker still
                # running there would have the card handed to another node as well
                halt()
                done.wait()
                raise
            done.wait()  # the worker must end before the packet commits; its error, if any, takes precedence
            if (failed := error()) is not None:
                raise failed
            return produced
        folder = ctx.run_worker(image, extra=dict(job.extra), inputs=dict(job.inputs), record=job.lens.record())
        return cls.convert(ctx, RawOutput(folder, cls.missing_frames), job)
