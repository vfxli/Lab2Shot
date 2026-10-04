"""Models that stay loaded between jobs: the worker's side of the core's resident processes (lab2shot/engine/resident.py).

A worker script ends with `serve(main)`. `worker.py <job.json>` runs that one job and exits (by hand, for debugging);
the core runs `worker.py --serve` and keeps the process: it sends jobs one after another, each running `main` as a
fresh process would. Between jobs the process keeps what its model loaders returned:

    @resident
    def load_model(checkpoint: Path, device: str):
        ...
        return model

The first call loads; later calls with the same arguments (in this job or a later one) return the same object. A loader
depends on its arguments only (plain values: str, numbers, Path, torch.device ...), is a module-level function, and
returns a model in eval mode that jobs do not change (whatever a job sets on it, every job sets). A job keeps nothing
else between jobs: no model in a global, no per-job state left on the model.

Memory, all generic (torch tensors are found by walking the loaded object and moved in place, so every reference to
them stays valid):
  - a process keeps what its last job used; models the job did not use are freed when it ends;
  - loading or bringing back a model first moves the models this job has not used off the GPU: to RAM when the
    machine keeps enough free (the core's keep_free_gb), else they are freed;
  - offload(model) inside a job: a model the job is done with leaves the GPU (DiffusionRenderer's two 7B models);
  - the core's "offload" command moves everything to RAM (the admin page, or another project needs the GPU); a
    model in RAM goes back to the GPU on its next use. `@resident(movable=False)` marks a model that holds GPU
    memory PyTorch does not manage (an onnxruntime CUDA session): leaving the GPU frees it.

Identical results: every job starts with its random generators seeded from the OS, as a new process starts them
(torch, numpy and Python all seed at random in a new process, so a worker whose result must repeat seeds them itself),
and loading never draws from the job's random numbers (the random state is put back after a loader runs): a job that
seeds gets the same numbers whether its models were loaded just now or kept. A failed job ends the process (the core starts a new one for the next job).

Protocol (--serve): commands are JSON lines on the process's original stdin (the worker's own stdin is /dev/null):
{"cmd": "job", "job": path, "keep_free_gb": n}, {"cmd": "offload", "keep_free_gb": n}, {"cmd": "exit"}. Replies are
events: job_end {ok, error, oom, incompatible_gpu, models, vram_mb, context_mb} and offloaded {models, vram_mb,
context_mb}. context_mb: what the process holds on its GPU beyond PyTorch's reserved memory (its CUDA context, and any
library's own allocations), which only ending the process frees; measured once, at the end of the first job that
brought CUDA up (the card's memory.used then, less before that job and less vram_mb), None until then or when it
cannot be measured (no nvidia-smi, the process sees no single GPU, another program's use moved too much meanwhile).
"""

from __future__ import annotations

import functools
import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from lab2shot_shared.memory import available_gb

DEPTH = 24  # how deep the search for a model's tensors goes into nested objects
PARENT_POLL_S = 2.0


# --------------------------------------------------------------------------- loaded models


@dataclass(eq=False)
class _Entry:
    label: str
    value: Any
    movable: bool
    job: int = 0  # the last job that used it
    moved: list | None = None  # while in RAM: [(tensor, device it came from)]


_ENTRIES: dict[tuple, _Entry] = {}
_JOB = 1  # the job running (one-shot runs have just this one)
_KEEP_FREE_GB: float | None = None  # memory to keep free when models go to RAM; None: not serving, models are freed


def resident(loader: Callable | None = None, *, movable: bool = True):
    """Keep what `loader` returns loaded between jobs (see the module doc). `movable=False`: the model holds GPU
    memory PyTorch does not manage, so leaving the GPU frees it instead of moving it to RAM."""

    def wrap(fn: Callable) -> Callable:
        if "<locals>" in fn.__qualname__:
            raise TypeError(f"@resident {fn.__qualname__}: a loader is a module-level function (its result may depend "
                            "on its arguments only)")
        name = f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def load(*args, **kwargs):
            key = (name, _frozen(args), _frozen(kwargs))
            entry = _ENTRIES.get(key)
            if entry is None:
                _make_room()
                before = _rng_state()
                value = fn(*args, **kwargs)
                _set_rng_state(before)  # loading draws none of the job's random numbers
                entry = _ENTRIES[key] = _Entry(_label(fn, args, kwargs), value, movable)
            elif entry.moved is not None:
                _make_room()
                _restore(entry)
            entry.job = _JOB
            return entry.value

        return load

    return wrap(loader) if loader is not None else wrap


def offload(value: Any) -> None:
    """A model (returned by a @resident loader) this job no longer needs leaves the GPU: to RAM for the next job, or
    freed when the machine is short of memory (then the caller drops its own references too)."""
    entry = next((e for e in _ENTRIES.values() if e.value is value), None)
    if entry is None:
        raise ValueError("offload(): not a model a @resident loader returned")
    _leave_gpu(entry)
    _free_cached()


def _frozen(value: Any):
    """A loader argument as part of the cache key."""
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return (type(value).__name__, value)
    if isinstance(value, Path):
        return ("Path", str(value))
    if isinstance(value, Enum):
        return (type(value).__qualname__, value.name)
    if isinstance(value, (tuple, list)):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _frozen(v)) for k, v in value.items()))
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, (torch.device, torch.dtype)):
        return (type(value).__name__, str(value))
    raise TypeError(f"@resident loader arguments are plain values (str, numbers, Path, torch.device ...), not {type(value).__name__}")


def _label(fn: Callable, args: tuple, kwargs: dict) -> str:
    """What the admin page shows: the loader's Path arguments (by their last two parts) and short string arguments
    other than a device; the loader's name when there are none."""
    parts = []
    for v in (*args, *kwargs.values()):
        if isinstance(v, Path):
            parts.append("/".join(v.parts[-2:]))
        elif isinstance(v, str) and len(v) <= 80 and v.split(":")[0] not in ("cuda", "cpu"):
            parts.append(v)
    return " · ".join(parts) or fn.__name__


def _tensors(value: Any) -> list:
    """Every torch tensor reachable from `value` (module parameters and buffers, attributes, containers)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return []
    from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

    atoms = (str, bytes, int, float, bool, complex, type(None), type, ModuleType, FunctionType, BuiltinFunctionType,
             MethodType, Path, torch.device, torch.dtype)
    found, seen, stack = [], set(), [(value, 0)]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen or depth > DEPTH or isinstance(obj, atoms):
            continue
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            found.append(obj)
            continue
        if isinstance(obj, torch.nn.Module):
            items = [*obj.parameters(recurse=False), *obj.buffers(recurse=False), *obj.children()]
            if not isinstance(obj, torch.jit.ScriptModule):
                items += [v for k, v in vars(obj).items() if k not in ("_parameters", "_buffers", "_modules")]
        elif isinstance(obj, dict):
            items = list(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            items = list(obj)
        else:
            items = list(getattr(obj, "__dict__", {}).values())
            items += [getattr(obj, s) for s in getattr(type(obj), "__slots__", ()) if isinstance(s, str) and hasattr(obj, s)]
        stack.extend((item, depth + 1) for item in items)
    return found


def _sizes(value: Any) -> tuple[int, int]:
    """(bytes on the GPU, bytes in RAM) of the tensors `value` holds, each storage counted once."""
    gpu, cpu, seen = 0, 0, set()
    for t in _tensors(value):
        if t.device.type == "meta":
            continue
        storage = t.untyped_storage()
        key = (str(t.device), storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)
        if t.device.type == "cpu":
            cpu += storage.nbytes()
        else:
            gpu += storage.nbytes()
    return gpu, cpu


def _move(t, device) -> None:
    """Move a tensor in place: every object holding it now holds it on `device`."""
    import torch

    if t.is_inference():
        with torch.inference_mode():
            t.data = t.data.to(device)
    else:
        t.data = t.data.to(device)


def _to_ram(entry: _Entry) -> bool:
    """Move the entry's GPU tensors to RAM if the machine then still has _KEEP_FREE_GB free; False when not."""
    if not entry.movable or _KEEP_FREE_GB is None:
        return False
    gpu, _ = _sizes(entry.value)
    if available_gb() - gpu / 2**30 < _KEEP_FREE_GB:
        return False
    moved = [(t, t.device) for t in _tensors(entry.value) if t.device.type not in ("cpu", "meta")]
    for t, _ in moved:
        _move(t, "cpu")
    entry.moved = (entry.moved or []) + moved
    return True


def _restore(entry: _Entry) -> None:
    for t, device in entry.moved:
        _move(t, device)
    entry.moved = None


def _on_gpu(entry: _Entry, gpu_bytes: int | None = None) -> bool:
    """Holds GPU memory: tensors on the GPU, or GPU memory PyTorch does not manage (movable=False)."""
    return not entry.movable or (_sizes(entry.value)[0] if gpu_bytes is None else gpu_bytes) > 0


def _leave_gpu(entry: _Entry) -> None:
    """To RAM, or freed."""
    if _on_gpu(entry) and not _to_ram(entry):
        _drop(entry)


def _drop(entry: _Entry) -> None:
    for key, e in list(_ENTRIES.items()):
        if e is entry:
            del _ENTRIES[key]


def _make_room() -> None:
    """Before loading or bringing back a model: the models this job has not used leave the GPU."""
    stale = [e for e in _ENTRIES.values() if e.job != _JOB and _on_gpu(e)]
    for entry in stale:
        _leave_gpu(entry)
    if stale:
        _free_cached()


def _free_cached() -> None:
    """After a job (and when a model is dropped): what the job freed goes back to the card and to the machine, so a
    process kept between jobs holds its models and nothing more. torch keeps freed GPU blocks and pinned host buffers
    in its caches, and glibc keeps the frames and arrays the job freed in its arenas (a buffer freed anywhere but at
    the top of a heap is never returned on its own): a kept process held up to 15 GB of RAM it no longer used."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        host = getattr(torch._C, "_host_emptyCache", None)  # pinned host memory cached by torch (torch >= 2.1)
        if host is not None:
            host()
    trim = _malloc_trim()
    if trim is not None:
        trim(0)


@functools.cache
def _malloc_trim():
    """glibc's malloc_trim, None on another C library."""
    import ctypes
    import platform

    if platform.libc_ver()[0] != "glibc":
        return None
    try:
        return getattr(ctypes.CDLL(None), "malloc_trim", None)
    except OSError:
        return None


def _status() -> dict:
    models = []
    for e in _ENTRIES.values():
        gpu, cpu = _sizes(e.value)
        models.append({"name": e.label, "gpu_mb": gpu >> 20, "ram_mb": cpu >> 20, "on_gpu": _on_gpu(e, gpu)})
    torch = sys.modules.get("torch")
    vram = torch.cuda.memory_reserved() >> 20 if torch is not None and torch.cuda.is_initialized() else 0
    return {"models": models, "vram_mb": vram, "context_mb": _CONTEXT_MB}


# --------------------------------------------------------------------------- the process's own share of its GPU

CONTEXT_MOST_MB = 4096  # a measured context beyond this is another program's change, not ours: not reported
SMI_TIMEOUT_S = 5.0
_CONTEXT_MB: int | None = None  # measured once (the module doc's context_mb)
_BEFORE_MB: int | None = None  # the card's memory.used when the job that brings CUDA up began


def _card_used_mb() -> int | None:
    """memory.used of the one GPU this process sees (CUDA_VISIBLE_DEVICES, as the core sets it), by nvidia-smi: read
    without creating a CUDA context. None: not exactly one GPU, or no answer."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible or "," in visible:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", visible, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=SMI_TIMEOUT_S)
        return int(out.stdout.strip().splitlines()[0]) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _cuda_up() -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and torch.cuda.is_initialized()


def _before_job() -> None:
    """Before a job: the card's use, while CUDA is not up in this process yet (what the job adds to it is ours)."""
    global _BEFORE_MB
    if _CONTEXT_MB is None and not _cuda_up():
        _BEFORE_MB = _card_used_mb()


def _after_job() -> None:
    """After the job (its cache freed): the first one that brought CUDA up measures the context once."""
    global _CONTEXT_MB, _BEFORE_MB
    if _CONTEXT_MB is not None or _BEFORE_MB is None or not _cuda_up():
        return
    after = _card_used_mb()
    if after is not None:
        reserved = sys.modules["torch"].cuda.memory_reserved() >> 20
        held = after - _BEFORE_MB - reserved
        if 0 < held <= CONTEXT_MOST_MB:
            _CONTEXT_MB = held
    _BEFORE_MB = None


# --------------------------------------------------------------------------- random state


def _rng_state() -> dict:
    import random

    import numpy as np

    state = {"random": random.getstate(), "numpy": np.random.get_state()}
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_initialized():
            state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    """Back to `state` (a generator that did not exist then keeps the job's random start)."""
    import random

    import numpy as np

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch = sys.modules.get("torch")
    if torch is not None and "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch is not None and "cuda" in state and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(state["cuda"])


def _seed_from_os() -> None:
    """Every generator seeded at random, as in a new process."""
    import random

    import numpy as np

    random.seed()
    np.random.seed()
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.seed()  # CPU and CUDA (also CUDA initialised later)


# --------------------------------------------------------------------------- serving


def serve(main: Callable[[str], None]) -> None:
    """A worker script's entry point: `if __name__ == "__main__": serve(main)`, `main(job_path)` runs one job."""
    from . import _pin_channel

    _pin_channel()  # upstream code replacing sys.stdout cannot cut the worker off from the core (see _pin_channel)
    args = sys.argv[1:]
    if args == ["--serve"]:
        _serve(main)
    elif len(args) == 1 and not args[0].startswith("-"):
        main(args[0])
    else:
        raise SystemExit("usage: worker.py <job.json>  |  worker.py --serve")


def _serve(main: Callable[[str], None]) -> None:
    from . import _emit

    commands = os.fdopen(os.dup(0), "r", encoding="utf-8")
    null = os.open(os.devnull, os.O_RDONLY)  # the worker's own stdin: nothing (upstream code asking a question gets EOF)
    os.dup2(null, 0)
    os.close(null)
    parent = int(os.environ.get("LAB2SHOT_CORE_PID") or os.getppid())  # the core may be gone before this line
    threading.Thread(target=_exit_with_parent, args=(parent,), daemon=True).start()
    code = 0
    for line in commands:
        cmd = json.loads(line)
        if cmd["cmd"] == "job":
            if not _job(main, cmd["job"], cmd["keep_free_gb"]):
                code = 1
                break
        elif cmd["cmd"] == "offload":
            _offload_all(cmd["keep_free_gb"])
            _emit("offloaded", **_status())
        elif cmd["cmd"] == "exit":
            break
    sys.stdout.flush()
    os._exit(code)  # no waiting on threads or exit handlers upstream left behind


def _exit_with_parent(parent: int) -> None:
    """The core went away (even killed): so does this process, also in the middle of a job."""
    while os.getppid() == parent:
        time.sleep(PARENT_POLL_S)
    os._exit(1)


def _job(main: Callable[[str], None], path: str, keep_free_gb: float) -> bool:
    """Run one job as a fresh process would; report how it ended. False: it failed (the process ends)."""
    from . import Failure, _emit, incompatible_gpu_failure, is_incompatible_gpu, is_oom, offline_failure

    global _JOB, _KEEP_FREE_GB
    _JOB += 1
    _KEEP_FREE_GB = keep_free_gb
    _seed_from_os()
    _before_job()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.reset_peak_memory_stats()
    error, oom, incompatible = None, False, False
    try:
        main(path)
    except SystemExit as exc:  # fail(), or the worker ending itself
        if exc.code not in (None, 0):
            incompatible = is_incompatible_gpu(exc)
            if incompatible:
                _emit("failed", **incompatible_gpu_failure(exc))
            error = "E-WORKER-GPUARCH" if incompatible else str(exc.code)
            oom = is_oom(exc)
            print(error, flush=True)  # the log's last line, as a process ending with it prints it
    except BaseException as exc:  # noqa: BLE001  (a bug or an upstream error: the job fails, the process ends)
        traceback.print_exc()  # its last line is the error, as in a process ending with it
        sys.stderr.flush()
        incompatible = is_incompatible_gpu(exc)
        # An incompatible GPU gets its own clear message (which torch, which card, what to do): the raw
        # "RuntimeError: CUDA error: no kernel image is available ..." reads like an internal bug or like running
        # out of memory.
        # A worker's own Failure says its code as fail() does; an upstream error after a refused connection most
        # likely comes from a model fetching a file at run time (E-WORKER-OFFLINE, see _forbid_downloads).
        said = (incompatible_gpu_failure(exc) if incompatible else {"code": exc.code, "params": exc.params}
                if isinstance(exc, Failure) else offline_failure())
        if said:
            _emit("failed", **said)
        error = said["code"] if said else f"{type(exc).__name__}: {exc}"
        oom = is_oom(exc)
    if error is None:
        for entry in [e for e in _ENTRIES.values() if e.job != _JOB]:
            _drop(entry)
    _free_cached()
    _after_job()
    _emit("job_end", ok=error is None, error=error, oom=oom, incompatible_gpu=incompatible, **_status())
    return error is None


def _offload_all(keep_free_gb: float) -> None:
    global _KEEP_FREE_GB
    _KEEP_FREE_GB = keep_free_gb
    for entry in list(_ENTRIES.values()):
        _leave_gpu(entry)
    _free_cached()
