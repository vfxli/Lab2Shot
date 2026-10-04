"""Which GPU architectures an extension's environment (or the core's own) can run: probed once (lab2shot_shared.
gpu_arch, no GPU touched; see its docstring), kept with the extension's install record and invalidated when the
environment changes (the fingerprint step_env() records). An environment with no record gets one the first time
anything asks (ensure), not a guess.

Why this exists: a torch build compiled for older GPUs only (e.g. torch 2.3, sm_50..sm_90) raises "no kernel image
is available for execution on the device" on a newer one (RTX 5090, sm_120, Blackwell) instead of computing, and
upstream error handling can turn that into something that looks like a memory problem. The queue (farm/scheduler/)
checks this before it ever starts a job on a card that cannot run it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from lab2shot_shared.gpu_arch import SCAN_VERSION, kernel_archs, probe_torch_archs

from ..io import atomic

CORE_KEY = "gpu_archs.core"  # database meta key for the main environment's own record


@dataclass(frozen=True)
class ExtensionArchRecord:
    """What is recorded for one environment: torch's own compiled architectures, and (when cuobjdump is on this
    machine) every compiled CUDA extension .so that does not simply follow torch's; a project's custom kernels can
    be built for fewer architectures than the torch they sit on."""

    torch_version: str
    archs: tuple[str, ...] | None  # None: unknown (torch missing, or the probe could not tell)
    kernels: dict[str, tuple[str, ...]] | None  # None: cuobjdump was not available to scan for these
    fingerprint: str  # the environment's step_env() fingerprint (the scanner's version is `scan`)
    scan: int = 0  # lab2shot_shared.gpu_arch.SCAN_VERSION this was scanned with (0: none recorded)

    def to_json(self) -> dict:
        return {**asdict(self), "kernels": None if self.kernels is None else dict(self.kernels)}

    @classmethod
    def from_json(cls, raw: dict) -> "ExtensionArchRecord":
        archs = raw.get("archs")
        kernels = raw.get("kernels")
        return cls(raw.get("torch_version") or "", tuple(archs) if archs else None,
                   None if kernels is None else {k: tuple(v) for k, v in kernels.items()}, raw.get("fingerprint", ""),
                   int(raw.get("scan") or 0))


def probe(python: Path, root: Path, fingerprint: str) -> ExtensionArchRecord:
    """Probe one environment now (python: its interpreter; root: its install folder, scanned for compiled .so).

    Only this environment (the .venv of `python`) and what the repository itself compiled are scanned; sibling
    environments (.venv / .venv-2 ... kept as backups from before an upgrade) are skipped entirely: the extension never
    loads their old kernels, and scanning them would misjudge usable cards as unusable."""
    torch = probe_torch_archs(python)
    kernels = kernel_archs(root, env=python.parent.parent)
    return ExtensionArchRecord(torch.torch_version, torch.archs, kernels, fingerprint, SCAN_VERSION)


def compiled_modules(kernels: dict[str, tuple[str, ...]] | None, roots: tuple[str, ...], base: Path) -> tuple[str, ...]:
    """Import names of the importable modules compiled in this environment, derived from the scanned .so paths
    rather than a hand-written list.

    Rationale: the post-install self-check imports only `lab2shot_worker` / `torch` / `torchvision` and the modules an
    extension lists in `EnvSpec.imports`, which is optional and usually empty. When a compiled artifact (e.g.
    `sam2/_C.so`) was built before an environment upgrade and no longer loads, and upstream swallows the exception
    with `except`, the self-check passes while the component is missing (masks silently left unfilled). Importing by
    scan result covers any third-party project automatically: whatever it compiles is imported by the self-check,
    without anyone having to declare it.

    The derivation is mechanical, with no guessing:
    1. the root the path lies under (site-packages, or a directory on the extension's PYTHONPATH), longest root
       first, otherwise `repo/sam2/sam2/_C.so` would be matched by `repo` and become `sam2.sam2._C`;
    2. the first segment of the file name (`_C.cpython-312-....so` -> `_C`);
    3. every level must be a package: each intermediate directory must contain `__init__.py`, otherwise it is not an
       importable module;
    4. `lib...` C++ shared libraries are skipped (OpenPose's `libopenpose`, Caffe's `libcaffe`, torchaudio's
       `libtorchaudio`): they are linked, not imported.

    Anything that cannot be derived is skipped: under-reporting is preferred to blocking an install with a false name."""
    out: list[str] = []
    ordered = sorted({r.strip("/") for r in roots if r.strip("/")}, key=len, reverse=True)
    for rel in sorted(kernels or {}):
        parts = rel.split("/")
        name = parts[-1].split(".")[0]
        if name.startswith("lib"):        # a linked C++ library, not an importable module
            continue
        for root in ordered:
            bits = root.split("/")
            if parts[:len(bits)] != bits:
                continue
            pkg = parts[len(bits):-1]
            if not all(part.isidentifier() for part in [*pkg, name]):
                break
            here = base / root
            if all((here := here / part).joinpath("__init__.py").exists() for part in pkg):
                out.append(".".join([*pkg, name]))
            break
    return tuple(dict.fromkeys(out))


def _current(cached: dict | None, fingerprint: str) -> bool:
    """Whether the record is still valid: the environment is unchanged and it was scanned with the current scan rules.

    The environment fingerprint alone is insufficient: when the scan rules change while the environment is
    byte-for-byte unchanged, a record from the old rules would stay in use, usable cards would be judged unusable and
    tasks would stall in the queue. Therefore
    `lab2shot_shared.gpu_arch.SCAN_VERSION` is compared as well: changing it rescans every extension on its next query."""
    return bool(cached) and cached.get("fingerprint") == fingerprint and int(cached.get("scan") or 0) == SCAN_VERSION


def record(ext) -> ExtensionArchRecord:
    """Probe `ext`'s environment now and save it with its install state (the installer's self-check records it too,
    once the environment is ready; ensure() fills in an environment that has no record yet).

    Only adds an entry to the record and never replaces the whole record. The install record identifies this
    environment (which repository, which packages, whether weights were verified, whether the self-check ran);
    `Extension.install_state()` returns `{}` when it cannot be read, and writing `{}` back would erase it. After that
    `built_for()` returns None and the extension shows as needing reinstallation on the admin page even though the
    environment and weights are present, and its nodes fail when cooked. Therefore, when the file exists but cannot be
    read (partially written or corrupt), nothing is written and the probe result is returned directly.

    The probe takes seconds and the installer may save the record meanwhile: the record is read again after the
    probe and the entry merged into it, both under the record's one-writer lock (spec.state_writing), so nothing the
    installer wrote is lost; an environment that changed during the probe (another fingerprint) is not described by
    it, and nothing is written. Writes use an atomic replace, as the installer does, so readers never see a partial
    file."""
    from .spec import state_writing

    ext = ext.env_owner  # an extension running in another's environment: that environment's record
    fingerprint = ext.install_state().get("env", {}).get("fingerprint", "")
    found = probe(ext.paths.python, ext.paths.root, fingerprint)
    with state_writing(ext.paths):
        state = ext.install_state()
        if not state and ext.paths.state_file.exists():
            return found  # an unreadable record must not be overwritten by this entry
        if state.get("env", {}).get("fingerprint", "") != fingerprint:
            return found  # the environment changed while it was probed: this is not its record
        state["gpu_archs"] = found.to_json()
        atomic.write_text(ext.paths.state_file, json.dumps(state, indent=2, ensure_ascii=False))
    return found


def recorded(ext) -> ExtensionArchRecord | None:
    """`ext`'s recorded architectures when the record is there and still current, else None (to be probed:
    `ensure`); never probes. The record of the environment it runs in (Extension.env_owner)."""
    state = ext.env_owner.install_state()
    cached = state.get("gpu_archs")
    return ExtensionArchRecord.from_json(cached) if _current(cached, state.get("env", {}).get("fingerprint", "")) else None


def ensure(ext) -> ExtensionArchRecord | None:
    """`ext`'s recorded architectures, probing (and saving) now if there is none yet or the environment changed
    since; None when the environment is not installed at all (nothing to probe)."""
    ext = ext.env_owner
    if (found := recorded(ext)) is not None:
        return found
    if not ext.paths.python.exists():
        return None
    return record(ext)


def core_fingerprint() -> str:
    """What identifies the main environment's build, for cache invalidation: its own interpreter path plus its
    installed torch's version when it has one (most core environments have none: GPU nodes live in extensions)."""
    try:
        import torch  # noqa: PLC0415

        return f"{sys.executable}:{torch.__version__}"
    except ImportError:
        return sys.executable


@lru_cache(maxsize=8)
def _probe_core(fingerprint: str) -> ExtensionArchRecord:
    """Only torch's own build is probed for the core: it never compiles a CUDA extension of its own (the core holds
    no project- or format-specific code), so scanning its whole environment for compiled kernels
    (as an extension's install folder gets: gpu_archs.probe) would only cost time for nothing. Cached in-process too
    (not just in the database): the fingerprint cannot change within one process."""
    torch = probe_torch_archs(Path(sys.executable))
    return ExtensionArchRecord(torch.torch_version, torch.archs, None, fingerprint, SCAN_VERSION)


def recorded_core() -> ExtensionArchRecord | None:
    """The main environment's recorded architectures when the database has them and they are still current, else
    None (to be probed: `ensure_core`); never probes."""
    from ..database import db

    cached = db().meta(CORE_KEY, None)
    return ExtensionArchRecord.from_json(cached) if _current(cached, core_fingerprint()) else None


def ensure_core() -> ExtensionArchRecord:
    """The main environment's own architectures (for a GPU node declared in the core rather than in an
    extension): cached in the database, reprobed when the interpreter or its torch changes."""
    from ..database import db

    if (found := recorded_core()) is not None:
        return found
    found = _probe_core(core_fingerprint())
    db().set_meta(CORE_KEY, found.to_json())
    return found
