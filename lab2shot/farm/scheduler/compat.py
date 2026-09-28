"""Whether a runtime's environment (an extension's, or the core's own) can run on a given GPU's architecture
(compute capability): a torch build compiled for older cards only raises "no kernel image is available for
execution on the device" on a newer one instead of computing (e.g. a torch 2.3 environment, sm_50..sm_90 only and
no PTX at all, on an RTX 5090, Blackwell sm_120). farm/scheduler/placement.py uses this to
keep a job from ever being placed on a card it cannot run on, and to explain why none fit.

Symmetric by construction: nothing here is specific to any one GPU generation, so an environment that runs on the
5090 but not the 4090 (or the other way around) is judged exactly the same way. For example, torch 2.3's build lists only up to sm_86 and has no PTX, yet genuinely runs on the RTX 4090 (sm_89, same
major generation "8": CUDA's own binary-compatibility guarantee, not PTX — see gpu_arch.compatible's docstring) and
only fails on the RTX 5090 (sm_120, a different major generation the build has nothing for).

Whether it runs is decided by declarations alone: the architectures the extension declares (Extension.env_archs,
and the install-time record of extensions/gpu_archs.py) plus the static judgement here. There is no other source, so
two places never come to different conclusions.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from lab2shot_shared.gpu_arch import cap_to_sm, compatible, kernels_incompatible

from ... import logs
from ...extensions import extensions, gpu_archs
from ...messages import Msg
from .inventory import GpuState


@dataclass(frozen=True)
class Fit:
    """Whether one runtime can run on one GPU. `known`: False means the environment was never probed (no install,
    or the probe could not tell) — never treated as compatible."""

    ok: bool | None  # True / False / None (unknown: treat as not fitting, but say so differently)
    bad_kernels: tuple[str, ...] = ()  # compiled .so files that do not cover this architecture (even if torch does)
    undeclared: tuple[str, ...] = ()  # refused: the extension's environment declares these architectures, not this one
    probing: bool = False  # unknown for a moment: the environment is being probed now (runtime_record)

    @property
    def known(self) -> bool:
        return self.ok is not None


def runtime_title(runtime: str) -> str:
    from ...adapters import project_of

    return project_of(runtime).title


PROBING = "probing"  # runtime_record: the environment is being probed now, on a thread of its own


def _stamp(ext) -> tuple:
    """What says an extension's install record changed: the file's time and size (a probe or an install writes it),
    and whether the environment is there at all."""
    try:
        st = ext.paths.state_file.stat()
        written = (st.st_mtime_ns, st.st_size)
    except OSError:
        written = None
    return written, ext.paths.python.exists()


class _Records:
    """The runtimes' recorded architectures as everything here reads them (the dispatcher holding the pools' lock
    for every GPU node it places, the cards page on a request's thread). A record is read from the extension's install
    record (the core's: the database) and kept while that file is unchanged, so a pass of the dispatcher reads no file.
    One missing or out of date is probed (gpu_archs: a subprocess, seconds) on a thread of its own, never on the
    caller's; until it is through, the runtime reads PROBING (a wait that ends by itself)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.kept: dict[str, tuple[tuple, object]] = {}  # runtime -> (its stamp when read, its record or None)
        self.probing: set[str] = set()

    def get(self, runtime: str):
        ext = None if runtime == "core" else extensions().get(runtime)
        if runtime != "core" and ext is None:
            return None
        stamp = ("core",) if ext is None else _stamp(ext)  # the core's cannot change within one process
        with self.lock:
            hit = self.kept.get(runtime)
            if hit is not None and hit[0] == stamp:
                return hit[1]
            if runtime in self.probing:
                return PROBING
        if ext is not None and not stamp[1]:
            found = None  # not installed: nothing to probe
        else:
            found = gpu_archs.recorded_core() if ext is None else gpu_archs.recorded(ext)
            if found is None:  # never probed, or its environment changed since
                with self.lock:
                    if runtime not in self.probing:
                        self.probing.add(runtime)
                        threading.Thread(target=self._probe, args=(runtime, ext), daemon=True,
                                         name=f"gpu-archs-{runtime}").start()
                return PROBING
        with self.lock:
            self.kept[runtime] = (stamp, found)
        return found

    def _probe(self, runtime: str, ext) -> None:
        try:
            found = gpu_archs.ensure_core() if ext is None else gpu_archs.ensure(ext)
        except Exception as exc:  # noqa: BLE001 (the record could not be written: unknown, never guessed compatible)
            logs.say(logs.get("farm"), Msg("E-FARM-INTERNAL", detail=f"{runtime}: {exc}"), logs.error_text(exc))
            found = gpu_archs.ExtensionArchRecord("", None, None, "")
        with self.lock:
            self.probing.discard(runtime)
            self.kept[runtime] = (("core",) if ext is None else _stamp(ext), found)


_RECORDS = _Records()


def runtime_record(runtime: str):
    """This runtime's recorded architectures (lab2shot.extensions.gpu_archs); PROBING while a probe of its
    environment runs (started here when it never was probed, or its environment changed since: `_Records`). None:
    the extension is not installed (nothing to probe)."""
    return _RECORDS.get(runtime)


def declared_archs(runtime: str) -> tuple[str, ...]:
    """The architectures an extension's environment is declared for (Extension.env_archs); () when it declares none."""
    ext = extensions().get(runtime)
    return tuple(getattr(ext, "env_archs", ()) or ()) if ext is not None else ()


def fit(runtime: str, gpu: GpuState) -> Fit:
    """Can `runtime`'s environment run on `gpu`? "core" is a special case: no core node needs a GPU today (every
    GPU node is an extension's), so an ordinary main environment — no torch installed at all, on purpose (the core
    depends on no paper's maths) — imposes no architecture restriction of its own; only a core environment that
    does have a torch build (were one ever added) is checked like an extension's."""
    if runtime == "core":
        core = runtime_record("core")
        if core is PROBING:
            return Fit(None, probing=True)
        if core is None or not core.torch_version:
            return Fit(True)
    if not gpu.compute_cap:
        # nvidia-smi always reports this for real NVIDIA hardware; not having it is an inventory gap (an unusual
        # driver, or a test double), not a sign this card cannot run something — nothing to restrict on, unlike an
        # environment that was probed and still came back unknown (rec.archs is None below).
        return Fit(True)
    target = cap_to_sm(gpu.compute_cap)
    rec = runtime_record(runtime)
    if rec is PROBING:
        return Fit(None, probing=True)
    if rec is None:
        # No record at all: the extension is not in the registry (a node type that reached the queue with no such
        # extension would already have failed to load, so this is not a real production case) or not installed yet
        # (ditto). Nothing to restrict on — unlike rec.archs is None below, which is a real, probed-and-still-
        # unknown environment and must never be guessed as compatible.
        return Fit(True)
    declared = declared_archs(runtime) if runtime != "core" else ()
    if declared and compatible(declared, target) is not True:  # the environment is built for other architectures
        return Fit(False, undeclared=declared)
    return static_fit(rec, target)


def static_fit(rec, target: str) -> Fit:
    """What the environment's probed architectures and compiled kernels alone say about `target` ("sm_120")."""
    if rec.archs is None:
        if not rec.torch_version and rec.kernels:
            # No torch here at all (a C++/CUDA tool, e.g. COLMAP's pycolmap SiftGPU): nothing to probe with
            # torch._C._cuda_getArchFlags, but its own compiled kernels were scanned directly and say something —
            # judge by those alone rather than calling the whole environment unknown.
            bad = kernels_incompatible(rec.kernels, target)
            return Fit(not bad, bad)
        return Fit(None)  # torch is used here but its build could not be read: a real unknown, never guessed compatible
    ok = compatible(rec.archs, target)
    bad = kernels_incompatible(rec.kernels, target) if ok else ()
    return Fit(ok and not bad, bad)


def _gpu_extensions() -> list:
    """Installed extensions with at least one node that runs on a GPU (the only ones a card's architecture matters
    to)."""
    from ...nodes import node_types

    on_gpu = {t.runtime for t in node_types().values() if t.cost.gpu}
    return [ext for ext in extensions().values() if ext.name in on_gpu and ext.paths.python.exists()]


ASSUMED_STATE, REFUSED_STATE, UNKNOWN_STATE = "assumed", "refused", "unknown"


def card_extensions(gpu: GpuState) -> dict[str, list[dict]]:
    """The admin GPU panel, per card: every installed GPU extension as assumed (its declared architectures and the
    static scan say it runs here), refused (they say it cannot) or unknown (the environment could not be probed),
    each with its reason as a catalogue message."""
    groups: dict[str, list[dict]] = {ASSUMED_STATE: [], REFUSED_STATE: [], UNKNOWN_STATE: []}
    if not gpu.compute_cap:
        return groups
    for ext in _gpu_extensions():
        state, msg = explain(ext.name, gpu)
        groups[state].append({"extension": ext.name, "title": ext.title, "message": msg.json()})
    for items in groups.values():
        items.sort(key=lambda i: i["title"].lower())
    return groups


def explain(runtime: str, gpu: GpuState) -> tuple[str, Msg]:
    """One extension on one card: its state (card_extensions) and why."""
    title = runtime_title(runtime)
    target = cap_to_sm(gpu.compute_cap)
    rec = runtime_record(runtime)
    if rec is PROBING:
        return UNKNOWN_STATE, Msg("N-GPU-PROBING", project=title)
    if rec is None:
        return UNKNOWN_STATE, Msg("W-GPU-NOTPROBED", project=title)
    got = fit(runtime, gpu)
    if got.undeclared:
        return REFUSED_STATE, Msg("W-GPU-ENVARCHS", project=title, gpus=gpu.short_name, arch=target, archs=list(got.undeclared))
    if got.ok is None:
        return UNKNOWN_STATE, Msg("W-GPU-NOARCH", project=title, extension=runtime)
    if got.ok is False:
        return REFUSED_STATE, _static_problem(title, rec, gpu)
    torch_bit = f"PyTorch {rec.torch_version}" if rec.torch_version else "CUDA"
    return ASSUMED_STATE, Msg("I-GPU-ASSUMED", project=title, arch=target, torch=torch_bit)


def _static_problem(title: str, rec, gpu: GpuState) -> Msg:
    """Why the static scan refuses one card."""
    target = cap_to_sm(gpu.compute_cap)
    if rec.archs is None:
        return Msg("W-GPU-KERNELS", project=title, kernels=list(kernels_incompatible(rec.kernels, target)[:3]) or "?",
                   gpus=gpu.short_name)
    torch_bit = f"PyTorch {rec.torch_version}" if rec.torch_version else "CUDA"
    bad = kernels_incompatible(rec.kernels, target)
    if compatible(rec.archs, target) and bad:
        return Msg("W-GPU-TORCHKERNELS", project=title, torch=torch_bit, gpus=gpu.short_name, kernels=list(bad[:3]))
    return Msg("W-GPU-ARCH", project=title, torch=torch_bit, gpus=gpu.short_name)


def wait_reason(runtimes: set[str], authorized: list[GpuState], inventory: list[GpuState]) -> Msg | None:
    """Why a GPU job with these runtimes cannot start on any authorized GPU right now (architecture, not memory or
    a paused switch), as a message (several runtimes: N-GPU-WAITSEVERAL); None when at least one authorized GPU can run
    every one of them, or when no GPU is authorized at all (placement.reason says that)."""
    if not authorized:
        return None
    problems = [_runtime_problem(runtime, authorized, inventory) for runtime in sorted(runtimes)
                if not any(fit(runtime, g).ok for g in authorized)]
    if not problems:
        return None
    return problems[0] if len(problems) == 1 else Msg("N-GPU-WAITSEVERAL", problems=problems)


def _runtime_problem(runtime: str, authorized: list[GpuState], inventory: list[GpuState]) -> Msg:
    title = runtime_title(runtime)
    rec = runtime_record(runtime)
    if rec is PROBING:  # not known yet: a wait that ends by itself when the probe is through
        return Msg("N-GPU-PROBING", project=title)
    authed_names = "、".join(sorted({g.short_name for g in authorized}))
    declared = declared_archs(runtime) if runtime != "core" else ()
    if declared and all(fit(runtime, g).undeclared for g in authorized):
        return Msg("W-GPU-ENVARCHS", project=title, gpus=authed_names, arch="、".join(sorted({cap_to_sm(g.compute_cap) for g in authorized})),
                   archs=list(declared))
    if rec is None:
        return Msg("W-GPU-NOTPROBED", project=title)
    if rec.archs is None:
        if not rec.torch_version and rec.kernels:
            # No torch at all here (a C++/CUDA tool, e.g. COLMAP): judged by its own compiled CUDA kernels directly.
            bad = next((kernels_incompatible(rec.kernels, cap_to_sm(g.compute_cap)) for g in authorized
                       if kernels_incompatible(rec.kernels, cap_to_sm(g.compute_cap))), ())
            return Msg("W-GPU-KERNELS", project=title, kernels=list(bad[:3]) or "未知", gpus=authed_names)
        if runtime == "core":
            return Msg("W-GPU-NOARCHCORE", project=title)
        return Msg("W-GPU-NOARCH", project=title, extension=runtime)
    torch_bit = f"PyTorch {rec.torch_version}" if rec.torch_version else "运行环境"
    # torch itself supports at least one authorized card, but its compiled CUDA kernels do not
    bad_kernels: tuple[str, ...] = ()
    for g in authorized:
        if compatible(rec.archs, cap_to_sm(g.compute_cap)):
            bad_kernels = kernels_incompatible(rec.kernels, cap_to_sm(g.compute_cap))
            if bad_kernels:
                break
    if bad_kernels:
        return Msg("W-GPU-TORCHKERNELS", project=title, torch=torch_bit, gpus=authed_names, kernels=list(bad_kernels[:3]))
    unauthorized_that_fit = sorted({g.short_name for g in inventory
                                    if g.uuid not in {a.uuid for a in authorized} and fit(runtime, g).ok})
    if unauthorized_that_fit:
        return Msg("W-GPU-AUTHORIZEOTHER", project=title, torch=torch_bit, gpus=authed_names, others=unauthorized_that_fit)
    return Msg("W-GPU-ARCH", project=title, torch=torch_bit, gpus=authed_names)
