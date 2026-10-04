"""Determine which GPU architectures an environment's torch (and its compiled CUDA extensions) can run on, without
accessing any GPU. A compute-capability mismatch (e.g. a torch 2.3 environment on an RTX 5090, sm_120, Blackwell)
raises "no kernel image is available for execution on the device" rather than an out-of-memory error.

torch.cuda.get_arch_list() returns [] whenever CUDA is unavailable in the current process, including when
CUDA_VISIBLE_DEVICES="" is set deliberately. This module must run that way, so that installing or checking an
extension never touches a GPU. torch._C._cuda_getArchFlags() returns the same information (the architectures the
torch build was compiled for) directly from build metadata without any GPU, and is the source this module relies on.
When neither is available (very old torch, or torch not installed) the result is unknown (None) and is never treated
as compatible: a queue that cannot determine compatibility must wait rather than guess.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PROBE_TIMEOUT_S = 60
CUOBJDUMP_TIMEOUT_S = 30
# Only .so files at least this large are inspected (smaller ones rarely contain compiled CUDA kernels). torch's own
# libraries and the NVIDIA / Triton pip wheels are not the extension's code and are skipped.
KERNEL_MIN_BYTES = 200_000
KERNEL_SKIP = ("/torch/lib/", "/nvidia/", "/triton/")
# Version of the scan rules; must be incremented whenever any of the skip rules below changes.
# Scan records are cached by environment fingerprint and are not rescanned while the environment is unchanged. This
# version is part of the fingerprint, so a rule change triggers a rescan of every extension. Otherwise records made
# under the old rules would retain libraries that the new rules skip (e.g. NVIDIA's bundled libaccinj64 / libcufftw /
# libnppc, which contain only sm_52), and the extension would remain marked as unable to run.
#
# An extension may have several side-by-side environments (third_party/<name>/.venv, .venv-ada-blackwell, .venv-2,
# ...; see extensions/spec.py ExtensionPaths.env). Only one is active; the others are backups kept from earlier
# upgrades. Scanning all of third_party/<name>/ would include stale kernels from inactive environments (e.g. the
# torchvision/_C.so of torch 2.4.0+cu121, which stops at sm_90), rejecting an active environment that supports sm_120
# and leaving the queue with no eligible machine. Therefore only the active environment (`env`) and the extension's
# own compiled code are scanned.
SCAN_VERSION = 4
# PTX is told by cuobjdump's "PTX file" lines, never by the module's name: cuobjdump names an embedded PTX module
# "<lib>.1.sm_50.ptx" (never compute_50), and taking that for binary code of sm_50 would lose PTX's forward
# compatibility: unirig (torch_scatter, torch_cluster, spconv, open3d, all with PTX) and faceanything (xformers, PTX
# compute_80) would be judged unable to run on sm_120 although the driver JIT-compiles their PTX there.
# The conda counterpart of the same rule: conda environments (EnvSpec.conda) place the package manager's C libraries
# under <prefix>/lib (libmagma, libnccl, VTK's libviskores_*, ...), while the extension's own compiled code lives in
# <prefix>/lib/pythonX.Y/site-packages. Like torch/lib in a uv environment, the former is not the extension's own
# kernel code, and using it to judge GPU compatibility would wrongly exclude the extension (e.g. the 22 VTK
# libviskores_*.so files pulled in by pyvista cover only sm_50..sm_90, while the extension itself runs on both
# sm_89 and sm_120). site-packages is still scanned (diff-gaussian-rasterization and simple-knn reside there).
_CONDA_LIB = re.compile(r"(^|/)\.venv[^/]*/lib/(?!python\d)")

_SM_RE = re.compile(r"\bsm_(\d{2,3})\b")
_COMPUTE_RE = re.compile(r"\bcompute_(\d{2,3})\b")

_PROBE_SCRIPT = """
import json
result = {"torch_version": None, "archs": None}
try:
    import torch
    result["torch_version"] = torch.__version__
except Exception:
    pass
try:
    flags = torch._C._cuda_getArchFlags()
    if flags:
        result["archs"] = flags.split()
except Exception:
    pass
print(json.dumps(result))
"""


@dataclass(frozen=True)
class ArchRecord:
    """Architectures an environment's torch was compiled for. `archs` is a tuple of tokens such as "sm_90" (a binary
    kernel for exactly that architecture) or "compute_90" (PTX, which the driver can JIT-compile for that architecture
    or any newer one). None means not probed or undeterminable; an unknown value is never treated as compatible."""

    torch_version: str
    archs: tuple[str, ...] | None


def probe_torch_archs(python: Path, timeout: float = PROBE_TIMEOUT_S) -> ArchRecord:
    """Import torch in the interpreter `python` with all GPUs hidden (CUDA_VISIBLE_DEVICES="") and read the
    architectures its build was compiled for. Never raises: a missing torch, a crash or a timeout yields an unknown
    record (torch_version "", archs None)."""
    import os

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # deliberately hides all GPUs; see the module docstring
    env.pop("CUDA_DEVICE_ORDER", None)
    try:
        out = subprocess.run([str(python), "-c", _PROBE_SCRIPT], capture_output=True, text=True,
                             timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired):
        return ArchRecord("", None)
    try:
        data = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return ArchRecord("", None)
    archs = data.get("archs")
    return ArchRecord(data.get("torch_version") or "", tuple(archs) if archs else None)


def other_envs(root: Path, env: Path | None) -> frozenset[str]:
    """Return the folder names of the side-by-side environments under `root` other than `env` (".venv",
    ".venv-ada-blackwell", ".venv-2", ...).

    These are backup environments kept from upgrades (installer/plan.py installs the new environment under a new name
    and keeps the old one for rollback). The extension does not run in them, so their kernels must not be used to
    judge GPU compatibility. When `env` is None (the active environment is unknown) an empty set is returned, so that
    everything is scanned rather than risking the exclusion of the active environment."""
    if env is None:
        return frozenset()
    try:
        active = env.resolve()
    except OSError:
        active = env
    out = set()
    for child in root.glob(".venv*"):
        if not child.is_dir():
            continue
        try:
            same = child.resolve() == active
        except OSError:
            same = child == env
        if not same:
            out.add(child.name)
    return frozenset(out)


def listed_archs(text: str) -> tuple[str, ...]:
    """The architectures in `cuobjdump --list-elf --list-ptx` output: an "ELF file" line is binary code (sm_XY), a
    "PTX file" line is PTX (compute_XY), whatever its file name says (cuobjdump names PTX "<lib>.1.sm_50.ptx"). A line
    without either prefix is read by its token alone."""
    archs: set[str] = set()
    for line in text.splitlines():
        head = line.lstrip()
        if head.startswith("PTX file"):
            archs |= {f"compute_{m}" for m in _SM_RE.findall(line) + _COMPUTE_RE.findall(line)}
        elif head.startswith("ELF file"):
            archs |= {f"sm_{m}" for m in _SM_RE.findall(line)}
        else:
            archs |= {f"sm_{m}" for m in _SM_RE.findall(line)} | {f"compute_{m}" for m in _COMPUTE_RE.findall(line)}
    return tuple(sorted(archs))


def kernel_archs(root: Path, env: Path | None = None, cuobjdump: str | None = None) -> dict[str, tuple[str, ...]] | None:
    """Return {relative .so path -> embedded architectures (sm_XY / compute_XY)} for every compiled CUDA extension
    under `root` (an extension's install folder), read with `cuobjdump` (static analysis; no GPU is accessed).
    Returns None when cuobjdump is not available; kernels are then omitted from the compatibility record. torch's own
    support is still known, but a project's custom op may fail undetected until cuobjdump is available.

    `env` is the extension's active environment (ExtensionPaths.venv). Other side-by-side environments (backups from
    earlier upgrades) are skipped entirely, since the extension never loads their kernels (see the comment above
    SCAN_VERSION)."""
    cuobjdump = cuobjdump or shutil.which("cuobjdump") or _default_cuobjdump()
    if cuobjdump is None:
        return None
    skip_envs = other_envs(root, env)
    found: dict[str, tuple[str, ...]] = {}
    for so in sorted(root.rglob("*.so")):
        try:
            if so.stat().st_size < KERNEL_MIN_BYTES:
                continue
        except OSError:
            continue
        rel = so.relative_to(root).as_posix()
        if rel.partition("/")[0] in skip_envs:
            continue
        if any(skip.strip("/") in rel for skip in KERNEL_SKIP) or _CONDA_LIB.search("/" + rel):
            continue
        try:
            out = subprocess.run([cuobjdump, "--list-elf", "--list-ptx", str(so)], capture_output=True, text=True,
                                 timeout=CUOBJDUMP_TIMEOUT_S)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = out.stdout
        if not text.strip():
            continue
        found[rel] = listed_archs(text)
    return found


def _default_cuobjdump() -> str | None:
    for candidate in sorted(Path("/usr/local").glob("cuda*/bin/cuobjdump"), reverse=True):
        if candidate.exists():
            return str(candidate)
    return None


def cap_to_sm(compute_cap: str) -> str:
    """Convert an nvidia-smi compute_cap ("8.9", "12.0") to the sm_XY token used by torch ("sm_89", "sm_120")."""
    major, _, minor = compute_cap.partition(".")
    return f"sm_{int(major)}{int(minor or 0)}"


# CUDA architectures by family, used to name the architectures an environment supports
# (third_party/<name>/.venv-<families>). A family covers all of its GPUs, so no GPU model appears in environment
# names or declarations.
FAMILIES = {"sm_75": "turing", "sm_80": "ampere", "sm_86": "ampere", "sm_87": "ampere", "sm_89": "ada", "sm_90": "hopper",
            "sm_100": "blackwell", "sm_120": "blackwell"}

# The architectures CUDA code can be compiled for (the setting build.archs), each named by its platform, never by a
# graphics card: what is compiled depends on these, not on the cards the building machine happens to have.
TARGET_LABELS = {"sm_75": "Turing", "sm_80": "Ampere (data center)", "sm_86": "Ampere", "sm_89": "Ada Lovelace",
                 "sm_90": "Hopper", "sm_100": "Blackwell (data center)", "sm_120": "Blackwell"}
# How the installer hands the chosen architectures to a build script (EnvSpec.build), which runs in the extension's own
# environment and cannot read the settings: "8.9;12.0", the form of TORCH_CUDA_ARCH_LIST
# (lab2shot_worker.build.cuda_build_env reads it)
ARCHS_ENV = "LAB2SHOT_CUDA_ARCHS"


def target_label(token: str) -> str:
    """A compile target as the menu and the admin page show it: "Ada Lovelace (sm_89)" (platform names: not translated)."""
    return f"{TARGET_LABELS.get(token, token)} ({token})"


def cap_of(token: str) -> str:
    """sm_89 -> "8.9", sm_120 -> "12.0": the form TORCH_CUDA_ARCH_LIST takes."""
    major, minor = _cc(token) or (0, 0)
    return f"{major}.{minor}"


def env_name(archs) -> str:
    """Return the side-by-side environment name for the declared architectures, e.g. ("sm_89", "sm_120") ->
    "ada-blackwell"; "" when none are declared (the active .venv)."""
    names: list[str] = []
    for a in archs:
        if a not in FAMILIES:
            raise ValueError(f"not a known CUDA architecture: {a!r} (one of {sorted(FAMILIES)})")
        if FAMILIES[a] not in names:
            names.append(FAMILIES[a])
    return "-".join(names)


def _cc(token: str) -> tuple[int, int] | None:
    """Return the compute capability (major, minor) of an sm_XY / compute_XY token, or None if it cannot be parsed.

    The last digit is the minor version and the remaining digits the major (consistent with all existing tokens:
    sm_86 -> 8.6, sm_120 -> 12.0, sm_100 -> 10.0). The trailing letter some CUDA 12 tokens carry for family-specific
    features (sm_90a) is ignored: it only narrows compatibility, and no probe here reports it separately from the
    plain architecture."""
    m = re.match(r"^(?:sm|compute)_(\d+?)(\d)[a-z]*$", token)
    return (int(m.group(1)), int(m.group(2))) if m else None


def compatible(archs: tuple[str, ...] | None, target_sm: str) -> bool | None:
    """Return whether code built for `archs` can run on `target_sm` (e.g. "sm_120").

    True if any of the following holds:
    - an exact binary match;
    - a binary (sm_XY) of the same major generation with an equal or lower minor version (CUDA's binary
      compatibility guarantee within a generation: a torch 2.3 build lists only up to sm_86 with no PTX, yet runs on
      an RTX 4090 (sm_89, major 8), but not on an RTX 5090 (sm_120, a different major generation));
    - PTX (compute_XY) for that architecture or any older one, regardless of major generation, since PTX is
      forward-compatible by design.
    False if `archs` is known and none of them covers the target. None if `archs` is unknown (not probed, or the
    probe failed); an unknown value is never assumed compatible."""
    if archs is None:
        return None
    if target_sm in archs:
        return True
    target = _cc(target_sm)
    if target is None:
        return False
    for a in archs:
        cc = _cc(a)
        if cc is None:
            continue
        if a.startswith("sm_") and cc[0] == target[0] and cc[1] <= target[1]:
            return True
        if a.startswith("compute_") and cc <= target:
            return True
    return False


def kernels_incompatible(kernels: dict[str, tuple[str, ...]] | None, target_sm: str) -> tuple[str, ...]:
    """Return the compiled .so files (from kernel_archs()) that contain neither binary code nor PTX usable on
    `target_sm`; () when `kernels` is None (not scanned) or all of them are compatible."""
    if not kernels:
        return ()
    return tuple(name for name, archs in kernels.items() if compatible(archs, target_sm) is not True)
