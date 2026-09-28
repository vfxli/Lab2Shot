"""Helpers for EnvSpec.build scripts: they run with the extension environment's own Python (the SDK is installed
there before the build step), standard library only."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

MAX_JOBS = "4"  # parallel compiler processes: nvcc takes GBs of RAM each and the machine is shared


def run(args: list, cwd: Path | None = None) -> None:
    """Run a build command, shown first in the install log; a failure stops the build."""
    args = [str(a) for a in args]
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def compute_caps() -> list[str]:
    """The machine's GPU architectures (e.g. ["8.9", "12.0"]), for TORCH_CUDA_ARCH_LIST."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted({line.strip() for line in out.splitlines() if line.strip()})


def pip_cuda_home(prefix: Path) -> Path | None:
    """CUDA_HOME of the pip CUDA toolkit inside an environment (EnvSpec.cuda_toolkit); None when there is none.
    The wheels ship only versioned runtime libraries: the plain libcudart.so is linked so that -lcudart finds
    this one and not another CUDA on the system."""
    nvcc = sorted(Path(prefix).glob("lib/python*/site-packages/nvidia/*/bin/nvcc"))
    if not nvcc:
        return None
    home = nvcc[-1].parent.parent
    lib = home / "lib"
    for versioned in sorted(lib.glob("libcudart.so.*"))[:1]:
        if not (lib / "libcudart.so").exists():
            (lib / "libcudart.so").symlink_to(versioned.name)
    return home


def cuda_build_env(prefix: Path) -> dict[str, str]:
    """Environment for compiling CUDA extensions inside an EnvSpec.build script with the pip CUDA toolkit of the
    extension's environment: the installer hands build scripts a CPU-only environment. The architectures compiled for
    are the ones the installer chose (ARCHS_ENV: the setting build.archs, narrowed to the extension's env_archs); only a
    script run by hand, without the installer, falls back to this machine's cards."""
    from lab2shot_shared.gpu_arch import ARCHS_ENV

    home = pip_cuda_home(prefix)
    if home is None:
        from . import fail

        fail("E-WORKER-NONVCC")
    env = os.environ.copy()
    env.update(CUDA_HOME=str(home), PATH=f"{home / 'bin'}:{env.get('PATH', '')}", FORCE_CUDA="1", MAX_JOBS=MAX_JOBS,
               TORCH_CUDA_ARCH_LIST=os.environ.get(ARCHS_ENV) or ";".join(compute_caps()) or "8.9")
    return env


def modernize_torch_sources(folder: Path) -> int:
    """Old CUDA extensions (DPVO, DROID-SLAM, lietorch, pointops2) written for torch 1.x, fixed in a
    copy of their sources for torch 2.9: tensor.type() passed to the AT_DISPATCH macros
    becomes tensor.scalar_type() (device().type() is left alone), and the C++ frontend's
    removed torch::linalg:: namespace becomes the ATen functions (torch::linalg_<name>).
    Returns the number of replacements."""
    fixes = ((re.compile(r"(?<!device\(\))\.type\(\)(?!\.)"), ".scalar_type()"),
             (re.compile(r"torch::linalg::(\w+)\("), r"torch::linalg_\1("))
    count = 0
    for path in folder.rglob("*"):
        if path.suffix in (".cu", ".cpp", ".h", ".cuh") and path.is_file():
            text = path.read_text()
            for pattern, repl in fixes:
                text, n = pattern.subn(repl, text)
                count += n
            path.write_text(text)
    return count
