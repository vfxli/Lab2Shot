"""The machine's build kit for compiled extensions: the CUDA toolkit and the C/C++ compiler the settings point at,
and whether nvcc accepts that compiler — nvcc refuses a GCC newer than it knows ("unsupported GNU version"), the one
compile failure users hit most and cannot read out of a page of build output. One table, used by the installer's
preflight (a check on every extension that compiles CUDA) and by `lab2shot setup` (扩展包编译与下载设置 → 检查编译工具)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .. import config
from ..messages import Msg

# nvcc release (major, minor) -> the newest GCC major it accepts (NVIDIA's "supported host compilers" per release).
# tuples, not floats: 12.10 as a float is 12.1 and would sort before 12.4
MAX_GCC = (((13, 0), 15), ((12, 8), 14), ((12, 4), 13), ((12, 0), 12), ((11, 4), 11), ((11, 1), 10), ((11, 0), 9), ((10, 2), 8))

Release = tuple[int, int]


def release_text(nvcc: Release) -> str:
    return f"{nvcc[0]}.{nvcc[1]}"


def cuda_home() -> Path:
    s = config.settings()
    return Path(s["build.cuda_home"] or os.environ.get("CUDA_HOME") or "/usr/local/cuda")


def nvcc_version(home: Path | None = None) -> Release | None:
    """nvcc's release ((12, 8)) at the toolkit home; None when there is no nvcc."""
    exe = (home or cuda_home()) / "bin" / "nvcc"
    if not exe.is_file():
        return None
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def compiler() -> tuple[str, str | None]:
    """(the C compiler the settings name, else `gcc`; its path or None). gcc, not cc: with build.cc empty torch passes
    no -ccbin, and nvcc then runs `gcc` from PATH — `cc` may point at another compiler (clang, an older gcc) and the
    check would judge the wrong one."""
    cc = config.settings()["build.cc"] or "gcc"
    return cc, shutil.which(cc)


def gcc_major(path: str) -> int | None:
    """The GCC major version of a compiler (None: not GCC, or it will not say)."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    if "gcc" not in out.lower() and "GNU" not in out:
        return None
    m = re.search(r"\b(\d+)\.\d+\.\d+\b", out)
    return int(m.group(1)) if m else None


def max_gcc(nvcc: Release) -> int:
    return next((g for release, g in MAX_GCC if tuple(nvcc) >= release), MAX_GCC[-1][1])


def pip_nvcc_release(cuda_toolkit: tuple[str, ...]) -> Release | None:
    """The release of an nvcc an extension brings from pip (EnvSpec.cuda_toolkit: "nvidia-cuda-nvcc==13.2.86" -> (13, 2))."""
    for pin in cuda_toolkit:
        m = re.match(r"nvidia-cuda-nvcc==(\d+)\.(\d+)", pin)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    return None


def problem(nvcc: Release | None = None) -> tuple[str, Msg | None]:
    """("ok" | "warning" | "blocked", why). `nvcc`: the release of the nvcc that will compile (an extension's own from
    pip); None: the machine toolkit the settings point at, which must then be there. Either way the host compiler is
    the machine's (settings build.cc), and nvcc must accept it."""
    if nvcc is None:
        home = cuda_home()
        nvcc = nvcc_version(home)
        if nvcc is None:
            return "blocked", Msg("E-TOOLCHAIN-NONVCC", folder=str(home))
    cc, path = compiler()
    if path is None:
        return "blocked", Msg("E-TOOLCHAIN-NOCC", compiler=cc)
    major = gcc_major(path)
    if major is None:
        return "warning", Msg("W-TOOLCHAIN-NOTGCC", compiler=path)
    if major > max_gcc(nvcc):
        return "blocked", Msg("E-TOOLCHAIN-GCCTOONEW", compiler=f"{path}（GCC {major}）", nvcc=release_text(nvcc), max=max_gcc(nvcc))
    return "ok", Msg("I-TOOLCHAIN-OK", nvcc=release_text(nvcc), compiler=f"{path}（GCC {major}）")
