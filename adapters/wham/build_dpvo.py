"""EnvSpec.build for WHAM: compile DPVO (visual odometry: camera rotation for moving
cameras) from WHAM's own submodule, run with the environment's Python.

The installer has checked out the repo's third-party/ submodules (DPVO, ViTPose) at
the commits WHAM records, and Eigen 3.4.0 (what DPVO's setup.py expects) into
<ext root>/eigen-3.4.0. DPVO is copied to <ext root>/build/DPVO (repo/ stays
untouched) together with Eigen and built with
the environment's pip CUDA 13.2 toolkit. Two source fixes for torch 2.9 in the copy:
the removed <THC/THCAtomics.cuh> header becomes <ATen/cuda/Atomic.cuh> (same
atomicAdd overloads), and tensor.type() in the dispatch macros becomes
tensor.scalar_type() (lab2shot_worker.build.modernize_torch_sources).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env, modernize_torch_sources

ext_root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])

build = ext_root / "build" / "DPVO"
if build.exists():
    shutil.rmtree(build)
shutil.copytree(repo / "third-party" / "DPVO", build, ignore=shutil.ignore_patterns(".git", "Pangolin", "datasets"))

shutil.copytree(ext_root / "eigen-3.4.0", build / "thirdparty" / "eigen-3.4.0", ignore=shutil.ignore_patterns(".git"))

kernel = build / "dpvo" / "altcorr" / "correlation_kernel.cu"
kernel.write_text(kernel.read_text().replace("#include <THC/THCAtomics.cuh>", "#include <ATen/cuda/Atomic.cuh>"))
modernize_torch_sources(build / "dpvo")

subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", "--no-cache", str(build)],
    env=cuda_build_env(prefix), check=True,
)
shutil.rmtree(build.parent)  # everything is in site-packages now
