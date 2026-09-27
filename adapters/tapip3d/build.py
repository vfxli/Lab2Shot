"""EnvSpec.build for TAPIP3D: compile pointops2's CUDA ops (the k-nearest-neighbour query TAPIP3D's 4D correlation
uses) from the pinned repository, with the environment's pip CUDA 13.2 toolkit. Runs with the environment's Python.

pointops2 is copied to <ext root>/build/pointops2 (repo/ stays untouched) and only its compiled module
`pointops2_cuda` is kept in site-packages: the worker imports pointops2's Python functions from the repository, as
TAPIP3D does (third_party.pointops2.functions.pointops). Old torch 1.x idioms in the copy are fixed for torch 2.x
(lab2shot_worker.build.modernize_torch_sources)."""

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

build = ext_root / "build" / "pointops2"
if build.exists():
    shutil.rmtree(build)
shutil.copytree(repo / "third_party" / "pointops2", build)
modernize_torch_sources(build / "src")

subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", "--no-cache", str(build)],
    env=cuda_build_env(prefix), check=True,
)
shutil.rmtree(build.parent)  # everything is in site-packages now
subprocess.run([sys.executable, "-c", "import torch, pointops2_cuda"], check=True)
print("pointops2 编译完成")
