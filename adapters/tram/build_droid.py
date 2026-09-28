"""EnvSpec.build for TRAM: compile TRAM's masked DROID-SLAM (droid_backends) and
lietorch, run with the environment's Python.

The installer has checked out the repo's submodules (DROID-SLAM's lietorch + Eigen,
DEVA) at the commits TRAM records. thirdparty/DROID-SLAM is copied to
<ext root>/build/DROID-SLAM (repo/ stays untouched) and built with the
environment's pip CUDA 13.2 toolkit. Changes to the copy: setup.py's
hard-coded -gencode list (sm_60 ... sm_86; CUDA 13 no longer supports sm_60 /
sm_70) is dropped so torch builds for the chosen compile targets (TORCH_CUDA_ARCH_LIST),
its two setup() calls (droid_backends, lietorch) are installed one after the
other, and tensor.type() in the dispatch macros becomes tensor.scalar_type()
for torch 2.9 (lab2shot_worker.build.modernize_torch_sources).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env, modernize_torch_sources

ext_root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])

build = ext_root / "build" / "DROID-SLAM"
if build.exists():
    shutil.rmtree(build)
source = repo / "thirdparty" / "DROID-SLAM"
unused = {"tartanair_tools", "evaluation_scripts", "misc", "data"}  # top level only (Eigen has its own src/misc)
shutil.copytree(source, build,
                ignore=lambda folder, names: [n for n in names if n == ".git" or (Path(folder) == source and n in unused)])

modernize_torch_sources(build / "src")
modernize_torch_sources(build / "thirdparty" / "lietorch" / "lietorch")
setup_py = build / "setup.py"
text = re.sub(r"\s*'-gencode=arch=compute_\d+,code=sm_\d+',?", "", setup_py.read_text())
head, *setups = text.split("\nsetup(")
if len(setups) != 2:
    raise SystemExit(f"DROID-SLAM setup.py: expected 2 setup() calls, found {len(setups)}")

env = cuda_build_env(prefix)
for body in setups:  # droid_backends, then lietorch
    setup_py.write_text(head + "\nsetup(" + body)
    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", "--no-cache", str(build)],
        env=env, check=True,
    )
shutil.rmtree(build.parent)  # everything is in site-packages now
