"""Compile upstream kernels in a separate build copy, preserving the checkout."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env

repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
build = root / "build" / ("wildpose-" + str(os.getpid()))
shutil.copytree(repo, build, ignore=shutil.ignore_patterns(".git", "__pycache__"))
# The installer hands this script a CPU-only environment; the extension's own
# pip CUDA toolkit (EnvSpec.cuda_toolkit) is what compiles lietorch and the
# droid backends, exactly as tapip3d's build does.
env = cuda_build_env(prefix)
for package in (build / "thirdparty/lietorch", build):
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", str(package)],
                   cwd=package, env=env, check=True)
