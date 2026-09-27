"""EnvSpec.build for Kimodo: MotionCorrection, the C++ post-processing Kimodo's generation calls to clean up foot
skating and pull the motion onto its constraints (kimodo/postprocess.py imports motion_correction), compiled from the
pinned repository into the environment. CMake finds pybind11 through the pip package's CMake config; its CMakeLists
fetches Eigen 3.4.0 itself. Runs with the environment's Python; the pinned repository is not modified (setuptools
builds in a copy of MotionCorrection)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pybind11

repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
env = os.environ.copy()
env["CMAKE_PREFIX_PATH"] = pybind11.get_cmake_dir()
env["CMAKE_BUILD_PARALLEL_LEVEL"] = env.get("MAX_JOBS", "4")
with tempfile.TemporaryDirectory() as tmp:
    source = Path(tmp) / "MotionCorrection"
    shutil.copytree(repo / "MotionCorrection", source)
    subprocess.check_call(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", str(source)],
                          env=env)
subprocess.check_call([sys.executable, "-c", "from motion_correction import motion_postprocess"], env=env)
print("MotionCorrection 编译完成")
