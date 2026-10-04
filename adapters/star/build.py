"""Compile official pointnet2 for actual GPUs, in a throwaway source copy.
The upstream setup hard-codes obsolete architectures removed by CUDA 12.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from lab2shot_worker.build import cuda_build_env

repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
env = cuda_build_env(prefix)
build_root = Path(os.environ["LAB2SHOT_EXT_ROOT"]) / "build"
build_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=build_root) as temporary:
    source = Path(temporary) / "pointnet2_ops_lib"
    shutil.copytree(repo / "submodules/pointnet2_ops_lib", source)
    setup = source / "setup.py"
    text = setup.read_text()
    text, changed = re.subn(r'os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*',
                           'os.environ["TORCH_CUDA_ARCH_LIST"] = os.environ["LAB2SHOT_POINTNET_ARCHS"]', text)
    if changed != 1:
        raise RuntimeError("Pinned pointnet2 setup architecture override changed")
    setup.write_text(text)
    env["LAB2SHOT_POINTNET_ARCHS"] = env["TORCH_CUDA_ARCH_LIST"]
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", str(source)],
                   env=env, check=True)
