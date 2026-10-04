"""Build PyTorch3D's CPU ops for its Meshes import; neural inference uses torch CUDA.
No PyTorch3D renderer or training metrics are invoked by the adapter.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""
env["PYTORCH3D_FORCE_NO_CUDA"] = "1"
build_root = Path(os.environ["LAB2SHOT_EXT_ROOT"]) / "build"
build_root.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=build_root) as temporary:
    source = Path(temporary) / "pytorch3d"
    shutil.copytree(Path(os.environ["LAB2SHOT_EXTRA_PYTORCH3D"]), source, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", str(source)],
                   env=env, check=True)
