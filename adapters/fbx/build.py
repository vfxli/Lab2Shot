"""Compile fbxio (fbxio.cpp, a pybind11 module over the Autodesk FBX SDK) into this environment.

Run by `lab2shot ext install fbx` (EnvSpec.build) with the extension's own Python, after the conda-forge packages
(Python 3.12, numpy, libxml2 2.x) are in place. Only the standard library and the worker SDK's build helpers
(lab2shot_worker.build, installed before this step) are used here.

    LAB2SHOT_EXT_REPO         third_party/fbx/repo: pybind11 at the pinned commit (its headers)
    LAB2SHOT_EXT_PREFIX       third_party/fbx/.venv: the conda prefix; the module goes into its site-packages
    LAB2SHOT_MANUAL_FBX_SDK   third_party/_fbx_sdk/<version>: the SDK the user installed (accepting its licence)
    CXX                       the compiler from config [build] cxx (e.g. g++-14); default c++

The SDK is linked in statically (lib/release/libfbxsdk.a): the module needs only the environment's libxml2 and zlib
at run time, found next to it (rpath $ORIGIN/../..). One translation unit: MAX_JOBS has nothing to share out.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
import time
from pathlib import Path

from lab2shot_worker import fail
from lab2shot_worker.build import run

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ["LAB2SHOT_EXT_REPO"])
PREFIX = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
SDK = Path(os.environ["LAB2SHOT_MANUAL_FBX_SDK"])


def compiler() -> str:
    name = os.environ.get("CXX") or "c++"
    path = shutil.which(name)
    if not path:
        fail("E-FBX-NOCOMPILER", name=name)
    return path


def main() -> None:
    start = time.time()
    archive = SDK / "lib" / "release" / "libfbxsdk.a"
    if not archive.is_file() or not (SDK / "include" / "fbxsdk.h").is_file():
        fail("E-FBX-SDKINCOMPLETE", path=str(SDK))
    if not (REPO / "include" / "pybind11" / "pybind11.h").is_file():
        fail("E-FBX-NOPYBIND", path=str(REPO))
    site = Path(sysconfig.get_paths()["purelib"])
    out = site / f"fbxio{sysconfig.get_config_var('EXT_SUFFIX')}"
    tmp = out.with_name(out.name + ".tmp")
    run([compiler(), "-O2", "-shared", "-fPIC", "-std=c++17", "-fvisibility=hidden", "-w",
         f"-I{sysconfig.get_paths()['include']}", f"-I{REPO / 'include'}", f"-I{SDK / 'include'}",
         HERE / "fbxio.cpp", archive, f"-L{PREFIX / 'lib'}", "-lxml2", "-lz", "-lpthread", "-ldl",
         "-Wl,-rpath,$ORIGIN/../..", "-o", tmp])
    tmp.replace(out)

    # Smoke test in a fresh interpreter: the module loads, writes a file and reads it back.
    run([sys.executable, "-c", (
        "import os, tempfile, numpy as np, fbxio\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    p = os.path.join(d, 't.fbx')\n"
        "    s = fbxio.create(24.0, 1, 2)\n"
        "    n = s.add_node(-1, 'box')\n"
        "    s.set_transform(n, np.array([1.0]), np.eye(4)[None])\n"
        "    s.save(p)\n"
        "    back = fbxio.open(p)\n"
        "    assert [x['name'] for x in back.nodes()] == ['box'], back.nodes()\n"
        "    assert back.info()['up'] == 'y' and back.info()['unit_cm'] == 1.0, back.info()\n"
        "print('fbxio OK: FBX SDK', fbxio.sdk_version())\n"
    )])
    print(f"built in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
