"""Compile Imath (with PyImath) and Alembic (with PyAlembic) into this environment.

Run by `lab2shot ext install alembic` (EnvSpec.build) with the extension's own
Python, after the conda-forge packages (Python 3.12, Boost.Python, numpy, cmake,
ninja) are in place. Only the standard library and the worker SDK's build helpers
(lab2shot_worker.build, installed before this step) are used here.

    LAB2SHOT_EXT_ROOT    third_party/alembic
    LAB2SHOT_EXT_REPO    third_party/alembic/repo   Alembic at the pinned commit (built out of tree, untouched)
    LAB2SHOT_EXT_PREFIX  third_party/alembic/.venv  conda prefix: everything is installed into it
    CC / CXX             compilers from config [build] (e.g. gcc-14); default cc / c++
    MAX_JOBS             parallel compile jobs

Result in the prefix: libImath / libPyImath / libAlembic, the Python modules
`imath`, `imathnumpy` and `alembic` (site-packages), and Alembic's command-line
tools (abcls, abctree, abcecho, abcechobounds, abcdiff, abcstitcher; abcconvert
needs HDF5 and is skipped).
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from lab2shot_worker import fail
from lab2shot_worker.build import run

IMATH = Path(os.environ["LAB2SHOT_EXTRA_IMATH"])  # checked out by the installer (extension.py extra_sources): tag v3.2.3

ROOT = Path(os.environ["LAB2SHOT_EXT_ROOT"])
REPO = Path(os.environ["LAB2SHOT_EXT_REPO"])
PREFIX = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
SRC = ROOT / "src"
BUILD = ROOT / "build"
PY = f"python{sys.version_info.major}.{sys.version_info.minor}"
# Absolute: CMake turns a relative PATH cache value into one below the current folder.
SITE = str(PREFIX / "lib" / PY / "site-packages")


def compiler(var: str, default: str) -> str:
    name = os.environ.get(var) or default
    path = shutil.which(name)
    if not path:
        fail("E-ALEMBIC-NOCOMPILER", name=name, setting=var.lower())
    return path


def cmake_project(name: str, src: Path, options: dict[str, str]) -> None:
    build = BUILD / name
    if build.exists():  # stale caches can point at a previous prefix
        shutil.rmtree(build)
    common = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_C_COMPILER": compiler("CC", "cc"),
        "CMAKE_CXX_COMPILER": compiler("CXX", "c++"),
        "CMAKE_INSTALL_PREFIX": str(PREFIX),
        "CMAKE_INSTALL_LIBDIR": "lib",
        "CMAKE_PREFIX_PATH": str(PREFIX),
        # Only this environment: the machine may have its own Boost/Imath in /usr.
        "CMAKE_FIND_USE_CMAKE_SYSTEM_PATH": "OFF",
        "CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH": "OFF",
        "Boost_ROOT": str(PREFIX),
        "Boost_NO_SYSTEM_PATHS": "ON",
        "Python_EXECUTABLE": sys.executable,
        "Python3_EXECUTABLE": sys.executable,
        "Python_FIND_STRATEGY": "LOCATION",
        "Python3_FIND_STRATEGY": "LOCATION",
        # Libraries in lib/, modules in lib/pythonX.Y/site-packages, tools in bin/:
        # all find each other and the conda libraries (Boost, libpython, libstdc++).
        "CMAKE_INSTALL_RPATH": "$ORIGIN/../lib;$ORIGIN/../..",
    }
    defs = [f"-D{k}={v}" for k, v in {**common, **options}.items()]
    run(["cmake", "-S", src, "-B", build, "-G", "Ninja", "--no-warn-unused-cli", *defs])
    run(["cmake", "--build", build, "--parallel", os.environ.get("MAX_JOBS", str(os.cpu_count() or 4))])
    run(["cmake", "--install", build, "--strip"])
    shutil.rmtree(build)


def main() -> None:
    start = time.time()
    imath_src = IMATH  # built out of source (cmake_project's build folder): the checkout itself is never written
    print(f"== Imath {IMATH.name} + PyImath (Boost.Python, {PY})", flush=True)
    cmake_project("imath", imath_src, {
        "PYTHON": "ON",
        "PYBIND11": "OFF",
        "BUILD_TESTING": "OFF",
        "PYTHON_INSTALL_DIR": SITE,
    })

    print(f"== Alembic {REPO} + PyAlembic", flush=True)
    cmake_project("alembic", REPO, {
        "USE_PYALEMBIC": "ON",
        "USE_BINARIES": "ON",
        "USE_TESTS": "OFF",
        "USE_EXAMPLES": "OFF",
        "USE_HDF5": "OFF",  # Ogawa only (the default format of Maya / Houdini since 2013)
        "USE_MAYA": "OFF",
        "USE_PRMAN": "OFF",
        "USE_ARNOLD": "OFF",
        "ALEMBIC_SHARED_LIBS": "ON",
        "ALEMBIC_DEBUG_WARNINGS_AS_ERRORS": "OFF",
        "ALEMBIC_PYTHON_INSTALL_DIR": SITE,
    })
    shutil.rmtree(BUILD, ignore_errors=True)

    # Smoke test in a fresh interpreter: the modules load and can write + read an archive.
    run([sys.executable, "-c", (
        "import imath, imathnumpy, alembic, tempfile, os\n"
        "from alembic import Abc, AbcGeom\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    p = os.path.join(d, 't.abc')\n"
        "    AbcGeom.OXform(Abc.OArchive(p).getTop(), 'x')  # temporaries: archive closed here\n"
        "    assert Abc.IArchive(p).getTop().getChild(0).getName() == 'x'\n"
        "print('PyAlembic OK:', Abc.GetLibraryVersion())\n"
    )])
    tools = sorted(p.name for p in (PREFIX / "bin").glob("abc*"))
    print(f"Alembic tools: {' '.join(tools)}")
    print(f"built in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
