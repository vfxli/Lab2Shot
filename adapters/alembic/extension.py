"""Alembic (.abc), a format module: read cameras, models, point clouds and curves from DCC files; write models (point
caches), cameras, point clouds and curves.

The official Alembic library with its Python bindings (PyAlembic) is not on PyPI
(the PyPI / conda-forge package called "alembic" is an unrelated database tool)
and conda-forge's Imath has no Python bindings, so both are compiled from their
pinned sources at install time (build.py) into this extension's own conda-forge
environment (Python 3.12 + Boost.Python). Nodes only: no `lab2shot run alembic` job.
"""

from __future__ import annotations

from lab2shot.sdk import BASIC, EnvSpec, Extension, GitSource, LicenseInfo

ALEMBIC_URL = "https://github.com/alembic/alembic.git"
ALEMBIC_COMMIT = "6f59e4d0c9012c67e242da510b31d96504db8e01"  # tag 1.8.12


class Alembic(Extension):
    name = "alembic"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Alembic"
    summary = "开放的图形交换框架：把复杂的动画场景烘成一份与软件无关、非过程化的几何结果"
    format_module = True  # reads and writes scene formats
    homepage = "https://github.com/alembic/alembic"
    source = GitSource(url=ALEMBIC_URL, commit=ALEMBIC_COMMIT)
    # Imath (BSD-3-Clause, tag v3.2.3): PyImath is built beside PyAlembic. Checked out by the installer like the repo
    # (mirror, retry, pinned commit) and handed to build.py as LAB2SHOT_EXTRA_IMATH — the script never fetches on its own
    extra_sources = {"imath": GitSource(url="https://github.com/AcademySoftwareFoundation/Imath.git", commit="5f27ba266d3ea1565e912570c30b5eafc89959f1")}
    license = LicenseInfo(
        tag=BASIC,
        name="BSD-3-Clause",
        url="https://github.com/alembic/alembic/blob/master/LICENSE.txt",
        summary=(
            "BSD-3，可商用、修改和再分发（需保留版权声明）；依赖的 Imath 同为 BSD-3，Boost 为 Boost 许可证，均可商用。"
            "安装时从源码编译，需要本机 C++ 编译器（config/local.toml 的 build 段 cc / cxx）"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # C++ dependencies from conda-forge (pinned); numpy is also PyImath's build dependency.
        conda=(
            "python=3.12.14",
            "libboost-python-devel=1.92.0",
            "numpy=2.5.3",
            "cmake=4.4.3",
            "ninja=1.13.2",
        ),
        # Imath + PyImath, then Alembic + PyAlembic, compiled into the environment.
        build="build.py",
    )


EXTENSION = Alembic()
