"""The env, packages and build steps: an extension's own environment, built from its EnvSpec into the install's target
folder (plan.target: never over the live environment).

    env       uv venv (or a conda prefix from a pinned standalone micromamba, never the user's own conda)
    packages  torch (its CUDA backend), requirements (or the hash lock), the CUDA toolkit wheels, compiled packages
              against that torch, the worker SDK
    build     the extension's build script with the environment's Python (EnvSpec.build)
"""

from __future__ import annotations

import os
import platform
import re
import shutil

from lab2shot_shared.gpu_arch import ARCHS_ENV, cap_of, target_label
from lab2shot_worker.build import MAX_JOBS, compute_caps, pip_cuda_home

from .. import config
from ..extensions.spec import InstallError, clean_environ
from ..messages import Msg
from . import sources
from .sources import NetworkFailure

MICROMAMBA_VERSION = "2.9.0-0"
MICROMAMBA_URL = f"https://github.com/mamba-org/micromamba-releases/releases/download/{MICROMAMBA_VERSION}/micromamba-linux-64"
MICROMAMBA_SHA256 = "366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3"
PACKAGES_GB = 6  # memory an install of packages or a compile wants free on top of the machine's reserve
BUILD_GB = 12


def tools_dir():
    """Shared installer tools (EnvSpec.conda): the pinned micromamba and its package cache."""
    return config.THIRD_PARTY_DIR / "_tools"


def target_archs(ext) -> list[str]:
    """The architectures `ext`'s CUDA code is compiled for: the setting build.archs, narrowed to the ones the extension
    declares when it declares any (Extension.env_archs: a torch without sm_120, say). Never the building machine's
    cards: what an install compiles depends on the repository and the settings only. [] when the two have nothing in
    common (the preflight blocks such an install, E-INSTALL-NOARCH)."""
    archs = list(config.settings().value("build.archs"))
    return [a for a in archs if a in ext.env_archs] if ext.env_archs else archs


def target_caps(ctx) -> list[str]:
    """target_archs as TORCH_CUDA_ARCH_LIST wants them ("8.9", "12.0"), said in the install's log; InstallError when
    there is none (a declaration and a setting with nothing in common)."""
    ext = ctx.ext
    archs = target_archs(ext)
    if not archs:
        raise InstallError(Msg("E-INSTALL-NOARCH", title=ext.title, declared=[target_label(a) for a in ext.env_archs]))
    ctx.sink.say(Msg("I-INSTALL-ARCHS", archs=[target_label(a) for a in archs]))
    return [cap_of(a) for a in archs]


def build_env(cuda: bool = True, cuda_home: str | None = None, caps: list[str] | None = None) -> dict[str, str]:
    """Toolchain for compiled torch extensions (detectron2 & co.). `cuda_home`: a toolkit inside the extension's own
    environment (EnvSpec.cuda_toolkit); default is the machine toolkit from the settings. On the same base as every
    other environment an extension's Python runs in (spec.clean_environ), never os.environ.copy(): that would carry
    the core's VIRTUAL_ENV / PYTHONPATH and the machine's own pip / uv index settings into the compile and the package
    install."""
    s = config.settings()
    env = clean_environ()
    if s["build.cc"]:
        env["CC"] = s["build.cc"]  # torch also passes $CC to nvcc as -ccbin
    if s["build.cxx"]:
        env["CXX"] = s["build.cxx"]
    env.setdefault("MAX_JOBS", MAX_JOBS)
    if not cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""  # torch.cuda.is_available() is what setup.py files check: hide the GPUs
        env["FORCE_CUDA"] = "0"
        return env
    cuda_home = cuda_home or s["build.cuda_home"] or env.get("CUDA_HOME") or "/usr/local/cuda"
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    # `caps`: the compile targets (target_caps); only a caller without an extension falls back to this machine's cards
    if caps := caps if caps is not None else compute_caps():
        env["TORCH_CUDA_ARCH_LIST"] = ";".join(caps)
    env["FORCE_CUDA"] = "1"
    return env


def step_env(ctx) -> None:
    """A fresh interpreter in the target folder: always from scratch (installing over an old environment would keep
    packages the spec no longer lists)."""
    paths = ctx.paths
    if paths.venv == ctx.ext.paths.venv and ctx.ext.paths.python.exists():
        raise InstallError(Msg("E-INSTALL-LIVEENV", title=ctx.ext.title))  # plan.target never builds over the live one
    ctx.mark_building()
    if paths.venv.exists():
        shutil.rmtree(paths.venv)
    if ctx.ext.env.conda:
        _conda_env(ctx)
    else:
        ctx.retry_command(["uv", "venv", "--no-project", "--python", ctx.ext.env.python, paths.venv], "Python")


def step_packages(ctx) -> None:
    ext, paths, env = ctx.ext, ctx.paths, ctx.ext.env
    python = paths.python
    pip = ["uv", "pip", "install", "--python", python]
    toolkit: list = []
    if env.cuda_toolkit:
        overrides = paths.root / f"cuda_toolkit_overrides{'-' + paths.env if paths.env else ''}.txt"
        overrides.write_text("\n".join(env.cuda_toolkit) + "\n")
        toolkit = [*env.cuda_toolkit, "--override", overrides]
    req_file = ext.adapter_dir / env.requirements
    lock = ext.adapter_dir / env.lock
    lock = lock if lock.is_file() else None
    wanted = ["-r", str(lock), "--require-hashes"] if lock else (["-r", str(req_file)] if req_file.is_file() else [])
    backend = ["--torch-backend", env.torch_backend] if env.torch_backend else []
    # `--require-hashes` wants a hash for every package on the command line: torch and the CUDA toolkit are given by
    # name, so in one command with the lock pip refuses at once. With a lock there are two rounds: by name, then the lock
    rounds = [[*env.torch, *toolkit], wanted] if lock and (env.torch or toolkit) else [[*env.torch, *toolkit, *wanted]]
    rounds = [r for r in rounds if r]
    if rounds:
        ctx.wait_memory(PACKAGES_GB)
        indexes = sources.package_indexes(lock, ctx.policy)
        for i, index in enumerate(indexes):
            if not index.official:
                ctx.sink.say(Msg("N-INSTALL-MIRROR", what="Python 包", mirror=index.name))
            try:
                for packages in rounds:
                    ctx.retry_command([*pip, *backend, *(["--index-url", index.url] if index.url else []), *packages], "Python 包")
                ctx.sink.say(Msg("I-INSTALL-SOURCE", what="Python 包", source=index.name))
                break
            except NetworkFailure as exc:
                if i == len(indexes) - 1:
                    if not lock and ctx.policy.pypi:
                        ctx.sink.say(Msg("N-INSTALL-NOLOCK", title=ext.title))
                    raise InstallError(Msg("E-INSTALL-NETWORK", what="Python 包", detail=str(exc)[-200:])) from exc
    if env.compiled:
        ctx.wait_memory(BUILD_GB)
        cuda_home = None
        if env.cuda_toolkit and env.compiled_cuda:
            home = pip_cuda_home(paths.venv)
            if home is None:
                raise InstallError(Msg("E-INSTALL-NONVCC"))
            cuda_home = str(home)
        # no idle timeout (retry_command leaves it off): uv prints nothing to a pipe while it compiles a package from
        # source or fetches a multi-GB wheel, and tens of minutes at nice 19 are normal; with it on the compile would be
        # killed as stalled, started over and reported as a network failure. A failed compile speaks for itself
        # (E-INSTALL-COMMAND); only output that clearly ends in a download or connection error is retried as the network
        caps = target_caps(ctx) if env.compiled_cuda else None
        ctx.retry_command([*pip, "--no-build-isolation", "--no-deps", *env.compiled], "编译的包",
                          env=build_env(env.compiled_cuda, cuda_home, caps))
    ctx.retry_command([*pip, "-e", config.WORKER_SDK_DIR], "Lab2Shot worker SDK")
    if not ext.env.build:  # the environment is finished here (else after its build script)
        ctx.record_env()


def step_build(ctx) -> None:
    ctx.wait_memory(BUILD_GB)
    ctx.run([ctx.paths.python, ctx.ext.adapter_dir / ctx.ext.env.build], cwd=ctx.paths.root, env=build_script_env(ctx))
    ctx.record_env()


def build_script_env(ctx) -> dict[str, str]:
    ext, paths = ctx.ext, ctx.paths
    env = build_env(cuda=False)
    for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PATH"] = f"{paths.venv / 'bin'}:{env.get('PATH', '')}"
    if ext.env.conda:
        env["CONDA_PREFIX"] = str(paths.venv)
    else:
        env.pop("CONDA_PREFIX", None)
    from .preflight import compiles_cuda

    if compiles_cuda(ext):  # the script compiles CUDA through lab2shot_worker.build.cuda_build_env, which reads this
        env[ARCHS_ENV] = ";".join(target_caps(ctx))
    env["LAB2SHOT_EXT_ROOT"] = str(paths.root)
    env["LAB2SHOT_EXT_REPO"] = str(paths.repo)
    env["LAB2SHOT_EXT_PREFIX"] = str(paths.venv)
    # every extra source (Extension.extra_sources: Imath, pytorch3d, dinov2 …) by name: LAB2SHOT_EXTRA_<FOLDER> — the checkout
    # the installer made (mirror, retry, pinned commit), so a build script never fetches from GitHub on its own. A new commit
    # waiting for the switch sits at <folder>.new (installer/run.py step_repo): the environment being built compiles that one
    for folder in ext.extra_sources:
        staged = paths.root / f"{folder}.new"
        env[f"LAB2SHOT_EXTRA_{folder.upper().replace('-', '_')}"] = str(staged if staged.is_dir() else paths.root / folder)
    script = (ext.adapter_dir / ext.env.build).read_text()
    from ..extensions import manual

    for w in ext.weights:
        if w.kind != "manual":
            continue
        var = f"LAB2SHOT_MANUAL_{w.source.upper()}"
        if var not in script:
            continue  # this build script does not read it (a manual item only some node needs)
        path = manual.item_of(ext, w.source).install.path()
        if path is None:
            raise InstallError(Msg("E-INSTALL-NEEDSMANUAL", title=manual.item_of(ext, w.source).title,
                                   reason=manual.explain(w.source, manual.state(w.source))))
        env[var] = str(path)
    return env


def _micromamba(ctx):
    """The pinned standalone micromamba binary (downloaded once, checked against its sha256)."""
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise InstallError(Msg("E-INSTALL-CONDAPLATFORM"))
    exe = tools_dir() / f"micromamba-{MICROMAMBA_VERSION}"
    if exe.exists():
        return exe
    download = exe.with_name(exe.name + ".download")
    sources.fetch_checked(MICROMAMBA_URL, download, MICROMAMBA_SHA256, ctx.sink, ctx.policy)
    download.chmod(0o755)
    download.replace(exe)
    return exe


def _conda_env(ctx) -> None:
    """The environment as a conda prefix at the target's venv folder, from conda-forge only."""
    env, prefix = ctx.ext.env, ctx.paths.venv
    exe = _micromamba(ctx)
    specs = list(env.conda)
    if not any(re.split(r"[=<>!~ ]", s, maxsplit=1)[0] == "python" for s in specs):
        specs.insert(0, f"python={env.python}")
    # everything micromamba keeps (package cache, env registry, ~/.mamba) stays in third_party/_tools
    mamba_env = {k: v for k, v in os.environ.items() if not k.startswith(("CONDA", "MAMBA_", "XDG_"))}
    mamba_env["HOME"] = str(tools_dir() / "home")
    mamba_env["MAMBA_ROOT_PREFIX"] = str(tools_dir() / "mamba")
    mamba_env["CONDA_PKGS_DIRS"] = str(tools_dir() / "mamba" / "pkgs")
    ctx.wait_memory(PACKAGES_GB)
    ctx.retry_command([exe, "create", "--yes", "--no-rc", "--override-channels", "--channel", "conda-forge", "--prefix", prefix, *specs],
                      "conda-forge", env=mamba_env)
    ctx.run([exe, "clean", "--all", "--yes", "--no-rc"], env=mamba_env)  # the prefix holds its own hard-linked copies
