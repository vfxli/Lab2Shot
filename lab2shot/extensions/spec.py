"""What an adapter declares about its third-party package."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, ClassVar, Literal

from ..errors import MessageError

if TYPE_CHECKING:
    from .manual import ManualItem


class InstallError(MessageError, RuntimeError):
    """An install step failed, with what the user reads (Extension.post_install raises it too)."""


@dataclass(frozen=True)
class GitSource:
    url: str
    commit: str  # full SHA: installs are reproducible


@dataclass(frozen=True)
class LicenseInfo:
    name: str
    url: str
    summary: str  # one plain-language line shown before install
    tag: str  # its licence class (nodes/tags.py LICENCES): basic, commercial, noncommercial or research


# CUDA compiler from pip for EnvSpec.cuda_toolkit, CUDA 13.2: its headers are the first that compile against
# glibc >= 2.43 (C23 rsqrt/rsqrtf). CCCL requires compiler == runtime headers, so cudart is lifted to 13.2 too
# (same major as torch's cu130 runtime: compatible).
CUDA_13_2_TOOLKIT = (
    "nvidia-cuda-nvcc==13.2.86",
    "nvidia-cuda-crt==13.2.86",
    "nvidia-nvvm==13.2.86",
    "nvidia-cuda-cccl==13.2.86",
    "nvidia-cuda-runtime==13.2.75",
)


@dataclass(frozen=True)
class EnvSpec:
    python: str  # e.g. "3.11"
    torch: tuple[str, ...] = ()  # e.g. ("torch==2.8.0", "torchvision==0.23.0"); () = no torch
    torch_backend: str = ""  # uv --torch-backend, e.g. "cu128"; "" = none
    # pip requirements, relative to the adapter folder (skipped if the file doesn't exist)
    requirements: str = "requirements.txt"
    # Packages compiled against the installed torch: installed afterwards with
    # --no-build-isolation --no-deps and the configured toolchain.
    compiled: tuple[str, ...] = ()
    # False: build C++ ops only (no CUDA toolkit needed, no kernels compiled).
    compiled_cuda: bool = True
    # CUDA compiler from pip wheels inside this environment (nvcc, crt, nvvm,
    # cccl and a matching cudart), used to build `compiled` instead of the
    # machine toolkit in [build] cuda_home. Installed as overrides, so they may
    # lift torch's pinned CUDA runtime to a newer minor of the same major.
    cuda_toolkit: tuple[str, ...] = ()
    # conda-forge packages (C++ libraries that have no wheels, e.g. Boost.Python),
    # e.g. ("libboost-python-devel=1.92.0", "cmake=4.4.3"). When set, the
    # environment is a conda prefix made by a pinned standalone micromamba
    # (third_party/_tools/, never the user's own conda) instead of a uv venv;
    # it still lives at .venv, so paths.python is unchanged. pip packages
    # (torch, requirements, worker SDK) are installed into it with uv afterwards.
    conda: tuple[str, ...] = ()
    # Build script, relative to the adapter folder, run last with the
    # environment's own Python, e.g. to compile C++ libraries from source into
    # the environment. It gets LAB2SHOT_EXT_ROOT / _REPO / _PREFIX, the configured
    # CC / CXX and MAX_JOBS, and the environment's bin/ first on PATH. Its text is
    # part of the environment fingerprint: editing it rebuilds on the next install.
    build: str = ""
    # Adapter files the build script compiles (e.g. a C++ binding), relative to the adapter folder: their text joins
    # the environment fingerprint like the build script's, so editing one rebuilds on the next install. A build
    # script also gets LAB2SHOT_MANUAL_<KEY> = the installed folder of every item the extension declares with
    # manual_weight(key) (an SDK the user downloaded by hand), and does not run until each one is ready.
    build_files: tuple[str, ...] = ()
    # What the build script puts into the checkout for upstream code that reads hard-coded relative paths
    # (a weight the repo expects at `ext/<x>/pretrained/<y>.pth`, an archive unpacked into `ext/<x>/data`):
    # paths relative to ExtensionPaths.root, one line each. Two uses, both read-only for the core:
    # `lab2shot ext adopt` checks they are there (an install record that says ready while a file upstream opens on
    # its ninth step is missing is a false green light), and the message names `lab2shot ext place <name>`, which
    # runs this build script with the single argument "place" so it puts them again, without downloading,
    # compiling, or touching the environment. A build script with `places` must honour that argument.
    places: tuple[str, ...] = ()
    # torch >= 2.6 loads checkpoints weights-only by default. True: upstream code in this environment loads pickled
    # checkpoints (each pinned and sha256-checked by the installer), so its workers run with the old behaviour.
    pickled_checkpoints: bool = False
    # Modules the installer's self-check imports in this environment (on the CPU, no footage), besides the worker SDK
    # and the pinned torch / torchvision it always tries: a package that installed but does not import (a missing
    # shared library, a wrong CUDA build) keeps the card grey instead of failing on the artist's first cook.
    imports: tuple[str, ...] = ()
    # Hash-pinned requirements (uv pip compile --generate-hashes), relative to the adapter folder. Only with one does
    # the installer ever take packages from a PyPI mirror (installed with --require-hashes: every wheel checked against
    # the official hash); without one, packages come from the official index only (lab2shot/installer/sources.py).
    lock: str = "requirements.lock"


@dataclass(frozen=True)
class Weight:
    key: str
    kind: Literal["hf", "url", "zip", "manual"]
    # Hugging Face repo id, URL, or for "manual" the key of a hand-downloaded item (lab2shot.extensions.manual.items():
    # a body model via body_model_weight(key), an extension's own item via manual_weight(item))
    source: str
    dest: str  # relative to the weights folder (a folder for hf/zip, a file for url); "" for manual
    gated: bool = False  # needs an approved access request on Hugging Face
    note: str = ""
    files: tuple[str, ...] = ()  # hf: fetch just these files/patterns
    sha256: str = ""  # url: of the file, zip: of the archive. Checked by the installer; a bad download is deleted
    revision: str = ""  # hf: the commit to fetch (must be pinned: a branch can change at any time)
    option: tuple[str, object] | None = None  # needed only when node parameter option[0] == option[1] (e.g. a model choice)
    notice: str = ""  # this weight's licence requires crediting it once it is actually there ("Built on NVIDIA Cosmos"):
    # help.py's 版权与许可声明 shows it, generically, for whichever weight of whichever extension declares one

    def __post_init__(self) -> None:
        # an hf snapshot without a revision follows main: once the repository author pushes new files, the next install
        # gets something other than the self-checked version, with no sha256 to verify (snapshots are pinned by
        # revision, not hash). This is refused at declaration rather than silently following main at install time
        if self.kind == "hf" and not self.revision:
            raise ValueError(f"weight {self.key!r}: a Hugging Face snapshot ({self.source}) must pin a revision (the commit it was checked with)")

    @property
    def page(self) -> str:
        """Where a person looks at this weight (and requests access when it is gated): the Hugging Face repository
        page of its files, else its URL (a hand download: its page is in lab2shot.extensions.manual)."""
        if self.kind == "hf":
            return f"https://huggingface.co/{self.source}"
        repo, sep, _ = self.source.partition("/resolve/")
        return repo if sep and repo.startswith("https://huggingface.co/") else self.source


def hf_file(repo: str, revision: str, filename: str, *, key: str, dest: str = "", note: str = "", sha256: str = "",
            **kw) -> Weight:
    """One file of a Hugging Face repository (a Space: repo "spaces/<owner>/<name>") at a pinned commit, fetched
    like any URL (large files in parallel parts; checked against `sha256`, the file's LFS id, when given; small
    git files such as config.json are pinned by the commit alone). `dest` defaults to <repo name>/<file>
    (lab2shot_worker.hf_dest, where the worker finds it)."""
    from lab2shot_worker import hf_dest

    return Weight(key=key, kind="url", source=f"https://huggingface.co/{repo}/resolve/{revision}/{filename}",
                  dest=dest or hf_dest(repo, filename), note=note, sha256=sha256, **kw)


def hf_weights(models: dict[str, tuple[str, str, dict[str, str], str]], licence_of) -> tuple[Weight, ...]:
    """The pinned files of a model choice's Hugging Face repository, for a parameter that picks one: `models` is
    {choice: (repo, revision, {file: sha256}, note)}, each file stored as weights/<choice>/<file> (key the same);
    `licence_of(choice)` is said in the weight's note."""
    return tuple(
        hf_file(repo, revision, filename, key=f"{key}/{filename}", dest=f"{key}/{filename}", sha256=sha,
                note=f"{repo}（{licence_of(key)}，{note}）" if filename != "config.json" else f"{repo} 的网络配置")
        for key, (repo, revision, files, note) in models.items()
        for filename, sha in files.items()
    )


ACTIVE_FILE = "active_env.json"  # third_party/<name>/active_env.json: which environment and checkout are live

# What of this process's environment never reaches an extension's Python (a worker, the self-check, a build script, uv
# installing into the extension's environment): the core's own interpreter (VIRTUAL_ENV / CONDA_PREFIX / PYTHONHOME), an
# inherited PYTHONPATH, a startup file, and the machine's pip / uv settings that would point package installs at
# another index or config file. Every package of an extension comes from the official index or a hash-checked mirror
# the settings name (installer/sources.py), never from wherever the shell that started the server happens to point.
CORE_ONLY_ENV = ("VIRTUAL_ENV", "PYTHONHOME", "CONDA_PREFIX", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE",
                 "PYTHONEXECUTABLE", "PYTHONSAFEPATH", "PIP_CONFIG_FILE", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL",
                 "PIP_FIND_LINKS", "PIP_TRUSTED_HOST", "PIP_REQUIRE_VIRTUALENV", "PIP_PYTHON", "UV_CONFIG_FILE",
                 "UV_INDEX", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS",
                 "UV_INSECURE_HOST", "UV_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_PROJECT", "UV_NO_CONFIG",
                 "UV_SYSTEM_PYTHON", "UV_TORCH_BACKEND", "UV_INDEX_STRATEGY", "UV_KEYRING_PROVIDER")


def clean_environ() -> dict[str, str]:
    """This process's environment without what is the core's alone (CORE_ONLY_ENV), and with the user site-packages of
    whoever runs the server kept out (PYTHONNOUSERSITE: a ~/.local package would shadow the extension's own). The one
    base every environment an extension's Python runs in is built on (run_env, installer/envbuild.py build_env)."""
    import os

    env = {k: v for k, v in os.environ.items() if k not in CORE_ONLY_ENV}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def active_env(root: Path) -> dict:
    """The installer's record of which side-by-side environment is live ({"current": {"env", "repo"}, "previous":
    ..., "building": ...}, lab2shot/installer/run.py writes it); {} before the installer switched one in."""
    import json

    try:
        return json.loads((root / ACTIVE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@dataclass(frozen=True)
class ExtensionPaths:
    root: Path
    # "": the live environment, third_party/<name>/.venv. Set: a side-by-side environment third_party/<name>/.venv-<env>,
    # named by the architectures it supports (Extension.env_archs -> "ada-blackwell"), with its own install_state;
    # the installer's pointer (active_env.json below) names the one it switched in, the one before stays for rollback.
    env: str = ""
    repo_dir: str = "repo"  # the checkout's folder: a new pinned commit is checked out beside the live one too

    @property
    def repo(self) -> Path:
        return self.root / self.repo_dir

    @property
    def weights(self) -> Path:
        return self.root / "weights"

    @property
    def venv(self) -> Path:
        return self.root / (".venv" if not self.env else f".venv-{self.env}")

    @property
    def python(self) -> Path:
        return self.venv / "bin" / "python"

    @property
    def state_file(self) -> Path:
        return self.root / ("install_state.json" if not self.env else f"install_state-{self.env}.json")


class Extension:
    """Base class for adapters. Subclass in adapters/<name>/extension.py."""

    name: ClassVar[str]
    title: ClassVar[str]
    summary: ClassVar[str]
    homepage: ClassVar[str]
    source: ClassVar[GitSource]
    license: ClassVar[LicenseInfo]
    env: ClassVar[EnvSpec]
    # a format module (FBX, Alembic: reads and writes a scene format, no model of its own): the setup menu lists these
    # apart (cli/setup.py). Where an extension's cards and nodes sit is the administrator's placing (lab2shot/categories.py)
    format_module: ClassVar[bool] = False
    weights: ClassVar[tuple[Weight, ...]] = ()
    # items this extension requires the user to download by hand (lab2shot.extensions.manual.ManualItem: what it is,
    # where to download it, how to recognise and install it) are declared here and referenced from `weights` with
    # manual_weight(item). Body models (smplx / smpl / mano / flame) are shared and not declared here; use
    # body_model_weight(key). The core names no assets
    manual_items: ClassVar[tuple["ManualItem", ...]] = ()
    # the extension's own lens model names used when it solves lenses (AnyCalib's cam_id): its lens intrinsics group
    # (nodes/lens.py LensGroup), by which 「LensDistortion」 lists models; both sides must match. The core provides the
    # COLMAP and 3DE4 groups
    lens_groups: ClassVar[tuple[Any, ...]] = ()
    # Submodules of the repository it needs (paths inside it), checked out at the commits the pinned repo records.
    submodules: ClassVar[tuple[str, ...]] = ()
    # More code from other repositories, each pinned: folder (relative to the extension's folder) -> source.
    extra_sources: ClassVar[dict[str, GitSource]] = {}
    # The pinned repository (or a folder in it, e.g. "src") is imported by the worker, not pip-installed: it goes on
    # the worker's PYTHONPATH. None: the worker imports nothing from the repository by path.
    import_repo: ClassVar[str | None] = None
    # Adapter files the worker imports besides worker.py (part of the worker's code identity, like the worker SDK).
    worker_modules: ClassVar[tuple[str, ...]] = ()
    # Extensions whose adapter code this one builds on (their folders must be there; they need not be installed): its
    # nodes may subclass theirs and its worker may import their worker_modules. Without one of them it counts as not
    # installed, with the reason (extensions/status.py), and its nodes are not loaded; nothing else notices.
    requires: ClassVar[tuple[str, ...]] = ()
    # The version of the adapter API (lab2shot.sdk.SDK_API) the adapter is written for; every adapter sets its own
    # (worker-node adapters: 2). One written for an API other than this core's is left out with the reason.
    sdk: ClassVar[int] = 1
    # The CUDA architectures ("sm_89", "sm_120") this extension's environment is built for, when it is not the plain
    # third_party/<name>/.venv: every path (paths.venv, paths.python, paths.state_file) moves to the side-by-side
    # environment named by their families (third_party/<name>/.venv-ada-blackwell, lab2shot_shared.gpu_arch.env_name),
    # and the scheduler places its jobs only on cards of these architectures (farm/scheduler/compat.py), whatever card
    # model they are. Never a card name: a machine mixes models of one architecture.
    env_archs: ClassVar[tuple[str, ...]] = ()
    # Why it combines with no other project, one line (a format library, a colour-chart detector ...). Only for a
    # project whose nodes connect to no other project's by the type rules. Informational: no code reads it.
    standalone: ClassVar[str] = ""
    # Disk the environment takes, in GB, for the installer's check before it starts (lab2shot/installer/preflight.py);
    # 0: estimated from the spec (a torch or conda environment about 15 GB, a plain one 3 GB). Model files come on top.
    disk_gb: ClassVar[float] = 0.0

    @property
    def adapter_dir(self) -> Path:
        return Path(inspect.getfile(type(self))).resolve().parent

    @property
    def year(self) -> int | None:
        """The paper / release year, written at the top of the project's own docs.md (`year = 2024`); None when absent.
        Read in one place: both the help page badge and the template card badge use it, so the year is not written
        twice."""
        import tomllib

        path = self.adapter_dir / "docs.md"
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
        if not text.startswith("+++"):
            return None
        head, _, _ = text[3:].partition("\n+++")
        try:
            got = tomllib.loads(head).get("year")
        except Exception:
            return None
        return int(got) if isinstance(got, int) or (isinstance(got, str) and got.isdigit()) else None

    @property
    def paths(self) -> ExtensionPaths:
        """The live environment and checkout: the installer's pointer (active_env.json) when it switched one in, else
        the architecture-named side-by-side one (env_name of the archs the environment covers)."""
        from lab2shot_shared.gpu_arch import env_name

        from ..config import THIRD_PARTY_DIR

        root = THIRD_PARTY_DIR / self.name
        live = active_env(root).get("current")
        if live:
            return ExtensionPaths(root, live.get("env", ""), live.get("repo", "repo"))
        return ExtensionPaths(root, env_name(self.env_archs))

    @property
    def worker_script(self) -> Path:
        return self.adapter_dir / "worker.py"

    @property
    def required_extensions(self) -> tuple[Extension, ...]:
        """The extensions it requires, as loaded (one that is not there is left out: then it is not ready)."""
        if not self.requires:
            return ()
        from .registry import extensions

        return tuple(extensions()[r] for r in self.requires if r in extensions())

    def install_state(self) -> dict:
        """What `lab2shot ext install` recorded (install_state.json); {} before the first install."""
        import json

        try:
            return json.loads(self.paths.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def base_env(self) -> dict[str, str]:
        """Every worker's environment: offline (all weights are local), caches inside the extension, growable
        GPU memory blocks (no fragmentation on long shots), no progress bars (workers report progress themselves)."""
        cache = self.paths.root / "cache"
        return {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "DISABLE_TELEMETRY": "1",
            "TORCH_HOME": str(self.paths.weights / "torch"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "MPLCONFIGDIR": str(cache / "matplotlib"),
            # ultralytics (used by the GVHMR and WHAM upstreams for their own YOLOv8x person detection), two settings:
            # (1) it writes settings.yaml into `get_user_config_dir()`, by default the current user's
            #    ~/.config/Ultralytics (ultralytics/utils/__init__.py:697
            #    `USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or ...)`). Extensions must not write into the
            #    user's home, so it points to the extension's own cache, as MPLCONFIGDIR above does.
            # (2) it probes the network on import (same file :687 `ONLINE = is_online()`; is_online at :516 connects
            #    to 1.1.1.1 and 8.8.8.8). Workers have no network (_forbid_downloads replaces socket.connect), so
            #    the probe is refused, 1.1.1.1 is recorded as a refused host, and every unrelated error afterwards
            #    mentions being unable to reach 1.1.1.1, which is misleading. It is therefore disabled.
            #    The value must be the string "True": :516 compares `str(os.getenv("YOLO_OFFLINE","")).lower() != "true"`,
            #    so "1" would silently have no effect.
            "YOLO_CONFIG_DIR": str(cache / "ultralytics"),
            "YOLO_OFFLINE": "True",
            "TOKENIZERS_PARALLELISM": "false",
            "TQDM_DISABLE": "1",
        } | self._allocator_env() | ({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"} if self.env.pickled_checkpoints else {}) \
          | ({"PYTHONPATH": str(self.paths.repo / self.import_repo)} if self.import_repo is not None else {})

    def _allocator_env(self) -> dict[str, str]:
        """Expandable segments, under the name the pinned torch reads (renamed in 2.9; 2.0 refuses the option)."""
        version = self._torch_version()
        if version < (2, 1):
            return {}
        return {"PYTORCH_ALLOC_CONF" if version >= (2, 9) else "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    def _torch_version(self) -> tuple[int, ...]:
        """The pinned torch version (EnvSpec.torch "torch==X.Y.Z"); a high number when torch is not pinned."""
        import re

        pins = [t for t in (self.env.torch or ()) if t.startswith("torch==")]
        m = re.match(r"torch==(\d+)\.(\d+)", pins[0]) if pins else None
        return (int(m.group(1)), int(m.group(2))) if m else (99,)

    def worker_env(self) -> dict[str, str]:
        """What this extension's worker needs on top of base_env() (weight paths, PYTHONPATH, ...)."""
        return {}

    def run_env(self, paths: ExtensionPaths | None = None, *, gpu: str | None = None) -> dict[str, str]:
        """The environment this extension's own Python runs in, wherever it is started: every worker process
        (engine/resident.py worker_environment) and the installer's self-check (installer/run.py). One rule, in one
        place, so what the self-check proves is what a worker then runs.

        `paths`: the environment being started (the installer's side-by-side one); the live one by default.
        `gpu`: CUDA_VISIBLE_DEVICES ("" hides every card); None keeps what this process sees.

        The worker SDK of this checkout comes first on PYTHONPATH, ahead of the copy pip installed into the
        environment (PYTHONPATH is searched before site-packages). That copy is editable and points at whichever
        checkout installed it, so a second checkout (a branch, a snapshot, an integration worktree) runs its own
        SDK whatever version that copy is. Then the folders of the extensions it requires (their worker_modules).
        Nothing of the core environment leaks in (clean_environ: VIRTUAL_ENV, PYTHONHOME, CONDA_PREFIX, an inherited
        PYTHONPATH, the machine's pip / uv indexes, the user site-packages), and the environment's own bin/ is first
        on PATH: upstream code that runs `python` gets this environment's interpreter, not the core's.
        """
        import os

        from ..config import WORKER_SDK_DIR, cpu_budget

        paths = paths or self.paths
        env = clean_environ()
        env["PATH"] = os.pathsep.join(filter(None, [str(paths.venv / "bin"), env.get("PATH", "")]))
        env.update(self.base_env())
        own = self.worker_env()
        env.update(own)
        if paths.repo != self.paths.repo:
            # worker_env hard-codes self.paths.repo (the live checkout; WHAM and SegAnyMo put it on PYTHONPATH), while
            # the self-check runs the new checkout beside it. If PYTHONPATH still pointed at the old one, the self-check
            # would import old code and pass falsely. Every entry pointing at the live checkout is redirected to the
            # checkout being run
            live, mine = str(self.paths.repo), str(paths.repo)
            rebase = lambda v: mine + v[len(live):] if v == live or v.startswith(live + os.sep) else v  # noqa: E731
            for key, value in list(env.items()):
                if key in own or key == "PYTHONPATH":
                    env[key] = os.pathsep.join(rebase(part) for part in value.split(os.pathsep))
        if paths.venv != self.paths.venv:
            # extra sources (dinov3, sam2, ...) were upgraded: the new commit is checked out beside as <folder>.new and
            # swapped in at activation (installer/run.py step_repo). The environment being built must compile and
            # self-check against the new copy, so every entry pointing at <folder> is redirected to <folder>.new
            # (otherwise, when extra sources and environment are upgraded together, the build script would compile the
            # old folder)
            for folder in self.extra_sources:
                live, staged = str(paths.root / folder), str(paths.root / f"{folder}.new")
                if not Path(staged).is_dir():
                    continue
                restage = lambda v, live=live, staged=staged: staged + v[len(live):] if v == live or v.startswith(live + os.sep) else v  # noqa: E731
                for key, value in list(env.items()):
                    if key in own or key == "PYTHONPATH":
                        env[key] = os.pathsep.join(restage(part) for part in value.split(os.pathsep))
        if self.import_repo is not None:  # the environment being started, not the live one
            env["PYTHONPATH"] = str(paths.repo / self.import_repo)
        # some workers take their repo from PYTHONPATH's first entry: the SDK goes after it, never before
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH", ""), str(WORKER_SDK_DIR),
                                                          *(str(r.adapter_dir) for r in self.required_extensions)]))
        if gpu is not None:
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = gpu
        # a cook must not occupy the whole machine (the admin setting 「保留核心数」, config.py worker_cpus): numeric
        # libraries read these limits. Set here so the self-check and workers get the same values; core affinity is the
        # hard limit (engine/resident.py hold_back). LAB2SHOT_CPU_BUDGET is read by projects with their own thread
        # parameter (COLMAP's num_threads defaults to -1 = machine core count, which oversubscribes after pinning)
        budget = str(cpu_budget())
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "LAB2SHOT_CPU_BUDGET"):
            env[name] = budget
        # when the environment ships its own ptxas, tell triton where it is. On Blackwell (sm_120) triton needs a
        # `ptxas-blackwell`, present in pip's triton wheel but not in conda-forge's; without it triton looks for
        # bin/ptxas under CONDA_PREFIX, but workers never activate the environment (they call .venv/bin/python
        # directly) and clean_environ removes CONDA_PREFIX, resulting in "Cannot find ptxas-blackwell". The decision is
        # based on whether the file exists, not on whether the environment is a conda environment.
        ptxas = paths.venv / "bin" / "ptxas"
        if ptxas.exists():
            env.setdefault("TRITON_PTXAS_BLACKWELL_PATH", str(ptxas))
        return env

    def post_install(self, run, paths: ExtensionPaths) -> None:
        """Optional step after env + weights, e.g. extracting rig data: the installer's one registered hook for what
        no declaration says (lab2shot/installer/plan.py runs it only when an extension overrides it).

        `run(args, **kw)` runs a command and raises on failure; `paths`: the environment and checkout being installed
        (not yet live: read the checkout from here, never from self.paths).
        """

