"""Before an install starts: everything it needs that the installer cannot fetch by itself, as one checklist. The
install starts only when nothing on it blocks; each row says what is missing and what to do, by message code.

    requires  extensions it builds on
    manual    files the user downloads by hand into downloads/ (SMPL-X, FLAME ...): which, from where, how it is named
    licence   a hand-downloaded file waiting for the user's own click on its licence: only the user accepts it
    gated     Hugging Face weights that need an approved access request: checked with the server's token
    disk      free space for the environment beside the live one
    gpu       whether the environment runs on each authorised card (its declared architectures and the live
              environment's recorded ones, when the spec did not change); unknown before it is built: said, not blocking
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..extensions.spec import Extension
from ..messages import Msg
from . import plan

LABELS = {"requires": "扩展包", "manual": "手动下载", "licence": "许可", "gated": "申请权限", "disk": "硬盘", "gpu": "显卡", "toolchain": "编译工具链"}
BLOCKED, WARNING, NOTICE, OK = "blocked", "warning", "notice", "ok"
HF_TIMEOUT_S = 15


@dataclass(frozen=True)
class Check:
    kind: str
    state: str
    message: Msg

    def json(self) -> dict:
        return {"kind": self.kind, "label": LABELS[self.kind], "state": self.state, "message": self.message.json()}


@dataclass(frozen=True)
class Checklist:
    name: str
    title: str
    checks: tuple[Check, ...]

    @property
    def ready(self) -> bool:
        return not any(c.state == BLOCKED for c in self.checks)

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.state == BLOCKED]

    def json(self) -> dict:
        return {"name": self.name, "title": self.title, "ready": self.ready, "checks": [c.json() for c in self.checks]}


def preflight(ext: Extension) -> Checklist:
    from ..extensions import manual

    manual.check()  # what was just dropped into the inbox is sorted (and installed when it installs by itself) first
    checks = [*requires(ext), *manual_files(ext), *gated(ext), disk(ext), *gpu(ext), *toolchain_check(ext)]
    return Checklist(ext.name, ext.title, tuple(checks))


def toolchain_check(ext: Extension) -> list[Check]:
    """An extension that compiles CUDA ops — through EnvSpec.compiled (compiled_cuda) or in its build script
    (EnvSpec.build: lab2shot_worker.build.cuda_build_env, FORCE_CUDA, CUDA_HOME) — the nvcc that will compile (its own
    from pip, EnvSpec.cuda_toolkit, else the machine toolkit the settings point at) must accept the machine's C
    compiler (installer/toolchain.py). One that compiles nothing has no row here."""
    from . import toolchain

    env = ext.env
    if not compiles_cuda(ext):
        return []
    state, why = toolchain.problem(toolchain.pip_nvcc_release(env.cuda_toolkit))
    return [Check("toolchain", {"ok": OK, "warning": WARNING, "blocked": BLOCKED}[state], why)]


def compiles_cuda(ext: Extension) -> bool:
    """Does installing this extension run nvcc? `compiled` with compiled_cuda, or a build script that sets up a CUDA
    build. The build script must be looked at too: several extensions (gvhmr build_pytorch3d.py, tram build_droid.py,
    wham build_dpvo.py) declare compiled_cuda=False or compiled=() yet compile CUDA kernels through cuda_build_env in
    their script; without this row an nvcc that rejects the machine's GCC only fails halfway through the build. Whether a
    script compiles CUDA is judged from its text (envbuild.build_script_env judges which manual files it reads the same
    way)."""
    env = ext.env
    if env.compiled and env.compiled_cuda:
        return True
    if not env.build:
        return False
    try:
        script = (ext.adapter_dir / env.build).read_text(encoding="utf-8")
    except OSError:
        return False
    return any(sign in script for sign in ("cuda_build_env", "FORCE_CUDA", "CUDA_HOME"))


def requires(ext: Extension) -> list[Check]:
    from ..extensions.status import missing_requirements

    if not ext.requires:
        return []
    lacking = missing_requirements(ext)
    if lacking:
        return [Check("requires", BLOCKED, Msg("B-INSTALL-NEEDSEXT", title=ext.title, needs=lacking))]
    return [Check("requires", OK, Msg("I-INSTALL-NEEDSOK", needs=list(ext.requires)))]


def manual_files(ext: Extension) -> list[Check]:
    from ..extensions import manual

    out = []
    for w in ext.weights:
        if w.kind != "manual" or w.option is not None:
            continue
        item = manual.item_of(ext, w.source)
        state = manual.state(item.key)
        if state == "ready":
            out.append(Check("manual", OK, Msg("I-INSTALL-MANUALOK", title=item.title)))
        elif state == "consent":
            out.append(Check("licence", BLOCKED, Msg("B-INSTALL-CONSENT", title=item.title)))
        else:
            out.append(Check("manual", BLOCKED, Msg("B-INSTALL-MANUAL", title=item.title, page=item.page, download=item.download,
                                                   filename=item.filename, inbox=manual.INBOX.name)))
    return out


def gated(ext: Extension) -> list[Check]:
    rows = plan.read_state(ext.paths).get("weights", {})
    out, seen = [], set()
    for w in ext.weights:
        if not w.gated or w.option is not None or w.kind == "manual" or rows.get(w.key) == "ok" or w.page in seen:
            continue
        seen.add(w.page)
        access = hf_access(w.page)
        if access == "ok":
            out.append(Check("gated", OK, Msg("I-INSTALL-GATEDOK", page=w.page)))
        elif access == "notoken":
            out.append(Check("gated", BLOCKED, Msg("B-INSTALL-HFTOKEN", page=w.page)))
        elif access == "gated":
            out.append(Check("gated", BLOCKED, Msg("B-INSTALL-GATED", page=w.page)))
        else:
            out.append(Check("gated", WARNING, Msg("W-INSTALL-GATEDUNKNOWN", page=w.page)))
    return out


def hf_access(page: str) -> str:
    """"ok" / "gated" (no approved request) / "notoken" (the server has no Hugging Face token) / "unknown" (could
    not ask within HF_TIMEOUT_S: the network) for the repository at `page` (https://huggingface.co/<repo>). Asked
    on the calling thread with the request's own timeout (Hugging Face's auth-check: 200 approved, 401 / 403 / 404 not
    for this token)."""
    import urllib.error
    import urllib.request

    from huggingface_hub import get_token

    from .sources import HF  # the token is sent to huggingface.co only (sources.request): a mirror in HF_ENDPOINT never sees it

    token = get_token()
    if not token:
        return "notoken"
    repo = page.removeprefix("https://huggingface.co/")
    kind = "space" if repo.startswith("spaces/") else "dataset" if repo.startswith("datasets/") else "model"
    repo = repo.split("/", 1)[1] if kind != "model" else repo
    ask = urllib.request.Request(f"{HF}/api/{kind}s/{repo}/auth-check",
                                 headers={"Authorization": f"Bearer {token}", "User-Agent": "lab2shot"})
    try:
        with urllib.request.urlopen(ask, timeout=HF_TIMEOUT_S):
            return "ok"
    except urllib.error.HTTPError as exc:
        return "gated" if exc.code in (401, 403, 404) else "unknown"
    except (OSError, ValueError):
        return "unknown"


def need_gb(ext: Extension) -> float:
    """What the environment beside the live one takes, as declared (Extension.disk_gb) or estimated: a torch or conda
    environment about 15 GB, a plain one 3 GB. Model files come on top (their sizes are not known before download)."""
    if ext.disk_gb:
        return float(ext.disk_gb)
    env = ext.env
    return 15.0 if (env.torch or env.conda or env.compiled) else 3.0


def free_gb(folder: Path) -> float:
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return shutil.disk_usage(folder).free / (1 << 30)


def disk(ext: Extension) -> Check:
    free, need = free_gb(config.THIRD_PARTY_DIR), need_gb(ext)
    if free < need:
        return Check("disk", BLOCKED, Msg("B-INSTALL-DISK", free=free, need=need))
    return Check("disk", OK, Msg("I-INSTALL-DISKOK", free=free, need=need))


def authorised_cards() -> list:
    """The cards the queue may give jobs to (farm/scheduler's inventory; nvidia-smi read once, no thread)."""
    from ..farm import gpus
    from ..farm.scheduler import GpuState

    authed = set(gpus.authorized())
    return [GpuState(g.uuid, g.index, g.name, g.short_name, g.compute_cap, g.memory_mb, g.used_mb, g.utilization,
                     g.temperature, True) for g in gpus.inventory() if g.uuid in authed]


def card_fit(ext: Extension, card, env_fp: str) -> bool | None:
    """Whether this spec runs on this card, by the same rule the queue places jobs with (farm/scheduler/compat.py
    fit: the extension's declared architectures, then the live environment's probed ones) — read only when the live
    environment was built for this very spec; None: not known before it is built. There is no second judgement (a
    probe record consulted before the declaration would let the preflight and the queue disagree)."""
    from ..farm.scheduler import compat

    if not card.compute_cap:
        return None
    if plan.built_for(ext.paths) == env_fp:
        return compat.fit(ext.name, card).ok
    return None


def gpu(ext: Extension) -> list[Check]:
    from ..nodes import node_types

    if not any(t.runtime == ext.name and t.cost.gpu for t in node_types().values()):
        return []
    cards = authorised_cards()
    if not cards:
        return [Check("gpu", NOTICE, Msg("N-INSTALL-NOGPU", project=ext.title))]
    env_fp = plan.env_fingerprint(ext)
    fits = {c.short_name: card_fit(ext, c, env_fp) for c in cards}
    bad = [n for n, ok in fits.items() if ok is False]
    good = [n for n, ok in fits.items() if ok is True]
    unknown = [n for n, ok in fits.items() if ok is None]
    if bad and not good and not unknown:
        return [Check("gpu", BLOCKED, Msg("B-INSTALL-GPUNONE", project=ext.title, gpus=bad))]
    if bad:
        return [Check("gpu", WARNING, Msg("W-INSTALL-GPUSOME", project=ext.title, bad=bad, others=good + unknown))]
    if unknown:
        return [Check("gpu", NOTICE, Msg("N-INSTALL-GPULATER", project=ext.title, gpus=unknown))]
    return [Check("gpu", OK, Msg("I-INSTALL-GPUOK", project=ext.title, gpus=good))]
