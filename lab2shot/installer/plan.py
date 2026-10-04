"""An install's steps, derived from what the extension declares (adapters/<name>/extension.py: source, env, weights,
post_install) — never written per extension.

    repo       the pinned checkout (and submodules, extra pinned code)
    env        the environment's interpreter (uv venv, or a conda prefix)
    packages   torch, requirements, the CUDA toolkit wheels, compiled packages, the worker SDK
    build      the build script (EnvSpec.build), only when there is one
    weights    model files: downloaded, checked against their sha256, shared by hash between extensions
    post       Extension.post_install, only when the extension overrides it (the one registered hook)
    place      what upstream reads from fixed paths in its checkout (EnvSpec.places), put there by the build script
    selfcheck  imports and a tiny CPU call in the new environment, required weights there, GPU architectures recorded
    switch     the new environment goes live (waiting for running jobs of this extension); the old one stays

Each step has a fingerprint of its inputs; a finished step whose fingerprint did not change is skipped, so running an
install again goes on from the step that failed. An environment is finished for a spec when its install record says so
(`built_for`: the one judgement, the installer's and the card's, extensions/status.py). Where it builds (`target`): the
live environment when it is finished for this spec, else a new one beside it. An environment's folder is named by the
architectures it is built for (Extension.env_archs -> lab2shot_shared.gpu_arch.env_name: .venv-ada-blackwell), with the
next free number when one of that name is there already (.venv-ada-blackwell-2); what it was built from is its install
record's fingerprint, never its name.
"""

from __future__ import annotations



import json
from collections.abc import Mapping
from dataclasses import dataclass

from ..extensions.spec import ACTIVE_FILE, Extension, ExtensionPaths, active_env
# what identifies a build and what an environment was finished for are read by the extensions' status too: they are the
# extensions' (extensions/build_state.py, below the installer); the installer writes the state they read
from ..extensions.build_state import (SAID_CALLS, SAID_MODULES, _content, _hash, _Unsaid, built_for,  # noqa: F401
                                      env_fingerprint, read_state, repo_fingerprint)

class Words(Mapping):
    """{id: its word's key}, read as {id: the word in the language now} (a step's, a check's name)."""

    def __init__(self, keys: dict[str, str]) -> None:
        self._keys = keys

    def __getitem__(self, id: str) -> str:
        from .. import i18n

        return i18n.t(self._keys[id])

    def __iter__(self):
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def key(self, id: str) -> str:
        """The key of `id`'s word (kept as a word and said in whoever's language reads it: messages.word_of)."""
        return self._keys[id]


# UI vocabulary: each step's short name on the progress bar (one word, never a sentence)
LABELS = Words({"repo": "install.step.repo", "env": "install.step.env", "packages": "install.step.packages",
                "build": "install.step.build", "weights": "install.step.weights", "post": "install.step.post",
                "place": "install.step.place",
                "selfcheck": "install.step.selfcheck", "switch": "install.step.switch"})
ENV_STEPS = ("env", "packages", "build")  # built into the target environment: one fingerprint for the three
ALWAYS = ("weights", "post", "place", "selfcheck", "switch")  # cheap to run again (each skips what is already right)


@dataclass(frozen=True)
class Step:
    id: str
    fingerprint: str

    @property
    def label(self) -> str:
        return LABELS[self.id]

    @property
    def always(self) -> bool:
        return self.id in ALWAYS


def has_post_install(ext: Extension) -> bool:
    return type(ext).post_install is not Extension.post_install


def steps(ext: Extension) -> list[Step]:
    """Every step installing `ext` runs, in order: an extension running in another's environment (runs_in) installs
    that one first (its steps, each skipped when done), then its own (own_steps)."""
    host = ext.host if ext.runs_in else None
    return [*(own_steps(host) if host is not None else ()), *own_steps(ext)]


def own_steps(ext: Extension) -> list[Step]:
    """The steps of `ext`'s own install record: all of them for an extension with its own environment; only its
    weights (and its post_install) for one running in another's (runs_in), whose environment that one builds."""
    env_fp, repo_fp = env_fingerprint(ext), repo_fingerprint(ext)
    weights_fp = _hash([[w.key, w.kind, w.source, w.dest, w.sha256, w.revision, w.files] for w in ext.weights])
    if ext.runs_in:
        return [Step("weights", weights_fp), *([Step("post", _hash([weights_fp, repo_fp]))] if has_post_install(ext) else [])]
    out = [Step("repo", repo_fp), Step("env", env_fp), Step("packages", env_fp)]
    if ext.env.build:
        out.append(Step("build", env_fp))
    out.append(Step("weights", weights_fp))
    if has_post_install(ext):
        out.append(Step("post", _hash([weights_fp, repo_fp])))
    if ext.env.places:
        out.append(Step("place", _hash([weights_fp, repo_fp, list(ext.env.places)])))
    out += [Step("selfcheck", env_fp), Step("switch", _hash([env_fp, repo_fp]))]
    return out


# ------------------------------------------------------------------ where it builds


def pointer(ext: Extension) -> dict:
    return active_env(ext.paths.root)


def write_pointer(ext: Extension, data: dict) -> None:
    root = ext.paths.root
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (ACTIVE_FILE + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(root / ACTIVE_FILE)


def slot(paths: ExtensionPaths) -> dict:
    return {"env": paths.env, "repo": paths.repo_dir}


def paths_of(ext: Extension, entry: dict | None) -> ExtensionPaths | None:
    if not entry:
        return None
    return ExtensionPaths(ext.paths.root, entry.get("env", ""), entry.get("repo", "repo"))


def target(ext: Extension, rebuild: bool = False) -> ExtensionPaths:
    """Where this install builds: the live environment when it is finished for this spec (nothing to build beside it);
    the one an earlier install began for this same spec (it goes on there); else, or when the person asked to `rebuild`,
    a new folder beside the live one. One running in another's environment (runs_in) builds none: its own folder."""
    live = ext.paths
    if ext.runs_in:
        return live
    env_fp = env_fingerprint(ext)
    repo_dir = live.repo_dir if read_state(live).get("repo", {}).get("commit") in (None, ext.source.commit) else f"repo-{ext.source.commit[:12]}"
    if not rebuild and built_for(live) == env_fp:
        return ExtensionPaths(live.root, live.env, repo_dir)
    building = paths_of(ext, pointer(ext).get("building"))
    if building is not None and not rebuild and read_state(building).get("plan") == env_fp:
        return building
    return ExtensionPaths(live.root, new_env(ext), repo_dir)


def new_env(ext: Extension) -> str:
    """The name of a new environment of `ext`: env_name of its architectures when no environment or install record of
    that name is there, else that name with the next free number (a forced rebuild, a changed spec beside the live one)."""
    from lab2shot_shared.gpu_arch import env_name

    base, root = env_name(ext.env_archs), ext.paths.root
    taken = lambda name: ExtensionPaths(root, name).venv.exists() or ExtensionPaths(root, name).state_file.exists()  # noqa: E731
    name, n = base, 1
    while taken(name):
        n += 1
        name = "-".join(part for part in (base, str(n)) if part)
    return name
