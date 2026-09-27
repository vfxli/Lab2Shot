"""An install's steps, derived from what the extension declares (adapters/<name>/extension.py: source, env, weights,
post_install) — never written per extension.

    repo       the pinned checkout (and submodules, extra pinned code)
    env        the environment's interpreter (uv venv, or a conda prefix)
    packages   torch, requirements, the CUDA toolkit wheels, compiled packages, the worker SDK
    build      the build script (EnvSpec.build), only when there is one
    weights    model files: downloaded, checked against their sha256, shared by hash between extensions
    post       Extension.post_install, only when the extension overrides it (the one registered hook)
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

import ast

from pathlib import Path

import json
from dataclasses import dataclass

from ..extensions.spec import ACTIVE_FILE, Extension, ExtensionPaths, active_env

# UI vocabulary: each step's short name on the progress bar (one word, never a sentence)
LABELS = {"repo": "代码", "env": "环境", "packages": "装包", "build": "编译", "weights": "权重", "post": "后处理",
          "selfcheck": "自检", "switch": "启用"}
ENV_STEPS = ("env", "packages", "build")  # built into the target environment: one fingerprint for the three
ALWAYS = ("weights", "post", "selfcheck", "switch")  # cheap to run again (each skips what is already right)


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


def _hash(parts) -> str:
    from ..io.digest import key

    return key(parts, 16)


def _content(path: Path) -> str:
    """The part of a build input that affects the build, so that editing its comments does not mark an installed
    environment as outdated. Python sources are reduced to their syntax tree without docstrings; other text files
    (requirements lists, shell fragments) lose comment lines, trailing comments and blank lines."""
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body
                    and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return ast.dump(tree)
    lines = (line.split(" #", 1)[0].rstrip() for line in text.splitlines() if not line.lstrip().startswith("#"))
    return "\n".join(line for line in lines if line)


def env_fingerprint(ext: Extension) -> str:
    """What identifies the environment's build (preflight.card_fit compares it with what the live environment was
    built for). Build inputs are compared by content, not by their literal text (`_content`)."""
    from ..extensions import manual

    env = ext.env
    req = _content(ext.adapter_dir / env.requirements)
    parts = [env.python, env.torch, env.torch_backend, env.compiled, req]
    if env.cuda_toolkit:
        parts.append(env.cuda_toolkit)
    if env.conda:
        parts.append(["conda", *env.conda])
    if env.build:
        script = (ext.adapter_dir / env.build).read_text()
        parts.append(["build", _content(ext.adapter_dir / env.build)])
    if env.build_files:
        parts.append(["build_files", *(_content(ext.adapter_dir / f) for f in env.build_files)])
    if env.build:
        used = {w.source: manual.item_of(ext, w.source).install.path() for w in ext.weights
                if w.kind == "manual" and f"LAB2SHOT_MANUAL_{w.source.upper()}" in script}
        if used:  # the build compiles against them: a new version rebuilds
            # by where it really is: a checkout that links another checkout's third_party sees the same SDK, not another path
            parts.append(["manual", *(str(path.resolve()) for path in used.values() if path is not None)])  # a missing one: preflight says so
    return _hash(parts)  # sort_keys on: a spec's parts never depend on dict order


def repo_fingerprint(ext: Extension) -> str:
    return _hash([ext.source.url, ext.source.commit, ext.submodules,
                  {k: [v.url, v.commit] for k, v in ext.extra_sources.items()}])


def has_post_install(ext: Extension) -> bool:
    return type(ext).post_install is not Extension.post_install


def steps(ext: Extension) -> list[Step]:
    env_fp, repo_fp = env_fingerprint(ext), repo_fingerprint(ext)
    weights_fp = _hash([[w.key, w.kind, w.source, w.dest, w.sha256, w.revision, w.files] for w in ext.weights])
    out = [Step("repo", repo_fp), Step("env", env_fp), Step("packages", env_fp)]
    if ext.env.build:
        out.append(Step("build", env_fp))
    out.append(Step("weights", weights_fp))
    if has_post_install(ext):
        out.append(Step("post", _hash([weights_fp, repo_fp])))
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


def read_state(paths: ExtensionPaths) -> dict:
    try:
        return json.loads(paths.state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def built_for(paths: ExtensionPaths) -> str | None:
    """The fingerprint the environment at `paths` was finished for (installer: after its last environment step), None
    when there is no environment there or none was finished."""
    if not paths.python.exists():
        return None
    return read_state(paths).get("env", {}).get("fingerprint")


def target(ext: Extension, force: bool = False) -> ExtensionPaths:
    """Where this install builds: the live environment when it is finished for this spec (nothing to build beside it);
    the one an earlier install began for this same spec (it goes on there); else a new folder beside the live one."""
    live = ext.paths
    env_fp = env_fingerprint(ext)
    repo_dir = live.repo_dir if read_state(live).get("repo", {}).get("commit") in (None, ext.source.commit) else f"repo-{ext.source.commit[:12]}"
    if not force and built_for(live) == env_fp:
        return ExtensionPaths(live.root, live.env, repo_dir)
    building = paths_of(ext, pointer(ext).get("building"))
    if building is not None and not force and read_state(building).get("plan") == env_fp:
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
