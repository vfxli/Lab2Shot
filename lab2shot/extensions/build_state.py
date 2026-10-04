"""What identifies an extension's build, and what its live environment and checkout hold: read-only.

The installer (lab2shot/installer) builds and writes the state (state.json, the placed files); the extensions' status
(extensions/status.py), a result's identity (lab2shot/adapters.py) and the installer itself read it here. It sits with
the extensions, below the installer (cli/check_arch.py LAYERS), so reading whether an extension is ready never reaches up
into the code that installs it."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from .spec import Extension, ExtensionPaths


def _hash(parts) -> str:
    from ..io.digest import key

    return key(parts, 16)


# Calls whose string arguments are words for a person (a progress line, a log line, an error's text), never a build input
SAID_CALLS = {"print", "SystemExit", "exit"}
SAID_MODULES = {"logging", "log", "logger", "LOG", "LOGGER"}  # log.info("..."), logging.warning("...")


class _Unsaid(ast.NodeTransformer):
    """A build script's syntax tree without what it says to a person: bare string statements (docstrings included) go,
    the string arguments of print / SystemExit / sys.exit / logging calls, of a raised exception and of an assert
    become one placeholder. Editing a message then leaves the fingerprint as it was; editing what the script does
    changes it."""

    @staticmethod
    def _blank(node: ast.expr) -> ast.expr:
        """Every string and f-string inside `node` (a message built with + or % too) as the placeholder."""
        class Words(ast.NodeTransformer):
            def visit_JoinedStr(self, n):
                return ast.Constant("")

            def visit_Constant(self, n):
                return ast.Constant("") if isinstance(n.value, str) else n

        return Words().visit(node)

    def _blank_call(self, call: ast.Call) -> None:
        call.args = [self._blank(a) for a in call.args]
        for kw in call.keywords:
            kw.value = self._blank(kw.value)

    def visit_Expr(self, node: ast.Expr):
        if isinstance(node.value, (ast.Constant, ast.JoinedStr)) and isinstance(getattr(node.value, "value", ""), str):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
        owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
        if (isinstance(func, ast.Name) and name in SAID_CALLS) or owner in SAID_MODULES or (owner, name) == ("sys", "exit"):
            self._blank_call(node)
        return node

    def visit_Raise(self, node: ast.Raise):
        self.generic_visit(node)
        if isinstance(node.exc, ast.Call):
            self._blank_call(node.exc)
        elif node.exc is not None:
            node.exc = self._blank(node.exc)
        return node

    def visit_Assert(self, node: ast.Assert):
        self.generic_visit(node)
        if node.msg is not None:
            node.msg = self._blank(node.msg)
        return node

    def generic_visit(self, node):
        super().generic_visit(node)
        if isinstance(getattr(node, "body", None), list) and not node.body:  # a block emptied by a removed statement
            node.body = [ast.Pass()]
        return node


def read_file(path: Path) -> str | None:
    """A build input's text as it is on disk now (None: not there): how env_fingerprint reads them by default."""
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _content(path: Path, read=read_file) -> str:
    """The part of a build input that affects the build, so that editing its comments or its messages does not mark
    an installed environment as outdated. Python sources are reduced to their syntax tree without docstrings and
    without what they say to a person (_Unsaid); other text files (requirements lists, lock files, shell fragments)
    lose comment lines, trailing comments and blank lines. `read` gives a file's text (None: not there): the one on
    disk, or another version's (installer/migrate.py compares an earlier release's files)."""
    text = read(path)
    if text is None:
        return ""
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text
        return ast.dump(_Unsaid().visit(tree))
    lines = (line.split(" #", 1)[0].rstrip() for line in text.splitlines() if not line.lstrip().startswith("#"))
    return "\n".join(line for line in lines if line)


def env_fingerprint(ext: Extension, read=read_file) -> str:
    """What identifies the environment's build (preflight.card_fit compares it with what the live environment was
    built for). Build inputs are compared by content, not by their literal text (`_content`); `read` gives their text
    (read_file: on disk now). An extension running in another's environment (Extension.runs_in) has that one's."""
    from . import manual

    if ext.env_owner is not ext:
        return env_fingerprint(ext.env_owner, read)
    env = ext.env
    # what step_packages installs from: the hash-pinned lock when there is one (it replaces the requirements), else
    # the requirements list
    lock = ext.adapter_dir / env.lock
    locked = read(lock)
    req = ["lock", _content(lock, read)] if locked is not None else _content(ext.adapter_dir / env.requirements, read)
    parts = [env.python, env.torch, env.torch_backend, env.compiled, req]
    if env.compiled and not env.compiled_cuda:  # the compiled packages are built without their CUDA kernels
        parts.append(["compiled_cuda", False])
    if env.cuda_toolkit:
        parts.append(env.cuda_toolkit)
    if env.conda:
        parts.append(["conda", *env.conda])
    if env.build:
        script = read(ext.adapter_dir / env.build) or ""
        parts.append(["build", _content(ext.adapter_dir / env.build, read)])
    if env.build_files:
        parts.append(["build_files", *(_content(ext.adapter_dir / f, read) for f in env.build_files)])
    if env.build:
        used = {w.source: manual.item_of(ext, w.source).install.path() for w in ext.weights
                if w.kind == "manual" and f"LAB2SHOT_MANUAL_{w.source.upper()}" in script}
        if used:  # the build compiles against them: a new version rebuilds
            # by where it really is: a checkout that links another checkout's third_party sees the same SDK, not another path
            parts.append(["manual", *(str(path.resolve()) for path in used.values() if path is not None)])  # a missing one: preflight says so
    return _hash(parts)  # sort_keys on: a spec's parts never depend on dict order


def repo_fingerprint(ext: Extension) -> str:
    if ext.env_owner is not ext:
        return repo_fingerprint(ext.env_owner)
    return _hash([ext.source.url, ext.source.commit, ext.submodules,
                  {k: [v.url, v.commit] for k, v in ext.extra_sources.items()}])


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


def located(paths: ExtensionPaths, rel: str) -> Path:
    """Where a declared place is: relative to the extension's folder, a path starting with repo/ inside the checkout
    `paths` uses (a side-by-side checkout repo-<commit> included)."""
    head, sep, rest = rel.partition("/")
    return paths.repo / rest if head == "repo" and sep else paths.root / rel


def missing(ext: Extension, paths: ExtensionPaths | None = None) -> list[str]:
    """The declared places not there now in `paths` (the live environment by default), as declared. Read-only."""
    paths = paths or ext.paths
    return [rel for rel in ext.env.places if not located(paths, rel).exists()]
