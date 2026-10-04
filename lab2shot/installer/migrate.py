"""One-time migrations of what is installed (third_party/), for code that changed how an installation is identified or
laid out without changing the environment itself. `lab2shot update` runs them with the new code after the database
upgrade (cli/update.py step 6) and undoes them on a rollback (`undo`); `lab2shot ext migrate` runs them by hand.

Each is idempotent: run on what is already migrated (or was never installed the old way), it changes nothing.

- fingerprints: the environment fingerprint is computed by a new rule (extensions/build_state.py: the lock when there is
  one, compiled_cuda, a build script's messages left out). An environment recorded under the earlier rule
  (`legacy_env_fingerprint`, the rule of the release before) whose build inputs are the same gets its record rewritten
  to the new fingerprint: the same environment, not built again. Inputs that really changed are left alone: the
  extension shows 「需重装」 (E-EXT-OUTDATED) as it should.
- bases: an extension now running in a base's environment (Extension.runs_in: alphagen, cleanplate in ltx) that was
  installed with an environment of its own: when the base has none yet, that environment, its checkout and the base's
  model files move into the base's folder, and the install records are split (the base's environment and weights; the
  feature's own weights). A copy left over once the base has one (a second feature) is left where it is and reported.

The journal (`journal`: a folder of the update's record) lists every change before it is made, the records it rewrote
kept beside it, so `undo` puts everything back as it was.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

JOURNAL = "migrate.json"


@dataclass
class Report:
    rewritten: list[tuple[str, str, str]] = field(default_factory=list)  # (extension, old fingerprint, new)
    outdated: list[str] = field(default_factory=list)  # recorded for other build inputs: install again
    moved: list[tuple[str, str]] = field(default_factory=list)  # (feature, base): its environment moved into the base
    split: list[str] = field(default_factory=list)  # features given an install record of their own weights
    left: list[str] = field(default_factory=list)  # copies a base no longer needs (paths): free to delete


# ------------------------------------------------------------------ the journal


class Journal:
    """What a migration changed, written before each change: undone in reverse by `undo`."""

    def __init__(self, folder: Path | None) -> None:
        self.folder = folder
        self.actions: list[dict] = []
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            try:
                self.actions = json.loads((folder / JOURNAL).read_text(encoding="utf-8"))["actions"]
            except (OSError, ValueError, KeyError):
                self.actions = []

    def _save(self) -> None:
        if self.folder is not None:
            tmp = self.folder / (JOURNAL + ".tmp")
            tmp.write_text(json.dumps({"actions": self.actions}, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.folder / JOURNAL)

    def write(self, path: Path, data: dict) -> None:
        """Write an install record (or a pointer), the one there before kept for undo."""
        if self.folder is not None:
            copy = None
            if path.is_file():
                copy = self.folder / f"{len(self.actions):03d}_{path.name}"
                shutil.copy2(path, copy)
            self.actions.append({"kind": "file", "path": str(path), "copy": str(copy) if copy else None})
            self._save()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".migrate.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def move(self, src: Path, dst: Path) -> None:
        self.actions.append({"kind": "move", "from": str(src), "to": str(dst)})
        self._save()
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)  # within third_party: one file system, nothing copied


def undo(folder: Path) -> list[str]:
    """Put back what the journal in `folder` says a migration changed, newest first. Returns what could not be put
    back (empty: all). Standard library only: the update's rollback runs it after the code went back."""
    try:
        actions = json.loads((folder / JOURNAL).read_text(encoding="utf-8"))["actions"]
    except FileNotFoundError:
        return []
    except (OSError, ValueError, KeyError) as exc:
        return [f"{folder / JOURNAL}: {exc}"]
    problems = []
    for a in reversed(actions):
        try:
            if a["kind"] == "move":
                src, dst = Path(a["from"]), Path(a["to"])
                if dst.exists() and not src.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    dst.rename(src)
            elif a["kind"] == "file":
                path = Path(a["path"])
                if a["copy"]:
                    shutil.copy2(a["copy"], path)
                else:
                    path.unlink(missing_ok=True)
        except OSError as exc:
            problems.append(f"{a}: {exc}")
    (folder / JOURNAL).rename(folder / (JOURNAL + ".undone"))
    return problems


# ------------------------------------------------------------------ the earlier rule


def _legacy_content(path: Path, read) -> str:
    """build_state._content as the release before had it: Python sources without their docstrings (messages kept)."""
    text = read(path)
    if text is None:
        return ""
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


def legacy_env_fingerprint(ext, read) -> str:
    """The environment fingerprint by the rule of the release before this one (installer/plan.py env_fingerprint of
    7440b01): the requirements list only (no lock), no compiled_cuda, a build script's messages counted. Kept here
    only to recognise records it wrote."""
    from ..extensions import manual
    from ..extensions.build_state import _hash

    env = ext.env
    parts = [env.python, env.torch, env.torch_backend, env.compiled, _legacy_content(ext.adapter_dir / env.requirements, read)]
    if env.cuda_toolkit:
        parts.append(env.cuda_toolkit)
    if env.conda:
        parts.append(["conda", *env.conda])
    if env.build:
        script = read(ext.adapter_dir / env.build) or ""
        parts.append(["build", _legacy_content(ext.adapter_dir / env.build, read)])
    if env.build_files:
        parts.append(["build_files", *(_legacy_content(ext.adapter_dir / f, read) for f in env.build_files)])
    if env.build:
        used = {w.source: manual.item_of(ext, w.source).install.path() for w in ext.weights
                if w.kind == "manual" and f"LAB2SHOT_MANUAL_{w.source.upper()}" in script}
        if used:
            parts.append(["manual", *(str(path.resolve()) for path in used.values() if path is not None)])
    return _hash(parts)


def files_at(commit: str):
    """A reader of build inputs as they were at `commit` of this checkout (None: not there then)."""
    from ..config import ROOT

    def read(path: Path) -> str | None:
        try:
            rel = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return None
        got = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], capture_output=True)
        return got.stdout.decode("utf-8") if got.returncode == 0 else None

    return read


def same_build(ext, recorded: str, since: str | None) -> bool:
    """Whether `recorded` is the earlier rule's fingerprint of the build inputs this code builds from: by the files on
    disk now, or by those of `since` (the release before) when they compute the same under the new rule."""
    from ..extensions.build_state import env_fingerprint, read_file

    current = env_fingerprint(ext)
    readers = [read_file] + ([files_at(since)] if since else [])
    return any(recorded == legacy_env_fingerprint(ext, r) and env_fingerprint(ext, r) == current for r in readers)


def _renamed(state: dict, old: str, new: str) -> dict:
    """The record with every value `old` (the environment fingerprint, wherever a step or the self-check recorded it)
    replaced by `new`; the switch step's, made of it and the checkout's, made again."""
    from ..extensions.build_state import _hash

    def walk(v):
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return new if v == old else v

    out = walk(state)
    steps = out.get("steps") or {}
    repo_fp = (steps.get("repo") or {}).get("fingerprint")
    switch = steps.get("switch") or {}
    if repo_fp and switch.get("fingerprint") == _hash([old, repo_fp]):
        switch["fingerprint"] = _hash([new, repo_fp])
    return out


# ------------------------------------------------------------------ the migrations


def fingerprints(journal: Journal, report: Report, since: str | None) -> None:
    from ..extensions import extensions
    from ..extensions.build_state import env_fingerprint, read_state
    from ..extensions.spec import state_writing

    for name, ext in sorted(extensions().items()):
        if ext.runs_in:
            continue  # its environment is its base's, migrated as that one
        paths = ext.paths
        if not paths.python.exists():
            continue
        with state_writing(paths):
            state = read_state(paths)
            recorded = (state.get("env") or {}).get("fingerprint")
            current = env_fingerprint(ext)
            if not recorded or recorded == current:
                continue
            if not same_build(ext, recorded, since):
                report.outdated.append(name)
                continue
            journal.write(paths.state_file, _renamed(state, recorded, current))
        report.rewritten.append((name, recorded, current))


def _only(state: dict, ext) -> dict:
    """What of an install record is about `ext`'s own weights: their states, pins and checked files."""
    keys = {w.key for w in ext.weights}
    dests = [f"weights/{w.dest}".rstrip("/") for w in ext.weights if w.dest]
    mine = lambda rel: any(rel == d or rel.startswith(d + "/") for d in dests)  # noqa: E731
    return {"weights": {k: v for k, v in (state.get("weights") or {}).items() if k in keys},
            "weights_pinned": {k: v for k, v in (state.get("weights_pinned") or {}).items() if k in keys},
            "verified": {k: v for k, v in (state.get("verified") or {}).items() if mine(k)}}


def bases(journal: Journal, report: Report) -> None:
    from ..config import THIRD_PARTY_DIR
    from ..extensions import extensions
    from ..extensions.build_state import built_for, env_fingerprint, read_file, read_state
    from ..extensions.spec import ExtensionPaths, active_env

    from .plan import ACTIVE_FILE

    for name, ext in sorted(extensions().items()):
        host = ext.host if ext.runs_in else None
        if host is None:
            continue
        root = THIRD_PARTY_DIR / name
        live = active_env(root).get("current") or {}
        old = ExtensionPaths(root, live.get("env", ""), live.get("repo", "repo"))  # where it was installed on its own
        if not old.python.exists():
            continue  # never installed on its own, or already moved
        state = read_state(old)
        recorded = (state.get("env") or {}).get("fingerprint")
        there = host.paths
        current = env_fingerprint(host)
        if not there.python.exists():
            same_checkout = (state.get("repo") or {}).get("commit") == host.source.commit
            if not same_checkout or recorded not in (current, legacy_env_fingerprint(host, read_file)):
                report.outdated.append(name)
                continue
            base = ExtensionPaths(there.root, old.env, old.repo_dir)
            journal.move(old.venv, base.venv)
            journal.move(old.repo, base.repo)
            for w in host.weights:  # the base's model files, from the folder of the feature they were fetched for
                if w.dest and (root / "weights" / w.dest).exists() and not (base.weights / w.dest).exists():
                    journal.move(root / "weights" / w.dest, base.weights / w.dest)
            own = {k: v for k, v in state.items() if k not in ("weights", "weights_pinned", "verified")}
            journal.write(base.state_file, _renamed({**own, **_only(state, host)}, recorded, current))
            journal.write(there.root / ACTIVE_FILE, {"current": {"env": old.env, "repo": old.repo_dir}})
            report.moved.append((name, host.name))
        elif built_for(there) == current:  # the base is there already: this copy is not used any more
            report.left += [str(p) for p in (old.venv, old.repo) if p.exists()]
            report.left += [str(root / "weights" / w.dest) for w in host.weights if w.dest and (root / "weights" / w.dest).exists()]
        mine = ext.paths.state_file  # the feature's own record: its weights alone
        if not read_state(ext.paths).get("weights"):
            journal.write(mine, _only(state, ext))
            report.split.append(name)


def migrate(journal: Path | None = None, since: str | None = None) -> Report:
    """Every migration, in order (bases first: a moved environment then gets its fingerprint rewritten if need be)."""
    j, report = Journal(journal), Report()
    bases(j, report)
    fingerprints(j, report, since)
    return report
