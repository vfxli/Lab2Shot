"""Adapter loading. Each adapters/<name>/ folder is one extension: extension.py defines its EXTENSION and the
optional nodes.py its NODES. Each extension is loaded once and independently. This module is the only place the core
imports adapters and the only place it imports modules by name (the layering rule); `import_core()` walks the
lab2shot package at server start so that no background task is the first to import a lab2shot module.

An extension that breaks a rule is recorded with the reason and excluded, so it cannot bring down the other
extensions or the application (`lab2shot ext list` shows it):
- a module that fails to import (for example, a partially edited file);
- a spec that is not an Extension, is named other than its folder (so two can't claim one name), or is written for
  another version of the adapter API (Extension.sdk, lab2shot.sdk.SDK_API), or points outside its folder (../);
- nodes that are not node types, whose ids are not "<extension>.<name>", that run in another extension's
  environment, or that repeat an id. In that case none of its nodes load, but its spec still does (so it can still be
  installed or removed).
An extension that builds on another (Extension.requires) loads its nodes after the required one, and only if the
required one loaded completely; otherwise it is reported as not ready with the reason (extensions/status.py) and the
other extensions are unaffected.

Each node class of an extension that loaded is stamped with its project (NodeDef.project, nodes/services.py
ProjectFacts: title, licence class, the tags its extension brings, whether it is usable, the spec its worker runs from),
so the nodes, the tags and the engine read the class and never reach up into the extension registry.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
import sys
import traceback
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ADAPTERS_DIR, ROOT
from .messages import Msg

if TYPE_CHECKING:
    from .extensions.spec import Extension
    from .nodes.base import NodeDef
    from .nodes.services import ProjectFacts


def import_core() -> int:
    """Import every module of the lab2shot package and return the count. The server calls this on its main thread at
    start, before any background task (server/restart.py import_everything): an import holds a lock, and a task
    importing a module while the next server starts after a restart's exec would block that start. The package is
    walked rather than listed, because an explicit list goes stale as soon as a task reaches a module it does not
    name. About 250 modules, 0.6 s."""
    import pkgutil

    import lab2shot

    count = 0
    for found in pkgutil.walk_packages(lab2shot.__path__, "lab2shot."):
        importlib.import_module(found.name)
        count += 1
    return count


@dataclass(frozen=True, eq=False)
class Adapters:
    extensions: dict[str, Extension]  # name (its folder) -> spec
    nodes: dict[str, type[NodeDef]]  # node id -> type, of every extension whose nodes loaded
    broken: dict[str, str]  # folder -> why it (or its nodes) was left out

    def why_missing(self, type_id: str) -> Msg:
        """Why there is no node type `type_id`: its extension (the id's first part) is not there, or its nodes did not
        load, or it has no such node."""
        from .extensions.status import extension_status, missing_requirements

        name = type_id.split(".")[0]
        if name in self.broken:
            return Msg("E-NODETYPE-BROKEN", extension=name, reason=self.broken[name])
        if name not in self.extensions:
            return Msg("E-NODETYPE-NOEXTENSION", extension=name)
        ext = self.extensions[name]
        if missing_requirements(ext):
            return Msg("E-NODETYPE-UNMET", extension=ext.title, reason=extension_status(ext)["reason"])
        return Msg("E-NODETYPE-NOTINEXTENSION", extension=ext.title)


@cache
def adapters() -> Adapters:
    """Every adapter folder, loaded."""
    return load(sorted(f for f in ADAPTERS_DIR.iterdir() if (f / "extension.py").is_file()))


def load(folders: list[Path]) -> Adapters:
    """The extensions of these adapter folders, loaded."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    specs: dict[str, Extension] = {}
    broken: dict[str, str] = {}
    for folder in folders:
        module = _import(folder.name, "extension", broken)
        if module is None:
            continue
        why = _spec_problem(folder.name, getattr(module, "EXTENSION", None))
        if why:
            broken[folder.name] = why.text
        else:
            specs[folder.name] = module.EXTENSION
    nodes: dict[str, type[NodeDef]] = {}
    whole: set[str] = set()  # extensions whose nodes loaded (or that have none)
    for name in _requirements_first(specs, broken):
        ext = specs[name]
        if not all(r in whole for r in ext.requires):
            continue  # not ready: extension_status says which one it lacks
        if not (ext.adapter_dir / "nodes.py").is_file():
            whole.add(name)
            continue
        module = _import(name, "nodes", broken)
        if module is None:
            continue
        declared = getattr(module, "NODES", ())
        why = _nodes_problem(name, declared, nodes)
        if why:
            broken[name] = f"nodes.py：{why.text}"
            continue
        project = project_facts(ext)
        for n in declared:
            n.project = project
        nodes.update((n.id, n) for n in declared)
        whole.add(name)
    return Adapters(specs, nodes, broken)


def project_facts(ext: Extension) -> ProjectFacts:
    """An extension's project as its node classes carry it: its title, its declared licence class, 需注册 from the
    hand-downloaded items it needs, whether it is ready now (extensions/status.py), and its spec."""
    from .extensions import status as ext_status
    from .extensions.manual import BODY_KEYS
    from .nodes.services import ProjectFacts
    from .nodes.tags import REGISTRATION

    def available() -> Msg | None:  # evaluated on every call: readiness can change at any time
        said = ext_status.extension_status(ext)["message"]  # None when ready
        return None if said is None else Msg(said["code"], **said["params"])

    # 需注册的手动下载：身体模型一律需要（官网注册），扩展自身的条目按其声明的 registration 判断。此处不查询
    # 注册表，因为本步骤在扩展加载期间执行，此时注册表尚未构建完成
    own = {it.key: it.registration for it in ext.manual_items}
    more = frozenset({REGISTRATION for w in ext.weights if w.kind == "manual" and (w.source in BODY_KEYS or own.get(w.source))})
    return ProjectFacts(ext.title, ext.license.tag, more, available, ext, _result_identity(ext))


@lru_cache(maxsize=256)
def _result_identity(ext: Extension) -> str:
    """返回决定该扩展计算结果的标识：代码（仓库提交）、环境（python / torch / 依赖 / 编译）和权重（声明的 sha256）。
    全部取自声明，不读取磁盘上的模型文件，因此开销很小。该标识计入节点指纹，重装扩展或更换权重后旧结果不再命中。"""
    from .installer.plan import env_fingerprint, repo_fingerprint
    from .io.digest import key

    weights = sorted(f"{w.key}:{getattr(w, 'sha256', '') or ''}" for w in ext.weights)
    return key([repo_fingerprint(ext), env_fingerprint(ext), weights], 16)  # 摘要只有一处实现（io/digest.py）


def project_of(runtime: str) -> ProjectFacts:
    """The project a runtime belongs to, whether or not it has node types: the core's, an extension's that loaded, or
    one never loaded (the strictest licence, never usable)."""
    from .nodes.services import CORE_PROJECT, unloaded_project

    if runtime == "core":
        return CORE_PROJECT
    ext = adapters().extensions.get(runtime)
    return project_facts(ext) if ext is not None else unloaded_project(runtime)


def _import(folder: str, part: str, broken: dict[str, str]):
    try:
        return importlib.import_module(f"adapters.{folder}.{part}")
    except Exception as exc:
        broken[folder] = f"{part}.py：{type(exc).__name__}: {exc}"
        traceback.print_exc()
        return None


def _spec_problem(folder: str, ext) -> Msg | None:
    from .extensions.spec import Extension
    from .sdk import SDK_API

    if not isinstance(ext, Extension):
        return Msg("E-ADAPTER-NOSPEC")
    if ext.name != folder:
        return Msg("E-ADAPTER-NAME", name=ext.name, folder=folder)
    if folder == "core":
        return Msg("E-ADAPTER-CORE")
    if ext.sdk != SDK_API:
        return Msg("E-ADAPTER-SDK", sdk=ext.sdk, current=SDK_API)
    for what, path in (("EnvSpec.requirements", ext.env.requirements), ("EnvSpec.build", ext.env.build),
                       *(("worker_modules", m) for m in ext.worker_modules)):
        if Path(path).is_absolute() or ".." in Path(path).parts:
            return Msg("E-ADAPTER-OUTSIDE", what=what, path=str(path))
    return None


def _nodes_problem(name: str, declared, taken: dict) -> Msg | None:
    from .nodes.base import NodeDef

    if not isinstance(declared, (tuple, list)) or not all(isinstance(n, type) and issubclass(n, NodeDef) for n in declared):
        return Msg("E-ADAPTER-NODES")
    ids = [n.id for n in declared]
    for n in declared:
        if not n.id.startswith(f"{name}."):
            return Msg("E-ADAPTER-NODEID", node=n.id, extension=name)
        if n.runtime != name:
            return Msg("E-ADAPTER-RUNTIME", node=n.id, runtime=n.runtime, extension=name)
        if ids.count(n.id) > 1 or n.id in taken:
            return Msg("E-ADAPTER-DUPLICATE", node=n.id)
    return None


def _requirements_first(specs: dict[str, Extension], broken: dict[str, str]) -> list[str]:
    """The extensions in folder order, each after the ones it requires; those that require each other (a cycle) are
    broken."""
    order: list[str] = []
    state: dict[str, str] = {}

    def visit(name: str, path: tuple[str, ...]) -> None:
        if state.get(name) == "done" or name not in specs:
            return
        if name in path:
            for n in path[path.index(name):]:
                broken[n] = Msg("E-ADAPTER-CYCLE", chain=" → ".join((*path[path.index(name):], name))).text
            return
        for r in specs[name].requires:
            visit(r, (*path, name))
        state[name] = "done"
        if name not in broken:
            order.append(name)

    for name in specs:
        visit(name, ())
    return order
