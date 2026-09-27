"""登记一份已安装的环境：核对其与声明是否一致，一致时才写入安装记录。

部分项目须先手工配置好环境才能编写 adapter（例如需要与自带的 Caffe 一起编译、多个环境相互借用工具链的情况）。
环境可以运行但安装器没有记录时，服务器会判定该环境不是按当前代码安装完成的（`E-EXT-OUTDATED`），
节点在页面上不可用。重新安装（数十 GB、数小时，且可能破坏已可运行的环境）代价过高，手写状态文件更不可取，
因此提供此途径。

此途径不是跳过检查的后门：登记前逐项核对，任一项不符即拒绝并说明差异：

1. 环境是否存在（`paths.python`）；
2. 代码是否位于声明的 commit（仓库本身及 `extra_sources` 的每一项）；
3. 声明的权重是否位于声明的位置，带 sha256 的是否一致；
3b. build 脚本放入检出中的文件（`EnvSpec.places`，上游按固定路径读取）是否存在。不核对这些文件，写出的
   「已就绪」即为虚假状态（缺少一个文件时，前面数十分钟的步骤会白白运行，直到最后一步才报错）。此处只检查、
   不放置：缺失时提示运行 `lab2shot ext place <名字>`；`adopt` 保持只读，不会破坏已可运行的环境。
4. 环境中的 Python 版本和固定的 torch 版本是否一致（在该环境中查询）；
5. 最后运行安装器自身的自检步骤（导入 worker SDK、torch 和扩展声明的模块，探测显卡架构），
   与正常安装使用同一段代码，不另行实现。

全部通过才完成登记：第 5 步的自检要读取安装记录中的环境指纹（run.py `_selfcheck` 读取 `state["env"]`），因此
记录在自检前写入；自检未通过时将状态文件恢复原状（原本不存在则删除），live 指针仅在自检通过后写入，
对外表现为未写入任何记录。
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from ..extensions.spec import Extension, InstallError
from ..messages import Msg
from . import plan
from .events import Sink
from .run import Context, Live, _head, install, locked
from .sources import Policy


def _python_version(python: Path) -> str:
    """环境自身报告的 `<主>.<次>` 版本（在该环境中查询，不依据文件名）。"""
    out = subprocess.run([str(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                         capture_output=True, text=True, timeout=120)
    return out.stdout.strip()


FREEZE_TIMEOUT_S = 600


def _frozen(python: Path) -> dict[str, str]:
    """环境中已安装的包：包名（小写，下划线替换为短横线）-> 版本。FREEZE_TIMEOUT_S 秒内无响应时抛出
    InstallError（E-ADOPT-FREEZETIMEOUT）；不得让 TimeoutExpired 直接抛出，否则页面上会显示为 500。"""
    try:
        out = subprocess.run(["uv", "pip", "freeze", "--python", str(python)], capture_output=True, text=True, timeout=FREEZE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise InstallError(Msg("E-ADOPT-FREEZETIMEOUT", seconds=FREEZE_TIMEOUT_S)) from exc
    got = {}
    for line in out.stdout.splitlines():
        name, sep, version = line.partition("==")
        if sep:
            got[name.strip().lower().replace("_", "-")] = version.strip()
    return got


def _pin_problem(pin: str, frozen: dict[str, str]) -> Msg | None:
    """核对一条固定版本的依赖（`torch==2.1.1+cu118`）与环境中实际安装的是否一致；一致时返回 None。
    本地版本号（`+cu118`）仅在声明中写明时才比较，否则只比较前面的部分。"""
    name, sep, want = pin.partition("==")
    key = name.strip().lower().replace("_", "-")
    have = frozen.get(key)
    if have is None:
        return Msg("E-ADOPT-NOPACKAGE", name=name)
    if not sep:
        return None
    want, have_cmp = want.strip(), have
    if "+" not in want:
        have_cmp = have.split("+")[0]
    return None if have_cmp == want else Msg("E-ADOPT-PIN", name=name, want=want, have=have)


def check(ext: Extension) -> list[Msg]:
    """列出环境与声明不一致之处，每项一句；全部一致时返回空列表。不写入、不修改任何内容。"""
    from lab2shot_shared.protocol import shown

    paths = ext.paths
    problems: list[Msg] = []
    if not paths.python.exists():
        return [Msg("E-ADOPT-NOENV", path=shown(paths.python))]

    # 2. 代码位于声明的 commit
    for where, src in [(paths.repo, ext.source), *((paths.root / f, s) for f, s in ext.extra_sources.items())]:
        head = _head(where)
        if not head:
            problems.append(Msg("E-ADOPT-NOTGIT", name=where.name, path=shown(where)))
        elif head != src.commit:
            problems.append(Msg("E-ADOPT-COMMIT", name=where.name, want=src.commit[:10], have=head[:10]))

    # 3. 权重位于声明的位置，带 sha256 的须一致
    from . import sources

    for w in ext.weights:
        if w.kind == "manual" or w.option is not None:
            continue
        dest = paths.weights / w.dest
        if not dest.exists():
            problems.append(Msg("E-ADOPT-NOWEIGHT", key=w.key, path=shown(dest)))
        elif w.sha256 and w.kind == "url" and not sources.matches(dest, w.sha256):
            problems.append(Msg("E-ADOPT-WEIGHTSHA", key=w.key, path=shown(dest)))

    # 3b. build 放入检出中的文件（只检查，不放置）
    from .place import missing

    if left := missing(ext):
        problems.append(Msg("E-ADOPT-NOTPLACED", name=ext.name, count=len(left), first=left[0]))

    # 4. Python 与固定的 torch 版本
    found = _python_version(paths.python)
    if found != ext.env.python:
        problems.append(Msg("E-ADOPT-PYTHON", want=ext.env.python, have=found))
    if ext.env.torch:
        try:
            frozen = _frozen(paths.python)
        except InstallError as exc:
            return problems + [exc.message]
        problems += [p for p in (_pin_problem(pin, frozen) for pin in ext.env.torch) if p is not None]
    return problems


def adopt(ext: Extension, sink: Sink, *, live: Live | None = None) -> None:
    """核对后登记这份已安装的环境。核对不通过时抛出 InstallError，不写入任何记录；自检未通过时将已写入的记录恢复原状。"""
    live = live or Live()
    with locked(ext):
        paths = ext.paths
        problems = check(ext)
        if problems:
            raise InstallError(Msg("E-ADOPT-MISMATCH", title=ext.title, name=ext.name, count=len(problems),
                                   detail="\n  ".join(m.text for m in problems[:10])))
        before = paths.state_file.read_bytes() if paths.state_file.exists() else None  # 自检未通过时恢复为此内容
        ctx = Context(ext, paths, sink, Policy.from_settings(), live, force=False)
        steps = {s.id: s for s in plan.steps(ext)}
        ctx.state["repo"] = {"url": ext.source.url, "commit": ext.source.commit, "dir": paths.repo_dir}
        ctx.state.setdefault("weights", {}).update(
            {w.key: "ok" for w in ext.weights if w.kind != "manual" and w.option is None})
        ctx.record_env()  # 环境指纹 + uv pip freeze：与安装器写入的内容相同
        done = {k: {"state": "done", "fingerprint": steps[k].fingerprint, "time": time.time()}
                for k in ("repo", "env", "packages", "build", "weights", "post") if k in steps}
        ctx.state.setdefault("steps", {}).update(done)
        ctx.save()
        sink.say(Msg("I-ADOPT-CHECKED", title=ext.title, commit=ext.source.commit[:10],
                     weights=len(ctx.state["weights"]), places=len(ext.env.places)))
    # 自检使用安装器自身的步骤（导入、权重、探测显卡架构），不另行实现。自检需要读取上面写入的环境指纹，因此先写记录；
    # 未通过时恢复，登记失败的环境不得保留「已安装」的记录
    try:
        install(ext, sink, only=("selfcheck",), live=live)
    except BaseException:
        with locked(ext):
            if before is None:
                paths.state_file.unlink(missing_ok=True)
            else:
                paths.state_file.write_bytes(before)
        raise
    with locked(ext):
        ptr = plan.pointer(ext)
        ptr["current"] = plan.slot(ext.paths)
        ptr.pop("building", None)
        plan.write_pointer(ext, ptr)
    sink.say(Msg("I-ADOPT-DONE", title=ext.title))


__all__ = ["adopt", "check"]
