"""将已下载的文件放置到上游固定读取的位置（`EnvSpec.places`），不做其他操作。

设为单独命令的原因：`adopt` 须完整核对，未核对这些文件时，其写出的「已就绪」即为虚假状态（缺少一个文件时，
前面数十分钟的步骤会白白运行，直到最后一步才报错）。但完整核对不意味着修改磁盘：`adopt` 保持只读，缺失时
说明缺少哪些文件及应运行的命令，放置由本模块完成。本命令也可单独使用：权重被误删需要重新放置时，无需重装整个环境。

实现方式：用扩展自身的环境运行其 build 脚本，仅额外传入参数 `place`。核心不了解任何项目放置的内容，
路径由 adapter 的声明（`EnvSpec.places`）给出，放置方式由 adapter 的 build 脚本实现。
"""

from __future__ import annotations

from ..extensions.spec import Extension, InstallError
from ..messages import Msg
from . import envbuild
from .events import Sink
from .run import Context, Live, locked
from .sources import Policy


def missing(ext: Extension) -> list[str]:
    """声明需放置的文件中当前缺失的部分（相对 root 的路径，原样返回）。只读。"""
    return [rel for rel in ext.env.places if not (ext.paths.root / rel).exists()]


def place(ext: Extension, sink: Sink, *, live: Live | None = None) -> None:
    """运行 build 脚本的放置部分（`<build> place`），完成后再次核对。"""
    if not ext.env.places:
        raise InstallError(Msg("E-PLACE-NOTHING", title=ext.title))
    with locked(ext):
        paths = ext.paths   # 当前启用的环境与检出（active_env.json 指向的版本）
        if not paths.python.exists():
            raise InstallError(Msg("E-ADOPT-NOENV", path=str(paths.python)))
        ctx = Context(ext, paths, sink, Policy.from_settings(), live or Live(), force=False)
        ctx.run([paths.python, ext.adapter_dir / ext.env.build, "place"],
                cwd=paths.root, env=envbuild.build_script_env(ctx))
    left = missing(ext)
    if left:
        raise InstallError(Msg("E-PLACE-STILLMISSING", title=ext.title, count=len(left),
                               detail="\n  ".join(left[:10])))
    sink.say(Msg("I-PLACE-DONE", title=ext.title, count=len(ext.env.places)))


__all__ = ["missing", "place"]
