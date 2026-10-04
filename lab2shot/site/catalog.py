"""The node catalogue as the pages and the tools see it, and the one place this process's layers are wired together.

install() hands the lower layers what only the top layer can give (nodes/services.py): the node types of the
extensions that loaded (lab2shot/adapters.py), where an upload reference is (lab2shot/transfer/uploads.py) and a
worker job run outside any cook (engine/external.py ask_worker). The server (server/app.py) and the command line
(cli/base.py) call it when they start, before they read a node type.

describe() is a node type as the catalogue lists it: its own declarations (NodeDef.describe) with what the top layer
knows of its project: the project's name and the tags that decide who may use it. Whether it is usable now is the
session's availability answer (server/available.py: node:<type id>), never a field of the catalogue.
"""

from __future__ import annotations

from pathlib import Path

from ..nodes import tags
from ..nodes.services import Services
from ..nodes.services import install as _install


class _PlanEnv:
    """The PlanEnv this process installs (nodes/services.py PlanEnv): each method reaches into a top-layer module
    the lower layers may not import themselves."""

    def upload(self, ref: str) -> Path:
        from ..transfer import uploads

        return uploads.resolve(ref)

    def declared_upload(self, ref: str) -> dict | None:
        from ..transfer import uploads

        if not uploads.is_ref(ref):
            return None
        return uploads.head_described(ref[len(uploads.PREFIX):].partition("/")[0])

    def may_draw_on(self, layer: Path, file: Path) -> bool:
        from ..transfer import uploads

        return uploads.may_draw_on(layer, file)

    def declared_layers(self, path: Path) -> dict | None:
        from ..transfer import uploads

        return uploads.declared_layers_for(path)

    def channel_plan(self, ref: str, wanted: list[str]) -> dict | None:
        from ..transfer import uploads

        if not uploads.is_ref(ref):
            return None
        return uploads.channel_plan(ref[len(uploads.PREFIX):].partition("/")[0], wanted)

    def content_id(self, path: Path) -> str:
        """返回文件的内容身份（nodes/services.py PlanEnv.content_id）。上传的素材直接取清单中的 sha256（上传按内容
        寻址，无额外开销）；其他路径（数据集、基准素材等服务器上的路径）计算内容哈希，按 (路径, 大小, 修改时间)
        缓存在进程内。每次提交计算时整表作废（content.forget_all），因此原地替换文件且修改时间未变的情况也能识别。

        只有本层了解「上传」（transfer/uploads.py 属于 top，io/content.py 属于 base）：与 upload() 相同，由本层
        把上层信息交给下层，下层不得反向读取上层。"""
        from ..io.content import content_id
        from ..transfer.uploads import sha_of

        return sha_of(path) or content_id(path)

    def ask_worker(self, node_type, params: dict, inputs: dict[str, Path]) -> Path:
        from ..engine.external import ask_worker

        return ask_worker(node_type, params, inputs)

    def holds(self, runtime: str, vram_gb: float) -> bool:
        from ..farm.queue import started
        from ..farm.scheduler import placement

        farm = started()
        return farm is not None and placement.holds(runtime, vram_gb, farm.host.snapshot())


def _extensions():
    from ..adapters import adapters

    return adapters()


def install() -> None:
    """Wire this process's services (idempotent: installing again changes nothing)."""
    _install(Services(extensions=_extensions, plan=_PlanEnv()))


def describe(node_type) -> dict:
    """A node type as the catalogue lists it (/api/catalog, through server/access.py describe_for): NodeDef.describe with its project's name and
    its tags (nodes/tags.py: the chips on the node, and who may use it)."""
    ext = node_type.project.extension
    # 三方节点额外附带项目的仓库与主页（参数面板标题行的 GitHub 标签）以及许可证（显示在面板底部，仅供参考，
    # 不参与可用性判断）
    links = ({"repo": ext.source.url, "homepage": ext.homepage,
              "licence": {"name": ext.license.name, "url": ext.license.url}}
             if ext is not None else None)
    return {**node_type.describe(), "project": node_type.project.title, "tags": sorted(tags.node_tags(node_type)),
            **({"links": links} if links else {})}
