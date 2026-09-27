"""外部文件的内容身份，作为节点指纹的组成部分。

不能仅依据大小和修改时间：CG 制作中常原地替换素材，部分软件直接改写文件内容而修改时间不变。仅比较大小和
修改时间会在素材已更换时命中旧缓存。

也不对每次请求都重新哈希整个文件：状态页每次刷新都要计算指纹（节点上的「已算过 / 没算过」状态），数百帧的
序列每次刷新都重读磁盘，开销不可接受。因此本模块读取文件计算摘要（实现见 io/digest.py，吞吐约 1.4 GB/s），
结果按 (路径, 大小, 修改时间) 缓存在进程内；提交计算时整表作废（forget_all），下一次计算指纹必定重新读盘。
由此状态页开销低，提交时的指纹准确。

本层只处理文件，不区分素材是否为上传所得：上传的素材本身按内容寻址（清单中记录每个文件的 sha256），相应的
捷径位于处理上传的上层（lab2shot/catalog.py 的 PlanEnv.content_id）。底层不得反向依赖上层（io 为底层，
transfer 为上层）。
"""

from __future__ import annotations

from pathlib import Path

_known: dict[tuple[str, int, int], str] = {}


def forget_all() -> None:
    """作废全部已缓存的内容身份。提交计算时调用一次，使本轮指纹按文件当前的实际内容计算。"""
    _known.clear()


def content_id(path: Path | str) -> str:
    """返回文件 `path` 的内容身份：读取文件计算 sha256，结果按 (路径, 大小, 修改时间) 缓存在进程内。"""
    from .digest import sha256  # 摘要只有一处实现（io/digest.py），此处不直接使用 hashlib

    p = Path(path)
    st = p.stat()
    at = (str(p), st.st_size, st.st_mtime_ns)
    got = _known.get(at)
    if got is None:
        got = _known[at] = sha256(p)
    return got
