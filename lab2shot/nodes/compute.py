"""GPU 计算量档位（低/中/高/超高）：为每个使用显卡的节点和每个模板标注档位，使用者在填写参数前即可估计其显存占用与速度。
危险参数的选项限制决定能否提交，本模块则提示提交后的大致开销。

数据来源：各 adapter 的 docs.md 中在 RTX 4090 上以默认参数测得的显存峰值与每帧耗时，即 NodeDef.cost 的 vram_gb 与
seconds_per_frame；测过的选项记录在 NodeDef.traits 中（nodes/applies.py）。本模块只负责将这两个数值换算为档位，并生成
面向使用者的提示文字；数值本身在各节点的 nodes.py 中声明（由 NodeDef.describe() 调用本模块的函数）。

档位规则：取显存与速度两个维度中较差的一项（任一维度达到超高即为超高，两个维度均满足才为低）。
    显存（默认参数下测得的峰值）      < 6 GB       低
                                    6–12 GB      中
                                    12–18 GB     高
                                    > 18 GB      超高
    每帧耗时（默认参数下测得；整段一次计算、按张标定等不适用逐帧计时的节点不计此项）
                                    < 0.2 秒/帧   低
                                    0.2–1 秒/帧   中
                                    1–5 秒/帧     高
                                    > 5 秒/帧     超高
"""

from __future__ import annotations

TIERS: tuple[str, ...] = ("低", "中", "高", "超高")
VRAM_BREAKS = (6.0, 12.0, 18.0)  # GB
TIME_BREAKS = (0.2, 1.0, 5.0)  # seconds/frame


def _tier_index(x: float | None, breaks: tuple[float, float, float]) -> int:
    if not x:  # 0 or None: not measured on this axis, so it does not raise the rating
        return 0
    for i, b in enumerate(breaks):
        if x < b:
            return i
    return 3


def compute_tier(vram_gb: float, seconds_per_frame: float | None) -> str:
    """Return 低/中/高/超高 from the measured peak VRAM (GB) and seconds per frame at default parameters, taking the
    worse of the two axes (see the module docstring)."""
    return TIERS[max(_tier_index(vram_gb, VRAM_BREAKS), _tier_index(seconds_per_frame, TIME_BREAKS))]


def compute_tip(tier: str, seconds_per_frame: float | None, note: str = "") -> str:
    """Return the rating's tooltip, including the node's measured time at default parameters. The measured VRAM and
    the card it was measured on are hardware information; they remain structured fields on the resolved cost
    (vram_gb, vram_measured, measured_on), which the server exposes only with the farm.cards permission."""
    tail = ""
    if seconds_per_frame:
        tail = f"（约 {seconds_per_frame * 1000:.0f} 毫秒/帧）" if seconds_per_frame < 0.1 else f"（约 {seconds_per_frame:g} 秒/帧）"
    return f"计算量：{tier}，默认参数实测{tail}{('，' + note) if note else ''}"


def combine(ratings) -> dict | None:
    """Return the worst of several {"tier", "tip"} ratings, or None when none is present."""
    present = [r for r in ratings if r]
    if not present:
        return None
    return max(present, key=lambda r: TIERS.index(r["tier"]))


def template_compute(nodes: list[dict]) -> dict | None:
    """Return a template's rating: the worst of its nodes' ratings, each evaluated at the parameters the template sets
    rather than at the node defaults, so a template that selects a heavier option is rated accordingly.
    `nodes` is a graph's "nodes" list (each {"type", "params", ...})."""
    from . import node_types
    from .applies import resolve_params

    types = node_types()
    ratings = []
    for n in nodes:
        t = types.get(n.get("type", ""))
        if t is None:
            continue
        try:
            params = t.load_params(n.get("params", {}))
        except Exception:  # stale or unknown parameter: skip the node rather than fail the whole listing
            continue
        ratings.append(resolve_params(t, params).cost.rating)
    return combine(ratings)
