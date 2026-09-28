"""GPU 计算量档位（低/中/高/超高）：为每个使用显卡的节点和每个模板标注档位，使用者在填写参数前即可估计其显存占用与速度。
危险参数的选项限制决定能否提交，本模块则提示提交后的大致开销。

数据来源：各 adapter 的 docs.md 中在 RTX 4090 上以默认参数测得的显存峰值与每帧耗时，即 NodeDef.cost 的 vram_gb 与
seconds_per_frame；测过的选项记录在 NodeDef.traits 中（nodes/applies.py）。本模块只负责将这两个数值换算为档位；
数值本身在各节点的 nodes.py 中声明（由 nodes/applies.py resolve_cost 调用本模块的函数）。

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
