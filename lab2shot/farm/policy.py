"""调度参数：the numbers the queue and the scheduler go by, in one place.

Every one of them is a setting the administrator can change (lab2shot/config.py, group 队列与显存), read here — once, where it is used — so no module
keeps its own copy and nothing has to be restarted for a change to apply:

    立即计算名额 queue.interactive_pool    light jobs running at once (default worked out from this machine's cores)
    排队上限     queue.waiting_max         light jobs waiting for that pool before one more is refused
    久等优先     queue.age_minutes         how long a job waits before it reserves its card (placement's ageing)
    单账号占卡   queue.account_share       most cards one account may hold at once, as a share of the cards taking jobs
    显存余量     queue.vram_margin_gb      headroom kept free beyond a job's own declared need
    显卡优先顺序 queue.gpu_order           CUDA numbers preferred first among the cards a job fits on

一套公式，2 / 4 / 8 卡通吃: the share is a percentage, never a number of cards, and `cards_per_account` turns it
into a number for whatever machine this is — so moving to a bigger or smaller workstation needs no setting changed.
On a single-card machine the rule turns itself off: the only card can never be "someone else's share".
"""

from __future__ import annotations

from ..config import settings
from ..data.units import PERCENT


def interactive_pool() -> int:
    return int(settings()["queue.interactive_pool"])


def waiting_max() -> int:
    return int(settings()["queue.waiting_max"])


def max_frames() -> int:
    """一次提交最多算多少帧；超过的提交由服务器拒绝并提示。"""
    return int(settings()["queue.max_frames"])


def age_s() -> float:
    """How long a job waits before it reserves the first of its eligible cards in placement order (placement.py's
    ageing)."""
    return float(settings()["queue.age_minutes"]) * 60.0


def margin_gb() -> float:
    return float(settings()["queue.vram_margin_gb"])


def gpu_order() -> list[int]:
    """The administrator's preferred cards (CUDA numbers, first preferred first); [] when not set."""
    from ..config import gpu_order as parse

    return parse(str(settings()["queue.gpu_order"]))


def share_percent() -> int:
    """一个账号最多占几成卡, as whole percent (PERCENT: no limit)."""
    return int(settings()["queue.account_share"])


def cards_per_account(cards: int) -> int:
    """Most cards one account may hold at once on a machine with `cards` cards taking jobs; 0: no limit at all —
    the share is 100%, or there are not at least two cards to divide (one card can never be halved, and holding it
    back from the only account asking would leave it idle for nobody). Worked out in whole numbers, so 「floor(卡数 ×
    占比)」 is exactly that and never a percent's rounding (5 × 60% is 3, not 3.0000000000000004)."""
    percent = share_percent()
    if cards < 2 or percent >= PERCENT:
        return 0
    return max(1, int(cards * percent // PERCENT))


def share_says(cards: int) -> str:
    """One line for the admin page: what the share works out to on this machine, and when it does not apply."""
    cap = cards_per_account(cards)
    percent = share_percent()
    if not cap:
        if cards >= 2:
            return "不限：「单账号占卡」是 100%"
        return f"不限：接任务的显卡{'一张也没有' if cards == 0 else '只有一张'}，这条不起作用"
    return f"{cap} 张（接任务 {cards} 张 × {percent}%），只在别人也在排队时起作用"
