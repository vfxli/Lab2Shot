"""The unit conversions a worker and the core both do (centimetres, Y up; one home for each number): a worker's
world in metres with OpenCV camera axes (+X right, +Y down, +Z forward) -> ours, centimetres with GL / USD camera axes
(+X right, +Y up, -Z forward)."""

from __future__ import annotations

import numpy as np

M_TO_CM = 100.0
CV_TO_GL = np.array([1.0, -1.0, -1.0])
# 帧率在 Lab2Shot 里不随数据流动：中间流只有帧号。要帧率的地方各有一个「帧率」参数（可接线）：写按时间存的文件的输出设置节点，
# 和按秒工作的动作模型节点（worker 的 job.fps）。这个数用在：
#   ① 场景文件（USD）的时基——一个时间码就是一帧，谁都不换算（lab2shot/io/usd.py apply_conventions）；
#   ② 上面那些「帧率」参数的默认值，以及上游 API 非要 fps、节点又没有这个参数时递给它的数。
# 电影的 24 帧。和 M_TO_CM 一样放在这里：核心和 worker 都要用同一个数。
DEFAULT_FPS = 24.0

# 常见帧率，NTSC 三个是精确的 24000/1001、30000/1001、60000/1001。帧率的来源只有两种写法，各按自己的精度认出它代表的标准帧率：
#   手填或文件记的帧率（23.976、59.94）：相对误差在 FPS_SNAP 内；23.976 与 24 只差 1e-3，所以 FPS_SNAP 要比它小得多；
#   文件记的每帧秒数（BVH 的 Frame Time）：按它写出的小数位数，哪个标准帧率的 1/帧率 舍入到这么多位正是它（0.0333 是 30，
#   不是 30.03）；几个都合（0.0417 是 24 也是 23.976）先取整数帧率，再取最近的
STANDARD_FPS = (12.0, 15.0, 24000 / 1001, 24.0, 25.0, 30000 / 1001, 30.0, 48.0, 50.0, 60000 / 1001, 60.0, 90.0, 100.0,
                120.0, 240.0)
FPS_SNAP = 2e-4


def standard_fps(fps: float) -> float:
    """The standard rate a written rate `fps` stands for (STANDARD_FPS, within FPS_SNAP), else `fps` itself."""
    nearest = min(STANDARD_FPS, key=lambda s: abs(fps / s - 1))
    return nearest if abs(fps / nearest - 1) <= FPS_SNAP else float(fps)


def fps_of_period(written: str) -> float:
    """The rate a written time per frame (BVH's Frame Time, as its text) stands for: a whole rate it is exactly, else
    the standard rate whose period
    rounds to exactly that text at its number of decimals (whole rates first, then the nearest: 0.04 is 25, not 24),
    else 1 / the period."""
    from decimal import Decimal

    period = Decimal(written)
    whole = round(1 / float(period))
    if whole > 0 and abs(1 / float(period) - whole) <= 1e-6 * whole:  # exactly a whole rate (0.1 is 10, not 12)
        return float(whole)
    places = max(-period.as_tuple().exponent, 0)
    fits = [s for s in STANDARD_FPS if round(Decimal(1) / Decimal(s), places) == period]
    return min(fits, key=lambda s: (s != round(s), abs(1 / s - float(period)))) if fits else 1.0 / float(period)
