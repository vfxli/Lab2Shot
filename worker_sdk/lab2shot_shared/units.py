"""The unit conversions a worker and the core both do (centimetres, Y up; one home for each number): a worker's
world in metres with OpenCV camera axes (+X right, +Y down, +Z forward) -> ours, centimetres with GL / USD camera axes
(+X right, +Y up, -Z forward)."""

from __future__ import annotations

import numpy as np

M_TO_CM = 100.0
CV_TO_GL = np.array([1.0, -1.0, -1.0])
# 帧率在 Lab2Shot 里不是数据：中间流只有帧号，帧率只在输出节点上出现一次。这个数只有两个用处，都是「非要一个数不可」的地方：
#   ① 场景文件（USD）的时基——一个时间码就是一帧，谁都不换算（lab2shot/io/usd.py apply_conventions）；
#   ② 输出设置节点「帧率」参数的默认值，以及上游 API 非要 fps 时递给它的数（worker 的 job.fps）。
# 电影的 24 帧。和 M_TO_CM 一样放在这里：核心和 worker 都要用同一个数。
DEFAULT_FPS = 24.0
