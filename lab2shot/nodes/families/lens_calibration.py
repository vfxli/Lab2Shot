"""镜头标定家族：一张画面 → 该镜头的参数（Focal Length、Filmback、镜头内参〔模型 + 畸变系数 + 主点 + 像素比〕）。

成员：`anycalib.calibrate`、`geocalib.calibrate`。共同声明为输入 `image.3`，输出 `focal` + `filmback` + `lens`
（均为 `value.*`）；成员按上游额外提供的内容自行追加端口（GeoCalib 的重力方向）。

「镜头标定」这一名称与交付物分类 `lens_calib`（menu/categories.json，位于 `camera_track` 下）及上游项目自身的叫法
（camera calibration / single-image calibration）一致，描述的是艺术家需要的结果（测量镜头），而非内部的数学操作。

不应与核心节点「LensDistortion」混淆：后者用已知镜头去畸变 / 加畸变；本家族位于其上游，负责测量出该镜头。

「镜头模型」在本家族中只有一个含义：`lens` 输出端口中的 model，即识别出的镜头类型。
「按哪种模型拟合」一律为节点参数 `fit_model`，标签为「拟合模型」。两者不得共用标签，
否则在节点图上无法区分哪个是要求、哪个是结果。
"""

from __future__ import annotations

from ..kit.ports import rgb_port
from ..base import Port
from .base import WorkerNode
from ...data.values import LENS
from ..values import FLOAT

# 「Focal Length」「Filmback」端口的说明，各成员共用（family.lens_calibration.<端口>.help）：上游给出像素单位的 Focal Length，
# 此处换算为项目标准单位毫米后输出（与 SMPL 参数写入「蒙皮角色」属于同一类）；Filmback 即节点参数原样输出，让下游无需重复填写。

class LensCalibration(WorkerNode):
    """镜头标定节点的共同声明：输入一张画面，输出该镜头的若干参数。

    成员通过 `outputs = LensCalibration.outputs + (...)` 追加上游额外提供的端口（GeoCalib 的重力方向和重力误差；
    AnyCalib 的主点包含在「镜头内参」中）。此处的端口均由各成员的上游实际提供（AnyCalib 取 `pred["intrinsics"]`，
    GeoCalib 取 `camera` 对象），「Filmback」除外：它是节点参数的透传，由成员在 `Official(ours=...)` 中登记。
    """

    # 该节点自行从画面估计畸变，因此不假定输入画面已去畸变（nodes/applies.py LENSES）
    lens = "any"
    inputs = (rgb_port(),)
    outputs = (
        Port("focal", FLOAT, unit="mm", words="family.lens_calibration.focal"),
        Port("filmback", FLOAT, unit="mm", words="family.lens_calibration.filmback"),
        # 镜头模型、畸变系数、主点、像素比打包为一份「镜头内参」（nodes/lens.py packed_lens），而非各设一个端口
        # every choice gives one, a pinhole too (no coefficients: an identity ST-map downstream): a port's type never
        # changes with a value
        Port("lens", LENS, words="family.lens_calibration.lens"),
    )
