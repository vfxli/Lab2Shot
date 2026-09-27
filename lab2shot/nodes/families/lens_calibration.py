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

from typing import ClassVar

from ..base import NodeDef, Port
from ..lens import LENS_HELP, only_when_distorting
from ...data.values import LENS
from ..values import FLOAT


# 「Focal Length」端口的说明，两个成员共用。上游给出像素单位的 Focal Length，此处换算为项目标准单位毫米后输出：
# 将上游的值转换为标准表示和标准单位不属于二次加工（与 SMPL 参数写入「蒙皮角色」属于同一类）。
FOCAL_HELP = (
    "Focal Length（mm）。**上游给的是 Focal Length（px）**（用「一个感光点有多宽」当尺子量出来的长度），这里换成毫米交出去。"
    "换算用的是节点上的「Filmback」：Focal Length（mm）= Focal Length（px）÷ 画面宽度 × Filmback。"
)
# 「Filmback」即节点参数原样输出。上游没有该项；透传是为了让下游（LensDistortion、创建相机）无需重复填写，
# 且保证与 Focal Length 的毫米换算始终使用同一数值。
FILMBACK_HELP = (
    "节点上填的「Filmback」原样交出去：Focal Length 是按它换算成毫米的，下游（「LensDistortion」「创建相机」）"
    "接这一根就不用再填一遍，两边永远是同一个数"
)


def focal_px_to_mm(focal_px: float, width: int | float, filmback_mm: float) -> float:
    """Focal Length（px）→ Focal Length（mm）。`width` 必须是 worker 写出的画面宽度，而非节点侧的宽度：
    两者目前相等，但不应依赖这一点。"""
    return float(focal_px) / float(width) * float(filmback_mm)


class LensCalibration(NodeDef):
    """镜头标定节点的共同声明：输入一张画面，输出该镜头的若干参数。

    成员通过 `outputs = LensCalibration.outputs + (...)` 追加上游额外提供的端口（GeoCalib 的重力方向和重力误差；
    AnyCalib 的主点已并入「镜头内参」）。此处的端口均由各成员的上游实际提供（AnyCalib 取 `pred["intrinsics"]`，
    GeoCalib 取 `camera` 对象），「Filmback」除外：它是节点参数的透传，由成员在 `Official(ours=...)` 中登记。
    """

    # 该节点自行从画面估计畸变，因此不假定输入画面已去畸变（nodes/applies.py LENSES）
    lens = "any"
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (
        Port("focal", FLOAT, "Focal Length", unit="mm", help=FOCAL_HELP),
        Port("filmback", FLOAT, "Filmback", unit="mm", help=FILMBACK_HELP),
        # 镜头模型、畸变系数、主点、像素比打包为一份「镜头内参」（nodes/lens.py packed_lens），不再各设一个端口
        Port("lens", LENS, "镜头内参", help=LENS_HELP),
    )

    # 「拟合模型」中确实会解出畸变的档位：「镜头内参」端口只在这些档位下有值，其他档位输出空数据包，
    # 因此在其他档位下该端口置灰且不可用（nodes/lens.py only_when_distorting），而不是隐藏。
    # 成员必须从自身的模型表推导该名单（AnyCalib 的 `LensModel` 即核心表的 id，GeoCalib 的 `TABLE_MODELS`），
    # 不得手写：这样新增拟合模型档位时无需回来修改此处。
    distorting_models: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kw) -> None:
        # 条件在此处施加，而不是在成员的 outputs 中：这些端口是家族声明的同一份，成员通过
        # `outputs = LensCalibration.outputs + (...)` 追加，无法在类体中修改家族的那一份。
        # 施加后得到新的 Port 对象（Port 为 frozen，使用 `dataclasses.replace`），赋给 `cls.outputs` 只在子类上生效，
        # 不会就地修改家族的那一份而影响其他成员。
        if "fit_model" in cls.Params.model_fields:
            if not cls.distorting_models:
                raise TypeError(f"{cls.__name__}: a node with a fit_model must declare `distorting_models` (which of "
                                f"its choices really solve a distortion), worked out from its own model table; without "
                                f"it the 镜头内参 output stays live on the choices that give nothing "
                                f"(lab2shot/nodes/families/lens_calibration.py)")
            cls.outputs = only_when_distorting(cls.outputs, *cls.distorting_models)
        super().__init_subclass__(**kw)  # 条件施加完成后再由 NodeDef 检查声明：check_declarations 需核对条件中的参数名是否存在
