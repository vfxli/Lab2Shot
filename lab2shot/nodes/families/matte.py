"""抠像：从画面中分离主体。

家族层统一处理两项内容，成员无需各自实现：

* Alpha：每个抠像模型都会输出，是本家族的主要结果；
* 前景：去除背景颜色后的前景色（解混，预乘 alpha）。只有官方模型本身计算前景的项目才有该端口
  （`Matting.foreground`）；只输出 alpha 的项目没有该端口，不以 alpha 乘原图的方式伪造前景。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..kit.ports import rgb_port
from ...data.packet import Packet
from ...data.payloads import UNIT
from ..applies import Cond, Cost
from ..base import Port
from .base import Job, MissingFrames, RawOutput, WorkerNode
from ..kit.maps import frame_maps, model_picture
from ..kit.ports import plate_mask_port

def foreground_port(when: Cond | None = None) -> Port:
    """抠像家族的「前景」输出端口（由 `Matting.foreground` 提供）。`when`：仅在某参数开启时存在该端口。"""
    # 说明是家族的（family.matte.foreground.help）；各节点的 waits 在 node.<类型>.port.foreground.waits
    return Port("foreground", "image.4", alpha=True, when=when, words="family.matte.foreground")


def foreground_entry(array: str = "foreground", alpha: str = "alpha") -> tuple:
    """写出「前景」的 frame_maps 条目：模型给出显示 sRGB、已预乘 alpha 的前景色 [H,W,3]，写为一张预乘的
    RGBA EXR（与 Nuke 一样使用预乘，data/types.py「图像序列」）。sRGB 即工作色彩空间（io/color.py），像素原样写出，
    不做任何色彩空间转换。"""
    convert, options = model_picture()

    def rgba(d):
        a = np.clip(np.asarray(d[alpha], np.float32), 0.0, 1.0)
        return np.concatenate([convert(np.asarray(d[array], np.float32)), a[..., None]], axis=-1)

    return "image.4", rgba, {**options, "alpha": True}


class Matting:
    """抠像节点的共同声明。`foreground`：官方模型除 alpha 外是否计算前景色。为 True 时增加「前景」输出端口，
    也可写为参数条件（Sapiens2 只有开启「精细抠像」才使用抠像模型）。默认为 False：只输出 alpha 的项目没有该端口。

    端口在此处添加，节点只需一行声明；写出由 foreground_entry() 统一实现。"""

    foreground: ClassVar[bool | Cond] = False

    def __init_subclass__(cls, **kw):
        if cls.foreground is not False and not any(p.name == "foreground" for p in cls.outputs):
            when = cls.foreground if isinstance(cls.foreground, Cond) else None
            cls.outputs = (*cls.outputs, foreground_port(when))
        super().__init_subclass__(**kw)


def matte(ctx, raw: RawOutput, image: Packet) -> dict[str, Packet]:
    """raw/frame_<n>.npz：alpha [H,W]，取值 0..1，分辨率与输入相同；同时预测前景色的模型还输出 foreground [H,W,3]
    （预乘 sRGB）。frame_maps 只写出有需要的内容。"""
    # alpha 取值为 0..1：显式声明显示范围，避免视图按 1%–99% 分位推测（一张 0.2–0.8 的 alpha 会被拉伸为 0–1 显示）。
    # `matte`：该通道是抠像的 alpha。接线时不考虑此标记（单通道即单通道），使用时才考虑：
    # 视图默认将其叠加在上游原图上，交付时写入 EXR 的 a 通道而不是 R。
    maps = {"alpha": ("image.1", "alpha", {"value_range": UNIT, "half": True, "matte": True})}
    stage = "write_alpha"
    if "foreground" in ctx.wanted:  # 仅在需要时才构建色彩转换（frame_maps 也只写出需要的端口）
        maps["foreground"] = foreground_entry()
        stage = "write_alpha_foreground"
    return frame_maps(ctx, raw, image, maps, stage=stage)


class MatteNode(Matting, WorkerNode):
    """本家族的形态：输入一段画面 → 输出一张软边 alpha。写出由 matte() 统一实现
    （worker 侧为 lab2shot_worker 的 matte 约定）。Job.notes：无。

    分为两档，区别只在于上游是否需要使用者先提供一张粗遮罩：

    | 档 | 上游输入 | 成员 |
    |---|---|---|
    | `MatteNode`（本类） | 只接受画面，由模型自行判断主体 | BiRefNet |
    | `GuidedMatte` | 画面 + 一张粗遮罩，以其为起点细化 | MatAnyone 2、SDMatte、VideoMaMa |

    `sapiens2.segment` 只混入 `Matting` 声明（有前景端口，没有粗遮罩端口），不属于这两档。"""

    inputs = (rgb_port(),)
    outputs = (Port("alpha", "image.1"),)
    cost = Cost(gpu=True)
    missing_frames = MissingFrames.SKIP

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        return matte(ctx, raw, job.plate)


class GuidedMatte(MatteNode):
    """有引导档：输入画面 + 一张粗遮罩（由 BiRefNet、SAM 3 提供），输出带发丝细节的软 alpha。

    `every_frame`：模型是否读取每一帧的遮罩（逐帧细化器、逐帧引导的模型），还是只需起始处的遮罩（MatAnyone 2
    从第一张含前景的遮罩开始传播）。这是这些节点输入声明之间唯一的差异，因此端口在此处据此构建，
    而不是在各适配器中重复声明。"""

    every_frame: ClassVar[bool] = True
    on_node = ("resolution", "erode_dilate")
    inputs = (rgb_port(), plate_mask_port(optional=False, words="family.matte.mask"))
    streams = True  # 逐帧写出 EXR（frame_maps）：每帧写完即为最终字节，可边算边看

    def __init_subclass__(cls, **kw):
        if "inputs" not in cls.__dict__:  # 需要自有端口的节点仍自行声明
            cls.inputs = (rgb_port(),
                          plate_mask_port(optional=False, every_frame=cls.every_frame, words="family.matte.mask"))
        super().__init_subclass__(**kw)

    @classmethod
    def prepare(cls, ctx) -> Job:
        return Job(ctx.input("image"), inputs=ctx.input_files("mask"))
