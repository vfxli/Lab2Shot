"""分割家族：画面 → 选中物体的 0/1 遮罩，加一张按物体编号的物体分割图。

家族层统一三件事，成员不各写一遍：

* 两个输出口：`mask`（所有选中物体合成一张 0/1 遮罩）和 `objects`（背景 0、第 k 个物体 k，
  相当于 Object ID pass）。口名、类型和 `objects` 的标签由家族定；同一个口名 + 同一个类型，标签必须一致。
* `mask` 永远是 `objects > 0`，两张图不各算各的。
* 写出：读 `objects.json`、把类别表（Cryptomatte 清单）挂在编号图上；编号图带 `classes` 时
  `frame_maps` 按最近邻重采样，不做插值（kit/maps.py `_entry`）。

成员只需给出：每帧原始结果怎么变成编号图（`label_map`）、物体叫什么（`class_list`）、
找到 / 没找到时报哪条消息（`found`）。

这个家族不是抠像（`families/matte.py`）：那边出软边 alpha（发丝、半透明边缘），
这边出 0/1 选区。
"""

from __future__ import annotations

import json
from typing import ClassVar

import numpy as np

from ..kit.ports import rgb_port
from ...data.packet import Packet
from ..applies import Cost
from ..base import Port
from ..kit.maps import frame_maps
from .base import MissingFrames, RawOutput, WorkerNode

OBJECTS_FILE = "objects.json"  # 每个成员的 worker 都写这一份：[{"id": …}, …]，物体的顺序就是编号的顺序
# 同一个口名 + 同一个类型只许一个标签。「运动」之类的限定词属于节点名（如「SegAnyMo 运动物体遮罩」），
# 不属于口：口的标签只说这个口上是什么数据
OBJECTS_LABEL = "物体分割"


class Segmentation(WorkerNode):
    """分割节点的共同声明：画面进，遮罩 + 物体分割图出。

    Raw contract：worker 写 `raw/objects.json`（物体表，顺序就是编号顺序）和每帧一个 npz；
    那个 npz 里装什么由项目自己说（SAM 3 是 `masks` [K,H,W]，SegAnyMo 是 `labels` [H,W]），
    `label_map()` 把它变成这一家统一的编号图。

    `mask_label`：遮罩口的标签（这个成员选出来的是什么）。口名 `mask` 和类型由家族定，
    标签按成员说：它描述这个口要接什么，各成员不同是正常的。

    `mask_half`：遮罩写成半精度并声明值域 0..1。成员之间目前不一致（SegAnyMo 是、SAM 3 不是），
    按成员声明；改它会改掉已交付 EXR 的位深。
    """

    mask_label: ClassVar[str] = "遮罩"
    mask_half: ClassVar[bool] = False
    inputs = (rgb_port(),)
    outputs = ()  # 每个成员在 __init_subclass__ 里按 mask_label 建；口名、类型、objects 的标签是家族给的
    cost = Cost(gpu=True)
    missing_frames = MissingFrames.SKIP

    def __init_subclass__(cls, **kw):
        if "outputs" not in cls.__dict__:
            cls.outputs = (Port("mask", "image.1", cls.mask_label), Port("objects", "image.1", OBJECTS_LABEL))
        super().__init_subclass__(**kw)

    @classmethod
    def found(cls, ctx, raw: RawOutput, objects: list) -> None:
        """报告找到了什么（消息编号由成员定）。空结果不算错误。"""

    @classmethod
    def class_list(cls, ctx, objects: list) -> list[dict]:
        """物体分割图的类别表（Cryptomatte 清单）：每个物体一条 {"index", "name", …}。"""
        raise NotImplementedError(f"{cls.id}: class_list() does not say what each object is called")

    @classmethod
    def label_map(cls, ctx, image: Packet, objects: list):
        """(一帧的 npz) -> 编号图 [H,W] float32：背景 0，第 k 个物体一个大于 0 的编号。"""
        raise NotImplementedError(f"{cls.id}: label_map() does not say how a frame's raw result becomes a label map")

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job) -> dict[str, Packet]:
        from ...data.payloads import UNIT

        objects = json.loads(raw.file(OBJECTS_FILE).read_text(encoding="utf-8"))
        cls.found(ctx, raw, objects)
        labels = cls.label_map(ctx, job.plate, objects)
        return frame_maps(ctx, raw, job.plate, {
            # 遮罩直接由编号图导出，不再读一遍原始结果
            "mask": ("image.1", lambda d: (labels(d) > 0).astype(np.float32),
                     {"value_range": UNIT, "half": True} if cls.mask_half else None),
            "objects": ("image.1", labels, {"value_range": (0, max(len(objects), 1)),
                                            "classes": cls.class_list(ctx, objects)}),
        }, stage="写出遮罩")
