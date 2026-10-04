"""AnyCalib 的「镜头内参组」：将 AnyCalib 自身的相机模型名（cam_id）映射到核心公式表中的模型。与 COLMAP 相同，参数按
公式表模型加系数的形式输出，「LensDistortion」使用同一张表接收。

官方能拟合的模型（third_party/anycalib/repo/anycalib/cameras/factory.py:11-27，README 129-145）：pinhole / simple_pinhole、
radial:k / simple_radial:k（Brown 多项式 k1..k4）、kb:k / simple_kb:k（Kannala-Brandt θ 多项式）、ucm / simple_ucm、
eucm / simple_eucm、division:k / simple_division:k；fov 仅有投影、不支持拟合。本模块仅提供公式与 COLMAP 模型严格一致的常用模型
（枕形 / 桶形畸变和鱼眼）：

| cam_id | 公式表 | 系数 |
|---|---|---|
| simple_pinhole | SIMPLE_PINHOLE | 无 |
| simple_radial:1 | SIMPLE_RADIAL | k |
| simple_radial:2 | RADIAL | k1 k2 |
| simple_kb:1 | SIMPLE_RADIAL_FISHEYE | k |
| simple_kb:2 | RADIAL_FISHEYE | k1 k2 |
| simple_kb:3 | OPENCV_FISHEYE | k1 k2 k3（k4 = 0） |
| simple_kb:4 | OPENCV_FISHEYE | k1 k2 k3 k4 |
| simple_division:1 | SIMPLE_DIVISION | k |

未提供的模型及原因：pinhole / radial:k / kb:k / division:k（fx ≠ fy 的双焦距版本：镜头表只有一个焦距加像素比，
对应关系不严格）；radial:3（对应 FULL_OPENCV，拟合会发散）；simple_radial:3 / :4、simple_kb 之外的四项
（公式表没有对应槽位）；ucm / simple_ucm（表里没有这个公式）；eucm / simple_eucm（表里有 EUCM 公式，但 AnyCalib 的 α β
和表的定义未逐项核对）；division:k ≥ 2；fov（官方不能拟合）。新增模型时，应先核对公式，再添加对应行。
"""

from __future__ import annotations

from types import MappingProxyType

from lab2shot.sdk import GroupModel, LensGroup

GROUP = LensGroup("anycalib", "AnyCalib", MappingProxyType({m.name: m for m in (
    GroupModel("simple_pinhole", "SIMPLE_PINHOLE", ()),
    GroupModel("simple_radial:1", "SIMPLE_RADIAL", ("k",)),
    GroupModel("simple_radial:2", "RADIAL", ("k1", "k2")),
    GroupModel("simple_kb:1", "SIMPLE_RADIAL_FISHEYE", ("k",)),
    GroupModel("simple_kb:2", "RADIAL_FISHEYE", ("k1", "k2")),
    GroupModel("simple_kb:3", "OPENCV_FISHEYE", ("k1", "k2", "k3")),
    GroupModel("simple_kb:4", "OPENCV_FISHEYE", ("k1", "k2", "k3", "k4")),
    GroupModel("simple_division:1", "SIMPLE_DIVISION", ("k",)),
)}))
