"""GeoCalib 的「镜头内参组」：它自己的三个相机模型（third_party/geocalib/repo/siclib/geometry/camera.py：Pinhole、
SimpleRadial 1 + k1 r²、SimpleDivisional 1 / (1 + k1 r²)）→ 核心公式表（nodes/lens.py LensGroup）。"""

from __future__ import annotations

from types import MappingProxyType

from lab2shot.sdk import GroupModel, LensGroup

GROUP = LensGroup("geocalib", "GeoCalib", MappingProxyType({m.name: m for m in (
    GroupModel("pinhole", "pinhole（无畸变）", "SIMPLE_PINHOLE", ()),
    GroupModel("simple_radial", "simple_radial（径向 k1）", "SIMPLE_RADIAL", ("k",)),
    GroupModel("simple_divisional", "simple_divisional（除法模型 k1，鱼眼）", "SIMPLE_DIVISION", ("k",)),
)}))
