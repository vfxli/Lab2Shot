"""COLMAP 的「镜头内参组」：清单就是「COLMAP 相机解算」节点上提供的那几档（nodes.py OFFERED 从这里取，
「LensDistortion」上 COLMAP 那一组也从这里取，两头一致）。模型名就是核心公式表的 id（COLMAP 官方名）。

提供哪几档、为什么只有这几档，见 nodes.py 里 PINHOLES 下面那段注释。"""

from __future__ import annotations

from lab2shot.sdk import LENS_TABLE, COLMAP_MODELS, core_group

# 在真值素材上验证过能解出来的模型；FULL_OPENCV 会发散，其余 8 个没验证过的不提供
TESTED = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV",
          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE")
# default: 「LensDistortion」 starts on this group (nodes/lens.py lens_groups lists it first)
GROUP = core_group("colmap", "COLMAP", tuple(m for m in COLMAP_MODELS if m in LENS_TABLE and m in TESTED), default=True)
