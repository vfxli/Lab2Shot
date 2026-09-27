"""家族和节点用的零件：端口、参数、结果图、相机、置信度、骨架对照表，以及算法模块。

算法模块：对齐（align）、去抖（deshake）、锁定 Focal Length（lock_focal）、重定时（retime）、
重定向（retarget）、UV 投影（projection）、方向场（orientation）、栅格化（raster）、地面和放平（ground）、
透过相机看场景（render）、深度反投影（unproject）。`lab2shot/data/` 只放数据的定义、契约和包的读写；
把一堆数变成另一堆数的计算都在这里。

这些不是家族。家族是「一类活」（`lab2shot/nodes/families/`：抠像、深度、人体动作……），
每个家族至少有一个节点继承它；这里的模块没有节点继承，是家族拼起来用的零件。
零件放在 `families/` 里会让对外报的家族数量失真，所以单独一个目录：
`families/` 里每个类都要有节点继承它，零件不放在那里。
"""

from __future__ import annotations
