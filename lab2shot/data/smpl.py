"""SMPL 家族的关节名称与身体部位：将 rig 的骨骼名与 SMPL 的关节对应起来。

本模块不包含数学计算。SMPL 家族的骨架、参数与关节变换之间的换算由 `lab2shot_shared.smpl` 实现
（核心与每个 worker 共用同一份实现），rig 与模型骨架不一致时的重定向由 `lab2shot_shared.motion.Retarget` 实现。
本模块仅做名称和层级的匹配：两侧的身体部位均由 `data/joints.py guess()` 识别，
并按身体区域向使用者报告配对结果（对齐所需的 aims 只看层级，在 `lab2shot_shared.motion.aims_toward`）。

SMPL / SMPL-X / MANO / FLAME / MHR 本质上是「蒙皮 + 权重 + 骨架动画（+ 表情）」，即已有的「蒙皮角色」，
不另设数据类型；本模块的函数提供 `nodes/kit/rig.py` 使用的关节表和配对。
"""

from __future__ import annotations


import numpy as np

from ..errors import Invalid
from ..messages import Msg



def _driving(names: list[str]) -> str:
    """将驱动某个部位的 rig 关节格式化为一段文字。成链的部位（脊柱、颈、每根手指）可能对应多个关节：
    不超过两个时全部列出（脊柱←Spine/Spine1，可看出使用了哪两节），更多时只写第一个并注明节数
    （左拇指←LeftHandThumb1（3 节））；若列出双手十根手指的全名，单行将超过两百字，难以阅读。"""
    return "/".join(names) if len(names) <= 2 else f"{names[0]}（{len(names)} 节）"


def pairing_notes(driven: dict[str, list[str]], have: set[str]) -> list[Msg]:
    """将推测的配对转换为节点上显示的消息（按身体区域分行，比每行一个关节的二十多行更清晰；区域表只有一份，
    在 data/joints.py REGIONS）：首先说明配对为推测结果，然后每个区域一行
    「左臂：左锁骨←LeftShoulder、左上臂←LeftArm、左前臂←LeftForeArm、左手←LeftHand」，最后一行列出未配对的部位。
    `driven`：推测出来的部位 -> 驱动它的关节名（成链的部位按层级顺序可有多个）；`have`：被驱动的那副骨架
    实际具有的推测部位（SMPL 没有手指和脸，不应报告为未配对）。只报告推测的部分：使用者在「对应关系」里
    亲手指定的部位不再重复说明。

    必须显示的原因：推测依赖 `data/joints.py guess()` 按名称和层级判断，可能猜错；显示后使用者能对照骨骼名
    找出错误，在「对应关系 · 编辑…」里改。
    """
    from .joints import REGIONS, part_label

    notes, missing = [Msg("I-SMPL-GUESS")], []
    for region, parts in REGIONS:
        said = [f"{part_label(p)}←{_driving(driven[p])}" for p in parts if p in driven]
        missing += [part_label(p) for p in parts if p in have and p not in driven]
        if said:
            notes.append(Msg("I-SMPL-PAIRED", region=region, pairs="、".join(said)))
    if missing:
        notes.append(Msg("I-SMPL-UNPAIRED", parts="、".join(missing)))
    return notes


def rows_of(body) -> list[tuple[str, str]]:
    """SMPL 家族身体的每个关节所属的部位：[(模型关节名, 部位)]，成链的部位（脊柱、颈、手指）按层级顺序排列。
    这是唯一的一份：自动配对与节点上可编辑的「对应关系」（nodes/kit/rig.py body_joints）读取同一张表，
    因此编辑器里模型那一侧列出的关节与自动推测的结果一致。"""
    from .joints import guess

    parts = guess(list(body.names), np.asarray(body.parents, np.int64))
    rows: list[tuple[str, str]] = []
    for part, where in parts.items():
        rows += [(body.names[j], part) for j in (where if isinstance(where, list) else [where])]
    return rows

