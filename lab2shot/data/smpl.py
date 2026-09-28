"""SMPL 家族的关节名称与身体部位：将 rig 的骨骼名与 SMPL 的关节对应起来。

本模块不包含数学计算。SMPL 家族的骨架、参数与关节变换之间的换算由 `lab2shot_shared.smpl` 实现
（核心与每个 worker 共用同一份实现），rig 与模型骨架不一致时的重定向由 `lab2shot_shared.motion.Retarget` 实现。
本模块仅做名称和层级的匹配：两侧的身体部位均由 `data/joints.py guess()` 识别，
生成 `Retarget.align` 所需的三项（pairs / aims / legs），并按身体区域向使用者报告配对结果。

SMPL / SMPL-X / MANO / FLAME / MHR 本质上是「蒙皮 + 权重 + 骨架动画（+ 表情）」，即已有的「蒙皮角色」，
不另设数据类型；本模块的函数提供 `nodes/kit/rig.py` 使用的关节表和配对。
"""

from __future__ import annotations


import numpy as np

from ..errors import Invalid
from ..messages import Msg

# 支持的身体仅限 SMPL 家族的三种（lab2shot_shared.smpl 的 BODIES，契约据此检查）。MANO 的手和 FLAME 的脸
# 各为模型的一部分，将其组合回身体不在本模块处理
FILE = "smpl.npz"


def aims_of(parents) -> list[int | None]:
    """Retarget.align 所需的 `aims`：每个模型关节的骨骼指向哪个关节。只有一个子关节时指向该子关节；
    分叉关节（骨盆、胸、SMPL-X 的头）和末端关节为 None，由 Retarget 沿用父关节的对齐。
    该结果仅由层级决定，不涉及姿势计算。"""
    kids: dict[int, list[int]] = {}
    for j, up in enumerate(parents):
        kids.setdefault(int(up), []).append(j)
    return [kids[j][0] if len(kids.get(j, ())) == 1 else None for j in range(len(parents))]


LEG_PARTS = ("l.thigh", "l.shin", "r.thigh", "r.shin")


# 向使用者报告推测的配对时按身体区域分行（每行一个区域，比每行一个关节的二十多行更清晰）。
# 部位 id 即 data/joints.py 的 PARTS，中文名也只在该处定义；约定 PARTS 中的每个部位都属于某个区域，
# 因此 joints.py 新增部位时不会在此遗漏。
REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("躯干", ("hips", "spine", "chest", "neck", "head")),
    ("面部", ("jaw", "l.eye", "r.eye")),
    ("左臂", ("l.clavicle", "l.upperarm", "l.forearm", "l.hand")),
    ("右臂", ("r.clavicle", "r.upperarm", "r.forearm", "r.hand")),
    ("左腿", ("l.thigh", "l.shin", "l.foot", "l.toe")),
    ("右腿", ("r.thigh", "r.shin", "r.foot", "r.toe")),
    ("左手", ("l.thumb", "l.index", "l.middle", "l.ring", "l.pinky")),
    ("右手", ("r.thumb", "r.index", "r.middle", "r.ring", "r.pinky")),
)


def _driving(names: list[str]) -> str:
    """将驱动某个部位的 rig 关节格式化为一段文字。成链的部位（脊柱、颈、每根手指）可能对应多个关节：
    不超过两个时全部列出（脊柱←Spine/Spine1，可看出使用了哪两节），更多时只写第一个并注明节数
    （左拇指←LeftHandThumb1（3 节））；若列出双手十根手指的全名，单行将超过两百字，难以阅读。"""
    return "/".join(names) if len(names) <= 2 else f"{names[0]}（{len(names)} 节）"


def pairing_notes(rows: list[tuple[str, str]], mapped: dict) -> list[Msg]:
    """将推测的配对转换为节点上显示的消息：首先说明配对为推测结果，然后每个区域一行
    「左臂：左锁骨←LeftShoulder、左上臂←LeftArm、左前臂←LeftForeArm、左手←LeftHand」，最后一行列出未配对的部位。

    必须显示的原因：配对完全依赖 `data/joints.py guess()` 按名称和层级推测，目前无法在节点上修改
    （关节映射表的行数随接入的身体变化，现有表格控件的行数固定）。若不显示推测结果，使用者只能看到
    扭曲的动作而无法得知哪个关节配错。显示后，使用者可以对照骨骼名找出错误，并回到 DCC 中修正名称。
    """
    from .joints import part_label

    driven: dict[str, list[str]] = {}  # 部位 -> 驱动它的 rig 关节（成链的部位按层级顺序可有多个）
    for name, part in rows:
        if mapped.get(name):
            driven.setdefault(part, []).append(str(mapped[name]))
    have = {part for _, part in rows}  # 该身体实际具有的部位（SMPL 没有手指和脸，不应报告为未配对）
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
    这是唯一的一份：自动配对（`match`）与节点上可编辑的关节映射表（nodes/kit/rig.py body_joints）读取同一张表，
    因此表格列出的行与自动推测的结果一致。"""
    from .joints import guess

    parts = guess(list(body.names), np.asarray(body.parents, np.int64))
    rows: list[tuple[str, str]] = []
    for part, where in parts.items():
        rows += [(body.names[j], part) for j in (where if isinstance(where, list) else [where])]
    return rows

