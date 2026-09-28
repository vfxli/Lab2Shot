"""动作重定向：将一段骨架动画迁移到另一副骨架上，包括两副 rig 的关节配对与动作转换。

静止姿势对齐、骨骼方向与扭转、腿长比例等数学计算均由 `lab2shot_shared.motion.Retarget` 完成
（核心与各 worker 共用同一实现）。本模块仅负责两项工作：

  1. 配对：两副 rig 的身体部位均由 `data/joints.py guess()` 按名称与层级识别，同一部位即配为一对；
     链状部位（脊椎、颈部、各手指）按在链上的相对位置分配（joints.spread）。
  2. 转换：`Retarget.to_model` 给出目标 rig 各关节的世界旋转与根位置；本模块按目标自身的骨骼长度
     做一次正向运动学，得到逐帧的 joint-to-parent 矩阵，交由 `data/animation.py write_animation`
     写成新的数据包。

动作重定向是「提取骨架」（蒙皮角色 → 骨架动画）的逆向步骤，不依赖第三方解算器，因此作为核心节点实现。
"""

from __future__ import annotations

import numpy as np

from ...errors import Invalid
from ...messages import Msg


def pair_rigs(target, source) -> tuple[list[tuple[int, int]], list, tuple[int, int, int, int], list[Msg]]:
    """两副 rig 的关节配对，返回 (pairs[(目标关节, 来源关节)], aims, legs, 面向使用者的配对说明)。

    `target` / `source` 均为 `data/animation.py Rig`。配对以身体部位为单位：两侧分别由 joints.guess() 识别
    髋、脊椎、颈部、头、四肢与手指，同一部位配为一对；链状部位按位置分配。
    目标 rig 无法识别髋或四段腿骨时抛出 E-RETARGET-MISSINGPARTS：Retarget 依据腿部确定朝向与长度比例，缺失则无法对齐。"""
    from ...data.joints import CHAINS, guess, part_label, spread
    from ...data.smpl import LEG_PARTS

    a = guess(list(target.names), target.parents)
    b = guess(list(source.names), source.parents)
    pairs: list[tuple[int, int]] = []
    rows: list[tuple[str, str]] = []  # (目标关节名, 部位)；生成配对说明时按部位分组
    mapped: dict[str, str | None] = {}
    for part in sorted(set(a) | set(b)):
        ja, jb = a.get(part), b.get(part)
        if part in CHAINS:
            chain_a = list(ja or [])
            picks = spread(len(chain_a), list(jb or []))
            for m, p in zip(chain_a, picks):
                rows.append((target.names[m], part))
                mapped[target.names[m]] = None if p is None else source.names[p]
                if p is not None:
                    pairs.append((int(m), int(p)))
        elif isinstance(ja, int):
            rows.append((target.names[ja], part))
            mapped[target.names[ja]] = None if not isinstance(jb, int) else source.names[jb]
            if isinstance(jb, int):
                pairs.append((int(ja), int(jb)))
    missing = [part_label(p) for p in ("hips", *LEG_PARTS) if not isinstance(a.get(p), int)]
    if missing:
        raise Invalid(Msg("E-RETARGET-MISSINGPARTS", parts=missing, rig="目标"))
    if not any(m == a["hips"] for m, _ in pairs):
        raise Invalid(Msg("E-RETARGET-MISSINGPARTS", parts=[part_label("hips")], rig="来源"))
    from ...data.smpl import aims_of, pairing_notes

    legs = tuple(int(a[p]) for p in LEG_PARTS)
    return sorted(pairs), aims_of(target.parents), legs, pairing_notes(rows, mapped)


def locals_from_world_rotations(target, rotations: np.ndarray, root: np.ndarray) -> np.ndarray:
    """目标 rig 每帧的 joint-to-parent [K,J,4,4]。根关节采用给定的旋转与位置；其余关节保留静止姿势中的
    骨骼偏移（长度不变），仅替换旋转。`rotations` [K,J,3,3] 为世界旋转，`root` [K,3] 为根关节的世界位置。"""
    from lab2shot_shared import motion as mo

    parents = np.asarray(target.parents, np.int64)
    rest_local = mo.local_from_world(np.asarray(target.bind, np.float64), parents)  # [J,4,4]，提供骨骼偏移

    k, j = len(rotations), len(parents)
    out = np.zeros((k, j, 4, 4))
    out[..., 3, 3] = 1.0
    out[:, :, :3, 3] = rest_local[None, :, :3, 3]
    for m in range(j):
        up = int(parents[m])
        out[:, m, :3, :3] = rotations[:, m] if up < 0 else rotations[:, up].transpose(0, 2, 1) @ rotations[:, m]
    root_joints = [m for m in range(j) if parents[m] < 0]
    for m in root_joints:
        out[:, m, :3, 3] = root
    return out
