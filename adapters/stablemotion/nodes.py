"""Nodes provided by the StableMotion extension."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Cost, DetectCleanupParams, PartMap, Licence, Measured, RigMotion, P, body_joints,
                          mapping_param, measured_param)

# StableMotion's skeleton is SMPL's, so its joints are SMPL's, read off the body itself (nodes/kit/rig.py):
# the same 22 joints and body parts the model side of 「对应关系」 shows, the person's joints driving them chosen there.
JOINTS = body_joints("smpl")


class StableMotionCleanup(RigMotion):
    id = "stablemotion.cleanup"
    version = 5  # 5：只为让半途代码算出的 4 的缓存重算；4：两副站着的基准姿势时躯干（根、脊柱、胸、颈、头）整体按根的对齐，不再逐骨对准（MHR 的人体曾被对成后仰 18°）；3：判坏帧按官方 100 帧窗口；回填用扩张后的坏帧；2：帧号按节点的「帧率」换算成时间（默认 24 与 1 版的固定时基相同）
    does = "cleanup"  # 修动作（骨骼动作家族的两件活之一，lab2shot/nodes/families/rig_motion.py）
    # 引的是官方 fix_globsmpl.py 的修复那一遍：input_motions（坏动作）+ label（上一遍 detect_labels 判出来的
    # 问题帧，同文件 105-117）进去，fixed_motion 出来。
    official = Official(
        cite="third_party/stablemotion/repo/sample/fix_globsmpl.py:121-239",
        takes={"character": "input_motions"},
        gives={"character": "fixed_motion", "labels": "label"},
    )
    # 不读画面，整段一起看；模型一次看 100 帧（自己的 20 帧/秒，5 秒），更长的分段重叠处理
    runtime = "stablemotion"
    licence = Licence(note=True)
    joints = JOINTS
    detects = ("problem_frames",)  # 它逐帧判断好坏：家族因此给它「问题帧」输出口，并让「只改问题帧」有意义
    # RTX 4090：60 秒的动捕 3.5 秒、215 MB 显存；判坏的帧越多，要重画的段越多
    cost = Cost(gpu=True, vram_gb=0.3, whole=True)

    class Params(DetectCleanupParams):
        mapping: list[PartMap] | None = mapping_param()
        quality: Literal["basic", "best"] = measured_param(
            {"basic": Measured(flat=True),
                     "best": Measured(flat=True)},
            default="basic", group="model")
        seed: int = P(10, group="model", ge=0)


NODES = (StableMotionCleanup,)
