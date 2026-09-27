"""Nodes provided by the StableMotion extension."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, Cost, DetectCleanupParams, JointMap, Licence, Measured, RigMotion, P, body_joints,
                          mapping_param, measured_param)

# StableMotion's skeleton is SMPL's, so the joint table is SMPL's, read off the body itself (nodes/kit/rig.py):
# the same 22 joints and the same body parts 「骨架动画转 SMPL」 pairs up, listed here so they can be corrected by hand.
JOINTS = body_joints("smpl")
MODEL_FPS = 20.0  # the model's own frame rate (BrokenAMASS is resampled to 20 fps)
WINDOW = 100  # frames it was trained on, at its own rate: 5 seconds


class StableMotionCleanup(RigMotion):
    id = "stablemotion.cleanup"
    does = "cleanup"  # 修动作（骨骼动作家族的两件活之一，lab2shot/nodes/families/rig_motion.py）
    # 引的是官方 fix_globsmpl.py 的修复那一遍：input_motions（坏动作）+ label（上一遍 detect_labels 判出来的
    # 问题帧，同文件 105-117）进去，fixed_motion 出来。
    official = Official(
        cite="third_party/stablemotion/repo/sample/fix_globsmpl.py:121-239",
        takes={"character": "input_motions"},
        gives={"character": "fixed_motion", "labels": "label"},
        note="「问题帧」曲线就是官方的 label：`detect_labels`（同文件 33-117）读第 233 个通道判每一帧好坏，"
             "out['label'] 再喂给 fix_motion。两遍都是官方 README 的那两条命令。",
    )
    # 不读画面，整段一起看；模型一次看 100 帧（自己的 20 帧/秒，5 秒），更长的分段重叠处理
    runtime = "stablemotion"
    licence = Licence(note="仅限研究：代码是 MIT，但放出来的权重在 AMASS 上训练，AMASS 只许非商业的学术研究。"
                           "要商用得按官方说明拿自己的动捕重训一个模型。")
    joints = JOINTS
    detects = ("问题帧",)  # 它逐帧判断好坏：家族因此给它「问题帧」输出口，并让「只改问题帧」有意义
    # RTX 4090：60 秒的动捕 3.5 秒、215 MB 显存；判坏的帧越多，要重画的段越多
    cost = Cost(gpu=True, vram_gb=0.3, whole="整段一次找问题帧，只有判坏的地方才按 5 秒一段重画")

    class Params(DetectCleanupParams):
        mapping: list[JointMap] = mapping_param(JOINTS)
        quality: Literal["basic", "best"] = measured_param(
            "质量", {"basic": Measured("官方基本推理：一次检测、一次重画", flat=True),
                     "best": Measured("官方增强推理：多次采样挑最好的一版，慢很多", flat=True)},
            default="basic", group="模型",
            option_labels={"basic": "基本", "best": "增强"},
            help="基本：官方 README 的第一条命令，检测一遍、修一遍。"
                 "增强：官方 README 的第二条命令（多次采样取平均来定问题帧，再多采几版按脚滑挑最好的一版，"
                 "并在去噪时用脚锁引导），论文的主结果就是这一档；慢很多，修不干净的镜头再用它")
        seed: int = P(10, label="随机种子", group="模型", ge=0,
                      help="扩散模型每次重画的结果不同，同一个种子得到同一个结果。修出来的那几帧不满意就换个数字重算。"
                           "10 是官方默认")


NODES = (StableMotionCleanup,)
