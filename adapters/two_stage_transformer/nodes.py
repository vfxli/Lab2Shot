"""Two-stage Transformer 扩展提供的节点（LaFAN1 权重，仅限研究用途）。"""

from __future__ import annotations

from lab2shot.sdk import Official, RigMotion, MotionGenParams, JointMap, P, humanoid_joints, mapping_param, Cost, Licence

# LaFAN1 的 22 个关节，均可由骨骼绑定的关节驱动（模型骨架取自其 BVH 文件）
JOINTS = humanoid_joints((("Hips", "hips", "Spine"), ("Spine", "spine", "Spine1"), ("Spine1", "spine", "Spine2"),
                          ("Spine2", "chest", "Neck"), ("Neck", "neck", "Head"), ("Head", "head", "up")),
                         legs=("UpLeg", "Leg", "Foot", "Toe"))


class TSTInbetween(RigMotion):
    id = "two_stage_transformer.inbetween"
    does = "generate"  # 生成动作（骨骼动作家族的两类任务之一，lab2shot/nodes/families/rig_motion.py）
    # 引用官方 Detail Transformer 的推理函数：输入关键帧的 positions / rotations / foot_contact，
    # 输出补间后的 pos_new / rot_new / c_out（eval_detail_model.py 使用同一路径）。
    official = Official(
        cite="third_party/two_stage_transformer/repo/packages/motion_inbetween/train/detail_model.py:515-584",
        takes={"character": "positions"},
        gives={"character": "pos_new"},
        note="接进来的动画在上游是 positions + rotations 一对（同一段 515-520），补出来的是 pos_new + rot_new "
             "一对（584）：一个「人物」口对应的就是这一对，不是两样东西。post_process 是官方的参数（517）。",
    )
    # 不声明 `unconstrained`：该模型以相邻两个关键帧之间为一段（见 worker.py 模块说明），
    # 没有关键帧时无法划分任何一段。因此「动画」口仍为必需输入，引擎在计划阶段拒绝未接入的情况。
    # docs.md：模型单次最多处理 65 帧（30 帧/秒），相邻关键帧间隔不超过 1.8 秒；更长的间隔须增加关键帧或改用 Kimodo
    runtime = "two_stage_transformer"
    # vram_gb：在 RTX 4090 上测得（docs.md）
    cost = Cost(gpu=True, vram_gb=0.2, whole="毫秒级、一整段一起补，不是逐帧的活")
    licence = Licence(note="Two-stage Transformer 的官方权重用 Ubisoft LaFAN1 动捕数据（CC BY-NC-ND 4.0）训练，补出来的动作只能用于研究和评估。")
    joints = JOINTS

    class Params(MotionGenParams):
        mapping: list[JointMap] = mapping_param(JOINTS)
        post_process: bool = P(True, label="衔接平滑", group="模型",
                               help="Two-stage Transformer 自带的后处理：把补出的曲线整体微调，让它在前一个关键帧之后和后一个关键帧之前的走势"
                                    "（速度）接得更顺。一般保持打开")


NODES = (TSTInbetween,)
