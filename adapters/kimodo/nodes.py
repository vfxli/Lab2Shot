"""Kimodo 扩展提供的节点。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, RigMotion, FreeMotionParams, MissingFrames, RawOutput,
                          JointMap, P, Param, Wired, Because, body_joints, humanoid_joints, mapping_param, Cost, OptionTrait,
                          Measured, ModelJoint)

# SOMA 的 30 关节骨架（kimodo.skeleton.SOMASkeleton30），去掉末端点（下巴、眼睛、指尖跟随父关节），即骨骼绑定驱动的关节。
# SOMA 将大腿称为 "Leg"，小腿称为 "Shin"。
SOMA = humanoid_joints((("Hips", "hips", "Spine1"), ("Spine1", "spine", "Spine2"), ("Spine2", "spine", "Chest"),
                        ("Chest", "chest", "Neck1"), ("Neck1", "neck", "Neck2"), ("Neck2", "neck", "Head"), ("Head", "head", "up")),
                       legs=("Leg", "Shin", "Foot", "ToeBase"))
# SMPL-X 的 22 个身体关节（kimodo.skeleton.SMPLXSkeleton22）。关节名和顺序与 SMPL-X 一致，因此直接取
# SMPL 表的前 22 项，接入「标准人」（SMPL-X 身体）时关节可一一对应（同一张表，data/smpl.py rows_of）
SMPLX22 = tuple(j for j in body_joints("smplx")[:22])
# 第 22 个之后的关节（手指、下巴、眼睛）不在该骨架中，指向它们的 aim 须清除，否则对齐静止姿势时找不到目标
SMPLX22 = tuple(ModelJoint(j.name, j.part, j.aim if any(k.name == j.aim for k in SMPLX22) or j.aim == "up" else None)
                for j in SMPLX22)
# Unitree G1 的 34 关节（kimodo.skeleton.G1Skeleton34）：每条腿有三个独立的髋关节轴（pitch / roll / yaw），
# 无法与人物骨骼中的单个大腿关节对应，因此该骨架不能由人物骨骼驱动，仅用于生成（「动画」口不可用）。
# 空表表示没有可配对的关节
G1: tuple[ModelJoint, ...] = ()

# 每个权重对应一副骨架：关节对照表随所选模型切换（mapping_param(follows=("model",)) + joints_of）
SKELETONS = {"soma": SOMA, "smplx": SMPLX22, "g1": G1}
MODELS = {"rp": ("Kimodo-SOMA-RP-v1.1", "soma"), "seed": ("Kimodo-SOMA-SEED-v1.1", "soma"),
          "rp_v1": ("Kimodo-SOMA-RP-v1", "soma"), "seed_v1": ("Kimodo-SOMA-SEED-v1", "soma"),
          "smplx": ("Kimodo-SMPLX-RP-v1", "smplx"),
          "g1": ("Kimodo-G1-RP-v1", "g1"), "g1_seed": ("Kimodo-G1-SEED-v1", "g1")}
ON_RIG_MODELS = tuple(k for k, (_, s) in MODELS.items() if SKELETONS[s])  # 可由人物骨骼驱动的模型
TEXT_RAM_GB = 20  # 文字编码时以 bfloat16 读入内存的 Llama 3 8B（16 GB）及 worker 其余部分


class KimodoMotion(RigMotion):
    id = "kimodo.motion"
    does = "generate"  # 生成动作（骨骼动作家族的两类任务之一，lab2shot/nodes/families/rig_motion.py）
    # 引用官方 post_process_motion 的签名及其声明的返回值：输入约束（关键帧姿势，即接入的动画），
    # 输出 local_rot_mats / root_positions / posed_joints / global_rot_mats。
    official = Official(
        cite="third_party/kimodo/repo/kimodo/postprocess.py:185-212",
        takes={"character": "constraint_lst"},
        gives={"character": "local_rot_mats"},
        note="接进来的动画在上游就是 constraint_lst（EndEffectorConstraintSet / FullBodyConstraintSet，"
             "kimodo/constraints.py）；「提示词」是参数不是口。官方吐的是骨架的局部旋转矩阵加根位置，"
             "和节点交出的骨架动画是同一样东西。",
    )
    # 该模型可在不接入动画时运行：官方 `kimodo_model.py` 中 `constraint_lst` 的说明为「Pass an empty list for unconstrained generation」。
    # 家族据此将「动画」口设为可选
    unconstrained = True
    # G1 机器人的两个权重使用另一副骨架，与人物骨骼不匹配（每条腿三个独立的髋关节轴）：此时「动画」口置灰，
    # 拒绝连线。原因通过 `Because` 单独说明，自动生成的提示会列出其余五个权重的名称，不能说明真实原因
    rig_when = Because(Param("model").one_of(*ON_RIG_MODELS), "N-KIMODO-ROBOTSKELETON")
    # 单次最长生成 10 秒（30 帧/秒，共 300 帧），关键帧宜少于 20 个；更长的镜头分段生成，相邻段共用边界关键帧
    runtime = "kimodo"
    # 提示词显示在节点上。该设置写在节点类而非模板卡中，从菜单新建的节点同样显示。
    # 家族默认值为 `("keys", "exact")`（rig_motion.py `_generating`），是关键帧补间流程的两个参数；
    # 在不接入动画、仅凭文字生成的流程中，提示词是唯一输入，须显示在节点上
    on_node = ("prompt", "keys", "exact")
    joints = SOMA
    # 显存在 RTX 4090 上测得
    traits = (OptionTrait(Param("prompt").set(), ram_gb=TEXT_RAM_GB),)  # 仅在有提示词时才将文字编码器读入内存
    cost = Cost(gpu=True, vram_gb=1.2, whole="一段 10 秒的镜头几秒钟就出，不是逐帧的活")

    @classmethod
    def joints_of(cls, params: dict) -> tuple[ModelJoint, ...]:
        """「模型」一换，关节对照表就换一套：SOMA 30 关节、SMPL-X 22 关节，G1 机器人没有（不能被骨骼带动）。"""
        return SKELETONS[MODELS.get(params.get("model") or "rp", MODELS["rp"])[1]]

    class Params(FreeMotionParams):
        mapping: list[JointMap] = mapping_param(SOMA, follows=("model",))
        model: Literal["rp", "seed", "rp_v1", "seed_v1", "smplx", "g1", "g1_seed"] = P(
            "rp", label="模型", group="模型",
            option_labels={"rp": "Rigplay", "seed": "SEED", "rp_v1": "Rigplay 旧版", "seed_v1": "SEED 旧版",
                           "smplx": "SMPL-X", "g1": "G1 机器人", "g1_seed": "G1 机器人 SEED"})
        prompt: str = P("", label="提示词", group="模型", placeholder="不写：只按关键帧补", lines=4)
        steps: Literal[25, 50, 100] = measured_param(
            "去噪步数", {25: Measured(flat=True), 50: Measured(flat=True), 100: Measured(flat=True)}, default=100,
            group="模型")
        # 「贴合关键帧」作用于关键帧：未接入动画时没有关键帧，该参数不生效（输入口是否可连接由 rig_when 控制，
        # 此处针对参数）。以「接入动画时生效」表述，比列举模型名更简洁，且反映真实原因
        guidance: float = P(2.0, label="贴合关键帧", group="模型", ge=0.0, le=5.0, applies=Wired("character"))
        seed: int = P(0, label="随机种子", group="模型", ge=0)
        post_process: bool = P(True, label="模型后处理", group="模型")

    @classmethod
    def prepare(cls, ctx):
        """主任务负责生成；有提示词时另读取提示词的嵌入，该嵌入由一个仅以提示词为键的独立任务计算
        （同一提示词只编码一次，与所用镜头无关）。"""
        job = super().prepare(ctx).with_(extra={"task": "generate"})
        prompt = ctx.params["prompt"].strip()
        if not prompt:
            return job
        ctx.stage("编码文字描述")
        # "text"：该任务需要文字编码器的权重（Weight.option）
        text = ctx.run_worker(None, params={"task": "text", "text": True, "prompt": prompt})
        return job.with_(inputs={"text": RawOutput(text, MissingFrames.FAIL).path("text.npy")})


NODES = (KimodoMotion,)
