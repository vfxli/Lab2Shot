"""Kimodo 扩展提供的节点。"""

from __future__ import annotations

from typing import Literal

from .extension import MODELS as WEIGHTS, OPTION_LICENCES
from lab2shot.sdk import (Official, licence_traits, measured_param, RigMotion, FreeMotionParams, MissingFrames, RawOutput,
                          Invalid, Msg, PartMap, P, Param, Wired, Because, body_joints, humanoid_joints, mapping_param, Cost, OptionTrait,
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

# 每个权重对应一副骨架：「对应关系」里模型那一侧随所选模型切换（mapping_param(follows=("model",)) + joints_of）
SKELETONS = {"soma": SOMA, "smplx": SMPLX22, "g1": G1}
# 「模型」选项 -> (权重文件夹名, 骨架)：由 extension.py 的 MODELS（唯一的一张表）派生
MODELS = {k: (repo.split("/")[-1], skeleton) for k, (repo, _rev, skeleton) in WEIGHTS.items()}
# 许可受限的选项（SMPL-X：仅限研究）由编辑器按 option_traits 统一注明（webui ui/controls.tsx optionText），名字里不另写；
# 选项名在 node."kimodo.motion".param.model.option.<值>
ON_RIG_MODELS = tuple(k for k, (_, s) in MODELS.items() if SKELETONS[s])  # 可由人物骨骼驱动的模型
TEXT_RAM_GB = 20  # 文字编码时以 bfloat16 读入内存的 Llama 3 8B（16 GB）及 worker 其余部分


class KimodoMotion(RigMotion):
    id = "kimodo.motion"
    version = 5  # 5：只为让半途代码算出的 4 的缓存重算；4：两副站着的基准姿势时躯干整体按根的对齐（motion.Retarget.align）；3：G1 权重按官方不做后处理；2：帧号按节点的「帧率」换算成时间（默认 24 与 1 版的固定时基相同）
    does = "generate"  # 生成动作（骨骼动作家族的两类任务之一，lab2shot/nodes/families/rig_motion.py）
    # 引用官方 post_process_motion 的签名及其声明的返回值：输入约束（关键帧姿势，即接入的动画），
    # 输出 local_rot_mats / root_positions / posed_joints / global_rot_mats。
    official = Official(
        cite="third_party/kimodo/repo/kimodo/postprocess.py:185-212",
        takes={"character": "constraint_lst"},
        gives={"character": "local_rot_mats"},
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
    # 有提示词时才把文字编码器读入内存；SMPL-X 权重只限研究（extension.py OPTION_LICENCES）
    traits = (OptionTrait(Param("prompt").set(), ram_gb=TEXT_RAM_GB), *licence_traits(OPTION_LICENCES))
    cost = Cost(gpu=True, vram_gb=1.2, whole=True)

    @classmethod
    def joints_of(cls, params: dict) -> tuple[ModelJoint, ...]:
        """「模型」一换，「对应关系」里模型那一侧就换一套：SOMA 30 关节、SMPL-X 22 关节，G1 机器人没有（不能被骨骼带动）。"""
        return SKELETONS[MODELS.get(params.get("model") or "rp", MODELS["rp"])[1]]

    class Params(FreeMotionParams):
        mapping: list[PartMap] | None = mapping_param(follows=("model",))
        model: Literal[tuple(MODELS)] = P("rp", group="model")  # type: ignore[valid-type]
        prompt: str = P("", group="model", lines=4)
        steps: Literal[25, 50, 100] = measured_param(
            {25: Measured(flat=True), 50: Measured(flat=True), 100: Measured(flat=True)}, default=100,
            group="model")
        # 「贴合关键帧」作用于关键帧：未接入动画时没有关键帧，该参数不生效（输入口是否可连接由 rig_when 控制，
        # 此处针对参数）。以「接入动画时生效」表述，比列举模型名更简洁，且反映真实原因
        guidance: float = P(2.0, group="model", ge=0.0, le=5.0, applies=Wired("character"))
        seed: int = P(0, group="model", ge=0)
        # 官方对 G1 机器人权重一律关掉后处理（generate.py:325-326、demo/ui.py:2846）：选 G1 时置灰，worker 也不做
        post_process: bool = P(True, group="model",
                               applies=Because(Param("model").one_of(*ON_RIG_MODELS), "N-KIMODO-G1NOPOSTPROCESS"))

    @classmethod
    def plan_refusals(cls, params, comes):
        """没接动画、提示词也空（也没有线接进提示词）：算之前就知道，「计算」置灰并写原因（B-KIMODO-WAITPROMPT）。
        提示词由别的节点接进来时要等它算出才知道是不是空，那时由 prepare 拒绝（E-KIMODO-NOPROMPT）。"""
        if str(params.get("prompt") or "").strip() or comes(cls.param_port("prompt").name) or comes("character"):
            return []
        return [(Msg("B-KIMODO-WAITPROMPT"), "character")]

    @classmethod
    def prepare(cls, ctx):
        """主任务负责生成；有提示词时另读取提示词的嵌入，该嵌入由一个仅以提示词为键的独立任务计算
        （同一提示词只编码一次，与所用镜头无关）。"""
        prompt = ctx.params["prompt"].strip()
        # 没接动画、也没写提示词：模型只会给一段原地站着的「待机」，不算，说清楚要什么。提示词常经「翻译」
        # 等节点接进来，算之前不知道是不是空，所以在这里（发 worker 之前，不占显卡）拒绝
        if not prompt and ctx.input("character") is None:
            raise Invalid(Msg("E-KIMODO-NOPROMPT"))
        # checkpoint：所选权重的文件夹名（worker 不另存一张权重表）
        job = super().prepare(ctx).with_(extra={"task": "generate", "checkpoint": MODELS[ctx.params["model"]][0]})
        if not prompt:
            return job
        ctx.stage("encode_prompt")
        # "text"：该任务需要文字编码器的权重（Weight.option）
        text = ctx.run_worker(None, params={"task": "text", "text": True, "prompt": prompt})
        return job.with_(inputs={"text": RawOutput(text, MissingFrames.FAIL).path("text.npy")})


NODES = (KimodoMotion,)
