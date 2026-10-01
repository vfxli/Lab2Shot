"""UnderPressure 扩展提供的节点。"""

from __future__ import annotations

from lab2shot.sdk import (Official, Cost, CleanupParams, PartMap, Licence, ModelJoint, RigMotion, P, Port,
                          curves_packet, mapping_param)

# 官方 vGRFs 的形状为 [T, 左右 2, 每只脚 16 个鞋垫单元]（models.py:116 vGRFs -> data.py:183 的 "[...] x F x LR x 16"），
# 每个值为该单元承受的力占体重的比例。曲线名按此顺序排列，每个单元一条
VGRF_CURVES = tuple(f"{side}{i + 1}" for side in ("左脚 ", "右脚 ") for i in range(16))

# UnderPressure 自身的 23 关节骨架（data.TOPOLOGY，去掉手部与脚尖末端点的 Xsens MVN 人体）。此处逐项列出，
# 因为这是该项目特有的骨架，无法从命名约定推断：脊柱到手的链为 clavicle / shoulder / elbow / wrist，
# 对应 CG 术语中的锁骨、上臂、前臂和手，与 SMPL 系列存在相同的一个关节的偏移。
JOINTS = (
    ModelJoint("pelvis", "hips", "spine_1"),
    ModelJoint("spine_1", "spine", "spine_2"),
    ModelJoint("spine_2", "spine", "spine_3"),
    ModelJoint("spine_3", "spine", "spine_4"),
    ModelJoint("spine_4", "chest", "neck"),
    ModelJoint("neck", "neck", "head"),
    ModelJoint("head", "head", "up"),
    *(j for side in ("right", "left") for j in (
        ModelJoint(f"{side}_clavicle", f"{side[0]}.clavicle", f"{side}_shoulder"),
        ModelJoint(f"{side}_shoulder", f"{side[0]}.upperarm", f"{side}_elbow"),
        ModelJoint(f"{side}_elbow", f"{side[0]}.forearm", f"{side}_wrist"),
        ModelJoint(f"{side}_wrist", f"{side[0]}.hand", None),
        ModelJoint(f"{side}_hip", f"{side[0]}.thigh", f"{side}_knee"),
        ModelJoint(f"{side}_knee", f"{side[0]}.shin", f"{side}_ankle"),
        ModelJoint(f"{side}_ankle", f"{side[0]}.foot", f"{side}_foot"),
        ModelJoint(f"{side}_foot", f"{side[0]}.toe", None),
    )),
)


class UnderPressureFootskate(RigMotion):
    id = "underpressure.footskate"
    version = 2  # 2：帧号按节点的「帧率」换算成时间（默认 24 与 1 版的固定时基相同）
    does = "cleanup"  # 修复动作（骨骼动作家族的两类任务之一，lab2shot/nodes/families/rig_motion.py）
    # 引用官方 demo.py 的三个流程：vGRFs（model.vGRFs 估计每帧每只脚各鞋垫单元承受的力，第 20-22 行）、
    # contacts（model.contacts 据此判断每帧哪只脚着地，第 37、127 行）和 cleanup
    # （Cleaner(angles, skeleton, trajectory) 带接触约束的 IK，返回相同的三项，第 135 行）。
    official = Official(
        cite=("third_party/underpressure/repo/demo.py:18-40", "third_party/underpressure/repo/demo.py:120-141"),
        takes={"character": "angles"},
        gives={"character": "angles", "contacts": "contacts", "vgrfs": "vGRFs"},
        note="① 「动画」进出都是同一组 angles + skeleton + trajectory（demo.py:135：cleaner(item[\"angles\"], "
             "item[\"skeleton\"], item[\"trajectory\"])）。② 没有「脚滑」输出口：官方只给 contacts 和 vGRFs 两样，"
             "没有逐帧的脚滑判断。所以节点声明 judges = False，没有「只改问题帧」「检测阈值」"
             "两个参数（lab2shot/nodes/families/rig_motion.py 的 _cleaned 只在 judges 时挑帧），"
             "整段都按模型改一遍。只改滑的那几帧要做成显式的核心节点（量脚滑 → 一条曲线 → "
             "清理节点的一个接线参数），不在本节点的范围内。③ 「足底力」是官方的 vGRFs（demo.py:22 "
             "model.vGRFs(...)），「脚接触」是官方的 contacts。",
    )
    # docs.md：不读取画面，整段联合处理（接触依据前后若干帧的运动判断）；模型按 100 帧/秒训练，帧率由节点换算
    runtime = "underpressure"
    licence = Licence(note="仅限研究：InterDigital 的评估许可只允许「fundamental research work」，"
                           "明文排除一切商业用途，包括放进任何提供给第三方的产品或服务。")
    joints = JOINTS
    contacts = True  # 模型判断哪只脚着地，家族据此提供「脚接触」输出口
    # 必须为 False：上游没有「脚滑」输出（demo.py 只有 contacts 和 vGRFs），worker 不写 result["labels"]。
    # `judges` 为真时家族会读取该键（families/rig_motion.py 的 `_cleaned`），在整段计算完成后引发 KeyError。
    # `judges` 附带的「只改问题帧」「检测阈值」两个参数依赖该曲线，此处没有判据
    judges = False
    # 官方的另一项输出（demo.py:22 model.vGRFs）：每帧、每只脚、16 个鞋垫单元各自承受的力（占体重的比例）。
    # 着地由其判定，因此比「脚接触」更原始，适用于动画师自行设定着地阈值的场合
    outputs = (*RigMotion.outputs, Port("vgrfs", "curves", "足底力"))
    # 不使用 GPU：网络仅有四层卷积，500 帧清理在 CPU 上耗时 2.4 秒、RTX 5090 上 2.7 秒，
    # 12000 帧的接触判断在 CPU 上耗时 0.17 秒，使用 GPU 没有收益（docs.md）
    cost = Cost(gpu=False, whole="整段一起优化，不是逐帧的活；在 CPU 上跑，不占显卡（60 秒的动捕约 34 秒）")

    # 使用 CleanupParams 而非 DetectCleanupParams：没有「脚滑」曲线，「只改问题帧」「检测阈值」缺少判据（同上文 `judges = False`）
    class Params(CleanupParams):
        mapping: list[PartMap] | None = mapping_param()
        contact_margin: int = P(5, label="接触余量", group="清理", ge=0, le=20)

    @classmethod
    def convert(cls, ctx, raw, job):
        """家族的输出，另加网络自身的 vGRFs（demo.py:22）。raw/vgrfs.npz 已对齐到节点发送的帧
        （worker 已从模型的 100 帧/秒时间轴重采样）。"""
        out = super().convert(ctx, raw, job)
        if "vgrfs" not in ctx.wanted:
            return out
        rig = job.notes["rig"]
        values = raw.arrays("vgrfs.npz")["values"]
        out["vgrfs"] = curves_packet(ctx.outputs["vgrfs"], list(rig.frames),
                                     list(VGRF_CURVES), values, extension=cls.runtime)
        return out


NODES = (UnderPressureFootskate,)
