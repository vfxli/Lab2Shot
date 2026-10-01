"""骨骼动作：输入一副骨骼，输出带有该段动作的同一副骨骼。模型负责生成动作或修复动作。

成员通过 `does` 声明所执行的工作（与其交付物子类使用同一词语）：

    "generate"  生成动作。接入动画时：动画师的关键帧作为约束，模型补全中间帧，结果回到该骨骼上；
                未接入动画时（`unconstrained` 的成员）：只按文字和帧范围生成整段，输出模型自身骨架的骨架动画。
    "cleanup"   清理。整段每一帧都发送给模型，模型修复其判定有问题的部分，正常的帧保持不变，
                并报告其发现（`detects` / `contacts`）：动画师交付的是完成的表演，
                一个静默重写且不作任何说明的节点无法使用。

两项工作属于同一家族：声明的端口完全相同，输入 `character`（`scene.skeleton|scene.character`），输出同一种 `character`，
使用同一 worker 约定 `lab2shot_worker.rig_motion`。清理一侧只多出几个可选输出端口（「问题帧」「脚接触」），
可选端口由成员按上游能力自行声明。真正的区别只有两处：发送哪些帧（关键帧 / 每一帧），以及结果如何回到骨骼
（`data.animation.on_rig` / `repaired`），两处都在本文件中各实现一次。「补帧还是清理」不体现在任何参数中，
由节点的 `does` 声明和节点名称确定，模板卡片仍可逐一区分。

两项工作共用的对应关系、关节配对、骨骼选择和 job 写出在 `kit/rig.py`（`RigModel`）中。
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np

from ...availability import Cond, Not
from ...data.packet import Packet
from ...data.units import DEFAULT_FPS
from ...errors import Invalid
from ...messages import Msg

from ..applies import Cost, Wired
from ..base import NodeParams, P, Port
from .base import Job, RawOutput, WorkerNode
from ..kit.rig import ModelJoint, RigModel, mapping_param, part_labels, skeleton_param  # noqa: F401 (the family's API)

GENERATE = "generate"  # 生成动作（交付物子类 motion_gen）
CLEANUP = "cleanup"  # 修复动作（交付物子类 cleanup）
JOBS = (GENERATE, CLEANUP)

GROUND_TOLERANCE_CM = 15.0  # 按键帧计算的最低脚部关节距 y = 0 超过此距离时，节点给出警告
ON_RIG = Wired("character")  # 这些参数只在接入动画时生效：它们描述的都是结果如何回到该骨骼上
OFF_RIG = Not(ON_RIG)  # 相反：只在未接入动画时生效的参数（起始帧号、结束帧号）
PERSON = "generated_01"  # 生成人物在 USD 中的名称（与 Sketch2Anim 的 sketch_01 做法相同）
CONTACT_CURVES = ("左脚跟", "左脚尖", "右脚跟", "右脚尖")  # 约定中的 contacts [T,4]，顺序与之一致
BAD_FRAME_SHARE = 0.6  # 判为有问题的帧超过此比例时，节点提示该动作可能超出模型的适用范围


# ------------------------------------------------------------------ 两侧共用的参数


def motion_fps_param(applies=None):
    """「帧率」：动画是几帧每秒。帧号本身不带时间，而模型按秒工作（各自的训练帧率：Kimodo 30、StableMotion 20、
    UnderPressure 100……），job 的 fps 就是它（kit/rig.py send、free_job）；可接线，模板从读取节点的「帧率」口接来。
    worker=False：它随 job 的 motion.npz / extra 交给 worker，不再作为参数另交一份。"""
    return P(DEFAULT_FPS, label="帧率", unit="fps", group="时间", gt=0, worker=False, applies=applies)


# ------------------------------------------------------------------ 生成动作一侧的参数


class MotionGenParams(NodeParams):
    """接入动画时使用的参数（关键帧、结果如何回到骨骼上）。各成员再添加自己的对应关系和模型参数。

    只能接入动画的成员使用本类（Two-stage Transformer）：其「动画」端口为必需输入，这些参数始终生效，
    因此不写 applies（为必需端口写 `Wired(...)` 相当于一个恒成立的条件）。
    可以不接入动画的成员使用下方的 FreeMotionParams。"""

    skeleton: str | None = skeleton_param()
    keys: str = P("", label="关键帧", group="关键帧", placeholder="自动", worker=False)
    exact: bool = P(True, label="关键帧精确", group="结果", worker=False)
    foot_lock: bool = P(True, label="脚锁定", group="结果", worker=False)
    fps: float = motion_fps_param()


class FreeMotionParams(MotionGenParams):
    """可以不接入动画、只按文字生成的成员（`unconstrained = True`）的参数：上述参数只在接入动画时生效
    （此处为其添加 applies，未接入动画时置灰并说明原因，位置不变），另加镜头的起止帧，
    它们只在未接入动画时生效（接入动画时帧范围由该动画决定，显式优先于隐式）。

    帧范围以起始帧号 / 结束帧号表示，而非「帧数」：这是 Nuke / Houdini 中定义镜头区间的方式，
    「起始帧号」与「读取视频」中的用语相同；而「帧数」在本项目中专指「模型一次处理的帧数」这类
    可能耗尽显存的参数，不用于描述镜头长度。"""

    keys: str = P("", label="关键帧", group="关键帧", placeholder="自动", worker=False, applies=ON_RIG)
    exact: bool = P(True, label="关键帧精确", group="结果", worker=False, applies=ON_RIG)
    foot_lock: bool = P(True, label="脚锁定", group="结果", worker=False, applies=ON_RIG)
    start_frame: int = P(1001, label="起始帧号", group="时间", worker=False, applies=OFF_RIG)
    end_frame: int = P(1120, label="结束帧号", group="时间", worker=False, applies=OFF_RIG)


# ------------------------------------------------------------------ 清理一侧的参数


class CleanupParams(NodeParams):
    """所有「动作清理」节点共有的参数；成员另外添加其对应关系（mapping_param）和模型自身的参数。"""

    skeleton: str | None = skeleton_param()
    fps: float = motion_fps_param()


class DetectCleanupParams(CleanupParams):
    """声明了 `judges` 的成员：它逐帧作出判断，因此动画师可以指定如何使用该判断。
    这与是否存在对应输出端口（`detects`）是两回事：judges=True、detects=() 的成员有参数而无输出端口。
    UnderPressure 两者都没有（上游只给触地和足底力，没有逐帧判断）：judges=False，整段都按模型改。"""

    only_bad: bool = P(True, label="只改问题帧", group="结果", worker=False)
    threshold: float = P(0.5, label="检测阈值", group="结果", ge=0.05, le=0.95, worker=False)


# ------------------------------------------------------------------ 家族


class RigMotion(RigModel, WorkerNode):
    """骨骼动作（见模块开头）：输入带有一段动作的骨骼，输出带有该段动作的同一副骨骼。

    生成动作（`does = "generate"`）：
    接入动画时，其关键帧作为约束，模型补全中间帧，动作回到该骨骼上（按对应关系配对、对齐静止姿势、
    缩放骨骼长度，见 lab2shot_worker.rig_motion；模型不具备的关节保留原动画，关键帧精确还原、脚部固定，
    见 data.animation.on_rig）。未接入动画时（`unconstrained` 的成员），只按文字和帧范围生成，输出模型自身骨架的
    一段骨架动画（骨骼名和关节轴经过 data.skeleton.rig_of_model，与其他骨架动画交付物约定一致）。

    清理（`does = "cleanup"`）：
    整段每一帧都发送出去（而非动画师的关键帧），worker 可以在动作旁另写一份 `labels`（其对每帧问题的判断）；
    结果按 data.animation.repaired 回到骨骼上，模型判定正常的帧可以完全不变（「只改问题帧」）。

    两者都不涉及画面。`Job.notes`："rig"（data.animation.read_rig）、"keys"（发送的帧）、"pairs"（模型关节 →
    人物关节序号）、"parts"（部位 → 模型关节）；未接入动画生成时 "rig" 为 None，另有 "frames"（结果的帧号）。"""

    # 该成员执行的工作，二者之一，必须声明。不提供默认值：未声明时在创建类时报错，
    # 以免「忘记声明」变成「静默按另一种工作运行」。
    does: ClassVar[str] = ""
    # rig → 模型骨架的对齐（lab2shot_shared.motion.Retarget.align）改过：同一套关节约定按关节坐标系、根按身体坐标系、
    # 静止姿势不是站姿时在发来的姿势里对齐——四个成员的结果都变了，旧缓存要重算
    version: ClassVar[int] = 11  # 进指纹：rig 与模型骨架的对齐（motion.Retarget.align）变了就加一
    cost = Cost(gpu=True)

    # --- 仅 does = "generate" 的成员声明 ---
    # 该模型能否在没有任何关键帧时生成（Kimodo 可以：`constraint_lst` 传空列表；
    # Two-stage Transformer 不能：它只在两个关键帧之间生成一段）。声明后，家族将「动画」端口改为可选端口。
    unconstrained: bool = False
    # 未接入动画时一次最多生成的帧数（安全上限，成员可自行设定；超出时拒绝并说明处理方法）
    most_free_frames: int = 3000
    # 「动画」端口的适用条件。部分项目的某些权重对应特殊骨架，与人物骨骼无法对应
    # （Kimodo 的 Unitree G1：一条腿有三个独立的髋关节轴），此时该端口置灰并说明原因，接线也被拒绝。
    # 未声明时始终可用。判断由服务器统一计算，前端组件中不写条件分支。
    rig_when: Cond | None = None

    # --- 仅 does = "cleanup" 的成员声明 ---
    detects: tuple[str, ...] = ()  # 该模型逐帧给出的判断（为空表示无，即没有「问题帧」输出）
    # 能否逐帧判断与该判断是否为上游输出是两回事：一个成员可以逐帧判断却不把判断交出去，
    # 这时仍需要「只改问题帧」「检测阈值」两个参数，否则整段都会被模型修改，原本不滑的动捕反而会出现滑动。
    # 因此拆为两项声明：`detects` 决定是否有输出端口，
    # `judges` 决定是否有这两个参数及相应逻辑（UnderPressure 两者都没有：judges=False）。声明了 detects 的必然也 judges（在 __init_subclass__ 中补上）。
    judges: bool = False
    contacts: bool = False  # 该模型是否判断哪只脚着地（否：没有「脚接触」输出）

    def __init_subclass__(cls, **kw):
        if cls.does not in JOBS:
            # 创建类时面向开发者的信息，不面向使用者，因此使用英文；面向使用者的中文一律在消息目录中
            raise TypeError(f"{cls.__name__}: a RigMotion member declares does = 'generate' (makes a take) or "
                            f"'cleanup' (repairs one), not {cls.does!r} (lab2shot/nodes/families/rig_motion.py)")
        # 在一种工作上写了另一种工作的声明：立即报错，避免其静默失效
        theirs = (("detects", cls.detects), ("contacts", cls.contacts), ("judges", cls.judges)) \
            if cls.does == GENERATE else (("unconstrained", cls.unconstrained),)
        if wrong := [name for name, on in theirs if on]:
            raise TypeError(f"{cls.__name__}: a does = {cls.does!r} member must not declare {wrong} — "
                            f"detects / contacts / judges belong to cleanup (what the model says it found), "
                            f"unconstrained belongs to generate (it can make a take with no keys at all)")
        if cls.does == GENERATE:
            cls._generating()
        else:
            cls._cleaning()
        super().__init_subclass__(**kw)

    # ------------------------------------------------------------------ 创建类：两种工作各自的端口和声明

    @classmethod
    def _generating(cls) -> None:
        """生成动作的成员：节点上直接显示关键帧的两个参数；能无约束生成的成员，「动画」端口改为可选端口。"""
        if not cls.on_node:
            cls.on_node = ("keys", "exact")
        if cls.unconstrained and not any(p.name == "character" and p.optional for p in cls.inputs):
            cls.inputs = tuple(
                Port(p.name, p.type, p.label, optional=True, applies=cls.rig_when,
                     help="接上动画：它的关键帧就是约束，模型只补关键帧之间的动作，结果回到这副骨骼上。"
                          "不接：只按文字和「起始帧号」「结束帧号」生成一整段新动作，交出模型自己的骨架动画")
                if p.name == "character" else p for p in cls.inputs)

    @classmethod
    def _cleaning(cls) -> None:
        """清理的成员：修复后的动作是主结果，其发现按声明作为旁边的输出端口。"""
        # 清理节点必须报告其发现：动画师交付的是完成的表演，一个静默重写且不作任何说明的节点无法使用。
        # 分类规则也正是依据这两个端口区分清理和补帧。
        if not (cls.detects or cls.contacts):
            raise TypeError(f"{cls.__name__}: a motion-cleanup node must declare `detects` or `contacts` "
                            f"(lab2shot/nodes/families/rig_motion.py: it has to say what it found)")
        if not cls.main:  # 修复后的动作是主结果，「问题帧」「脚接触」「足底力」是其旁边的说明
            cls.main = "character"
        # 该端口定义在家族上：成员声明其能力，端口即随之出现；适配器中只需一行声明
        if cls.detects:  # 有输出端口的必然也能计算出判断
            cls.judges = True
        # `judges` 与这两个参数必须同时存在或同时不存在：`judges` 为真时家族会读取 worker 的 `result["labels"]`
        # （见 _cleaned），而「只改问题帧」「检测阈值」判断的正是该曲线。参数存在而 judges 为假时，这两个参数
        # 没有判据；judges 为真而参数不存在时，cook 会在 worker 运行完之后才出现 KeyError。因此在创建类时检查。
        has_params = any("only_bad" in getattr(base, "__annotations__", {}) for base in cls.Params.__mro__)
        if cls.judges != has_params:
            raise TypeError(
                f"{cls.__name__}: judges is {cls.judges} while the two judging parameters are "
                f"{'present' if has_params else 'absent'}; they go together. With judges, the family reads the "
                f"worker's result['labels'] (_cleaned below) and those parameters decide on that curve: one without "
                f"the other either lies to the artist or raises KeyError mid-cook. To judge, inherit "
                f"DetectCleanupParams and have the worker write labels; not to judge, inherit CleanupParams "
                f"and set judges = False")
        if cls.detects and not any(p.name == "labels" for p in cls.outputs):
            cls.outputs = (*cls.outputs, Port("labels", "curves", cls.detects[0]))
        if cls.contacts and not any(p.name == "contacts" for p in cls.outputs):
            cls.outputs = (*cls.outputs, Port("contacts", "curves", "脚接触"))

    # ------------------------------------------------------------------ 发送给 worker 的内容

    @staticmethod
    def key_frames(text: str, rig) -> list[int]:
        """关键帧：列出的帧，否则为骨骼动画中有采样的帧。"""
        from ...data.animation import parse_frames

        if text.strip():
            keys = parse_frames(text)
            outside = [f for f in keys if f not in set(rig.frames)]
            if outside:
                raise Invalid(Msg("E-MOTIONGEN-KEYOUTSIDE", frame=outside[0], first=rig.frames[0], last=rig.frames[-1]))
        else:
            keys = rig.keys
            if len(keys) == len(rig.frames):
                raise Invalid(Msg("E-MOTIONGEN-BAKED", first=rig.frames[0], last=rig.frames[-1]))
        if len(keys) < 2:
            raise Invalid(Msg("E-MOTIONGEN-FEWKEYS", count=len(keys)))
        return keys

    @classmethod
    def prepare(cls, ctx) -> Job:
        """发送给 worker 的帧：清理发送每一帧，生成动作发送动画师的关键帧（未接入动画时不发送任何帧）。"""
        if cls.does == CLEANUP:
            return cls.cleanup_job(ctx)
        if cls.unconstrained and ctx.input("character") is None:
            return cls.free_job(ctx)
        return cls.keyed_job(ctx)

    @classmethod
    def cleanup_job(cls, ctx) -> Job:
        """清理：发送整段每一帧的世界姿势。"""
        rig = cls.rig(ctx)
        pairs, parts = cls.driven(ctx, rig)
        ctx.stage("整理动作")
        job = cls.send(ctx, rig, pairs, parts, list(rig.frames), rig.world())
        if cls.judges:  # worker 同样按节点的「检测阈值」截断其判断，使其重绘的帧与节点保留的帧一致；
            # 「只改问题帧」关闭时，worker 重绘每一帧（该开关由家族定义），而不只是判定有问题的帧
            job = replace(job, extra={**job.extra, "threshold": float(ctx.params["threshold"]), "repaint_all": not ctx.params["only_bad"]})
        return job

    @classmethod
    def keyed_job(cls, ctx) -> Job:
        """生成动作且接入动画：动画师的关键帧即约束，只发送这些帧的世界姿势。"""
        rig = cls.rig(ctx)
        keys = cls.key_frames(ctx.params["keys"], rig)  # 关键帧先于对应关系处理：无论骨骼关节如何命名，
        pairs, parts = cls.driven(ctx, rig)  # 烘焙文件都应首先报告
        ctx.stage("整理关键帧")
        poses = rig.world()[rig.index(keys)]
        feet = [pairs[parts[x]] for x in ("l.foot", "r.foot", "l.toe", "r.toe") if parts.get(x) in pairs]
        low = float(poses[:, feet][..., 1, 3].min())
        if abs(low) > GROUND_TOLERANCE_CM:  # 模型站立在 y = 0 上
            ctx.say("W-MOTIONGEN-OFFGROUND", low=low, port="character")
        return cls.send(ctx, rig, pairs, parts, keys, poses)

    @classmethod
    def free_job(cls, ctx) -> Job:
        """未接入动画：镜头的起止帧和「帧率」即 worker 所需的全部信息（约束为空）。模型按自己的帧率生成，
        worker 按「帧率」把它重采样成起止帧之间每帧一个姿势。"""
        first, last = int(ctx.params["start_frame"]), int(ctx.params["end_frame"])
        fps = float(ctx.params["fps"])
        if last < first:
            raise Invalid(Msg("E-MOTIONGEN-BADRANGE", first=first, last=last))
        length = last - first + 1
        if length > cls.most_free_frames:  # 服务器端同样拒绝：不允许绕过网页提交会耗尽机器资源的数值
            raise Invalid(Msg("E-MOTIONGEN-TOOLONG", frames=length, most=cls.most_free_frames,
                              seconds=round(cls.most_free_frames / fps)))
        return Job(None, extra={"length": length, "fps": fps},
                   notes={"rig": None, "keys": [], "frames": list(range(first, last + 1)), "fps": fps})

    # ------------------------------------------------------------------ 结果如何回到骨骼

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from lab2shot_worker.rig_motion import read_result

        result = read_result(raw.folder)
        if cls.does == CLEANUP:
            return cls._cleaned(ctx, result, job)
        if job.notes["rig"] is None:
            return cls.free_animation(ctx, result, job)
        return cls._generated(ctx, result, job)

    @classmethod
    def _generated(cls, ctx, result: dict, job: Job) -> dict[str, Packet]:
        """接入动画的生成：补全的动作放回人物骨骼，关键帧精确还原、脚部固定。"""
        from ...data.animation import on_rig, write_animation

        rig, keys, pairs, parts = (job.notes[k] for k in ("rig", "keys", "pairs", "parts"))
        ctx.stage("动作放回人物骨骼")
        feet = [tuple(pairs[parts[f"{s}.{x}"]] for x in ("thigh", "shin", "foot")) for s in ("l", "r")] if ctx.params["foot_lock"] else None
        local, stats = on_rig(rig, keys, result, ctx.params["exact"], feet)
        info = {"extension": cls.runtime, "inbetween": {**result["info"], **stats}, "keys": keys}
        return {"character": write_animation(ctx.input("character"), ctx.outputs["character"], rig, local, info)}

    @classmethod
    def free_animation(cls, ctx, result: dict, job: Job) -> dict[str, Packet]:
        """未接入动画时的结果：模型自身骨架的一段骨架动画，骨骼名和关节轴换为 CG 约定（rig_of_model），
        与「提取骨架」「Sketch2Anim 动作生成」的输出类型相同，重定向时可被 HumanIK 识别。"""
        from ...data.payloads import SCENE_FILE, scene_packet
        from ...data.skeleton import rig_of_model
        from ...io.usd import create_stage, save_stage, write_rig

        frames = job.notes["frames"]
        names = [str(n) for n in result["names"]]
        parents = np.asarray(result["parents"], np.int64)
        bind = np.asarray(result["rest"], np.float64)  # 模型静止姿势的关节到世界矩阵，单位厘米
        anim = np.asarray(result["anim"], np.float64)  # [F,J,4,4] 逐帧的关节到世界矩阵，单位厘米
        if len(anim) != len(frames):  # worker 已重采样到所要求的帧数：不一致说明两者不同步
            raise Invalid(Msg("E-MOTIONGEN-FRAMES", node=ctx.label, made=len(anim), want=len(frames)))
        ctx.stage("写骨架动画")
        names_cg, bind_cg, anim_cg = rig_of_model(names, parents, bind, anim)
        out = ctx.outputs["character"]
        info = {"extension": cls.runtime, "motion": result["info"]}
        stage = create_stage(frames, info)
        write_rig(stage, PERSON, names_cg, parents, bind_cg, anim_cg, frames,
                  custom_data={"generated_by": f"Lab2Shot {cls.runtime}"})
        save_stage(stage, out / SCENE_FILE)
        packet = scene_packet(out, frames, "scene.skeleton", people=[PERSON])
        packet.meta[cls.runtime] = result["info"]
        return {"character": packet}

    @classmethod
    def _cleaned(cls, ctx, result: dict, job: Job) -> dict[str, Packet]:
        """清理：修复后的动作放回人物骨骼（模型判定正常的帧可以完全不变），并在旁边输出其发现。"""
        from ...data.animation import model_at, repaired, write_animation
        from ...data.payloads import curves_packet
        from ...data.units import PERCENT
        from lab2shot_shared import motion as mo

        rig, sent = job.notes["rig"], job.notes["keys"]
        ctx.stage("清理过的动作放回人物骨骼")
        at = model_at(rig, sent, result)  # 骨骼每一帧在模型时间轴上的位置
        timeline = np.arange(len(result["rotations"]))
        labels = (mo.resample_values(timeline, result["labels"], at) if cls.judges
                  else np.zeros((len(at), 0)))  # 模型的判断，对应到骨骼的帧上
        keep = None
        if cls.judges and ctx.params["only_bad"]:
            keep = labels[:, 0] <= ctx.params["threshold"]
        local, stats = repaired(rig, sent, result, keep)
        if cls.judges:
            bad = float((labels[:, 0] > ctx.params["threshold"]).mean())
            if bad > BAD_FRAME_SHARE:
                ctx.say("W-CLEANUP-MOSTLYBAD", share=round(PERCENT * bad), port="character")
        info = {"extension": cls.runtime, "cleanup": {**result["info"], **stats}}
        out = {"character": write_animation(ctx.input("character"), ctx.outputs["character"], rig, local, info)}
        frames = rig.frames
        if cls.detects:
            out["labels"] = curves_packet(ctx.outputs["labels"], frames, list(cls.detects), labels,
                                          extension=cls.runtime)
        if cls.contacts:
            feet = mo.resample_values(timeline, result["contacts"], at)
            out["contacts"] = curves_packet(ctx.outputs["contacts"], frames, list(CONTACT_CURVES), feet,
                                            extension=cls.runtime)
        return out
