"""学习式重定向的前后两段：「重定向预处理」和「重定向后处理」。中间一段是解算器（families/rig_retarget.py）。

预处理把人手要修的三件事放在一处：两副骨架的部位对应、两侧的参考姿态（初始姿态对齐）、哪些骨骼不送进解算器；
三件都在视图里编辑（「rig_pair」手柄），每件都有一个「自动」，自动的结果写进参数，看得见、改得了。结果写进 USD 骨架
（kit/retarget_prepare.py stamp），不走包的 meta。解算器原生输出什么就是什么；位移和髋高放在后处理里，明着做。

「动作重定向」（core/scene.py Retarget）是一体的核心算法，不在这里。
"""

from __future__ import annotations

import copy
import json
from typing import Literal

import numpy as np

from ...errors import Invalid
from ... import i18n
from ...messages import Both, Msg
from ...recent import Recent, packet_key
from ..applies import Param
from ..base import Info, NodeDef, NodeParams, P, Port
from ..handles import RigPair
from ..kit.rig_map import (JointPose, PartMap, auto_record_param, ignore_param, pose_param, rig_map_param, rule_param,
                           scale_param, side_parts)
from .scene import Retarget, _stored

_RIGS: Recent = Recent(8)  # (包, 骨架路径) -> Rig：手柄反复读同一副骨架
_SIDES: Recent = Recent(8)  # (两个包, 两条骨架路径, 相关参数) -> RetargetPrepare._sides 的结果


def _moved(name: str, group: str):
    """与「动作重定向」相同的参数声明，只换分组。"""
    field = copy.deepcopy(Retarget.Params.model_fields[name])
    field.json_schema_extra["group"] = group
    return field


REST_PARAMS = ("motion_rest", "motion_rest_frame", "target_rest", "target_rest_frame")


def on_own_ground(reference: np.ndarray, rig, feet: list[int]) -> np.ndarray:
    """参考姿态（世界 [J,4,4]）上下平移到这一侧动画站的地面上：参考姿态里最低的脚踝，挪到动画每帧最低脚踝的中位数
    （走路时就是支撑脚，一段坡上取中间）。学习式解算器把参考姿态的髋高当站立高度（SATA tpose_height_to_skel 拿动画
    的髋高减它）：绑定姿势不在动画的地面上——相机空间解出的人绑定在第一帧相机下，悬在 1.4 米高——走路的人就被当成
    髋低了 1.4 米，解成跪姿。只挪参考姿态，动画、绑定姿势都不动；只有一帧的一侧（目标常取首帧）挪的就是它自己的差。"""
    if not feet:
        return reference
    lowest = np.min(rig.world()[:, feet, 1, 3], axis=1)
    out = np.array(reference, np.float64, copy=True)
    out[:, 1, 3] += float(np.median(lowest)) - float(np.min(out[feet, 1, 3]))
    return out


class RetargetPrepare(NodeDef):
    """重定向预处理：两副骨架的部位对应、两侧的参考姿态、不送进解算器的骨骼，交出标注过的「源」「目标」。

    「动作」接要搬的动作（骨架动画或蒙皮角色），「目标」接要套上动作的角色。两个输出是输入原样，加上各自骨架上的
    标注（kit/retarget_prepare.py）：这一侧的部位表、忽略的骨骼、修正后的参考姿态。动画、绑定姿势都不动。

    参考姿态先按「基准姿势」从绑定姿势、首帧或指定帧取（动作的绑定姿势不是站姿时自动挑最像目标的一帧），
    再叠上「初始姿势」的逐关节修正。「自动姿态」把两边摆成标准 T 姿并转到同一个朝前方向，写成修正行；「自动对应」
    把按名字和层级的推测写成对应行；「自动忽略」按所选解算器的规则（kit/retarget_needs.py）写出忽略名单。默认就是自动：
    「自动姿态」「自动忽略」没执行过、参数又是空的，计算按它们的建议（kit/retarget_prepare.py effective）。被忽略的
    骨骼在参考姿态里保留修正后的局部，跟着父骨骼走。

    「动作尺寸」「目标尺寸」：学习式解算器只见过人体尺寸，8 米高的角色要先缩到人的大小。两个输出包里这一侧的骨架
    （绑定姿势、动画平移、参考姿态）和蒙皮网格整体乘这个系数（kit/retarget_prepare.py scale_rig），系数记在骨架上，
    「重定向后处理」按它还原成原尺寸。留空 = 自动：腿长在人体尺寸范围内不缩放，否则缩到标准人体（auto_scale）；1 = 不缩放。
    """

    id = "retarget_prepare"
    tool = False  # a stage of the retarget cards (prepare → solver → post), not a tool of its own (engine/node_tools.py)
    category = "scene_convert"
    version = 9  # 9：只为让半途代码的缓存重算；8：参考姿态站在这一侧动画的地面上（绑定姿势悬空的人：SATA 当成髋低了 1.4 米、解成跪姿）；7：有蒙皮权重时不驱动任何顶点的关节不参与推测对应、作忽略建议；6：尺寸归一（动作尺寸、目标尺寸，默认自动；缩放时站到地面上）
    inputs = (Port("motion", "scene.skeleton|scene.character"),
              Port("target", "scene.character|scene.skeleton"))
    outputs = (Port("source", "scene.skeleton|scene.character", type_from="input:motion"),
               Port("target", "scene.character|scene.skeleton", type_from="input:target"))
    handles = (RigPair(src="motion", dst="target", src_skeleton="motion_skeleton", dst_skeleton="target_skeleton",
                       mapping="mapping", src_pose="motion_pose", dst_pose="target_pose", src_ignore="motion_ignore",
                       dst_ignore="target_ignore", ignore_rule="ignore_rule", auto_record="auto_record",
                       src_scale="motion_scale", dst_scale="target_scale"),)

    class Params(NodeParams):
        motion_skeleton: str | None = _moved("motion_skeleton", "skeleton")
        target_skeleton: str | None = _moved("target_skeleton", "skeleton")
        mapping: list[PartMap] | None = rig_map_param(("motion", "target", "motion_skeleton", "target_skeleton",
                                                       *REST_PARAMS, "motion_pose", "target_pose"), group="skeleton")
        motion_pose: list[JointPose] = pose_param()
        target_pose: list[JointPose] = pose_param()
        motion_ignore: list[str] = ignore_param()
        target_ignore: list[str] = ignore_param()
        ignore_rule: str = rule_param()
        motion_scale: float | None = scale_param()
        target_scale: float | None = scale_param()
        auto_record: str = auto_record_param()
        motion_rest: Literal["bind", "first", "frame"] = _moved("motion_rest", "advanced")
        motion_rest_frame: int | None = _moved("motion_rest_frame", "advanced")
        target_rest: Literal["bind", "first", "frame"] = _moved("target_rest", "advanced")
        target_rest_frame: int | None = _moved("target_rest_frame", "advanced")

    # 「忽略规则」「自动记录」本身不算结果参数（affects_result=False）：记录里的快照是页面写状态行用的，每次手动编辑都会
    # 更新，进指纹会让下游的解算器白白重算。但「默认自动」让其中两件事决定结果（kit/retarget_prepare.py effective）：
    # 哪个「自动」执行过（没执行过、名单又空就按建议算），以及按建议算忽略时用的是哪条规则。指纹只收这两件
    AUTO = "_auto"

    @classmethod
    def affecting_params(cls) -> set[str]:
        return super().affecting_params() | {cls.AUTO}

    @classmethod
    def fingerprint_params(cls, params: dict) -> dict:
        from ..kit.retarget_prepare import auto_done

        done = auto_done(params.get("auto_record") or "")
        rule = params.get("ignore_rule") or "" if "ignore" not in done else ""
        return {**params, cls.AUTO: {"done": sorted(k for k in ("pose", "ignore") if k in done), "rule": rule}}

    @classmethod
    def info(cls, params, inputs):
        """帧来自「动作」：目标常常只有一帧的静止姿势，「目标」输出原样带着它自己的帧。"""
        return Info.merge(inputs.get("motion", []))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        from ..kit.retarget_needs import rule_label, rule_owners
        from ..kit.rig_map import skeleton_choice

        out = {}
        for port, name in (("motion", "motion_skeleton"), ("target", "target_skeleton")):
            if inputs.get(port) is not None:
                out.update(skeleton_choice(inputs[port], name))
        owners = rule_owners()
        out["ignore_rule"] = {"options": list(owners), "labels": {k: str(rule_label(k, o)) for k, o in owners.items()}}
        return out

    @staticmethod
    def _rest_args(params: dict) -> dict:
        return {"motion_rest": params.get("motion_rest") or "bind", "motion_frame": params.get("motion_rest_frame"),
                "target_rest": params.get("target_rest") or "first", "target_frame": params.get("target_rest_frame")}

    @classmethod
    def _sides(cls, params: dict, inputs: dict, *, strict: bool = False):
        """两副骨架、对应关系和未修正的参考姿态（kit/retarget.py rests：选的姿势、动作折叠时自动挑的帧，不摆 T、不修正；
        只配手时是两边的绑定姿势，hand_bases）：(动作 Rig, 目标 Rig, Mapping, Rests)。对应关系或参考帧不成时后两项是
        None（`strict`：照样报出来），读不到骨架时 None。手柄每改一次参数都要它：按输入包和相关参数记住。"""
        from ...data.animation import read_rig
        from ..kit.retarget import hand_bases, resolve_mapping, rests

        def rig(port: str, path):
            packet = inputs.get(port)
            if packet is None:
                return None

            def read():
                try:
                    return read_rig(packet, path, least_frames=1)
                except Invalid:
                    if strict:
                        raise
                    return None

            key = not strict and _stored(packet) and (packet_key(packet), path)
            return _RIGS.get_or(key, read) if key else read()

        source, target = rig("motion", params.get("motion_skeleton")), rig("target", params.get("target_skeleton"))
        if source is None or target is None:
            return None
        args = cls._rest_args(params)
        key = not strict and _stored(inputs["motion"]) and _stored(inputs["target"]) and (
            packet_key(inputs["motion"]), packet_key(inputs["target"]), source.path, target.path,
            json.dumps([params.get("mapping"), args], sort_keys=True, default=str))
        if key and (kept := _SIDES.get(key)) is not None:
            return kept
        try:
            m = resolve_mapping(params.get("mapping"), source, target, "legs")
            base = hand_bases(source, target) if m.hands else rests(source, target, m, **args)
            got = source, target, m, base
        except Invalid:
            if strict:
                raise
            got = source, target, None, None
        return _SIDES.put(key, got) if key else got

    @classmethod
    def handle_data(cls, params: dict, inputs: dict) -> dict[int, dict]:
        """「rig_pair」手柄的数据（kit/rig_map.py rig_pair_data）：两侧骨架未修正的参考姿态、部位、「自动姿态」的建议行，
        部位表，推测的对应，和每条忽略规则在两侧的建议名单（kit/retarget_needs.py suggest_ignored）。页面预选的规则
        是通用规则：预处理不往下游看解算器（数据流不倒灌）。"""
        from ...data.joints import auto_rows, joint_keys, part_names
        from ..kit.retarget import REST_LABEL, corrected, required, rest_of, rest_pose
        from ...data.skeleton_recognition import recognize
        from ..kit.retarget_needs import GENERIC_ID, GROUP_OF, all_rules, ignore_reasons, rule_label, rule_owners
        from ..kit.retarget import ground_of
        from ..kit.retarget_prepare import auto_pose, auto_scale, ground_level, leg_length
        from ..kit.rig_map import rig_pair_data, rig_pair_side

        got = cls._sides(params, inputs)
        if got is None:
            return {}
        source, target, m, base = got
        args = cls._rest_args(params)
        sides, parts_of = [], []
        for k, (rig, port, key, which, frame, side) in enumerate((
                (source, "motion", "motion_pose", "motion_rest", "motion_frame", "src"),
                (target, "target", "target_pose", "target_rest", "target_frame", "dst"))):
            keys = joint_keys(rig.names, rig.parents)
            if base is not None:
                before = base.source_before if k == 0 else base.target_before
                said = base.source_said if k == 0 else base.target_said
                parts = getattr(m, side)
            else:
                try:
                    before = rest_pose(rig, args[which], args[frame], i18n.Word("role.motion" if k == 0 else "role.target"))
                except Invalid:
                    return {}
                said = REST_LABEL[args[which]].format(frame=args[frame])
                parts = side_parts(rig.names, rig.parents, params.get("mapping"), side, rest_of(rig), rig.weighted)
            unknown = corrected(before, rig.parents, keys, params.get(key) or [])[1]
            body = base.bodies[1 - k] if base is not None and base.bodies else None  # bodies: (target, motion)
            size = None
            if m is not None and not m.hands:
                leg = leg_length(m, side, before)
                ground = ground_level(inputs[port], rig, ground_of(rig, parts)[0], before)
                size = {"auto": auto_scale(leg), "leg_cm": round(leg, 2), "ground": round(ground, 4)}
            sides.append(rig_pair_side(inputs[port].fingerprint, rig.path, keys, rig.parents, before, parts, said,
                                       unknown, body, auto_pose(before, rig.parents, keys, parts),
                                       recognize(rig.names, rig.parents, rest_of(rig), weights=rig.weighted).report(),
                                       size))
            parts_of.append(parts)
        owners = rule_owners()
        rules = []
        for rule, needs in all_rules().items():
            why = {col: ignore_reasons(needs, rig.names, rig.parents, parts, rest_of(rig), rig.weighted)
                   for col, rig, parts in (("src", source, parts_of[0]), ("dst", target, parts_of[1]))}
            keys = {col: joint_keys(rig.names, rig.parents) for col, rig in (("src", source), ("dst", target))}
            # the model's fixed skeleton (JointNeeds.fixed_parts): how many joints each part takes, and the parts it
            # cannot do without — the editor checks the mapping against the chosen rule as it is edited, with the
            # same numbers the solver checks at its cook (retarget_needs.check)
            counts = needs.counts()
            rules.append({"id": rule, "label": rule_label(rule, owners[rule]), "node": owners[rule],
                          "slots": counts,
                          "required": sorted(part for part in counts if GROUP_OF.get(part, "body") not in needs.optional),
                          "ignore": {col: [keys[col][j] for j in sorted(got)] for col, got in why.items()},
                          "why": {col: {keys[col][j]: m.text for j, m in sorted(got.items())} for col, got in why.items()}})
        guessed = [part_names(rig.names, rig.parents, rest_of(rig), rig.weighted) for rig in (source, target)]
        hands = m.hands if m is not None else ()
        return {0: rig_pair_data(*sides, required("legs", hands), auto_rows(*guessed), rules, GENERIC_ID)}

    @classmethod
    def cook(cls, ctx):
        from lab2shot_shared import motion as mo

        from ...data.joints import joint_keys
        from ..kit.retarget import corrected, notes
        from ..kit.retarget import ground_of
        from ..kit.retarget_prepare import auto_scale, effective_scale, ground_level, leg_length, prepared_packet, sizing_for

        ctx.stage("match_joints")
        source, target, m, base = cls._sides(ctx.params, {k: ctx.input(k) for k in ("motion", "target")}, strict=True)
        for note in notes(m):  # 推测的部位要说清推测成了什么
            ctx.say(note.code, **note.params)
        ctx.stage("rest_pose")
        p = {**ctx.params, **cls._effective(ctx.params, source, target, m, base)}
        legs = {"motion": leg_length(m, "src", base.source_before), "target": leg_length(m, "dst", base.target_before)}
        scales = dict(zip(("motion", "target"), effective_scale(p, {k: auto_scale(v) for k, v in legs.items()})))
        out, said = {}, []
        for port, src_port, rig, before, rows_key, ignore_key, parts, who, which in (
                ("source", "motion", source, base.source_before, "motion_pose", "motion_ignore", m.src, i18n.Word("role.motion"), "motion"),
                ("target", "target", target, base.target_before, "target_pose", "target_ignore", m.dst, i18n.Word("role.target"), "target")):
            keys = joint_keys(rig.names, rig.parents)
            reference, unknown, posed = corrected(before, rig.parents, keys, p[rows_key])
            reference = on_own_ground(reference, rig, ground_of(rig, parts)[0])
            if unknown:
                ctx.say("W-RETARGET-POSEJOINT", rig=who, count=len(unknown),
                        joints=i18n.Both.of(lambda: i18n.separator().join(unknown[:8]) + (i18n.t("list.etc") if len(unknown) > 8 else "")))
            have = set(keys)
            ignored = [n for n in dict.fromkeys(p[ignore_key]) if n in have]
            if lost := [n for n in p[ignore_key] if n not in have]:
                ctx.say("W-RETARGETPREP-IGNOREJOINT", rig=who, count=len(lost),
                        joints=i18n.Both.of(lambda: i18n.separator().join(lost[:8]) + (i18n.t("list.etc") if len(lost) > 8 else "")))
            local = mo.local_from_world(np.linalg.inv(rig.placement[0]) @ reference, rig.parents)
            info = cls._reference_info(p, which, rig, base, port == "source")
            sizing = sizing_for(scales[which], rig.placement[0], ground_level(ctx.input(src_port), rig,
                                                                               ground_of(rig, parts)[0], before)) \
                if scales[which] != 1.0 else None
            out[port] = prepared_packet(ctx.input(src_port), ctx.outputs[port], rig,
                                        {part: [keys[j] for j in js] for part, js in parts.items()}, ignored, local,
                                        info, *([sizing] if sizing else []))
            said.append((len(rig.names), len(ignored), posed))
        ctx.say("I-RETARGETPREP-DONE", src_joints=said[0][0], src_ignored=said[0][1], src_posed=said[0][2],
                dst_joints=said[1][0], dst_ignored=said[1][1], dst_posed=said[1][2],
                src_rest=base.source_said, dst_rest=base.target_said)
        ctx.say("I-RETARGETPREP-SIZE", src=scales["motion"], dst=scales["target"], src_leg=legs["motion"],
                dst_leg=legs["target"], src_how=i18n.Word("value.auto" if p.get("motion_scale") is None else "value.typed"),
                dst_how=i18n.Word("value.auto" if p.get("target_scale") is None else "value.typed"))
        return out

    @classmethod
    def _effective(cls, params: dict, source, target, m, base) -> dict:
        """默认自动（kit/retarget_prepare.py effective）：两侧实际用的初始姿势行和忽略名单，作为参数的同名项。没执行过的
        「自动」按这一次的建议（auto_pose、当前忽略规则的 suggest_ignored；规则不在了按通用规则）。"""
        from ...data.joints import joint_keys
        from ..kit.retarget import rest_of
        from ..kit.retarget_needs import GENERIC, all_rules, suggest_ignored
        from ..kit.retarget_prepare import auto_pose, effective

        needs = all_rules().get(params.get("ignore_rule") or "", GENERIC)
        auto = {"pose": {}, "ignore": {}}
        for side, rig, before, parts in (("motion", source, base.source_before, m.src),
                                         ("target", target, base.target_before, m.dst)):
            auto["pose"][side] = auto_pose(before, rig.parents, joint_keys(rig.names, rig.parents), parts)
            auto["ignore"][side] = suggest_ignored(needs, rig.names, rig.parents, parts, rest_of(rig), rig.weighted)
        got = effective(params, auto)
        return dict(zip(("motion_pose", "target_pose", "motion_ignore", "target_ignore"), got))

    @classmethod
    def _reference_info(cls, params: dict, which: str, rig, base, source: bool) -> dict:
        """PREPARED_KEY 里参考姿态的来源：bind / first / frame 与帧号（动作的绑定姿势不是站姿、自动挑了一帧时是那一帧），
        和节点版本。"""
        rest = params[f"{which}_rest"]
        info = {"reference": rest, "version": cls.version}
        if source and base.picked >= 0:
            info["frame"] = int(rig.frames[base.picked])
        elif rest == "first":
            info["frame"] = int(rig.frames[0])
        elif rest == "frame":
            info["frame"] = int(params[f"{which}_rest_frame"])
        return info


class RetargetPost(NodeDef):
    """重定向后处理：解算器交出的骨架动画上，明着做位移和髋高，并还原尺寸。

    解算器（学习式重定向）原生输出什么就是什么：身体的旋转、模型自己的根位移。预处理把目标缩放过（「目标尺寸」不是 1）
    时，解算结果是缩放后的大小：这里先按骨架上记的系数还原（骨骼平移、髋的位置，以及交出的骨架的绑定姿势），输出和原
    目标角色同尺寸，能直接蒙到原角色上。然后按「位移轨迹」二选一：跟随源动作 = 髋高按两边腿长的比和两边脚踝离脚底的差
    换算（与「动作重定向」同一套，kit/retarget.py measure / ground_of、motion.hips_path），水平位移按「水平位移」：
    按腿长缩放（默认：步子按腿长比例放大缩小，脚不滑）或按画面原样（1 : 1，贴合视频里那个人的位置）；保留模型 = 模型给的
    轨迹不动（只还原尺寸）。两种都能再加「高度偏移」。改的是髋这根骨骼的动画曲线，不是 USD 的变换（与「3D 变换」不同）；
    落地接「自动落地」。部位、参考姿态和两侧尺寸都从预处理的标注里取，计算都在原尺寸里做。
    """

    id = "retarget_post"
    tool = False  # a stage of the retarget cards, as RetargetPrepare
    category = "scene_convert"
    version = 6  # 6：「跟随源动作」的高度沿源站立帧的身体上方量，「按画面原样」逐帧跟着脚抬（motion.up_over / hips_path）；5：只为让半途代码的缓存重算；4：「按腿长缩放」时起伏也按腿长放大（跳跃、跨栏）；默认「保留模型」；3：按预处理的尺寸还原（含站到地面的平移）；水平位移默认按腿长缩放
    inputs = (Port("skeleton", "scene.skeleton"),
              Port("source", "scene.skeleton|scene.character"),
              Port("target", "scene.character|scene.skeleton"))
    outputs = (Port("skeleton", "scene.skeleton"),)

    class Params(NodeParams):
        # 默认不修正：解算器原样给的轨迹；要按源动作修正髋的轨迹（按腿长放大步子和起伏、贴合画面）时手动选
        root_motion: Literal["source", "model"] = P(
            "model", group="translate")
        stride: Literal["legs", "plate"] = P(
            "legs", group="translate",
            applies=Param("root_motion").one_of("source"))
        height_offset: float = copy.deepcopy(Retarget.Params.model_fields["height_offset"])

    @classmethod
    def info(cls, params, inputs):
        return Info.merge(inputs.get("skeleton", []))

    @classmethod
    def cook(cls, ctx):
        from lab2shot_shared import motion as mo

        from ...data.animation import read_rig, skeleton_animation, sole_height
        from ..kit.retarget import MEASURABLE_CM, body_up, ground_of, measure
        from ..kit.retarget_prepare import bases, prepared, restore_size

        p = ctx.params
        src_sized = prepared(ctx.input("source"), side=i18n.Word("role.source"), least_frames=2)
        dst_sized = prepared(ctx.input("target"), side=i18n.Word("role.target"))
        src, dst = src_sized.unscaled(), dst_sized.unscaled()  # 以下都在原尺寸里算
        mapping, base = bases(src, dst)
        source, goal = src.rig, dst.rig
        packet = ctx.input("skeleton")
        anim = read_rig(packet, goal.path)
        if anim.frames != source.frames or list(anim.names) != list(goal.names) or list(anim.parents) != list(goal.parents):
            raise Invalid(Msg("E-RETARGETPOST-MISMATCH", frames=len(anim.frames), joints=len(anim.names),
                              source_frames=len(source.frames), target_joints=len(goal.names)))
        hip = mapping.dst["hips"][0]
        local = dst_sized.sizing.inverse().local(anim.local, anim.parents)  # 解算结果是变换后的目标：骨骼平移、髋的位置还原
        inv = np.linalg.inv(anim.placement)

        def into(path):  # world positions [F,3] into the skeleton's space
            return (inv[:, :3, :3] @ path[..., None])[..., 0] + inv[:, :3, 3]

        info = {"root_motion": p["root_motion"], "height_offset_cm": p["height_offset"], "target_scale": dst_sized.scale,
                "source_scale": src_sized.scale}
        if p["root_motion"] == "source":
            positions = source.world()[..., :3, 3]
            source_cm = float(np.median(measure("legs", mapping, "src", positions)))
            target_cm = float(measure("legs", mapping, "dst", goal.world()[:1, ..., :3, 3])[0])
            if min(source_cm, target_cm) <= MEASURABLE_CM:
                raise Invalid(Msg("E-RETARGET-NOLENGTH", what=i18n.Word("retarget.scale_by.legs"), rig=i18n.Word("role.motion" if source_cm <= MEASURABLE_CM else "role.target")))
            feet_s, feet_t = ground_of(source, mapping.src)[0], ground_of(goal, mapping.dst)[0]
            # 脚底在包里的（缩放过的）网格上量，再换回原尺寸：同一个世界里的长度正好乘了系数
            soles = []
            for pkt, sized, feet, body in ((ctx.input("source"), src_sized, feet_s, base.bodies[1]),
                                           (ctx.input("target"), dst_sized, feet_t, base.bodies[0])):
                h = sole_height(pkt, sized.rig, feet, body_up(sized.rig, body))
                soles.append(None if h is None else h / sized.scale)
            sole_delta = soles[1] - soles[0] if None not in soles else 0.0
            ratio = target_cm / source_cm
            up = base.bodies[1].up_over(positions)
            # 「按腿长缩放」：整段动作按腿长等比放大——步子（水平）和起伏（髋离地的高度：跳起、跨栏、蹲下）一起，脚不滑、
            # 跳到半空脚也不会穿到地下（只抬一个常数时，大个子的腿按大比例收起，髋却只升到源那么高）。「按画面原样」：
            # 水平 1 : 1 贴着画面，高度跟着脚抬（逐帧按髋离最低脚踝的高度，楼梯、坡地上站得住，hips_path lift）
            legs = p["stride"] == "legs"
            path = mo.hips_path(positions, (mapping.src["l.thigh"][0], mapping.src["r.thigh"][0]), feet_s, ratio, not legs,
                                sole_delta + p["height_offset"], up)
            if legs:  # 水平位移从第一帧起按腿长比例：步子跟着腿长变，脚不滑
                u = np.asarray(up, np.float64) / np.linalg.norm(up)
                flat = path - (path @ u)[:, None] * u
                path = path + (ratio - 1.0) * (flat - flat[:1])
            local = mo.set_hips(local, anim.parents, {}, hip, (mapping.dst["l.thigh"][0], mapping.dst["r.thigh"][0]),
                                into(path))
            info.update(leg_ratio=ratio, sole_delta_cm=sole_delta, stride=p["stride"])
            ctx.say("I-RETARGETPOST-SOURCE", ratio=ratio, target=target_cm, source=source_cm, sole=sole_delta,
                    offset=p["height_offset"],
                    stride=(i18n.Word("retarget.stride.legs", ratio=f"{ratio:.3f}") if p["stride"] == "legs" else i18n.Word("retarget.stride.as_is")))
        else:
            if p["height_offset"]:
                path = anim.world(local)[:, hip, :3, 3].copy()
                path[:, 1] += p["height_offset"]
                local = mo.set_world(local, anim.parents, {}, {hip: into(path)})
            ctx.say("I-RETARGETPOST-MODEL", offset=p["height_offset"])
        out = skeleton_animation(packet, ctx.outputs["skeleton"], anim, local, {"retarget_post": info})
        if not dst_sized.sizing.identity:
            restore_size(out, goal.path, dst_sized.sizing)  # 交出的骨架的绑定姿势也回到原尺寸、原位置
            ctx.say("I-RETARGETPOST-SIZE", scale=dst_sized.scale)
        return {"skeleton": out}


NODES = (RetargetPrepare, RetargetPost)
