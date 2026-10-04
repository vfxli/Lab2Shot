"""Driving a model's own skeleton from a production rig, and transferring the model's motion back onto that rig.

Both tasks of the rig-motion family (families/rig_motion.py), motion generation and cleanup, follow the same steps:
the node reads the character's skeleton from the scene, matches the rig's joints to the model's through its
「对应关系」 (the body parts of the two skeletons), sends the rig's world poses through the shared worker contract (lab2shot_worker.rig_motion), and applies the
result to the rig, leaving joints the model does not have untouched. Only the middle step differs: one generates
motion (between an animator's keys, or from a text prompt alone), the other repairs broken frames.

This module therefore holds the shared parts (the 「对应关系」, joint matching, skeleton choice and the job), and the
family adds only its parameters and its interpretation of the result. `RigModel` is a building block rather than a
node family, so it does not subclass NodeDef (the family is `class RigMotion(RigModel, WorkerNode)`), and it lives
in kit/ rather than families/, which holds only node families that nodes subclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...data.joints import LANDMARKS, LEGS
from ...errors import Invalid
from ... import i18n
from ...messages import Msg

from ..base import P, Port
from ..families.base import Job
from ..handles import RigPair
from .rig_map import rig_map_param, skeleton_choice


@dataclass(frozen=True)
class ModelJoint:
    """A joint of a model's skeleton that a rig joint drives: its body part (data.joints.PARTS; the rows of a chain
    part, "spine" or "neck", in the model's order) and the joint its bone points to for the rest-pose alignment
    ("up" for a head that stands straight, None to take its parent's)."""

    name: str
    part: str
    aim: str | None = None


ARM_PARTS = ("clavicle", "upperarm", "forearm", "hand")
LEG_PARTS = ("thigh", "shin", "foot", "toe")
REQUIRED_PARTS = ("hips", *LEGS, "l.foot", "r.foot")
# what a body part hangs from, for a model skeleton declared by parts only (RigModel.model_parents): the first of these
# the model has
BODY_PARENT = {"spine": ("hips",), "chest": ("spine", "hips"), "neck": ("chest", "spine"), "head": ("neck", "chest", "spine"),
               "jaw": ("head",), "eye": ("head",), "clavicle": ("chest", "spine"), "upperarm": ("clavicle", "chest", "spine"),
               "forearm": ("upperarm",), "hand": ("forearm",), "thigh": ("hips",), "shin": ("thigh",), "foot": ("shin",),
               "toe": ("foot",), **{f: ("hand",) for f in ("thumb", "index", "middle", "ring", "pinky")}}


def _body_parent(part: str, have: dict) -> str | None:
    """The part `part` hangs from among those the model has so far (`have`), on its own side: BODY_PARENT."""
    side, _, name = part.rpartition(".")
    for up in BODY_PARENT.get(name, ()):
        for candidate in ((f"{side}.{up}",) if side else ()) + (up,):
            if candidate in have:
                return candidate
    return None


def humanoid_joints(trunk: tuple[tuple[str, str, str | None], ...], legs: tuple[str, str, str, str],
                    arms: tuple[str, str, str, str] = ("Shoulder", "Arm", "ForeArm", "Hand")) -> tuple[ModelJoint, ...]:
    """A model skeleton named the Mixamo way (LaFAN1, SOMA): the trunk's (name, part, aim) rows, then for the left and
    the right side the arm (clavicle to hand) and the leg (thigh to toe), each joint "Left" / "Right" + its name,
    aimed at the next one of its chain."""
    out = [ModelJoint(*row) for row in trunk]
    for side, s in (("Left", "l"), ("Right", "r")):
        for names, parts in ((arms, ARM_PARTS), (legs, LEG_PARTS)):
            out += [ModelJoint(side + name, f"{s}.{part}", side + names[i + 1] if i + 1 < len(names) else None)
                    for i, (name, part) in enumerate(zip(names, parts))]
    return tuple(out)


def part_labels(joints: tuple[ModelJoint, ...]) -> list[str]:
    """The label of each model joint in messages (左大腿, 脊柱 2, ...)."""
    from ...data.joints import part_label

    out = []
    for j in joints:
        label = part_label(j.part)
        chain = [k for k in joints if k.part == j.part]
        out.append(f"{label} {chain.index(j) + 1}" if len(chain) > 1 else label)
    return out


def mapping_param(follows: tuple[str, ...] = ()):
    """The 「对应关系」 of a node that drives a model's skeleton: which of the person's rig joints drives each body part
    of its model, edited in the view as every such node and 「动作重定向」 edit theirs (the "rig_pair" handle,
    kit/rig_map.py): the person's skeleton on one side, the model's joints on the other, fixed by the node (a tree of
    names, no positions). Empty = every part guessed.

    `follows`: the parameters of this node whose change replaces the model's side (Kimodo's 「模型」 selects among
    three skeletons: SOMA with 30 joints, SMPL-X with 22, G1 with 34). A node that declares them implements
    `joints_of(params)`, and `derive` rebuilds the rows from it (the same mechanism by which LensDistortion's
    distortion parameters follow 「镜头模型」; the framework provides only P(derived_from) for this purpose)."""
    # the parameters it follows go with the question too: the editor's model side is the model those select
    return rig_map_param(("character", "skeleton", *follows), follows)


def skeleton_param():
    """The 骨骼 choice every rig-driven node has: which character in the scene this node works on."""
    return P(None, widget="choice", group="people", choices_from=("character",), worker=False)


class RigModel:
    """Shared parts of a node that drives a model's skeleton from a production rig (see the module docstring).

    The family mixes it in ahead of WorkerNode: `class RigMotion(RigModel, WorkerNode)`. It declares the ports common
    to both tasks (a bare skeleton or a skinned character, as stored in the file; the output is of the same kind), the
    model's joints (`joints`), and the three shared steps: reading the rig (`rig`), matching the joints
    (`driven`) and writing the job for the worker (`send`).
    """

    # A bare skeleton or a skinned character, as stored in the file; the output is of the same kind.
    inputs = (Port("character", "scene.skeleton|scene.character", words="family.rig_motion.character"),)
    outputs = (Port("character", "scene.skeleton|scene.character", type_from="input:character", words="family.rig_motion.character"),)
    joints: tuple[ModelJoint, ...] = ()  # the model joints a rig can drive (the model side of its 「对应关系」)
    # the 「对应关系」 edited in the view: the person's skeleton against the model's own (fixed, drawn as a tree)
    handles = (RigPair(src="character", dst=None, src_skeleton="skeleton", mapping="mapping"),)

    @classmethod
    def joints_of(cls, params: dict) -> tuple[ModelJoint, ...]:
        """本次使用的模型骨架。默认返回节点声明的骨架；每套权重对应一副骨架的节点（Kimodo：
        SOMA 30 / SMPL-X 22 / G1 34）按「模型」参数返回对应骨架，「对应关系」里模型那一侧随之更换
        （mapping_param(follows=…) + derive）。"""
        return cls.joints

    @classmethod
    def model_parts(cls, params: dict) -> dict[str, list[str]]:
        """The model's joints by body part, in its order (a chain part: its joints from the root outwards)."""
        out: dict[str, list[str]] = {}
        for j in cls.joints_of(params):
            out.setdefault(j.part, []).append(j.name)
        return out

    @classmethod
    def derive(cls, params: dict) -> dict:
        """「模型」变化时重建对应关系里模型那一侧：每个部位已选的人物关节（src）保留，模型关节（dst）换成新模型的，
        新模型没有的部位去掉。值为空（全自动）时不用动。仅对声明了 follows 的节点调用。"""
        spec = next((q for q in cls.param_specs() if q["name"] == "mapping"), None)
        if not spec or not spec["derived_from"] or params.get("mapping") is None:
            return {}
        parts = cls.model_parts(params)
        chosen = {r["part"]: r.get("src") or [] for r in params["mapping"] if isinstance(r, dict) and "part" in r}
        return {"mapping": [{"part": p, "src": list(chosen[p]), "dst": joints} for p, joints in parts.items()
                            if p in chosen]}

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """The 骨骼 choice: the skeletons of the scene wired in."""
        src = inputs.get("character")
        return skeleton_choice(src, "skeleton") if src is not None else {}

    @classmethod
    def model_parents(cls, params: dict) -> list[int]:
        """The model skeleton's hierarchy, for the view's tree of its side: the model declares joints by body part
        only (ModelJoint), so each joint hangs from the joint before it in its chain, a chain's first joint from the
        last joint of the part the body hangs it from (BODY_PARENT: the spine from the hips, an arm from the chest, a
        finger from the hand), the hips at the root (a part whose parent part the model lists later is a root too)."""
        joints = cls.joints_of(params)
        last: dict[str, int] = {}
        out = []
        for k, j in enumerate(joints):
            if j.part in last:
                out.append(last[j.part])
            else:
                up = _body_parent(j.part, last)
                out.append(last[up] if up else -1)
            last[j.part] = k
        return out

    @classmethod
    def handle_data(cls, params: dict, inputs: dict) -> dict[int, dict]:
        """The 「对应关系」 in the view (the "rig_pair" handle, kit/rig_map.py rig_pair_data): the person's skeleton in
        its bind pose with its body parts as the cook reads them (side_parts: the rows given, the guess for the rest),
        and the model's fixed skeleton (names, hierarchy, parts) beside it."""
        from ...data.animation import bind_skeleton
        from ...data.joints import auto_rows, joint_keys, part_names
        from ...data.skeleton_recognition import recognize
        from .rig_map import fixed_side, rig_pair_data, rig_pair_side, side_parts

        src = inputs.get("character")
        found = bind_skeleton(src, params.get("skeleton")) if src is not None else None
        if not found:
            return {}
        names = [j.name for j in cls.joints_of(params)]
        model = cls.model_parts(params)
        model_parts = {p: [names.index(n) for n in js] for p, js in model.items()}
        rows = [{**r, "dst": []} for r in params.get("mapping") or [] if isinstance(r, dict)]
        rest = np.asarray(found["world"], np.float64)[:, :3, 3]  # 识别引擎按位置、对称一起判断（data/skeleton_recognition.py）
        weights = found.get("weighted")  # 不驱动任何顶点的关节不参与对应（data/skeleton_recognition.py）
        parts = side_parts(found["names"], found["parents"], rows, "src", rest, weights)
        person = rig_pair_side(src.fingerprint, found["path"], joint_keys(found["names"], found["parents"]),
                               found["parents"], found["world"], parts, i18n.t("retarget.rest.bind"),
                               recognition=recognize(found["names"], found["parents"], rest, weights=weights).report())
        guessed = part_names(found["names"], found["parents"], rest, weights)
        return {0: rig_pair_data(person, fixed_side(names, cls.model_parents(params), model_parts), REQUIRED_PARTS,
                                 auto_rows(guessed, model))}

    @classmethod
    def mapping(cls, params: dict, names: list[str], parents, rest=None, weights=None) -> dict[str, int]:
        """Model joint -> rig joint index. A part the 「对应关系」 lists takes the rig joints given there, every other
        part the guess (data/joints.py guess); a chain part's model joints go over its rig joints by their place along
        it (joints.spread), so the result reverses exactly back onto the rig (motion.Retarget.to_production) — the
        spreading by length of 「动作重定向」 is not reversible and does not apply here. The body parts every model needs
        are validated; one rig joint may drive one model joint only. The worker gets the result as a table model joint
        -> rig joint (lab2shot_worker.rig_motion.write_job)."""
        from ...data.joints import CHAINS, auto_rows, check_mapping, merged_rows, part_joints, part_label, part_names, spread

        joints = cls.joints_of(params)
        labels = dict(zip((j.name for j in joints), part_labels(joints)))
        parts = cls.model_parts(params)
        given = params["mapping"] or []
        unknown = sorted({n for r in given for n in r["dst"]} - set(labels))
        if unknown:
            raise Invalid(Msg("E-RIG-UNKNOWNJOINTS", joints=unknown))
        names = list(names)
        # the rows a cook uses and their joints: the same two functions as 「动作重定向」 (merged_rows, part_joints);
        # only the person's side is read, the model's side is the node's own
        rows, _ = merged_rows([{**r, "dst": []} for r in given],
                              auto_rows(part_names(names, parents, rest, weights), {}))
        check_mapping(rows, {"src": (names, parents, i18n.Word("role.person"))})
        rig_parts = foot_hinges(part_joints(rows, names, parents, "src"), parents)
        out: dict[str, int] = {}
        for part, model in parts.items():
            rig = rig_parts.get(part, [])
            picks = spread(len(model), rig) if part in CHAINS else [rig[0] if rig else None] * len(model)
            for m, r in zip(model, picks):
                if r is not None:
                    out[m] = r
        last = {j.part: j.name for j in joints}
        missing = [labels[last[p]] for p in REQUIRED_PARTS if last.get(p) not in out]
        if missing:
            raise Invalid(Msg("E-RIG-MISSINGPARTS", parts=missing))
        part_of = {j.name: j.part for j in joints}
        taken: dict[int, str] = {}
        for m, r in out.items():
            if r in taken:
                raise Invalid(Msg("E-MAP-SHARED", first=part_label(part_of[taken[r]]), second=part_label(part_of[m]),
                                  joint=names[r], side=i18n.Both.of(lambda: i18n.Word("role.person"))))
            taken[r] = m
        return out

    @classmethod
    def rig(cls, ctx):
        """The rig this node works on, sampled at every frame of the packet."""
        from ...data.animation import read_rig

        return read_rig(ctx.input("character"), ctx.params["skeleton"])

    @classmethod
    def driven(cls, ctx, rig) -> tuple[dict[str, int], dict[str, str]]:
        """The match between the rig's joints and the model's: (model joint -> rig joint index, body part -> model
        joint)."""
        from lab2shot_shared import motion as mo

        pairs, parts = (cls.mapping(ctx.params, rig.names, rig.parents, np.asarray(rig.bind, np.float64)[:, :3, 3],
                                    rig.weighted),
                        {j.part: j.name for j in cls.joints_of(ctx.params)})
        # the worker aligns the rig by its body (motion.Retarget.align: Body.of_hierarchy); a rig with no trunk to find
        # is taken as standing, which a folded rest pose is not
        legs = [pairs[parts[x]] for x in LEGS]
        marks = [pairs[parts[x]] for x in LANDMARKS if x in parts and parts[x] in pairs]
        # the same body the worker aligns by: the same landmarks, the rig's rest as sent (rig.skeleton())
        if not mo.Body.of_hierarchy(rig.parents, (legs[0], legs[2]), (legs[1], legs[3]), marks,
                                    rig.skeleton().rest_positions).trunk:
            ctx.say("W-BODY-NOTRUNK", rig=i18n.Word("role.person"))
        return pairs, parts

    @classmethod
    def send(cls, ctx, rig, pairs: dict[str, int], parts: dict[str, str], frames: list[int],
             poses: np.ndarray) -> Job:
        """The job for the worker: the rig, the given frames, and the rig's world pose on each of them."""
        from lab2shot_worker.rig_motion import write_job

        # 「帧率」（families/rig_motion.py motion_fps_param）：worker 按它把帧号换算成秒，再放到模型自己的时间轴上
        motion = write_job(ctx.work / "motion.npz", rig.skeleton(), frames, poses, float(ctx.params["fps"]),
                           pairs, {j.name: j.aim for j in cls.joints_of(ctx.params) if j.aim},
                           tuple(parts[x] for x in LEGS), tuple(parts[x] for x in LANDMARKS if x in parts))
        return Job(None, inputs={"motion": motion}, notes={"rig": rig, "keys": frames, "pairs": pairs, "parts": parts})


def foot_hinges(rig_parts: dict[str, list[int]], parents) -> dict[str, list[int]]:
    """The rig joint that turns each foot: the one its toe hangs from. The 「对应关系」 names the ankle (l.foot), and in
    most rigs the toe hangs right under it; SAM 3D Body's MHR hangs it under three joints of the foot's own
    (LeftFoot → l_talocrural → l_subtalar → l_transversetarsal → LeftToeBase), and the ankle's bend happens there, at
    l_talocrural and l_subtalar (12° and 7° on a walk), while LeftFoot itself hardly turns (4°). Driven by LeftFoot,
    the model's ankle stood still and its toe took the whole bend: StableMotion called every frame of such a walk broken
    (LaFAN1's walk with those feet: 100%; without them: 1%). Driven by the toe's parent, the model's ankle turns as the
    foot does, and the result goes back onto that same joint (motion.Retarget.to_production), the joints between keeping
    their own turn."""
    out = dict(rig_parts)
    for side in ("l", "r"):
        foot, toe = out.get(f"{side}.foot"), out.get(f"{side}.toe")
        if not foot or not toe:
            continue
        hinge, above = int(parents[toe[0]]), int(parents[toe[0]])
        while above >= 0 and above != foot[0]:
            above = int(parents[above])
        if hinge != foot[0] and above == foot[0]:
            out[f"{side}.foot"] = [hinge]
    return out


def body_joints(body: str) -> tuple[ModelJoint, ...]:
    """The joints of a model whose skeleton belongs to the SMPL family ("smpl", "smplh", "smplx"): every joint
    of that body, its body part, and the joint its bone points at.

    `humanoid_joints` provides the same table written by hand for Mixamo-style naming; this function derives it from
    the body itself, using the table data/smpl.py maintains (rows_of) and the hierarchy (motion.aims_toward), so that the model side of a node's 「对应关系」
    and the automatic guess that fills it (RigModel.choices / mapping: guess() over the same body) list the same
    joints in the same order. Read by 「StableMotion 动捕清理」 (SMPL) and 「Kimodo 动作生成」 (its SMPL-X 22 weights)."""

    from lab2shot_shared import motion as mo
    from lab2shot_shared import smpl as S

    from ...data.smpl import rows_of

    b = S.body(body)
    part_of = dict(rows_of(b))
    aims = mo.aims_toward(b.parents)
    return tuple(ModelJoint(name, part_of[name], None if aims[j] is None else b.names[aims[j]])
                 for j, name in enumerate(b.names) if name in part_of)
