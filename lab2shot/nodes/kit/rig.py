"""Driving a model's own skeleton from a production rig, and transferring the model's motion back onto that rig.

Both tasks of the rig-motion family (families/rig_motion.py), motion generation and cleanup, follow the same steps:
the node reads the character's skeleton from the scene, matches the rig's joints to the model's through the joint
table, sends the rig's world poses through the shared worker contract (lab2shot_worker.rig_motion), and applies the
result to the rig, leaving joints the model does not have untouched. Only the middle step differs: one generates
motion (between an animator's keys, or from a text prompt alone), the other repairs broken frames.

This module therefore holds the shared parts (the joint table, joint matching, skeleton choice and the job), and the
family adds only its parameters and its interpretation of the result. `RigModel` is a building block rather than a
node family, so it does not subclass NodeDef (the family is `class RigMotion(RigModel, WorkerNode)`), and it lives
in kit/ rather than families/, which holds only node families that nodes subclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...errors import Invalid
from ...messages import Msg

from ...data.units import DEFAULT_FPS
from ..base import NodeParams, P, Port
from ..families.base import Job


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
LEGS = ("l.thigh", "l.shin", "r.thigh", "r.shin")
REQUIRED_PARTS = ("hips", *LEGS, "l.foot", "r.foot")


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
    """The label of each model joint as shown in the joint table (左大腿, 脊柱 2, ...)."""
    from ...data.joints import part_label

    out = []
    for j in joints:
        label = part_label(j.part)
        chain = [k for k in joints if k.part == j.part]
        out.append(f"{label} {chain.index(j) + 1}" if len(chain) > 1 else label)
    return out


class JointMap(NodeParams):
    """A row of the joint table: a model joint and the rig joint that drives it."""

    name: str = P(..., label="模型关节", widget="fixed")
    label: str = P(..., label="部位", widget="fixed")
    joint: str | None = P(None, label="人物关节", widget="choice")


def mapping_rows(joints: tuple[ModelJoint, ...]) -> list[dict]:
    """关节对照表的行，模型的每个关节对应一行。"""
    return [{"name": j.name, "label": label, "joint": None} for j, label in zip(joints, part_labels(joints))]


def mapping_param(joints: tuple[ModelJoint, ...], follows: tuple[str, ...] = ()):
    """The joint table of a node that drives a model's skeleton: one row per joint of its model.

    `follows`: the parameters of this node whose change replaces the whole table (Kimodo's 「模型」 selects among
    three skeletons: SOMA with 30 joints, SMPL-X with 22, G1 with 34). A node that declares them implements
    `joints_of(params)`, and `derive` rebuilds the rows from it. This is the same mechanism by which LensDistortion's
    distortion parameters follow 「镜头模型」; the framework provides only P(derived_from) for this purpose."""
    return P(mapping_rows(joints), label="关节映射", widget="table", group="人物", choices_from=("character", "skeleton"),
             worker=False, validate_default=True, derived_from=follows)


def skeleton_param():
    """The 骨骼 choice every rig-driven node has: which character in the scene this node works on."""
    return P(None, label="骨骼", widget="choice", group="人物", choices_from=("character",), worker=False,
             placeholder="第一个")


class RigModel:
    """Shared parts of a node that drives a model's skeleton from a production rig (see the module docstring).

    The family mixes it in ahead of WorkerNode: `class RigMotion(RigModel, WorkerNode)`. It declares the ports common
    to both tasks (a bare skeleton or a skinned character, as stored in the file; the output is of the same kind), the
    joint table's rows (`joints`), and the three shared steps: reading the rig (`rig`), matching the joints
    (`driven`) and writing the job for the worker (`send`).
    """

    # A bare skeleton or a skinned character, as stored in the file; the output is of the same kind.
    inputs = (Port("character", "scene.skeleton|scene.character", "动画"),)
    outputs = (Port("character", "scene.skeleton|scene.character", "动画", type_from="input:character"),)
    joints: tuple[ModelJoint, ...] = ()  # the model joints a rig can drive (its joint table's rows)

    @classmethod
    def joints_of(cls, params: dict) -> tuple[ModelJoint, ...]:
        """本次使用的模型骨架。默认返回节点声明的骨架；每套权重对应一副骨架的节点（Kimodo：
        SOMA 30 / SMPL-X 22 / G1 34）按「模型」参数返回对应骨架，关节对照表随之更换
        （mapping_param(follows=…) + derive）。"""
        return cls.joints

    @classmethod
    def derive(cls, params: dict) -> dict:
        """「模型」变化时重新生成关节对照表；已选定的人物关节按模型关节名保留（两副骨架中同名的关节
        无需重选）。仅对声明了 follows 的节点调用。"""
        spec = next((q for q in cls.param_specs() if q["name"] == "mapping"), None)
        if not spec or not spec["derived_from"]:
            return {}
        chosen = {r["name"]: r.get("joint") for r in params.get("mapping") or [] if isinstance(r, dict)}
        return {"mapping": [{**row, "joint": chosen.get(row["name"])} for row in mapping_rows(cls.joints_of(params))]}

    @classmethod
    def rows(cls, params: dict) -> list[tuple[str, str]]:
        return [(j.name, j.part) for j in cls.joints_of(params)]

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        from ...data.animation import skeletons
        from ...data.joints import auto_mapping

        src = inputs.get("character")
        found = skeletons(src) if src is not None else []
        if not found:
            return {}
        chosen = next((s for s in found if s["path"] == params.get("skeleton")), found[0])
        auto = auto_mapping(cls.rows(params), chosen["joints"], chosen["parents"])
        return {"skeleton": {"options": [s["path"] for s in found], "auto": found[0]["path"]},
                "mapping": {"options": chosen["joints"], "auto": {n: v or "" for n, v in auto.items()}, "none": "不映射"}}

    @classmethod
    def mapping(cls, params: dict, names: list[str], parents) -> dict[str, int]:
        """Model joint -> rig joint index. Choices in the table take precedence over the automatic guess; the body
        parts required by every model are validated."""
        from ...data.joints import auto_mapping

        joints = cls.joints_of(params)
        labels = dict(zip((j.name for j in joints), part_labels(joints)))
        chosen = {r["name"]: r["joint"] for r in params["mapping"]}
        unknown = sorted(set(chosen) - set(labels))
        if unknown:
            raise Invalid(Msg("E-RIG-UNKNOWNJOINTS", joints=unknown))
        auto = auto_mapping(cls.rows(params), names, parents)
        out: dict[str, int] = {}
        for j in joints:
            name = auto[j.name] if chosen.get(j.name) is None else chosen[j.name] or None
            if name is None:
                continue
            if name not in names:
                raise Invalid(Msg("E-RIG-NOJOINT", part=labels[j.name], joint=name))
            out[j.name] = names.index(name)
        parts = {j.part: j.name for j in joints}
        missing = [labels[parts[p]] for p in REQUIRED_PARTS if parts.get(p) not in out]
        if missing:
            raise Invalid(Msg("E-RIG-MISSINGPARTS", parts=missing))
        taken: dict[int, str] = {}
        for m, r in out.items():
            if r in taken:
                raise Invalid(Msg("E-RIG-SHARED", first=labels[taken[r]], second=labels[m], joint=names[r]))
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
        return cls.mapping(ctx.params, rig.names, rig.parents), {j.part: j.name for j in cls.joints_of(ctx.params)}

    @classmethod
    def send(cls, ctx, rig, pairs: dict[str, int], parts: dict[str, str], frames: list[int],
             poses: np.ndarray) -> Job:
        """The job for the worker: the rig, the given frames, and the rig's world pose on each of them."""
        from lab2shot_worker.rig_motion import write_job

        # DEFAULT_FPS 为固定时基（帧号即时间码）；模型重采样需要以秒为单位的时间线
        motion = write_job(ctx.work / "motion.npz", rig.skeleton(), frames, poses, DEFAULT_FPS,
                           pairs, {j.name: j.aim for j in cls.joints_of(ctx.params) if j.aim},
                           tuple(parts[x] for x in LEGS))
        return Job(None, inputs={"motion": motion}, notes={"rig": rig, "keys": frames, "pairs": pairs, "parts": parts})


def body_joints(body: str) -> tuple[ModelJoint, ...]:
    """The joint table of a model whose skeleton belongs to the SMPL family ("smpl", "smplh", "smplx"): every joint
    of that body, its body part, and the joint its bone points at.

    `humanoid_joints` provides the same table written by hand for Mixamo-style naming; this function derives it from
    the body itself, using the table data/smpl.py maintains (rows_of, aims_of), so that a node's editable joint table
    and the automatic guess that fills it (RigModel.choices / mapping: auto_mapping over the same rows) list the same
    joints in the same order. Read by 「StableMotion 动捕清理」 (SMPL) and 「Kimodo 动作生成」 (its SMPL-X 22 weights)."""

    from lab2shot_shared import smpl as S

    from ...data.smpl import aims_of, rows_of

    b = S.body(body)
    part_of = dict(rows_of(b))
    aims = aims_of(b.parents)
    return tuple(ModelJoint(name, part_of[name], None if aims[j] is None else b.names[aims[j]])
                 for j, name in enumerate(b.names) if name in part_of)
