"""场景：合成场景、按种类取出、按分区取出、删除属性、相机空间转换、3D 变换、自动落地（可先按重力方向放平）、重定时、
重采样曲线、标准人、烘焙成模型、提取骨架、动作重定向、线性蒙皮变形、场景投影成 2D。"""

from __future__ import annotations

import json
from typing import Literal

import numpy as np

from ..kit.ports import rgb_port
from ..kit.ports import normal_port
from ...errors import Invalid
from ...recent import Recent, packet_key
from ...messages import Msg
from ..base import Info, NodeDef, NodeParams, P, Port, colorspace_param
from ..tags import NONCOMMERCIAL
from ..expects import DistinctNames, SameShot
from ..handles import Places, Poses
from ...data.types import DEFORMING
from ...availability import Not
from ..applies import Licence, Param, Wired, fact
from ..kit.rig_map import (JointPose, PartMap, expression_choice, pose_param, rig_map_choice, rig_map_param,
                           side_parts, skeleton_handle)


class UsdPack(NodeDef):
    id = "core.usd_pack"
    category = "scene_build"
    inputs = (Port("scene", "scene|scene[]", "场景", multi=True,
                   expects=(DistinctNames(),)),)
    outputs = (Port("scene", "scene", "场景"),)

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import pack

        return {"scene": pack(ctx.inputs["scene"], ctx.outputs["scene"])}


class Take(NodeDef):
    id = "core.take"
    category = "scene_build"
    inputs = (Port("scene", "scene", "场景"),)
    # 每种类型一个输出，类型与导入节点一致（nodes/formats.py SELECTIONS）
    outputs = (Port("camera", "scene.camera", "相机", narrows="input:scene", may_be_empty=True), Port("models", "scene.model", "模型", narrows="input:scene", may_be_empty=True),
               Port("points", "scene.points", "点云", narrows="input:scene", may_be_empty=True), Port("curves", "scene.curves", "三维曲线", narrows="input:scene", may_be_empty=True),
               Port("skeletons", "scene.skeleton", "骨架动画", narrows="input:scene", may_be_empty=True),
               Port("characters", "scene.character", "蒙皮角色", narrows="input:scene", may_be_empty=True))

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import only_kind
        from ...data.types import SCENE_KINDS
        from ..base import empty_packet
        from ..formats import SELECTIONS

        src, out, none = ctx.input("scene"), {}, []
        for port in cls.outputs:  # 只取出有连线的种类，其余种类完全不取出
            if port.name not in ctx.wanted:
                continue
            kind = SELECTIONS[port.name]
            packet = only_kind(src, kind, ctx.outputs[port.name])
            if packet is None:
                none.append(SCENE_KINDS[kind].label)
                packet = empty_packet(ctx, port.name)
            out[port.name] = packet
        if none:  # 只针对有连线的种类提示：其他种类没有下游在等待
            ctx.say("N-TAKE-NONE", node=cls.label, kinds="、".join(none))
        return out


class CameraSpaceConvert(NodeDef):
    """将场景从一台相机的空间转换到另一台相机的空间。

    接入的各第三方项目使用各自的空间：WHAM 的人物位于其自身的世界并配有其自身的相机，ViPE 的相机位于 ViPE 的
    世界，使用者的真实相机从 USD 导入。各项目只保证自身数据在自身相机下与画面一致。要让它们处于同一个世界，
    需用两台相机计算一个变换并施加到数据上：只要一个项目能输出逐帧的内参和外参，其空间即可转换到任何其他项目的空间。

    三个输入、一个输出：「场景」为待转换的数据，「来源相机」为其当前所在的相机空间，「目标相机」为转换目标的
    相机世界；输出仍为「场景」，类型与输入一致（输入人物则输出人物，输入点云则输出点云）。

    「来源相机」留空表示单位相机：SAM 3D Body、HaMeR、SMIRK、MediaPipe 面部等官方实现只输出人物在相机前的位置，
    未解算相机运动，相当于来源相机每一帧都位于原点、朝向默认且静止。因此留空时每帧直接乘以目标相机的
    cam_to_world（data/scene.py place_through），且只能逐帧贴合，因为静止相机没有可拟合的轨迹。

    内参只做核对，不参与变换：两台相机焦距不同时，不存在能使两边与画面都对齐的刚性变换（透视程度不同），
    节点会拦下并提示使用目标相机的内参重新解算。模板已通过「拆分相机」将目标相机的焦距和片门提供给解算器，
    因此正常链路中两边内参一致。

    唯一的参数「贴合方式」：
    - 逐帧贴合（默认）：每帧 T(f) = 目标(f) × 来源(f)⁻¹。内参一致时为精确解而非近似；静止镜头同样适用。
      代价是前后距离会随解算器逐帧的深度噪声抖动。
    - 整段平滑：由两条轨迹拟合一个带尺度的常量变换（lab2shot_shared/poses.py scaled_align），施加于整段。
      动作连贯、脚部不滑动，但每帧不严格贴合；剩余误差会被报告，误差大说明两边不像同一段素材。尺度作用于整个场景，
      包括人物身高：目标世界的尺度由目标相机决定，人物不随之缩放则投影无法对齐；需要真实身高时应调整目标世界的比例。

    刚性拟合必须带尺度：ViPE 轨迹的尺度来自深度模型，WHAM 世界的尺度来自人物身高，两条轨迹长度不同，
    不带尺度的常量刚性拟合会把差异全部平均到人物上，导致人物无法与画面贴合。
    本节点不提高精度：只改变位置，场景自身的数据不做任何修改。
    """

    id = "core.camera_space"
    same_on_cards = True  # 各卡上公开的这块参数一样（NodeDef.same_on_cards）
    version = 1
    on_node = ("fit",)
    # 与「法线空间转换」属于同一类：更换空间，数据本身不变
    category = "geometry_tools"
    # 只接受场景中的以下内容：人物、网格、骨架、曲线、点云。不接受相机：用一台相机去放置另一台相机在逻辑上不成立，
    # 而宽类型 `scene` 会包含相机，因此此处逐项列出。需要移动整台相机时使用「3D 变换」。
    inputs = (Port("scene", "scene.character|scene.model|scene.skeleton|scene.curves|scene.points", "场景",
                   help="要搬的东西：人、网格、骨架、曲线、点云"),
              # 必须设置 optional=True：提交前的检查（engine/graph.py）要求所有非可选端口已连接，不考虑 applies
              Port("source_camera", "scene.camera", "来源相机", optional=True, expects=(SameShot(of="scene", size=False),),
                   help="场景现在待在哪台相机的空间里：有自己世界的解算器（WHAM、GVHMR）把它的「参照相机」接过来。"
                        "留空表示场景就在相机前面、没有世界位置（SAM 3D Body、HaMeR、SMIRK、MediaPipe 面部），"
                        "每帧直接按目标相机摆进世界"),
              Port("target_camera", "scene.camera", "目标相机", expects=(SameShot(of="scene", size=False),),
                   help="搬到哪台相机的世界里：解出来的那台（ViPE、COLMAP…）或者「导入 USD」进来的真相机。"
                        "两台相机的焦距、片门必须一样（模板用「拆分相机」把目标相机的喂给解算器）"))
    outputs = (Port("scene", "scene.character|scene.model|scene.skeleton|scene.curves|scene.points", "场景",
                    type_from="input:scene", help="同一份场景，在目标相机的世界里"),)

    class Params(NodeParams):
        fit: Literal["per_frame", "whole"] = P(
            "per_frame", label="贴合方式", group="转换",
            option_labels={"per_frame": "逐帧贴合", "whole": "整段平滑"},
            # 来源相机留空即单位相机，没有可拟合的轨迹，只能逐帧贴合
            applies=Wired("source_camera"))

    LENS_TOLERANCE = 0.02  # 两台相机的视角相差超过此比例即视为不是同一镜头
    MOVED_CM = 10.0  # 相机中心的散布（均方根）至少达到此值，整段平滑模式才从轨迹读取比例

    @classmethod
    def cook(cls, ctx):
        from ...data.camera import CameraSamples
        from ...data.scene import place_through, transform

        scene, target = ctx.input("scene"), ctx.input("target_camera")
        frames = [int(f) for f in (scene.meta.get("frames") or target.meta.get("frames") or [])]
        if not frames:  # 静止场景：按相机的第一帧放置
            frames = [int((target.meta.get("frames") or [0])[0])]
        to = CameraSamples.from_packet(target, frames)
        to_mats = cls._poses(to, len(frames))
        source = ctx.input("source_camera")  # 可选端口未连接时为 None
        if source is None:  # 单位相机：场景位于相机前方，每帧按目标相机放入世界
            ctx.say("I-CAMSPACE-PLACED", frames=len(frames))
            return {"scene": place_through(scene, ctx.outputs["scene"], to_mats, frames)}
        fr = CameraSamples.from_packet(source, frames)
        cls._same_lens(fr, to)
        fr_mats = cls._poses(fr, len(frames))
        if ctx.params["fit"] == "per_frame":
            mats = to_mats @ np.linalg.inv(fr_mats)
            ctx.say("I-CAMSPACE-PERFRAME", frames=len(frames))
            return {"scene": place_through(scene, ctx.outputs["scene"], mats, frames)}
        # 整段平滑：两条轨迹 → 一个带尺度的常量变换（两台相机单位均为厘米，误差也以厘米计）
        from lab2shot_shared.poses import rotation_deg, scaled_align

        # 两台相机都确实移动过才从轨迹读取比例：相机中心散布（均方根）不足 MOVED_CM 时比例按 1 处理并给出提示。
        # 在几厘米长的轨迹上读出的比例只是噪声（近似静止的镜头上可拟合出 0.034，人物被缩小到 3%），只有旋转和平移可信。
        spreads = [cls._spread(m) for m in (fr_mats, to_mats)]
        moved = min(spreads) >= cls.MOVED_CM
        fix, scale, spread, turn_off, left_cm = scaled_align(fr_mats, to_mats, fit_scale=moved)
        if not moved:
            ctx.say("W-CAMSPACE-STILL", spread=min(spreads), least=cls.MOVED_CM)
        ctx.say("I-CAMSPACE-FITTED", scale=scale, turn=float(rotation_deg(fix[:3, :3] / scale)),
                move=float(np.linalg.norm(fix[:3, 3])), turn_off=turn_off, error=left_cm)
        if left_cm > 25.0:  # 两台相机偏差达到此程度，说明不像是同一镜头的解算结果
            ctx.say("W-CAMSPACE-OFF", error=left_cm, turn_off=turn_off)
        return {"scene": transform(scene, ctx.outputs["scene"], fix)}

    @staticmethod
    def _spread(mats: np.ndarray) -> float:
        """相机中心到其均值的均方根距离（cm）：衡量该段相机的移动幅度。"""
        c = np.asarray(mats, np.float64)[:, :3, 3]
        return float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean()))

    @staticmethod
    def _poses(cam, count: int) -> np.ndarray:
        """无论数据包提供的是什么，均返回 [F,4,4] 的 cam_to_world（单个静止位姿，或没有时位于原点）。"""
        m = cam.cam_to_world
        if m is None:
            return np.repeat(np.eye(4)[None], count, 0)
        m = np.asarray(m, np.float64)
        return np.repeat(m[None], count, 0) if m.ndim == 2 else m

    @classmethod
    def _same_lens(cls, a, b) -> None:
        """两台相机的视角（焦距 / 水平片门）逐帧一致，否则拦下：焦距不同时不存在使两边都与画面对齐的刚性变换。"""
        from ...data.units import PERCENT

        def fov(c):
            ap = np.asarray(c.h_aperture_mm, np.float64)
            return np.asarray(c.focal_mm, np.float64) / np.where(ap > 0, ap, np.nan)
        fa, fb = np.broadcast_arrays(fov(a), fov(b))
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = float(np.nanmax(np.abs(fa / fb - 1.0))) if fa.size else 0.0
        if np.isfinite(diff) and diff > cls.LENS_TOLERANCE:
            raise Invalid(Msg("B-CAMSPACE-LENS", diff=diff * PERCENT))


class Transform3D(NodeDef):
    id = "core.transform"
    version = 2
    on_node = ("translate", "rotate", "scale")
    category = "scene_build"
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)
    # 变换手柄，以及计算时的放置方式。手柄作用于输入的场景：还没算（或已过期）时视图按当前参数摆放上游的场景
    handles = (Places(translate="translate", rotate="rotate", scale="scale", source="scene"),)

    class Params(NodeParams):
        translate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="移动", widget="vec3", group="变换")
        rotate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="旋转", widget="vec3", group="变换")
        scale: float = P(1.0, label="缩放", gt=0, group="变换")

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import transform

        return {"scene": transform(ctx.input("scene"), ctx.outputs["scene"], cls.places.matrix(ctx.params))}


class AutoGround(NodeDef):
    id = "core.auto_ground"
    version = 2
    on_node = ("source", "percentile")
    category = "scene_build"
    # 按输入场景自身的尺寸将其放平，每次处理一组帧；相机随之旋转后透传。
    # 只有一个输入端口：相机本身已在场景中（合成场景会把相机和点云打包在一起），再单独开一个「相机」端口会使
    # 同一台相机有两条输入路径。场景中有多台相机时拒绝并说明如何选择（data/scene.py the_camera）。
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)

    class Params(NodeParams):
        gravity: tuple[float, float, float] | None = P(
            None, label="重力方向", widget="vec3", group="放平", per_frame=True,
        )
        gravity_error: float | None = P(
            None, label="重力误差", unit="°", ge=0, group="放平", per_frame=True, applies=Param("gravity").set(),
        )
        source: Literal["people", "points", "none"] = P(
            "people", label="地面依据", group="地面",
            option_labels={"people": "人物脚底", "points": "点云", "none": "不落地"},
        )
        percentile: float = P(
            10.0, label="落地帧比例", ge=1.0, le=50.0, group="地面", widget="slider",
            applies=Param("source").one_of("people"),
        )

    @classmethod
    def _gravity(cls, ctx, scene) -> tuple[list[int], np.ndarray, np.ndarray] | None:
        """(帧, 每帧相机坐标系中的上方向 [F,3], 每帧允许的偏差 [F] 度)，来自「重力方向」/「重力误差」：
        填写的值（相机各帧相同）或连线提供的值（每帧一个）；未提供「重力方向」时为 None。"""
        if ctx.params["gravity"] is None:
            return None
        wired = ctx.values.get("gravity")
        if wired is not None and wired.per_frame:
            frames = list(wired.frames)
            up = np.asarray(wired.values, np.float64)
        else:
            frames = [int(f) for f in scene.meta["frames"]] or [0]
            up = np.tile(np.asarray(ctx.params["gravity"], np.float64), (len(frames), 1))
        error = ctx.values.get("gravity_error")
        if error is not None:
            sigma = error.at(frames)
        else:
            sigma = np.full(len(frames), float(ctx.params["gravity_error"] or 1.0))
        return frames, up, sigma

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import transform
        from ..kit.ground import ground_height, level_rotation, points_ground

        src = ctx.input("scene")
        gravity = cls._gravity(ctx, src)
        m = np.eye(4)
        if gravity is not None:
            m[:3, :3], found = level_rotation(*gravity, src)
            dropped = Msg("I-GROUND-DROPPED", count=found["dropped"]) if found["dropped"] else ""
            ctx.say("I-GROUND-LEVELED", tilt=found["tilt_deg"], frames=found["frames"], spread=found["spread_deg"], dropped=dropped)
            if found["spread_deg"] > SPREAD_WARN_DEG:
                ctx.say("W-GROUND-SPREAD", spread=found["spread_deg"], port="gravity")
        source = ctx.params["source"]
        if source == "none" and gravity is None:
            raise Invalid(Msg("E-GROUND-NOTHING"))
        if source == "none":
            floor = 0.0
        elif source == "points":
            floor, share = points_ground(src, m[:3, :3])
            ctx.say("I-GROUND-POINTS", floor=floor, share=share, lift=-floor)
            if share < GROUND_SHARE_WARN:
                ctx.say("W-GROUND-FEWPOINTS", share=share)
        else:
            floor = ground_height(src, ctx.params["percentile"], m[:3, :3])
            ctx.say("I-GROUND-HEIGHT", floor=floor, lift=-floor)
        m[1, 3] -= floor
        return {"scene": transform(src, ctx.outputs["scene"], m)}


SPREAD_WARN_DEG = 5.0  # 各帧的上方向在世界中的分歧超过此值：相机与重力方向不一致
GROUND_SHARE_WARN = 0.1


class CurvesResample(NodeDef):
    id = "core.curves_resample"
    category = "scene_convert"
    inputs = (Port("curves", "scene.curves", "三维曲线"),)
    outputs = (Port("curves", "scene.curves", "三维曲线"),)
    on_node = ("per_curve",)

    class Params(NodeParams):
        # 只提供若干固定档位，不允许任意填写：点数 x 曲线条数决定内存和文件大小，
        # 误填 100000 会使一个有 10 万条的发型变成 100 亿个点。
        per_curve: Literal[8, 16, 24, 32, 48, 64, 100] = P(
            24, label="每条点数", group="曲线",
            option_labels={"8": "8", "16": "16", "24": "24", "32": "32", "48": "48", "64": "64", "100": "100"})

    @classmethod
    def cook(cls, ctx):
        from ...data.curves import resample

        packet, strands, points = resample(ctx.input("curves"), int(ctx.params["per_curve"]), ctx.outputs["curves"])
        ctx.say("I-CURVES-RESAMPLED", strands=strands, points=points, each=int(ctx.params["per_curve"]))
        return {"curves": packet}


class TakeSubset(NodeDef):
    id = "core.take_subset"
    # 「分区」的选项来自接入的网格（choices_from），不是可在节点上直接修改的简单参数：在面板中选择，节点上不显示
    category = "scene_build"
    # 名称与「按种类取出」采用同一套说法（一个按三维数据的种类区分，一个按网格自身的分区区分）。Houdini 的 Blast
    # 最为接近，但其含义是「删除其他部分」而非「取出这一部分」，且对非建模使用者不是通用术语，因此保留中文名。
    inputs = (Port("model", "scene.model", "模型"),)
    outputs = (Port("model", "scene.model", "模型", type_from="input:model"),)

    class Params(NodeParams):
        subset: str = P("", label="分区", widget="choice", group="网格", choices_from=("model",),
                        placeholder="先接上模型")

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """「分区」列出的是接入网格自带的分区名（data/subsets.py 读取 GeomSubset），而非固定的列表。"""
        from ...data.subsets import scene_subsets

        src = inputs.get("model")
        have = scene_subsets(src) if src is not None else {}
        return {"subset": {"options": list(have),
                           "labels": {name: f"{name}（{faces} 面）" for name, faces in have.items()},
                           "empty": "选一个分区" if have else "这个网格没有分区"}}

    @classmethod
    def cook(cls, ctx):
        from ...data.subsets import only_subset, scene_subsets

        src, name = ctx.input("model"), ctx.params["subset"]
        have = scene_subsets(src)
        if not name:
            raise Invalid(Msg("B-SUBSET-NOTCHOSEN", names=_said(have) or "（这个网格没有分区）"))
        got = only_subset(src, name, ctx.outputs["model"])
        if got is None:
            raise Invalid(Msg("E-SUBSET-MISSING", name=name, names=_said(have) or "（这个网格没有分区）"))
        packet, meshes, faces = got
        ctx.say("I-SUBSET-TOOK", name=name, meshes=meshes, faces=faces)
        return {"model": packet}


class AttribDelete(NodeDef):
    id = "core.attrib_delete"
    # 属于「组装与变换」类：与「按种类取出」「按分区取出」一样不改变数据本身，
    # 只移除场景中的一部分内容（分别按种类、按面、按点上的属性值）。
    category = "scene_build"
    # Houdini 中对应的节点是 AttribDelete；「属性」在中文 CG 领域是通用词，不属于只以英文流通的术语
    # （如 STMap、LensDistortion、CornerPin），因此保留中文名。
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)

    class Params(NodeParams):
        attributes: str = P("", label="属性名", group="属性", placeholder="先接上数据")

    @classmethod
    def cook(cls, ctx):
        from ...data.attributes import drop, matching, named
        from ...data.packet import copy_packet

        src, patterns = ctx.input("scene"), str(ctx.params["attributes"] or "").strip()
        have = named(src)
        if not have:
            # 该数据的点上本来没有属性（相机、只有点的点云等）：无可删除不属于错误。
            # 原样向下传递并在节点上提示；上游改为带属性的数据后即开始生效。
            ctx.say("I-ATTR-NOTHING")
            # 输出必须是本节点自己的数据包（下游按其指纹查找文件），因此复制一份，
            # 而不是将上游数据包原样传出（data/packet.py copy_packet）。
            out = ctx.outputs["scene"]
            out.mkdir(parents=True, exist_ok=True)
            return {"scene": copy_packet(src, out)}
        # 未匹配也不属于错误：数据没有丢失，只是尚未指定要删除的内容。
        # 原样向下传递，并在节点上保留一条警告（不会自动消失，日志中也有记录），整条链不会因一个尚未填写的节点而中断。
        hit = matching(have, patterns) if patterns else []
        if not hit:
            ctx.say("W-ATTR-NONAMES" if not patterns else "W-ATTR-NOMATCH", names=_attrs(have))
            out = ctx.outputs["scene"]
            out.mkdir(parents=True, exist_ok=True)
            return {"scene": copy_packet(src, out)}
        packet, gone = drop(src, hit, ctx.outputs["scene"])
        ctx.say("I-ATTR-DROPPED", count=gone, names="、".join(hit))
        return {"scene": packet}


def _attrs(have: list[dict]) -> str:
    return "、".join(f"{a['name']}（{a['type']} · {a['per']}）" for a in have)


def _said(have: dict) -> str:
    return "、".join(f"{name}（{faces} 面）" for name, faces in have.items())


class BakeModel(NodeDef):
    id = "core.bake_model"
    category = "scene_convert"
    inputs = (Port("character", "scene.character", "蒙皮角色"),)
    outputs = (Port("model", "scene.model", "模型", kinds=(DEFORMING,)),)
    converts = ("scene.character", "scene.model")  # 只接受「模型」的位置接入了「蒙皮角色」时，一键插入本节点进行转换

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import bake

        return {"model": bake(ctx.input("character"), ctx.outputs["model"])}


class ExtractSkeleton(NodeDef):
    id = "core.extract_skeleton"
    category = "scene_convert"
    inputs = (Port("character", "scene.character", "蒙皮角色"),)
    outputs = (Port("skeleton", "scene.skeleton", "骨架动画"),)
    converts = ("scene.character", "scene.skeleton")

    @classmethod
    def cook(cls, ctx):
        from ...data.scene import skeleton_only

        return {"skeleton": skeleton_only(ctx.input("character"), ctx.outputs["skeleton"])}


# 包都按 recent.packet_key（账号、指纹、代次）做键，和状态回复的手柄数据同一条：重算过的包重新读，别的账号的不串
_RIGS: Recent = Recent(8)  # (包, 骨架路径, least_frames) -> Rig：弹窗和手柄反复读同一副骨架
_BASES: Recent = Recent(8)  # (两个包, 两条骨架路径, 相关参数) -> Retarget._bases 的结果
_PICKED: Recent = Recent(8)  # (两个包, 两条骨架路径, 除「初始姿势」外的相关参数) -> 动作自动挑的帧（Rests.picked）


def _stored(packet) -> bool:
    """包在缓存里（按指纹找得到的就是它）：拆出来放在临时文件夹里的一条（server/app.py representative）不记。"""
    from ...data.packet import packet_dir
    from ...data.store import NoAccount

    try:
        return packet_dir(packet.fingerprint) == packet.dir
    except NoAccount:  # read outside an account (a script, a test): not remembered
        return False


class Retarget(NodeDef):
    """动作重定向：一副骨架的动作换到另一副骨架上，交出目标层级上的骨架动画。

    「动作」接骨架动画或蒙皮角色（导入的 BVH、解算出来的人、动作生成的结果），「目标」接要套上动作的骨架或蒙皮角色，
    只用它的骨架：关节、层级、绑定姿势（对齐方向），以及第一帧每个关节的平移和缩放（骨长、比例）；一帧的角色
    （「标准人」、刚自动绑定的角色）就够，DCC 里也是这样。交出的只是骨架动画：要看蒙皮效果接「线性蒙皮变形」，
    要交付骨架接「FBX / USD 输出设置」。

    两边的部位对应在「对应关系」里（HumanIK 式的人形部位槽，NodeDef.choices 给编辑器两副骨架），没写的部位按关节名
    和层级推测；链状部位（脊柱、颈、手指）两边节数不同时，源链的弯曲按骨长分摊到目标链的每一节。数学在
    `lab2shot_shared.motion.BodyRetarget` / `hips_path`，对应关系与写回在 `nodes/kit/retarget.py`。

    两边有一边没有髋、都有手时（HaMeR 的 MANO 手、只有手的绑定）只配手：必需部位是手腕和五指；目标只有手时手腕 1 : 1
    跟着动作，目标的手长在身体上时手腕和手臂保持目标自己的姿势、只传手指；髋高的四个参数不起作用
    （kit/retarget.py hands_of / retarget_hands）。
    """

    id = "core.retarget"
    category = "scene_convert"
    same_on_cards = True  # 13 张重定向卡的参数块一样（NodeDef.same_on_cards）
    version = 23  # 进指纹：对齐或写回的算法变了就加一，旧结果重算
    inputs = (Port("motion", "scene.skeleton|scene.character", "动作",
                   help="动作从哪来：一段骨架动画（导入的 BVH、动作生成、解算出来的人），或者另一个角色"),
              Port("target", "scene.character|scene.skeleton", "目标",
                   help="动作要套到谁身上：你自己的角色或骨架，只用它的骨架和绑定姿势，一帧就够"))
    outputs = (Port("skeleton", "scene.skeleton", "骨架动画"),)
    # 两边的基准姿势在 3D 视图里逐关节摆（「初始姿势」，kit/retarget.py corrected）：舞台要的骨架由 handle_data 给
    handles = (Poses(pose="motion_pose", source="motion", skeleton="motion_skeleton"),
               Poses(pose="target_pose", source="target", skeleton="target_skeleton"))

    class Params(NodeParams):
        motion_skeleton: str | None = P(None, label="动作骨骼", widget="choice", group="骨架", choices_from=("motion",),
                                        placeholder="第一个")
        target_skeleton: str | None = P(None, label="目标骨骼", widget="choice", group="骨架", choices_from=("target",),
                                        placeholder="第一个")
        # scale_by 也在 choices_from 里：编辑器标的「必需」跟着「髋高依据」变（kit/retarget.py required）
        mapping: list[PartMap] | None = rig_map_param(("motion", "target", "motion_skeleton", "target_skeleton", "scale_by",
                                                       "motion_rest", "motion_rest_frame", "target_rest", "target_rest_frame",
                                                       "rest_fix", "motion_pose", "target_pose"), group="骨架")
        # 两边对齐用的基准姿势（kit/retarget.py rests）：「对应关系」弹窗的 3D 画的就是它们，一份
        motion_rest: Literal["bind", "first", "frame"] = P(
            "bind", label="动作基准姿势", group="骨架",
            option_labels={"bind": "绑定姿势", "first": "第一帧", "frame": "指定帧"},
            help="动作那副骨架拿哪个姿势当基准和目标对齐：绑定姿势（文件里网格绑定时的样子；不是站姿时——BVH 的零姿势——"
                 "自动挑动作里最像目标的一帧）、第一帧，或指定的一帧")
        motion_rest_frame: int | None = P(None, label="动作基准帧", group="骨架", placeholder="帧号",
                                          applies=Param("motion_rest").one_of("frame"))
        target_rest: Literal["bind", "first", "frame"] = P(
            "first", label="目标基准姿势", group="骨架",
            option_labels={"bind": "绑定姿势", "first": "第一帧", "frame": "指定帧"},
            help="目标角色拿哪个姿势当基准：第一帧（DCC 打开文件看到的样子）、绑定姿势（网格绑定时的样子：AccuRIG 导出的 FBX "
                 "绑定姿势是 A、第一帧是 T），或指定的一帧")
        target_rest_frame: int | None = P(None, label="目标基准帧", group="骨架", placeholder="帧号",
                                          applies=Param("target_rest").one_of("frame"))
        rest_fix: Literal["none", "tpose"] = P(
            "none", label="基准姿势摆正", group="骨架", option_labels={"none": "不摆正", "tpose": "自动摆成 T 姿"},
            help="两边的基准姿势差很多（A 对 T、折叠的零姿势）时，先按配上的部位把两边的四肢都摆成标准 T 姿（上臂、前臂水平，"
                 "腿竖直，脚朝前）再对齐")
        # 基准姿势（选的姿势、摆正之后）上的逐关节修正：对齐按修正后的算（kit/retarget.py rests）
        motion_pose: list[JointPose] = pose_param("动作初始姿势", "动作")
        target_pose: list[JointPose] = pose_param("目标初始姿势", "目标")
        size_by: Literal["none", "height", "manual"] = P(
            "none", label="角色缩放", group="位移",
            option_labels={"none": "不缩放", "height": "按身高自动", "manual": "手填倍率"},
            help="把整个角色（骨架和蒙皮）放大或缩小：按身高自动 = 动作里的人的身高 ÷ 角色的身高（腿长加髋到头，逐帧量取中位数；"
                 "一边没有头时按腿长）；手填倍率 = 用下面填的。缩放后再算髋高，角色和画面里的人一样大时髋高比例接近 1")
        size: float = P(1.0, label="缩放倍率", group="位移", ge=0.01, le=100.0, unit="倍",
                        applies=Param("size_by").one_of("manual"), help="整个角色放大这么多倍，1 = 原大")

        # 水平位移 1 : 1 照抄动作（角色要踩在画面里那个人的位置上：整条轨迹乘比例会越走越偏）；
        # 只有髋的高度按比例 = 依据量出来的比（目标 ÷ 动作，动作逐帧量、取中位数）× 修正系数，腿长的角色髋抬高、脚才
        # 落地。依据按镜头选：走路、站立看腿，全身入画看身高，够东西、手接触看臂长；「不缩放」时比例就是修正系数本身。
        scale_by: Literal["legs", "height", "arms", "none"] = P(
            "legs", label="髋高依据", group="位移",
            option_labels={"legs": "按腿长", "height": "按身高", "arms": "按臂长", "none": "不缩放（1 : 1）"},
            help="髋的高度按哪个量算比例：目标的这个量 ÷ 动作的（逐帧量、取中位数）。走路、站立按腿长；全身入画按身高；"
                 "够东西、手要接触按臂长；不缩放就只用修正系数。水平位移总是 1 : 1 跟着动作走，角色踩在人的位置上")
        scale: float | None = P(1.0, label="髋高倍率", group="位移", ge=0.01, le=100.0, unit="倍", placeholder="1",
                                help="再把髋抬高 / 压低这么多倍（只动髋的高度，不缩放人）；「不缩放」时它就是髋高比例本身")
        # 固定抬高：髋高 = 动作的髋高 + (比例 − 1) × 髋离地高度的中位数，楼梯、梯子、跳跃都对，深蹲差几厘米；
        # 按比例：离地高度逐帧乘比例，平地（含蹲）都对，地面一升高（楼梯）就错。
        height_mode: Literal["lift", "proportional"] = P(
            "lift", label="髋高跟法", group="位移",
            option_labels={"lift": "固定抬高（楼梯、梯子、跳跃也对）", "proportional": "按比例（只适合平地）"},
            help="髋的高度怎么跟着动作：固定抬高 = 动作的髋高加一个固定量，地面升降（楼梯、梯子、跳）都对，深蹲时差几厘米；"
                 "按比例 = 离地高度逐帧乘比例，平地上蹲下站起都对，地面一升高就错")
        height_offset: float = P(0.0, label="高度偏移", group="位移", ge=-100.0, le=100.0, unit="cm",
                                 help="整段再抬高（正）或压低（负）这么多厘米，脚浮空、穿地时直接按看到的量填。两边脚踝离脚底的"
                                      "高度差已经自动算进去了（两边都有网格时）")
        # 角色缩放：整个目标（骨架 + 蒙皮）均匀放大到和动作里的人一样大，根关节上加缩放（kit/retarget.py sized）；
        # 髋高比例按放大后的目标算，所以「按身高自动」时髋高比例接近 1，脚底对齐照常

    @classmethod
    def info(cls, params, inputs):
        """帧来自「动作」：目标只提供骨架（常常只有一帧的静止姿势），它的帧不属于结果。"""
        return Info.merge(inputs.get("motion", []))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        from ...data.joints import part_joints
        from ..kit.retarget import has_body, hands_of, merged_parts, required
        from ..kit.rig_map import pose_handle, rig_side, skeleton_choice

        out = {}
        for port, name in (("motion", "motion_skeleton"), ("target", "target_skeleton")):
            if inputs.get(port) is not None:
                out.update(skeleton_choice(inputs[port], name))
        if inputs.get("motion") is not None and inputs.get("target") is not None:
            src = rig_side(inputs["motion"], params.get("motion_skeleton"), "动作", pose_handle(cls, "motion"))
            dst = rig_side(inputs["target"], params.get("target_skeleton"), "目标", pose_handle(cls, "target"))
            # 只有手的两边（hands_of）标手腕和五指、整个身体按「髋高依据」：部位按计算时同一份（推测 + 手配的行）
            if src and dst:  # the joints' own names (rig_side "joints"), as resolve_mapping reads them
                rows = merged_parts(params.get("mapping"), src["joints"], src["parents"], dst["joints"], dst["parents"])[0]
                have = [part_joints(rows, side["joints"], side["parents"], col) for side, col in ((src, "src"), (dst, "dst"))]
                hands = hands_of(*have, (has_body(src["joints"], src["parents"], have[0]),
                                         has_body(dst["joints"], dst["parents"], have[1])))
            else:
                hands = ()
            out["mapping"] = rig_map_choice(src, dst, required(params.get("scale_by") or "legs", hands))
            if "rig" in out["mapping"] and not hands:
                cls._rest_choice(params, inputs, out["mapping"]["rig"])
        return out

    @classmethod
    def _rest_choice(cls, params: dict, inputs: dict, rig: dict) -> None:
        """弹窗两侧标出对齐用的是哪个基准姿势，并带出两副基准姿势（修正后，kit/retarget.py rests，和计算同一份）里
        对应骨头的方向差。"""
        from ..kit.retarget import rest_diffs

        got = cls._bases(params, inputs)
        if got is None or got[2] is None:
            return  # 配对还有问题：编辑器照常标红
        m, r = got[2], got[3]
        rig["src"]["pose"], rig["dst"]["pose"] = cls._pose_said(r)
        rig["diffs"] = rest_diffs(r, m)

    @classmethod
    def _bases(cls, params: dict, inputs: dict):
        """两副骨架、对应关系和对齐用的基准姿势（kit/retarget.py rests / 只配手时 hand_bases），和计算同一份：
        (动作 rig, 目标 rig, Mapping, Rests)；对应关系还不成（缺部位、基准帧不对）时后两项是 None，读不到骨架时 None。
        弹窗（choices）和手柄（handle_data）每改一次参数都要它：按输入包和相关参数记住（_BASES），骨架按包和骨架路径
        另记（_RIGS），拖一次手柄只重算基准姿势，不重读两副骨架的全部帧。"""
        from ...data.animation import read_rig
        from ..kit.retarget import hand_bases, resolve_mapping, rests

        def rig(port: str, path, least: int):
            packet = inputs.get(port)
            if packet is None:
                return None
            def read():
                try:
                    return read_rig(packet, path, least_frames=least)
                except Invalid:
                    return None

            key = _stored(packet) and (packet_key(packet), path, least)
            return _RIGS.get_or(key, read) if key else read()

        source, target = rig("motion", params.get("motion_skeleton"), 2), rig("target", params.get("target_skeleton"), 1)
        if source is None or target is None:
            return None
        args = cls._rest_args(params)
        rigs = _stored(inputs["motion"]) and _stored(inputs["target"]) and (
            packet_key(inputs["motion"]), packet_key(inputs["target"]), source.path, target.path)
        # the frame picked for a folded source depends on neither pose correction: kept apart, so a pose edit does
        # not compare every frame of the take again
        chosen = rigs and (rigs, json.dumps([params.get("mapping"), params.get("scale_by"),
                                             {k: v for k, v in args.items() if not k.endswith("_pose")}],
                                            sort_keys=True, default=str))
        key = chosen and (chosen, json.dumps([args["motion_pose"], args["target_pose"]], sort_keys=True, default=str))
        if key and (kept := _BASES.get(key)) is not None:
            return kept
        try:
            m = resolve_mapping(params.get("mapping"), source, target, params.get("scale_by") or "legs")
            if m.hands:
                base = hand_bases(source, target, args)
            else:
                base = rests(source, target, m, **args, picked=_PICKED.get(chosen) if chosen else None)
                if chosen:
                    _PICKED.put(chosen, base.picked)
            got = source, target, m, base
        except Invalid:
            got = source, target, None, None
        return _BASES.put(key, got) if key else got

    @staticmethod
    def _pose_said(r) -> tuple[str, str]:
        return r.source_said, r.target_said + ("（摆成 T 姿）" if r.fixed else "")

    @staticmethod
    def _rest_args(params: dict) -> dict:
        return {"motion_rest": params.get("motion_rest") or "bind", "motion_frame": params.get("motion_rest_frame"),
                "target_rest": params.get("target_rest") or "first", "target_frame": params.get("target_rest_frame"),
                "fix": params.get("rest_fix") or "none", "motion_pose": params.get("motion_pose") or [],
                "target_pose": params.get("target_pose") or []}

    @classmethod
    def handle_data(cls, params: dict, inputs: dict) -> dict[int, dict]:
        """两个「骨架姿势」手柄（动作 0、目标 1）要画的（kit/rig_map.py skeleton_handle）：关节、层级、对齐用的基准
        姿势在「初始姿势」修正之前的局部（正交的轴、cm，根关节含 Skeleton prim 的摆放），左右镜像的关节对，参数里
        骨架没有的关节名，和蒙皮预览要的包与骨架路径。对应关系还不成时按各自选的姿势画（不挑帧、不摆 T），部位按
        side_parts。"""
        from ...data.joints import joint_keys
        from ..kit.retarget import corrected, rest_pose

        got = cls._bases(params, inputs)
        if got is None:
            return {}
        source, target, m, r = got
        args = cls._rest_args(params)
        out = {}
        for k, (rig, port, key, which, frame, side) in enumerate((
                (source, "motion", "motion_pose", "motion_rest", "motion_frame", "src"),
                (target, "target", "target_pose", "target_rest", "target_frame", "dst"))):
            keys = joint_keys(rig.names, rig.parents)  # how parameters name its joints
            if r is not None:
                before = r.source_before if k == 0 else r.target_before
                pose, unknown = cls._pose_said(r)[k], r.unknown[k]
                parts = getattr(m, side)
            else:
                try:
                    before = rest_pose(rig, args[which], args[frame], "动作" if k == 0 else "目标")
                except Invalid:
                    continue
                unknown = corrected(before, rig.parents, keys, args[key])[1]
                pose = ""
                parts = side_parts(rig.names, rig.parents, params.get("mapping"), side)
            body = r.bodies[1 - k] if r is not None and r.bodies else None  # bodies: (target, motion)
            out[k] = skeleton_handle(inputs[port].fingerprint, rig.path, keys, rig.parents, before, parts, pose,
                                     unknown, body)
        return out

    @classmethod
    def cook(cls, ctx):
        from ...data.animation import read_rig, sole_height
        from ..kit.retarget import (GROUND_SAID, LEGS_VARY, SCALE_BY, body_up, character_size, ground_of, rests,
                                    notes, resolve_mapping, retarget)

        p = ctx.params
        motion, target = ctx.input("motion"), ctx.input("target")
        # 目标只需带骨架（least_frames=1）：只读它的关节、层级、绑定姿势和第一帧；动作全部来自「动作」口。
        # 来源仍要求一段动作（一帧没有可迁移的动作）。
        source, goal = read_rig(motion, p["motion_skeleton"]), read_rig(target, p["target_skeleton"], least_frames=1)
        ctx.stage("配对关节")
        mapping = resolve_mapping(p["mapping"], source, goal, p["scale_by"])
        for note in notes(mapping):  # 推测的部位要说清推测成了什么
            ctx.say(note.code, **note.params)
        ctx.stage("换动作")
        if mapping.hands:  # 只配手：没有髋，髋高的四个参数都不起作用
            r = retarget(source, goal, mapping, p["scale_by"], 1.0, rest=cls._rest_args(p))
            cls._unknown_joints(ctx, r)
            ctx.say("I-RETARGET-HANDS", joints=r.joints, total=len(goal.names),
                    hands="、".join({"l": "左手", "r": "右手"}[s] for s in mapping.hands),
                    wrist="目标的手长在身体上：手腕和手臂保持目标自己的姿势，只传手指相对手腕的弯曲" if r.wrist_kept
                    else "手腕 1 : 1 跟着动作")
            return cls._written(ctx, target, goal, source, r, {"hands": list(mapping.hands)})
        size = character_size(p["size_by"], p["size"], source, goal, mapping)
        # 两副基准姿势和两边的身体坐标系取一次（rests）：提示、脚底、对齐都用这一份
        base = rests(source, goal, mapping, **cls._rest_args(p), size=size.factor)
        for who, body in (("动作", base.bodies[1]), ("目标", base.bodies[0])):
            if not body.trunk:  # 躯干定身体的「上」和是不是站着：对齐用的这一个没有，就说
                ctx.say("W-BODY-NOTRUNK", rig=who)
        # 脚底对脚底：两边离地参考关节（ground_of：脚踝，没配脚时往下找）离脚底的高度差（两边都有网格才知道；BVH、
        # 只有关节的解算按一样处理）
        (ground_s, how_s), (ground_t, how_t) = ground_of(source, mapping.src), ground_of(goal, mapping.dst)
        soles = (sole_height(motion, source, ground_s, body_up(source, base.bodies[1])),
                 sole_height(target, goal, ground_t, body_up(goal, base.bodies[0])))
        if size.basis == "legs":
            ctx.say("W-RETARGET-SIZELEGS", rig="动作" if "head" not in mapping.src else "目标")
        if None not in soles:
            soles = (soles[0], soles[1] * size.factor)  # 目标放大了，踝离脚底也跟着放大
        sole_delta = soles[1] - soles[0] if None not in soles else 0.0
        r = retarget(source, goal, mapping, p["scale_by"], p["scale"] if p["scale"] is not None else 1.0,
                     p["height_mode"] == "lift", sole_delta + p["height_offset"], size.factor, cls._rest_args(p), base)
        cls._unknown_joints(ctx, r)
        low, median, high = r.measured_cm
        if r.by != "none" and high - low > LEGS_VARY * median:
            ctx.say("W-RETARGET-LEGVARIES", what=r.by_label, low=low, high=high, median=median)
        factor = f" × 修正 {r.factor:g}" if r.factor != 1.0 else ""
        how = f"按{r.by_label}：目标 {r.target_cm:.1f} cm ÷ 动作 {median:.1f} cm{factor}" if r.by != "none" else f"不缩放{factor}"
        for who, said in (("动作", how_s), ("目标", how_t)):
            if said != "ankles":
                how += f"；{who}离地高度按{GROUND_SAID[said]}量"
        if None not in soles:
            how += f"；脚底对齐 {sole_delta:+.1f} cm（离脚底：目标 {soles[1]:.1f}、动作 {soles[0]:.1f}）"
        if p["height_offset"]:
            how += f"；高度偏移 {p['height_offset']:+g} cm"
        if size.basis in ("height", "legs"):
            how += (f"；角色缩放 ×{size.factor:.3f}（按{SCALE_BY[size.basis]}：动作 {size.source_cm:.1f} cm ÷ "
                    f"目标 {size.target_cm:.1f} cm）")
        elif size.basis == "manual":
            how += f"；角色缩放 ×{size.factor:g}（手填）"
        how += f"；{r.rest_said}"
        ctx.say("I-RETARGET-DONE", joints=r.joints, total=len(goal.names), scale=round(r.scale, 3), how=how)
        return cls._written(ctx, target, goal, source, r, {"scale": r.scale, "scale_by": r.by, "factor": r.factor,
                                                            "size": r.size, "size_by": size.basis})

    @staticmethod
    def _unknown_joints(ctx, r) -> None:
        """「初始姿势」里写了骨架没有的关节（换了骨架、改了骨骼路径）：跳过，说一声。"""
        for who, names in zip(("动作", "目标"), r.unknown):
            if names:
                ctx.say("W-RETARGET-POSEJOINT", rig=who, joints="、".join(names[:8]) + (" 等" if len(names) > 8 else ""),
                        count=len(names))

    @classmethod
    def _written(cls, ctx, target, goal, source, r, said: dict) -> dict:
        """结果写回目标自己的层级，交出骨架动画（整个身体和只配手共用）。"""
        from dataclasses import replace

        from ...data.animation import placement_at, skeleton_animation
        from ..kit.retarget import sized, target_locals

        # 帧号来自「动作」；目标的 Skeleton prim 在这些帧上的摆放（可能被动画过的「3D 变换」挪着）
        placement = placement_at(target, goal.path, source.frames)
        local = target_locals(sized(goal, r.size), source.frames, placement, r)
        info = {"retarget": {"joints": r.joints, **said, "from": source.path}}
        # 帧率不属于数据的属性，写出文件时由输出设置节点指定（data/units.py DEFAULT_FPS）。
        rig = replace(goal, frames=list(source.frames))
        return {"skeleton": skeleton_animation(target, ctx.outputs["skeleton"], rig, local, info)}


class LinearSkin(NodeDef):
    """线性蒙皮变形：把一段骨架动画挂到蒙皮角色的骨架上，交出带这段动画的蒙皮角色。

    只校验、不猜测：两副骨架的关节名和父子关系必须逐一相同（顺序可以不同，按名字对齐），否则报 E-SKIN-HIERARCHY
    并列出不同之处；动画来自另一副骨架时先接「动作重定向」。蒙皮计算本身不在这里：视图用 GPU 蒙皮，交付走 UsdSkel，
    要网格缓存接「烘焙成模型」。骨长和角色的不一样时照挂（DCC 里连动画也是这样），只提示 W-SKIN-LENGTHS。
    """

    id = "core.linear_skin"
    category = "scene_convert"
    inputs = (Port("skeleton", "scene.skeleton", "骨架动画",
                   help="要挂上的动画：骨架层级要和角色的完全一样（通常是「动作重定向」交出的）"),
              Port("character", "scene.character", "蒙皮角色",
                   help="挂动画的角色：网格、蒙皮、blend shape 原样，原来的动画被换掉"))
    outputs = (Port("character", "scene.character", "蒙皮角色"),)

    class Params(NodeParams):
        skeleton_path: str | None = P(None, label="动画骨骼", widget="choice", group="骨架", choices_from=("skeleton",),
                                      placeholder="第一个")
        character_skeleton: str | None = P(None, label="角色骨骼", widget="choice", group="骨架",
                                           choices_from=("character",), placeholder="第一个")

    @classmethod
    def info(cls, params, inputs):
        """帧来自骨架动画：角色常常只有一帧的静止姿势。"""
        return Info.merge(inputs.get("skeleton", []))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        from ..kit.rig_map import skeleton_choice

        out = {}
        for port, name in (("skeleton", "skeleton_path"), ("character", "character_skeleton")):
            if inputs.get(port) is not None:
                out.update(skeleton_choice(inputs[port], name))
        return out

    @classmethod
    def cook(cls, ctx):
        from dataclasses import replace

        from ...data.animation import placement_at, read_rig, same_hierarchy, write_animation
        from ..kit.retarget import lengths_differ, skin_locals

        p = ctx.params
        anim = read_rig(ctx.input("skeleton"), p["skeleton_path"])
        character = read_rig(ctx.input("character"), p["character_skeleton"], least_frames=1)
        ctx.stage("核对层级")
        order = same_hierarchy(anim, character)
        far = lengths_differ(anim, character, order)
        if far:
            ctx.say("W-SKIN-LENGTHS", count=len(far), joints="、".join(far[:8]) + (" 等" if len(far) > 8 else ""))
        ctx.stage("挂动画")
        placement = placement_at(ctx.input("character"), character.path, anim.frames)
        local = skin_locals(anim, character, order, placement)
        ctx.say("I-SKIN-DONE", joints=len(order), frames=len(anim.frames))
        rig = replace(character, frames=list(anim.frames))
        info = {"linear_skin": {"from": anim.path}}
        return {"character": write_animation(ctx.input("character"), ctx.outputs["character"], rig, local, info)}


class ExpressionRetarget(NodeDef):
    """表情重定向（ARKit52）：按 ARKit 52 个 blendshape 的名字配对，不是逐点的面部重定向。一段表情（面部解算的「表情曲线」、或带 blendshape 动画的角色）换到另一个角色的 blendshape 上，交出
    这段表情动起来的角色。

    「表情」和「目标」按名字对应（「对应关系」，和「动作重定向」同一个编辑器，部位槽换成表情槽）：ARKit 的 52 个名字
    （MediaPipe 的表情曲线、Character Creator / MetaHuman 的 ARKit 形变）不管大小写、分隔符、L / R 写法都认得，
    FLAME 的眼皮（eyelid_left）认作 eyeBlinkLeft；FLAME 的 expression_00… 是 PCA 分量，只对得上同样带这些分量的
    FLAME 角色。认不出、认错的在编辑器里手动配。数值原样搬（0..1 的权重），目标没配上的形变保持 0；关节的动画
    不动。名字表在 data/expressions.py，写出在 data/animation.py write_blend_weights。
    """

    id = "core.expression_retarget_arkit52"
    same_on_cards = True  # 各卡上公开的这块参数一样（NodeDef.same_on_cards）
    category = "scene_convert"
    inputs = (Port("expressions", "curves|scene.character", "表情",
                   help="表情从哪来：面部解算的「表情曲线」（MediaPipe 的 ARKit 52、SMIRK / Pixel3DMM 的 FLAME），或带表情动画的角色"),
              Port("target", "scene.character", "目标",
                   help="表情要套到谁脸上：带 blendshape 的蒙皮角色（一帧就够，带着身体动画也行）"))
    outputs = (Port("character", "scene.character", "蒙皮角色"),)

    class Params(NodeParams):
        source_skeleton: str | None = P(None, label="表情骨骼", widget="choice", group="表情", choices_from=("expressions",),
                                        placeholder="第一个有表情的")
        target_skeleton: str | None = P(None, label="目标骨骼", widget="choice", group="表情", choices_from=("target",),
                                        placeholder="第一个")
        mapping: list[PartMap] | None = rig_map_param(("expressions", "target", "source_skeleton", "target_skeleton"),
                                                      group="表情")

    @classmethod
    def info(cls, params, inputs):
        """帧来自「表情」：目标常常只有一帧的静止姿势。"""
        return Info.merge(inputs.get("expressions", []))

    @classmethod
    def _curves(cls, packet, path: str | None):
        """「表情」口的曲线：(所在路径, 名字, 权重 [F,C], 帧)。"""
        from ...data.animation import blend_weights
        from ...data.payloads import read_curves

        frames = [int(f) for f in packet.meta.get("frames") or []]
        if packet.type == "curves":
            return "表情曲线", [str(n) for n in packet.meta.get("names") or []], read_curves(packet), frames
        where, names, weights = blend_weights(packet, path)
        return where, names, weights, frames

    @classmethod
    def _target(cls, packet, path: str | None) -> tuple[str, list[str]]:
        from ...data.animation import blend_shapes, skeletons

        found = [s["path"] for s in skeletons(packet)]
        where = path if path in found else (found[0] if found else "")
        return where, (blend_shapes(packet, where) if where else [])

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        from ..kit.rig_map import skeleton_choice

        out = {}
        src, dst = inputs.get("expressions"), inputs.get("target")
        if src is not None and src.type != "curves":
            out.update(skeleton_choice(src, "source_skeleton"))
        if dst is not None:
            out.update(skeleton_choice(dst, "target_skeleton"))
        if src is not None and dst is not None:
            where, curves, _, _ = cls._curves(src, params.get("source_skeleton"))
            path, shapes = cls._target(dst, params.get("target_skeleton"))
            out["mapping"] = expression_choice(curves, shapes, where, path)
        return out

    @classmethod
    def cook(cls, ctx):
        from ...data.animation import write_blend_weights
        from ...data.expressions import auto_rows, check_mapping, merged_rows

        p = ctx.params
        where, curves, values, frames = cls._curves(ctx.input("expressions"), p["source_skeleton"])
        if not curves:
            raise Invalid(Msg("E-EXPRMAP-NOCURVES"))
        path, shapes = cls._target(ctx.input("target"), p["target_skeleton"])
        if not shapes:
            raise Invalid(Msg("E-EXPRMAP-NOSHAPES", path=path or "（没有骨架）"))
        ctx.stage("配对表情")
        rows, guessed = merged_rows(p["mapping"], auto_rows(curves, shapes))
        check_mapping(rows, curves, shapes)
        pairs = [(r["src"][0], r["dst"][0]) for r in rows if r["src"] and r["dst"]]
        if not pairs:
            raise Invalid(Msg("E-EXPRMAP-NOPAIR", curves="、".join(curves[:4]), shapes="、".join(shapes[:4])))
        said = [f"{d}←{s}" for s, d in pairs if any(r["part"] in guessed and r["src"] == [s] for r in rows)]
        if said:
            ctx.say("I-EXPRMAP-GUESS", count=len(said), pairs="、".join(said[:12]) + (" 等" if len(said) > 12 else ""))
        ctx.stage("写表情")
        weights = np.zeros((len(frames), len(shapes)), np.float32)
        for s, d in pairs:
            weights[:, shapes.index(d)] = np.nan_to_num(values[:, curves.index(s)])
        unused = [c for c in curves if c not in {s for s, _ in pairs}]
        ctx.say("I-EXPRMAP-DONE", driven=len(pairs), total=len(shapes), frames=len(frames),
                unused=("、".join(unused[:6]) + (" 等" if len(unused) > 6 else "")) if unused else "没有")
        info = {"expression_retarget": {"from": where, "pairs": len(pairs)}}
        return {"character": write_blend_weights(ctx.input("target"), ctx.outputs["character"], path, frames, shapes,
                                                 weights, info)}


class StandardHuman(NodeDef):
    """标准人：一个通用的 T-pose 人体，带蒙皮和骨架，无需任何输入即可输出。

    只有骨架的结果（Kimodo / Sketch2Anim 动作生成、动捕清理、导入的 BVH）在视图中只显示为若干线段，
    难以判断动作质量。将动作施加到身体上即可直观判断：「动作重定向」`core.retarget`（动作接骨架动画，目标接本节点）
    把动作换到这副骨架上，「线性蒙皮变形」`core.linear_skin`（骨架动画接重定向的结果，蒙皮角色接本节点）挂回身体；
    本节点的一个输出口同时连这两处。本节点提供一个随时可用、符合内部标准的身体。

    身体读取的是由管理员自行下载并同意许可的 SMPL-X 中性体型（`data/skeleton.py neutral_body`：官方的
    静止姿势网格、蒙皮权重、面片、关节回归器）。骨骼名、关节轴、每点受影响的骨骼数与解算器输出蒙皮角色时
    经过同一处转换（`data/skeleton.py body_character`），因此与 GVHMR / WHAM / 自动绑定输出的人物采用同一约定。

    SMPL-X 的模型文件需在官网注册后下载，许可仅限非商业科研用途，因此本节点自行声明 `Licence`
    （核心节点默认为「基础」，不同的许可由 `NodeDef.licence` 声明）。许可仅为标签，不影响节点设计。
    """

    id = "core.standard_human"
    # 在菜单中位于「读取 · 三维文件」类别，与「导入 USD」「导入 BVH」并列：使用者要找的是获取一个人体的途径。
    # 代码放在本文件中，是因为其产物是蒙皮角色并经过 data/skeleton.py 的转换，
    # 而不是因为它读取文件（它没有任何文件参数）。
    category = "read_scene"
    on_node = ("height_cm",)
    licence = Licence(NONCOMMERCIAL, registration=True, uses=("SMPL-X",), note="身体是 SMPL-X 的中性体型：模型文件要在 smpl-x.is.tue.mpg.de 注册后自己下载"
                                     "（后台「扩展包」的「手动下载」），非商用：许可只许非商业的科研、教学和艺术项目，不许再分发。"
                                     "这个节点本身不带任何模型数据。")
    outputs = (Port("character", "scene.character", "蒙皮角色"),)

    class Params(NodeParams):
        # 只有身高一个参数。「动作重定向」按目标自身的骨骼长度算髋高比例（水平位移 1 : 1 跟着动作），因此身高决定了
        # 预览中人物的大小和髋的高度；需要与镜头中真人的尺寸一致时，只需修改此值。
        # 下游的「3D 变换」也能缩放，但那需要使用者先知道模型自身的高度并自行计算比例。
        # 未提供体型（胖瘦）参数：SMPL-X 的体型由 300 个 betas 表示，官方未定义哪个方向、多少算「胖」，不自行设定档位。
        height_cm: float | None = P(None, label="身高", unit="cm", ge=100.0, le=250.0, group="人物",
                                    placeholder="模型的 172")

    @classmethod
    def info(cls, params, inputs):
        """一个静止姿势而非一段镜头：与 HDRI、照片一样声明为 still，使其单帧不会混入
        下游的帧范围（base.py Info.merge：存在其他输入时 still 输入不参与）。"""
        return Info((STANDARD_HUMAN_FRAME,), 0, 0, True)

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import SCENE_FILE, scene_packet
        from ...data.skeleton import neutral_body
        from ...io.usd import create_stage, save_stage, write_character

        ctx.stage("读身体模型")
        character, said = neutral_body(ctx.params["height_cm"])
        ctx.stage("写蒙皮角色")
        out, frames = ctx.outputs["character"], [STANDARD_HUMAN_FRAME]
        out.mkdir(parents=True, exist_ok=True)
        stage = create_stage(frames, {"standard_human": said})
        write_character(stage, STANDARD_HUMAN, character, frames)
        save_stage(stage, out / SCENE_FILE)
        ctx.say("I-BODY-MADE", model=said["model"], height=said["height_cm"], joints=said["joints"],
                vertices=said["vertices"])
        return {"character": scene_packet(out, frames, "scene.character",
                                         people=[STANDARD_HUMAN], still=True)}


STANDARD_HUMAN = "standard_human"  # 该人物在 USD 中的名称
STANDARD_HUMAN_FRAME = 1001  # 静止姿势只占一帧；帧号按影视惯例取 1001


class SceneRender(NodeDef):
    id = "core.scene_render"
    on_node = ("width", "height")
    keeps_overscan = False  # 渲染到相机所见的画面框内
    category = "scene_convert"
    inputs = (
        Port("scene", "scene", "场景", multi=True),
        Port("camera", "scene.camera", "相机", optional=True, expects=(SameShot(),)),
        rgb_port(optional=True),
    )
    outputs = (Port("mask", "image.1", "遮罩"), Port("depth", "image.1", "深度图", means=("scale",)), normal_port())

    class Params(NodeParams):
        width: int | None = P(None, label="画面宽度", unit="px", ge=16, le=16384, group="画面", placeholder="相机的",
                              applies=Not(Wired("image")))
        height: int | None = P(None, label="画面高度", unit="px", ge=16, le=16384, group="画面", placeholder="相机的",
                               applies=Not(Wired("image")))
        space: Literal["camera", "world"] = P("camera", label="法线坐标系", group="法线",
                                              option_labels={"camera": "相机", "world": "世界"})

    @classmethod
    def info(cls, params, inputs):
        infos = inputs.get("image") or [i for port in ("camera", "scene") for i in inputs.get(port, [])]
        info = Info.merge(infos)
        if not inputs.get("image") and params["width"] and params["height"]:
            info = Info(info.frames, params["width"], params["height"], info.still)
        return info

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import SIGNED, UNIT, ExrWriter
        from ...data.frames import union
        from ...data.scene import camera_of
        from ..kit.render import render_layer_frames
        from ...io.usd import camera_resolution

        scenes, camera, image, p = ctx.inputs["scene"], ctx.input("camera"), ctx.input("image"), ctx.params
        if image is not None:
            frames, (w, h) = image.meta["frames"], (image.meta["width"], image.meta["height"])
        else:
            frames = (camera.meta["frames"] if camera is not None else []) or union(scenes)
            if (p["width"] is None) != (p["height"] is None):
                raise Invalid(Msg("E-PROJECT-SIZEPAIR"))
            cam_stage, cam_prim = camera_of(scenes, camera)  # 读取 prim 期间保持 stage 存活
            w, h = (p["width"], p["height"]) if p["width"] else camera_resolution(cam_prim) or (0, 0)
            if not w:
                raise Invalid(Msg("E-PROJECT-NOSIZE"))
        if not frames:
            raise Invalid(Msg("E-PROJECT-NOFRAMES"))
        world = p["space"] == "world"
        writers = {}  # 只处理有连线的输出；光栅化无论如何只运行一次
        if "mask" in ctx.wanted:
            writers["mask"] = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True)
        if "depth" in ctx.wanted:
            writers["depth"] = ExrWriter(ctx.outputs["depth"], 1, validity=True, scale="metric")
        if "normal" in ctx.wanted:
            writers["normal"] = ExrWriter(ctx.outputs["normal"], 3, validity=True, value_range=SIGNED, half=True, space=p["space"])
        ctx.stage("投影场景")
        seen = shaded = False
        for done, (f, layers, cam_to_world) in enumerate(render_layer_frames(scenes, camera, frames, w, h), 1):
            r = cam_to_world[:3, :3] / np.linalg.norm(cam_to_world[:3, :3], axis=0, keepdims=True)
            if "mask" in writers:
                writers["mask"].add(f, layers.coverage)
            if "depth" in writers:
                writers["depth"].add(f, layers.depth, layers.hit)
            if "normal" in writers:
                writers["normal"].add(f, layers.normal @ r.T if world else layers.normal, layers.shaded)
            seen, shaded = seen or bool(layers.hit.any()), shaded or bool(layers.shaded.any())
            ctx.progress(done, len(frames))
        if not seen:
            ctx.say("N-PROJECT-NOTHINGSEEN")
        elif not shaded:
            ctx.say("N-PROJECT-POINTSONLY")
        return {port: w.packet() for port, w in writers.items()}


# 对齐帧只在实际改变速度时生效（速度 1 表示只做偏移）：只在此处声明
SPEED_CHANGES = fact("speed_changes").true()


class Retime(NodeDef):
    id = "core.retime"
    version = 3  # a frame whose source time falls in a gap of the source, or of an object's own, is not made up from either side
    on_node = ("offset", "speed")
    category = "scene_build"
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)

    class Params(NodeParams):
        offset: int = P(0, label="偏移", unit="帧", group="时间")
        speed: float = P(1.0, label="速度", group="时间", gt=0)
        anchor: int | None = P(None, label="对齐帧", unit="帧", group="时间", placeholder="第一帧",
                               applies=SPEED_CHANGES)

    fact_labels = {"speed_changes": "速度不是 1"}

    @classmethod
    def facts(cls, params):
        from ..applies import Fact

        return {"speed_changes": Fact(abs(float(params.get("speed") or 1.0) - 1.0) > 1e-9)}  # facts() 必须开销小且不得抛出异常

    @classmethod
    def info(cls, params, inputs):
        """输出的帧：重定时的结果覆盖的帧与输入不同，因此需要告知节点图（base.py info）。"""
        got = Info.merge(inputs.get("scene") or [])
        if not got.frames:
            return got
        timing = cls.timing(params, got.frames)
        frames = timing.frames()
        return Info(tuple(frames), got.width, got.height, got.still)

    @classmethod
    def timing(cls, params: dict, frames):
        """本节点参数在时间上的含义（kit/retime.py Timing）：只定义一处，由 `info` 和 `cook` 读取。"""
        from ..kit.retime import Timing

        source = tuple(sorted(int(f) for f in frames))
        anchor = params["anchor"]
        speed = float(params["speed"] or 1.0)
        return Timing(speed if speed > 0 else 1.0, int(params["offset"]), source[0] if anchor is None else int(anchor), source)

    @classmethod
    def cook(cls, ctx):
        from ..kit.retime import retime

        p, src = ctx.params, ctx.input("scene")
        frames = [int(f) for f in src.meta["frames"]]
        if not frames:  # 固定机位或静止模型：对每一帧都成立，因此无需移动
            ctx.say("N-RETIME-STILL")
            return {"scene": pass_scene(ctx, src)}
        timing = cls.timing(p, frames)
        if not timing.changes_speed and not timing.offset:
            ctx.say("N-RETIME-NOTHING")
            return {"scene": pass_scene(ctx, src)}
        out, report = retime(src, ctx.outputs["scene"], timing)
        if not report["frames"]:
            raise Invalid(Msg("E-RETIME-NOFRAMES", first=frames[0], last=frames[-1]))
        ctx.say("I-RETIME-MAP", speed=timing.speed, offset=timing.offset,
                was=f"{frames[0]}–{frames[-1]}", now=f"{report['first']}–{report['last']}")
        if report["interpolated"]:
            ctx.say("N-RETIME-INTERP", count=report["interpolated"], total=len(report["frames"]))
        if report["gaps"]:
            ctx.say("N-RETIME-GAP", count=report["gaps"])
        if report["hidden"]:
            ctx.say("N-RETIME-HIDDEN", count=report["hidden"])
        if report["nearest"]:
            ctx.say("N-RETIME-NEAREST", count=report["nearest"])
        return {"scene": out}


def pass_scene(ctx, src):
    """场景原样透传到本节点自己的文件夹中（结果始终位于其节点放置的位置，且保持原有的场景类型：
    经过「重定时」的点云仍是点云）。"""
    from ...data.scene import transform

    return transform(src, ctx.outputs["scene"], np.eye(4))


NODES = (UsdPack, Take, TakeSubset, AttribDelete, CameraSpaceConvert, Transform3D, AutoGround, Retime, CurvesResample,
         StandardHuman, BakeModel, ExtractSkeleton, Retarget, LinearSkin, ExpressionRetarget, SceneRender)
