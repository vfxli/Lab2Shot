"""场景：合成场景、取出、3D 变换、自动落地（可先按重力方向放平）、重定时、重采样曲线、标准人、烘焙成模型、提取骨架、
线性蒙皮变形、场景投影成 2D。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ..base import HEAVY, Info, NodeDef, NodeParams, P, Port
from ..tags import RESEARCH
from ..expects import DistinctNames, SameShot
from ..handles import Places
from ...data.types import DEFORMING
from ...availability import Not
from ..applies import Cost, Licence, Param, Wired, fact


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
    outputs = (Port("camera", "scene.camera", "相机", narrows="input:scene"), Port("models", "scene.model", "模型", narrows="input:scene"),
               Port("points", "scene.points", "点云", narrows="input:scene"), Port("curves", "scene.curves", "三维曲线", narrows="input:scene"),
               Port("skeletons", "scene.skeleton", "骨架动画", narrows="input:scene"),
               Port("characters", "scene.character", "蒙皮角色", narrows="input:scene"))

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
            applies=Wired("source_camera"),
            help="逐帧贴合：每一帧按两台相机算一个变换，画面上严格对上，静止镜头也成立，但前后距离可能跟着解算器抖。"
                 "整段平滑：两条相机轨迹拟合一个常量变换（带尺度），动作连贯、脚不滑，但每帧不严格贴；"
                 "拟合剩下的误差会报出来。来源相机留空时只有逐帧一种")

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
        # 在几厘米长的轨迹上读出的比例只是噪声（近似静止的镜头上曾拟合出 0.034，人物被缩小到 3%），只有旋转和平移可信。
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
        def fov(c):
            ap = np.asarray(c.h_aperture_mm, np.float64)
            return np.asarray(c.focal_mm, np.float64) / np.where(ap > 0, ap, np.nan)
        fa, fb = np.broadcast_arrays(fov(a), fov(b))
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = float(np.nanmax(np.abs(fa / fb - 1.0))) if fa.size else 0.0
        if np.isfinite(diff) and diff > cls.LENS_TOLERANCE:
            raise Invalid(Msg("B-CAMSPACE-LENS", diff=diff * 100))


class Transform3D(NodeDef):
    id = "core.transform"
    version = 2
    on_node = ("translate", "rotate", "scale")
    category = "scene_build"
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)
    handles = (Places(translate="translate", rotate="rotate", scale="scale"),)  # 变换手柄，以及计算时的放置方式

    class Params(NodeParams):
        translate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="移动", help="整个场景（相机和人一起）平移，单位厘米，Y 向上。比如脚在地面下 10 厘米就 Y 填 10", widget="vec3", group="变换")
        rotate: tuple[float, float, float] = P((0.0, 0.0, 0.0), label="旋转", help="整个场景绕原点旋转，单位度，按 X、Y、Z 顺序（和 Houdini 默认一致）。常用来把地面转平或调整朝向", widget="vec3", group="变换")
        scale: float = P(1.0, label="缩放", help="整个场景统一缩放（相机位置一起缩放，画面对位不变）。人物尺寸不对时用，如 1.05 放大 5%", gt=0, group="变换")

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
            help="每帧相机里朝上的方向（相机坐标：X 右、Y 上、Z 朝向看画面的人）。一般接 GeoCalib 的「重力方向」"
                 "（点参数旁边的「提升到节点」），每帧一个；留空 = 不转正。相机从场景里找，"
                 "场景里不止一台相机时会拒绝并说清怎么挑（接「按种类取出」留下要的那一台）",
        )
        gravity_error: float | None = P(
            None, label="重力误差", unit="°", ge=0, group="放平", per_frame=True, applies=Param("gravity").set(),
            help="每帧重力方向可能差多少度（接 GeoCalib 的「重力误差」）：误差大的帧少算一些；留空 = 每帧一样看待",
        )
        source: Literal["people", "points", "none"] = P(
            "people", label="地面依据", group="地面",
            option_labels={"people": "人物脚底", "points": "点云", "none": "不落地"},
            help="人物脚底：人走在地上时最准；点云：场景里没有人物时，取相机下方的点里最密的那一层高度当地面（地面、路面），"
                 "场景要是正的（给了重力方向，或者本来就是 Y 朝上），否则斜的地面分不出一层；不落地：只按重力方向转正，"
                 "不上下移动（比如只有一台相机）",
        )
        percentile: float = P(
            10.0, label="落地帧比例", help="用多少比例的帧判断地面：取脚最低的这部分帧当作踩在地上（跳起来的帧不算）。人一直在走就用 10；经常跳就调低到 3–5", ge=1.0, le=50.0, group="地面", widget="slider",
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
        elif ctx.input("camera") is not None:
            ctx.say("N-GROUND-CAMERAUNUSED", port="camera")
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
            option_labels={"8": "8", "16": "16", "24": "24", "32": "32", "48": "48", "64": "64", "100": "100"},
            help="每条曲线重采样成多少个点。发丝交给 DCC 一般 16–32 个点就够看不出差别；要保住细小的卷曲用 48 以上。"
                 "这几档是实测过、不会把内存撑爆的值")

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
                        placeholder="先接上模型",
                        help="取网格上的哪一个分区（选项是接进来的网格自己带的分区名，如 FLAME 的 scalp、face、neck）。"
                             "网格上有哪些分区、各多少面，在节点的「数据信息」里看得到")

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
        attributes: str = P("", label="属性名", group="属性", placeholder="先接上数据",
                       help="要删掉哪些属性：空格或逗号分开，可以用 * 通配（如 `track_*` 删掉所有跟踪点的编号，"
                            "`*` 删掉全部）。这份数据上现在有哪些属性，在节点的「数据信息」里看得到；"
                            "填了但没命中的名字，算完会在节点上说一句")

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


class Retarget(NodeDef):
    """线性蒙皮变形：将一段骨架动画施加到另一副骨架 / 蒙皮角色上。

    「提取骨架」是正向操作（蒙皮角色 → 骨架动画），本节点为反向操作：将导入的 BVH 或解算得到的动作施加到
    使用者自己的角色上。数学实现在 `lab2shot_shared.motion.Retarget`，关节配对和正向运动学在 `nodes/kit/retarget.py`，
    此处只声明端口、参数和说明。
    """

    id = "core.retarget"
    category = "scene_convert"
    inputs = (Port("character", "scene.character|scene.skeleton", "目标",
                   help="动作要套到谁身上：你自己的角色或骨架。交出来的就是它，网格和蒙皮一个字节不动"),
              Port("motion", "scene.skeleton|scene.character", "动作",
                   help="动作从哪来：一段骨架动画（导入的 BVH、解算出来的人），或者另一个角色"))
    outputs = (Port("character", "scene.character|scene.skeleton", "目标", type_from="input:character"),)

    @classmethod
    def cook(cls, ctx):
        from ...data.animation import read_rig, write_animation
        from ..kit.retarget import locals_from_world_rotations, pair_rigs
        from lab2shot_shared.motion import Retarget as Fit

        target, source = ctx.input("character"), ctx.input("motion")
        ctx.stage("配对关节")
        # 目标只需带骨架，不要求自带动画（least_frames=1）：此步骤只读取其关节、层级和静止姿势
        # （Rig.skeleton() 使用 bind 姿势和第一帧的摆放），动作全部来自「动作」端口。
        # 「自动绑定」刚生成的角色只有一帧静止姿势，按 2 帧要求会在这条链的第一步就被拒绝。
        # 来源端仍要求一段动作（一帧没有可迁移的动作）。
        t_rig, s_rig = read_rig(target, least_frames=1), read_rig(source)
        pairs, aims, legs, notes = pair_rigs(t_rig, s_rig)
        for note in notes:  # 关节配对是推测得出的：需说明推测结果
            ctx.say(note.code, **note.params)
        ctx.stage("换动作")
        fit = Fit.align(t_rig.skeleton(), s_rig.skeleton(), pairs, aims, legs)
        rotations, root = fit.to_model(s_rig.world())
        local = locals_from_world_rotations(t_rig, rotations, root)
        ctx.say("I-RETARGET-DONE", joints=len(pairs), total=len(t_rig.names), scale=round(float(fit.scale), 3))
        info = {"retarget": {"joints": len(pairs), "scale": float(fit.scale), "from": s_rig.path}}
        rig = t_rig.__class__(**{**t_rig.__dict__, "frames": list(s_rig.frames)})
        # 帧号来自「动作」端口（rig 按 s_rig.frames 构建）：目标只提供骨架和网格。
        # 帧率不属于数据的属性，写出文件时由输出设置节点指定（data/units.py DEFAULT_FPS）。
        return {"character": write_animation(target, ctx.outputs["character"], rig, local, info)}



class StandardHuman(NodeDef):
    """标准人：一个通用的 T-pose 人体，带蒙皮和骨架，无需任何输入即可输出。

    只有骨架的结果（Kimodo / Sketch2Anim 动作生成、动捕清理、导入的 BVH）在视图中只显示为若干线段，
    难以判断动作质量。将动作施加到身体上即可直观判断，该步骤由「线性蒙皮变形」`core.retarget`
    完成（目标接本节点输出的人体，动作接骨架动画）；本节点提供一个随时可用、符合内部标准的身体。

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
    licence = Licence(RESEARCH, note="身体是 SMPL-X 的中性体型：模型文件要在 smpl-x.is.tue.mpg.de 注册后自己下载"
                                     "（后台「扩展包」的「手动下载」），许可只许非商业科研，不许再分发。"
                                     "这个节点本身不带任何模型数据。")
    outputs = (Port("character", "scene.character", "蒙皮角色"),)

    class Params(NodeParams):
        # 只有身高一个参数。「线性蒙皮变形」按目标自身的骨骼长度计算比例，因此身高决定了预览中人物的
        # 大小以及动作的步幅；需要与镜头中真人的尺寸一致时，只需修改此值。
        # 下游的「3D 变换」也能缩放，但那需要使用者先知道模型自身的高度并自行计算比例。
        # 未提供体型（胖瘦）参数：SMPL-X 的体型由 300 个 betas 表示，官方未定义哪个方向、多少算「胖」，不自行设定档位。
        height_cm: float | None = P(None, label="身高", unit="cm", ge=100.0, le=250.0, group="人物",
                                    placeholder="模型的 172",
                                    help="这个人多高，单位厘米（脚底在 y = 0，整个人等比缩放）。"
                                         "留空 = SMPL-X 中性体型自己的身高 172 厘米。"
                                         "「线性蒙皮变形」按目标自己的骨头长度算比例，所以这个数也决定套上来那段动作的步子有多大")

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
    # 开销：在每一帧光栅化所有网格和点
    cost = Cost(lane=HEAVY)
    on_node = ("width", "height")
    keeps_overscan = False  # 渲染到相机所见的画面框内
    category = "scene_convert"
    inputs = (
        Port("scene", "scene", "场景", multi=True),
        Port("camera", "scene.camera", "相机", optional=True, expects=(SameShot(),)),
        Port("image", "image.3", "RGB", optional=True),
    )
    outputs = (Port("mask", "image.1", "遮罩"), Port("depth", "image.1", "深度图", means=("scale",)), Port("normal", "image.3", "法线图", means=("space",)))

    class Params(NodeParams):
        width: int | None = P(None, label="画面宽度", unit="px", ge=16, le=16384, group="画面", placeholder="相机的",
                              applies=Not(Wired("image")), help="没接画面时输出多宽；留空用相机记下的分辨率。接了画面就和画面一样大、一样的帧")
        height: int | None = P(None, label="画面高度", unit="px", ge=16, le=16384, group="画面", placeholder="相机的",
                               applies=Not(Wired("image")), help="没接画面时输出多高；留空用相机记下的分辨率")
        space: Literal["camera", "world"] = P("camera", label="法线坐标系", group="法线",
                                              option_labels={"camera": "相机", "world": "世界"},
                                              help="相机：X 向右、Y 向上、Z 朝向镜头；世界：场景的世界方向，镜头动了法线也不变。点云没有法线，只进遮罩和深度图")

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
    on_node = ("offset", "speed")
    category = "scene_build"
    inputs = (Port("scene", "scene", "场景"),)
    outputs = (Port("scene", "scene", "场景", type_from="input:scene"),)

    class Params(NodeParams):
        offset: int = P(0, label="偏移", unit="帧", group="时间",
                        help="结果整体往后挪几帧，负数往前。1001 起的镜头填 +12 就从 1013 起")
        speed: float = P(1.0, label="速度", group="时间", gt=0,
                         help="一帧走几帧源上的帧，和 Nuke Retime 的 speed 一样。1 = 不动一帧；"
                              "1.25 = 30 帧的动作放进 24 帧的镜头（30 ÷ 24），中间的帧插出来；0.5 = 慢一半、帧数翻倍")
        anchor: int | None = P(None, label="对齐帧", unit="帧", group="时间", placeholder="第一帧",
                               applies=SPEED_CHANGES,
                               help="改速度时哪一帧保持不动。留空 = 接进来的第一帧")

    fact_labels = {"speed_changes": "速度不是 1"}

    @classmethod
    def facts(cls, params):
        from ..applies import Fact

        return {"speed_changes": Fact(abs(float(params.get("speed") or 1.0) - 1.0) > 1e-9)}  # facts() 必须开销小且不得抛出异常

    @classmethod
    def info(cls, params, inputs):
        """输出的帧及其速率：重定时是唯一一个结果覆盖的帧与输入不同的节点，因此需要告知节点图（base.py info）。"""
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

        first, last = int(frames[0]), int(frames[-1])
        anchor = params["anchor"]
        speed = float(params["speed"] or 1.0)
        return Timing(speed if speed > 0 else 1.0, int(params["offset"]),
                      first if anchor is None else int(anchor), first, last)

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
        if report["nearest"]:
            ctx.say("N-RETIME-NEAREST", count=report["nearest"])
        return {"scene": out}


def pass_scene(ctx, src):
    """场景原样透传到本节点自己的文件夹中（结果始终位于其节点放置的位置，且保持原有的场景类型：
    经过「重定时」的点云仍是点云）。"""
    from ...data.scene import transform

    return transform(src, ctx.outputs["scene"], np.eye(4))


NODES = (UsdPack, Take, TakeSubset, AttribDelete, CameraSpaceConvert, Transform3D, AutoGround, Retime, CurvesResample,
         StandardHuman, BakeModel, ExtractSkeleton, Retarget, SceneRender)
