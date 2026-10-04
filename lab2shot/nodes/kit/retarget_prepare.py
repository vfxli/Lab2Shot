"""「重定向预处理」的结果：写在 USD 骨架上的标注，以及「自动姿态」的算法。

预处理（core/retarget.py RetargetPrepare）交出「源」「目标」两个包：输入原样，加上各自 UsdSkel.Skeleton prim 上的四项
标注（像 HumanIK 的 characterization，两侧各自标部位，配对按部位）：

    customData lab2shot:retarget:parts       部位 id -> 关节名列表（data/joints.py joint_keys，链按根到梢）
    customData lab2shot:retarget:ignored     不送进解算器的关节名
    attribute  lab2shot:retarget:referencePose  matrix4d[]，关节局部（与 joints 同序），不随时间变：修正后的参考姿态
    customData lab2shot:retarget:prepared    骨架路径、参考姿态从哪来（bind / first / frame 与帧号）、节点版本
    customData lab2shot:retarget:scale       尺寸归一：{"factor": 系数, "offset": 骨架空间里的平移}（没有 = 没缩放）

参考姿态只写在这里：动画、bindTransforms、restTransforms 都不动，蒙皮和 DCC 里的绑定不受影响。

例外是「尺寸归一」（scale_rig）：学习式解算器只见过站在地面上的人体尺寸，8 米高的角色要先缩到人的大小、站到地面上再
送进去。系数不是 1 时，这一侧在骨架自己的空间里做一次 p ↦ 系数·p + offset（Sizing）：绑定姿势、静止姿势、动画、参考
姿态的平移按它变（子关节的局部平移只乘系数，根关节的再加 offset），旋转不变；蒙皮网格的 geomBindTransform 左乘它（点、
混合变形都不用改）。offset 让这一侧在世界里正好是「绕地面原点缩放、脚底落在 y = 0」：世界里 p ↦ 系数·(p − (0, 地面, 0))，
地面 = 参考姿态里最低的脚踝减去脚踝离脚底的高度（kit/retarget.py ground_of、data/animation.py sole_height）。只缩不挪
（绕原点）时，缩小的角色悬在地面下方（实测 Kinematic Refinement、SATA 的腿方向一致度 0.7 → 1.0）。骨架 prim 的摆放
不动。系数和 offset 记在 SCALE_KEY 上，「重定向后处理」按它把结果还原成原尺寸、原位置（restore_size），解算器不知道这件
事。系数 1 什么都不做（不挪到地面）：与没有这一步完全一样。

解算器（families/rig_retarget.py）和「重定向后处理」只认带这些标注的输入。写（stamp）与读（prepared）只在本模块：
其余代码不直接碰这些键，数据也不走包的 meta。

「自动姿态」不是另一套隐式步骤：按「摆 T 姿 + 朝前方向对齐」算出参考姿态（auto_reference），再换算成相对未修正参考的
逐关节 T·R·S 行（pose_rows，「初始姿势」参数的格式），点「自动姿态」就是把这些行写进参数，手动修正就是在这些行上改。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lab2shot_shared import motion as mo

from ...data.joints import PARTS, joint_keys
from ...errors import Invalid
from ... import i18n
from ...messages import Msg

PARTS_KEY = "lab2shot:retarget:parts"
IGNORED_KEY = "lab2shot:retarget:ignored"
REFERENCE_ATTR = "lab2shot:retarget:referencePose"
PREPARED_KEY = "lab2shot:retarget:prepared"
SCALE_KEY = "lab2shot:retarget:scale"

# 「自动尺寸」：标准人体的腿长（大腿根 → 膝 → 踝，cm；身高 170 cm 左右的成人约 85），一侧的腿长在标准的这个倍数范围
# 内时不缩放（系数 1）；超出时缩放到正好标准腿长
STANDARD_LEG_CM = 85.0
NORMAL_SIZE = (0.67, 1.5)


# ---------------------------------------------------------------- 写


@dataclass(frozen=True)
class Sizing:
    """尺寸归一：骨架自己的空间里 p ↦ factor·p + offset（offset 只加在根关节的局部上，和每个关节的骨架空间矩阵上）。
    factor 1 = 不变（offset 也是 0）。"""

    factor: float = 1.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def identity(self) -> bool:
        return self.factor == 1.0

    def inverse(self) -> "Sizing":
        if self.identity:
            return self
        d = np.asarray(self.offset, np.float64)
        return Sizing(1.0 / self.factor, tuple(float(v) for v in -d / self.factor))

    def skel(self, m: np.ndarray) -> np.ndarray:
        """骨架空间里的矩阵 [...,4,4]（bindTransforms、关节到骨架）：平移 factor·t + offset，旋转不变。"""
        out = np.array(m, np.float64)
        if not self.identity:
            out[..., :3, 3] = self.factor * out[..., :3, 3] + np.asarray(self.offset)
        return out

    def local(self, m: np.ndarray, parents) -> np.ndarray:
        """关节局部 [...,J,4,4]（动画、restTransforms、参考姿态）：子关节的平移乘 factor，根关节的再加 offset。"""
        out = np.array(m, np.float64)
        if not self.identity:
            out[..., :3, 3] *= self.factor
            roots = np.flatnonzero(np.asarray(parents) < 0)
            out[..., roots, :3, 3] += np.asarray(self.offset)
        return out

    def matrix(self) -> np.ndarray:
        out = np.diag([self.factor, self.factor, self.factor, 1.0])
        out[:3, 3] = self.offset
        return out

    def as_data(self) -> dict:
        return {"factor": float(self.factor), "offset": [float(v) for v in self.offset]}


def sizing_for(factor: float, placement: np.ndarray, ground: float) -> Sizing:
    """世界里「绕地面原点 (0, ground, 0) 缩放 `factor` 倍、地面落到 y = 0」（p ↦ factor·(p − G)）换到骨架自己的空间
    （`placement`：骨架 prim 到世界 [4,4]，第一帧）：factor·p + offset，offset = P⁻¹ 作用在点 factor·(t − G) 上。"""
    if factor == 1.0:
        return Sizing()
    t = np.asarray(placement, np.float64)[:3, 3]
    point = factor * (t - np.array([0.0, ground, 0.0]))
    d = np.linalg.solve(np.asarray(placement, np.float64)[:3, :3], point - t)
    return Sizing(float(factor), tuple(float(v) for v in d))


def ground_level(packet, rig, feet: list[int], pose: np.ndarray) -> float:
    """一侧的地面高度（世界 y）：姿势 `pose`（世界 [J,4,4]，参考姿态）里最低的脚踝（`feet`，kit/retarget.py ground_of）
    减去脚踝离脚底的高度（data/animation.py sole_height，按绑定姿势和蒙皮网格量；没有网格时 0：脚踝就当脚底）。"""
    from ...data.animation import sole_height

    ankle = float(min(np.asarray(pose, np.float64)[j, 1, 3] for j in feet))
    sole = sole_height(packet, rig, feet) if feet else None
    return ankle - (sole or 0.0)


# ---------------------------------------------------------------- 写


def stamp(stage, rig, parts: dict[str, list[str]], ignored: list[str], reference: np.ndarray, info: dict,
          sizing: Sizing = Sizing()) -> None:
    """在 `stage`（当前编辑层）里给 `rig`（data/animation.py Rig）的 Skeleton prim 写上标注。`parts`：部位 ->
    关节名（joint_keys）；`ignored`：忽略的关节名；`reference`：修正后的参考姿态，关节局部 [J,4,4]（根关节相对
    Skeleton prim，与 joints 同序；已经按 `sizing` 变过）；`info`：参考姿态的来源与节点版本（PREPARED_KEY 里的其余项）；
    `sizing`：尺寸归一（数据本身由 scale_rig 变换，这里只记下来；不缩放不写）。

    customData 的字典按层合并：输入要是已经预处理过一次，弱层里的部位会漏进来。所以每个部位都写（没有的写空表），
    这一层的部位表就是完整的一份。"""
    from pxr import Sdf, Vt

    prim = stage.GetPrimAtPath(rig.path)
    prim.SetCustomDataByKey(PARTS_KEY, {p: Vt.StringArray([str(n) for n in parts.get(p, [])]) for p in PARTS})
    prim.SetCustomDataByKey(IGNORED_KEY, Vt.StringArray([str(n) for n in ignored]))
    prim.CreateAttribute(REFERENCE_ATTR, Sdf.ValueTypeNames.Matrix4dArray, custom=True).Set(_usd_matrices(reference))
    prim.SetCustomDataByKey(PREPARED_KEY, {"skeleton": rig.path, **info})
    if not sizing.identity:
        prim.SetCustomDataByKey(SCALE_KEY, {"factor": float(sizing.factor), "offset": Vt.DoubleArray(list(sizing.offset))})


def _usd_matrices(m: np.ndarray):
    from pxr import Vt

    return Vt.Matrix4dArray.FromNumpy(np.ascontiguousarray(np.transpose(np.asarray(m, np.float64), (0, 2, 1))))


def _np_matrices(values) -> np.ndarray:
    return np.transpose(np.array([np.array(x) for x in values], np.float64).reshape(-1, 4, 4), (0, 2, 1))


def _size_joints(stage, path: str, sizing: Sizing) -> None:
    """骨架自己的 bindTransforms（骨架空间）、restTransforms（局部）按 `sizing` 变换（scale_rig 的一半，不碰动画和网格）。"""
    from pxr import UsdSkel

    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(path))
    parents = UsdSkel.Topology(skel.GetJointsAttr().Get() or []).GetParentIndices()
    bind, rest = skel.GetBindTransformsAttr(), skel.GetRestTransformsAttr()
    if bind and bind.Get():
        bind.Set(_usd_matrices(sizing.skel(_np_matrices(bind.Get()))))
    if rest and rest.Get():
        rest.Set(_usd_matrices(sizing.local(_np_matrices(rest.Get()), parents)))


def scale_rig(stage, path: str, sizing: Sizing) -> None:
    """在 `stage` 的当前编辑层里把 `path` 那副骨架连同蒙在它上面的网格按 `sizing` 变换（骨架自己的空间）：
    bindTransforms、restTransforms、它的动画的每个平移样本（translations：子关节乘系数，根关节再加 offset；旋转、缩放不变），
    蒙皮网格的 geomBindTransform 左乘它。骨架 prim 和网格 prim 自己的变换不动。系数 1 什么都不写。"""
    from pxr import Gf, Usd, UsdSkel, Vt

    from ...data.evaluate import skin_bindings

    if sizing.identity:
        return
    skel = UsdSkel.Skeleton(stage.GetPrimAtPath(path))
    _size_joints(stage, path, sizing)
    query = UsdSkel.Cache().GetSkelQuery(skel)
    anim = query.GetAnimQuery()
    if anim:
        prim = UsdSkel.Animation(anim.GetPrim())
        order = [str(j) for j in query.GetJointOrder()]
        parents = np.asarray(query.GetTopology().GetParentIndices())
        roots = {order[j] for j in np.flatnonzero(parents < 0)}
        anim_joints = [str(j) for j in prim.GetJointsAttr().Get() or []]
        is_root = np.array([j in roots for j in anim_joints], bool)
        attr = prim.GetTranslationsAttr()

        def sized(value):
            t = np.asarray(value, np.float64).reshape(-1, 3) * sizing.factor
            t[is_root[:len(t)]] += np.asarray(sizing.offset)
            return Vt.Vec3fArray.FromNumpy(t.astype(np.float32))

        # 先全部读出再写：写进这一层的第一个样本之后，再读别的时刻就读到这一层的（已经变换过的）值了
        default = attr.Get(Usd.TimeCode.Default())
        samples = [(t, attr.Get(t)) for t in attr.GetTimeSamples()]
        if default is not None:
            attr.Set(sized(default))
        for t, value in samples:
            attr.Set(sized(value), t)
    m = sizing.matrix()
    for binding, _ in skin_bindings(stage):
        if str(binding.GetSkeleton().GetPrim().GetPath()) != path:
            continue
        for target in binding.GetSkinningTargets():
            geom = np.array(target.GetGeomBindTransform(Usd.TimeCode.Default())).T  # column vectors
            UsdSkel.BindingAPI(target.GetPrim()).CreateGeomBindTransformAttr().Set(Gf.Matrix4d((m @ geom).T.tolist()))


def restore_size(packet, path: str, sizing: Sizing) -> None:
    """「重定向后处理」交出的骨架动画还原成原尺寸、原位置：它是从变换过的目标写出来的（skeleton_animation 把整个场景
    压平），骨架的 bindTransforms、restTransforms（和跟过来的参考姿态标注）按 `sizing` 的逆变换，去掉 SCALE_KEY。动画本身
    调用方已经按原尺寸算好。"""
    from pxr import Usd, UsdSkel

    from ...data.payloads import SCENE_FILE

    if sizing.identity:
        return
    back = sizing.inverse()
    stage = Usd.Stage.Open(str(packet.path(SCENE_FILE)))
    _size_joints(stage, path, back)
    prim = stage.GetPrimAtPath(path)
    attr = prim.GetAttribute(REFERENCE_ATTR)
    if attr and attr.Get():  # 标注跟着压平的场景过来了：参考姿态也回去
        parents = UsdSkel.Topology(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or []).GetParentIndices()
        attr.Set(_usd_matrices(back.local(_np_matrices(attr.Get()), parents)))
    if prim.GetCustomDataByKey(SCALE_KEY) is not None:
        prim.ClearCustomDataByKey(SCALE_KEY)
    stage.GetRootLayer().Save()


def prepared_packet(src, out, rig, parts: dict[str, list[str]], ignored: list[str], reference: np.ndarray, info: dict,
                    sizing: Sizing = Sizing()):
    """预处理的一个输出包：输入的场景原样（作为下层），上面一层是 stamp 的标注，缩放时还有 scale_rig 的变换
    （`reference` 是原尺寸的，这里一起变换）；类型、帧和 meta 跟输入一样。"""
    from pxr import Usd

    from ...data.payloads import SCENE_FILE, scene_packet
    from ...io import usd

    frames = [int(f) for f in src.meta.get("frames") or []]
    out.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out / SCENE_FILE))
    usd.apply_conventions(stage, frames)
    stage.GetRootLayer().subLayerPaths = [str(src.path(SCENE_FILE))]
    scale_rig(stage, rig.path, sizing)
    stamp(stage, rig, parts, ignored, sizing.local(reference, rig.parents), info, sizing)
    stage.GetRootLayer().Save()
    return scene_packet(out, frames, src.type, **{k: v for k, v in src.meta.items() if k != "frames"})


# ---------------------------------------------------------------- 读


@dataclass
class Prepared:
    """预处理过的一副骨架：骨架本身（Rig，全部帧）和它的四项标注。"""

    packet: object
    rig: object  # data/animation.py Rig
    keys: list[str]  # 每个关节在参数和标注里的名字（joint_keys）
    parts: dict[str, list[int]]  # 部位 -> 关节序号（链按根到梢），这一侧的全部部位
    ignored: list[str]  # 忽略的关节名，骨架上有的
    reference_local: np.ndarray  # [J,4,4] 修正后的参考姿态，关节局部（根关节相对 Skeleton prim）
    info: dict = field(default_factory=dict)
    sizing: Sizing = field(default_factory=Sizing)  # 尺寸归一（SCALE_KEY）：rig 与 reference_local 都是变换之后的

    @property
    def scale(self) -> float:
        return self.sizing.factor

    def unscaled(self) -> "Prepared":
        """还原成原尺寸、原位置的同一份（rig 的绑定姿势、动画平移和参考姿态按逆变换）：后处理在原尺寸里算位移、髋高。
        网格还是包里变换过的那份（data/animation.py sole_height 量出来的高度要再除以系数）。"""
        from dataclasses import replace

        if self.sizing.identity:
            return self
        back = self.sizing.inverse()
        rig = replace(self.rig, bind=back.skel(self.rig.bind), local=back.local(self.rig.local, self.rig.parents))
        return replace(self, rig=rig, reference_local=back.local(self.reference_local, self.rig.parents), sizing=Sizing())

    @property
    def path(self) -> str:
        return self.rig.path

    @property
    def parents(self) -> np.ndarray:
        return np.asarray(self.rig.parents)

    @property
    def ignored_joints(self) -> list[int]:
        at = {k: j for j, k in enumerate(self.keys)}
        return sorted(at[n] for n in self.ignored if n in at)

    @property
    def reference(self) -> np.ndarray:
        """参考姿态，joint-to-world [J,4,4]（含 Skeleton prim 在第一帧的摆放）。"""
        return self.rig.placement[0] @ mo.world_from_local(self.reference_local, self.parents)

    def part_names(self) -> dict[str, list[str]]:
        return {p: [self.keys[j] for j in js] for p, js in self.parts.items()}

    def sent_parts(self) -> dict[str, list[int]]:
        """去掉忽略的关节之后的部位（空的部位不列）：真正送进解算器的。"""
        skip = set(self.ignored_joints)
        out = {p: [j for j in js if j not in skip] for p, js in self.parts.items()}
        return {p: js for p, js in out.items() if js}

    def has_meshes(self) -> bool:
        """有网格蒙皮在这副骨架上（data/evaluate.py skin_bindings）。"""
        from ...data.evaluate import skin_bindings
        from ...data.scene import open_scene

        stage = open_scene([self.packet])
        return any(str(b.GetSkeleton().GetPrim().GetPath()) == self.path and list(b.GetSkinningTargets())
                   for b, _ in skin_bindings(stage))

    def mapping_with(self, other: "Prepared"):
        """这一侧（驱动）配 `other`（被驱动）的对应关系（kit/retarget.py Mapping）：两侧同一部位配对，链内两边节数不同
        时由重定向按骨长分摊。每个部位都写成一行（两侧都没有的写两栏空），resolve_mapping 就不再推测任何部位：
        对应关系完全是预处理定下的那一份。"""
        from .retarget import resolve_mapping

        mine, theirs = self.part_names(), other.part_names()
        rows = [{"part": p, "src": mine.get(p, []), "dst": theirs.get(p, [])} for p in PARTS]
        return resolve_mapping(rows, self.rig, other.rig, "legs")


def stamped(packet) -> list[str]:
    """包里预处理过的骨架路径（场景顺序）。"""
    from pxr import UsdSkel

    from ...data.scene import open_scene

    stage = open_scene([packet])
    return [str(p.GetPath()) for p in stage.Traverse()
            if p.IsA(UsdSkel.Skeleton) and p.GetCustomDataByKey(PREPARED_KEY) is not None]


def prepared(packet, path: str | None = None, *, side: str = "", least_frames: int = 1) -> Prepared:
    """读回 stamp 写的标注：`path` 那副骨架（None：第一副预处理过的）。没有预处理过的骨架时报 E-RETARGET-UNPREPARED
    （`side`：消息里怎么称呼这个输入）。"""
    from pxr import UsdSkel

    from ...data.animation import read_rig
    from ...data.scene import open_scene

    found = stamped(packet)
    if not found or (path and path not in found):
        raise Invalid(Msg("E-RETARGET-UNPREPARED", side=side or i18n.Word("role.input")))
    rig = read_rig(packet, path or found[0], least_frames=least_frames)
    stage = open_scene([packet])
    prim = stage.GetPrimAtPath(rig.path)
    keys = joint_keys(rig.names, rig.parents)
    at = {k: j for j, k in enumerate(keys)}
    parts = {}
    for part, names in dict(prim.GetCustomDataByKey(PARTS_KEY) or {}).items():
        joints = [at[str(n)] for n in names if str(n) in at]
        if joints:
            parts[str(part)] = joints
    ignored = [str(n) for n in prim.GetCustomDataByKey(IGNORED_KEY) or [] if str(n) in at]
    got = UsdSkel.Skeleton(prim).GetPrim().GetAttribute(REFERENCE_ATTR).Get()
    local = np.transpose(np.array([np.array(m) for m in got or []], np.float64).reshape(-1, 4, 4), (0, 2, 1))
    if len(local) != len(keys):
        raise Invalid(Msg("E-RETARGET-UNPREPARED", side=side or i18n.Word("role.input")))
    info = {str(k): v for k, v in dict(prim.GetCustomDataByKey(PREPARED_KEY) or {}).items()}
    return Prepared(packet, rig, keys, parts, ignored, local, info, _sizing_of(prim.GetCustomDataByKey(SCALE_KEY)))


def _sizing_of(data) -> Sizing:
    """SCALE_KEY 的值读成 Sizing；没有或读不懂的是不缩放。"""
    try:
        factor = float(data["factor"])
        offset = tuple(float(v) for v in data["offset"])
    except (TypeError, KeyError, ValueError):
        return Sizing()
    return Sizing(factor, offset) if factor > 0 and len(offset) == 3 else Sizing()


def bases(src: Prepared, dst: Prepared):
    """解算器和后处理用的对应关系和两副参考姿态：(kit/retarget.py Mapping, Rests)。参考姿态就是预处理写下的那两副，
    这里不再挑帧、摆 T、修正；两边的身体坐标系按它们算（two_bodies）。"""
    from .retarget import Rests, two_bodies

    m = src.mapping_with(dst)
    ps, pt = src.reference, dst.reference
    tb, sb = two_bodies(m, dst.rig, pt, src.rig, ps)
    said = i18n.Word("retarget.rest.prepared")
    return m, Rests(mo.Skeleton(src.rig.names, src.parents, ps), mo.Skeleton(dst.rig.names, dst.parents, pt), said, said,
                    False, ps, pt, bodies=(tb, sb))


# ---------------------------------------------------------------- 自动姿态


class _Axes:
    """t_posed 要的「身体坐标系」：这里固定成场景的上方和按髋定的左方（reference_axes），不随姿势变。"""

    def __init__(self, axes: np.ndarray) -> None:
        self._axes = axes

    def axes(self, _positions) -> np.ndarray:
        return self._axes


def reference_axes(pose: np.ndarray, parts: dict[str, list[int]], parents=None) -> np.ndarray:
    """参考坐标系 [3,3]（列：上、前、左）：上通常是场景的 Y；给了 `parents` 时按识别引擎从这副姿势估的身体上方向
    （data/skeleton_recognition.py frame_of：躯干方向与两腿方向的反向，两者一致才可信）——它与 Y 差过 45° 时（Z 向上
    导进来、躺着的参考姿势）以它为上。左按两条大腿根在水平面上的投影。脊柱的弯曲是解剖形态，不拿来当上方。两条大腿根
    在水平面上重合时定不了朝向（E-RETARGET-FACING）：要手动改对应关系或姿态。"""
    up = np.array([0.0, 1.0, 0.0])
    if parents is not None:
        from ...data.skeleton_recognition import frame_of

        found = frame_of(parents, np.asarray(pose, np.float64)[:, :3, 3], parts)
        if found.up is not None and found.up_conf >= 0.6 and float(np.dot(found.up, up)) < np.cos(np.radians(45)):
            up = found.up
    left = pose[parts["l.thigh"][0], :3, 3] - pose[parts["r.thigh"][0], :3, 3]
    left = np.array(left, np.float64)
    left -= np.dot(left, up) * up
    length = np.linalg.norm(left)
    if length < 1e-3:
        raise Invalid(Msg("E-RETARGET-FACING"))
    return mo.aim_frame(up, left / length)


def auto_reference(pose: np.ndarray, parents, parts: dict[str, list[int]]) -> np.ndarray:
    """「自动姿态」要的参考姿态 [J,4,4]（世界）：四肢摆成标准 T 姿（kit/retarget.py t_posed，身体坐标系按 reference_axes：
    A 姿、手臂下垂都摆到 T），再绕髋把整副骨架转到朝前方向对齐（上 = +Y、左 = +X、面朝 +Z）。各家模型训练用的骨架朝向不同，两边都转到同一个
    朝向，学习式解算器看到的就是同一种输入。"""
    from .retarget import t_posed

    axes = reference_axes(pose, parts, parents)
    posed = t_posed(pose, parents, parts, _Axes(axes))
    turn = mo.aim_frame(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])) @ axes.T
    anchor = posed[parts["hips"][0], :3, 3].copy()
    posed[:, :3, :3] = turn @ posed[:, :3, :3]
    posed[:, :3, 3] = (posed[:, :3, 3] - anchor) @ turn.T + anchor
    return posed


STILL = 1e-9  # 修正量的每个数都不超过这么多：这个关节没修正，不写行


def pose_rows(before: np.ndarray, after: np.ndarray, parents, keys: list[str]) -> list[dict]:
    """把姿势 `before` 摆成 `after`（世界 [J,4,4]，同一副骨架、各关节轴长不变）写成「初始姿势」的行（JointPose：关节局部
    @ T·R·S，关节自己的正交坐标系里）：kit/retarget.py corrected(before, rows) 正好得到 `after`。每个关节的修正
    D = 修正前局部⁻¹ · 修正后局部，拆成平移和 XYZ 旋转（缩放总是 1）；没变的关节不写。"""
    from lab2shot_shared.poses import xyz_euler_deg

    from .retarget import pose_frames

    a = mo.local_from_world(pose_frames(before), parents)
    b = mo.local_from_world(pose_frames(after), parents)
    rows = []
    for j, key in enumerate(keys):
        d = np.linalg.inv(a[j]) @ b[j]
        if np.abs(d - np.eye(4)).max() <= STILL:
            continue
        rotate = xyz_euler_deg(mo.orthonormal(d[None, :3, :3]))[0]
        rows.append({"joint": key, "translate": [float(v) for v in d[:3, 3]], "rotate": [float(v) for v in rotate],
                     "scale": [1.0, 1.0, 1.0]})
    return rows


def auto_pose(before: np.ndarray, parents, keys: list[str], parts: dict[str, list[int]]) -> list[dict]:
    """「自动姿态」的建议：未修正的参考姿态 `before` 到 auto_reference 的逐关节行（pose_rows）。缺两条大腿、髋，或定不了
    朝向时没有建议（[]），页面就不给这个按钮的结果。"""
    if not all(p in parts for p in ("hips", "l.thigh", "r.thigh")):
        return []
    try:
        after = auto_reference(before, parents, parts)
    except Invalid:
        return []
    return pose_rows(before, after, parents, keys)


# ---------------------------------------------------------------- 自动尺寸


def leg_length(m, col: str, pose: np.ndarray) -> float:
    """一侧（`col`：src / dst）在一副姿势（世界 [J,4,4]）里的腿长：大腿 + 小腿 + 到脚，左右平均（kit/retarget.py
    measure 的「按腿长」，后处理算髋高用的同一个量）。与姿势无关，只看骨长。"""
    from .retarget import measure

    if m.hands or not all(q in m.src and q in m.dst for q in ("l.thigh", "l.shin", "r.thigh", "r.shin")):
        return 0.0  # 只配手、没有腿：没有身体尺寸可比（auto_scale 给 1）
    return float(measure("legs", m, col, np.asarray(pose, np.float64)[None, :, :3, 3])[0])


def auto_scale(leg_cm: float) -> float:
    """「自动尺寸」的系数：腿长在标准腿长（STANDARD_LEG_CM）的 NORMAL_SIZE 倍范围内是 1（人体尺寸，不动）；超出时
    缩放到正好标准腿长，取三位有效数字（参数里写得下、看得懂）。量不出腿长时 1。"""
    from .retarget import MEASURABLE_CM

    if not np.isfinite(leg_cm) or leg_cm <= MEASURABLE_CM:
        return 1.0
    ratio = leg_cm / STANDARD_LEG_CM
    if NORMAL_SIZE[0] <= ratio <= NORMAL_SIZE[1]:
        return 1.0
    return float(f"{1.0 / ratio:.3g}")


def effective_scale(params: dict, auto: dict) -> tuple[float, float]:
    """预处理实际用的 (动作尺寸, 目标尺寸)：参数填了就是参数（1 = 不缩放）；留空 = 自动（`auto`：{"motion": 系数,
    "target": 系数}，auto_scale 这一次的建议）。和姿态、忽略的默认自动不同，空着永远是自动：一个系数没有「空 = 不改」
    的意思，留空的框写着「自动」就得是自动，所以不看「自动记录」。"""
    out = []
    for side in ("motion", "target"):
        v = params.get(f"{side}_scale")
        out.append(float(v) if v is not None and float(v) > 0 else float(auto.get(side, 1.0)))
    return out[0], out[1]


# ---------------------------------------------------------------- 默认自动


def auto_done(record: str) -> dict:
    """「自动记录」参数（JSON）读成字典；空或读不懂的是 {}（三个「自动」都还没执行过）。"""
    import json

    try:
        got = json.loads(record or "{}")
    except ValueError:
        return {}
    return got if isinstance(got, dict) else {}


def effective(params: dict, auto: dict) -> tuple[list[dict], list[dict], list[str], list[str]]:
    """预处理实际用的 (动作初始姿势, 目标初始姿势, 动作忽略, 目标忽略)。默认自动是服务端的规则，命令行、批量没有页面
    也一样：「自动姿态」从没执行过（自动记录里没有 pose）而某一侧的行是空的，这一侧用自动姿态的行；「自动忽略」从没
    执行过（没有 ignore）而两侧名单都空，用当前忽略规则的建议。执行过之后参数就是参数本身，执行后又全部复位的空行
    也照空行算。`auto`：这一次的建议，{"pose": {"motion": 行, "target": 行}, "ignore": {"motion": [名], "target": [名]}}
    （auto_pose、suggest_ignored）。"""
    done = auto_done(params.get("auto_record") or "")
    poses = [params.get(f"{side}_pose") or [] for side in ("motion", "target")]
    if "pose" not in done:
        poses = [rows or list(auto["pose"][side]) for rows, side in zip(poses, ("motion", "target"))]
    ignored = [params.get(f"{side}_ignore") or [] for side in ("motion", "target")]
    if "ignore" not in done and not any(ignored):
        ignored = [list(auto["ignore"][side]) for side in ("motion", "target")]
    return poses[0], poses[1], ignored[0], ignored[1]
