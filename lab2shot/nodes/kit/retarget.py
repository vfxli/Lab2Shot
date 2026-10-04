"""「动作重定向」与「线性蒙皮变形」的核心步骤：两副骨架的部位对应、把一副骨架的动作换到另一副骨架上、
把一段骨架动画挂到层级完全相同的蒙皮角色上。

数学在 `lab2shot_shared.motion.BodyRetarget`（静止姿势对齐：保留目标自己的基准姿势、传转动差，或逐根骨骼按方向；
链状部位按骨长分摊弯曲）与 `hips_path`（水平 1:1 跟着动作，
髋高按比例跟着脚抬高或逐帧缩放，再加脚底差和高度偏移），核心与 worker 共用一份实现。本模块只做三件事：

  1. 对应关系（`resolve_mapping`）：节点参数里写了的部位以参数为准，没写的部位由 `data/joints.py guess()`（识别
     引擎 data/skeleton_recognition.py：名字、层级、位置、对称综合）推测；`joints.check_mapping` 校验（权威在这里，网页上的错误色只是按同样的规则做显示）；只对推测的部位
     生成说明（data/smpl.py pairing_notes）。
  2. 换动作（`retarget`）：算出目标每个配上的关节的世界旋转和髋关节的位置；髋高比例按「髋高依据」两边各量一次
     （`measure`）。
  3. 写回目标自己的层级（`target_locals`）：没配上的关节局部旋转取对齐用的基准姿势（rests：默认目标的第一帧）、局部平移和缩放取
     参考帧（第一帧），
     配上的关节只换旋转、保留自己的缩放（`motion.set_world`）；世界量先换回骨架自己的空间（Skeleton prim 可能被
     导入的单位缩放或「3D 变换」挪过）。

这两个节点不依赖第三方解算器，因此是核心节点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from lab2shot_shared import motion as mo
from lab2shot_shared.poses import trs_matrix

from ...data.joints import FINGERS, LANDMARKS, LEGS, LIMB_BONES, joint_keys
from ... import i18n
from ...errors import Invalid
from ...messages import Both, Msg

# 「动作重定向」必须两边都有的部位：髋和两条腿定朝前方向（motion.BodyRetarget.align）和髋点（两条大腿中间，hips_path）
REQUIRED = ("hips", *LEGS)
# 「髋高依据」另要的部位：身高量到头，臂长量上臂和前臂（measure）
BASIS_PARTS = {"legs": (), "height": ("head",), "arms": ("l.upperarm", "l.forearm", "r.upperarm", "r.forearm"), "none": ()}


# E-RETARGET-MISSINGPARTS 的 {why}：缺的部位是拿来做什么的，按「髋高依据」说
WHY = i18n.Words("retarget.why.", ("legs", "none", "height", "arms"))


def why_hands() -> str:
    return i18n.Word("retarget.why.hands")


BODY_PARTS = ("hips", *LEGS, "spine", "chest")  # 配上其中任何一个，这一边就是有身体的骨架


def has_body(names, parents, parts: dict[str, list[int]]) -> bool:
    """这副骨架按对应关系和名字算不算一整个身体：决定「只配手」（hands_of，计算与编辑器同一个）。这一边已经定下的对应
    关系（推测加手配之后的部位 -> 关节 `parts`）里髋、腿、脊柱、胸配上了任何一个，或名字推测出髋（data/joints.py
    guess）。所以有身体的骨架把髋那一行清空是缺部位（E-RETARGET-MISSINGPARTS），不会变成一只单独的手；名字认不出、
    只配了手的整个人仍是只配手（手长在身体上由 hand_on_body 按层级看）。"""
    from ...data.joints import guess

    return any(p in parts for p in BODY_PARTS) or "hips" in guess(list(names), parents)


def hangs_on_body(parents, hand: int) -> bool:
    """一只手长在身体上（层级）：它的子树不到整副骨架的一半——上面还挂着手臂和身体，名字认不出也一样。"""
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)

    def size(j: int) -> int:
        return 1 + sum(size(c) for c in kids.get(j, []))

    return size(hand) * 2 < len(parents)


def hands_of(src, dst, bodies: tuple[bool, bool]) -> tuple[str, ...]:
    """只配手的重定向：两边有一边的骨架没有身体（has_body：HaMeR 的 MANO 手、只有手的绑定）时，两边都配了的那几只手
    （"l" / "r"）；两边的骨架都有身体（整个身体）时为空。`src` / `dst`：部位 -> 关节（只看有哪些部位），`bodies`：
    两边 has_body。"""
    if all(bodies):
        return ()
    return tuple(s for s in ("l", "r") if f"{s}.hand" in src and f"{s}.hand" in dst)


def required(by: str, hands: tuple[str, ...] = ()) -> tuple[str, ...]:
    """两边都必须配上的部位：整个身体按「髋高依据」`by`（髋、两腿，身高加头、臂长加上臂前臂）；只有手时（hands_of）
    是手腕和五根手指（髋高比例不适用）。计算时 resolve_mapping 按它报缺，编辑器按它标必需（Retarget.choices），一份。"""
    if hands:
        return tuple(p for s in hands for p in (f"{s}.hand", *(f"{s}.{f}" for f in FINGERS)))
    return REQUIRED + BASIS_PARTS.get(by, ())


MEASURABLE_CM = 1e-3  # a length the 髋高依据 measures at or below this (cm) is a degenerate skeleton, not a body
LEGS_VARY = 0.03  # 源每帧的腿长相差超过中位数的 3%：多半是解算时没锁体型，提示一次（W-RETARGET-LEGVARIES）


@dataclass
class Mapping:
    """一次 cook 用的对应关系：每个部位两边的关节序号，以及哪些部位是推测的。"""

    rows: list[dict]  # 参数格式（名字），给说明用
    src: dict[str, list[int]]  # 部位 -> 动作骨架的关节
    dst: dict[str, list[int]]  # 部位 -> 目标骨架的关节
    guessed: set[str]
    hands: tuple[str, ...] = ()  # 只配手（hands_of）：配的是哪几只手；空 = 整个身体
    bodies: tuple[bool, bool] = (True, True)  # 两边骨架有没有身体（has_body）
    stale: list[tuple[str, list[str]]] = field(default_factory=list)  # (部位, 骨架上没有的关节)：换过骨架，这次按推测（drop_stale）


def merged_parts(mapping: list[dict] | None, src_names, src_parents, dst_names, dst_parents,
                 src_rest=None, dst_rest=None, src_weights=None, dst_weights=None) -> tuple[list[dict], set[str]]:
    """参数里的对应关系配上推测补齐的部位（data/joints.py merged_rows）：(行, 推测的部位)。计算（resolve_mapping）和编辑器
    （Retarget.choices：只配手时标哪些必需）都从这里得到部位，所以用户手配的行两边同样算数。`*_rest`：两侧的静止位置
    [J,3]：识别引擎按名字、层级、位置、对称一起判断（data/skeleton_recognition.py；自动绑定的 bone_0、bone_1……
    靠位置认出）。`*_weights`：两侧带蒙皮权重的关节（data/animation.py Rig.weighted；不驱动顶点的关节不参与对应）。"""
    from ...data.joints import auto_rows, merged_rows, part_names

    return merged_rows(mapping, auto_rows(part_names(list(src_names), src_parents, src_rest, src_weights),
                                          part_names(list(dst_names), dst_parents, dst_rest, dst_weights)))


def rest_of(rig) -> np.ndarray:
    """一副骨架绑定姿势里各关节的位置 [J,3]（骨架自己的空间）：识别引擎（data/skeleton_recognition.py）按它判断部位、
    角色和朝向——推测对应、忽略建议、编辑器里的依据都传这同一份，识别结果按骨架缓存共用。"""
    return np.asarray(rig.bind, np.float64)[:, :3, 3]


def drop_stale(mapping: list[dict] | None, src_names, src_parents, dst_names, dst_parents) -> tuple[list[dict] | None, list[tuple[str, list[str]]]]:
    """参数里的对应关系，去掉点了这两副骨架上没有的关节的行：(留下的行, [(去掉的部位, 它点的、骨架上没有的关节)])。换了骨架（换了角色或动作）时，
    之前点的、自动对应写下的行指的是旧骨架的关节——这些部位这次按新骨架推测，并说一句（notes：W-MAP-STALE），
    而不是整个节点报错、连编辑器都进不去（初始姿势、忽略骨骼对没有的关节同样是跳过并说明）。某一边故意留空（这一边
    不配）的行不算：只看写了却找不到的关节名。"""
    from ...data.joints import joint_keys

    if not mapping:
        return mapping, []
    keys = {"src": set(joint_keys(list(src_names), src_parents)), "dst": set(joint_keys(list(dst_names), dst_parents))}
    kept, gone = [], []
    for row in mapping:
        missing = [n for c in ("src", "dst") for n in (row.get(c) or [] if isinstance(row, dict) else []) if n not in keys[c]]
        if missing:
            gone.append((str(row.get("part", "")), missing))
        else:
            kept.append(row)
    return kept, gone


def resolve_mapping(mapping: list[dict] | None, source, target, by: str = "legs") -> Mapping:
    """参数里的对应关系（None = 全部推测）配上推测补齐的部位，校验后换成关节序号。`source` / `target` 是
    data/animation.py Rig。两边缺了按「髋高依据」`by` 必需的部位（required）时报 E-RETARGET-MISSINGPARTS，
    说清是哪一边、缺的部位拿来做什么。"""
    from ...data.joints import check_mapping, part_joints, part_label

    mapping, stale = drop_stale(mapping, source.names, source.parents, target.names, target.parents)
    rows, guessed = merged_parts(mapping, source.names, source.parents, target.names, target.parents,
                                 rest_of(source), rest_of(target), source.weighted, target.weighted)
    check_mapping(rows, {"src": (list(source.names), source.parents, i18n.Word("role.motion"), source.bind[:, :3, 3]),
                         "dst": (list(target.names), target.parents, i18n.Word("role.target"), target.bind[:, :3, 3])})
    src = part_joints(rows, list(source.names), source.parents, "src")
    dst = part_joints(rows, list(target.names), target.parents, "dst")
    bodies = has_body(source.names, source.parents, src), has_body(target.names, target.parents, dst)
    hands = hands_of(src, dst, bodies)
    for have, rig in ((src, i18n.Word("role.motion")), (dst, i18n.Word("role.target"))):
        missing = [part_label(p) for p in required(by, hands) if p not in have]
        if missing:
            why = why_hands() if hands else WHY.get(by, WHY["legs"])
            raise Invalid(Msg("E-RETARGET-MISSINGPARTS", parts=missing, rig=rig, why=why))
    return Mapping(rows, src, dst, guessed, hands, bodies, stale)


def notes(m: Mapping) -> list[Msg]:
    """推测出来的那些部位的说明（手动指定的部位不再说明；全是手动的就一条也没有）。"""
    from ...data.joints import part_label
    from ...data.smpl import pairing_notes

    said = []
    if m.stale:
        said.append(Msg("W-MAP-STALE", parts=[part_label(part) for part, _ in m.stale],
                        joints=[n for _, gone in m.stale for n in gone][:6]))
    if not m.guessed:
        return said
    driven = {r["part"]: list(r["src"]) for r in m.rows if r["part"] in m.guessed and r["src"] and r["dst"]}
    have = {r["part"] for r in m.rows if r["part"] in m.guessed and r["dst"]}
    return said + pairing_notes(driven, have)


@dataclass
class Result:
    rotations: dict[int, np.ndarray]  # 目标关节 -> 世界旋转 [F,3,3]
    placed: dict[int, np.ndarray]  # 放到给定世界位置的目标关节 -> 位置 [F,3]：整个身体是髋；只有手是每只手腕
    scale: float  # 最终的髋高比例（依据量出来的比 × 修正系数）；水平位移不缩放；只有手时 1
    by: str  # 依据：legs / height / arms / none
    factor: float  # 修正系数
    measured_cm: tuple[float, float, float]  # 源的依据量逐帧的最小、中位、最大（世界单位，cm；不缩放时全 0）
    target_cm: float  # 目标的依据量
    joints: int  # 转动的目标关节数
    size: float = 1.0  # 角色缩放：目标整体放大的倍数（根关节的局部加均匀缩放，sized）
    rest_said: str = ""  # 两边用的基准姿势，给 I-RETARGET-DONE
    rest: np.ndarray | None = None  # 目标对齐用的基准姿势（骨架自己的空间，joint-to-skeleton [J,4,4]）：没配上的关节保持它
    unknown: tuple[list[str], list[str]] = ([], [])  # 两边「初始姿势」里骨架没有的关节名（Rests.unknown）
    wrist_kept: bool = False  # 只配手而目标的手长在身体上：手腕不动，只传手指（retarget_hands）
    # 整个身体：(移动的关节 = 髋, 两条大腿, 髋点每帧的世界位置 [F,3])——转完之后按层级把两大腿中点放到这里
    hips: tuple[int, tuple[int, int], np.ndarray] | None = None

    @property
    def by_label(self) -> str:
        return Both.of(lambda: SCALE_BY[self.by])


@dataclass
class Size:
    """「角色缩放」：目标整体放大的倍数，和它从哪来（给 I-RETARGET-DONE 说清）。"""

    factor: float
    basis: str  # none / manual / height / legs（按身高缩放、一边没有头时退回腿长）
    source_cm: float = 0.0  # 动作这边的身高（或腿长），逐帧中位数
    target_cm: float = 0.0  # 目标的，没缩放时


def character_size(by: str, manual: float, source, target, m: Mapping) -> Size:
    """「角色缩放」的倍数：`by` none 不缩放、manual 用 `manual`、height = 动作的身高 ÷ 目标的身高（measure 的同一条链：
    腿长 + 髋到头沿脊柱的骨长，动作逐帧量取中位数，目标量参考帧）；两边有一边没配上头时退回腿长比（basis "legs"，
    调用方提示 W-RETARGET-SIZELEGS）。只配手时没有身体可比，不缩放。"""
    if by == "none" or m.hands:
        return Size(1.0, "none")
    if by == "manual":
        return Size(float(manual), "manual")
    basis = "height" if "head" in m.src and "head" in m.dst else "legs"
    src = float(np.median(measure(basis, m, "src", source.world()[..., :3, 3])))
    dst = float(measure(basis, m, "dst", target.world()[:1, :, :3, 3])[0])
    if min(src, dst) <= MEASURABLE_CM:
        raise Invalid(Msg("E-RETARGET-NOLENGTH", what=SCALE_BY[basis], rig=i18n.Word("role.motion" if src <= MEASURABLE_CM else "role.target")))
    return Size(src / dst, basis, src, dst)


def sized(rig, factor: float):
    """目标整体放大 `factor` 倍：根关节的局部变换加均匀缩放（参考帧起每帧都带），子关节的平移在父关节空间里跟着放大，
    蒙皮网格按关节的世界矩阵跟着放大。放在根关节上而不是 Skeleton prim 上：「动作重定向」交出的是骨架动画，
    「线性蒙皮变形」只搬关节的局部（prim 的摆放不跟过去），交付的 USD / FBX 里它就是根关节动画上的 scale，DCC 读得到。"""
    from dataclasses import replace

    if factor == 1.0:
        return rig
    local = np.array(rig.local, np.float64)
    for j in np.flatnonzero(np.asarray(rig.parents) < 0):
        local[:, j, :3, :3] *= factor
    return replace(rig, local=local)


SCALE_BY = i18n.Words("retarget.scale_by.", ("legs", "height", "arms", "none"))
# 依据量沿哪些部位量（按顺序连成一条或几条链，取几条链的平均）；两边都配了的部位才算
TRUNK = ("hips", "spine", "chest", "neck", "head")


def measure(by: str, m: Mapping, col: str, positions: np.ndarray) -> np.ndarray:
    """一边（`col`：src 动作 / dst 目标）的依据量，逐帧 [F]：腿长（大腿 + 小腿 + 到脚）、身高（腿长 + 髋到头沿脊柱的
    骨长）、臂长（上臂 + 前臂 + 到手），左右取平均；只量两边都配了的部位。链状部位（脊柱、颈）按这一边自己的节数连，
    两边节数不同也各量各的。必需的部位 resolve_mapping 已按同一个 `by` 查过。"""
    side, other = (m.src, m.dst) if col == "src" else (m.dst, m.src)

    def chain(parts: tuple[str, ...]) -> list[int]:
        return [j for q in parts if q in side and q in other for j in side[q]]

    def legs() -> list[list[int]]:
        return [chain((f"{s}.thigh", f"{s}.shin", f"{s}.foot")) for s in ("l", "r")]

    if by == "legs":
        return mo.chain_lengths(positions, legs())
    if by == "height":
        return mo.chain_lengths(positions, legs()) + mo.chain_lengths(positions, [chain(TRUNK)])
    if by == "arms":
        return mo.chain_lengths(positions, [chain((f"{s}.upperarm", f"{s}.forearm", f"{s}.hand")) for s in ("l", "r")])
    return np.zeros(positions.shape[0])


GROUND_SAID = i18n.Words("retarget.ground.", ("ankles", "below", "knees"))


def ground_of(rig, side: dict[str, list[int]]) -> tuple[list[int], str]:
    """一边离地高度从哪几个关节量（髋高的「跟着脚抬高 / 按比例」、脚底对齐都用它，一处定）：两只脚踝（配上的「脚」）；
    没配脚时是每条腿小腿下面那一节（层级里小腿的子关节，取离小腿最远的那个：脚踝常常没配但在骨架里）；小腿下面
    什么都没有时是小腿本身（膝）。返回 (关节, 说法：GROUND_SAID 的键)，完成句里说明没按脚踝量的时候。"""
    if all(f"{s}.foot" in side for s in ("l", "r")):
        return [side[f"{s}.foot"][0] for s in ("l", "r")], "ankles"
    rest = np.asarray(rig.bind, np.float64)[:, :3, 3]
    out, how = [], "below"
    for s in ("l", "r"):
        if f"{s}.foot" in side:
            out.append(side[f"{s}.foot"][0])
            continue
        shin = side[f"{s}.shin"][0]
        kids = [j for j, p in enumerate(rig.parents) if p == shin]
        if kids:
            out.append(max(kids, key=lambda j: float(np.linalg.norm(rest[j] - rest[shin]))))
        else:
            out.append(shin)
            how = "knees"
    return out, how


REST_LABEL = ("bind", "first", "frame")  # retarget.rest.<which>, {frame}: rest_said


def rest_said(which: str, frame) -> str:
    return i18n.Word(f"retarget.rest.{which}", **({"frame": frame} if which == "frame" else {}))
# 基准姿势摆成 T 姿时每根肢体骨头（data/joints.py LIMB_BONES，同一张表）的朝向：身体坐标系里的方向（motion.Body.axes
# 的列：0 上、1 前、2 左，左边的手臂朝左、右边的朝右）；锁骨只放平（None：去掉抬起、保留前后的角度——各家锁骨前后
# 本来就不一样，A 姿里却都压低了 16–27°）；脚只转向正前（"heading"：保留它向下的斜度——踝比前脚掌高，脚平踩地面时
# 踝到脚尖这根骨头本来就朝下，斜多少每副骨架不同：UniRig 的 37°、SOMA 的 23°。放平它，站着时脚尖就翘起来）
T_TOWARD = {"clavicle": None, "upperarm": (2, 1), "forearm": (2, 1), "thigh": (0, -1), "shin": (0, -1), "foot": "heading"}


def _toward(part: str) -> tuple[int, int] | str | None:
    """T_TOWARD for one side's part: the right side's arms point the other way along the body's left axis."""
    got = T_TOWARD[part[2:]]
    return got if got is None or isinstance(got, str) else (got[0], -got[1] if part.startswith("r.") and got[0] == 2 else got[1])


T_BONES = tuple((a, b, _toward(a)) for a, b in LIMB_BONES)


def rest_pose(rig, which: str, frame: int | None, said: str) -> np.ndarray:
    """一副骨架的基准姿势（joint-to-world [J,4,4]，含 Skeleton prim 的摆放）：bind = 文件里的绑定姿势（USD
    bindTransforms / FBX cluster 的 TransformLinkMatrix，网格绑定时的样子）；first = 它的第一帧（DCC 打开文件看到的样子：
    AccuRIG 的 FBX 绑定姿势是 A、第一帧是 T）；frame = 指定的那一帧，不在这副骨架的帧里报 E-RETARGET-RESTFRAME。"""
    if which == "bind":
        return rig.placement[0] @ np.asarray(rig.bind, np.float64)
    if which == "first":
        return rig.world()[0]
    if frame is None or int(frame) not in rig.frames:
        raise Invalid(Msg("E-RETARGET-RESTFRAME", rig=said, frame=frame if frame is not None else i18n.Word("retarget.frame_unset"),
                          first=rig.frames[0], last=rig.frames[-1]))
    return rig.world()[rig.frames.index(int(frame))]


def sized_pose(pose: np.ndarray, parents, factor: float) -> np.ndarray:
    """一副姿势 [J,4,4]（世界）放大 `factor` 倍，和 sized 放大骨架的是同一个变换：绕每个根关节自己缩放（sized 把
    缩放乘在根关节的局部上），子关节跟着。基准姿势无论取绑定姿势、第一帧还是指定帧，都在这里放大一次。"""
    if factor == 1.0:
        return pose
    pose = np.asarray(pose, np.float64)
    parents = np.asarray(parents)
    root = np.arange(len(parents))
    for j in range(len(parents)):  # parents come before children
        if parents[j] >= 0:
            root[j] = root[parents[j]]
    scale = np.diag([factor, factor, factor, 1.0])
    about = pose[root] @ scale @ np.linalg.inv(pose[root])
    return about @ pose


def body_of(side: dict[str, list[int]], parents, pos=None, marks=LANDMARKS) -> mo.Body:
    """一边的身体坐标系（motion.Body）：两条大腿、两条小腿（必需部位，按对应关系），躯干是从髋到手臂与头分开的地方
    （motion.trunk_of，和模型类节点同一个函数）：`marks` 里这一边配上的手、头（两边一起用时只取两边都配了的，
    two_bodies）定终点，不够两个时按层级（`pos`：这一副姿势的关节位置，看左右对称和在不在髋上）。挂在髋下的道具、
    尾巴、裙摆，手上的道具都不算。"""
    thighs, shins = (side["l.thigh"][0], side["r.thigh"][0]), (side["l.shin"][0], side["r.shin"][0])
    return mo.Body.of_hierarchy(parents, thighs, shins, [side[p][0] for p in marks if p in side], pos)


def two_bodies(m: Mapping, target, pt: np.ndarray, source, ps: np.ndarray) -> tuple[mo.Body, mo.Body]:
    """两边的身体坐标系（目标，动作），躯干终点用两边都配上的手、头：一边少配了头或一只手，另一边也不用它，两边的
    「上」才是同一组部位量出来的。`pt` / `ps`：两边的姿势（joint-to-world）。"""
    marks = tuple(p for p in LANDMARKS if p in m.dst and p in m.src)
    return (body_of(m.dst, target.parents, pt[:, :3, 3], marks), body_of(m.src, source.parents, ps[:, :3, 3], marks))


def body_up(rig, body: mo.Body) -> np.ndarray:
    """一副骨架在绑定姿势里（放到第一帧的世界里，和 sole_height 量脚底的同一副）身体的「上」：脚底对齐沿它量。
    `body`：对齐用的那一个（Rests.bodies），躯干是同一组关节。"""
    pose = rig.placement[0] @ np.asarray(rig.bind, np.float64)
    return body.axes(pose[:, :3, 3])[:, 0]


def t_posed(pose: np.ndarray, parents, side: dict[str, list[int]], body: mo.Body | None = None) -> np.ndarray:
    """一个姿势 [J,4,4] 按已配对部位的关节位置摆成标准 T 姿：上臂、前臂水平朝外，大腿、小腿竖直朝下，脚朝前（T_BONES，
    方向取这副骨架的身体坐标系 `body`（对齐用的那一个，Rests.bodies；没给时 body_of 这一边），面朝哪都行、没配躯干也行），锁骨放平（去掉抬起、保留前后）。每根骨头绕它的
    起点转最小的角度（不加扭转），子树整体跟着转；脊柱、头、手指保持原样。两副骨架绑定姿势差很多（A 对 T、BVH 的折叠零姿势）时，两边都摆成
    同一个 T 再对齐。"""
    pose = np.array(pose, np.float64)
    parents = np.asarray(parents)
    axes = (body or body_of(side, parents, pose[:, :3, 3])).axes(pose[:, :3, 3])
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)

    def subtree(j: int) -> list[int]:
        out, todo = [], [j]
        while todo:
            k = todo.pop()
            out.append(k)
            todo += kids.get(k, [])
        return out

    for a, b, toward in T_BONES:
        if a not in side or b not in side:
            continue
        ja, jb = side[a][0], side[b][0]
        cur = pose[jb, :3, 3] - pose[ja, :3, 3]
        if np.linalg.norm(cur) <= MEASURABLE_CM:
            continue
        if toward == "heading":  # turned to face forward, its slope kept: the same rise, the heading straight ahead
            rise = cur @ axes[:, 0]
            flat = np.linalg.norm(cur - rise * axes[:, 0])
            want = rise * axes[:, 0] + flat * axes[:, 1]
        elif toward is None:  # level: the same heading, no rise
            want = cur - (cur @ axes[:, 0]) * axes[:, 0]
            if np.linalg.norm(want) < 0.5 * np.linalg.norm(cur):
                # a clavicle mostly along up (a folded zero pose's lies along the spine): levelled it would point
                # wherever its small remainder does, so it goes out to its own side instead
                want = (1 if a.startswith("l.") else -1) * axes[:, 2]
        else:
            want = toward[1] * axes[:, toward[0]]
        turn = mo.between(cur / np.linalg.norm(cur), want / np.linalg.norm(want))
        at = pose[ja, :3, 3].copy()
        for k in subtree(ja):
            pose[k, :3, :3] = turn @ pose[k, :3, :3]
            pose[k, :3, 3] = at + turn @ (pose[k, :3, 3] - at)
    return pose


def pose_frames(pose: np.ndarray) -> np.ndarray:
    """姿势 [J,4,4] 的关节坐标系：轴正交化（去掉缩放，Skeleton prim 的单位缩放也在里面），位置照旧（cm）。修正和
    「骨架姿势」手柄的局部 / 世界变换都在这组坐标系里。"""
    out = np.array(pose, np.float64)
    out[:, :3, :3] = mo.orthonormal(out[:, :3, :3])
    return out


def corrected(pose: np.ndarray, parents, names, rows) -> tuple[np.ndarray, list[str], int]:
    """基准姿势 [J,4,4]（世界）按「初始姿势」`rows` 修正：每个写了的关节局部 = 基准局部 @ D（这一行的平移 / 旋转 /
    按轴缩放，poses.trs_matrix，关节自己的正交坐标系里），按层级正向算回世界，子关节跟着走。返回 (修正后的姿势，参数里骨架没有的关节名，修正了几个关节)；
    每个关节自己的轴长（缩放）照旧，修正里的缩放只体现在子关节的位置上。修正总是叠在原大的姿势上（位移是按原大的
    角色填的），「角色缩放」在修正之后才放大（rests）。"""
    pose = np.asarray(pose, np.float64)
    at = {n: j for j, n in enumerate(names)}
    unknown = [r["joint"] for r in rows or () if r.get("joint") not in at]
    known = [r for r in rows or () if r.get("joint") in at]
    if not known:
        return pose, unknown, 0
    frames = pose_frames(pose)
    local = mo.local_from_world(frames, parents)
    for r in known:
        d = trs_matrix(r.get("translate") or (0.0, 0.0, 0.0), r.get("rotate") or (0.0, 0.0, 0.0), r.get("scale") or 1.0)
        local[at[r["joint"]]] = local[at[r["joint"]]] @ d
    world = mo.world_from_local(local, parents)
    out = pose.copy()
    out[:, :3, 3] = world[:, :3, 3]
    out[:, :3, :3] = mo.orthonormal(world[:, :3, :3]) * mo.signed_scales(pose[:, :3, :3])[:, None, :]
    return out, unknown, len(known)


@dataclass
class Rests:
    """两边对齐用的基准姿势（「动作静止姿势」「目标静止姿势」「基准姿势摆正」），重定向和视图里编辑时画的同一份。"""

    source: mo.Skeleton
    target: mo.Skeleton
    source_said: str  # 给 I-RETARGET-DONE：动作用的是哪个姿势（含「自动挑了第 N 帧」）
    target_said: str
    fixed: bool  # 两边摆成了 T 姿
    # 「初始姿势」修正之前的两副（选的姿势、自动挑帧、摆 T 之后）：「骨架姿势」手柄的「修正前」
    source_before: np.ndarray | None = None
    target_before: np.ndarray | None = None
    unknown: tuple[list[str], list[str]] = ([], [])  # 两边「初始姿势」里骨架没有的关节名（W-RETARGET-POSEJOINT）
    posed: tuple[int, int] = (0, 0)  # 两边修正了几个关节
    bodies: tuple[mo.Body, mo.Body] | None = None  # 两边的身体坐标系（body_of：目标，动作）；只配手时没有
    picked: int = -1  # 动作自动挑的那一帧在它帧里的序号（-1 没挑）：只随骨架、对应关系和选的姿势变，调用方可以记住
    held: tuple[frozenset, frozenset] = (frozenset(), frozenset())  # 两边「初始姿势」写了的关节（序号）：以用户为准（matched_limbs）

    def said(self) -> str:
        """I-RETARGET-DONE 里说基准的那一句。"""
        fixes = i18n.separator().join(i18n.t("retarget.base.fixed_count", who=i18n.t(who), n=n)
                                      for who, n in zip(("role.motion", "role.target"), self.posed) if n)
        return (i18n.t("retarget.base.said", source=self.source_said, target=self.target_said)
                + (i18n.t("retarget.base.tposed") if self.fixed else "")
                + (i18n.t("retarget.base.fixes", fixes=fixes) if fixes else ""))


def rests(source, target, m: Mapping, motion_rest: str = "bind", motion_frame: int | None = None,
          target_rest: str = "first", target_frame: int | None = None, fix: str = "none", motion_pose=None,
          target_pose=None, size: float = 1.0, picked: int | None = None) -> Rests:
    """重定向对齐所用的两副基准姿势，一处定，计算、弹窗和「骨架姿势」手柄都调它：各自按选的姿势取（rest_pose）；动作取
    绑定姿势而它不是站姿（BVH 的折叠零姿势）时，自动在动作自己的帧里挑最像目标基准姿势的一帧（motion.closest_pose，
    和模型类节点的对齐同一条规则）；`fix` tpose 时两边再摆成标准 T 姿（t_posed）；再叠上「初始姿势」的逐关节修正
    （corrected）；最后目标按「角色缩放」`size` 放大（sized_pose，`target` 是原大的）。对齐用的是这一副，修正前的
    （原大）也带着给手柄看。`picked`：上一次挑出的帧序号（Rests.picked），给了就不再挑——只改「初始姿势」时挑帧的
    结果不变，而挑帧要把动作的每一帧比一遍。"""
    pt = rest_pose(target, target_rest, target_frame, i18n.Word("role.target"))
    ps = rest_pose(source, motion_rest, motion_frame, i18n.Word("role.motion"))
    # in both languages (messages.Both): said in I-RETARGET-DONE / I-RETARGETPREP-DONE, read in whoever's language
    source_said = Both.of(lambda: rest_said(motion_rest, motion_frame))
    tb, sb = two_bodies(m, target, pt, source, ps)
    k = -1
    if motion_rest == "bind" and not sb.stands(ps[:, :3, 3]):
        if picked is None:
            singles, _ = _parts(m)
            picked = mo.closest_pose(mo.Skeleton(target.names, target.parents, pt),
                                     mo.Skeleton(source.names, source.parents, ps), source.world(),
                                     sorted(singles.items()), mo.aims_toward(target.parents, set(singles)), (tb, sb))[1]
        k = picked
        if k >= 0:
            ps = source.world()[k]
            source_said = i18n.Word("retarget.rest.picked", frame=source.frames[k])
    if fix == "tpose":
        ps, pt = t_posed(ps, source.parents, m.src, sb), t_posed(pt, target.parents, m.dst, tb)
    fs, us, ns = corrected(ps, source.parents, joint_keys(source.names, source.parents), motion_pose)
    ft, ut, nt = corrected(pt, target.parents, joint_keys(target.names, target.parents), target_pose)
    ft = sized_pose(ft, target.parents, size)
    return Rests(mo.Skeleton(source.names, source.parents, fs), mo.Skeleton(target.names, target.parents, ft),
                 source_said, Both.of(lambda: rest_said(target_rest, target_frame)), fix == "tpose", ps, pt, (us, ut),
                 (ns, nt), (tb, sb), k, (_held(source, motion_pose), _held(target, target_pose)))


def _held(rig, rows) -> frozenset:
    """一副骨架「初始姿势」里写了的关节（序号；骨架没有的名字不算，corrected 同样跳过）。"""
    at = {n: j for j, n in enumerate(joint_keys(rig.names, rig.parents))}
    return frozenset(at[r["joint"]] for r in rows or () if r.get("joint") in at)


# 「动作重定向」保留目标自己的基准姿势时（retarget），四肢的哪几根骨头两边的基准方向差得多就先按动作对上：差不到
# LIMB_SAME 度是同一个姿势里各副骨架自己的样子（手臂略垂、腿略分），保留；超过 LIMB_POSE 度是另一个姿势（A 姿对 T 姿
# 上臂差 35–50°、手臂垂下对平举差 90°），整根对上；中间平滑过渡（smoothstep），差别跨过界限时结果不跳
LIMB_SAME, LIMB_POSE = 15.0, 30.0
MATCHED_LIMBS = ("upperarm", "forearm", "thigh", "shin")  # 只看四肢的长骨：脚的斜度、锁骨、脊柱、手指是骨架自己的样子


def matched_limbs(target: mo.Skeleton, source: mo.Skeleton, m: Mapping, turn: np.ndarray,
                  held: tuple[frozenset, frozenset] = (frozenset(), frozenset())) -> tuple[mo.Skeleton, list[str]]:
    """保留目标基准姿势的对齐（retarget）之前，四肢长骨（MATCHED_LIMBS：上臂、前臂、大腿、小腿，data/joints.py
    LIMB_BONES 的顺序，父骨在前）两边基准姿势差了一整个姿势的，目标这根骨头绕起点转到动作的方向（子树跟着转、
    最小转动不加扭转，t_posed 同样的转法）：`turn` 是目标身体到动作身体的转动（hips_turn），差的角度在动作那边量，
    转多少按 LIMB_SAME / LIMB_POSE 平滑过渡。两边基准姿势的「初始姿势」写了这根骨头起点关节的（`held`：动作、目标）
    以用户为准，不转——用户已经把两边摆成了同一个姿势，剩下的差别是有意的。返回 (对齐用的目标骨架, 转了的部位名)；
    写回没配上的关节仍用原来的基准姿势（Result.rest：整棵子树一起转，它们相对父关节的样子不变）。"""
    from ...data.joints import part_label

    pose = np.array(target.rest, np.float64)
    parents = np.asarray(target.parents)
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)
    pp = source.rest_positions
    said = []
    for a, b in LIMB_BONES:
        if a[2:] not in MATCHED_LIMBS or not all(q in m.dst and q in m.src for q in (a, b)):
            continue
        ja, jb, sa, sb = m.dst[a][0], m.dst[b][0], m.src[a][0], m.src[b][0]
        if ja in held[1] or sa in held[0]:
            continue
        cur, want = turn @ (pose[jb, :3, 3] - pose[ja, :3, 3]), pp[sb] - pp[sa]
        if min(np.linalg.norm(cur), np.linalg.norm(want)) <= MEASURABLE_CM:
            continue
        cur, want = cur / np.linalg.norm(cur), want / np.linalg.norm(want)
        angle = float(np.degrees(np.arccos(np.clip(cur @ want, -1.0, 1.0))))
        w = float(mo.smoothstep((angle - LIMB_SAME) / (LIMB_POSE - LIMB_SAME)))
        if w <= 0.0:
            continue
        axis = np.cross(cur, want)
        if np.linalg.norm(axis) < 1e-9:  # opposite directions (w > 0 rules out the same): any axis across the bone
            axis = np.cross(cur, [1.0, 0.0, 0.0] if abs(cur[0]) < 0.9 else [0.0, 1.0, 0.0])
        spin = turn.T @ mo.axis_angle(axis / np.linalg.norm(axis), np.radians(angle) * w) @ turn  # in the target's frame
        _turn_subtree(pose, kids, ja, spin)
        foot = m.dst.get(f"{a[:2]}foot", [None])[0]
        if a[2:] in ("thigh", "shin") and foot is not None:
            # the foot goes along with the leg but keeps the way it stands: a standing rest pose has its sole on the
            # ground whatever the leg's angle (the hand, which nothing holds, turns with the arm)
            _turn_subtree(pose, kids, foot, spin.T)
        said.append(Both.of(lambda a=a: part_label(a)))  # said again in whoever's language (I-RETARGET-DONE)
    return mo.Skeleton(target.names, target.parents, pose), said


def _turn_subtree(pose: np.ndarray, kids: dict[int, list[int]], joint: int, spin: np.ndarray) -> None:
    """一副姿势 [J,4,4]（世界）里 `joint` 和它的整棵子树绕 `joint` 的位置转 `spin`（原地改）。"""
    at = pose[joint, :3, 3].copy()
    todo = [joint]
    while todo:
        k = todo.pop()
        pose[k, :3, :3] = spin @ pose[k, :3, :3]
        pose[k, :3, 3] = at + spin @ (pose[k, :3, 3] - at)
        todo += kids.get(k, [])


def _parts(m: Mapping, only: tuple[str, ...] | None = None) -> tuple[dict[int, int], list]:
    """BodyRetarget.align 的一对一关节（目标 -> 源）和链状部位；`only`：只取这些部位（只配手时一只手的）。"""
    from ...data.joints import CHAIN_ENDS, CHAINS

    def end(q):
        return (m.dst[q][0], m.src[q][0]) if q and q in m.dst and q in m.src and q not in CHAINS else None

    keep = (lambda p: True) if only is None else (lambda p: p in only)
    singles = {m.dst[p][0]: m.src[p][0] for p in m.dst if p not in CHAINS and p in m.src and keep(p)}
    chains = [mo.Chain(m.dst[p], m.src.get(p, []), end(CHAIN_ENDS[p][0]), end(CHAIN_ENDS[p][1]))
              for p in CHAINS if p in m.dst and keep(p)]
    return singles, chains


def retarget(source, target, m: Mapping, by: str, factor: float, lift: bool = True, extra: float = 0.0,
             size: float = 1.0, rest: dict | None = None, base: Rests | None = None) -> Result:
    """源 rig 的动作 → 目标每个配上的关节的世界旋转和髋关节的世界位置（第二、三节的公式）。髋高比例 = 依据 `by`
    量出来的比（目标 ÷ 源逐帧的中位数）× `factor`；`by` 为 none 时就是 `factor`。`lift`：髋按固定量抬高（楼梯、梯子也对），
    否则离地高度逐帧乘比例（只适合平地）。`extra`：每帧再加的高度（cm）：两边踝离脚底的差 + 用户填的偏移。
    `size`：目标先整体放大这么多倍（「角色缩放」，sized），髋高比例按放大后的目标量。`rest`：rests 的参数（两边的
    基准姿势、摆不摆成 T 姿），空 = 节点的默认（动作绑定姿势、目标第一帧、不摆正）；`base`：已经按它们取好的两副基准
    姿势（rests，调用方先取来提示、量脚底时，对齐就用同一份）。只配手（`m.hands`）时见 retarget_hands（不缩放）。"""
    if m.hands:
        return retarget_hands(source, target, m, by, factor, rest)
    base = base or rests(source, target, m, **(rest or {}), size=size)
    target = sized(target, size)
    singles, chains = _parts(m)
    source_world, target_world = source.world(), target.world()
    driven = set(singles) | {t for ch in chains for t in ch.target}
    skel_t, skel_s = base.target, base.source
    aims = mo.aims_toward(target.parents, driven)
    # the clavicle turns as the chest does, from its base pose, and is not aimed along its bone: the rigs put it
    # differently (SAM 3D Body's from the sternum forwards and up, AccuRIG's from behind the neck outwards), and copying
    # its direction turns AccuRIG's clavicle 36–40° off its own rest against the chest where the source's is 7° off
    # its own: raised shoulders. That holds between two standing base poses only: a folded one (a BVH zero pose as the
    # target's first frame) has its clavicle along the spine, and following the chest from there keeps it pointing up
    # (92° off, the shoulder 19 cm out of place) — the clavicle is then aimed like any limb bone
    both_stand = base.bodies[0].stands(skel_t.rest_positions) and base.bodies[1].stands(skel_s.rest_positions)
    for s in ("l", "r"):
        if f"{s}.clavicle" in m.dst and both_stand:
            aims[m.dst[f"{s}.clavicle"][0]] = None
    # the hips fork (legs and spine): turned by the two base poses' own axes (motion.hips_turn), with or without an
    # upper body mapped — a picked frame or a first frame may face any way, a rig may lie any way. Levelled (only the
    # way the body faces) between two standing base poses the user chose; a frame picked from the take holds the take's
    # own lean, which stays (as Retarget.align): levelled, LAFAN dance → AccuRIG's man leans 3.6° more than the source
    # over the take, 1.3° kept (walk → woman 4.3° / 1.6°)
    fixed = {m.dst["hips"][0]: mo.hips_turn(skel_t.rest_positions, skel_s.rest_positions, *base.bodies,
                                                   level=both_stand and base.picked < 0)}
    refs = mo.body_refs(skel_t, skel_s, base.bodies)
    trunk = next((p for p in ("neck", "head") if p in m.dst and p in m.src), None)
    if "chest" in m.dst and "chest" in m.src and trunk:
        # the chest forks (the neck and both clavicles), so aims_toward gives it no bone; following the spine's
        # alignment, two spines of different rest shapes (SOMA and the standard human differ 36° at Spine1→Spine2)
        # would turn the chest, and the neck's root and the shoulders with it (chest→neck 44° off). It aims along the
        # trunk's branch: at the neck's first joint (a chain's first joints correspond in both skeletons), else the head
        pair = ((m.dst["chest"][0], m.src["chest"][0]), (m.dst[trunk][0], m.src[trunk][0]))
        turn = mo.bone_turn(skel_t, skel_s, *pair, refs)
        if turn is not None:
            fixed = {**(fixed or {}), pair[0][0]: turn}
    # two standing base poses the user chose (not a frame picked for a folded source): the target keeps its own base
    # pose — with the source in its base pose the target is in its own (「初始姿势」 corrections included), every joint
    # turns from there as the source's turns from its own (BodyRetarget keep: one turn, the two bodies' axes). A limb
    # bone whose two rest directions differ by a whole pose (A against T, arms down against out) is matched onto the
    # source's first (matched_limbs). A picked frame (an arbitrary frame of the take) or a folded base pose holds no
    # rest of its own to keep: aligned bone by bone as before, each bone onto the source's direction
    keep, matched = None, []
    if both_stand and base.picked < 0:
        keep = fixed[m.dst["hips"][0]]
        skel_t, matched = matched_limbs(skel_t, skel_s, m, keep, base.held)
    fit = mo.BodyRetarget.align(skel_t, skel_s, singles, chains, aims, refs, target_world[0, :, :3, 3],
                                source_world[0, :, :3, 3], None if keep is not None else fixed, keep)
    rotations = fit.rotations(source_world)
    def aligned() -> str:  # in the language now; said in both (messages.Both) with base.said() below
        if keep is None:
            return i18n.t("retarget.aligned.bones", why=i18n.t("retarget.aligned.picked" if base.picked >= 0 else "retarget.aligned.folded"))
        return i18n.t("retarget.aligned.kept") + (i18n.t("retarget.aligned.limbs", n=len(matched), names=i18n.separator().join(str(x) for x in matched))
                                                  if matched else "")

    positions = source_world[..., :3, 3]
    measured = measure(by, m, "src", positions)
    target_cm = float(measure(by, m, "dst", target_world[:1, :, :3, 3])[0])
    median = float(np.median(measured))
    if by != "none" and min(median, target_cm) <= MEASURABLE_CM:  # a degenerate skeleton: no ratio (inf / 0 otherwise)
        raise Invalid(Msg("E-RETARGET-NOLENGTH", what=SCALE_BY[by], rig=i18n.Word("role.motion" if median <= MEASURABLE_CM else "role.target")))
    s = (target_cm / median if by != "none" else 1.0) * float(factor)
    # the lift (跟着脚抬高) is measured from the same joints as the sole alignment (animation.py sole_height): ground_of
    # heights along the source body's own up over the frames it stands (Body.up_over), not the world's Y
    path = mo.hips_path(positions, (m.src["l.thigh"][0], m.src["r.thigh"][0]), ground_of(source, m.src)[0], s, lift,
                        extra, base.bodies[1].up_over(positions))
    hips_t = m.dst["hips"][0]
    if hips_t not in rotations:  # 髋总是一对一的部位，这里只防御
        raise Invalid(Msg("E-RETARGET-MISSINGPARTS", parts=[_part_label("hips")], rig=i18n.Word("role.target"), why=WHY["legs"]))
    # the hip point goes on the path once every rotation is set (target_locals → motion.set_hips): where the thighs
    # hang comes from the hierarchy, not from an offset fixed on the hips joint
    return Result(rotations, {}, s, by, float(factor),
                  (float(measured.min()), median, float(measured.max())), target_cm, len(rotations), float(size),
                  Both.of(lambda: base.said() + aligned()), np.linalg.inv(target.placement[0]) @ base.target.rest, base.unknown,
                  hips=(hips_t, (m.dst["l.thigh"][0], m.dst["r.thigh"][0]), path))


def retarget_hands(source, target, m: Mapping, by: str, factor: float, rest: dict | None = None) -> Result:
    """只配手（HaMeR 的 MANO 手、只有手的绑定）：每只手单独对齐——手腕按两只手各自的轴（沿手指、手掌法向）对上，
    手指链照常按骨长分摊弯曲，扭转参考换成手掌法向（身体的「前方、上方」对一只随便摆着的手没有意义）。手腕放到哪
    由目标的形态定（hand_on_body）：目标只有手时手腕 1 : 1 跟着动作（位置和朝向）；目标的手长在身体上（整个身体的
    角色配一只 MANO 手）时，手腕和手臂保持目标自己的基准姿势，只传手指相对手腕的弯曲——一只单独的手的世界位置和
    朝向对一条手臂没有意义，写上去手就离开前臂。没有髋，髋高比例不适用（`by` / `factor` 只原样记下）。基准是两边
    的绑定姿势（hand_bases），`rest` 里的「初始姿势」照样叠上去（其余基准姿势参数只管整个身体）。"""
    source_world, target_world = source.world(), target.world()
    base = hand_bases(source, target, rest)
    skel_t, skel_s = base.target, base.source
    rotations: dict[int, np.ndarray] = {}
    placed: dict[int, np.ndarray] = {}
    for side in m.hands:
        parts = (f"{side}.hand", *(f"{side}.{f}" for f in FINGERS))
        singles, chains = _parts(m, parts)
        driven = set(singles) | {t for ch in chains for t in ch.target}

        def pair(part: str) -> tuple[int, int]:
            return m.dst[part][0], m.src[part][0]

        refs, fixed = mo.hand_refs(skel_t, skel_s, pair(f"{side}.hand"), pair(f"{side}.middle"), pair(f"{side}.index"),
                                   pair(f"{side}.pinky"))
        fit = mo.BodyRetarget.align(skel_t, skel_s, singles, chains, mo.aims_toward(target.parents, driven), refs,
                                    target_world[0, :, :3, 3], source_world[0, :, :3, 3], fixed)
        got = fit.rotations(source_world)
        wrist_t, wrist_s = pair(f"{side}.hand")
        if hand_on_body(m, side, target.parents):
            # the fingers as they bend against the wrist, the wrist where the target's own base pose holds it
            held = mo.orthonormal(skel_t.rest[wrist_t, :3, :3])
            back = np.swapaxes(got.pop(wrist_t), -1, -2)
            got = {j: held @ (back @ r) for j, r in got.items()}
        else:
            placed[wrist_t] = source_world[:, wrist_s, :3, 3]
        rotations.update(got)
    return Result(rotations, placed, 1.0, by, float(factor), (0.0, 0.0, 0.0), 0.0, len(rotations),
                  rest_said=Both.of(base.said), rest=np.linalg.inv(target.placement[0]) @ skel_t.rest, unknown=base.unknown,
                  wrist_kept=any(hand_on_body(m, s, target.parents) for s in m.hands))


def hand_on_body(m: Mapping, side: str, parents) -> bool:
    """目标的这只手长在身体上：对应关系或名字说目标是身体（has_body）、这一边配了前臂，或层级上它只是整副骨架的一小部分
    （hangs_on_body：名字认不出、只配了手的整个人）。只配手时手腕不能照抄一只单独的手的世界位置（retarget_hands）。"""
    return m.bodies[1] or f"{side}.forearm" in m.dst or hangs_on_body(parents, m.dst[f"{side}.hand"][0])


def hand_bases(source, target, rest: dict | None = None) -> Rests:
    """只配手时两边的基准姿势：绑定姿势，叠上「初始姿势」（rests 的另一半规则都是给整个身体的：挑站姿的帧、摆 T 姿）。"""
    rest = rest or {}
    ps, pt = source.placement[0] @ np.asarray(source.bind, np.float64), target.placement[0] @ np.asarray(target.bind, np.float64)
    fs, us, ns = corrected(ps, source.parents, joint_keys(source.names, source.parents), rest.get("motion_pose"))
    ft, ut, nt = corrected(pt, target.parents, joint_keys(target.names, target.parents), rest.get("target_pose"))
    return Rests(mo.Skeleton(source.names, source.parents, fs), mo.Skeleton(target.names, target.parents, ft),
                 REST_LABEL["bind"], REST_LABEL["bind"], False, ps, pt, (us, ut), (ns, nt))


def target_locals(target, frames, placement: np.ndarray, r: Result) -> np.ndarray:
    """目标 rig 在 `frames` 上的 joint-to-parent [F,J,4,4]，写的是 retarget 的结果 `r`。

    起点是目标的参考帧（第一帧）：每个关节的局部平移和缩放取参考帧的，局部旋转取对齐用的基准姿势的（Result.rest；
    没给时是绑定姿势。没配上的关节于是跟着父关节、保持基准姿势的朝向）；配上的关节转到给定的世界旋转、保留自己的
    缩放（motion.set_world）；整个身体再按髋点放好（motion.set_hips：两大腿中点放到 Result.hips 的路径上，大腿挂在
    哪由层级算），只配手时手腕放到 Result.placed。结果里的世界量先换回骨架自己的空间再写局部（`placement` [F,4,4]
    是目标 Skeleton prim 在这些帧上的摆放）——根关节的局部是相对 Skeleton prim 的，直接写世界量在 prim 自己带变换时
    （导入的单位缩放、上游「3D 变换」）位置就错了。"""
    parents = np.asarray(target.parents, np.int64)
    ref = np.array(target.local[0], np.float64)
    scale = np.linalg.norm(ref[:, :3, :3], axis=-2)  # 每列长度 = 每个轴的缩放
    bind_local = mo.local_from_world(np.asarray(target.bind if r.rest is None else r.rest, np.float64), parents)
    ref[:, :3, :3] = mo.orthogonal(bind_local[:, :3, :3]) * scale[:, None, :]  # a mirrored joint stays mirrored
    local = np.broadcast_to(ref, (len(frames), *ref.shape)).copy()
    inv = np.linalg.inv(placement)
    # a world rotation into the skeleton's space through the prim's whole linear part, a mirror included: the joint's
    # axes in the world are its rotation with x turned by its handedness there (the prim's times its own in the
    # skeleton's space, motion.orthonormal's convention), taken back by the prim's inverse; set_world puts the joint's
    # own mirror back. Turning by the prim's rotation alone drops a mirrored prim's mirror (130–170 cm off)
    own = mo.handedness(mo.world_from_local(local[0], parents)[:, :3, :3])
    prim = mo.handedness(placement[:, :3, :3])

    def into(p):  # a world position [F,3] into the skeleton's space
        return (inv[:, :3, :3] @ np.asarray(p, np.float64)[..., None])[..., 0] + inv[:, :3, 3]

    def turned(j: int, q: np.ndarray) -> np.ndarray:
        sign = prim * own[j]
        axes = np.asarray(q, np.float64) * np.stack([sign, np.ones_like(sign), np.ones_like(sign)], -1)[:, None, :]
        return mo.orthonormal(inv[:, :3, :3] @ axes)

    rot = {j: turned(j, q) for j, q in r.rotations.items()}
    if r.hips is not None:
        move, thighs, at = r.hips
        return mo.set_hips(local, parents, rot, move, thighs, into(at))
    return mo.set_world(local, parents, rot, {j: into(p) for j, p in r.placed.items()})


def skin_locals(anim, character, order: list[int], placement: np.ndarray) -> np.ndarray:
    """「线性蒙皮变形」：骨架动画的局部矩阵按角色的关节顺序排好 [F,J,4,4]。根关节换参考系：骨架动画的 Skeleton prim
    可能被「3D 变换」挪过，角色自己的 prim 保持不动，两者的差值折进根关节（`placement` 是角色的 prim 在动画各帧的
    摆放）。"""
    local = np.array(anim.local[:, order], np.float64)
    parents = np.asarray(character.parents, np.int64)
    fix = np.linalg.inv(placement) @ anim.placement  # [F,4,4]
    for j in np.flatnonzero(parents < 0):
        local[:, j] = fix @ local[:, j]
    return local


STILL_CM = 1e-3  # 局部平移逐帧变化不超过这么多（cm）就是一根固定长度的骨头


def lengths_differ(anim, character, order: list[int], tolerance: float = 0.05) -> list[str]:
    """骨长对不上的关节（动画第一帧的局部平移长度和角色参考帧的相差超过 5%）：照挂，只提示（W-SKIN-LENGTHS）。

    只比骨头：根关节（它的平移是整副骨架的摆放）和动画里局部平移逐帧在变的关节（那是位移，不是骨长）不比。
    按层级和动画判断、不看名字：「动作重定向」把位移写在髋上，髋不一定是根（AccuRIG / CC 的髋挂在不动的
    RL_BoneRoot 下面），DCC 里 K 了位移的关节也一样；只排除根关节时这类骨架每次都报髋「骨长不一致」。
    不带动任何顶点的关节也不比（_drives_mesh）：它的长度改不了网格上的任何东西（例如 Kimodo SOMA 的
    LeftHandThumbEnd / RightHandThumbEnd 在标准人上没有蒙皮权重、也没有子关节，每次默认都报）。"""
    moved = np.asarray(anim.local)[:, order][..., :3, 3]  # [F,J,3]，按角色的关节顺序
    a = np.linalg.norm(moved[0], axis=-1)
    b = np.linalg.norm(character.local[0][:, :3, 3], axis=-1)
    parents = np.asarray(character.parents)
    bone = (parents >= 0) & (np.ptp(moved, axis=0).max(axis=-1) <= STILL_CM) & _drives_mesh(character)
    far = (np.abs(a - b) > tolerance * np.maximum(b, 1e-6)) & (np.maximum(a, b) > 1e-3) & bone
    return [character.names[j] for j in np.flatnonzero(far)]


def _drives_mesh(character) -> np.ndarray:
    """Per joint: whether it moves some vertex of the character's mesh, itself (a skin weight above zero:
    Rig.weighted) or through a joint under it. Every joint when nothing says which deform (no mesh bound)."""
    parents = np.asarray(character.parents)
    weighted = getattr(character, "weighted", None)
    if weighted is None:
        return np.ones(len(parents), bool)
    drives = np.zeros(len(parents), bool)
    drives[[j for j in weighted if 0 <= j < len(parents)]] = True
    for j in sorted(range(len(parents)), key=lambda j: -_depth(parents, j)):  # children before their parents
        if drives[j] and parents[j] >= 0:
            drives[parents[j]] = True
    return drives


def _depth(parents: np.ndarray, j: int) -> int:
    d = 0
    while parents[j] >= 0:
        j, d = int(parents[j]), d + 1
    return d


def _part_label(part: str) -> str:
    from ...data.joints import part_label

    return part_label(part)
