"""两副骨架之间按身体部位的对应、两侧参考姿态的修正、忽略哪些关节：参数的声明、每行的格式和视图要的数据。

「重定向预处理」（core/retarget.py）、「动作重定向」（core/scene.py Retarget：两副使用者的骨架）和骨骼动作家族的四个
模型类节点（kit/rig.py RigModel：人物骨骼 → 模型自己的骨架，Kimodo / StableMotion / Two-stage Transformer /
UnderPressure）共用同一种参数和同一个视图编辑模式（nodes/handles.py RigPair，「rig_pair」手柄）。每行的约定、推测和
校验在 data/joints.py；这里只有节点声明和手柄要的数据（NodeDef.handle_data：rig_pair_data）。单独成一个模块，是因为
core/ 的节点也要用它，而 kit/rig.py 依赖 families（Job），从核心节点引它会循环导入。
"""

from __future__ import annotations

from ..base import NodeParams, P


class PartMap(NodeParams):
    """「对应关系」的一行：一个身体部位，驱动它的关节（src）和被驱动的关节（dst），都用关节名（约定见
    data/joints.py「a mapping as a node parameter」）。字段名不用 from / to：from 是 Python 关键字，pydantic 的别名
    在参数写出（model_dump）时不生效，存进节点图的会变成 from_，所以直接用 src / dst。"""

    part: str = P(..., widget="fixed")
    src: list[str] = P([])
    dst: list[str] = P([])


def rig_map_param(choices_from: tuple[str, ...], follows: tuple[str, ...] = (), group: str = "people"):
    """「对应关系」参数：值是 PartMap 的列表，空 = 全部按名字和层级推测；列表里没有的部位也按推测。在视图里编辑
    （「rig_pair」手柄，widget "rig_map"：面板上只有一行摘要和「在视图里编辑」）；两副骨架、部位表和推测结果由节点的
    handle_data 给出（rig_pair_data）。`choices_from` 是它依赖的输入口和参数；`follows`：换了就要重建模型一侧的参数
    （Kimodo 的「模型」，derive）。"""
    return P(None, widget="rig_map", group=group, choices_from=choices_from, worker=False,
             derived_from=follows)


class JointPose(NodeParams):
    """「动作初始姿势」「目标初始姿势」的一行：一个关节在基准姿势上的修正（kit/retarget.py corrected）。修正后的局部 =
    基准姿势的局部 @ 平移 · 旋转 · 缩放（顺序和旋转约定同「3D 变换」：先缩放、再按 X、Y、Z 转、再平移），都在这个
    关节自己的坐标系里（正交化后的轴，cm），子关节按层级跟着走。没修正的关节不写。"""

    joint: str = P(...)
    translate: list[float] = P([0.0, 0.0, 0.0], unit="cm")
    rotate: list[float] = P([0.0, 0.0, 0.0], unit="°")
    scale: list[float] = P([1.0, 1.0, 1.0], unit="x")


def pose_param(label: str = "", group: str = "skeleton"):
    """「初始姿势」参数：每个修正过的关节一行（JointPose），在视图里用手柄编辑（「rig_pair」手柄的 src_pose /
    dst_pose，widget "skeleton_pose"：面板上只有一句「修正了 N 个关节」和清空）。空 = 不修正。"""
    return P([], widget="skeleton_pose", group=group, worker=False)


def ignore_param(label: str = "", group: str = "skeleton"):
    """「忽略的骨骼」参数：不送进解算器的关节名（data/joints.py joint_keys），展开后的完整名单（选一个关节连同它下面的
    一起选，写进来的是每一个）。在视图里点选（「rig_pair」手柄的 src_ignore / dst_ignore，widget "joint_list"）。"""
    return P([], widget="joint_list", group=group, worker=False)


def scale_param(label: str = "", group: str = "skeleton"):
    """「动作尺寸」「目标尺寸」：这一侧送进解算器之前整体缩放的系数（kit/retarget_prepare.py scale_rig），1 = 不缩放；
    留空 = 自动（auto_scale：腿长不在人体尺寸范围时缩到标准人体）。「rig_pair」手柄的 src_scale / dst_scale：视图里这一侧
    按它画，功能条上有「自动尺寸」。"""
    return P(None, group=group, gt=0.0, le=1000.0, unit="x", worker=False)


def rule_param(group: str = "skeleton"):
    """「忽略规则」：「自动忽略」按哪条规则建议（kit/retarget_needs.py all_rules：通用，加每个解算器声明的）。名单
    执行过或改过之后，结果只认名单本身；还没动过时计算按这条规则的建议算，它经节点的 fingerprint_params 进指纹。
    选项只看装了哪些解算器，不依赖任何输入（choices_from 为空：页面不等上游算完就能写出规则的名字）。"""
    return P("generic", widget="choice", group=group, choices_from=(), worker=False,
             affects_result=False)


def auto_record_param():
    """几个「自动」（对应、姿态、忽略、尺寸）最近一次执行的快照（JSON：{"mapping": 行, "pose": {"motion": 行, "target": 行},
    "ignore": {"rule": id, "motion": [名], "target": [名]}, "size": {"motion": 系数, "target": 系数}}）：页面拿它和当前参数比，写「已自动 · 之后手动改了 N 个」
    这类状态行。不在面板里、不进 worker；快照本身不影响结果，但「哪个自动执行过」决定默认自动是否生效，经节点的
    fingerprint_params 进指纹（core/retarget.py RetargetPrepare）。"""
    return P("", widget="hidden_json", group="skeleton", panel=False, worker=False, affects_result=False)


def skeleton_choice(packet, param: str) -> dict:
    """「骨骼」下拉的选项：场景里的每副骨架，空 = 第一个。"""
    from ...data.animation import skeletons

    found = skeletons(packet)
    return {param: {"options": [s["path"] for s in found], "auto": found[0]["path"] if found else ""}}


def skeleton_handle(packet: str, path: str, names, parents, before, parts: dict[str, list[int]],
                    pose: str = "", unknown=(), body=None) -> dict:
    """一副骨架在视图里要画的（「rig_pair」手柄的一侧，rig_pair_side；nodes/handles.py Poses 的也是这一份）：蒙皮预览要的包指纹与骨架路径、关节名、父子、姿势叫什么，
    修正前的基准姿势（`before`，joint-to-world [J,4,4]）只发局部，左右镜像的关节对和对称面，参数里骨架没有的关节名。
    世界和修正后的都由舞台自己按层级算：世界 = 正向累乘局部，修正后 = 局部 @ D（kit/retarget.py corrected 的同一个
    公式，参数里的行就在舞台手上）——四份矩阵只发一份，大骨架也只有几十 KB。`body`：对齐用的身体（Rests.bodies），
    对称面按它；没给时按这一边自己（只读手柄、对应关系还不成时）。"""
    return {"packet": packet, "path": path, "names": list(names), "parents": [int(p) for p in parents], "pose": pose,
            "before": {"local": _local_matrices(before, parents)},
            "mirror": _mirror_pairs(parts), "mirror_plane": _mirror_plane(before, parts, parents, body), "unknown": list(unknown),
            "units": {"translate": "cm", "rotate": "°"}, "rotation": "XYZ", "order": "local @ T·R·S"}


def rig_pair_side(packet: str, path: str, names, parents, before, parts: dict[str, list[int]], pose: str = "",
                  unknown=(), body=None, auto_pose: list[dict] | None = None, recognition: dict | None = None,
                  size: dict | None = None) -> dict:
    """「rig_pair」手柄的一侧（有位置的骨架）：skeleton_handle 的全部字段，加这一侧的部位 -> 关节序号（参数行 + 推测
    合并后，同 side_parts）、「自动姿态」的建议行（只有带姿态角色的节点给，kit/retarget_prepare.py auto_pose）、
    识别引擎对这副骨架的判断（data/skeleton_recognition.py Recognition.report：每个部位的置信度与依据、没分配的部位、
    身体坐标系；只是给页面看的，页面不靠它做任何识别），「自动尺寸」的建议（只有带尺寸角色的节点给，`size`：
    {"auto": 系数, "leg_cm": 原尺寸的腿长, "ground": 这一侧地面的世界高度}，kit/retarget_prepare.py auto_scale、
    ground_level；系数不是 1 时世界里 p ↦ 系数·(p − (0, ground, 0))，页面按实际用的系数这样画这一侧），和
    fixed = False。"""
    out = skeleton_handle(packet, path, names, parents, before, parts, pose, unknown, body)
    out.update(parts={p: [int(j) for j in js] for p, js in parts.items()}, fixed=False)
    if auto_pose is not None:
        out["auto_pose"] = auto_pose
    if recognition is not None:
        out["recognition"] = recognition
    if size is not None:
        out["size"] = size
    return out


def fixed_side(names: list[str], parents, parts: dict[str, list[int]]) -> dict:
    """「rig_pair」手柄的模型一侧（RigPair dst=None：模型自己的固定骨架）：只有关节名、父子和部位，没有位置，视图里
    只画树。"""
    return {"fixed": True, "names": list(names), "parents": [int(p) for p in parents],
            "parts": {p: [int(j) for j in js] for p, js in parts.items()}}


def rig_pair_data(src: dict, dst: dict, required: tuple[str, ...], auto_mapping: list[dict], rules: list[dict] | None = None,
                  default_rule: str = "") -> dict:
    """「rig_pair」手柄的全部数据（NodeDef.handle_data 里这个手柄的那一项）：两侧（rig_pair_side / fixed_side），部位表
    （data/joints.py part_rows：id、中文名、区域、是不是链、是不是必需），推测的对应（参数格式），和带忽略角色的节点
    才有的忽略规则（每条规则两侧的建议名单、声明它的解算器节点类型 id，通用规则 ""）与页面预选的规则。"""
    from ...data.joints import part_rows

    out = {"src": src, "dst": dst, "parts_table": part_rows(required), "auto_mapping": auto_mapping}
    if rules is not None:
        out.update(rules=rules, default_rule=default_rule)
    return out


def side_parts(names, parents, rows, col: str, rest=None, weights=None) -> dict[str, list[int]]:
    """一副骨架的部位 -> 关节序号，给手柄的镜像对和对称面：和计算读对应关系同一套（data/joints.py merged_rows +
    part_joints）——参数里写了的行为准，留空的那一列就是「这一边不配」，没写的部位按推测。`col`：src / dst；`names` 是
    关节自己的名字，行按 joint_keys 指关节（重名的按路径）。`rest`、`weights`：见 data/joints.py guess。"""
    from ...data.joints import auto_rows, merged_rows, part_joints, part_names

    guessed = part_names(list(names), parents, rest, weights)  # rest：识别引擎按位置、对称一起判断（data/skeleton_recognition.py）
    merged, _ = merged_rows([r for r in rows or () if isinstance(r, dict) and "part" in r], auto_rows(guessed, guessed))
    try:
        return part_joints(merged, list(names), parents, col)
    except KeyError:  # a row naming a joint this skeleton lacks (the mapping check says so on the cook): the guess
        return part_joints(auto_rows(guessed, guessed), list(names), parents, col)


def _local_matrices(pose, parents) -> list[list[float]]:
    """一副姿势在正交坐标系（kit/retarget.py pose_frames）下的局部（根关节的就是世界），每个 16 个数、按列
    （THREE.Matrix4.fromArray 的顺序）。"""
    from lab2shot_shared import motion as mo

    from .retarget import pose_frames

    local = mo.local_from_world(pose_frames(pose), parents)
    return [[round(float(v), 6) for v in m.T.ravel()] for m in local]


def _mirror_pairs(parts: dict[str, list[int]]) -> list[list[int]]:
    """左右镜像的关节对：同一个部位左右两边（l.X / r.X）的第 k 个关节。"""
    out = []
    for part, left in parts.items():
        if part.startswith("l.") and (right := parts.get("r." + part[2:])):
            out += [[int(a), int(b)] for a, b in zip(left, right)]
    return out


def _mirror_plane(pose, parts: dict[str, list[int]], parents, body=None) -> dict | None:
    """镜像按的平面（世界）：法向是身体坐标系的「左」（kit/retarget.py body_of，和对齐用的同一个），过两大腿中点。
    各家骨架左右关节的轴一般不对称，镜像只能在世界里按这个面反射，不能在局部里翻轴的正负。缺大腿或小腿时 None。"""
    import numpy as np

    from .retarget import body_of

    if not all(parts.get(p) for p in ("l.thigh", "r.thigh", "l.shin", "r.shin")):
        return None
    pos = np.asarray(pose, np.float64)[:, :3, 3]
    body = body or body_of(parts, parents, pos)
    axes = body.axes(pos)
    return {"normal": [round(float(v), 6) for v in axes[:, 2]],
            "point": [round(float(v), 4) for v in (pos[body.thighs[0]] + pos[body.thighs[1]]) / 2]}


def expression_map_param(choices_from: tuple[str, ...], group: str = "expression"):
    """「表情重定向（ARKit52）」的「对应关系」：和身体的同一种行（PartMap：部位 = 表情槽，src 一条曲线，dst 一个形变），
    在面板上是一张表（widget "expression_map"：每个表情槽一行，曲线 → 形变，下拉选；选项由 expression_choice 给）。
    空 = 全部按名字推测。"""
    return P(None, widget="expression_map", group=group, choices_from=choices_from, worker=False)


def expression_choice(curves: list[str], shapes: list[str], source: str, target: str) -> dict:
    """「表情重定向（ARKit52）」的「对应关系」表格要的数据（NodeDef.choices 里 mapping 那一项）：每个表情槽一行
    （data/expressions.py slot_rows：每条源曲线一个槽，ARKit 的按 ARKit 名、按区域排），这一行可选的曲线（这个槽的）
    和形变（目标的全部），推测的那一对（auto：参数格式的行，ARKit 名、常见别名按名字认）。`source` / `target`：两侧
    所在的路径，只用来显示。两侧有一侧是空的只给一句说明。"""
    from ...data.expressions import auto_rows, slot, slot_rows

    if not curves or not shapes:
        from ... import i18n

        return {"options": [], "empty": i18n.t("expressions.no_curves" if not curves else "expressions.no_shapes")}
    auto = {r["part"]: r for r in auto_rows(curves, shapes)}
    rows = [{**row, "curves": [c for c in curves if slot(c) == row["id"]], "auto": auto.get(row["id"])}
            for row in slot_rows(curves)]
    return {"options": [], "rows": rows, "shapes": list(shapes), "source": source, "target": target}
