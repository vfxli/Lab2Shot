"""「对应关系」参数（两副骨架之间按身体部位的对应）：参数的声明、每行的格式和编辑器要的数据。

「动作重定向」（core/scene.py Retarget：两副使用者的骨架）和骨骼动作家族的四个模型类节点（kit/rig.py RigModel：
人物骨骼 → 模型自己的骨架，Kimodo / StableMotion / Two-stage Transformer / UnderPressure）共用同一种参数和同一个
编辑器（webui/src/editor/RigMap.tsx，widget "rig_map"）。每行的约定、推测和校验在 data/joints.py；这里只有
节点声明、NodeDef.choices 和「骨架姿势」手柄（NodeDef.handle_data）要用的几样。单独成一个模块，是因为 core/scene.py 也要用它，而 kit/rig.py 依赖
families（Job），从核心节点引它会循环导入。
"""

from __future__ import annotations

from ..base import NodeParams, P


class PartMap(NodeParams):
    """「对应关系」的一行：一个身体部位，驱动它的关节（src）和被驱动的关节（dst），都用关节名（约定见
    data/joints.py「a mapping as a node parameter」）。字段名不用 from / to：from 是 Python 关键字，pydantic 的别名
    在参数写出（model_dump）时不生效，存进节点图的会变成 from_，所以直接用 src / dst。"""

    part: str = P(..., label="部位", widget="fixed")
    src: list[str] = P([], label="驱动")
    dst: list[str] = P([], label="被驱动")


def rig_map_param(choices_from: tuple[str, ...], follows: tuple[str, ...] = (), group: str = "人物"):
    """「对应关系」参数：HumanIK 式的人形部位槽编辑器（widget "rig_map"，webui/src/editor/RigMap.tsx，装在「弹窗编辑
    参数」框架里）。值是 PartMap 的列表，空 = 全部按名字和层级推测；列表里没有的部位也按推测。编辑器要的数据
    （两副骨架的关节、父子，部位表，推测结果）由节点的 choices 给出（rig_map_choice；两副骨架的 3D 走「骨架姿势」手柄），
    `choices_from` 是它依赖的输入口和参数。`follows`：换了就要重建模型一侧的参数（Kimodo 的「模型」，derive）。"""
    return P(None, label="对应关系", widget="rig_map", group=group, choices_from=choices_from, worker=False,
             derived_from=follows, placeholder="自动")


class JointPose(NodeParams):
    """「动作初始姿势」「目标初始姿势」的一行：一个关节在基准姿势上的修正（kit/retarget.py corrected）。修正后的局部 =
    基准姿势的局部 @ 平移 · 旋转 · 缩放（顺序和旋转约定同「3D 变换」：先缩放、再按 X、Y、Z 转、再平移），都在这个
    关节自己的坐标系里（正交化后的轴，cm），子关节按层级跟着走。没修正的关节不写。"""

    joint: str = P(..., label="关节")
    translate: list[float] = P([0.0, 0.0, 0.0], label="位移", unit="cm")
    rotate: list[float] = P([0.0, 0.0, 0.0], label="旋转", unit="°")
    scale: list[float] = P([1.0, 1.0, 1.0], label="缩放", unit="倍")


def pose_param(label: str, side: str, group: str = "骨架"):
    """「初始姿势」参数：每个修正过的关节一行（JointPose），在 3D 舞台上用「骨架姿势」手柄编辑（nodes/handles.py
    Poses，widget "skeleton_pose"：面板上只有一句「修正了 N 个关节」和清空）。空 = 不修正。"""
    return P([], label=label, widget="skeleton_pose", group=group, worker=False,
             help=f"在 3D 视图里点{side}骨架的关节，用旋转 / 位移 / 缩放手柄把基准姿势摆对（比如 A 姿的手臂抬平），"
                  "对齐按修正后的姿势算；子关节跟着走，左右可以镜像")


def skeleton_choice(packet, param: str) -> dict:
    """「骨骼」下拉的选项：场景里的每副骨架，空 = 第一个。"""
    from ...data.animation import skeletons

    found = skeletons(packet)
    return {param: {"options": [s["path"] for s in found], "auto": found[0]["path"] if found else ""}}


def rig_side(packet, path: str | None, label: str, handle: int | None = None) -> dict:
    """编辑器里可点选的一侧：骨架路径、关节名、父子，和画它的「骨架姿势」手柄在节点 handles 里的序号（pose_handle）；
    所选的骨骼不在场景里时是 {}。不带姿势：两副骨架画在舞台上，数据走手柄（NodeDef.handle_data → skeleton_handle），
    弹窗只管对应关系。"""
    from ...data.animation import skeletons

    found = skeletons(packet)
    one = next((s for s in found if s["path"] == path), None) if path else (found[0] if found else None)
    if not one:
        return {}
    from ...data.joints import joint_keys

    parents = [int(p) for p in one["parents"]]
    # "names": how rows and the editor name each joint (joint_keys: the path where names repeat); "joints": the
    # joints' own names, which the part guess reads — a path's words would add every ancestor's to the guess
    return {"label": label, "fixed": False, "path": one["path"], "names": joint_keys(one["joints"], parents),
            "joints": list(one["joints"]), "parents": parents, "handle": handle}


def pose_handle(node, port: str) -> int | None:
    """节点里画 `port` 这个输入的骨架的「骨架姿势」手柄的序号（nodes/handles.py Poses），没有时 None。"""
    return next((k for k, h in enumerate(node.handles) if h.kind == "skeleton_pose" and h.source == port), None)


def skeleton_handle(packet: str, path: str, names, parents, before, parts: dict[str, list[int]],
                    pose: str = "", unknown=(), body=None) -> dict:
    """一个「骨架姿势」手柄要画的（nodes/handles.py Poses）：蒙皮预览要的包指纹与骨架路径、关节名、父子、姿势叫什么，
    修正前的基准姿势（`before`，joint-to-world [J,4,4]）只发局部，左右镜像的关节对和对称面，参数里骨架没有的关节名。
    世界和修正后的都由舞台自己按层级算：世界 = 正向累乘局部，修正后 = 局部 @ D（kit/retarget.py corrected 的同一个
    公式，参数里的行就在舞台手上）——四份矩阵只发一份，大骨架也只有几十 KB。`body`：对齐用的身体（Rests.bodies），
    对称面按它；没给时按这一边自己（只读手柄、对应关系还不成时）。"""
    return {"packet": packet, "path": path, "names": list(names), "parents": [int(p) for p in parents], "pose": pose,
            "before": {"local": _local_matrices(before, parents)},
            "mirror": _mirror_pairs(parts), "mirror_plane": _mirror_plane(before, parts, parents, body), "unknown": list(unknown),
            "units": {"translate": "cm", "rotate": "°"}, "rotation": "XYZ", "order": "local @ T·R·S"}


def bind_handle(packet, path: str | None, rows) -> dict | None:
    """只读的「骨架姿势」手柄：一副骨架的绑定姿势（模型类节点的人物骨架，kit/rig.py RigModel），部位按 side_parts。
    读不到骨架时 None。"""
    from ...data.animation import bind_skeleton

    found = bind_skeleton(packet, path)
    if not found:
        return None
    from ...data.joints import joint_keys

    return skeleton_handle(packet.fingerprint, found["path"], joint_keys(found["names"], found["parents"]),
                           found["parents"], found["world"], side_parts(found["names"], found["parents"], rows, "src"),
                           "绑定姿势")


def side_parts(names, parents, rows, col: str) -> dict[str, list[int]]:
    """一副骨架的部位 -> 关节序号，给手柄的镜像对和对称面：和计算读对应关系同一套（data/joints.py merged_rows +
    part_joints）——参数里写了的行为准，留空的那一列就是「这一边不配」，没写的部位按推测。`col`：src / dst；`names` 是
    关节自己的名字，行按 joint_keys 指关节（重名的按路径）。"""
    from ...data.joints import auto_rows, merged_rows, part_joints, part_names

    guessed = part_names(list(names), parents)
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


def rig_map_choice(src: dict, dst: dict, required: tuple[str, ...]) -> dict:
    """「对应关系」编辑器要的全部数据（NodeDef.choices 里 mapping 那一项）：两侧骨架（`src` 驱动、`dst` 被驱动；
    模型一侧 fixed、只有关节名和每个部位的关节），部位表（id、中文名、区域、是不是链、是不是必需），和推测的对应
    关系（参数格式，编辑器里标「自动」的就是它）。两侧有一侧读不到时只给一句说明。"""
    from ...data.joints import auto_rows, part_names, part_rows

    if not src or not dst:
        return {"options": [], "empty": "所选的骨骼不在场景里"}

    def parts(side: dict) -> dict[str, list[str]]:
        return side["parts"] if side.get("fixed") else part_names(side.get("joints", side["names"]), side["parents"])

    return {"options": [], "rig": {"src": src, "dst": dst, "parts": part_rows(required),
                                   "auto": auto_rows(parts(src), parts(dst))}}


def expression_choice(curves: list[str], shapes: list[str], source: str, target: str) -> dict:
    """「表情重定向（ARKit52）」的「对应关系」编辑器要的数据：同一个编辑器、同一种行，部位槽换成表情槽（data/expressions.py：
    每条源曲线一个槽，ARKit 的按 ARKit 名、按区域分组），两侧是扁平的名字列表（没有层级、没有绑定姿势，编辑器
    不给 3D），推测按名字（ARKit 名、常见别名）。`source` / `target`：两侧所在的路径，只用来显示。"""
    from ...data.expressions import auto_rows, slot_rows

    if not curves or not shapes:
        return {"options": [], "empty": "表情这边没有曲线" if not curves else "目标角色没有 blendshape"}

    def side(label: str, names: list[str], path: str, noun: str, item: str) -> dict:
        return {"label": label, "fixed": False, "names": list(names), "parents": [-1] * len(names), "path": path,
                "noun": noun, "item": item}

    return {"options": [], "rig": {"src": side("表情", curves, source, "曲线", "条曲线"),
                                   "dst": side("目标", shapes, target, "形变", "个形变"),
                                   "parts": slot_rows(curves), "auto": auto_rows(curves, shapes), "slot": "表情"}}
