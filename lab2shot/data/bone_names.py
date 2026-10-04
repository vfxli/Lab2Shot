"""项目的骨骼命名规范：给骨骼起名的唯一定义（设计说明：/home/lidong26/Lab2Shot_dev/设计_骨骼命名规范.md）。

采用 Maya HumanIK / Mixamo 体系的名字（Maya 优先：HumanIK 的「自动角色化」按这套名字直接认出每个部位；Mixamo、
MotionBuilder、Blender Rigify 与 UE 的重定向也都认得）。不带 mixamorig: 前缀——那是 Mixamo 自己的命名空间。

- 部位（data/joints.py PARTS）-> 标准名：单关节部位查 STANDARD；链（脊柱、颈、手指）按 chain_name 编号：
  Spine, Spine1, Spine2…（胸接在脊柱后面，取下一个号）；Neck, Neck1…；LeftHandThumb1–3 …；
- 没有部位的骨骼（辅助骨、道具、末端点、面部、认不出的额外骨）按规则命名（rule_names）：
  「最近的有部位的祖先的标准名_类别序号」，类别取识别引擎的角色（data/skeleton_recognition.py roles）：
  Helper（扭转 / 辅助 / 控制骨）、Prop（道具 / 挂点）、End（末端点；一个祖先下只有一个时不编号，如 Head_End）、
  Face（头下的面部骨）、Extra（其余）；序号在同一祖先、同一类别里按关节在骨架里的先后从 1 数起；没有有部位的祖先的
  用 Root。规则只看骨架本身（层级 + 识别结果），同一副骨架永远得到同一组名字；
- 所有名字只用 [A-Za-z0-9_]，在一副骨架里唯一（重名按 lab2shot_shared.names.unique 的 _2、_3）。

谁用：data/joints.py cg_names（解算器交付的蒙皮角色 / 骨架、自动绑定的蒙皮角色都经 data/skeleton.py rig_of_model
走到这里）。别处要给骨骼起名，读这里，不要另写一张表。
"""

from __future__ import annotations

import re

SIDE_WORD = {"l": "Left", "r": "Right"}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# 单关节部位的标准名（Maya HumanIK 的骨骼名）
STANDARD: dict[str, str] = {
    "hips": "Hips", "head": "Head", "jaw": "Jaw",
    **{f"{s}.{p}": f"{SIDE_WORD[s]}{n}" for s in ("l", "r") for p, n in (
        ("clavicle", "Shoulder"), ("upperarm", "Arm"), ("forearm", "ForeArm"), ("hand", "Hand"),
        ("thigh", "UpLeg"), ("shin", "Leg"), ("foot", "Foot"), ("toe", "ToeBase"), ("eye", "Eye"))},
}
# 链的第一节；往后编号 1、2、3…（Spine, Spine1, Spine2；Neck, Neck1）
CHAIN_STEM = {"spine": "Spine", "neck": "Neck"}
# 没有部位的骨骼的类别（识别引擎的角色 -> 名字里的词）
ROLE_WORD = {"helper": "Helper", "prop": "Prop", "end": "End", "face": "Face"}
OTHER_WORD = "Extra"
ROOT_WORD = "Root"


def chain_name(part: str, k: int) -> str:
    """链部位第 k 节（从 0 起）的标准名：手指 LeftHandIndex1、LeftHandIndex2…（从 1 起，手名 + 指名）；脊柱
    Spine、Spine1、Spine2…，颈 Neck、Neck1…（第一节不带号）。"""
    s, _, name = part.rpartition(".")
    if name in FINGERS:
        return f"{SIDE_WORD[s]}Hand{name.title()}{k + 1}"
    stem = CHAIN_STEM.get(part, re.sub(r"[^A-Za-z0-9]", "", part.title()) or "Bone")
    return stem if k == 0 else f"{stem}{k}"


def part_names(parts: dict) -> dict[int, str]:
    """识别出的部位（部位 -> 关节序号，链为列表；data/joints.py guess 的结构）-> {关节: 标准名}。胸是脊柱的下一个号
    （两节脊柱时 Spine2，没有脊柱时 Spine）。"""
    out: dict[int, str] = {}
    for part, where in parts.items():
        if isinstance(where, list):
            for k, j in enumerate(where):
                out[int(j)] = chain_name(part, k)
        elif part == "chest":
            out[int(where)] = chain_name("spine", len(parts.get("spine") or []))
        elif part in STANDARD:
            out[int(where)] = STANDARD[part]
    return out


def rule_names(parents, named: dict[int, str], roles=None) -> dict[int, str]:
    """没有标准名的关节按规则起的名字（模块说明）：{关节: 名字}。`named`：已有标准名的关节；`roles`：每个关节的角色
    （识别引擎 Recognition.roles 的 [(角色, 理由)]，没有时都算 Extra）。"""
    parents = [int(p) for p in parents]
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(p, []).append(j)
    order, stack = [], list(reversed(kids.get(-1, [])))
    while stack:  # 深度优先、子关节按序号：「先后」与骨架的层级顺序一致，和关节在文件里怎么排无关
        j = stack.pop()
        order.append(j)
        stack.extend(reversed(kids.get(j, [])))
    seen = set(order)
    order += [j for j in range(len(parents)) if j not in seen]  # 不成树的关节（环）：放最后
    anchor: dict[int, str] = {}
    groups: dict[tuple[str, str], list[int]] = {}
    for j in order:
        p = parents[j]
        anchor[j] = named[j] if j in named else (anchor.get(p, ROOT_WORD) if p >= 0 else ROOT_WORD)
        if j not in named:
            role = roles[j][0] if roles is not None and j < len(roles) else ""
            word = ROLE_WORD.get(role, OTHER_WORD)
            groups.setdefault((anchor.get(p, ROOT_WORD) if p >= 0 else ROOT_WORD, word), []).append(j)
    out: dict[int, str] = {}
    for (base, word), js in groups.items():
        if base == ROOT_WORD and word == OTHER_WORD and len(js) == 1 and parents[js[0]] < 0:
            out[js[0]] = ROOT_WORD  # 认不出的单个根关节就叫 Root
            continue
        for k, j in enumerate(js):
            out[j] = f"{base}_{word}" if word == "End" and len(js) == 1 else f"{base}_{word}{k + 1}"
    return out


def safe(name: str) -> str:
    """名字只留 [A-Za-z0-9_]（别的字符换成 _，不以数字开头）：Maya、FBX、USD 的节点名都收。"""
    out = re.sub(r"[^A-Za-z0-9_]", "_", str(name)) or "Bone"
    return out if not out[0].isdigit() else f"_{out}"
