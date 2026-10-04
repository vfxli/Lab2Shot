"""学习式重定向解算器要什么骨骼：解算器声明需求，「重定向预处理」按需求建议忽略哪些关节，解算器按同一份需求检查输入。

每个解算器节点类声明 `needs`（一条或几条规则，规则 id -> JointNeeds）；预处理的「忽略规则」下拉列出通用规则和所有
已加载的解算器的规则（all_rules），点「自动忽略」就是 suggest_ignored 的结果写进参数。解算器自己不丢骨骼：送进来的
不合它的规则时报错（check），让人回到预处理里改——哪些骨骼没进模型，在预处理的视图里看得见，不在解算器里暗中发生。
以后新接一个解算器只需声明 `needs`。
"""

from __future__ import annotations

from dataclasses import dataclass

from ...data.joints import FINGERS, REGION_OF, joint_keys, spread
from ... import i18n
from ...messages import Msg

# 部位属于哪一组：面部（下巴、眼）、手指、其余都是身体（躯干和四肢）
GROUP_OF = {part: ("face" if region == "face" else "fingers" if region in ("l_hand", "r_hand") else "body")
            for part, region in REGION_OF.items()}
GROUPS = ("body", "fingers", "face")
SIDE_SAID = {"src": "role.source", "dst": "role.target"}  # their words (i18n)


@dataclass(frozen=True)
class JointNeeds:
    """一条规则：解算器收哪些骨骼。"""

    # 下拉里的名字：node.<类型>.needs.<规则 id>.label（rule_label）
    groups: frozenset[str]  # 接受的部位组：GROUPS 里的 "body"、"fingers"、"face"
    # 固定部位表（部位, 链内第几节，-1 = 适配层自己补的末端标记）；空 = 骨架可变。非空时表外的部位不收，链内超出声明
    # 节数的中间关节也不收（保留哪几节按 data/joints.py spread：首尾总在）
    fixed_parts: tuple[tuple[str, int], ...] = ()
    end_sites: bool = True  # 末端点（没有子关节的关节）送不送
    meshes: tuple[str, ...] = ()  # 哪一侧必须带蒙皮网格（"src" / "dst"）
    # 不属于任何部位的关节（根骨、乳房、道具挂点）送不送：只有通用规则送，解算器的规则都按部位收
    loose: bool = False
    # 固定部位表里可以缺的组（适配层自己补标记）：MeshRet 的手指槽，AccuRIG 每指 3 节、没手指的骨架也要能用。
    # 缺了不算问题；有就照表收
    optional: frozenset[str] = frozenset()

    def counts(self) -> dict[str, int]:
        """固定部位表里每个部位声明了几节（末端标记不算）。"""
        out: dict[str, int] = {}
        for part, k in self.fixed_parts:
            if k >= 0:
                out[part] = max(out.get(part, 0), k + 1)
        return out


GENERIC_ID = "generic"
GENERIC = JointNeeds(groups=frozenset(GROUPS), loose=True)


def rule_label(rule: str, owner: str) -> str:
    """一条忽略规则在下拉里的名字（当前语言）：通用规则 needs.generic，节点声明的 node.<类型>.needs.<规则>.label。"""
    from ... import i18n
    from .. import text
    from ..registry import node_types

    if not owner:
        return i18n.Word("needs.generic")
    t = node_types()[owner]
    return text.word(t, "needs", rule, "label") or rule


def rule_owners() -> dict[str, str]:
    """每条忽略规则是哪个节点类型声明的（规则 id -> 节点 id；通用规则 ""），顺序同 all_rules：通用规则在前，再是所有
    已加载、声明了 `needs` 的节点类的规则，按节点名排序。两个节点声明了同一个规则 id 是声明错误（TypeError），不能让
    一条规则悄悄盖掉另一条。"""
    from ..registry import node_types

    owner = {GENERIC_ID: ""}
    for t in sorted(node_types().values(), key=lambda t: (t.subtitle or t.id, t.id)):
        for rule in getattr(t, "needs", None) or {}:
            if rule in owner:
                raise TypeError(f"{t.__name__}: ignore rule {rule!r} is already declared by {owner[rule] or 'core'}")
            owner[rule] = t.id
    return owner


def all_rules() -> dict[str, JointNeeds]:
    """预处理的「忽略规则」下拉：规则 id -> 需求，顺序和唯一性见 rule_owners。"""
    from ..registry import node_types

    types = node_types()
    return {rule: GENERIC if not node else types[node].needs[rule] for rule, node in rule_owners().items()}


def _children(parents) -> dict[int, list[int]]:
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)
    return kids


def _part_of(parts: dict[str, list[int]]) -> dict[int, str]:
    return {j: part for part, joints in parts.items() for j in joints}


def _refused(needs: JointNeeds, parts: dict[str, list[int]], parents) -> set[int]:
    """规则不收的关节（ignored 之前）：见 _refused_why。"""
    return set(_refused_why(needs, parts, parents))


def _refused_why(needs: JointNeeds, parts: dict[str, list[int]], parents) -> dict[int, Msg]:
    """规则不收的关节和理由：不属于接受的组的部位、没有部位的关节（loose 时除外）、固定骨架表外的部位、固定链
    超出声明节数的中间关节、不要末端点时的末端点。"""
    from ...data.joints import part_label

    part_of = _part_of(parts)
    counts = needs.counts()
    kids = _children(parents)
    out: dict[int, Msg] = {}
    for j in range(len(parents)):
        part = part_of.get(j)
        if part is None:
            if not needs.loose:
                out[j] = Msg("I-RETARGETNEEDS-WHYNOPART")
        elif GROUP_OF.get(part, "body") not in needs.groups or (needs.fixed_parts and part not in counts):
            out[j] = Msg("I-RETARGETNEEDS-WHYPART", part=part_label(part))
        elif not needs.end_sites and j not in kids:
            out[j] = Msg("I-RETARGETNEEDS-WHYEND")
    for part, want in counts.items():
        chain = parts.get(part, [])
        if len(chain) > want:
            kept = {j for j in spread(want, chain) if j is not None}
            for j in chain:
                if j not in kept:
                    out[j] = Msg("I-RETARGETNEEDS-WHYCHAIN", part=part_label(part), want=want)
    return out


def ignore_reasons(needs: JointNeeds, names, parents, parts: dict[str, list[int]], rest=None,
                   weights=None) -> dict[int, Msg]:
    """一副骨架按规则建议忽略的关节 -> 理由：先是识别引擎认作辅助或道具的（data/skeleton_recognition.py recognize 的
    角色：名字说的扭转 / 控制 / 道具，`rest` 给了时还有位置说的——与部位关节重合、躺在肢体骨上；对应关系已经当作身体
    部位用的关节不算），再是规则不收的（_refused_why）。`names`：关节自己的名字；`parts`：部位 -> 关节序号（参数行 +
    推测合并后）；`rest` [J,3]：静止位置，`weights`：带蒙皮权重的关节（与推测部位时同一份，识别结果在缓存里共用；
    给了时不驱动任何顶点的关节也是辅助 / 末端）。"""
    from ...data.skeleton_recognition import recognize

    part_of = _part_of(parts)
    roles = recognize([str(n) for n in names], parents, rest, weights=weights).roles
    out = {j: why for j, (role, why) in enumerate(roles) if j not in part_of and role in ("helper", "prop")}
    for j, why in _refused_why(needs, parts, parents).items():
        out.setdefault(j, why)
    return out


def suggest_ignored(needs: JointNeeds, names, parents, parts: dict[str, list[int]], rest=None,
                    weights=None) -> list[str]:
    """一副骨架按规则建议忽略的关节（joint_keys，骨架顺序）：ignore_reasons 的关节。"""
    keys = joint_keys(list(names), parents)
    return [keys[j] for j in sorted(ignore_reasons(needs, names, parents, parts, rest, weights))]


def _few(names: list[str], most: int = 6) -> str:
    from ... import i18n

    return i18n.Both.of(lambda: i18n.separator().join(names[:most]) + (i18n.t("list.more", count=len(names)) if len(names) > most else ""))


def _label(part: str, k: int, count: int) -> str:
    from ...data.joints import part_label

    from ... import i18n

    return i18n.Both.of(lambda: part_label(part) + (i18n.t("joints.segment", n=k + 1) if count > 1 or part.split(".")[-1] in FINGERS else ""))


def check(needs: JointNeeds, src, dst) -> list[Msg]:
    """解算器 prepare 时按规则检查预处理交来的两侧（kit/retarget_prepare.py Prepared）：缺固定部位、链的节数多于模型的、
    送了规则不收的关节、缺蒙皮网格。每个问题一条 I-RETARGETNEEDS-* 片段，解算器把它们放进 E-LEARNEDRETARGET-NEEDS；
    空 = 合规则。"""
    out: list[Msg] = []
    counts = needs.counts()
    for side, got in (("src", src), ("dst", dst)):
        said = i18n.Word(SIDE_SAID[side])
        parts = got.sent_parts()
        missing = [_label(part, k, counts[part]) for part, k in needs.fixed_parts
                   if k >= 0 and len(parts.get(part, [])) <= k and GROUP_OF.get(part, "body") not in needs.optional]
        if missing:
            out.append(Msg("I-RETARGETNEEDS-MISSING", side=said, parts=_few(missing)))
        for part, want in counts.items():
            if len(parts.get(part, [])) > want:
                out.append(Msg("I-RETARGETNEEDS-CHAIN", side=said, part=_label(part, 0, 1), count=len(parts[part]),
                               want=want))
        refused = _refused(needs, got.parts, got.parents) - set(got.ignored_joints) - {
            j for part in counts for j in parts.get(part, [])}  # a long chain is said once above
        if refused:
            out.append(Msg("I-RETARGETNEEDS-REFUSED", side=said, count=len(refused),
                           joints=_few([got.keys[j] for j in sorted(refused)])))
        if side in needs.meshes and not got.has_meshes():
            out.append(Msg("I-RETARGETNEEDS-MESH", side=said))
    return out
