"""骨架识别引擎：一副骨架的每个关节是什么身体部位、什么角色，整副骨架朝哪、是什么姿态。

对应关系（data/joints.py guess / part_names）、修正姿态（nodes/kit/retarget_prepare.py auto_pose）、忽略建议
（nodes/kit/retarget_needs.py suggest_ignored）都只读这里的一份结果（recognize），按「骨架本身」（名字 + 父子 +
静止位置）缓存在进程里：编辑器改参数、计算、三个「自动」读到的是同一份识别。

没有一条证据是硬门槛。几种读法各给一份完整的部位分配，再按证据综合（设计_骨架识别.md §2–§3）：

- 名字读法（_by_names，原来的 guess）：名字说出髋、脚、手、头，层级补全其余。名字是好的时它最准；
- 结构读法（_Structure）：不看名字（名字只在平手时作参考），按树的形状和静止位置找人形——成对的、拓扑同构且
  长度相当的两条长肢（腿、臂），髋是两腿与躯干的共同祖先、胸是两臂分开处；上方向 = 躯干方向与两腿方向的反向，
  面朝方向由脚尖、拇指、眼睛的朝向综合，左 = 上 × 前；肢体内按骨长比例（大腿 ≈ 小腿、上臂 ≈ 1.2 × 前臂）
  对齐出各段，扭转 / 辅助关节不挤掉主关节；手指按分叉、拇指按最分开的那一支、其余四指按掌上的排列；
  名字认不出的骨架靠它；
- 名字读法 + 按位置改左右（名字说的左右与位置矛盾、而面朝方向可信时，以位置为准）。

每种读法都用它自己的部位估一次身体坐标系，再逐部位打分：几何是否合理（腿自上而下、脚在地面、手在臂端、
左右与位置一致……）、名字是否支持（同义词软匹配，见 CONCEPTS）、几种读法是否一致。按身体区域（躯干、面部、
每条臂、每条腿、每只手的手指）取得分最高的读法，再算每个部位的置信度（几条证据的 noisy-OR）与依据（中文短句）。
置信度低于 THRESHOLD 的部位不分配：宁可不认，不要认错。

角色：每个关节是 主干 / 肢体 / 手指 / 面部 / 辅助（扭转、共享骨、控制骨）/ 末端 / 道具 / 未知，带理由；
忽略建议读辅助与道具。
"""

from __future__ import annotations

import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from ..messages import Msg
from .joints import part_label
from ..io.digest import sha256

VERSION = 2  # 改了识别规则就加一：缓存键里有它
THRESHOLD = 0.5  # 部位置信度低于它不分配

SIDES = ("l", "r")
CHAINS = ("spine", "neck", *(f"{s}.{f}" for s in ("l", "r") for f in ("thumb", "index", "middle", "ring", "pinky")))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ARM = ("clavicle", "upperarm", "forearm", "hand")
LEG = ("thigh", "shin", "foot", "toe")
TRUNK = ("hips", "spine", "chest", "neck", "head")
FACE = ("jaw", "l.eye", "r.eye")
# 选读法的区域：一个区域整块取自同一种读法，免得一条腿的大腿来自一种读法、小腿来自另一种
REGIONS = (("trunk", TRUNK), ("face", FACE),
           *((f"{s}.arm", tuple(f"{s}.{p}" for p in ARM)) for s in SIDES),
           *((f"{s}.leg", tuple(f"{s}.{p}" for p in LEG)) for s in SIDES),
           *((f"{s}.fingers", tuple(f"{s}.{f}" for f in FINGERS)) for s in SIDES))
ROLES = ("trunk", "limb", "finger", "face", "helper", "end", "prop", "unknown")

# ------------------------------------------------------------------ names: soft evidence

# 部位概念的同义词（小写英文词，或中日文字）：命中加分，不命中不扣到出局。匹配按「词」：名字分词后的单词、
# 相邻两三个词连起来（Left Up Leg → upleg），长度 ≥ 4 的词也认前缀（shoulders、thighs）；中日文按包含。越长的
# 同义词越具体：LeftForeArm 里 forearm 比 arm 具体，是前臂。
CONCEPTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "hips": (("hips", "hip", "pelvis", "cog", "center", "centre"), ("センター", "腰", "骨盤", "髋", "胯", "骨盆")),
    "spine": (("spine", "abdomen", "abdomenlower", "abdomenupper", "waist", "back", "lowerback", "torso", "belly"),
              ("上半身", "脊", "背", "腹")),
    "chest": (("chest", "chestlower", "chestupper", "upperchest", "thorax", "ribcage"), ("胸", "上半身2", "上半身２")),
    "neck": (("neck", "necklower", "neckupper"), ("首", "颈", "頸")),
    "head": (("head", "skull"), ("頭", "头")),
    "jaw": (("jaw", "chin", "mandible"), ("顎", "あご", "下巴")),
    "eye": (("eye", "eyeball"), ("目", "眼")),
    "clavicle": (("clavicle", "collar", "shoulder", "scapula", "collarbone"), ("肩", "锁骨", "鎖骨")),
    "upperarm": (("upperarm", "arm", "shldr", "humerus", "uparm"), ("腕", "上臂", "上腕")),
    "forearm": (("forearm", "lowerarm", "elbow", "ulna", "radius", "lowarm"), ("ひじ", "肘", "前臂", "前腕")),
    "hand": (("hand", "wrist"), ("手首", "手腕", "手")),
    "thigh": (("thigh", "upleg", "upperleg", "femur", "thighbend"), ("大腿", "太もも", "腿根")),
    "shin": (("shin", "calf", "leg", "lowerleg", "knee", "lowleg", "tibia"), ("ひざ", "膝", "小腿", "すね")),
    "foot": (("foot", "ankle"), ("足首", "脚踝", "脚")),
    "toe": (("toe", "toes", "toebase", "ball"), ("つま先", "足先", "趾")),
    "thumb": (("thumb",), ("親指", "拇指", "大拇")),
    "index": (("index", "fore", "pointer"), ("人指", "人差", "食指")),
    "middle": (("middle", "mid"), ("中指",)),
    "ring": (("ring",), ("薬指", "无名指", "無名指")),
    "pinky": (("pinky", "pinkie", "little"), ("小指",)),
}
# 只说「腿」「臂」「手指」的泛称：对一组部位都给一点支持（MMD 的 左足 是大腿）
GENERIC_WORDS = {"leg": ("thigh", "shin", "foot"), "足": ("thigh", "shin", "foot"), "finger": FINGERS,
                 "fingers": FINGERS, "指": FINGERS}
HELPER_WORDS = {"twist", "roll", "helper", "ribbon", "corrective", "offset", "aux", "ik", "fk", "pole", "ctrl",
                "share", "sharebone"}
HELPER_CJK = ("捩", "補助", "ダミー", "IK", "ＩＫ")
PROP_WORDS = {"prop", "props", "weapon", "attach", "attachment", "socket", "gun", "sword", "holster"}
END_WORDS = {"end", "top", "nub", "site", "tip", "effector"}
PALM_WORDS = {"metacarpal", "metacarpals", "carpal", "palm", "meta"}
END_CJK = ("先",)
LEFT_WORDS, RIGHT_WORDS = {"left", "l", "lf", "lft"}, {"right", "r", "rt", "rgt"}


@dataclass
class _Name:
    """一个名字说了什么（_read_name）：每个概念的支持强度（0–1），左右（"l" / "r" / None），是否像辅助、道具、末端。"""

    concepts: dict[str, float]
    side: str | None
    helper: bool
    prop: bool
    end: bool
    said: dict[str, str]
    palm: bool = False  # 概念 -> 命中的词（依据里写）


def _read_name(name: str) -> _Name:
    from .joints import tokens

    words = tokens(name)
    raw = name.rsplit(":", 1)[-1]
    grams = set(words)
    for k in (2, 3):
        grams |= {"".join(words[i:i + k]) for i in range(len(words) - k + 1)}
    found: dict[str, float] = {}
    said: dict[str, str] = {}

    def hit(concept: str, word: str, strength: float) -> None:
        if strength > found.get(concept, 0.0):
            found[concept], said[concept] = strength, word

    for concept, (latin, cjk) in CONCEPTS.items():
        for w in latin:
            if w in grams or (len(w) >= 4 and any(g.startswith(w) and len(g) <= len(w) + 2 for g in grams)):
                hit(concept, w, min(1.0, 0.55 + 0.07 * len(w)))
        for w in cjk:
            if w in raw:
                hit(concept, w, min(1.0, 0.6 + 0.2 * len(w)))
    for w, concepts in GENERIC_WORDS.items():
        if w in grams or (not w.isascii() and w in raw):
            for c in concepts:
                hit(c, w, 0.3)
    if found:  # 越具体越可信：比最强的弱很多的概念只算一半（LeftForeArm 的 arm、UpLeg 的 leg）
        best = max(found.values())
        found = {c: (v if v >= best - 0.05 else v * 0.5) for c, v in found.items()}
    left = bool(set(words) & LEFT_WORDS) or "左" in raw or "ひだり" in raw
    right = bool(set(words) & RIGHT_WORDS) or "右" in raw or "みぎ" in raw
    side = "l" if left and not right else "r" if right and not left else None
    helper = bool(set(words) & HELPER_WORDS) or any(w in raw for w in HELPER_CJK)
    prop = bool(set(words) & PROP_WORDS)
    end = bool(set(words) & END_WORDS) or (raw.endswith(END_CJK) and len(raw) > 1 and "つま先" not in raw)
    return _Name(found, side, helper, prop, end, said, bool(set(words) & PALM_WORDS))


# which body region a concept belongs to, and the regions a name may stray into without contradicting (the chest
# named Spine2, the clavicle named Shoulder, SMPL's thigh named hip and its toe named foot)
CONCEPT_REGION = {**{c: "trunk" for c in ("hips", "spine", "chest", "neck", "head")},
                  **{c: "arm" for c in ARM}, **{c: "leg" for c in LEG}, **{c: "finger" for c in FINGERS},
                  "jaw": "face", "eye": "face"}
NEAR = {("trunk", "leg"), ("leg", "trunk"), ("arm", "finger"), ("finger", "arm"), ("trunk", "face"),
        ("face", "trunk"), ("trunk", "arm"), ("arm", "trunk")}


def _name_support(name: _Name, part: str) -> float:
    """名字对「这个关节是 part」的支持，-1..1：命中这个概念为正；说的是另一侧、或明确说是身体另一个区域的部位
    （叫 Hand 的关节在腿上）为负（矛盾）；什么都没说、或说的是相邻的部位（胸叫 Spine2）为 0。"""
    side, _, concept = part.rpartition(".")
    if side and name.side and name.side != side:
        return -0.8
    got = name.concepts.get(concept, 0.0)
    if got > 0:
        return got
    mine = CONCEPT_REGION[concept]
    other = max((v for c, v in name.concepts.items()
                 if CONCEPT_REGION[c] != mine and (CONCEPT_REGION[c], mine) not in NEAR), default=0.0)
    return -0.5 * other if other >= 0.6 else 0.0


# ------------------------------------------------------------------ the tree


class _Tree:
    """一副骨架的层级与（可选的）静止位置：子关节、深度、骨长、往下最远能到多远（reach）、子树、最近公共祖先。"""

    def __init__(self, parents, pos: np.ndarray | None) -> None:
        self.parents = [int(p) for p in parents]
        n = self.n = len(self.parents)
        self.kids: list[list[int]] = [[] for _ in range(n)]
        roots = []
        for j, p in enumerate(self.parents):
            if 0 <= p < n and p != j:
                self.kids[p].append(j)
            else:
                self.parents[j] = -1
                roots.append(j)
        order, seen = [], set()
        stack = list(reversed(roots))
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            order.append(j)
            stack.extend(reversed(self.kids[j]))
        order += [j for j in range(n) if j not in seen]  # a cycle (not a tree): kept, never followed
        self.order = order
        self.depth = [0] * n
        for j in order:
            p = self.parents[j]
            self.depth[j] = self.depth[p] + 1 if p >= 0 else 0
        self.size = [1] * n
        for j in reversed(order):
            p = self.parents[j]
            if p >= 0:
                self.size[p] += self.size[j]
        self.pos = pos
        if pos is not None:
            self.blen = [float(np.linalg.norm(pos[j] - pos[self.parents[j]])) if self.parents[j] >= 0 else 0.0
                         for j in range(n)]
            self.reach = [0.0] * n
            for j in reversed(order):
                p = self.parents[j]
                if p >= 0:
                    self.reach[p] = max(self.reach[p], self.blen[j] + self.reach[j])
            span = pos.max(0) - pos.min(0)
            self.ext = float(np.linalg.norm(span)) or 0.0
        self._shape: dict[int, int] = {}
        self._shapes: dict[tuple, int] = {}

    def ancestors(self, j: int) -> list[int]:
        out = []
        while j >= 0 and len(out) <= self.n:
            out.append(j)
            j = self.parents[j]
        return out

    def lca(self, a: int, b: int) -> int:
        up = set(self.ancestors(a))
        return next((j for j in self.ancestors(b) if j in up), -1)

    def is_below(self, j: int, top: int) -> bool:
        """`j` is `top` or under it."""
        return top in self.ancestors(j)

    def path(self, top: int, bottom: int) -> list[int]:
        """Joints from below `top` down to `bottom` (top excluded), [] when `top` is not above `bottom`."""
        up = self.ancestors(bottom)
        return list(reversed(up[: up.index(top)])) if top in up else []

    def subtree(self, j: int) -> list[int]:
        out, stack = [], [j]
        while stack:
            k = stack.pop()
            out.append(k)
            stack.extend(self.kids[k])
        return out

    def shape(self, j: int) -> int:
        """The topology of j's subtree as an id: two subtrees of the same id are isomorphic."""
        if j not in self._shape:
            for k in reversed(self.subtree(j)):
                if k not in self._shape:
                    key = tuple(sorted(self._shape[c] for c in self.kids[k]))
                    self._shape[k] = self._shapes.setdefault(key, len(self._shapes))
        return self._shape[j]

    def line(self, j: int, avoid=()) -> list[int]:
        """From `j` down its main line: at every fork the child reaching furthest (the arm into the middle finger,
        not into a twist leaf beside it)."""
        out = [j]
        while True:
            nxt = [c for c in self.kids[out[-1]] if c not in avoid]
            if not nxt or len(out) > self.n:
                return out
            out.append(max(nxt, key=lambda c: (self.blen[c] + self.reach[c], -c)))

    def d(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.pos[a] - self.pos[b]))


def _unit(v) -> np.ndarray | None:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else None


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ------------------------------------------------------------------ the result


@dataclass
class Frame:
    """一副骨架的身体坐标系（静止姿态，骨架自己的空间）：上、前、左（单位向量；估不出为 None），面朝方向的可信度
    （0–1：脚尖、拇指、眼睛几条线索一致而且清楚时高），上方向的可信度，姿态类型（"T" / "A" / "other" / ""），
    身高（沿上方向的跨度）。"""

    up: np.ndarray | None = None
    forward: np.ndarray | None = None
    left: np.ndarray | None = None
    up_conf: float = 0.0
    facing_conf: float = 0.0
    pose: str = ""
    height: float = 0.0

    def report(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 4) for x in v]  # noqa: E731
        return {"up": r(self.up), "forward": r(self.forward), "left": r(self.left), "up_conf": round(self.up_conf, 2),
                "facing_conf": round(self.facing_conf, 2), "pose": self.pose, "height": round(self.height, 4)}


@dataclass
class Recognition:
    """recognize 的结果。`parts` 同 joints.guess：部位 -> 关节序号（链为列表），只含置信度够的；`confidence`、
    `evidence`：每个部位（含没分配的候选）的置信度与依据（Msg）；`roles`：每个关节的角色（ROLES）与理由（Msg，未知为
    None）；`frame`：身体坐标系；
    `rejected`：置信度不够而没分配的部位 -> 候选关节；`readings`：每个区域取自哪种读法。"""

    parts: dict
    confidence: dict[str, float]
    evidence: dict[str, list]
    roles: list[tuple[str, object]]
    frame: Frame
    rejected: dict = field(default_factory=dict)
    readings: dict[str, str] = field(default_factory=dict)

    def report(self) -> dict:
        """给编辑器的（handle_data）：每个部位的置信度与依据，没分配的部位为什么没分配，身体坐标系。"""
        out = {p: {"confidence": round(self.confidence[p], 2), "evidence": [m.text for m in self.evidence.get(p, [])]}
               for p in self.confidence}
        for p in self.rejected:
            out[p]["assigned"] = False
        return {"parts": out, "frame": self.frame.report(), "threshold": THRESHOLD}


def _joints_of(where) -> list[int]:
    return list(where) if isinstance(where, list) else [where]


# ------------------------------------------------------------------ frame from any reading


def _frame(tree: _Tree, parts: dict) -> Frame:
    """一种读法的部位 -> 身体坐标系。上 = 躯干方向（髋 -> 头 / 胸）与两腿方向反向的综合，两者不一致（静止姿态
    是折叠的、不是站姿：LAFAN 的零旋转）时上方向不可信；前 = 脚尖、拇指、眼睛、膝盖几条线索的加权；左 = 上 × 前。"""
    f = Frame()
    pos = tree.pos
    if pos is None or tree.ext <= 0:
        return f
    one = lambda p: parts[p][0] if isinstance(parts.get(p), list) and parts[p] else parts.get(p)  # noqa: E731
    hips = one("hips")
    top = next((j for j in (one("head"), (parts.get("neck") or [None])[-1] if parts.get("neck") else None,
                            one("chest"), (parts.get("spine") or [None])[-1] if parts.get("spine") else None)
                if j is not None), None)
    feet = [j for j in (one(f"{s}.foot") or one(f"{s}.shin") for s in SIDES) if j is not None]
    trunk = _unit(pos[top] - pos[hips]) if hips is not None and top is not None and top != hips else None
    legs = _unit(np.mean(pos[feet], 0) - pos[hips]) if hips is not None and feet else None
    if trunk is not None and legs is not None:
        agree = float(np.dot(trunk, -legs))
        f.up = _unit(trunk - legs)
        f.up_conf = _clip01((agree - 0.3) / 0.5)
    elif trunk is not None:
        f.up, f.up_conf = trunk, 0.4
    elif legs is not None:
        f.up, f.up_conf = -legs, 0.4
    if f.up is None:
        return f
    proj = pos @ f.up
    f.height = float(proj.max() - proj.min())
    flat = lambda v: v - np.dot(v, f.up) * f.up  # noqa: E731
    cues, weight = np.zeros(3), 0.0
    for s in SIDES:
        foot, toe = one(f"{s}.foot"), one(f"{s}.toe")
        if foot is not None and toe is not None:
            v = flat(pos[toe] - pos[foot])
            if np.linalg.norm(v) > 0.3 * (tree.d(toe, foot) or 1) and np.linalg.norm(v) > 1e-3 * tree.ext:
                cues += 3 * v / np.linalg.norm(v)
                weight += 3
        hand, thumb = one(f"{s}.hand"), parts.get(f"{s}.thumb")
        others = [parts[f"{s}.{k}"][0] for k in FINGERS[1:] if parts.get(f"{s}.{k}")]
        if hand is not None and thumb and others:
            arm = _unit(pos[hand] - pos[one(f"{s}.forearm")]) if one(f"{s}.forearm") is not None else None
            v = flat(pos[thumb[0]] - np.mean(pos[others], 0))
            if arm is not None:
                v = v - np.dot(v, arm) * arm
            if np.linalg.norm(v) > 2e-3 * tree.ext:
                cues += v / np.linalg.norm(v)
                weight += 1
    eyes = [one(f"{s}.eye") for s in SIDES if one(f"{s}.eye") is not None]
    head = one("head")
    if eyes and head is not None:
        v = flat(np.mean(pos[eyes], 0) - pos[head])
        if np.linalg.norm(v) > 2e-3 * tree.ext:
            cues += v / np.linalg.norm(v)
            weight += 1
    for s in SIDES:  # a bent knee points forward
        thigh, shin, foot = one(f"{s}.thigh"), one(f"{s}.shin"), one(f"{s}.foot")
        if None not in (thigh, shin, foot):
            v = flat(pos[shin] - (pos[thigh] + pos[foot]) / 2)
            if np.linalg.norm(v) > 0.01 * tree.ext:
                cues += 0.5 * v / np.linalg.norm(v)
                weight += 0.5
    if weight > 0 and np.linalg.norm(cues) > 1e-9:
        f.forward = cues / np.linalg.norm(cues)
        f.left = _unit(np.cross(f.up, f.forward))
        f.facing_conf = _clip01(float(np.linalg.norm(cues)) / weight) * _clip01(weight / 3.0) * (0.5 + 0.5 * f.up_conf)
    for s in SIDES:  # T or A: how far the arm hangs below the horizontal
        up_arm, hand = one(f"{s}.upperarm"), one(f"{s}.hand")
        if up_arm is not None and hand is not None and (v := _unit(pos[hand] - pos[up_arm])) is not None:
            below = float(np.degrees(np.arcsin(np.clip(-np.dot(v, f.up), -1, 1))))
            f.pose = "T" if below < 25 else "A" if below < 70 else "other"
            break
    return f


# ------------------------------------------------------------------ the structure reading


@dataclass
class _Reading:
    """一种读法：部位、每个部位这种读法自己的把握（0–1）和依据、它的身体坐标系。"""

    name: str
    parts: dict
    sure: dict[str, float] = field(default_factory=dict)
    said: dict[str, list[str]] = field(default_factory=dict)
    frame: Frame | None = None


class _Structure:
    """按形状和位置读人形（名字只在平手时参考）。"""

    LIMB = 0.12  # 一条肢体至少这么长（相对骨架包围盒对角线）
    PAIRS = 14  # 最多看这么多对候选肢体

    def __init__(self, tree: _Tree, names: list[_Name], inert=frozenset()) -> None:
        self.t, self.names = tree, names
        self.inert = inert  # 不驱动任何顶点的关节（_inert）：不当肢体的候选，位置照用
        self._matched: dict = {}
        self._reach: dict = {}
        self.extras: dict = {}
        self.avoid: set[int] = {j for j in range(tree.n) if names[j].prop}

    def line(self, j: int) -> list[int]:
        """The main line from `j`, past the joints the pair's other side has no counterpart for (a prop)."""
        return self.t.line(j, self.avoid)

    def reach(self, j: int) -> float:
        """How far below `j` its joints reach, the avoided ones (a prop) left out."""
        tag = (id(self.avoid), len(self.avoid))
        got = self._reach.get((j, tag))
        if got is None:
            t = self.t
            for k in reversed(t.subtree(j)):  # children before their parent: no recursion down a long chain
                if (k, tag) not in self._reach:
                    self._reach[(k, tag)] = max((t.blen[c] + self._reach[(c, tag)] for c in t.kids[k]
                                                 if c not in self.avoid), default=0.0)
            got = self._reach[(j, tag)]
        return got

    # -- candidates

    def match(self, a: int, b: int):
        """Two subtrees laid over each other, children paired greedily by length and size (a prop in one hand, a twist
        leaf on one side only stay unpaired): (paired joints, a's reach along paired joints, b's, the unpaired)."""
        key = (a, b)
        if key in self._matched:
            return self._matched[key]
        t = self.t
        L = lambda c: t.blen[c] + t.reach[c]  # noqa: E731
        costs = []
        for c in t.kids[a]:
            for d in t.kids[b]:
                # by the bone to the child (a prop further down does not change it) and the size of what hangs there
                top = max(t.blen[c], t.blen[d], 1e-3 * (t.ext or 1))
                diff = abs(t.blen[c] - t.blen[d]) / top + 0.3 * abs(t.size[c] - t.size[d]) / max(t.size[c], t.size[d]) + (
                    0.0 if t.shape(c) == t.shape(d) else 0.15) + 0.1 * abs(L(c) - L(d)) / max(L(c), L(d), 1e-9)
                if diff < 0.7:
                    costs.append((diff, c, d))
        costs.sort()
        used_a, used_b, count, ra, rb, extra = set(), set(), 1, 0.0, 0.0, set()
        for _, c, d in costs:
            if c in used_a or d in used_b:
                continue
            used_a.add(c)
            used_b.add(d)
            n, x, y, e = self.match(c, d)
            count += n
            ra, rb = max(ra, t.blen[c] + x), max(rb, t.blen[d] + y)
            extra |= e
        for c in t.kids[a]:
            if c not in used_a:
                extra |= set(t.subtree(c))
        for d in t.kids[b]:
            if d not in used_b:
                extra |= set(t.subtree(d))
        self._matched[key] = got = (count, ra, rb, extra)
        return got

    def pairs(self) -> list[tuple[int, int, int, float]]:
        """成对的长肢：同一个关节下的两个子关节，两棵子树叠得上（match：大部分关节配得上对，配上的部分一样长）、
        根骨等长。(分叉, a, b, 对称度)；配不上对的关节（一只手里的道具）记在 self.extras。"""
        t = self.t
        out = []
        for f in range(t.n):
            ks = [c for c in t.kids[f] if t.blen[c] + t.reach[c] >= self.LIMB * t.ext and not self.names[c].prop
                  and c not in self.inert]
            if len(ks) > 40:  # a face rig's fan of short strands: no limbs there
                ks = sorted(ks, key=lambda c: -(t.blen[c] + t.reach[c]))[:40]
            for a, b in combinations(ks, 2):
                n, ra, rb, extra = self.match(a, b)
                if n < 0.6 * max(t.size[a], t.size[b]):
                    continue
                la, lb = t.blen[a] + ra, t.blen[b] + rb
                if min(la, lb) < self.LIMB * t.ext or max(la, lb) > 1.35 * min(la, lb):
                    continue
                if abs(t.blen[a] - t.blen[b]) > 0.04 * t.ext + 0.3 * max(t.blen[a], t.blen[b]):
                    continue
                sym = (n / max(t.size[a], t.size[b])) * (1 - abs(la - lb) / max(la, lb))
                self.extras[(a, b)] = extra | {j for j in extra for j in t.subtree(j)}
                out.append((f, a, b, sym, min(la, lb)))
        out.sort(key=lambda q: -q[4] * q[3])
        return [q[:4] for q in out[: self.PAIRS]]

    def _words(self, joints, concepts) -> float:
        return sum(max((self.names[j].concepts.get(c, 0.0) for c in concepts), default=0.0) for j in joints)

    def finger_fork(self, j: int) -> int:
        """How many finger-like branches `j` has: three or more children (through a palm joint) of comparable length."""
        t = self.t
        ls = [t.blen[c] + self.reach(c) for c in t.kids[j] if c not in self.avoid]
        ls = [x for x in ls if x > 1e-6 * t.ext]
        if len(ls) < 2:
            return len(ls)
        top = max(ls)
        return sum(1 for x in ls if x >= top / 3.0)

    def hypotheses(self):
        """Every (legs, arms) pair of candidate pairs, scored: legs go down from the hips to the ground, the trunk goes
        up the other way, the arms part at the top of the trunk, the two pairs spread along the same sideways axis."""
        t, pos = self.t, self.t.pos
        pairs = self.pairs()
        subs = {}
        for f, a, b, _ in pairs:
            extra = self.extras.get((a, b), set())
            for c in (a, b):
                subs[c] = (subs[c] if c in subs else set(t.subtree(c))) - extra
        out = []
        for legs in pairs:
            for arms in [*pairs, None]:
                if arms is legs:
                    continue
                h = self._score(legs, arms, subs)
                if h is not None:
                    out.append(h)
        out.sort(key=lambda h: -h["score"])
        return out

    def _score(self, legs, arms, subs):
        t, pos = self.t, self.t.pos
        fL, a1, b1, sym_l = legs
        sub_l = subs[a1] | subs[b1]
        if arms is not None:
            fA, a2, b2, sym_a = arms
            sub_a = subs[a2] | subs[b2]
            if sub_a & sub_l or fA in sub_l or fL in sub_a:
                return None
            hips = t.lca(fL, fA)
            if hips < 0 or hips == fA:
                return None
            chest = fA
        else:
            hips, chest, sym_a, sub_a = fL, None, 0.0, set()
        self.avoid = {j for j in range(t.n) if self.names[j].prop} | self.extras.get((a1, b1), set()) | (
            self.extras.get((arms[1], arms[2]), set()) if arms is not None else set())
        ends = [self.line(a1)[-1], self.line(b1)[-1]]
        leg_dir = _unit(np.mean(pos[ends], 0) - pos[hips])
        if leg_dir is None:
            return None
        if chest is not None:
            trunk = _unit(pos[chest] - pos[hips])
            if trunk is None:
                return None
            cos_t = float(np.dot(trunk, -leg_dir))
            up = _unit(trunk - leg_dir)
        else:
            others = [c for c in t.kids[hips] if c not in (a1, b1)]
            if not others:
                return None
            top = max((k for c in others for k in t.subtree(c)), key=lambda k: float(np.dot(pos[k] - pos[hips], -leg_dir)))
            trunk = _unit(pos[top] - pos[hips])
            if trunk is None:
                return None
            cos_t = float(np.dot(trunk, -leg_dir))
            up = _unit(trunk - leg_dir)
        if up is None:
            return None
        proj = pos @ up
        ground, height = float(proj.min()), float(np.ptp(proj)) or 1.0
        c_trunk = _clip01((cos_t - 0.3) / 0.5)
        low = min(float(proj[list(subs[a1])].min()), float(proj[list(subs[b1])].min()))
        c_ground = _clip01(1 - (low - ground) / (0.1 * height))
        downs = [float(np.dot(_unit(pos[e] - pos[c]) if _unit(pos[e] - pos[c]) is not None else up, -up))
                 for c, e in zip((a1, b1), ends)]
        c_down = _clip01((min(downs) - 0.45) / 0.4)
        c_lat = 1.0
        if arms is not None:
            def lateral(a, b):
                v = np.mean(pos[list(subs[a])], 0) - np.mean(pos[list(subs[b])], 0)
                return _unit(v - np.dot(v, up) * up)
            la, lb = lateral(a1, b1), lateral(a2, b2)
            c_lat = _clip01((abs(float(np.dot(la, lb))) - 0.5) / 0.4) if la is not None and lb is not None else 0.0
            c_armtop = _clip01(1 - (float(proj[chest]) - float(proj[hips]) < 0) * 1.0)
        else:
            c_armtop = 0.5
        quality = (max(c_trunk, 1e-3) * max(c_ground, 1e-3) * max(c_down, 1e-3) * max(c_lat, 1e-3)) ** 0.25
        score = 4 * quality + 0.5 * c_armtop + 0.5 * (sym_l + sym_a)
        if arms is not None:
            forks = [max(self.finger_fork(k) for k in self.line(a)) for a in (a2, b2)]
            score += 0.4 * min(1.0, min(forks) / 3)
            heads = [c for c in t.kids[chest] if c not in (a2, b2)]
            if heads and max(float(proj[list(t.subtree(c))].max()) for c in heads) > float(proj[chest]) + 0.03 * height:
                score += 0.6
            leg_w = self._words(sub_l, ("thigh", "shin", "foot", "toe"))
            arm_w = self._words(sub_a, ("clavicle", "upperarm", "forearm", "hand", *FINGERS))
            wrong = self._words(sub_l, ("upperarm", "forearm", "hand", *FINGERS)) + self._words(
                sub_a, ("thigh", "shin", "foot", "toe"))
            total = leg_w + arm_w + wrong
            if total > 0:
                score += 1.0 * (leg_w + arm_w - wrong) / total
        else:
            score -= 1.5
        score += 0.3 * (t.blen[a1] + t.reach[a1]) / t.ext
        return {"legs": legs, "arms": arms, "hips": hips, "chest": chest, "up": up, "quality": quality,
                "avoid": set(self.avoid),
                "score": score, "checks": {"I-SKREC-TRUNKLEGS": c_trunk, "I-SKREC-LEGSGROUND": c_ground,
                                           "I-SKREC-LEGSDOWN": c_down, "I-SKREC-PAIRSAXIS": c_lat}}

    # -- one hypothesis into parts

    def read(self) -> list[_Reading]:
        t = self.t
        if t.pos is None or t.ext <= 0 or t.n < 6:
            return []
        hyps = self.hypotheses()
        out = []
        if hyps and hyps[0]["quality"] >= 0.3:
            best = hyps[0]
            differs = [h for h in hyps[1:] if h["legs"] != best["legs"] or (
                h["arms"] is not None and best["arms"] is not None and h["arms"] != best["arms"])]
            margin = best["score"] - differs[0]["score"] if differs else 3.0
            out.append(self.body(best, margin, "structure"))
            if differs and differs[0]["quality"] > 0.5:
                out.append(self.body(differs[0], 0.0, "structure.second"))
        else:
            hand = self.lone_hand()
            if hand is not None:
                out.append(hand)
        return out

    def body(self, h: dict, margin: float, label: str) -> _Reading:
        t, pos = self.t, self.t.pos
        q = h["quality"]
        base = _clip01(q) * (0.55 + 0.4 * _clip01(margin / 1.5))
        parts: dict = {}
        sure: dict[str, float] = {}
        said: dict[str, list[str]] = {}

        def put(part, where, s, *why):
            parts[part] = where
            sure[part] = s
            said[part] = list(why)

        hips, chest = h["hips"], h["chest"]
        self.avoid = h["avoid"]
        height = float(np.ptp(pos @ h["up"])) or 1.0
        for k in t.ancestors(hips)[1:]:  # a joint just above the leg fork named the hips (Biped's Pelvis) is them
            if t.d(k, hips) > 0.06 * height:
                break
            if self.names[k].concepts.get("hips", 0) >= 0.6 and not self.names[k].helper:
                hips = k
                break
        put("hips", hips, base, Msg("I-SKREC-HIPS"))
        fL, a1, b1, _ = h["legs"]
        self._axis = (pos[hips], h["up"], float(np.ptp(pos @ h["up"])) or 1.0)
        legs = [self.leg(a) for a in (a1, b1)]
        arms = []
        if h["arms"] is not None:
            fA, a2, b2, _ = h["arms"]
            arms = [self.arm(a) for a in (a2, b2)]
            trunk = t.path(hips, chest)[:-1]
            spine = [j for j in trunk if not self.names[j].helper] or trunk
            if spine:
                put("spine", spine, base, Msg("I-SKREC-SPINE"))
            shoulders = [a["joints"][1] for a in arms if a["joints"][1] is not None]
            head, neck, head_sure, why = self.head(chest, (a2, b2), h["up"], shoulders)
            proj = pos @ h["up"]
            if head is not None and not neck and spine and t.parents[head] == chest and float(proj[chest]) >= float(
                    np.mean(proj[[a2, b2]])) - 0.005 * t.ext:
                # the arms part above the chest's height, right under the head: that joint is the neck's base (3ds
                # Max Biped hangs its clavicles under Bip01 Neck); the chest is the one below it
                neck, chest = [chest], spine[-1]
                spine = spine[:-1]
                if spine:
                    put("spine", spine, base, Msg("I-SKREC-SPINE"))
                else:
                    parts.pop("spine", None)
                put("chest", chest, base * 0.9, Msg("I-SKREC-CHESTBELOW"))
            else:
                put("chest", chest, base, Msg("I-SKREC-CHEST"))
            if head is not None:
                if neck:
                    put("neck", neck, base * head_sure, Msg("I-SKREC-NECK"))
                put("head", head, base * head_sure, why)
        # sides, by the facing; the frame wants the parts sided first: read it on side-less copies
        proto = dict(parts)
        for k, (leg, arm) in enumerate(zip(legs, arms or [None, None])):
            s = SIDES[k]
            for p, j in zip(LEG, leg["joints"]):
                if j is not None:
                    proto[f"{s}.{p}"] = j
            if arm:
                for p, j in zip(ARM, arm["joints"]):
                    if j is not None:
                        proto[f"{s}.{p}"] = j
                for f, chain in arm["fingers"].items():
                    proto[f"{s}.{f}"] = chain
        frame = _frame(t, proto)
        if frame.up is None:
            frame.up = h["up"]
        eyes, jaw = self.face(parts.get("head"), frame)
        if eyes:
            proto["l.eye"], proto["r.eye"] = eyes[0], eyes[1]
            frame = _frame(t, proto)
        if frame.up is None:
            frame.up = h["up"]
        groups = [("腿", (a1, b1), legs)]
        if arms:
            groups.append(("臂", (h["arms"][1], h["arms"][2]), arms))
        for label_cn, (pa, pb), limbs in groups:
            side_of, side_sure, why = self.sides(pa, pb, frame, hips)
            for limb, root in zip(limbs, (pa, pb)):
                s = side_of[root]
                names = LEG if label_cn == "腿" else ARM
                for p, j in zip(names, limb["joints"]):
                    if j is not None:
                        put(f"{s}.{p}", j, base * side_sure * limb["fit"], *limb["why"].get(p, []), why)
                for f, chain in limb.get("fingers", {}).items():
                    put(f"{s}.{f}", chain, base * side_sure * limb["finger_sure"].get(f, 0.0),
                        *limb["finger_why"].get(f, []), why)
        if eyes and frame.left is not None:
            l, r = eyes[:2] if float(np.dot(pos[eyes[0]] - pos[eyes[1]], frame.left)) > 0 else (eyes[1], eyes[0])
            sure_eye = base * frame.facing_conf * eyes[2]
            put("l.eye", l, sure_eye, Msg("I-SKREC-EYES"))
            put("r.eye", r, sure_eye, Msg("I-SKREC-EYES"))
        if jaw is not None:
            put("jaw", jaw[0], base * jaw[1], Msg("I-SKREC-JAW"))
        said["hips"] += [Msg(k) for k, v in h["checks"].items() if v >= 0.8]
        reading = _Reading(label, parts, sure, said)
        reading.frame = _frame(t, parts) if any(p.startswith(("l.", "r.")) for p in parts) else frame
        if reading.frame.up is None:
            reading.frame = frame
        return reading

    def sides(self, a: int, b: int, frame: Frame, hips: int):
        """Which of a pair is the left: by the facing (left = up × forward) when that is clear; otherwise by the names
        on each side; with neither, not known (low)."""
        t, pos = self.t, self.t.pos
        ca, cb = np.mean(pos[t.subtree(a)], 0), np.mean(pos[t.subtree(b)], 0)
        geo = None
        if frame.left is not None and frame.facing_conf > 0.05:
            spread = float(np.dot(ca - cb, frame.left)) / (t.ext or 1)
            sep = _clip01(abs(spread) / 0.04)
            geo = ("l" if spread > 0 else "r", frame.facing_conf * sep)
        votes = {"l": 0.0, "r": 0.0}
        for root, sign in ((a, 1), (b, -1)):
            for j in t.subtree(root):
                s = self.names[j].side
                if s:
                    votes[s if sign > 0 else ("r" if s == "l" else "l")] += 1
        named = None
        if votes["l"] + votes["r"] > 0:
            s = "l" if votes["l"] > votes["r"] else "r"
            named = (s, abs(votes["l"] - votes["r"]) / (votes["l"] + votes["r"]))
        if geo is not None and geo[1] >= 0.45:
            s, sure = geo
            why = Msg("I-SKREC-SIDEPOSNAMES" if named and named[0] != s and named[1] > 0.5 else "I-SKREC-SIDEPOS")
            sure = min(1.0, 0.55 + 0.45 * sure + (0.2 if named and named[0] == s else 0.0))
        elif named is not None and named[1] > 0.5:
            s, sure = named[0], 0.85
            why = Msg("I-SKREC-SIDENAME")
        elif geo is not None:
            s, sure = geo[0], 0.3 + 0.5 * geo[1]
            why = Msg("I-SKREC-SIDEWEAK")
        else:
            s, sure, why = "l", 0.2, Msg("I-SKREC-SIDENONE")
        other = "r" if s == "l" else "l"
        return {a: s, b: other}, sure, why

    def leg(self, root: int) -> dict:
        """A leg's line as thigh, shin, foot, toe: the knee where thigh ≈ shin, the ankle near the ground where the
        line turns forward, the toe after it."""
        t, pos = self.t, self.t.pos
        line = self.line(root)
        best, best_cost = None, None
        n = len(line)
        for i in range(n):
            for k in range(i + 1, n):
                for a in range(k + 1, n):
                    u, e, w = line[i], line[k], line[a]
                    l1, l2 = t.d(u, e), t.d(e, w)
                    if l1 <= 1e-9 or l2 <= 1e-9:
                        continue
                    cost = 2.0 * abs(np.log(l1 / l2))
                    cost -= 0.4 * (self.said(u, "thigh") + self.said(e, "shin") + self.said(w, "foot"))
                    nxt = line[a + 1] if a + 1 < n else None
                    if nxt is not None and (v := _unit(pos[nxt] - pos[w])) is not None and (s := _unit(pos[w] - pos[e])) is not None:
                        cost += 0.8 * max(0.0, float(np.dot(v, s)) - 0.5)  # the foot turns away from the shin
                    # the ankle is near the bottom of the leg: what is left below it is short
                    rest = self.reach(w)
                    cost += 1.5 * max(0.0, rest / (l2 or 1) - 0.6)
                    cost += 0.15 * i  # nearer the hips: no long bone above the thigh within the leg
                    cost += 0.5 * min(1.5, abs(np.log(max(self.lateral(u), 1e-3) / 0.055)))  # the hip socket
                    if i > 0 and t.d(line[0], u) > 0.5 * l1:
                        cost += 1.0
                    if best_cost is None or cost < best_cost:
                        best, best_cost = (i, k, a), cost
        if best is None:
            return {"joints": [line[0], line[1] if n > 1 else None, line[2] if n > 2 else None, None], "fit": 0.4,
                    "why": {}}
        i, k, a = best
        toe = line[a + 1] if a + 1 < n and not self.names[line[a + 1]].end else None
        fit = _clip01(1.1 - 0.35 * best_cost)
        ratio = t.d(line[i], line[k]) / max(t.d(line[k], line[a]), 1e-9)
        why = {"thigh": [Msg("I-SKREC-THIGH", ratio=round(ratio, 2))], "shin": [Msg("I-SKREC-KNEE")],
               "foot": [Msg("I-SKREC-ANKLE")], "toe": [Msg("I-SKREC-TOE")]}
        return {"joints": [line[i], line[k], line[a], toe], "fit": max(fit, 0.55), "why": why}

    def arm(self, root: int) -> dict:
        """An arm's line as clavicle, upper arm, forearm, hand (upper arm ≈ 1.2 × forearm; the hand where the line
        forks into the fingers), and the hand's fingers."""
        t = self.t
        line = self.line(root)
        n = len(line)
        best, best_cost = None, None
        for i in range(n):
            for k in range(i + 1, n):
                for a in range(k + 1, n):
                    u, e, w = line[i], line[k], line[a]
                    l1, l2 = t.d(u, e), t.d(e, w)
                    if l1 <= 1e-9 or l2 <= 1e-9:
                        continue
                    cost = 2.0 * abs(np.log(l1 / (1.2 * l2)))
                    cost -= 0.4 * (self.said(u, "upperarm") + self.said(e, "forearm") + self.said(w, "hand"))
                    if t.kids[w]:  # the hand carries a palm and fingers: not a joint inside them
                        cost += 1.5 * max(0.0, 0.35 - self.reach(w) / l2)
                    forks = self.finger_fork(w)
                    cost -= 0.3 + 0.3 * min(forks, 5) / 5 if forks >= 2 else 0.0
                    if self.names[w].end:
                        cost += 1.0
                    elif not t.kids[w] and any(self.finger_fork(x) >= 2 for x in line[k + 1:a]):
                        cost += 0.6  # past the fork into the fingers
                    if t.kids[w] and forks < 2 and self.reach(w) > 0.25 * l2:
                        cost += 0.3
                    cost += 1.2 * max(0.0, self.reach(w) / l2 - 1.0)
                    if i > 1:
                        cost += 0.3 * (i - 1)
                    cost += min(1.5, abs(np.log(max(self.lateral(u), 1e-3) / 0.11)))  # the shoulder: ~0.11 H out
                    if i > 0 and t.d(line[0], u) > 0.9 * l1:
                        cost += 1.0
                    # among joints at one place (a hand and its coincident finger base) the first
                    if a > 0 and t.d(line[a - 1], w) < 1e-4 * t.ext:
                        cost += 0.2
                    if k > 0 and t.d(line[k - 1], e) < 1e-4 * t.ext:
                        cost += 0.2
                    if best_cost is None or cost < best_cost:
                        best, best_cost = (i, k, a), cost
        if best is None:
            return {"joints": [None, None, None, None], "fit": 0.0, "why": {}, "fingers": {}, "finger_sure": {},
                    "finger_why": {}}
        i, k, a = best
        clav = line[0] if i > 0 else None
        fit = _clip01(1.2 - 0.35 * best_cost)
        ratio = t.d(line[i], line[k]) / max(t.d(line[k], line[a]), 1e-9)
        why = {"clavicle": [Msg("I-SKREC-CLAVICLE")], "upperarm": [Msg("I-SKREC-UPPERARM", ratio=round(ratio, 2))],
               "forearm": [Msg("I-SKREC-ELBOW")],
               "hand": [Msg("I-SKREC-HANDFORK" if self.finger_fork(line[a]) >= 2 else "I-SKREC-HANDEND")]}
        fingers, sure, fwhy = self.fingers(line[a], line[k])
        return {"joints": [clav, line[i], line[k], line[a]], "fit": max(fit, 0.55), "why": why, "fingers": fingers,
                "finger_sure": sure, "finger_why": fwhy}

    def said(self, j: int, part: str) -> float:
        """The name's support for joint j being `part` (no side), clipped to 0..1: a tie-breaker only."""
        return max(0.0, _name_support(self.names[j], part))

    def lateral(self, j: int) -> float:
        """How far a joint is from the trunk's axis (the line up through the hips), as a share of the height."""
        if getattr(self, "_axis", None) is None:
            return 0.11
        o, up, height = self._axis
        v = self.t.pos[j] - o
        return float(np.linalg.norm(v - np.dot(v, up) * up)) / height

    def finger_chains(self, hand: int) -> list[list[int]]:
        """The hand's finger chains: each branch's main line, through a palm joint holding several fingers, without a
        metacarpal near the wrist (its knuckle is the finger's start), a tip without children past three joints, or a
        joint named an end."""
        t = self.t
        roots = []
        for c in t.kids[hand]:
            if c in self.avoid or t.blen[c] + self.reach(c) <= 1e-6 * (t.ext or 1):
                continue
            if self.finger_fork(c) >= 2 and t.d(hand, c) < 0.6 * min((t.d(hand, g) for g in t.kids[c]), default=0):
                roots += [g for g in t.kids[c]]
            else:
                roots.append(c)
        out = []
        for r in roots:
            if self.names[r].helper or self.names[r].prop:
                continue
            chain = [j for j in self.line(r) if not self.names[j].end]
            while chain and self.names[chain[0]].palm:
                chain = chain[1:]
            while len(chain) >= 2 and (t.d(hand, chain[0]) < 0.05 * t.d(hand, chain[1]) or (
                    len(chain) >= 3 and t.d(hand, chain[0]) < 0.35 * t.d(hand, chain[1]))):
                chain = chain[1:]  # a metacarpal near the wrist, or a finger base on the hand itself
            while len(chain) > 3 and not t.kids[chain[-1]]:
                chain = chain[:-1]
            if len(chain) >= 2 or (chain and any(self.names[chain[0]].concepts.get(f, 0) >= 0.6 for f in FINGERS)):
                out.append(chain)
        return out

    def fingers(self, hand: int, elbow: int):
        """Which chain is which finger: the thumb the one branching off furthest from the forearm's line and nearest
        the wrist; the others in their row across the palm from the thumb's side."""
        t, pos = self.t, self.t.pos
        chains = self.finger_chains(hand)
        if len(chains) < 2:
            return {}, {}, {}
        along = _unit(pos[hand] - pos[elbow])
        if along is None:
            return {}, {}, {}

        def away(c):
            # the branch's direction: to its first joint off the wrist (an end site counts: where it points is known)
            off = next((j for j in [*c, *t.subtree(c[-1])] if t.d(j, hand) > 1e-3 * t.ext), c[0])
            v = _unit(pos[off] - pos[hand])
            div = 1.0 - float(np.dot(v, along)) if v is not None else 0.0
            near = t.d(c[0], hand) / max(max(t.d(x[0], hand) for x in chains), 1e-9)
            return div + 0.4 * (1 - near) + 0.1 * (3 - min(len(c), 3)) * 0

        scores = [away(c) for c in chains]
        order = sorted(range(len(chains)), key=lambda i: -scores[i])
        thumb = chains[order[0]]
        gap = scores[order[0]] - scores[order[1]]
        others = [c for c in chains if c is not thumb]
        names_all = [self.names[j] for c in chains for j in c]
        out, sure, why = {"thumb": thumb}, {"thumb": _clip01(0.55 + gap)}, {"thumb": [Msg("I-SKREC-THUMB")]}
        if len(others) >= 2:
            bases = np.array([pos[c[0]] for c in others])
            axis = np.linalg.svd(bases - bases.mean(0))[2][0]
            idx = sorted(range(len(others)), key=lambda i: float(np.dot(bases[i], axis)))
            if np.linalg.norm(bases[idx[-1]] - pos[thumb[0]]) < np.linalg.norm(bases[idx[0]] - pos[thumb[0]]):
                idx.reverse()
            ordered = [others[i] for i in idx]
        else:
            ordered = others
        full = len(ordered) == 4
        for f, c in zip(FINGERS[1:], ordered):
            out[f] = c
            sure[f] = 0.85 if full else 0.4
            why[f] = [Msg("I-SKREC-FINGERROW" if full else "I-SKREC-FINGERROWUNSURE")]
        del names_all
        return out, sure, why

    def head(self, chest: int, arms, up, shoulders):
        """From the chest the branch rising highest: its line is the neck up to the head. The head is where that line
        forks (the eyes, the jaw, a face rig hang from it; of joints at one place the first), or a joint named a head;
        a line ending without a fork ends in the head, unless its last joint sits a skull's height above the
        shoulders — that is the head's top (an end site), the joint before it the head."""
        t, pos = self.t, self.t.pos
        cands = [c for c in t.kids[chest] if c not in arms]
        if not cands:
            return None, [], 0.0, ""
        proj = pos @ up
        c = max(cands, key=lambda c: float(proj[t.subtree(c)].max()) - 0.5 * float(
            np.linalg.norm((pos[c] - pos[chest]) - np.dot(pos[c] - pos[chest], up) * up)))
        if float(proj[t.subtree(c)].max()) < float(proj[chest]) + 0.02 * (t.ext or 1):
            return None, [], 0.0, ""
        ground = float(proj.min())
        sh = float(np.mean(proj[list(shoulders)])) - ground if shoulders else float(proj[chest]) - ground
        line = [c]
        while True:
            ks = [k for k in t.kids[line[-1]] if not self.names[k].end]
            if not ks:
                break
            if len(ks) >= 2 and float(proj[line[-1]]) - ground > 1.04 * sh:
                break
            line.append(max(ks, key=lambda k: float(proj[t.subtree(k)].max()) + 1e-6 * t.size[k]))
        named = [k for k, j in enumerate(line) if self.names[j].concepts.get("head", 0) >= 0.6
                 and not self.names[j].helper]
        if named:
            line = line[: named[0] + 1]
            why = Msg("I-SKREC-HEADNAMED")
        elif len([k for k in t.kids[line[-1]] if not self.names[k].end]) >= 2:
            height = float(np.ptp(proj)) or 1.0
            def off_head(a, b):  # b hangs just off a (a face root in front of the head), or sits on it
                v = pos[b] - pos[a]
                d = float(np.linalg.norm(v))
                return d < 0.01 * t.ext or (d < 0.03 * height and float(np.dot(v, up)) < 0.7 * d)
            while len(line) > 1 and off_head(line[-2], line[-1]):
                line.pop()
            why = Msg("I-SKREC-HEADFORK")
        else:
            why = Msg("I-SKREC-HEADTOP")
            over = (float(proj[line[-1]]) - ground - sh) / max(sh, 1e-9)
            if len(line) > 1 and over > 0.155:
                line.pop()
                why = Msg("I-SKREC-HEADEND")
        head = line[-1]
        neck = line[:-1]
        neck = [j for j in neck if not self.names[j].helper] or neck
        return head, neck, 0.95, why

    def face(self, head, frame: Frame):
        """Eyes: a mirrored pair of small branches (a joint, perhaps with its end) under the head, in front of it and
        not below it, the most forward such pair; the jaw: a joint under the head in the middle, in front, not above
        the eyes, carrying the most (teeth, tongue). Sure only when the head leaves no other choice."""
        t, pos = self.t, self.t.pos
        if head is None or frame.up is None:
            return None, None
        up, fwd, H = frame.up, frame.forward, frame.height or t.ext
        under = [j for j in t.subtree(head) if j != head and t.depth[j] - t.depth[head] <= 3]
        rel = {j: pos[j] - pos[head] for j in under}
        hz = lambda v: v - np.dot(v, up) * up  # noqa: E731
        ahead = lambda j: float(np.dot(rel[j], fwd)) if fwd is not None else 0.0  # noqa: E731
        small = [j for j in under if t.size[j] <= 2 and not self.names[j].end]
        found = []
        for a, b in combinations(small, 2):
            if t.depth[a] != t.depth[b]:
                continue
            ha, hb = hz(rel[a]), hz(rel[b])
            sep = float(np.linalg.norm(ha - hb))
            if not 0.01 * H <= sep <= 0.08 * H:
                continue
            if abs(float(np.dot(rel[a] - rel[b], up))) > 0.01 * H or float(np.dot(rel[a], up)) < -0.02 * H:
                continue
            if abs(float(np.linalg.norm(ha)) - float(np.linalg.norm(hb))) > 0.2 * sep:
                continue
            if frame.left is not None and abs(float(np.dot(ha + hb, frame.left))) > 0.5 * sep:
                continue
            if fwd is not None and min(ahead(a), ahead(b)) < 0.01 * H:
                continue
            found.append(((ahead(a) + ahead(b)) / 2, a, b))
        found.sort(key=lambda f: (-round(f[0] / (0.005 * H)), t.depth[f[1]]))
        simple = len(under) <= 12
        eyes = None
        if found and fwd is not None:
            clear = len(found) == 1 or found[0][0] - found[1][0] > 0.01 * H or t.depth[found[1][1]] > t.depth[found[0][1]]
            eyes = (found[0][1], found[0][2], 0.75 if clear and simple else 0.2)
        jaw = None
        if fwd is not None:
            top = float(np.dot(rel[eyes[0]], up)) if eyes else 0.04 * H
            cands = [j for j in under if t.depth[j] - t.depth[head] <= 2 and not self.names[j].end
                     and (frame.left is None or abs(float(np.dot(rel[j], frame.left))) < 0.01 * H)
                     and ahead(j) > 0.005 * H and float(np.dot(rel[j], up)) <= top
                     and (not eyes or j not in eyes[:2])
                     and all(float(np.linalg.norm(rel[k])) < 0.03 * H for k in t.path(head, j)[:-1])
                     and float(np.linalg.norm(rel[j])) < 0.08 * H]
            if cands:
                cands.sort(key=lambda j: (-t.size[j], float(np.linalg.norm(rel[j]))))
                clear = len(cands) == 1 or t.size[cands[0]] > t.size[cands[1]]
                jaw = (cands[0], 0.6 if clear and simple else 0.2)
        return eyes, jaw

    def lone_hand(self) -> _Reading | None:
        """A skeleton that is one hand (no legs, no trunk): the joint where four or five finger chains part; which
        finger is which as on a body's hand, the side only from the names or the model's declaration."""
        t = self.t
        forks = [j for j in range(t.n) if self.finger_fork(j) >= 4]
        if not forks:
            return None
        hand = min(forks, key=lambda j: t.depth[j])
        if t.size[hand] < 0.6 * t.n:
            return None
        elbow = t.parents[hand] if t.parents[hand] >= 0 else None
        chains = self.finger_chains(hand)
        if elbow is None:  # the wrist is the root: the forearm's direction is wrist -> middle of the knuckles
            ref = np.mean([t.pos[c[0]] for c in chains], 0) if chains else None
            if ref is None:
                return None
            fake = self.t.pos[hand] - (ref - self.t.pos[hand])
            t.pos = np.vstack([t.pos, fake[None]])
            elbow = t.n
            try:
                fingers, sure, why = self.fingers(hand, elbow)
            finally:
                t.pos = t.pos[:-1]
        else:
            fingers, sure, why = self.fingers(hand, elbow)
        return _Reading("structure.hand", {"hand": hand, **fingers}, {"hand": 0.8, **sure},
                        {"hand": [Msg("I-SKREC-LONEHAND")], **why})


# ------------------------------------------------------------------ judging the readings


def _check_part(tree: _Tree, frame: Frame, parts: dict, part: str) -> tuple[float, list[str]] | None:
    """一种读法里一个部位在几何上合不合理（0–1，和不合理的理由）；没有可信的身体坐标系时 None（不评）。"""
    if tree.pos is None or frame.up is None or frame.up_conf < 0.3:
        return None
    pos, up = tree.pos, frame.up
    one = lambda p: _joints_of(parts[p])[0] if parts.get(p) not in (None, []) else None  # noqa: E731
    h = lambda j: float(np.dot(pos[j], up))  # noqa: E731
    tol = 0.02 * (frame.height or tree.ext)
    side, _, name = part.rpartition(".")
    j = one(part)
    if j is None:
        return None
    bad = []
    hips = one("hips")
    if name in ("thigh", "shin", "foot"):
        seq = [one(f"{side}.{p}") for p in ("thigh", "shin", "foot")]
        k = ("thigh", "shin", "foot").index(name)
        if k > 0 and seq[k - 1] is not None and h(seq[k - 1]) < h(j) - tol:
            bad.append(Msg("I-SKREC-NOTDOWN"))
        if k < 2 and seq[k + 1] is not None and h(seq[k + 1]) > h(j) + tol:
            bad.append(Msg("I-SKREC-NOTUP"))
        if name == "foot" and frame.height and h(j) - float((pos @ up).min()) > 0.2 * frame.height:
            bad.append(Msg("I-SKREC-OFFGROUND"))
    if name in ("chest", "neck", "head") and hips is not None and h(j) < h(hips) - tol:
        bad.append(Msg("I-SKREC-BELOWHIPS"))
    if name == "hand" and hips is not None:
        up_arm = one(f"{side}.upperarm")
        if up_arm is not None and h(j) < h(hips) - 0.35 * (frame.height or tree.ext):
            bad.append(Msg("I-SKREC-HANDLOW"))
    if side and frame.left is not None and frame.facing_conf >= 0.5:
        center = pos[hips] if hips is not None else pos.mean(0)
        lateral = float(np.dot(pos[j] - center, frame.left))
        if name in FINGERS and one(f"{side}.hand") is not None:
            lateral = float(np.dot(pos[one(f"{side}.hand")] - center, frame.left))
        if abs(lateral) > 0.005 * tree.ext and (lateral > 0) != (side == "l"):
            bad.append(Msg("I-SKREC-WRONGSIDE"))
    return (1.0 if not bad else max(0.0, 1.0 - 0.6 * len(bad))), bad


def _sided_by_position(tree: _Tree, frame: Frame, parts: dict) -> dict | None:
    """名字读法按位置改左右：面朝方向可信、名字说的左右整组（腿、臂、手指、眼）与位置相反时换过来。没改返回 None。"""
    if frame.left is None or frame.facing_conf < 0.5 or tree.pos is None:
        return None
    one = lambda p: _joints_of(parts[p])[0] if parts.get(p) not in (None, []) else None  # noqa: E731
    hips = one("hips")
    center = tree.pos[hips] if hips is not None else tree.pos.mean(0)
    groups = {"leg": LEG, "arm": ARM, "fingers": FINGERS, "eye": ("eye",)}
    out, changed = dict(parts), False
    for g, names in groups.items():
        votes = 0.0
        for s in SIDES:
            for n in names:
                j = one(f"{s}.{n}")
                if j is None:
                    continue
                if g == "fingers" and one(f"{s}.hand") is not None:
                    j = one(f"{s}.hand")
                lat = float(np.dot(tree.pos[j] - center, frame.left))
                if abs(lat) > 0.005 * tree.ext:
                    votes += 1 if (lat > 0) == (s == "l") else -1
        if votes < 0:
            for n in names:
                a, b = parts.get(f"l.{n}"), parts.get(f"r.{n}")
                out.pop(f"l.{n}", None), out.pop(f"r.{n}", None)
                if a is not None:
                    out[f"r.{n}"] = a
                if b is not None:
                    out[f"l.{n}"] = b
            changed = True
    return out if changed else None


def _family(r: _Reading) -> str:
    return r.name.split(".")[0]


def _same(a, b) -> bool:
    return _joints_of(a) == _joints_of(b)


def _consistent(tree: _Tree, parts: dict) -> dict:
    """去掉互相冲突的部位：一个关节只给一个部位（后判的让位）、链要沿一条线往下、髋要在两腿和躯干之上。"""
    out, taken = {}, {}
    for p in sorted(parts, key=lambda p: REGION_ORDER.get(p, 99)):
        js = _joints_of(parts[p])
        if any(j in taken for j in js):
            continue
        if any(not tree.is_below(b, a) or a == b for a, b in zip(js, js[1:])):
            continue
        out[p] = parts[p]
        for j in js:
            taken[j] = p
    hips = out.get("hips")
    if hips is not None:
        for p in [p for p in out if p != "hips" and p not in ("jaw",)]:
            if p in ("l.eye", "r.eye"):
                continue
            if not all(tree.is_below(j, hips) for j in _joints_of(out[p])):
                del out[p]
    return out


REGION_ORDER = {p: i for i, p in enumerate(
    ("hips", "chest", "head", "spine", "neck", *(f"{s}.{p}" for s in SIDES for p in LEG),
     *(f"{s}.{p}" for s in SIDES for p in ARM), *(f"{s}.{f}" for s in SIDES for f in FINGERS), "jaw", "l.eye",
     "r.eye"))}


def _judge(tree: _Tree, names: list[_Name], readings: list[_Reading], has_rest: bool) -> Recognition:
    """几种读法按区域打分取最好的，再算每个部位的置信度与依据。"""
    frames = {r.name: (r.frame or _frame(tree, r.parts)) for r in readings}
    scores: dict[str, dict[str, float]] = {}
    checks: dict[str, dict[str, tuple]] = {}
    for r in readings:
        fr = frames[r.name]
        checks[r.name] = {p: _check_part(tree, fr, r.parts, p) for p in r.parts}
    chosen: dict[str, str] = {}
    parts: dict = {}
    for region, members in REGIONS:
        best, best_score = None, 0.0
        for r in readings:
            total = 0.0
            for p in members:
                if p not in r.parts:
                    continue
                js = _joints_of(r.parts[p])
                geo = checks[r.name][p]
                g = geo[0] if geo is not None else 0.5
                nm = float(np.mean([_name_support(names[j], p) for j in js]))
                agree = sum(1 for o in readings if _family(o) != _family(r) and p in o.parts and _same(o.parts[p], r.parts[p]))
                total += 1.0 + 1.5 * g + 1.0 * nm + 0.6 * min(agree, 2) + 0.2 * r.sure.get(p, 0.0)
                if geo is not None and geo[0] < 0.5:
                    total -= 2.0
            if total > best_score + 1e-9:
                best, best_score = r, total
        if best is not None:
            chosen[region] = best.name
            for p in members:
                if p in best.parts:
                    parts[p] = best.parts[p]
    by_name = {r.name: r for r in readings}
    region_of = {p: region for region, members in REGIONS for p in members}
    parts = _consistent(tree, parts)
    confidence, evidence, rejected, final = {}, {}, {}, {}
    for p in sorted(parts, key=lambda p: REGION_ORDER.get(p, 99)):
        src = by_name[chosen[region_of[p]]]
        js = _joints_of(parts[p])
        nm = float(np.mean([_name_support(names[j], p) for j in js]))
        geo = checks[src.name].get(p)
        agree = [o.name for o in readings if _family(o) != _family(src) and p in o.parts and _same(o.parts[p], parts[p])]
        ev = []
        strengths = []
        if src.name.startswith("names"):
            strengths.append(0.6)
            ev += src.said.get(p, [])
        else:
            strengths.append(src.sure.get(p, 0.0))
            ev += src.said.get(p, [])
        if nm > 0:
            strengths.append(0.6 * nm)
            word = next((names[j].said.get(p.rpartition(".")[2]) for j in js if p.rpartition(".")[2] in names[j].said), None)
            if word and not any(e.code in ("I-SKREC-NAMES", "I-SKREC-NAMEWORD") for e in ev):
                ev.append(Msg("I-SKREC-NAMEWORD", word=word))
        for o in agree:
            strengths.append(0.5 if by_name[o].name.startswith("names") else max(0.4, by_name[o].sure.get(p, 0.0)))
        if agree:
            ev.append(Msg("I-SKREC-AGREE"))
        if geo is not None and geo[0] >= 1.0:
            strengths.append(0.3)
        c = 1.0 - float(np.prod([1.0 - _clip01(s) for s in strengths]))
        if geo is not None:
            if geo[0] < 1.0:
                c *= 0.35 + 0.65 * geo[0]
                ev += list(geo[1])
        if nm < 0:
            c *= 1.0 + 0.35 * nm
            ev.append(Msg("I-SKREC-NAMECONTRA" if not src.name.startswith("names") else "I-SKREC-NAMECONFLICT"))
        confidence[p] = c
        evidence[p] = list({m.text: m for m in ev}.values())
        if c >= THRESHOLD:
            final[p] = parts[p]
        else:
            rejected[p] = parts[p]
            evidence[p].append(Msg("I-SKREC-LOW", confidence=round(c, 2), threshold=THRESHOLD))
    final = _consistent(tree, final)
    frame = _frame(tree, final) if has_rest else Frame()
    if has_rest and frame.up is None and readings:
        frame = frames.get(chosen.get("trunk", ""), Frame()) or Frame()
    return Recognition(final, confidence, evidence, [], frame, rejected, chosen)


def _roles(tree: _Tree, names: list[_Name], parts: dict, frame: Frame, inert=frozenset()) -> list[tuple[str, str]]:
    """每个关节的角色（ROLES）和理由（Msg）。辅助：名字说的扭转 / 辅助 / 控制骨，或与一个部位关节重合、躺在一根
    肢体骨上而自己不是部位、下面也没有部位；道具：名字说的道具 / 武器 / 挂点；末端：没有子关节、名字说是末端
    或接在部位末端；不驱动任何顶点的（`inert`，_inert）：末端或辅助；面部：头下的其余关节。"""
    part_of = {j: p for p, w in parts.items() for j in _joints_of(w)}
    under_part = set()
    for j in part_of:
        under_part.update(tree.ancestors(j))
    head = parts.get("head")
    out: list[tuple[str, str]] = []
    pos = tree.pos
    eps = 0.004 * (tree.ext or 1) if pos is not None else 0
    part_pts = list(part_of) if pos is not None else []
    for j in range(tree.n):
        nm = names[j]
        if j in part_of:
            p = part_of[j]
            region = p.rpartition(".")[2]
            role = ("trunk" if p in TRUNK else "finger" if region in FINGERS else "face" if p in FACE else "limb")
            out.append((role, Msg("I-SKREC-ROLEPART", part=part_label(p))))
            continue
        if nm.prop:
            out.append(("prop", Msg("I-SKREC-ROLEPROP")))
            continue
        if nm.helper and j not in under_part:
            out.append(("helper", Msg("I-SKREC-ROLEHELPER")))
            continue
        if not tree.kids[j] and (nm.end or tree.parents[j] in part_of and len(tree.kids[tree.parents[j]]) == 1):
            out.append(("end", Msg("I-SKREC-ROLEEND")))
            continue
        if pos is not None and j not in under_part:
            twin = next((k for k in part_pts if k != j and float(np.linalg.norm(pos[k] - pos[j])) <= eps
                         and not tree.is_below(k, j)), None)
            if twin is not None:
                out.append(("helper", Msg("I-SKREC-ROLETWIN", part=part_label(part_of[twin]))))
                continue
            p = tree.parents[j]
            if p in part_of and not any(k in part_of for k in tree.subtree(j)):
                nxt = next((c for c in tree.kids[p] if c in part_of), None)
                if nxt is not None and part_of[p].rpartition(".")[2] in ARM + LEG:
                    a, b = pos[p], pos[nxt]
                    seg = b - a
                    L = float(np.dot(seg, seg))
                    if L > 0:
                        tpar = float(np.dot(pos[j] - a, seg)) / L
                        dist = float(np.linalg.norm(a + np.clip(tpar, 0, 1) * seg - pos[j]))
                        if -0.05 <= tpar <= 1.05 and dist <= 0.012 * tree.ext and all(
                                float(np.linalg.norm(pos[k] - a - np.clip(float(np.dot(pos[k] - a, seg)) / L, 0, 1) * seg))
                                <= 0.012 * tree.ext for k in tree.subtree(j)):
                            out.append(("helper", Msg("I-SKREC-ROLEONBONE", part=part_label(part_of[p]))))
                            continue
        if j in inert:
            out.append(("end", Msg("I-SKREC-ROLEUNWEIGHTEDEND")) if not tree.kids[j] else
                       ("helper", Msg("I-SKREC-ROLEUNWEIGHTED")))
            continue
        if head is not None and tree.is_below(j, head):
            out.append(("face", Msg("I-SKREC-ROLEFACE")))
            continue
        if nm.helper:
            out.append(("helper", Msg("I-SKREC-ROLEHELPER")))
            continue
        out.append(("unknown", None))
    return out


# ------------------------------------------------------------------ the entry


def frame_of(parents, positions, parts: dict) -> Frame:
    """一副姿势（关节位置 [J,3]）按给定的部位（部位 -> 关节序号，链为列表）估的身体坐标系：识别时每种读法用的同一个
    函数（_frame）。「自动姿态」（nodes/kit/retarget_prepare.py reference_axes）按它定上方向。"""
    tree = _Tree(parents, np.asarray(positions, np.float64))
    return _frame(tree, {p: (list(v) if len(v) > 1 or p in CHAINS else v[0]) if isinstance(v, (list, tuple)) else v
                         for p, v in parts.items() if v is not None and (not isinstance(v, (list, tuple)) or v)})


_CACHE: OrderedDict = OrderedDict()
_CACHE_SIZE = 64
_LOCK = threading.Lock()


def fingerprint(names, parents, rest=None, lone_side=None) -> str:
    """一副骨架本身的指纹（缓存键）：名字、父子、静止位置（相对尺度取 6 位有效数字）、单手声明、引擎版本。"""
    parts = [f"v{VERSION}|{lone_side}|", "\x00".join(str(n) for n in names).encode("utf-8", "surrogatepass"),
             np.asarray([int(p) for p in parents], np.int64).tobytes()]
    if rest is not None:
        r = np.asarray(rest, np.float64)
        scale = float(np.abs(r).max()) or 1.0
        parts.append(np.round(r / scale, 6).tobytes())
    return sha256(*parts)  # the one digest (io/digest.py)


def recognize(names, parents, rest_world=None, *, weights=None, lone_side: str | None = None) -> Recognition:
    """识别一副骨架（模块说明）。`rest_world` [J,3] 静止姿态的世界位置（或 [J,4,4] 取平移）；没有时只有名字读法
    （层级 + 名字，与原来的 guess 相同）。`weights`：可选，哪些关节带蒙皮权重——关节序号的集合（data/animation.py
    Rig.weighted），或每个关节一个数 / 一行数（[J] / [J,V]，非零即有）；None = 不知道（只有骨架、没有网格：BVH、
    纯骨架 FBX），每个关节都读。给了时第一步排除不驱动任何顶点的关节（_inert）：它们不当任何部位的候选。
    `lone_side`：只有一只手、名字不说哪只手的骨架是哪只手（"l" / "r"）。
    结果按指纹缓存在进程里；返回的是共享的对象，调用方不要改它（joints.guess 等给的是拷贝）。"""
    names = [str(n) for n in names]
    rest = None
    if rest_world is not None:
        rest = np.asarray(rest_world, np.float64)
        if rest.ndim == 3:
            rest = rest[:, :3, 3]
        if rest.shape != (len(names), 3) or not np.all(np.isfinite(rest)):
            rest = None
    weighted = None
    if isinstance(weights, (set, frozenset)):
        weighted = frozenset(int(j) for j in weights if 0 <= int(j) < len(names))
    elif weights is not None:
        w = np.asarray(weights)
        weighted = frozenset(int(j) for j in np.nonzero(w.reshape(len(names), -1).any(axis=1))[0]) \
            if w.size and len(w) == len(names) else None
    key = fingerprint(names, parents, rest, lone_side) + (
        "|w" + sha256(str(sorted(weighted))) if weighted is not None else "")
    with _LOCK:
        got = _CACHE.get(key)
        if got is not None:
            _CACHE.move_to_end(key)
            return got
    rec = _recognize(names, parents, rest, weighted, lone_side)
    with _LOCK:
        _CACHE[key] = rec
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return rec


def _inert(tree: _Tree, weighted) -> frozenset:
    """不驱动任何顶点的关节：它自己和它下面的关节都没有蒙皮权重（转动它，网格一点不动：道具、挂点、控制骨、末端点、
    没蒙皮的面部骨）。按子树而不是按关节自己：CC / AccuRIG 的大腿、小腿、上臂、前臂本身不带权重（权重刷在它们下面的
    扭转骨上），转它们照样带动网格，是部位。没有权重信息（`weighted` 为 None）或一个带权重的关节都没有时为空：
    不排除任何关节。"""
    if not weighted:
        return frozenset()
    drives = set()
    for j in weighted:
        drives.update(tree.ancestors(j))
    return frozenset(j for j in range(tree.n) if j not in drives)


def _without(reading: _Reading, inert: frozenset, dropped: dict) -> None:
    """一种读法去掉落在不驱动顶点的关节上的部位（链去掉这几节，整条都落在上面就整条去掉）；去掉的记在 `dropped`
    （部位 -> 关节），给依据用。"""
    for p in list(reading.parts):
        js = _joints_of(reading.parts[p])
        keep = [j for j in js if j not in inert]
        if len(keep) == len(js):
            continue
        dropped.setdefault(p, [j for j in js if j in inert])
        if keep and isinstance(reading.parts[p], list):
            reading.parts[p] = keep
        else:
            del reading.parts[p]
            reading.sure.pop(p, None)
            reading.said.pop(p, None)


def _excluded(rec: Recognition, dropped: dict, names: list[str]) -> Recognition:
    """读法给过、却因为落在不驱动顶点的关节上而去掉的部位，最后也没有别的关节当它的：记为没分配，依据写明。"""
    for p, js in dropped.items():
        if p in rec.parts:
            continue
        rec.rejected.setdefault(p, js if len(js) > 1 or p in CHAINS else js[0])
        rec.confidence.setdefault(p, 0.0)
        rec.evidence.setdefault(p, []).append(Msg("I-SKREC-INERT", joint=names[js[0]]))
    return rec


def _recognize(names, parents, rest, weighted, lone_side) -> Recognition:
    from .joints import _by_names

    tree = _Tree(parents, rest)
    read = [_read_name(n) for n in names]
    # 第一步：有蒙皮权重时，不驱动任何顶点的关节不参与对应（不当任何部位的候选）；它们照样在树里、位置照用（骨架的
    # 朝向、头顶末端点、肢体的走向都还看得到它们）
    inert = _inert(tree, weighted)
    dropped: dict = {}
    readings = []
    by_names = {p: v for p, v in _by_names(names, np.asarray(tree.parents), lone_side).items() if v != []}
    if by_names:
        r = _Reading("names", by_names, {p: 0.6 for p in by_names},
                     {p: [Msg("I-SKREC-NAMES")] for p in by_names})
        _without(r, inert, dropped)
        by_names = r.parts
        if by_names:
            readings.append(r)
    if rest is not None and tree.ext > 0:
        structure = _Structure(tree, read, inert).read()
        for r in structure:
            _without(r, inert, dropped)
        readings += structure
        if by_names:
            frame = _frame(tree, by_names)
            if frame.left is None:
                frame = next((s.frame for s in structure if s.frame is not None and s.frame.left is not None), frame)
            fixed = _sided_by_position(tree, frame, by_names)
            if fixed is not None:
                said = {p: [Msg("I-SKREC-NAMES"), Msg("I-SKREC-NAMESSIDEFIXED")] for p in fixed}
                readings.append(_Reading("names.sided", fixed, {p: 0.6 for p in fixed}, said, frame))
        for r in readings:
            if r.name == "structure.hand":
                side = lone_side
                if side not in SIDES:  # the names' sides, when they agree
                    said = {read[j].side for v in r.parts.values() for j in _joints_of(v)} - {None}
                    side = said.pop() if len(said) == 1 else None
                if side not in SIDES:
                    continue
                lone_side_ = side
                r.parts = {(f"{lone_side_}.{p}"): v for p, v in r.parts.items()}
                r.sure = {(f"{lone_side_}.{p}"): v for p, v in r.sure.items()}
                r.said = {(f"{lone_side_}.{p}"): v for p, v in r.said.items()}
        readings = [r for r in readings if all("." in p or p in TRUNK or p in FACE for p in r.parts)]
    if not readings:
        return _excluded(Recognition({}, {}, {}, _roles(tree, read, {}, Frame(), inert), Frame()), dropped, names)
    if rest is None:
        r = readings[0]
        conf = {p: 0.6 for p in r.parts}
        rec = Recognition(dict(r.parts), conf, {p: list(r.said.get(p, [])) for p in r.parts}, [], Frame(), {},
                          {region: r.name for region, _ in REGIONS})
        rec.roles = _roles(tree, read, rec.parts, rec.frame, inert)
        return _excluded(rec, dropped, names)
    rec = _judge(tree, read, readings, True)
    rec.roles = _roles(tree, read, rec.parts, rec.frame, inert)
    return _excluded(rec, dropped, names)
