"""Which of a rig's joints is which body part, guessed from the joint names and the hierarchy (like Maya's HumanIK
characterization): the in-betweening models' skeletons are matched to a production rig through these parts.

Names only have to say what a hand, a foot, the head and the hips are, which finger a finger joint belongs to, and
which side (Maya / HumanIK, Mixamo, Unreal, SMPL, 3ds Max Biped, Blender-style suffixes, namespaces such as
mixamorig:); the rest follows from the hierarchy: the leg is the path from the hips to the foot, the arm the path from
where it leaves the spine to the hand, the chest is where the arms leave the spine, a finger is the chain of that
finger's joints under a hand. Twist, roll and helper joints in a chain are skipped.

The second half of the file is what a bone is called in CG (cg_names): the Mixamo names a delivered skeleton carries.
Its companion is the CG joint orientation in data/skeleton.py.
"""

from __future__ import annotations

import re

import numpy as np

SIDES = ("l", "r")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # the five finger chains of a hand, starting from the thumb
LIMB_PARTS = ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe") + FINGERS + ("eye",)
# parts that are several joints (a model may have fewer or more than the rig): the spine, the neck, and every finger
CHAINS = ("spine", "neck") + tuple(f"{s}.{f}" for s in SIDES for f in FINGERS)
PARTS = ("hips", "spine", "chest", "neck", "head", "jaw") + tuple(f"{s}.{p}" for s in SIDES for p in LIMB_PARTS)
PART_LABELS = {"hips": "髋", "spine": "脊柱", "chest": "胸", "neck": "颈", "head": "头", "jaw": "下巴", "eye": "眼",
               "clavicle": "锁骨", "upperarm": "上臂", "forearm": "前臂", "hand": "手",
               "thigh": "大腿", "shin": "小腿", "foot": "脚", "toe": "脚趾",
               "thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小指"}


def part_label(part: str) -> str:
    """The display name of a body part: "l.thigh" -> 左大腿, "hips" -> 髋. This is the project's only table of body
    part display names (read by the in-betweening family's joint mapping table and by 「骨架动画转 SMPL」's pairing notes)."""
    side, _, name = part.rpartition(".")
    return {"l": "左", "r": "右", "": ""}[side] + PART_LABELS[name]


# a rig's own prefix, only when a separator or a capital letter follows it ("rig_Spine", "DEF-thigh"; never "right")
_PREFIX = re.compile(r"^(?i:mixamorig\d*|bip\d*|biped\d*|cc_base|def|org|mch|jnt|jt|bn|bone|rig|sk|character\d*)(?=[_\-. ]|[A-Z])[_\-. ]?")
_HELPER = {"twist", "roll", "bend", "helper", "ribbon", "corrective", "offset", "aux", "ik", "fk", "pole", "ctrl"}
_END = {"end", "top", "nub", "site", "tip", "effector"}
_FINGER = {"thumb", "index", "middle", "ring", "pinky", "pinkie", "finger", "fingers", "carpal"}  # words marking a joint as part of a finger
# Which of the five fingers a joint belongs to. This differs from _FINGER, which also accepts generic words such as
# "finger"; only specific finger names are listed here, with the variants found in models and rigs (the index finger
# also appears as fore / point, the little finger as little)
_FINGER_WORDS = {"thumb": {"thumb"}, "index": {"index", "fore", "point"}, "middle": {"middle", "mid"},
                 "ring": {"ring"}, "pinky": {"pinky", "pinkie", "little"}}


def tokens(name: str) -> list[str]:
    """A joint name as lower-case words: namespaces and rig prefixes dropped, camelCase, digits and separators split
    ("mixamorig:LeftUpLeg" -> left up leg, "thigh_l" -> thigh l, "Bip01 L Thigh" -> l thigh)."""
    name = re.split(r"[:|]", name)[-1]
    name = _PREFIX.sub("", name) or name
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)
    return [p.lower() for p in parts]


def side(words: list[str]) -> str | None:
    left = any(w in ("left", "l", "lf", "lft") for w in words)
    right = any(w in ("right", "r", "rt", "rgt") for w in words)
    return "l" if left and not right else "r" if right and not left else None


def _kind(words: list[str]) -> str | None:
    """What a single joint's name says it is: hand, foot, head, jaw, eye, hips, or None."""
    w = set(words)
    if w & _HELPER or w & _END:
        return None
    if w & {"hand", "wrist"} and not w & _FINGER:
        return "hand"
    if w & {"foot", "ankle"} and not w & {"toe", "toes", "ball"}:
        return "foot"
    if "head" in w:
        return "head"
    if "jaw" in w:
        return "jaw"
    if w & {"eye", "eyes"} and not w & {"lid", "brow", "ball"}:  # the eyeball itself, not eyelids or brows
        return "eye"
    if w & {"hips", "pelvis"} or (w == {"hip"}) or (w & {"hip"} and side(words) is None):
        return "hips"
    return None


def _ancestors(j: int, parents) -> list[int]:
    out = []
    while j >= 0:
        out.append(j)
        j = int(parents[j])
    return out  # j first, the root last


def _path(top: int, bottom: int, parents) -> list[int]:
    """Joints from below `top` down to `bottom` (top excluded), or [] when `top` is not above `bottom`."""
    up = _ancestors(bottom, parents)
    return list(reversed(up[: up.index(top)])) if top in up else []


def guess(names: list[str], parents, lone_side: str | None = None) -> dict:
    """The body parts of a rig: {"hips": j, "spine": [j...], "chest": j, "neck": [j...], "head": j, "l.hand": j,
    "l.index": [j, j, j], ...} (joint indices; a part not found is left out; a chain part is a list).

    `lone_side` is for a skeleton that is one hand or one side only and whose joint names never say which
    ("wrist", "index1", ...; the model records the side beside its joints, as MANO does): "l" or "r"."""
    parents = np.asarray(parents)
    words = [tokens(n) for n in names]
    kinds = [_kind(w) for w in words]
    sides = [side(w) for w in words]
    depth = [len(_ancestors(j, parents)) for j in range(len(names))]

    def top_most(kind: str, s: str | None = None) -> int | None:
        found = [j for j in range(len(names)) if kinds[j] == kind and (s is None or sides[j] == s)]
        return min(found, key=lambda j: depth[j]) if found else None

    def plain(path: list[int]) -> list[int]:
        return [j for j in path if not set(words[j]) & _HELPER]

    out: dict = {}
    feet = {s: top_most("foot", s) for s in SIDES}
    hips = top_most("hips")
    if hips is None and all(f is not None for f in feet.values()):  # where the legs meet
        common = [j for j in _ancestors(feet["l"], parents) if j in _ancestors(feet["r"], parents)]
        hips = common[0] if common else None

    def loose() -> dict:
        """Parts recognised by name alone, independent of the body hierarchy: jaw, eyes, hands and fingers. These are
        also recognised on skeletons without hips (HaMeR solves one hand, SMIRK a face only)."""
        if (jaw := top_most("jaw")) is not None:
            out["jaw"] = jaw
        for s in SIDES:
            if (eye := top_most("eye", s)) is not None:
                out[f"{s}.eye"] = eye
        _hands_and_fingers(out, names, parents, words, depth, lone_side, top_most)
        return out

    if hips is None:
        return loose()
    out["hips"] = hips
    for s, foot in feet.items():
        leg = plain(_path(hips, foot, parents)) if foot is not None else []
        if len(leg) >= 3:
            out[f"{s}.thigh"], out[f"{s}.shin"], out[f"{s}.foot"] = leg[-3:]
            below = [c for c in range(len(names)) if foot in _ancestors(c, parents)[1:] and not set(words[c]) & (_END | _HELPER)]
            named = [c for c in below if set(words[c]) & {"toe", "toes", "ball"}]  # past any ankle sub-joints (MHR)
            toes = named or [c for c in below if parents[c] == foot]
            if toes:
                out[f"{s}.toe"] = min(toes, key=lambda c: depth[c])
    head = top_most("head")
    spine = plain(_path(hips, head, parents))[:-1] if head is not None else []  # between the hips and the head
    if head is not None:
        out["head"] = head
    for s in SIDES:
        hand = top_most("hand", s)
        if hand is None or not spine:
            continue
        branch = next((j for j in _ancestors(hand, parents) if j in spine or j == hips), None)
        arm = plain(_path(branch, hand, parents)) if branch is not None else []
        if len(arm) >= 3:
            out[f"{s}.upperarm"], out[f"{s}.forearm"], out[f"{s}.hand"] = arm[-3:]
            if len(arm) >= 4:
                out[f"{s}.clavicle"] = arm[-4]
            if branch in spine and "chest" not in out:
                out["chest"] = branch
    if "chest" in out:
        c = spine.index(out["chest"])
        out["spine"], out["neck"] = spine[:c], spine[c + 1:]
    elif spine:
        out["spine"], out["neck"] = spine, []
    return loose()


def _hands_and_fingers(out: dict, names, parents, words, depth, lone_side, top_most) -> None:
    """The five finger chains of each hand. Hands are taken from those already recognised in `out`; for a skeleton with
    a single hand whose names do not state the side (MANO), the model's declared `lone_side` decides. Fingers are
    recognised by name (thumb / index / middle / ring / pinky) and ordered into chains by hierarchy, skipping helper and
    end joints, the same method as for the rest of the body, without model-specific special cases."""
    hands = {s: out[f"{s}.hand"] for s in SIDES if f"{s}.hand" in out}
    if not hands and lone_side in SIDES and (lone := top_most("hand")) is not None:
        hands[lone_side] = lone
        out[f"{lone_side}.hand"] = lone
    for s, hand in hands.items():
        below = [j for j in range(len(names)) if hand in _ancestors(j, parents)[1:]
                 and not set(words[j]) & (_HELPER | _END)]
        for finger, hints in _FINGER_WORDS.items():
            chain = sorted((j for j in below if set(words[j]) & hints), key=lambda j: depth[j])
            if chain:
                out[f"{s}.{finger}"] = chain


def spread(count: int, joints: list[int]) -> list[int | None]:
    """`count` model joints of a chain over the rig's chain `joints`, by their place along it: the first on the first,
    the last on the last; model joints left over (the rig's chain is shorter) get None."""
    if not joints:
        return [None] * count
    picks = [0] if count == 1 else [int(round(k * (len(joints) - 1) / (count - 1))) for k in range(count)]
    out, used = [], set()
    for i in picks:
        out.append(None if i in used else joints[i])
        used.add(i)
    return out


def auto_mapping(rows: list[tuple[str, str]], names: list[str], parents) -> dict[str, str | None]:
    """Model joints [(name, part)] -> the rig joint each goes to (a name, or None). Rows of a chain part (CHAINS: the
    spine, the neck, each finger) are the model's chain in order."""
    parts = guess(names, parents)
    out: dict[str, str | None] = {}
    for part in CHAINS:
        chain = [n for n, p in rows if p == part]
        for n, j in zip(chain, spread(len(chain), parts.get(part, []))):
            out[n] = None if j is None else names[j]
    for n, p in rows:
        if p not in CHAINS:
            j = parts.get(p)
            out[n] = None if j is None else names[j]
    return out


# ------------------------------------------------------------------ CG bone names (Mixamo convention)

# The CG name of each bone. ML body models name joints while CG names bones, offset by one: SMPL's left_shoulder is
# the upper arm, left_hip the thigh, left_foot the toes. Animators would misread SMPL names and HumanIK would not match
# them, so names are converted to the Mixamo convention before delivery; it is the common vocabulary for retargeting
# between DCCs (recognised by Maya HumanIK, MotionBuilder, Blender Rigify and UE Retarget).
# No mixamorig: prefix: it is Mixamo's own namespace and would only interfere.
SIDE_WORD = {"l": "Left", "r": "Right"}
CG_NAMES = {
    "hips": "Hips", "chest": "Spine2", "neck": "Neck", "head": "Head", "jaw": "Jaw",
    **{f"{s}.{p}": f"{SIDE_WORD[s]}{n}" for s in SIDES for p, n in (
        ("clavicle", "Shoulder"), ("upperarm", "Arm"), ("forearm", "ForeArm"), ("hand", "Hand"),
        ("thigh", "UpLeg"), ("shin", "Leg"), ("foot", "Foot"), ("toe", "ToeBase"), ("eye", "Eye"))},
}
# multi-joint parts: the spine is Spine / Spine1 (the chest takes Spine2); extra neck joints are Neck1, Neck2, ...
CG_CHAINS = {"spine": ("Spine", "Spine1"), "neck": ("Neck", "Neck1", "Neck2")}


def cg_chain_name(part: str, k: int) -> str:
    """The CG bone name of joint k (from 0) of a chain. Fingers are LeftHandIndex1, LeftHandIndex2, ... (Mixamo numbers
    from 1 and appends the finger name to the hand); spine and neck follow CG_CHAINS, numbered on beyond the table."""
    s, _, name = part.rpartition(".")
    if name in FINGERS:
        return f"{SIDE_WORD[s]}Hand{name.title()}{k + 1}"
    fixed = CG_CHAINS.get(part, ())
    return fixed[k] if k < len(fixed) else f"{(fixed or (part.title(),))[0]}{k}"


def cg_names(names: list[str], parents, lone_side: str | None = None) -> list[str]:
    """The CG name of every bone of this skeleton: body parts recognised by guess() get Mixamo names; unrecognised
    joints (the model's own helpers) keep their names. Returns a list as long as `names`, with duplicates numbered.

    `lone_side`: see guess(); for a single-hand model whose names do not state the side, it decides left or right."""
    parts = guess(names, parents, lone_side)
    out = list(names)
    for part, where in parts.items():
        if isinstance(where, list):
            for k, j in enumerate(where):
                out[j] = cg_chain_name(part, k)
        elif part in CG_NAMES:
            out[where] = CG_NAMES[part]
    # an unrecognised end joint without children (the palm joint after SMPL's wrist) is named "<parent>_End" by CG
    # convention (as Mixamo's LeftToe_End, HeadTop_End). This applies only when it is the parent's single end child;
    # when a bone has several end children (the head with the jaw and two eyes), none of them is the head's end and
    # each keeps the model's own name
    children: dict[int, list[int]] = {}
    for i, up in enumerate(parents):
        children.setdefault(int(up), []).append(i)
    for i, n in enumerate(out):
        if n != names[i] or i in children:
            continue
        up = int(parents[i])
        if up >= 0 and out[up] != names[up] and len(children.get(up, ())) == 1:
            out[i] = f"{out[up]}_End"
    seen: dict[str, int] = {}
    for i, n in enumerate(out):  # number duplicate names (a model joint colliding with a standard name): bone names must be unique in a skeleton
        if n in seen:
            seen[n] += 1
            out[i] = f"{n}{seen[n]}"
        else:
            seen[n] = 0
    return out
