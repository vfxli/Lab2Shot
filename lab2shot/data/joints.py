"""Which of a rig's joints is which body part, guessed from the joint names and the hierarchy (like Maya's HumanIK
characterization): the in-betweening models' skeletons are matched to a production rig through these parts.

Names only have to say what a hand, a foot, the head and the hips are, which finger a finger joint belongs to, and
which side (Maya / HumanIK, Mixamo, Unreal, SMPL, 3ds Max Biped, Blender-style suffixes, namespaces such as
mixamorig:); the rest follows from the hierarchy: the leg is the path from the hips to the foot, the arm the path from
where it leaves the spine to the hand, the chest is where the arms leave the spine, a finger is the chain of that
finger's joints under a hand. Twist, roll and helper joints in a chain are skipped.

The second half of the file is what a bone is called in CG (cg_names): the names of the project's naming convention
(data/bone_names.py: Maya HumanIK / Mixamo) a delivered skeleton carries. Its companion is the CG joint orientation in
data/skeleton.py.
"""

from __future__ import annotations

from .. import i18n

import re

import numpy as np

SIDES = ("l", "r")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # the five finger chains of a hand, starting from the thumb
LIMB_PARTS = ("clavicle", "upperarm", "forearm", "hand", "thigh", "shin", "foot", "toe") + FINGERS + ("eye",)
# parts that are several joints (a model may have fewer or more than the rig): the spine, the neck, and every finger
CHAINS = ("spine", "neck") + tuple(f"{s}.{f}" for s in SIDES for f in FINGERS)
PARTS = ("hips", "spine", "chest", "neck", "head", "jaw") + tuple(f"{s}.{p}" for s in SIDES for p in LIMB_PARTS)
# each part's words: joints.part.<name> (part_label); one table of display names, in the catalogues
def part_label(part: str) -> str:
    """The display name of a body part: "l.thigh" -> 左大腿, "hips" -> 髋. This is the project's only table of body
    part display names, read by the 「对应关系」 editor of 「动作重定向」 and of the skeleton-motion nodes (the parts'
    labels come from the server: part_rows; 「StableMotion 动捕清理」, 「Kimodo 动作生成」, 「Two-stage Transformer 动作补帧」,
    「UnderPressure 脚滑清理」, nodes/kit/rig.py part_labels), and by 「动作重定向」's notes on what it paired
    (data/smpl.py pairing_notes)."""
    from .. import i18n

    side, _, name = part.rpartition(".")

    def said() -> str:
        word = i18n.t(f"joints.part.{name}")
        return i18n.t("joints.sided", side=i18n.t(f"joints.side.{side}"), part=word) if side else word

    return i18n.Both.of(said)  # every language: a message naming it reads in its reader's


# The body regions a mapping is shown and reported in: the 「对应关系」 editor groups its slots by them, the notes of
# 「动作重定向」 write one line per region (data/smpl.py pairing_notes). Every part of PARTS belongs to one region, so a
# part added above cannot be left out there.
REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("trunk", ("hips", "spine", "chest", "neck", "head")),
    ("face", ("jaw", "l.eye", "r.eye")),
    ("l_arm", ("l.clavicle", "l.upperarm", "l.forearm", "l.hand")),
    ("r_arm", ("r.clavicle", "r.upperarm", "r.forearm", "r.hand")),
    ("l_leg", ("l.thigh", "l.shin", "l.foot", "l.toe")),
    ("r_leg", ("r.thigh", "r.shin", "r.foot", "r.toe")),
    ("l_hand", ("l.thumb", "l.index", "l.middle", "l.ring", "l.pinky")),
    ("r_hand", ("r.thumb", "r.index", "r.middle", "r.ring", "r.pinky")),
)
REGION_OF = {part: region for region, parts in REGIONS for part in parts}


def region_label(region: str) -> str:
    """A region's name in the language now (joints.region.<id>)."""
    from .. import i18n

    return i18n.Word(f"joints.region.{region}")
# Both thighs and shins: what a leg length is measured on (「动作重定向」's scale by legs and its forward direction,
# the motion models' legs, motion.Retarget.align) — the parts every retarget and every rig model needs besides the hips
LEGS = ("l.thigh", "l.shin", "r.thigh", "r.shin")
# the parts where a body's trunk ends: it runs from the hips up to where these part (motion.trunk_of: the chest)
LANDMARKS = ("l.hand", "r.hand", "head")
# the limb bones, each (part, the part it runs to): the base pose's T (kit/retarget.py t_posed) goes over this list
LIMB_BONES = tuple((f"{s}.{a}", f"{s}.{b}") for s in SIDES for a, b in (
    ("clavicle", "upperarm"), ("upperarm", "forearm"), ("forearm", "hand"), ("thigh", "shin"), ("shin", "foot"),
    ("foot", "toe")))
# What a chain part hangs from and ends in (「动作重定向」 spreads a chain's bend between the two; a finger ends free)
CHAIN_ENDS = {"spine": ("hips", "chest"), "neck": ("chest", "head"),
              **{f"{s}.{f}": (f"{s}.hand", None) for s in SIDES for f in FINGERS}}


# a rig's own prefix ("rig_Spine", "DEF-thigh", "mixamorigHips", "Bip01Spine"): stripped only at a word boundary
# (_strip_prefix), never out of a word ("right", "RIGHT_FOOT", "SKULL", "DEFORM")
_PREFIX = re.compile(r"^(?i:mixamorig\d*|biped\d*|bip\d*|cc_base|def|org|mch|jnt|jt|bn|bone|rig|sk|character\d*)")


def _strip_prefix(name: str) -> str:
    """The name without a rig prefix (_PREFIX) when a word boundary follows it: a separator (dropped with it), or a
    capital that starts a word — after a lower-case letter or a digit ("rigSpine", "Bip01Spine"), or a capital followed
    by a lower-case letter ("DEFThigh"). A capital followed by another capital is inside an upper-case word: RIGHT_FOOT
    keeps its RIG (stripping it would leave "HT_FOOT" and lose the side)."""
    m = _PREFIX.match(name)
    if m is None:
        return name
    head, rest = m.group(0), name[m.end():]
    if rest[:1] in ("_", "-", ".", " "):
        return rest[1:]
    starts_word = rest[:1].isupper() and (head[-1].islower() or head[-1].isdigit() or rest[1:2].islower())
    return rest if starts_word else name
_HELPER = {"twist", "roll", "bend", "helper", "ribbon", "corrective", "offset", "aux", "ik", "fk", "pole", "ctrl"}
# helper words for a joint that lies on another joint's bone (_segments); "bend" is not one: Daz names its limb joints
# *Bend
_TWIST = _HELPER - {"bend"}
_END = {"end", "top", "nub", "site", "tip", "effector"}
_FINGER = {"thumb", "index", "middle", "ring", "pinky", "pinkie", "finger", "fingers"}  # words marking a joint as part of a finger
# words marking a joint inside the palm: a finger's metacarpal (UE5's index_metacarpal_l) belongs to the hand, and the
# finger chain starts after it, at the knuckle — both sides' chains then start at the same place, which the bend
# spread over a chain by arc length (motion.BodyRetarget) needs
_PALM = {"metacarpal", "metacarpals", "carpal", "palm", "meta"}
# Which of the five fingers a joint belongs to. This differs from _FINGER, which also accepts generic words such as
# "finger"; only specific finger names are listed here, with the variants found in models and rigs (the index finger
# also appears as fore / point, the little finger as little)
_FINGER_WORDS = {"thumb": {"thumb"}, "index": {"index", "fore", "point"}, "middle": {"middle", "mid"},
                 "ring": {"ring"}, "pinky": {"pinky", "pinkie", "little"}}
# words that name a finger's phalanx; "middle" is both a finger and a phalanx (L_Index_Middle), and on a joint that
# names another finger too, the phalanx reading is the one taken (_hands_and_fingers)
_PHALANX = {"proximal", "intermediate", "middle", "mid", "distal"}


def tokens(name: str) -> list[str]:
    """A joint name as lower-case words: namespaces and rig prefixes dropped, camelCase, digits and separators split
    ("mixamorig:LeftUpLeg" -> left up leg, "thigh_l" -> thigh l, "Bip01 L Thigh" -> l thigh)."""
    name = name.rsplit(":", 1)[-1]  # a namespace (Maya's, Mixamo's mixamorig:) is not part of what the joint is
    # "/" and "|" are characters of the name here, never path separators: tokens reads a joint's own name (guess), not
    # a key (joint_keys); split on them it lost the side of 「Left/Hand」 (the words below keep it)
    name = _strip_prefix(name) or name
    parts = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)
    return [p.lower() for p in parts]


def helper_bone(name: str) -> bool:
    """A joint whose own name says it is a helper bone (a twist or roll joint, an IK / pole control, a prop or weapon
    slot), not a body part a retarget solver should be given: the name evidence of the recognition engine
    (data/skeleton_recognition.py _read_name, its helper and prop words). The ignore suggestions read the engine's
    joint roles, which add what the positions say (nodes/kit/retarget_needs.py suggest_ignored); a joint the mapping
    uses as a body part is never taken for a helper there, whatever its name (AccuRIG's only neck joints are
    NeckTwist01/02)."""
    from .skeleton_recognition import _read_name

    got = _read_name(name)
    return got.helper or got.prop


def side(words: list[str]) -> str | None:
    left = any(w in ("left", "l", "lf", "lft") for w in words)
    right = any(w in ("right", "r", "rt", "rgt") for w in words)
    return "l" if left and not right else "r" if right and not left else None


def _kind(words: list[str]) -> str | None:
    """What a single joint's name says it is: hand, foot, head, jaw, eye, hips, or None."""
    w = set(words)
    if w & _HELPER or w & _END:
        return None
    if w & {"hand", "wrist"} and not w & (_FINGER | _PALM):  # a finger or palm joint named after its hand
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


def guess(names: list[str], parents, lone_side: str | None = None, rest=None, weights=None) -> dict:
    """The body parts of a rig: {"hips": j, "spine": [j...], "chest": j, "neck": [j...], "head": j, "l.hand": j,
    "l.index": [j, j, j], ...} (joint indices; a part not found is left out; a chain part is a list).

    The skeleton recognition engine's parts (data/skeleton_recognition.py recognize: names, hierarchy, rest positions,
    symmetry, judged together; a part it is not sure of is left out). `lone_side` is for a skeleton that is one hand or
    one side only and whose joint names never say which ("wrist", "index1", ...; the model records the side beside its
    joints, as MANO does): "l" or "r". `rest` [J,3] (rest positions), when the caller has them: without them only the
    names and the hierarchy are read (_by_names). `weights`: which joints carry skin weights (data/animation.py
    Rig.weighted), when the skeleton has a skinned mesh: joints that drive no vertex are no candidates (recognize)."""
    from .skeleton_recognition import recognize

    got = recognize(names, parents, rest, weights=weights, lone_side=lone_side).parts
    return {p: (list(v) if isinstance(v, list) else v) for p, v in got.items()}


def _by_names(names: list[str], parents, lone_side: str | None = None) -> dict:
    """The name reading of the recognition engine (data/skeleton_recognition.py): names say what a hand, a foot, the
    head and the hips are, the hierarchy gives the rest. Same result shape as guess()."""
    parents = np.asarray(parents)
    words = [tokens(n) for n in names]
    kinds = [_kind(w) for w in words]
    sides = [side(w) for w in words]
    depth = [len(_ancestors(j, parents)) for j in range(len(names))]

    def top_most(kind: str, s: str | None = None) -> int | None:
        found = [j for j in range(len(names)) if kinds[j] == kind and (s is None or sides[j] == s)]
        return min(found, key=lambda j: depth[j]) if found else None

    def limb(path: list[int]) -> list[int]:
        """One joint per segment of a limb's path (_segments): a helper joint is not dropped by its name, it is taken
        into the segment it lies on, and the segment is represented by its first joint."""
        return [seg[0] for seg in _segments(path, words)]

    def plain(path: list[int]) -> list[int]:
        """The path without its helper joints (twist, roll, ik, ...): a limb often chains upperarm -> upperarm_twist ->
        forearm, and the twist is not a part. The hierarchy outranks the names: when the filter would empty the path
        the joints stay (a rig whose only neck joints are NeckTwist01/02 has a neck)."""
        return [j for j in path if not set(words[j]) & _HELPER] or list(path)

    out: dict = {}
    feet = {s: top_most("foot", s) for s in SIDES}
    hips = top_most("hips")
    if all(f is not None for f in feet.values()):
        # the hips are the joint both legs and the trunk hang below (motion.hips_joint, the rule the mapping check
        # E-MAP-HIPS reads too): the one named so when it is, else the nearest above both — a COG rig names its leg
        # fork Pelvis and hangs the spine from COG above it
        from lab2shot_shared.motion import hips_joint, trunk_top

        marks = [j for j in (top_most("head"), top_most("hand", "l"), top_most("hand", "r")) if j is not None]
        legs = list(feet.values())
        found = hips_joint(parents, -1 if hips is None else hips, legs, trunk_top(parents, legs, marks))
        hips = found if found >= 0 else hips

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
        leg = limb(_path(hips, foot, parents)) if foot is not None else []
        if len(leg) >= 3:
            out[f"{s}.thigh"], out[f"{s}.shin"], out[f"{s}.foot"] = leg[-3:]
            below = [c for c in range(len(names)) if foot in _ancestors(c, parents)[1:] and not set(words[c]) & (_END | _HELPER)]
            named = [c for c in below if set(words[c]) & {"toe", "toes", "ball"}]  # past any ankle sub-joints (MHR)
            toes = named or [c for c in below if parents[c] == foot]
            if toes:  # the highest one; among equals the one the chain goes on from (not a helper hanging beside it)
                has_kids = {int(parents[c]) for c in range(len(names))}
                out[f"{s}.toe"] = min(toes, key=lambda c: (depth[c], c not in has_kids))
    head = top_most("head")
    hands = [h for h in (top_most("hand", s) for s in SIDES) if h is not None]
    if head is not None:
        trunk = _path(hips, head, parents)[:-1]  # between the hips and the head
    elif len(hands) == 2:
        # no joint is named a head (Blender Rigify's DEF skeleton: the head is DEF-spine.006): the arms still branch
        # off the trunk, so the trunk runs up to where the two arms part (the hands' deepest common ancestor, the
        # chest), which the arm loop below then takes as the chest
        shared = [j for j in _ancestors(hands[0], parents) if j in _ancestors(hands[1], parents)]
        trunk = _path(hips, shared[0], parents) if shared else []
    else:
        trunk = []
    spine = plain(trunk)
    if head is not None:
        out["head"] = head
    for s in SIDES:
        hand = top_most("hand", s)
        if hand is None or not spine:
            continue
        branch = next((j for j in _ancestors(hand, parents) if j in spine or j == hips), None)
        arm = limb(_path(branch, hand, parents)) if branch is not None else []
        if len(arm) >= 3:
            out[f"{s}.upperarm"], out[f"{s}.forearm"], out[f"{s}.hand"] = arm[-3:]
            if len(arm) >= 4:
                out[f"{s}.clavicle"] = arm[-4]
            if branch in spine and "chest" not in out:
                # the chest is where the arms branch off the trunk, but never a joint named a neck: 3ds Max Biped hangs
                # its clavicles under Bip01 Neck, whose parent in the trunk is the chest
                while "neck" in words[branch] and trunk.index(branch) > 0:
                    branch = trunk[trunk.index(branch) - 1]
                out["chest"] = branch
    if "chest" in out:
        # split the trunk at the chest, then drop the helpers of each half (plain(): a half the filter would empty
        # keeps its joints — CC / AccuRIG rigs have no neck joints but NeckTwist01/02)
        k = trunk.index(out["chest"])
        out["spine"], out["neck"] = plain(trunk[:k]), plain(trunk[k + 1:])
    elif spine:
        out["spine"], out["neck"] = spine, []
    for part in [p for p in out if p.startswith("l.")]:  # a guess never gives one joint to both sides
        other = "r." + part[2:]
        if other in out and out[other] == out[part]:
            del out[part], out[other]
    return loose()


# words that name a limb bone of their own: a joint whose name adds one of these to the joint before it is the next
# bone (LeftArm → LeftForeArm, arm → lowerarm), not a variant lying on that bone (_segments)
_LIMB = {"fore", "forearm", "lower", "lowerarm", "upper", "upperarm", "up", "low", "elbow", "knee", "shin", "calf", "leg",
         "thigh", "foot", "ankle", "hand", "wrist", "arm", "shoulder", "clavicle", "collar", "toe", "ball", "shldr"}


def _base(words: list[str]) -> frozenset[str]:
    """What a joint's name says it is, without helper words and numbers: "lThighBend" and "lThighTwist" are both the
    left thigh, "upperarm_twist_01_l" is the left upper arm."""
    return frozenset(w for w in words if w not in _HELPER and not w.isdigit())


def _segments(path: list[int], words: list[list[str]]) -> list[list[int]]:
    """A limb's path (joints from its root down to its end) as segments, each one limb bone with the joints that lie
    on it. The hierarchy gives the joints and their order; no joint is dropped for its name. A joint joins the segment
    before it when its name names that segment's joint and more (its base contains the segment's: Daz's lThighBend →
    lThighTwist, AdvancedSkeleton's Hip_L → HipPart1_L, upperarm → upperarm_twist_01): a variant of the bone it lies
    on, whatever the extra word is (twist, roll, part, bend …) — unless the extra word names a limb bone itself
    (_LIMB: Mixamo's LeftArm → LeftForeArm is two bones). A joint named a twist (_TWIST) lies on the bone before it
    too, unless the joint after it has its name (then it is the bone's own joint). Anything else starts a segment:
    Daz's arm with its twists stripped (lCollar → lShldrBend → lForearmBend) is three bones — Daz names its limb joints
    *Bend, so "bend" says nothing of a joint lying on another's bone."""
    out: list[list[int]] = []
    for i, j in enumerate(path):
        base = _base(words[j])
        variant = out and base >= (first := _base(words[out[-1][0]])) and not (base - first) & _LIMB
        # a twist lying on the bone before it, named after the joint it turns towards (SAM 3D Body's l_wrist_twist
        # between LeftForeArm and LeftHand): not a bone of its own unless the next joint carries its name too
        twist = bool(set(words[j]) & _TWIST) and not (i + 1 < len(path) and _base(words[path[i + 1]]) == base)
        if out and (variant or twist):
            out[-1].append(j)
        else:
            out.append([j])
    return out


def _hands_and_fingers(out: dict, names, parents, words, depth, lone_side, top_most) -> None:
    """The five finger chains of each hand. Hands are taken from those already recognised in `out` (the end of an arm);
    a skeleton without a body to hang them on (a hand alone: HaMeR's MANO delivered as RightHand / RightHandIndex1…,
    a hand-only rig) takes the highest joint each side's name calls a hand; for a single hand whose names do not state
    the side (MANO's own "wrist"), the model's declared `lone_side` decides. A finger is one branch of the hand (a palm
    or helper joint holding several is passed through), its chain that branch's main line; which finger it is its names
    say (thumb / index / middle / ring / pinky, or Biped's number), so a joint belongs to one finger only."""
    hands = {s: out[f"{s}.hand"] for s in SIDES if f"{s}.hand" in out}
    if not hands:
        for s in SIDES:
            if (h := top_most("hand", s)) is not None:
                hands[s] = out[f"{s}.hand"] = h
    if not hands and lone_side in SIDES and (lone := top_most("hand")) is not None:
        hands[lone_side] = lone
        out[f"{lone_side}.hand"] = lone
    kids: dict[int, list[int]] = {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)
    skip = _HELPER | _END | _PALM

    def size(j: int) -> int:
        return 1 + sum(size(c) for c in kids.get(j, []))

    def roots(j: int) -> list[int]:
        """The fingers' first joints under `j`: its children, through a palm or helper joint that holds several."""
        out = []
        for c in kids.get(j, []):
            out += roots(c) if set(words[c]) & skip else [c]
        return out

    def which(chain: list[int]) -> tuple[int, int] | None:
        """The finger a chain is (FINGERS order) and where it starts in the chain: at the first of its joints whose
        name says one finger (the joints before it name none: LAFAN's LeftFingerBase is the hand's), a word that also
        names a phalanx (the middle phalanx: L_Index_Middle) giving way to another finger word on the same joint."""
        for i, j in enumerate(chain):
            hit = {k for k, hints in enumerate(_FINGER_WORDS.values()) if set(words[j]) & hints}
            if (n := _finger_number(words[j])) is not None:
                hit.add(n)
            if len(hit) > 1:
                hit -= {k for k, (f, hints) in enumerate(_FINGER_WORDS.items())
                        if set(words[j]) & hints <= _PHALANX}
            if len(hit) == 1:
                return hit.pop(), i
        return None

    for s, hand in hands.items():
        found: dict[int, list[int]] = {}
        for root in roots(hand):
            chain = [root]  # one finger is one branch of the hand, down its main line (the child holding the most)
            while nxt := [c for c in kids.get(chain[-1], []) if not set(words[c]) & skip]:
                chain.append(max(nxt, key=size))
            # a finger bends at three joints; a fourth at its end without children is its tip (FBX / USD Mixamo's
            # Index4, where a BVH has an End Site): no joint of the chain, only where its last bone points
            # (motion.chain_bones' tip). Counted, one skeleton's finger has a joint more than the other's and the bend
            # spread by arc length is smeared (up to 15°). A third joint without children (SMPL-X, MANO, UE5) is a joint
            while len(chain) > 3 and chain[-1] not in kids:
                chain.pop()
            got = which(chain)
            if got is None:
                continue
            k, start = got
            chain = chain[start:]
            if len(chain) > len(found.get(k, [])):
                found[k] = chain
        for k, chain in found.items():
            out[f"{s}.{FINGERS[k]}"] = chain


def _finger_number(words: list[str]) -> int | None:
    """3ds Max Biped numbers its fingers instead of naming them: Finger0 is the thumb, Finger1 the index … Finger4 the
    little finger, the joints along one Finger1, Finger11, Finger12 (the first digit is the finger)."""
    if "finger" not in words:
        return None
    digits = next((w for w in words[words.index("finger") + 1:] if w.isdigit()), None)
    return int(digits[0]) if digits is not None and int(digits[0]) < 5 else None


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


# ------------------------------------------------------------------ a mapping as a node parameter (「对应关系」)
# One row per body part: {"part": id of PARTS, "src": [joint names], "dst": [joint names]}. `src` drives, `dst` is driven:
# in 「动作重定向」 src is the motion's skeleton and dst the target's; in the skeleton-motion nodes src is the person's rig
# and dst the model's own joints (fixed by the node). A part the parameter does not list is guessed (guess()); a part
# listed with an empty column is left out on purpose. Joints by name, not index: an asset exported again with its
# joints in another order keeps its mapping. The view's editing mode (the "rig_pair" handle, webui/src/view/rigPair.tsx)
# shows the same rules from the data NodeDef.handle_data gives; this is the authority, checked when the node cooks.


def joint_keys(names: list[str], parents) -> list[str]:
    """How parameters name each joint (「对应关系」 rows, 「初始姿势」 rows, the editors' lists): its name when no other
    joint of the skeleton has it, otherwise its path of names from the root ("Hips/…/LeftHand/Index1": two hands whose
    fingers share names), with lab2shot_shared.names.unique's "_2" for the rare paths that still repeat. Names alone
    are not unique in FBX or USD, and a mapping by name would give both hands' Index1 to one finger."""
    from collections import Counter

    from lab2shot_shared.names import join_path, unique

    names = [str(n) for n in names]
    count = Counter(names)

    def path(j: int) -> str:  # names joined by "/", a "/" inside a name escaped (lab2shot_shared/names.py join_path)
        return join_path(names[k] for k in reversed(_ancestors(j, parents)))[1:]

    keys, taken = [], set(n for n in names if count[n] == 1)
    for j, n in enumerate(names):
        key = n if count[n] == 1 else unique(path(j), taken)
        taken.add(key)
        keys.append(key)
    return keys


def part_names(names: list[str], parents, rest=None, weights=None) -> dict[str, list[str]]:
    """guess() as joint keys (joint_keys: the name, or the path where names repeat), every part a list (a single
    joint's part: one key). `rest`, `weights`: see guess()."""
    parts = guess(list(names), parents, rest=rest, weights=weights)
    keys = joint_keys(names, parents)
    return {part: [keys[j] for j in (where if isinstance(where, list) else [where])] for part, where in parts.items()}


def part_joints(rows: list[dict], names: list[str], parents, col: str) -> dict[str, list[int]]:
    """The rows' joints of one side (`col`: src / dst) as indices, by joint_keys: the one reading of a mapping, for
    「动作重定向」 and the model nodes alike. The rows are check_mapping's."""
    at = {k: j for j, k in enumerate(joint_keys(names, parents))}
    return {r["part"]: [at[k] for k in r[col]] for r in rows if r.get(col)}


def part_rows(required: tuple[str, ...] = ()) -> list[dict]:
    """Every body part as the 「对应关系」 editor lists its slots: id, label, region, whether it is a chain (several
    joints), whether the node needs it, and for a chain that ends in another part the part it ends in ("end": the
    spine the chest, the neck the head; CHAIN_ENDS, the one table — the editor draws a chain's bone lengths up to it)."""
    return [{"id": p, "label": part_label(p), "region": region_label(REGION_OF[p]), "region_id": REGION_OF[p], "chain": p in CHAINS, "required": p in required,
             **({"end": CHAIN_ENDS[p][1]} if p in CHAIN_ENDS and CHAIN_ENDS[p][1] else {})}
            for p in PARTS]


def auto_rows(src: dict[str, list[str]], dst: dict[str, list[str]]) -> list[dict]:
    """The guessed mapping as rows of the parameter (PARTS order): every part either side has."""
    return [{"part": p, "src": list(src.get(p, [])), "dst": list(dst.get(p, []))} for p in PARTS if p in src or p in dst]


def merged_rows(mapping: list[dict] | None, auto: list[dict]) -> tuple[list[dict], set[str]]:
    """The rows a cook uses: the parameter's rows, and the guess for every part the parameter does not list.
    Returns (rows in PARTS order, the parts that were guessed)."""
    given = {r["part"]: r for r in mapping or []}
    guessed = {r["part"]: r for r in auto}
    rows = [given.get(p) or guessed[p] for p in PARTS if p in given or p in guessed]
    rows += [r for p, r in given.items() if p not in PARTS]  # refused by check_mapping (E-MAP-PART)
    return rows, {p for p in guessed if p not in given}


def below(j: int, top: int, parents) -> bool:
    """Joint `j` is somewhere under `top` (not `top` itself)."""
    return top in _ancestors(int(parents[j]), parents) if int(parents[j]) >= 0 else False


def check_mapping(rows: list[dict], sides: dict[str, tuple]) -> None:
    """Check the rows of a mapping against the skeletons they name, `sides` {"src" / "dst": (joint names, parents,
    what the skeleton is called in a message: 动作 / 目标 / 人物[, rest positions [J,3]])} — only the sides given are
    checked (a model's own joints are the node's); the positions, where given, find the trunk's top as the bodies do
    (motion.trunk_top). Raises Invalid with the first problem: a part that does not exist (E-MAP-PART), a joint
    the skeleton lacks (E-MAP-NOJOINT), several joints on a part that takes one (E-MAP-ONEJOINT), a chain whose joints
    do not run down one line of the hierarchy (E-MAP-CHAIN), one joint on two parts (E-MAP-SHARED), hips that do not
    have both thighs below them (E-MAP-HIPS)."""
    from ..errors import Invalid
    from ..messages import Msg

    for row in rows:
        if row["part"] not in PARTS:
            raise Invalid(Msg("E-MAP-PART", part=row["part"]))
    for col, (names, parents, said, *rest) in sides.items():
        parents = np.asarray(parents)
        names = joint_keys(list(names), parents)  # rows name joints by key: a repeated name by its path
        taken: dict[str, str] = {}
        for row in rows:
            part, joints = row["part"], list(row[col])
            for n in joints:
                if n not in names:
                    same = [k for k in names if k.rsplit("/", 1)[-1] == n]
                    if len(same) > 1:  # a bare name that several joints share: which one is meant is not known
                        raise Invalid(Msg("E-MAP-DUPNAME", part=part_label(part), joint=n, side=said,
                                          paths=i18n.Both.of(lambda: i18n.separator().join(same[:4]) + (i18n.t("list.etc") if len(same) > 4 else ""))))
                    raise Invalid(Msg("E-MAP-NOJOINT", part=part_label(part), joint=n, side=said))
            if part not in CHAINS and len(joints) > 1:
                raise Invalid(Msg("E-MAP-ONEJOINT", part=part_label(part), count=len(joints), side=said))
            for a, b in zip(joints, joints[1:]):
                if not below(names.index(b), names.index(a), parents):
                    raise Invalid(Msg("E-MAP-CHAIN", part=part_label(part), joint=b, above=a, side=said))
            for n in joints:
                if n in taken and taken[n] != part:
                    raise Invalid(Msg("E-MAP-SHARED", first=part_label(taken[n]), second=part_label(part), joint=n,
                                      side=said))
                taken[n] = part
        # the hips carry the body: the joint moved to place it (motion.set_hips) must have both thighs and the trunk
        # below it (motion.hips_joint, the rule the guess reads too); a thigh beside it stays behind (the take stands
        # still), a spine above it (CC / AccuRIG's Pelvis, whose parent Hip carries the spine) leaves the upper body
        # where it was while the legs walk off
        first = {r["part"]: r[col][0] for r in rows if r.get(col)}
        if "hips" in first:
            from lab2shot_shared.motion import hips_joint, trunk_top

            at = {k: names.index(k) for k in first.values()}
            hips = at[first["hips"]]
            legs = [at[first[p]] for p in ("l.thigh", "r.thigh") if p in first]
            # the trunk's top from the mapped hands and head, else from the hierarchy — a mapping of the legs alone
            # still moves the whole skeleton by the hips, so they must carry the trunk the skeleton has
            top = trunk_top(parents, legs, [at[first[p]] for p in LANDMARKS if p in first],
                            rest[0] if rest else None) if legs else -1
            if legs and hips_joint(parents, hips, legs, top) != hips:
                raise Invalid(_hips_wrong(first, at, hips, top, names, parents, said))


def _hips_wrong(first: dict, at: dict, hips: int, top: int, names, parents, said: str):
    """E-MAP-HIPS for hips that leave a thigh or the trunk outside them: the mapped part that is not below them, or the
    trunk's top itself when that is a joint no part names (the one message, built here only)."""
    from ..messages import Msg

    part = next((p for p in ("l.thigh", "r.thigh", "spine", "chest", "neck", *LANDMARKS)
                 if p in first and not below(at[first[p]], hips, parents)), None)
    if part is not None:
        return Msg("E-MAP-HIPS", hips=first["hips"], joint=first[part], part=part_label(part), side=said)
    return Msg("E-MAP-HIPS", hips=first["hips"], joint=names[top], part=region_label("trunk"), side=said)


# ------------------------------------------------------------------ CG bone names (the project's naming convention)

# The CG name of each bone. ML body models name joints while CG names bones, offset by one: SMPL's left_shoulder is
# the upper arm, left_hip the thigh, left_foot the toes. Animators would misread SMPL names and HumanIK would not match
# them, so names are converted to the project's convention before delivery: Maya HumanIK / Mixamo names, the one
# definition in data/bone_names.py (recognised by Maya HumanIK, MotionBuilder, Blender Rigify and UE Retarget).
from .bone_names import SIDE_WORD, STANDARD as CG_NAMES, chain_name as cg_chain_name  # noqa: E402,F401


def cg_names(names: list[str], parents, lone_side: str | None = None, rest=None, weights=None,
             others: str = "keep") -> list[str]:
    """The CG name of every bone of this skeleton (data/bone_names.py): body parts recognised by guess() get the
    convention's names. Returns a list as long as `names`, with duplicates numbered.

    `others`: what an unrecognised joint is called. "keep": its own name (a body model's own helpers, named by the
    model), an end joint without children that is its recognised parent's single child "<parent>_End" (Mixamo's
    LeftToe_End); "rule": the convention's rule names (bone_names.rule_names: <nearest recognised ancestor>_Helper1,
    _End, _Extra1 …, by the engine's joint roles) — for a skeleton whose own names say nothing (auto-rigging's bone_0,
    bone_1 …).
    `lone_side`: see guess(); for a single-hand model whose names do not state the side, it decides left or right.
    `rest` [J,3], `weights`: see guess() (the model deliveries do not pass them: a model's own names are read; an
    auto-rig passes both, its parts are found by shape)."""
    from .bone_names import part_names as standard, rule_names

    parts = guess(names, parents, lone_side, rest, weights)
    out = list(names)
    named = standard(parts)
    for j, n in named.items():
        out[j] = n
    if others == "rule":
        from .skeleton_recognition import recognize

        roles = recognize([str(n) for n in names], parents, rest, weights=weights, lone_side=lone_side).roles
        for j, n in rule_names(parents, named, roles).items():
            out[j] = n
        return unique_names(out, first=set(named))
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
    # a recognised joint keeps its CG name when an unrecognised one already has it (a helper the model calls Spine1):
    # HumanIK reads the convention's names, the helper's own name matters to no one
    return unique_names(out, first=set(named))


def unique_names(names: list[str], first=()) -> list[str]:
    """`names` with every repeat renamed so that all are different (bone names must be unique in a skeleton), by the
    one tie-break every writer uses (lab2shot_shared.names.unique: "_2", "_3"…), checked against every name in the
    list so a renamed one never lands on another. The joints in `first` (indices) keep their names before the others
    do; otherwise the earlier one keeps it. The list stays in order: each name stands at its joint's index."""
    from lab2shot_shared.names import unique

    order = sorted(range(len(names)), key=lambda i: (i not in first, i))
    taken, seen, out = set(names), set(), list(names)
    for i in order:
        n = names[i]
        if n in seen:
            n = unique(n, taken)
            taken.add(n)
        seen.add(n)
        out[i] = n
    return out
