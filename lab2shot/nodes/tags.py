"""Tags: what a node, a choice of one of its parameters and a template are, for managing them all the same way — which
users may use them, and the chips a page shows on them. Derived here, in one place; nothing else decides it.

    basic          基础      the core's nodes and the format modules (extensions that declare it): every user has it
    commercial     可商用    its licence allows commercial use
    noncommercial  非商用    its licence forbids commercial use
    research       仅限研究  its licence allows only academic research (stricter than 非商用: no production at all)
    registration   需注册    it needs a model each user must register for (SMPL, SMPL-X, MANO, FLAME: manual items)

Where they come from:
  - an extension declares its licence class once: LicenseInfo(tag=...) (basic, commercial, noncommercial, research),
    which the extension loader stamps on its node classes (NodeDef.project, lab2shot/adapters.py project_of);
  - a node whose licence differs from its extension's says so itself: NodeDef.licence (nodes/applies.py Licence);
  - a parameter value that switches to restricted parts tags that choice with its class, 非商用 or 仅限研究 (an
    OptionTrait(..., licence=...), made from the extension's OPTION_LICENCES table by licence_traits: Kimodo's SMPL-X
    weights are 仅限研究); the node itself keeps its class, nodes/applies.py resolves it, and an account is judged on the
    choice as on a node (may: someone given only 非商用 does not get a 仅限研究 choice);
  - 需注册 comes from the hand-downloaded items the extension needs (ManualItem.registration), or a node's own
    Licence(registration=True) (the core's 「标准人」, SMPL-X);
  - a template carries every tag of the nodes (and choices) it uses.

A user may use what carries only tags they were given (accounts: the administrator ticks them; ALLOWED_NEW for a new
user) plus IMPLIED; the administrator may use everything. What a user may not use does not exist for them
(server/access.py). A later kind of tag (需显卡, a department's) is one more entry in TAGS and a rule in node_tags,
not a new path through the server or the pages.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import provide_choices

BASIC, COMMERCIAL, NONCOMMERCIAL, RESEARCH, REGISTRATION = "basic", "commercial", "noncommercial", "research", "registration"
LICENCES = (BASIC, COMMERCIAL, NONCOMMERCIAL, RESEARCH)  # the licence classes: every node has exactly one
IMPLIED = frozenset({BASIC})  # every user has these
ALLOWED_NEW = frozenset({COMMERCIAL})  # what a new user may use: the lowest licence risk


@dataclass(frozen=True)
class Tag:
    label: str
    tip: str
    implied: bool = False  # every user has it: not something the administrator gives


TAGS: dict[str, Tag] = {
    BASIC: Tag("基础", "Lab2Shot 自己的节点和读写文件格式的模块：每个用户都能用", implied=True),
    COMMERCIAL: Tag("可商用", "许可证允许商业使用（仍要遵守许可证里的其他条件，比如署名）"),
    NONCOMMERCIAL: Tag("非商用", "许可证禁止商业使用：只能用于研究、学习等非商业用途，不能用在商业项目的镜头里"),
    RESEARCH: Tag("仅限研究", "许可证只允许学术研究：比非商用更严，不能用于任何生产"),
    REGISTRATION: Tag("需注册", "要用每个人在官网注册后才能下载的人体、手或面部模型（SMPL、SMPL-X、MANO、FLAME），它们的许可证只许非商业的科研、教学和艺术项目"),
}


def account_choices() -> tuple[tuple[str, str], ...]:
    """What an account may be given (every tag but the implied ones), as 用户's 可用 offers it: also the options of the
    setting 注册可用模型类别 (lab2shot/config.py provide_choices, below)."""
    return tuple((k, t.label) for k, t in TAGS.items() if not t.implied)


provide_choices("account_tags", account_choices)


def describe() -> dict[str, dict]:
    """Every tag, for the pages: id -> {label, tip, implied}."""
    return {k: {"label": t.label, "tip": t.tip, "implied": t.implied} for k, t in TAGS.items()}


def node_tags(node_type) -> frozenset[str]:
    """A node type's tags: its licence class (its own licence, else its project's, never less strict than the data it
    uses: DATA_LICENCES of its own and its extension's `uses`) and what its project brings (nodes/services.py
    ProjectFacts, stamped by the extension loader)."""
    ext = getattr(node_type.project, "extension", None)
    uses = (*node_type.licence.uses, *(ext.license.uses if ext is not None else ()))
    cls = min([node_type.licence.tag or node_type.project.licence, *(DATA_LICENCES[u] for u in uses)], key=STRICTNESS.index)
    return frozenset({cls, *node_type.project.tags, *({REGISTRATION} if node_type.licence.registration else ())})


def commercial(tags: frozenset[str]) -> bool:
    """May results made with these tags be used commercially?"""
    return not tags & {NONCOMMERCIAL, RESEARCH, REGISTRATION}


# data and body models whose licence binds whatever runs on them or was trained on them, one class each wherever they
# are used (LicenseInfo.uses, Licence.uses; lab2shot check numbers: licence_words_agree). AMASS (and HumanML3D, made
# from it), BEDLAM, 3DPW and Human3.6M allow only non-commercial scientific research; LaFAN1 is CC BY-NC-ND; the body
# models allow non-commercial research, education and artistic projects
DATA_LICENCES = {"AMASS": RESEARCH, "BEDLAM": RESEARCH, "3DPW": RESEARCH, "Human3.6M": RESEARCH,
                 "LaFAN1": NONCOMMERCIAL, "SMPL": NONCOMMERCIAL, "SMPL-X": NONCOMMERCIAL, "MANO": NONCOMMERCIAL,
                 "FLAME": NONCOMMERCIAL}

# strictest first: what a card says in one word is the worst its parts allow. 需注册 sits
# between 非商用 and 可商用: it is not a licence class of its own, but a model behind a registration may not be used
# commercially either (commercial() counts it), so it must never be hidden behind 可商用.
STRICTNESS = (RESEARCH, NONCOMMERCIAL, REGISTRATION, COMMERCIAL, BASIC)


def strictest(tags: frozenset[str]) -> str:
    """The licence word a card, a node and a route show for these tags: the strictest of them (「仅限研究」 over 「非商用」
    over 「需注册」 over 「可商用」 over 「基础」), and 「需注册」 after it when that is not already the word — registering
    is a step of its own, not a licence class, and must not hide behind 「非商用」 (「非商用 · 需注册」). "" when there is
    nothing to say. The one spelling of it: a page never picks among the tags itself, it prints this."""
    word = next((TAGS[k].label for k in STRICTNESS if k in tags), "")
    return f"{word} · {TAGS[REGISTRATION].label}" if REGISTRATION in tags and word != TAGS[REGISTRATION].label else word


def label(tag: str) -> str:
    """A tag's word as pages and option names show it (「仅限研究」)."""
    return TAGS[tag].label


def least_strict(routes: list[frozenset[str]]) -> frozenset[str]:
    """Of several routes' tags (a card's, engine/templates.py route_tags), the one whose strictest tag is the least
    strict (the first on a tie): what the card offers at best."""
    rank = lambda tags: next((i for i, k in enumerate(STRICTNESS) if k in tags), len(STRICTNESS))  # noqa: E731
    return max(routes, key=rank, default=frozenset())


def may(tags: frozenset[str], allowed: frozenset[str] | None) -> bool:
    """May someone given `allowed` (None: the administrator, everything) use what carries `tags`?"""
    return allowed is None or tags <= allowed | IMPLIED


def graph_tags(data: dict, among: list[str] | None = None) -> frozenset[str]:
    """Every tag of a graph (a template's JSON): its nodes' and their chosen values'; a node type this server does
    not have counts as the strictest (RESEARCH), so it is never offered as usable. `among`: only these nodes (a card's
    delivery route with its defaults, engine/templates.py delivered_by); None: every node."""
    from . import node_types
    from .applies import resolve_params
    from .params import param_defaults

    types = node_types()
    out: set[str] = set()
    for n in data.get("nodes", []):
        if among is not None and n.get("id") not in among:
            continue
        t = types.get(n.get("type"))
        out |= {RESEARCH} if t is None else resolve_params(t, {**param_defaults(t.Params), **(n.get("params") or {})}).licence.tags
    return frozenset(out)
