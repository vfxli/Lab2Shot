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
  - a parameter value that switches to non-commercial parts (an OptionTrait(..., noncommercial=True): e.g. the original
    weights) tags that choice 非商用 (the node itself keeps its class; nodes/applies.py resolves it);
  - 需注册 comes from the hand-downloaded items the extension needs (ManualItem.registration);
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
    REGISTRATION: Tag("需注册", "要用每个人在官网注册后才能下载的人体、手或面部模型（SMPL、SMPL-X、MANO、FLAME），它们的许可证也只许非商用科研"),
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
    """A node type's tags: its licence class (its own licence, else its project's) and what its project brings
    (nodes/services.py ProjectFacts, stamped by the extension loader)."""
    return frozenset({node_type.licence.tag or node_type.project.licence, *node_type.project.tags})


def commercial(tags: frozenset[str]) -> bool:
    """May results made with these tags be used commercially?"""
    return not tags & {NONCOMMERCIAL, RESEARCH, REGISTRATION}


# strictest first: what a card says in one word is the worst its parts allow. 需注册 sits
# between 非商用 and 可商用: it is not a licence class of its own, but a model behind a registration may not be used
# commercially either (commercial() counts it), so it must never be hidden behind 可商用.
STRICTNESS = (RESEARCH, NONCOMMERCIAL, REGISTRATION, COMMERCIAL, BASIC)


def strictest(tags: frozenset[str]) -> str:
    """The one licence word a card shows for these tags: the strictest of them (「仅限研究」 over 「非商用」 over
    「需注册」 over 「可商用」 over 「基础」). "" when there is nothing to say. A page never picks among the tags
    itself: one rule, on the server."""
    return next((TAGS[k].label for k in STRICTNESS if k in tags), "")


def may(tags: frozenset[str], allowed: frozenset[str] | None) -> bool:
    """May someone given `allowed` (None: the administrator, everything) use what carries `tags`?"""
    return allowed is None or tags <= allowed | IMPLIED


def graph_tags(data: dict) -> frozenset[str]:
    """Every tag of a graph (a template's JSON): its nodes' and their chosen values'; a node type this server does
    not have counts as the strictest (RESEARCH), so it is never offered as usable."""
    from . import node_types
    from .applies import resolve_params
    from .params import _defaults

    types = node_types()
    out: set[str] = set()
    for n in data.get("nodes", []):
        t = types.get(n.get("type"))
        out |= {RESEARCH} if t is None else resolve_params(t, {**_defaults(t.Params), **(n.get("params") or {})}).licence.tags
    return frozenset(out)
