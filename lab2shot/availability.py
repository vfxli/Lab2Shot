"""Availability: which parts of a page are available to the current viewer, declared once as data and resolved in
one place.

A subject is one thing a page shows: a node's parameter, a node's output, a section or a button of the admin page,
an action on an account's row, and later the installer's and the benchmarks' controls. Each subject declares a
condition (Cond) as plain, three-valued data: holds() returns True, False or None (unknown). Each condition has one
kind, and the kind alone determines the effect when the condition does not hold (POLICY):

    capability  the session's role has it (roles.Can)             False: hidden     not known: hidden
    data        what is wired, set, chosen; the rights still fresh False: greyed, with why (a message)
                (nodes/applies.Wired, roles.RightsFresh)            not known: shown
    cook        a fact only cooking tells (nodes/applies.fact)     False: greyed, with why
                                                                    not known: shown, with "known once it cooks" (pending)

All takes the strictest of its parts and AnyOf the most lenient (available < pending < inactive < hidden). Not negates
a leaf that can state why its opposite does not hold (a wire, a value set).

resolve(subjects, facts, words) produces the single answer every page reads (Availability): which subjects are
available, which are greyed and why, and which wait for a cook; a subject in none of these is hidden. `words` supplies
the names a message uses (a node type's labels); `facts` is whatever the leaves read (nodes/applies.NodeFacts,
roles.SessionFacts). The web page reads the answer through one module (webui/src/api/applies.ts) and never computes
any of it itself."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from .messages import Msg


class Level(IntEnum):
    """What a condition makes of its subject, from the most lenient to the strictest."""

    AVAILABLE = 0
    PENDING = 1  # shown as applying; a cook determines whether it does
    INACTIVE = 2  # shown greyed, with the reason
    HIDDEN = 3  # not available to this session: not shown at all


AVAILABLE, PENDING, INACTIVE, HIDDEN = Level.AVAILABLE, Level.PENDING, Level.INACTIVE, Level.HIDDEN

DATA, COOK, CAPABILITY = "data", "cook", "capability"


@dataclass(frozen=True)
class Policy:
    false: Level  # the condition does not hold
    unknown: Level  # it is not known whether it holds


# condition kind -> effect on the subject when the condition does not hold; the only definition of the hidden-or-greyed rule
POLICY: dict[str, Policy] = {
    CAPABILITY: Policy(false=HIDDEN, unknown=HIDDEN),
    DATA: Policy(false=INACTIVE, unknown=AVAILABLE),
    COOK: Policy(false=INACTIVE, unknown=PENDING),
}

_NAMES = (frozenset(), frozenset(), frozenset())


class Cond:
    """A condition (see the module docstring). A leaf implements holds(); why(), the reason it does not hold (for a
    subject it greys; a capability hides its subject and gives no reason); until(), for a cook kind not yet known; and
    names(), the parameters, inputs and facts it reads (verified to exist when a node class is created)."""

    kind: str = DATA

    def holds(self, f: Any) -> bool | None:
        raise NotImplementedError

    def why(self, f: Any, words: Any) -> Msg:
        raise NotImplementedError

    def until(self, f: Any, words: Any) -> Msg:
        raise NotImplementedError

    def names(self) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        return _NAMES

    def outcome(self, f: Any) -> tuple[Level, Cond]:
        """The level this condition gives its subject, and the leaf that decided it (whose why() or until() is reported)."""
        h = self.holds(f)
        return (AVAILABLE if h is True else POLICY[self.kind].false if h is False else POLICY[self.kind].unknown), self

    def leaves(self) -> Iterator[Cond]:
        yield self


@dataclass(frozen=True)
class Because(Cond):
    """A condition with an explicitly authored reason.

    Normally the reason for greying is generated mechanically by the condition (「「模型」选「甲」「乙」时才用」), which
    is sufficient and most precise in most cases. Some conditions have a real reason the generated sentence cannot
    express: for example, Kimodo's 「动画」 port cannot be connected with the two G1 robot weights not because of the
    model choice, but because that robot skeleton has three hip joint axes per leg and does not match a human
    skeleton. In such cases this class supplies an accurate message that artists can understand.

    Only the message is replaced; whether the condition holds is still decided by the wrapped condition. The sentence
    lives in the message catalogue as usual; the code holds only its code and parameters."""

    cond: Cond
    code: str
    params: tuple[tuple[str, Any], ...] = ()

    def __init__(self, cond: Cond, code: str, **params: Any):
        object.__setattr__(self, "cond", cond)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "params", tuple(sorted(params.items())))

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.cond.kind

    def holds(self, f: Any) -> bool | None:
        return self.cond.holds(f)

    def why(self, f: Any, words: Any) -> Msg:
        return Msg(self.code, **dict(self.params))

    def why_not(self, f: Any, words: Any) -> Msg:
        return Msg(self.code, **dict(self.params))

    def until(self, f: Any, words: Any) -> Msg:
        return Msg(self.code, **dict(self.params))

    def names(self):
        return self.cond.names()


def _union(conds) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    parts = [c.names() for c in conds]
    return tuple(frozenset().union(*(p[i] for p in parts)) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class _Several(Cond):
    """Conditions taken together (All, AnyOf)."""

    conds: tuple[Cond, ...]

    def __init__(self, *conds: Cond):
        object.__setattr__(self, "conds", tuple(conds))

    def names(self):
        return _union(self.conds)

    def leaves(self) -> Iterator[Cond]:
        for c in self.conds:
            yield from c.leaves()


class All(_Several):
    """Every part holds: the strictest part decides (the first of equals)."""

    def holds(self, f: Any) -> bool | None:
        got = [c.holds(f) for c in self.conds]
        return False if False in got else None if None in got else True

    def outcome(self, f: Any) -> tuple[Level, Cond]:
        return max((c.outcome(f) for c in self.conds), key=lambda o: o[0], default=(AVAILABLE, self))


class AnyOf(_Several):
    """One part holds: the most lenient part decides (the first of equals)."""

    def holds(self, f: Any) -> bool | None:
        got = [c.holds(f) for c in self.conds]
        return True if True in got else None if None in got else False

    def outcome(self, f: Any) -> tuple[Level, Cond]:
        got = [c.outcome(f) for c in self.conds]
        if not got:
            return HIDDEN, self
        best = min(got, key=lambda o: o[0])
        # When no alternative holds, the reason must name every one of them. Reporting only one gives an incomplete
        # reason: if 「序列图输出设置」's 「格式」 is greyed with 「接了「图像」才用」 although connecting R, G and B also
        # works, a user following that message may still take a path that does not apply.
        same = [leaf for lvl, leaf in got if lvl is best[0]]
        return (best[0], _Either(tuple(same))) if best[0] is not AVAILABLE and len(same) > 1 else best


@dataclass(frozen=True)
class _Either(Cond):
    """Combines the reasons of several failing alternatives into one message (AnyOf.outcome). It decides nothing and
    only provides the message."""

    parts: tuple[Cond, ...]

    def holds(self, f: Any) -> bool | None:
        return False

    def why(self, f: Any, words: Any) -> Msg:
        return _joined([c.why(f, words) for c in self.parts])

    def until(self, f: Any, words: Any) -> Msg:
        return _joined([c.until(f, words) for c in self.parts])

    def names(self):
        return _union(self.parts)

    def leaves(self) -> Iterator[Cond]:
        for c in self.parts:
            yield from c.leaves()


@dataclass(frozen=True)
class Not(Cond):
    """The negation of a leaf that can state why its opposite does not hold (why_not), e.g. Not(Wired("camera")): a
    connected camera determines what the parameter would otherwise set."""

    cond: Cond

    def __post_init__(self) -> None:
        if not callable(getattr(self.cond, "why_not", None)):
            raise TypeError(f"Not() takes a condition that says why its opposite does not hold (a wire, a value set), not {self.cond!r}")

    @property
    def kind(self) -> str:  # type: ignore[override]
        return self.cond.kind

    def holds(self, f: Any) -> bool | None:
        h = self.cond.holds(f)
        return None if h is None else not h

    def why(self, f: Any, words: Any) -> Msg:
        return self.cond.why_not(f, words)  # type: ignore[attr-defined]

    def names(self):
        return self.cond.names()


def level(cond: Cond | None, f: Any) -> Level:
    """The level `cond` gives its subject (available when there is no condition)."""
    return AVAILABLE if cond is None else cond.outcome(f)[0]


@dataclass(frozen=True)
class Availability:
    """The resolved answer (see the module docstring). A subject in none of these fields is hidden."""

    available: tuple[str, ...] = ()
    inactive: Mapping[str, Msg] = None  # type: ignore[assignment]
    pending: Mapping[str, Msg] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inactive", dict(self.inactive or {}))
        object.__setattr__(self, "pending", dict(self.pending or {}))

    def __add__(self, other: Availability) -> Availability:
        return Availability(self.available + other.available, {**self.inactive, **other.inactive}, {**self.pending, **other.pending})

    def json(self) -> dict:
        return {"available": sorted(self.available), "inactive": {k: m.json() for k, m in self.inactive.items()},
                "pending": {k: m.json() for k, m in self.pending.items()}}


NOTHING = Availability()


def resolve(subjects: Mapping[str, Cond], f: Any, words: Any = None, prefix: str = "") -> Availability:
    """Resolve each subject's condition on `f` (see the module docstring); `prefix` is prepended to each id (a table
    row, e.g. "layers[2].")."""
    available: list[str] = []
    inactive: dict[str, Msg] = {}
    pending: dict[str, Msg] = {}
    for name, cond in subjects.items():
        lvl, leaf = cond.outcome(f)
        if lvl is AVAILABLE:
            available.append(prefix + name)
        elif lvl is PENDING:
            pending[prefix + name] = leaf.until(f, words)
        elif lvl is INACTIVE:
            inactive[prefix + name] = leaf.why(f, words)
    return Availability(tuple(available), inactive, pending)


def _joined(said: list[Msg]) -> Msg:
    """Combine the reasons of several alternatives into one message. Identical messages are reported once (several
    admin page sections may all ask to re-enter the password); distinct reasons (「接了「图像」才用」 and
    「接了「R」才用」) are all listed, so the user sees the complete reason."""
    seen: list[Msg] = []
    for m in said:
        if not any(m.code == k.code and m.params == k.params for k in seen):
            seen.append(m)
    return seen[0] if len(seen) == 1 else Msg("I-APPLIES-EITHER", reasons=seen)
