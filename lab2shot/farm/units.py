"""计算单元 (设计 五, 十三.10): a job's work cut into pieces that can run on different cards at the same time.

A unit is **one item of the innermost 逐项处理 block** — the nodes of the block for that one item, on one card (a card
keeps its models loaded, so the same node's next item there costs no reload). Everything outside every block is one
unit, as a whole job was before (the empty item path). So the unit of an instance is simply its item path
(engine/scopes.py Inst.path) and the units of a job are the item paths its `Evaluation.order` walks — nothing new to
derive, and nested blocks fall out of it (the innermost path is the unit).

One `Ledger` per job. Every card working on the job runs its own `Engine` over the same targets (engine/cook.py
`cook(units=...)`), and the ledger answers, for each instance a card reaches:

    mine(inst)    this card's? — an item nobody has taken is taken now, unless this card is already on another item
                  and another card is working on this job: then it is left for that one;
    wait(inst)    True: another card had it and is through (done, failed or skipped) — what it left is on disk, so
                  this card plans on from there; False: nobody took it after all, so it is this card's now.

A card never stands still while there is work: the engine takes the first instance left whose inputs are through
(here or on another card) and that the ledger gives it, and only waits when everything left is another card's —
always for an instance that comes before the one it wants, so waiting can never close a circle. A card alone on a
job takes every unit as it reaches it, so one card cooks exactly as it did before.

Fairness (返工计划 三.9): a job borrows a second card only while no waiting job would be placed there — the queue's
order of waiting jobs (farm/queue.py `_line`) always comes first, so one account's many items never hold a card
another account is waiting for. That check is the scheduler's own `place()`, never a second rule written here.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..engine.scopes import Inst, ItemPath

GRACE_S = 0.5  # how long a card leaves an item for another card of the same job before taking it itself


@dataclass
class Ledger:
    """Who cooks which item of one job, and how far it got. Every method takes the lock: the cards call them from
    their own threads."""

    stop: threading.Event  # the job's: a cancelled job wakes everyone waiting here
    cond: threading.Condition = field(default_factory=threading.Condition)
    cards: set[str] = field(default_factory=set)  # the cards working on this job now ("" a job without a GPU)
    owners: dict[ItemPath, str] = field(default_factory=dict)  # item path -> the card that took it
    finished: set[Inst] = field(default_factory=set)  # instances that came to an end (cooked, failed or skipped)
    counted: set[ItemPath] = field(default_factory=set)  # items a card is through with: what 「3/12 条」 counts
    known: set[ItemPath] = field(default_factory=set)  # every item path the job has (from the plan, as the items
    # become known: a block's list is only known once the node above it cooked)
    closed: bool = False  # the driver is through: whatever could be cooked is (a card still waiting stops waiting)
    on_progress = None  # the farm sets it: 「3/12 条」 told to whoever follows the job

    # ------------------------------------------------------------------ the cards

    def joins(self, card: str) -> None:
        with self.cond:
            self.cards.add(card)
            self.cond.notify_all()

    def leaves(self, card: str) -> None:
        with self.cond:
            self.cards.discard(card)
            self.cond.notify_all()

    # ------------------------------------------------------------------ who cooks what

    def mine(self, card: str, inst: Inst) -> bool:
        with self.cond:
            self.known.add(inst.path)
            owner = self.owners.get(inst.path)
            if owner is not None:
                return owner == card
            if inst.path and len(self.cards) > 1 and self._busy(card):
                return False  # this card is on another item and another card is here: leave this one for it
            self.owners[inst.path] = card
            return True

    def close(self) -> None:
        """The driver's cook ended: nothing more will be cooked, so no card waits for anything any more (a job that
        failed in the middle leaves instances nobody will ever finish; without this a borrowed card would wait for
        them until the job's own timeout)."""
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def wait(self, card: str, inst: Inst) -> bool:
        """True: another card is through with it (or nothing more will be: `closed`, `stop`). False: nobody took it,
        so it is `card`'s now (`mine` says so too)."""
        with self.cond:
            until = time.monotonic() + GRACE_S
            while not self.stop.is_set() and not self.closed:
                if inst in self.finished:
                    return True
                owner = self.owners.get(inst.path)
                if owner == card:
                    return False
                if owner is None and (len(self.cards) <= 1 or time.monotonic() >= until):
                    self.owners[inst.path] = card  # nobody came for it
                    return False
                self.cond.wait(0.05)
            return True  # stopped or closed: this card has nothing left to do here

    def has_ended(self, inst: Inst) -> bool:
        with self.cond:
            return inst in self.finished

    def elsewhere(self, card: str, inst: Inst) -> bool:
        """Another card took that item (so this one can only wait for it)."""
        with self.cond:
            owner = self.owners.get(inst.path)
            return owner is not None and owner != card

    def _busy(self, card: str) -> bool:
        """(Holding the lock) is `card` on an item of its own that is not through yet?"""
        return any(p and owner == card and p not in self.counted for p, owner in self.owners.items())

    def ended(self, inst: Inst) -> None:
        """That instance came to an end on some card (its result, its error or its skip was said)."""
        with self.cond:
            self.finished.add(inst)
            self.cond.notify_all()

    def item_done(self, path: ItemPath) -> None:
        """A card is through with that item (it moved on to another one, or its cook ended)."""
        with self.cond:
            if not path or path in self.counted:
                return
            self.counted.add(path)
            self.cond.notify_all()
        if self.on_progress is not None:
            self.on_progress(self.progress())

    # ------------------------------------------------------------------ what the queue shows and the farm asks

    def note(self, paths) -> None:
        """The item paths the job's plan has now (the farm reads them from the driver's evaluation as the items
        become known)."""
        with self.cond:
            self.known.update(paths)

    def free_items(self) -> int:
        """Items nobody has taken yet: whether another card would have anything to do at all."""
        with self.cond:
            return sum(1 for p in self.known if p and p not in self.owners)

    def progress(self) -> dict:
        """What the queue shows: 「3/12 条」 — the items of the job that are through, of those known so far. A job
        with no block has one unit and says nothing (items 0)."""
        with self.cond:
            items = [p for p in self.known if p]
            return {"done": len([p for p in self.counted if p]), "items": len(items),
                    "running": len([p for p, owner in self.owners.items() if p and p not in self.counted])}

    def wake(self) -> None:
        with self.cond:
            self.cond.notify_all()


class Units:
    """One card's view of the ledger, as engine/cook.py takes it (`cook(units=...)`). A card cooks one item at a
    time, so the moment it takes another one, the one before is through (`item_done`) — which is what 「3/12 条」
    counts, with nothing else having to report it."""

    def __init__(self, ledger: Ledger, card: str) -> None:
        self.ledger, self.card, self.on = ledger, card, None

    def mine(self, inst: Inst) -> bool:
        if not self.ledger.mine(self.card, inst):
            return False
        self._took(inst.path)
        return True

    def wait(self, inst: Inst) -> bool:
        if self.ledger.wait(self.card, inst):
            return True
        self._took(inst.path)  # nobody came for it: this card has it now
        return False

    def done(self, inst: Inst) -> bool:
        """Is that instance through (on any card)? What the engine asks before taking what comes after it."""
        return self.ledger.has_ended(inst)

    def others(self, inst: Inst) -> bool:
        return self.ledger.elsewhere(self.card, inst)

    def _took(self, path: ItemPath) -> None:
        if self.on is not None and self.on != path:
            self.ledger.item_done(self.on)
        self.on = path

    def over(self) -> None:
        """This card's cook ended: the item it was on is through."""
        if self.on is not None:
            self.ledger.item_done(self.on)
            self.on = None


def watch(ledger: Ledger, emit):
    """An `emit` that tells the ledger when an instance came to an end, and passes every event on unchanged. The
    engine says node_done / error / skipped per instance (with its item path), so nothing else has to be taught."""
    ends = ("node_done", "error", "skipped")

    def emitting(event: dict) -> None:
        if event.get("type") in ends and event.get("node"):
            ledger.ended(Inst(event["node"], tuple(event.get("path") or ())))
        emit(event)

    return emitting
