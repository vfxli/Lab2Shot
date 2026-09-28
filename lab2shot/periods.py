"""The periods the admin overview's 「今天和最近」 counts in, by the server's local time: 今日 (since midnight),
近 7 天 (today and the six days before it), 本周 (since Monday) and 本月 (since the 1st).

The counting modules (accounts, registration, farm/usage, feedback, traffic) each read their rows per local day
since the earliest of the starts, in one indexed query (`SELECT date(<time>, 'unixepoch', 'localtime') ...
WHERE <time> >= ?  GROUP BY 1`), and sum them here: a period is a range of days, so a row is in it when its day is.
SQLite's 'localtime' and Python's are the same process's time zone, so both name a moment by the same day."""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Iterable
from dataclasses import dataclass

@dataclass(frozen=True)
class Periods:
    """The first day (YYYY-MM-DD, local) of each period; `start`, the earliest of them, and `since`, its midnight as
    seconds since the epoch (what the queries take)."""

    today: str
    days7: str
    week: str
    month: str
    start: str
    since: float

    def first(self, period: str) -> str:
        return getattr(self, period)

    def sums(self, rows: Iterable[tuple[str, int]], periods: Iterable[str]) -> dict[str, int]:
        """(day, n) rows summed into each of `periods`: the rows of the days since its first day."""
        rows = list(rows)
        return {p: sum(n for day, n in rows if day >= self.first(p)) for p in periods}

    def distinct(self, rows: Iterable[tuple[str, object]], periods: Iterable[str]) -> dict[str, int]:
        """How many different things the (day, thing) rows name within each of `periods` (people who logged in)."""
        rows = list(rows)
        return {p: len({thing for day, thing in rows if day >= self.first(p)}) for p in periods}


def now(t: float | None = None) -> Periods:
    """The periods that hold the moment `t` (now when None)."""
    today = dt.datetime.fromtimestamp(time.time() if t is None else t).date()
    days7 = today - dt.timedelta(days=6)
    week = today - dt.timedelta(days=today.weekday())  # Monday
    month = today.replace(day=1)
    start = min(days7, week, month)
    since = dt.datetime.combine(start, dt.time()).timestamp()  # local midnight
    return Periods(today.isoformat(), days7.isoformat(), week.isoformat(), month.isoformat(), start.isoformat(), since)


# the day a stored moment falls on, local: the column every counting query groups by (the column name is the caller's
# own, never anything a user sent)
def local_day(column: str) -> str:
    return f"date({column}, 'unixepoch', 'localtime')"
