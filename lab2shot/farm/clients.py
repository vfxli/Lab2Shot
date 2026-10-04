"""Who started a job: its account (lab2shot/accounts.py; every request carries one), and what the request showed
about where it came from, for the administrator (the address, the User-Agent, and what a client says of itself: a DCC
plugin's application, computer name and OS user, a browser's platform and screen). The account is who it is; the rest
is only what was seen. Client does not keep the account's Chinese name and department as its own fields: `who` and
`full()` (below) read them from the account, fresh, every time — a renamed or moved account never leaves a stale copy
sitting in a job's record (history, usage and feedback read the same way: joining the live users table, not the
record's old snapshot).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Client:
    user: int  # the account: whose job it is
    username: str
    app: str = "web"  # web / maya / houdini / nuke / cli / script ...
    details: dict = field(default_factory=dict)  # auth.details(): ip, user_agent, hostname, os user, platform ...

    @property
    def who(self) -> str:
        """Short label: 张三（zhangsan）, 「已删除的用户」 once deleted (the account's current name: accounts.User.label)."""
        from .. import accounts
        from ..errors import NotFound

        try:
            return accounts.get(self.user).label
        except NotFound:  # the account row was purged; its jobs in memory and in the records still say whose they were
            return accounts.deleted_label()

    def full(self) -> dict:
        """What the administrator sees (and the job's record keeps): its own fields plus the account's current name,
        department and label — read live, never a client's own (a client's `name` or `id` is only ever in `details`,
        with no special meaning). A purged account reads as 「已删除的用户」 rather than raising, since its jobs can
        outlive the account row in memory and the admin queue must still list them."""
        from .. import accounts
        from ..errors import NotFound

        try:
            a = accounts.get(self.user)
        except NotFound:
            return {**asdict(self), "who": accounts.deleted_label(), "name": accounts.deleted_label(), "department": ""}  # its words
        return {**asdict(self), "who": a.label, "name": a.name, "department": accounts.department_label(a.department)}  # its words

    @classmethod
    def of(cls, account, details: dict | None = None) -> Client:
        """The client of a request by `account` (accounts.User), with what the request showed."""
        said = dict(details or {})
        return cls(account.id, account.username, str(said.pop("app", "") or "web")[:40], said)

    @classmethod
    def from_record(cls, record: dict, account) -> Client:
        """A job's client again, from its record (a job queued again after a restart)."""
        return cls.of(account, {**record.get("details", {}), "app": record.get("app", "web")})
