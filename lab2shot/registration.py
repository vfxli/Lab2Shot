"""Registering oneself (自行注册): an account made by the person who will use it, on the login page.

An account made this way is an ordinary account: accounts.create makes it, with the same checks as an account the
administrator makes (username, password, Chinese name, 环节, tags), in the same users table, and it is managed the same
way afterwards (用户: edit, extend, disable, quota, delete). Its role is always 普通用户. What the settings group 注册
decides for it (lab2shot/config.py register.*): whether registering is open at all (开放注册, off by default), whether it
needs an invite code (邀请码验证), how long the account lasts (注册账号有效期), its disk quota (注册账号配额) and what it
may use (注册可用模型类别, the same tags as 用户's 可用). The only thing kept besides the account is where it came
from (the `registrations` table: when, from which address and with which code), for the invite list's 「谁用它注册了」,
the limits below and bulk disabling.

Designed against someone who can send from as many addresses as they like: every limit is the server's, so calling
the API directly bypasses none of them (the HTTP side, server/register.py, adds the proof of work, the decoy field and
the minimum time to fill the form, and the per-client request rates):

    site-wide     at most register.per_hour accounts an hour and register.per_day a day, from everyone together;
                  beyond that registering pauses by itself (paused()), the admin overview says so, and the
                  registration that reached the limit is logged (W-REGISTER-PAUSED)
    per address   at most PER_IP accounts from one address and PER_NET from its network (/24, IPv6 /64) within an
                  hour and a day, when the address tells this client apart (server/auth.py client_source): behind a
                  tunnel that only passes the connection through, every request is this machine's loopback, and
                  counting that would stop everyone after the first few; then only the site-wide limits count
    invite codes  (邀请码, when 邀请码验证 is on): each with an optional number of uses and expiry, and a switch; a
                  code is claimed in the same transaction that makes the account, so two registrations never both
                  take its last use. The code is kept, so the 邀请码 page can show and copy it at any time; a typed code
                  is looked up by its sha256 and then compared in constant time; logs and audit lines name a code
                  by its first HINT characters only, never in full. A wrong, disabled, expired or used-up
                  code all get the same answer (E-REGISTER-INVITE): no one learns whether a code exists.

A username already taken is refused with the same message as on the admin page: someone choosing a username has to
learn that it is taken. It is the last check, after the proof of work, the limits and the invite code, so learning it
costs as much as registering.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import unicodedata

from . import accounts, logs, roles, terms
from .accounts import Actor, User, now
from .config import settings
from .database import db
from .errors import Forbidden, Invalid, NotFound, TooManyTries
from .messages import Msg
from .nodes import tags as node_tags
from .periods import Periods, local_day
from .text import plain_text

log = logs.get("auth")

# A random code: CODE_LEN characters of CODE_CHARS (no 0/O, 1/I/L), shown in groups of four: about 79 bits, never
# guessed. A code the administrator types is TYPED_MIN..TYPED_MAX letters and digits.
CODE_CHARS = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
CODE_LEN = 16
TYPED_MIN, TYPED_MAX = 6, 32
HINT = 4  # the code's characters logs and audit lines name it by
NOTE_MOST = 64  # characters of a code's 备注 (made plain text: text.py plain_text)
USES_MOST = 100_000

HOUR, DAY = 3600, 86400
PER_IP = {HOUR: 3, DAY: 10}  # accounts one address may register within an hour, a day
PER_NET = {HOUR: 6, DAY: 30}  # the same for its /24 (IPv6: /64): the next address over is usually the same sender


# ------------------------------------------------------------------ invite codes


def normal_code(text: object) -> str:
    """A code as it is kept and compared: NFC, upper case, without the spaces and dashes a person adds when copying it."""
    s = unicodedata.normalize("NFC", str(text or "")).upper()
    return "".join(ch for ch in s if ch not in " -\t")


def _hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def random_code() -> str:
    return "".join(secrets.choice(CODE_CHARS) for _ in range(CODE_LEN))


def shown_code(code: str) -> str:
    """A code as it is handed over: groups of four (ABCD-EFGH-...); normal_code reads it back."""
    return "-".join(code[i:i + 4] for i in range(0, len(code), 4))


def _typed(text: object) -> str:
    code = normal_code(text)
    if not (TYPED_MIN <= len(code) <= TYPED_MAX) or not code.isascii() or not code.isalnum():
        raise Invalid(Msg("E-INVITE-CODE", min=TYPED_MIN, max=TYPED_MAX))
    return code


def _uses(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= USES_MOST:
        raise Invalid(Msg("E-INVITE-USES", most=USES_MOST))
    return value


def _expiry(value: object, t: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= t:
        raise Invalid(Msg("E-INVITE-EXPIRYPAST"))
    return float(value)


def _state(r, t: float) -> str:
    """Whether a code can be used now, in words (the invite list)."""
    if not r["enabled"]:
        return "已停用"
    if r["expires"] is not None and r["expires"] <= t:
        return "已过期"
    if r["uses_max"] is not None and r["used"] >= r["uses_max"]:
        return "已用完"
    return "可以用"


def _invite(r, t: float, users: list[dict]) -> dict:
    return {"id": r["id"], "code": r["code"], "hint": r["hint"], "note": r["note"], "uses_max": r["uses_max"], "used": r["used"],
            "expires": r["expires"], "enabled": bool(r["enabled"]), "created": r["created"], "created_by": r["created_by"],
            "state": _state(r, t), "usable": _state(r, t) == "可以用", "accounts": users}


def invites() -> list[dict]:
    """Every code, newest first, each with the accounts registered with it (the account as it is now)."""
    t = now()
    who: dict[int, list[dict]] = {}
    for r in db().rows("SELECT g.invite_id, g.at, g.ip, u.id, u.username, u.name, u.enabled, u.deleted FROM registrations g "
                       "JOIN users u ON u.id = g.user_id WHERE g.invite_id IS NOT NULL ORDER BY g.at"):
        who.setdefault(r["invite_id"], []).append({"id": r["id"], "username": r["username"], "name": r["name"], "at": r["at"],
                                                   "ip": r["ip"], "enabled": bool(r["enabled"]), "deleted": r["deleted"]})
    return [_invite(r, t, who.get(r["id"], [])) for r in db().rows("SELECT * FROM invites ORDER BY created DESC, id DESC")]


def invite(invite_id: int) -> dict:
    r = db().row("SELECT * FROM invites WHERE id = ?", (invite_id,))
    if r is None:
        raise NotFound(Msg("E-INVITE-NONE"))
    return next(i for i in invites() if i["id"] == invite_id)


def create_invite(code: str | None, note: object, uses_max: object, expires: object, by: Actor) -> dict:
    """A new code: the one typed (checked, and never one that exists already) or a random one (kept in groups of four,
    as it is handed over). Returns the code's row, which holds the code."""
    t = now()
    kept = _typed(code) if code else random_code()
    shown = kept if code else shown_code(kept)
    uses, until, text = _uses(uses_max), _expiry(expires, t), plain_text(note, NOTE_MOST)
    with db().write() as c:
        if c.execute("SELECT 1 FROM invites WHERE code_hash = ?", (_hash(kept),)).fetchone():
            raise Invalid(Msg("E-INVITE-TAKEN"))
        new = c.execute("INSERT INTO invites (code, code_hash, hint, note, uses_max, used, expires, enabled, created, created_by, "
                        "created_by_id) VALUES (?, ?, ?, ?, ?, 0, ?, 1, ?, ?, ?)",
                        (shown, _hash(kept), kept[:HINT], text, uses, until, t, by.label, by.id)).lastrowid
    return invite(new)


def change_invite(invite_id: int, note: object, uses_max: object, expires: object, enabled: bool) -> dict:
    """Set a code's 备注, 可用次数 (None: no limit; never below what it was already used), 到期 (None: never) and
    switch. An expiry already past is refused like a new one: to stop a code at once, switch it off."""
    old = invite(invite_id)
    uses, text = _uses(uses_max), plain_text(note, NOTE_MOST)
    until = old["expires"] if expires == old["expires"] else _expiry(expires, now())
    if uses is not None and uses < old["used"]:
        raise Invalid(Msg("E-INVITE-USESBELOW", used=old["used"]))
    with db().write() as c:
        c.execute("UPDATE invites SET note = ?, uses_max = ?, expires = ?, enabled = ? WHERE id = ?",
                  (text, uses, until, int(bool(enabled)), invite_id))
    return invite(invite_id)


def delete_invite(invite_id: int) -> dict:
    """Remove a code. The accounts registered with it stay as they are; their registrations keep the code's hint."""
    old = invite(invite_id)
    with db().write() as c:
        c.execute("DELETE FROM invites WHERE id = ?", (invite_id,))
    return old


def _usable_row(row, typed: str, t: float) -> bool:
    return row is not None and hmac.compare_digest(normal_code(row["code"]).encode(), typed.encode()) and _state(row, t) == "可以用"


def invite_usable(code: object) -> bool:
    """Whether a code could be used now (read only: registering claims it, _claim). For the HTTP side's count of
    wrong codes (server/register.py, auth.guarded), which must count a wrong code and nothing else."""
    typed = normal_code(code)[:TYPED_MAX + 8]
    return bool(typed) and _usable_row(db().row("SELECT * FROM invites WHERE code_hash = ?", (_hash(typed),)), typed, now())


def _claim(c, code: str, t: float):
    """Take one use of a code inside the caller's transaction: its row, or E-REGISTER-INVITE for any code that cannot
    be used (unknown, off, expired, used up: one answer for all)."""
    typed = normal_code(code)[:TYPED_MAX + 8]
    r = c.execute("SELECT * FROM invites WHERE code_hash = ?", (_hash(typed),)).fetchone() if typed else None
    if not _usable_row(r, typed, t):
        raise Invalid(Msg("E-REGISTER-INVITE"))
    c.execute("UPDATE invites SET used = used + 1 WHERE id = ?", (r["id"],))
    return r


# ------------------------------------------------------------------ the limits


def net_of(ip: str) -> str:
    """The network an address counts under (/24, IPv6 /64); "" for what is not an address."""
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return ""
    return str(ipaddress.ip_network(f"{a}/{24 if a.version == 4 else 64}", strict=False))


def counts(t: float | None = None) -> dict:
    """Accounts registered by everyone within the last hour and day, against the limits, and whether that pauses
    registering now: for the admin overview, the settings' 注册 group and every registration."""
    t = now() if t is None else t
    s = settings()
    hour = db().row("SELECT COUNT(*) AS n FROM registrations WHERE at > ?", (t - HOUR,))["n"]
    day = db().row("SELECT COUNT(*) AS n FROM registrations WHERE at > ?", (t - DAY,))["n"]
    per_hour, per_day = int(s["register.per_hour"]), int(s["register.per_day"])
    paused = "hour" if hour >= per_hour else "day" if day >= per_day else ""
    return {"open": bool(s["register.open"]), "invite": bool(s["register.invite"]), "hour": hour, "day": day,
            "per_hour": per_hour, "per_day": per_day, "paused": paused}


def new_accounts(p: Periods) -> dict:
    """The admin overview's 注册: accounts made in 今日, 本周 and 本月, all of them and those that registered themselves
    (the rest an administrator made: 后台「用户」, the command line). A made account deleted since still counts (its row
    stays); one removed for good (永久删除) does not, nor the built-in administrator."""
    rows = db().rows(f"SELECT {local_day('u.created')} AS day, r.user_id IS NOT NULL AS self, COUNT(*) AS n FROM users u "
                     "LEFT JOIN registrations r ON r.user_id = u.id WHERE u.created >= ? AND u.id != ? GROUP BY 1, 2",
                     (p.since, accounts.ADMIN_ID))
    shown = ("today", "week", "month")
    made = p.sums(((r["day"], r["n"]) for r in rows), shown)
    themselves = p.sums(((r["day"], r["n"]) for r in rows if r["self"]), shown)
    return {k: {"all": made[k], "self": themselves[k]} for k in shown}


def paused(t: float | None = None) -> Msg | None:
    """Why registering is paused now (the site-wide limits), or None."""
    c = counts(t)
    return Msg("E-REGISTER-PAUSED") if c["paused"] else None


def address_problem(ip: str, t: float) -> Msg | None:
    """Why this address may not register another account now (PER_IP, PER_NET), or None."""
    net = net_of(ip)
    for window, most in PER_IP.items():
        if db().row("SELECT COUNT(*) AS n FROM registrations WHERE ip = ? AND at > ?", (ip, t - window))["n"] >= most:
            return Msg("E-REGISTER-ADDRESS")
    for window, most in PER_NET.items():
        if db().row("SELECT COUNT(*) AS n FROM registrations WHERE net = ? AND at > ?", (net, t - window))["n"] >= most:
            return Msg("E-REGISTER-ADDRESS")
    return None


# ------------------------------------------------------------------ registering


def register(username: object, name: object, department: object, password: str, again: str, code: object,
             agreed: int, ip: str, apart: bool) -> tuple[User, dict | None]:
    """Make the account of someone registering themself, if registering is open, they agreed to the 用户协议 and
    隐私政策 and no limit stands in the way: accounts.create with the settings' expiry and tags and the role 普通用户,
    the invite code's use (when codes are required), the agreement (lab2shot/terms), its quota and the record of where
    it came from, all in one transaction, which the new password is hashed before (accounts.new_password: never under
    the database's one writer). `agreed`: the version of the texts the form showed with
    its box ticked (0: not ticked); one that is no longer current is refused, to be read again. Returns the account and
    the code's row it used (None without one). `apart`: whether `ip` tells this client apart from others
    (server/auth.py client_source); only then do the per-address limits count. The caller has checked the proof of
    work."""
    s, t = settings(), now()
    if not s["register.open"]:
        raise Forbidden(Msg("E-REGISTER-CLOSED"))
    if not agreed:
        raise Invalid(Msg("E-REGISTER-TERMS"))
    if password != again:
        raise Invalid(Msg("E-REGISTER-AGAIN"))
    if s["register.invite"] and not str(code or "").strip():
        raise Invalid(Msg("E-REGISTER-NOINVITE"))
    expires, hashed = t + int(s["register.days"]) * DAY, accounts.new_password(password)
    with db().write() as c:  # the limits are read under the write lock: two registrations at once never both pass the last place
        if problem := paused(t):
            raise TooManyTries(problem)
        if apart and (problem := address_problem(ip, t)):
            raise TooManyTries(problem)
        used = _claim(c, str(code or ""), t) if s["register.invite"] else None
        user = accounts.create(str(username or ""), hashed, str(name or ""), str(department or ""), expires,
                               list(s["register.tags"]), role=roles.DEFAULT, by="本人")
        c.execute("INSERT INTO registrations (user_id, at, invite_id, invite_hint, ip, net) VALUES (?, ?, ?, ?, ?, ?)",
                  (user.id, t, used["id"] if used else None, used["hint"] if used else "", ip[:100], net_of(ip)))
        terms.record(c, user.id, agreed, ip)
        # its quota, the settings' for a self-registered account (server/quota.py reads users.quota_gb)
        c.execute("UPDATE users SET quota_gb = ? WHERE id = ?", (float(s["register.quota_gb"]), user.id))
    after = counts(t + 0.001)
    if after["paused"]:  # this one reached a site-wide limit: registering pauses from now on
        window = "一小时" if after["paused"] == "hour" else "一天"
        most = after["per_hour"] if after["paused"] == "hour" else after["per_day"]
        logs.say(log, Msg("W-REGISTER-PAUSED", window=window, count=after[after["paused"]], most=most))
    return user, (dict(used) if used else None)


def invite_label(used: dict | None) -> str:
    """The code a registration used, as the logs say it: its hint and 备注, never the code."""
    if used is None:
        return "没用邀请码"
    return f"{used['hint']}…" + (f"（{used['note']}）" if used["note"] else "")


def tags_label(given: frozenset[str] | list[str]) -> str:
    return "、".join(node_tags.TAGS[t].label for t in sorted(given)) or "只有基础"


# ------------------------------------------------------------------ bulk disabling


def registered(invite_id: int | None = None, since: float | None = None, until: float | None = None) -> list[dict]:
    """The accounts that registered themselves with this code, or within this time (both: both), oldest first, as
    they are now (not deleted ones)."""
    if invite_id is None and since is None and until is None:
        raise Invalid(Msg("E-REGISTER-FILTER"))
    rows = db().rows("SELECT u.*, g.at AS registered, g.invite_hint, g.ip AS registered_ip FROM registrations g "
                     "JOIN users u ON u.id = g.user_id WHERE u.deleted IS NULL "
                     "AND (? IS NULL OR g.invite_id = ?) AND (? IS NULL OR g.at >= ?) AND (? IS NULL OR g.at < ?) "
                     "ORDER BY g.at", (invite_id, invite_id, since, since, until, until))
    return [{"id": r["id"], "username": r["username"], "name": r["name"], "department": r["department"], "role": r["role"],
             "enabled": bool(r["enabled"]), "registered": r["registered"], "invite": r["invite_hint"],
             "ip": r["registered_ip"]} for r in rows]


def disable(ids: list[int]) -> list[str]:
    """Switch these accounts off (accounts.update, as 用户 does: their sessions end at once). Returns their usernames."""
    done = []
    for i in ids:
        u = accounts.update(i, enabled=False)
        done.append(u.username)
    return done

