"""User accounts: who may use this server, and what is theirs.

Every user logs in with an account. Each account has a role (lab2shot/roles.py: 管理员, 二级管理员, 普通用户), which
determines what it may do beyond the editor. The built-in administrator account (admin, ADMIN_ID; it has no password
until one is set with `./setup.sh`) is always 管理员. The 管理员 and 二级管理员 create other accounts on the admin page
(用户); while 开放注册 is on, people also make their own on the login page (lab2shot/registration.py), through the
same create() and with the same checks. Each account has a username; a password initially set by the administrator
(the user may change it by supplying the old one); a Chinese name and a 环节 (setting people.departments, used by the
statistics); an expiry date (after which login is refused and live sessions end immediately; the administrator can
extend it); an enabled flag; and the tags of what it may use (nodes/tags.py: 可商用 unless the administrator grants
more). A user sees only their own jobs, outputs, uploads, results and feedback (server/access.py); the administrator
sees everything.

Each account may be logged in in only one place per kind. 在线 means real presence, not a live login: the server keeps
in memory when each web/client session last made a request (session(), on every request; nothing is written for
it), and a session that made one within ONLINE_S is online (presence()). An open page always makes one at least once a
minute (its poll of the server's state), so a page left open counts and a closed one drops out within ONLINE_S. After
a restart that memory is empty, and the last activity shown falls back to sessions.seen (written at most every
SEEN_EVERY_S) and the last login. The admin page reports this per account, never a raw session count.

A new login of kind "web" (a browser) or "client" (a DCC plugin or `lab2shot login`) immediately ends every other
session of the same account and kind (start()). The displaced page learns this on its next request
(a 401 stating when, from where and from which device the new login came: server/access.py, server/auth.py) and
offers 重新登录, which in turn displaces the newer session. The machine token (kind "machine") is exempt and never
counts as online, nor as anyone's activity.

Stored in the database (database/schema.py); passwords are kept only as salted slow hashes (scrypt), never in plaintext:

    users       the accounts; a deleted account row remains (its jobs appear in the statistics as 「已删除的用户」)
                and keeps its username: nobody can take it until the account is purged (create)
    sessions    the credential a browser (a cookie), a client (DCC plugin or command line: a token in ~/.lab2shot/)
                or the command line on this machine (kind "machine": machine_token) holds after login, stored as its
                sha256 only. A session lasts SESSION_S (TOKEN_S for a client) from its last use; a machine-token
                session lasts MACHINE_S from creation and is replaced once it is MACHINE_ROTATE_S old
                (machine_token). Administrator rights on a browser session last ADMIN_S from password entry, after
                which the admin page asks for the password again (the editor remains usable). A web/client row ended
                by a later login of the same account is kept for a while with `replaced_at`/`replaced_by`, which the
                displaced page reads, and then removed (prune_logins)
    login_log   every login attempt (successful or not, with the reason), for the 用户 page's 最近登录; kept for
                LOGIN_LOG_KEPT_S, then removed (prune_logins, the queue's tidy); of the failed ones only the newest
                FAILED_KEPT per account, and of usernames that are no account (log_login)
    meta        the 口令 (recovery for the administrator's password; set only by `lab2shot admin passphrase` on
                this machine)

The machine token's plaintext is never stored in the database (and therefore never in a backup). It exists only in
MACHINE_FILE under the work folder, readable only by the account that runs the server (0600); the server accepts it
only from this machine's own loopback address with no proxy in between (server/auth.py credential).
"""

from __future__ import annotations

import hashlib
import os
import hmac
import re
import secrets
import time
from dataclasses import dataclass

from . import roles
from .config import settings
from .database import db, json_of, json_text
from .errors import Invalid, NotFound
from .messages import Msg
from .nodes import tags as node_tags
from .periods import Periods, local_day

ADMIN_ID = 1  # the baseline migration creates the administrator's account first (database/schema.py)
ADMIN_NAME = "admin"  # the built-in administrator's username on a new installation
DELETED = "已删除的用户"  # label under which a deleted account's jobs are counted

MIN_CHARS, MAX_CHARS = 8, 128  # length limits for a password and the 口令
SESSION_S = 30 * 86400  # a browser stays logged in this long after its last use
TOKEN_S = 180 * 86400  # the same for a DCC plugin or the command line
ADMIN_S = 3 * 86400  # duration of administrator rights on a browser, from password entry
SEEN_EVERY_S = 600  # a session's last use is written at most this often
# 在线: a web/client session that made a request within this many seconds (presence()). An open page asks the server
# at least once a minute (the editor's /api/load poll, webui/src/editor/Chrome.tsx, backs off to 60 s at most; a page
# without it polls webui/src/state/server.ts, which then does the same), so the window is that minute plus room for a
# slow answer; a tab the browser hides stops asking and drops out.
ONLINE_S = 90
MACHINE_S = 7 * 86400  # lifetime of a machine-token session from creation
MACHINE_ROTATE_S = 86400  # a machine token older than this is replaced on request (the old one ends immediately)
MACHINE_FILE = "auth/machine-token"  # plaintext token under the work folder, mode 0600, never in the database

LOGIN_LOG_KEPT_S = 180 * 86400  # login_log entries older than this are removed (prune_logins, the queue's tidy)
FAILED_KEPT = 1000  # failed attempts kept in login_log per account, and for usernames that are no account (log_login)
ENDED_LOGGED = 20  # at most this many displaced sessions are listed in one login_log row
REPLACED_KEPT_S = 86400  # retention of a displaced session row (its notice) or a lapsed one

# 异地频繁 (the 用户 page's 最近登录 panel): more than this many distinct IPs or devices within 7 days, or this many
# failures on one day, is shown as a warning. The tooltip states the same thresholds.
SUSPICIOUS_IPS_7D = 3
SUSPICIOUS_DEVICES_7D = 3
SUSPICIOUS_FAILURES_DAY = 5

# scrypt: 32 MB and ~0.1 s per check (the server checks one at a time: server/auth.py)
COST = 2**15
BLOCK = 8
PARALLEL = 1

PASSPHRASE = "auth.passphrase"

USERNAME = re.compile(r"^[a-z][a-z0-9_.-]{2,31}$")
NAME_MIN, NAME_MAX = 2, 6
DOTS = "·•・‧∙"  # middle-dot variants accepted in input; normalised to ·
_HAN = r"㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f"
_NAME = re.compile(rf"^[{_HAN}]+(·[{_HAN}]+)*$")


def now() -> float:
    """The clock for accounts, sessions, registrations and expiry."""
    return time.time()


# ------------------------------------------------------------------ hashes and rules


def hash_secret(text: str) -> str:
    """scrypt$<n>$<r>$<p>$<salt>$<key>: the parameters are stored with the hash, so they can be strengthened later."""
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(text.encode("utf-8"), salt=salt, n=COST, r=BLOCK, p=PARALLEL, maxmem=2**27, dklen=32)
    return f"scrypt${COST}${BLOCK}${PARALLEL}${salt.hex()}${key.hex()}"


def matches(text: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, key = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(text.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), maxmem=2**27,
                             dklen=len(key) // 2)
    except ValueError:
        return False
    return hmac.compare_digest(got.hex(), key)


_DUMMY: list[str] = []


def dummy_check(text: str) -> None:
    """Take as long as a password check, so an unknown username costs the same time as a wrong password."""
    if not _DUMMY:
        _DUMMY.append(hash_secret(secrets.token_hex(8)))
    matches(text, _DUMMY[0])


def sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


# a browser's User-Agent, checked in order (Edge and Opera's also carry Chrome's and Safari's own tokens)
_BROWSERS = (("Edg/", "Edge"), ("OPR/", "Opera"), ("Chrome/", "Chrome"), ("CriOS/", "Chrome"), ("Firefox/", "Firefox"),
             ("FxiOS/", "Firefox"), ("Safari/", "Safari"))
_SYSTEMS = (("Windows", "Windows"), ("Mac OS X", "macOS"), ("Android", "Android"), ("iPhone", "iOS"), ("iPad", "iOS"),
            ("Linux", "Linux"))


def device_label(kind: str, agent: str, app: str = "") -> str:
    """The short device label shown for a login (最近登录, the displaced-session notice). A browser cannot identify
    the specific computer it runs on (只有浏览器名和系统), so "Chrome · Windows" means "this browser on this computer",
    as the 用户 page's tooltip explains. A client (a DCC plugin, the command line) is named by its self-reported
    `app` (maya, houdini, nuke, cli, ...), not by the User-Agent of its HTTP library."""
    if kind != "web":
        return app or "客户端"
    browser = next((label for key, label in _BROWSERS if key in agent), "浏览器")
    system = next((label for key, label in _SYSTEMS if key in agent), "未知系统")
    return f"{browser} · {system}"


def rule_problem(text: str, what: str = "密码") -> Msg | None:
    """Why a new password or 口令 cannot be used (None if it can); the pages apply the same check before submitting.
    `what` is the term used in the message (密码, 新密码, 口令)."""
    if len(text) < MIN_CHARS:
        return Msg("E-ACCOUNT-SECRETSHORT", what=what, min=MIN_CHARS)
    if len(text) > MAX_CHARS:
        return Msg("E-ACCOUNT-SECRETLONG", what=what, max=MAX_CHARS)
    if not text.strip():
        return Msg("E-ACCOUNT-SECRETBLANK", what=what)
    return None


def check_username(text: object) -> str:
    name = str(text or "").strip().lower()
    if not USERNAME.match(name):
        raise Invalid(Msg("E-ACCOUNT-USERNAME"))
    return name


def check_name(name: object) -> str:
    """The Chinese name in stored form (trimmed, middle dots normalised); raises Invalid stating what is wrong."""
    text = str(name or "").strip()
    for dot in DOTS:
        text = text.replace(dot, "·")
    if not text:
        raise Invalid(Msg("E-ACCOUNT-NONAME", min=NAME_MIN, max=NAME_MAX))
    if re.search(r"\s", text):
        raise Invalid(Msg("E-ACCOUNT-NAMESPACE", name=text))
    if not _NAME.match(text):
        if re.search(r"[A-Za-z]", text):
            raise Invalid(Msg("E-ACCOUNT-NAMELATIN", name=text))
        raise Invalid(Msg("E-ACCOUNT-NAMECHARS", name=text))
    chars = len(text.replace("·", ""))
    if not NAME_MIN <= chars <= NAME_MAX:
        raise Invalid(Msg("E-ACCOUNT-NAMELENGTH", name=text, count=chars, min=NAME_MIN, max=NAME_MAX))
    return text


def departments() -> list[str]:
    return list(settings()["people.departments"])


def check_department(dept: object) -> str:
    text = str(dept or "").strip()
    if text not in departments():
        raise Invalid(Msg("E-ACCOUNT-DEPARTMENT", departments=departments()))
    return text


def check_tags(given: object) -> frozenset[str]:
    """The tags an account may be given: any except the implied ones, which every account has."""
    found = frozenset(str(t) for t in (given or []))
    unknown = found - set(node_tags.TAGS)
    if unknown:
        raise Invalid(Msg("E-ACCOUNT-TAGS", tags=sorted(unknown)))
    return found - node_tags.IMPLIED


# ------------------------------------------------------------------ accounts


@dataclass(frozen=True)
class User:
    id: int
    username: str
    name: str
    department: str
    role: str  # admin / deputy / user (lab2shot/roles.py): permissions beyond the editor
    tags: frozenset[str]  # what it may use (nodes/tags.py), excluding the implied tags
    expires: float | None  # None: never (the built-in administrator account)
    enabled: bool
    deleted: float | None
    created: float
    last_login: float | None
    password_set: float | None
    password_by: str
    no_password: bool  # the built-in administrator account has no password yet, so no one can log in as it

    @property
    def capabilities(self) -> frozenset[str]:
        """The capabilities of its role (lab2shot/roles.py); a session holds them only while its rights last (Session)."""
        return roles.capabilities(self.role)

    @property
    def owner(self) -> bool:
        """The built-in administrator account (ADMIN_ID): always 管理员, never disabled, expired or deleted."""
        return self.id == ADMIN_ID

    @property
    def allowed(self) -> frozenset[str] | None:
        """What it may use, for nodes/tags.py may(): None (everything) when it has nodes.all."""
        return None if "nodes.all" in self.capabilities else self.tags

    @property
    def label(self) -> str:
        """Display form: 张三（zhangsan）, or 「已删除的用户」 once deleted."""
        return DELETED if self.deleted else f"{self.name}（{self.username}）"

    def usable_now(self) -> Msg | None:
        """Why the account cannot log in or stay logged in now (None if it can)."""
        if self.deleted:
            return Msg("E-ACCOUNT-GONE")
        if not self.enabled:
            return Msg("E-ACCOUNT-DISABLED")
        if self.expires is not None and self.expires <= now():
            return Msg("E-ACCOUNT-EXPIRED")
        return None

    def public(self) -> dict:
        """The fields sent to the account's own pages."""
        return {"id": self.id, "username": self.username, "name": self.name, "department": self.department,
                "role": self.role, "role_label": roles.label(self.role), "tags": sorted(self.tags), "expires": self.expires}

    def full(self) -> dict:
        """The fields the 用户 page shows: `state` (可以用, or the reason it is not usable; 未设密码 for the built-in
        administrator account without a password) and `state_tip`."""
        problem = self.usable_now()
        state = "未设密码" if self.no_password else problem.text if problem else "可以用"
        tip = "还没设密码：在服务器上执行 ./setup.sh，选「账号与安全 → 设置管理员密码」" if self.no_password else state
        return {**self.public(), "owner": self.owner, "all_nodes": self.allowed is None, "enabled": self.enabled, "deleted": self.deleted, "created": self.created,
                "last_login": self.last_login, "password_set": self.password_set, "password_by": self.password_by,
                "no_password": self.no_password, "usable_now": problem is None, "state": state, "state_tip": tip}


def _user(r) -> User:
    return User(id=r["id"], username=r["username"], name=r["name"], department=r["department"], role=r["role"],
                tags=frozenset(json_of(r["tags"])), expires=r["expires"], enabled=bool(r["enabled"]),
                deleted=r["deleted"], created=r["created"], last_login=r["last_login"],
                password_set=r["password_set"], password_by=r["password_by"],
                no_password=r["id"] == ADMIN_ID and not r["hash"])


def get(user_id: int) -> User:
    r = db().row("SELECT * FROM users WHERE id = ?", (user_id,))
    if r is None:
        raise NotFound(Msg("E-ACCOUNT-NOUSER"))
    return _user(r)


def admin() -> User:
    return get(ADMIN_ID)


def by_username(username: str) -> User | None:
    r = db().row("SELECT * FROM users WHERE username = ? AND deleted IS NULL", (username.strip().lower(),))
    return _user(r) if r else None


def listing() -> list[dict]:
    """Every account (deleted ones last), each with its job count, its last job and its presence (presence(): where
    it is online now, a request within ONLINE_S, and otherwise when and where it was last active; never a raw session
    count). The username on the 用户 page opens 最近登录 for details."""
    rows = db().rows("""
        SELECT u.*, (SELECT COUNT(*) FROM jobs j WHERE j.user_id = u.id) AS jobs,
               (SELECT MAX(submitted) FROM jobs j WHERE j.user_id = u.id) AS last_job
        FROM users u ORDER BY u.deleted IS NOT NULL, CASE u.role WHEN 'admin' THEN 0 WHEN 'deputy' THEN 1 ELSE 2 END, u.created""")
    here = presence()
    nobody = {"online": [], "active": None, "where": None}
    return [{**_user(r).full(), "jobs": r["jobs"], "last_job": r["last_job"],
             # the account's own disk quota (None: the default from the settings). Usage requires a disk scan and is
             # requested separately by the detail page (server/quota.py)
             "quota_gb": r["quota_gb"],
             "presence": here.get(r["id"], nobody)} for r in rows]


def create(username: str, password: str, name: str, department: str, expires: float, allowed: object,
           role: str = roles.DEFAULT, by: str = "管理员") -> User:
    """Create an account (server/users.py checks which roles the caller may create: lab2shot/roles.py). `by`: who
    set its first password (the role label of the creator)."""
    username, role = check_username(username), roles.check(role)
    if problem := rule_problem(password):
        raise Invalid(problem)
    name, department, given = check_name(name), check_department(department), check_tags(allowed)
    if expires <= now():
        raise Invalid(Msg("E-ACCOUNT-EXPIRYPAST"))
    from .library import RESERVED_USERNAMES

    if by_username(username) or username in RESERVED_USERNAMES:  # admin / adapter are the other two template folders (library.py)
        raise Invalid(Msg("E-ACCOUNT-USERNAMETAKEN", username=username))
    # A deleted account keeps its username until it is purged: what is kept under the username (its templates,
    # work/users/<username>/, library.py) is still the deleted account's, and purging it removes that folder.
    if db().row("SELECT 1 FROM users WHERE username = ?", (username,)):
        raise Invalid(Msg("E-ACCOUNT-USERNAMEDELETED", username=username))
    t = now()
    with db().write() as c:
        uid = c.execute("INSERT INTO users (username, hash, name, department, role, tags, expires, enabled, created, "
                        "password_set, password_by) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                        (username, hash_secret(password), name, department, role, json_text(sorted(given)), expires, t, t,
                         by)).lastrowid
    return get(uid)


def update(user_id: int, *, name: str | None = None, department: str | None = None, expires: float | None = None,
           enabled: bool | None = None, allowed: object = None, role: str | None = None) -> User:
    """Change an account (for the built-in administrator account, only its name and department). Disabling it or
    setting an expiry in the past ends its sessions immediately (they are checked on every request); a new role with
    fewer rights also takes effect immediately, while a role with more rights takes effect at the next login, since
    rights are granted on password entry."""
    u = get(user_id)
    if u.deleted:
        raise NotFound(Msg("E-ACCOUNT-NOUSER"))
    changes: dict[str, object] = {}
    if name is not None:
        changes["name"] = check_name(name)
    if department is not None:
        changes["department"] = check_department(department) if department or not u.owner else ""
    if not u.owner:
        if expires is not None:
            changes["expires"] = float(expires)
        if enabled is not None:
            changes["enabled"] = int(bool(enabled))
        if allowed is not None:
            changes["tags"] = json_text(sorted(check_tags(allowed)))
        if role is not None:
            changes["role"] = roles.check(role)
    elif expires is not None or enabled is not None or allowed is not None or (role is not None and role != u.role):
        raise Invalid(Msg("E-ACCOUNT-ADMINFIXED", username=u.username))
    if changes:
        with db().write() as c:
            c.execute(f"UPDATE users SET {', '.join(f'{k} = ?' for k in changes)} WHERE id = ?", (*changes.values(), user_id))
    return get(user_id)


def set_password(user_id: int, new: str, by: str, keep: str | None = None) -> None:
    """Set a new password (by: 本人 / 管理员 / 命令行). Every session of the account ends except `keep` (the token of
    the browser that made the change)."""
    if problem := rule_problem(new, "新密码"):
        raise Invalid(problem)
    with db().write() as c:
        c.execute("UPDATE users SET hash = ?, password_set = ?, password_by = ? WHERE id = ?",
                  (hash_secret(new), now(), by, user_id))
        c.execute("DELETE FROM sessions WHERE user_id = ? AND token != ?", (user_id, sha(keep) if keep else ""))


def password_hash(user_id: int) -> str:
    r = db().row("SELECT hash FROM users WHERE id = ?", (user_id,))
    return r["hash"] if r else ""


def delete(user_id: int) -> dict:
    """Delete an account (never the built-in administrator account). In this order: it is switched off and its
    sessions end (nothing new comes from it), its live jobs stop and are waited for until they have ended (nothing of
    theirs is written any more; E-QUEUE-STILLSTOPPING, the account left switched off, when one does not end), then its
    tasks' outputs are removed (transfer/outputs.py; its cache, uploads and task folders go by housekeeping:
    farm/disk.py) and it is marked deleted. Its job records remain for the statistics (as 「已删除的用户」) and its
    feedback remains for the administrator. Returns what was removed."""
    u = get(user_id)
    if u.owner:
        raise Invalid(Msg("E-ACCOUNT-ADMINDELETE", username=u.username))
    if u.deleted:
        raise NotFound(Msg("E-ACCOUNT-NOUSER"))
    from .farm import farm
    from .transfer import outputs

    with db().write() as c:
        c.execute("UPDATE users SET enabled = 0 WHERE id = ?", (user_id,))
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    stopped = farm().stop_user(user_id)
    gone = outputs.remove_account(user_id)
    with db().write() as c:
        c.execute("UPDATE users SET deleted = ? WHERE id = ?", (now(), user_id))
    return {"jobs_stopped": stopped, "outputs": gone}


# On permanent deletion, these tables only record who did something: the column pointing at the account is cleared
# and the rows are kept (shown as 「已删除的用户」; the account row is gone and the username may be reused). The
# statistics use LEFT JOIN users and count a missing account as 「已删除的用户」 (farm/usage.py _uses). The other tables that name the account (sessions, tasks, task_group_names, registrations) are
# declared ON DELETE CASCADE and are removed with the account row, as permanent deletion requires; the confirmation
# dialog states this. Node graphs saved on the server are files (work/users/<username>/templates/,
# lab2shot/library.py), deleted with the account's folder; its task folders, its cache and its uploads are deleted with
# it (farm/disk.py forget_account), so the job records kept afterwards no longer have a graph.
KEEPS_RECORDS = ("jobs", "feedback", "login_log", "admin_actions")


def purge(user_id: int) -> dict:
    """Permanently delete an already deleted account: the account row is removed and the username may be reused (not
    before: create).

    Job records, feedback, login records and the admin action log are kept (KEEPS_RECORDS) but no longer identify the
    account; the statistics show them as 「已删除的用户」. Returns the number of rows kept per table, which the
    confirmation dialog and the audit log report.

    Applies only to accounts already deleted (deletion and permanent deletion are two separate steps, to prevent
    accidental loss). The built-in administrator account (ADMIN_ID) can never be deleted."""
    u = get(user_id)
    if u.owner:
        raise Invalid(Msg("E-ACCOUNT-ADMINDELETE", username=u.username))
    if not u.deleted:
        raise Invalid(Msg("E-ACCOUNT-NOTDELETED", username=u.username))
    from .farm import farm
    from .farm.disk import forget_account
    from .library import remove_user

    # a job of it still to finish would write into (and make again) the cache and folders removed below; a deleted
    # account gets no new one (no session, and a held job is not queued again for it: farm/queue.py _unpark)
    if live := farm().live_of(user_id):
        raise Invalid(Msg("E-ACCOUNT-JOBSLIVE", username=u.username, count=len(live)))

    kept = {t: db().row(f"SELECT COUNT(*) AS n FROM {t} WHERE user_id = ?", (user_id,))["n"] for t in KEEPS_RECORDS}
    graphs = remove_user(u.username)
    forget_account(user_id)  # its tasks, cache and uploads: an id SQLite may reuse never finds anything of it
    with db().write() as c:
        for table in KEEPS_RECORDS:
            c.execute(f"UPDATE {table} SET user_id = NULL WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return {"jobs": kept["jobs"], "feedback": kept["feedback"], "graphs": graphs}


def login(username: str, password: str) -> User | None:
    """The account these credentials open, or None. An unknown username and a wrong password give the same result and
    take the same time."""
    u = by_username(username) if USERNAME.match(username.strip().lower()) else None
    if u is None:
        dummy_check(password)
        return None
    return u if matches(password, password_hash(u.id)) else None


# ------------------------------------------------------------------ sessions


# The kinds of session (sessions.kind, login_log.kind) and the word the 用户 page shows each by: a browser, a client
# (a DCC plugin or `lab2shot login`), the command line on this machine (machine_token)
SESSION_KINDS = {"web": "浏览器", "client": "插件", "machine": "本机令牌"}


@dataclass(frozen=True)
class Session:
    token: str  # sha256 of the token
    user: User
    kind: str  # a key of SESSION_KINDS
    expires: float
    admin_until: float  # rights beyond the editor last until then (roles.py); for a browser, ADMIN_S from password entry

    @property
    def capabilities(self) -> frozenset[str]:
        """The capabilities beyond the editor it holds now (lab2shot/roles.py): those of the account's role, while its
        rights last (a browser: ADMIN_S from password entry; the machine token: always; a client token: never)."""
        return self.user.capabilities if self.admin_until > now() else frozenset()

    def can(self, capability: str) -> bool:
        return capability in self.capabilities

    @property
    def lapsed(self) -> bool:
        """The role has rights, but they have expired on this session; entering the password again restores them."""
        return bool(self.user.capabilities) and self.kind == "web" and self.admin_until <= now()


def start(user: User, kind: str, ip: str = "", agent: str = "", replaces: str = "", device_id: str = "",
          hostname: str = "", app: str = "") -> tuple[str, Session]:
    """Create a session for a user who has just entered the password (a browser cookie or a client token).
    `replaces` is the token the browser presented (this request's own cookie); it is ended regardless of its owner and
    never promoted, so a token planted in a browser before login never becomes its session. Every other live session
    of the same account and the same kind is also ended: at most one browser and, separately, one client per account
    may be online (a browser and a DCC plugin may be online together; a second browser or plugin displaces the first
    of its kind). Ended sessions are kept for a while as a displacement notice (kicked_info) for their holders. The
    machine token is not affected. Every attempt (here, a success) is written to login_log. Returns (token, session)."""
    token = secrets.token_urlsafe(32)
    t = now()
    life = TOKEN_S if kind == "client" else SESSION_S
    admin_until = t + ADMIN_S if user.capabilities and kind == "web" else 0.0
    device = device_label(kind, agent, app)
    ended: list[dict] = []
    with db().write() as c:
        if replaces:
            c.execute("DELETE FROM sessions WHERE token = ?", (sha(replaces),))
        if kind in ("web", "client"):
            rows = c.execute("SELECT token, kind, device FROM sessions WHERE user_id = ? AND kind = ?",
                             (user.id, kind)).fetchall()
            if rows:
                info = json_text({"at": t, "ip": ip, "device": device, "kind": kind})
                for r in rows:
                    if len(ended) < ENDED_LOGGED:
                        ended.append({"kind": r["kind"], "device": r["device"]})
                    c.execute("UPDATE sessions SET expires = ?, replaced_at = ?, replaced_by = ? WHERE token = ?",
                              (t, t, info, r["token"]))
        c.execute("INSERT INTO sessions (token, user_id, kind, created, expires, admin_until, seen, ip, agent, "
                  "device_id, device, hostname) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (sha(token), user.id, kind, t, t + life, admin_until, t, ip[:100], agent[:300], device_id[:100],
                   device, hostname[:100]))
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", (t, user.id))
        log_login(user.username, user.id, True, "", kind, ip, agent, device, device_id, hostname, ended)
    return token, Session(sha(token), user, kind, t + life, admin_until)


def kicked_info(token: str) -> dict | None:
    """Why `token` no longer works when another login ended it (start()): {"at", "ip", "device", "kind"} of that
    login. None when it was never a session or was not ended by a login (guessed, expired, logged out, account
    deleted)."""
    if not token or len(token) > 100:
        return None
    r = db().row("SELECT replaced_by FROM sessions WHERE token = ? AND replaced_at IS NOT NULL", (sha(token),))
    return json_of(r["replaced_by"]) if r else None


def once_issued(token: str) -> bool:
    """Whether this server issued `token` and still has a record of it: a live session, one ended by another login,
    or one that lapsed (kept for REPLACED_KEPT_S). A page left open keeps sending such a token (a displaced tab, a
    laptop reopened the next morning); that indicates a legitimate user whose login ended rather than a probe. A token
    this server never issued is treated as a probe (server/access.py)."""
    if not token or len(token) > 100:
        return False
    return db().row("SELECT 1 AS known FROM sessions WHERE token = ?", (sha(token),)) is not None


def session(token: str) -> Session | None:
    """The live session a token opens: the session is not expired and its account is enabled, not expired and not
    deleted (checked on every call, so disabling or expiry ends it immediately). A machine-token session (kind
    "machine": the command line on this machine, created by machine_token) expires MACHINE_S after creation and is
    never renewed here."""
    if not token or len(token) > 100:
        return None
    key = sha(token)
    r = db().row("SELECT s.kind, s.expires AS s_expires, s.admin_until, s.seen, s.device AS s_device, "
                 "s.hostname AS s_hostname, u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?", (key,))
    t = now()
    if r is None or r["s_expires"] <= t:
        return None
    user = _user(r)
    if user.usable_now():
        return None
    expires = r["s_expires"]
    if r["kind"] != "machine":  # 在线: this request, in memory only (the machine token is nobody's presence)
        _ACTIVE[key] = _Active(t, user.id, r["kind"], r["s_device"] or "", r["s_hostname"] or "")
    if r["kind"] != "machine" and t - r["seen"] > SEEN_EVERY_S:  # in use: renew from now
        # start() writes only web / client / machine (machine is excluded above). The renewal length must depend on
        # the kind; otherwise a client token would shrink from TOKEN_S to SESSION_S on its first renewal.
        expires = t + (TOKEN_S if r["kind"] == "client" else SESSION_S)
        with db().write() as c:
            c.execute("UPDATE sessions SET seen = ?, expires = ? WHERE token = ?", (t, expires, key))
    return Session(key, user, r["kind"], expires, r["admin_until"])


def end(token: str) -> None:
    with db().write() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (sha(token),))


def end_all() -> int:
    """Every browser and client is out (`lab2shot admin logout`); returns how many sessions there were."""
    with db().write() as c:
        return c.execute("DELETE FROM sessions").rowcount


@dataclass(frozen=True)
class _Active:
    """A web/client session's latest request (session()): when, whose, and where from (its device and computer)."""
    at: float
    user_id: int
    kind: str
    device: str
    hostname: str


# session key (sha256 of the token) -> its latest request; memory only, empty after a restart (presence() then falls
# back to sessions.seen). Written by request threads one key at a time; readers take a copy.
_ACTIVE: dict[str, _Active] = {}


def _where(kind: str, device: str, hostname: str) -> dict:
    return {"kind": kind, "device": device, "hostname": hostname}


def presence() -> dict[int, dict]:
    """Per account (only those with any web/client activity on record): {"online": [where...], "active": time of the
    last activity or None, "where": where that was}. `where` is {kind, device, hostname}: a browser as
    "Chrome · Windows", a DCC plugin or the command line by its app and computer. Online: a request within ONLINE_S on
    a session still live (not ended, expired or replaced); the machine token never counts. The last activity is the
    newest of the requests remembered since this server started, sessions.seen and the last login."""
    t = now()
    seen = dict(_ACTIVE)  # a copy: request threads keep writing
    out: dict[int, dict] = {}

    def note(user_id: int, at: float | None, where: dict | None) -> dict:
        p = out.setdefault(user_id, {"online": [], "active": None, "where": None})
        if at and (p["active"] is None or at > p["active"]):
            p["active"], p["where"] = at, where
        return p

    for r in db().rows("SELECT token, user_id, kind, device, hostname, seen, expires, replaced_at FROM sessions "
                       "WHERE kind IN ('web', 'client')"):
        a = seen.pop(r["token"], None)
        where = _where(r["kind"], a.device if a else r["device"] or "", a.hostname if a else r["hostname"] or "")
        p = note(r["user_id"], max(r["seen"] or 0, a.at if a else 0), where)
        if a and t - a.at <= ONLINE_S and r["expires"] > t and r["replaced_at"] is None:
            p["online"].append(where)
    for a in seen.values():  # a session since ended (logged out, displaced, pruned): still its account's activity
        note(a.user_id, a.at, _where(a.kind, a.device, a.hostname))
    for r in db().rows("SELECT id, last_login FROM users WHERE last_login IS NOT NULL"):
        note(r["id"], r["last_login"], None)  # nothing else on record (a restart, the session long gone): the login
    for p in out.values():
        p["online"].sort(key=lambda w: w["kind"] != "web")  # the browser first
    return out


def online(roles_shown: set[str] | None = None) -> dict:
    """Accounts online now (presence(): a request within ONLINE_S; the machine token never counts), with browsers
    and clients counted separately (start()): a count of each and who/where, for the admin overview's 在线 tile and
    tooltip. `roles_shown`: only accounts of these roles (those the viewer manages)."""
    here = {i: p["online"] for i, p in presence().items() if p["online"]}
    if not here:
        return {"count": 0, "browser": 0, "client": 0, "who": {"browser": [], "client": []}, "window_s": ONLINE_S}
    marks = ",".join("?" * len(here))  # placeholders only: the ids are the server's own integers
    users = [r for r in db().rows(f"SELECT id, name, username, department, role FROM users WHERE id IN ({marks})",
                                  tuple(here)) if roles_shown is None or r["role"] in roles_shown]

    def line(r, w: dict) -> str:
        at = f" · {w['hostname']}" if w["kind"] == "client" and w["hostname"] else ""
        return f"{r['name']}（{r['username']}）{' · ' + r['department'] if r['department'] else ''} · {w['device']}{at}"

    browser = [line(r, w) for r in users for w in here[r["id"]] if w["kind"] == "web"]
    client = [line(r, w) for r in users for w in here[r["id"]] if w["kind"] == "client"]
    return {"count": len(users), "browser": len(browser), "client": len(client),
            "who": {"browser": browser, "client": client}, "window_s": ONLINE_S}


def logins_in(p: Periods) -> dict:
    """The admin overview's 访问: accounts online now (online()), and in 今日, 近 7 天 and 本月 the accounts that logged
    in, their successful logins and the failed attempts (login_log: browsers and clients alike, the administrator's own
    too; a failure counts whether or not the username was an account, each keeping only its newest FAILED_KEPT)."""
    day = local_day("at")
    ok = db().rows(f"SELECT {day} AS day, user_id, COUNT(*) AS n FROM login_log WHERE at >= ? AND ok = 1 GROUP BY 1, 2",
                   (p.since,))
    failed = db().rows(f"SELECT {day} AS day, COUNT(*) AS n FROM login_log WHERE at >= ? AND ok = 0 GROUP BY 1", (p.since,))
    shown = ("today", "days7", "month")
    people = p.distinct(((r["day"], r["user_id"]) for r in ok), shown)
    logins = p.sums(((r["day"], r["n"]) for r in ok), shown)
    fails = p.sums(((r["day"], r["n"]) for r in failed), shown)
    return {"online": online(),
            **{k: {"people": people[k], "logins": logins[k], "failed": fails[k]} for k in shown}}


def prune_logins() -> None:
    """Housekeeping (the queue's tidy): remove login_log entries older than LOGIN_LOG_KEPT_S, and session rows kept
    only for a displacement notice (replaced_at) or that lapsed long ago without renewal; and forget the remembered
    requests of sessions gone from the table, except each account's newest (its last activity)."""
    t = now()
    with db().write() as c:
        c.execute("DELETE FROM login_log WHERE at < ?", (t - LOGIN_LOG_KEPT_S,))
        c.execute("DELETE FROM sessions WHERE replaced_at IS NOT NULL AND replaced_at < ?", (t - REPLACED_KEPT_S,))
        c.execute("DELETE FROM sessions WHERE kind != 'machine' AND replaced_at IS NULL AND expires < ?", (t - REPLACED_KEPT_S,))
        live = {r["token"] for r in c.execute("SELECT token FROM sessions").fetchall()}
    # the in-memory latest requests of sessions no longer on record: only each account's newest is kept (its last
    # activity, presence())
    newest: dict[int, str] = {}
    for key, a in sorted(dict(_ACTIVE).items(), key=lambda kv: kv[1].at):
        if key not in live:
            if (old := newest.get(a.user_id)) is not None:
                _ACTIVE.pop(old, None)
            newest[a.user_id] = key


# ------------------------------------------------------------------ login_log: every attempt, for 最近登录


DEVICE_ID_MIN = 16  # a shorter device id is never taken as known (an empty one least of all)


def known_device(user_id: int, device_id: str) -> bool:
    """Has this account logged in from this device (the random id its browser or client keeps and sends at every
    login) within LOGIN_LOG_KEPT_S? server/auth.py counts such a try apart from strangers' wrong passwords, so none of
    those ever slows the account's owner on their own device."""
    return len(device_id) >= DEVICE_ID_MIN and db().row(
        "SELECT 1 FROM login_log WHERE user_id = ? AND ok = 1 AND device_id = ? LIMIT 1", (user_id, device_id[:100])) is not None


def log_login(username: str, user_id: int | None, ok: bool, reason: str, kind: str, ip: str, agent: str, device: str,
              device_id: str, hostname: str, ended: list[dict]) -> None:
    """Record one login attempt (server/auth.py's login/token routes call this on failure, start() on success): the
    username entered and the account it resolved to, if any (shown only to the administrator; the login response is
    "用户名或密码不对" in either case), whether it succeeded and why not, and which other sessions it ended (kind and
    device of each, from start())."""
    with db().write() as c:
        c.execute("INSERT INTO login_log (at, username, user_id, ok, reason, kind, ip, agent, device, device_id, "
                  "hostname, ended) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (now(), username[:64], user_id, int(ok), reason[:200], kind, ip[:100], agent[:300], device[:60],
                   device_id[:100], hostname[:100], json_text(ended)))
        if not ok:  # strangers' failures, however many, keep only the newest FAILED_KEPT per account (and of unknown names)
            c.execute("DELETE FROM login_log WHERE ok = 0 AND user_id IS ? AND id <= (SELECT id FROM login_log "
                      "WHERE ok = 0 AND user_id IS ? ORDER BY id DESC LIMIT 1 OFFSET ?)", (user_id, user_id, FAILED_KEPT))


def login_recent(user_id: int, limit: int = 50) -> list[dict]:
    """The account's most recent login attempts, newest first (用户 page's 最近登录 table)."""
    rows = db().rows("SELECT * FROM login_log WHERE user_id = ? ORDER BY at DESC LIMIT ?", (user_id, limit))
    return [{"at": r["at"], "ok": bool(r["ok"]), "reason": r["reason"], "kind": r["kind"], "ip": r["ip"],
             "device": r["device"], "device_id": r["device_id"], "hostname": r["hostname"],
             "ended": json_of(r["ended"])} for r in rows]


def online_now(user_id: int) -> list[dict]:
    """Where this account is online now (a request within ONLINE_S, presence()): at most one row per kind (a browser
    and a client may be live together, start()), each {kind, ip, device, hostname, seen}, `seen` its latest request.
    An empty list means 不在线."""
    t, seen = now(), dict(_ACTIVE)
    rows = db().rows("SELECT token, kind, ip, device, hostname FROM sessions WHERE user_id = ? AND kind IN ('web', 'client') "
                     "AND expires > ? AND replaced_at IS NULL ORDER BY kind", (user_id, t))
    return [{"kind": r["kind"], "ip": r["ip"], "device": r["device"], "hostname": r["hostname"], "seen": a.at}
            for r in rows if (a := seen.get(r["token"])) is not None and t - a.at <= ONLINE_S]


def login_summary(user_id: int) -> dict:
    """7- and 30-day counts for the 用户 page's 最近登录 panel (successful logins, distinct IPs, distinct devices,
    failures), and whether the 7-day window looks like 异地频繁 (SUSPICIOUS_*, stated in its tooltip)."""
    t = now()

    def window(days: int) -> dict:
        rows = db().rows("SELECT ip, device, ok FROM login_log WHERE user_id = ? AND at >= ?", (user_id, t - days * 86400))
        return {"logins": sum(1 for r in rows if r["ok"]), "ips": len({r["ip"] for r in rows if r["ok"]}),
                "devices": len({r["device"] for r in rows if r["ok"]}), "failures": sum(1 for r in rows if not r["ok"])}

    d7, d30 = window(7), window(30)
    by_day: dict[str, int] = {}
    for r in db().rows("SELECT at FROM login_log WHERE user_id = ? AND ok = 0 AND at >= ?", (user_id, t - 7 * 86400)):
        key = time.strftime("%Y-%m-%d", time.localtime(r["at"]))
        by_day[key] = by_day.get(key, 0) + 1
    worst_day = max(by_day.values(), default=0)
    suspicious = d7["ips"] > SUSPICIOUS_IPS_7D or d7["devices"] > SUSPICIOUS_DEVICES_7D or worst_day >= SUSPICIOUS_FAILURES_DAY
    return {"7d": d7, "30d": d30, "suspicious": suspicious, "worst_day_failures": worst_day}


# ------------------------------------------------------------------ the 口令, the machine token


def passphrase() -> dict | None:
    """The stored 口令 ({"hash", "set"}), or None until one is set on this machine."""
    return db().meta(PASSPHRASE)


def set_passphrase(new: str) -> None:
    """Called only by `lab2shot admin passphrase`; no server route calls it."""
    if problem := rule_problem(new, "口令"):
        raise Invalid(problem)
    db().set_meta(PASSPHRASE, {"hash": hash_secret(new), "set": now()})


def machine_token() -> str:
    """The token the command line on this machine sends (server/auth.py MACHINE_HEADER); it acts as the
    administrator. Its plaintext exists only in the work folder's MACHINE_FILE (0600); the database keeps its sha256 as
    a `sessions` row of kind "machine", read through session(). A token older than MACHINE_ROTATE_S, or whose session
    is missing or expired, is replaced and every other machine session ends. It is never ended by logins (start()) and
    never renewed on use."""
    path = settings().work_dir / MACHINE_FILE
    t = now()
    plain = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    if plain:
        row = db().row("SELECT created, expires FROM sessions WHERE token = ? AND kind = 'machine'", (sha(plain),))
        if row is not None and row["expires"] > t and t - row["created"] < MACHINE_ROTATE_S:
            return plain
    plain = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.with_name(path.name + ".new")
    fd = os.open(fresh, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(plain)
    os.chmod(fresh, 0o600)
    with db().write() as c:
        c.execute("DELETE FROM sessions WHERE kind = 'machine'")
        c.execute("INSERT INTO sessions (token, user_id, kind, created, expires, admin_until, seen, ip, agent) "
                  "VALUES (?, ?, 'machine', ?, ?, ?, ?, '', '本机命令行')",
                  (sha(plain), ADMIN_ID, t, t + MACHINE_S, t + MACHINE_S, t))
    os.replace(fresh, path)
    return plain
