"""User accounts: who may use this server, and what is theirs.

Every user logs in with an account. Each account has a role (lab2shot/roles.py: 管理员, 二级管理员, 普通用户), which
determines what it may do beyond the editor. The built-in administrator account (admin, ADMIN_ID; it has no password
until one is set with `./setup.sh`) is always 管理员. The 管理员 and 二级管理员 create other accounts on the admin page
(用户); while 开放注册 is on, people also make their own on the login page (lab2shot/site/registration.py), through the
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

A DCC plugin's embedded web window (kind "embedded", start_embedded) logs in through its plugin's login (kind "client")
with a one-time ticket (server/auth.py Tickets): it hangs under that login (sessions.parent) and lives only while it
does — the plugin logging out, being displaced by another plugin login, its token lapsing or the account being disabled
ends it at once (session() checks its parent on every request). It displaces nothing and nothing displaces it but its
parent; it has no rights beyond the editor (like its parent), and lasts EMBED_S from its last use, never past its parent.

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
from collections.abc import Callable
from dataclasses import dataclass

from . import i18n, roles
from .config import settings
from .database import db, json_of, json_text
from .errors import Invalid, NotFound
from .messages import Msg
from .nodes import tags as node_tags
from .periods import Periods, local_day

ADMIN_ID = 1  # the baseline migration creates the administrator's account first (database/schema.py)
ADMIN_NAME = "admin"  # the built-in administrator's username on a new installation
# names no account may take: the template folders' owners besides the accounts' (site/library.py OWNER_ADMIN, OWNER_ADAPTER)
RESERVED_USERNAMES = frozenset({"admin", "adapter"})
def deleted_label() -> str:
    """The label under which a deleted account's jobs are counted, in the language now (account.deleted)."""
    return i18n.t("account.deleted")


def kept_deleted() -> str:
    """What is written into the records in place of a purged account's name (the server's records are English)."""
    return i18n.t("account.deleted", in_lang=i18n.LOG_LANG)


def label_of(name: str, username: str, lang: str | None = None) -> str:
    """An account as people read it: 张三（zhangsan） (account.label: the brackets are the language's)."""
    return i18n.t("account.label", in_lang=lang, name=name, username=username)


def labels_of(name: str, username: str) -> set[str]:
    """Every way a record may have written the account (its label in each language)."""
    return {label_of(name, username, lang) for lang in i18n.LANGS}

MIN_CHARS, MAX_CHARS = 8, 128  # length limits for a password and the 口令
SESSION_S = 30 * 86400  # a browser stays logged in this long after its last use
TOKEN_S = 180 * 86400  # the same for a DCC plugin or the command line
ADMIN_S = 3 * 86400  # duration of administrator rights on a browser, from password entry
EMBED_S = 86400  # a plugin's embedded window stays logged in this long after its last use (never past its parent's)
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
NAME_MIN, NAME_MAX = 2, 6  # a Chinese name: Han characters
LATIN_MIN, LATIN_MAX = 2, 20  # an English name: letters and digits, starting with a letter
DOTS = "·•・‧∙"  # middle-dot variants accepted in input; normalised to ·
# Han characters (CJK Unified Ideographs, Extension A, Compatibility, Extensions B and on), by code point
_HAN = "".join(f"{chr(a)}-{chr(b)}" for a, b in ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x3134F)))
_NAME = re.compile(rf"^[{_HAN}]+(·[{_HAN}]+)*$")
_LATIN = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")  # no spaces, no other characters


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
        return app or i18n.t("account.device.client", in_lang=i18n.LOG_LANG)  # kept in the records: the server's English
    browser = next((label for key, label in _BROWSERS if key in agent), None) or i18n.t("account.device.browser", in_lang=i18n.LOG_LANG)
    system = next((label for key, label in _SYSTEMS if key in agent), None) or i18n.t("account.device.unknown_system", in_lang=i18n.LOG_LANG)
    return f"{browser} · {system}"


def rule_problem(text: str, what: object = None) -> Msg | None:
    """Why a new password or 口令 cannot be used (None if it can); the pages apply the same check before submitting.
    `what` is the term used in the message (密码, 新密码, 口令; None: 密码, account.secret.password)."""
    what = i18n.t("account.secret.password") if what is None else what
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
    """The name in stored form (trimmed, middle dots normalised): a Chinese name (NAME_MIN–NAME_MAX Han characters,
    parts joined by ·) or an English one (LATIN_MIN–LATIN_MAX letters and digits, starting with a letter); raises
    Invalid stating what is wrong."""
    text = str(name or "").strip()
    for dot in DOTS:
        text = text.replace(dot, "·")
    if not text:
        raise Invalid(Msg("E-ACCOUNT-NONAME", min=NAME_MIN, max=NAME_MAX, latin_min=LATIN_MIN, latin_max=LATIN_MAX))
    if re.search(r"\s", text):
        raise Invalid(Msg("E-ACCOUNT-NAMESPACE", name=text))
    if _LATIN.match(text):
        if not LATIN_MIN <= len(text) <= LATIN_MAX:
            raise Invalid(Msg("E-ACCOUNT-NAMELATINLENGTH", name=text, count=len(text), min=LATIN_MIN, max=LATIN_MAX))
        return text
    if not _NAME.match(text):
        raise Invalid(Msg("E-ACCOUNT-NAMECHARS", name=text))
    chars = len(text.replace("·", ""))
    if not NAME_MIN <= chars <= NAME_MAX:
        raise Invalid(Msg("E-ACCOUNT-NAMELENGTH", name=text, count=chars, min=NAME_MIN, max=NAME_MAX))
    return text


def departments() -> list[str]:
    """The departments an account may be given (the setting people.departments): ids of the factory list
    (lab2shot/departments.json), and whatever an administrator added, as written."""
    return list(settings()["people.departments"])


def department_label(value: str) -> str:
    """How a department shows, the one rule: a factory one by its words (department.<id>, in the language now), one
    an administrator added (user data) as written; "" none."""
    return (i18n.lookup(f"department.{value}") if value and value.isidentifier() and value.isascii() else None) or value


def departments_shown() -> list[dict]:
    """The departments with how each shows: what a page offers to pick ({value, label}; the value is stored)."""
    return [{"value": d, "label": department_label(d)} for d in departments()]


def check_department(dept: object) -> str:
    text = str(dept or "").strip()
    if text not in departments():
        raise Invalid(Msg("E-ACCOUNT-DEPARTMENT", departments=i18n.separator().join(department_label(d) for d in departments())))
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
class Actor:
    """Who did something, as a record keeps it (an invite code's maker, a version of the terms, a role's rights, the
    notice, a feedback's answer): the account, by which permanent deletion finds it (PURGE), and how it is named then
    (User.label, the one way a record names a person). `id` None: the program or the command line on this machine."""

    id: int | None
    label: str

    @classmethod
    def of(cls, user: "User") -> "Actor":
        return cls(user.id, user.label)


SYSTEM = Actor(None, "")  # the program's own doing (its own terms, a default)

# who set an account's password, as users.password_by keeps it: an id said in the reader's language (password_by_text),
# never words (they would stay in the language of whoever set it)
BY_SELF, BY_PASSPHRASE, BY_COMMAND_LINE = "self", "passphrase", "command_line"
BY_ROLE = "role:"  # + the role of the account that set it (roles.py): an administrator, a deputy


def by_role(role: str) -> str:
    """password_by for a password an account of `role` set for another."""
    return BY_ROLE + role


def password_by_text(by: str) -> str:
    """Who set a password (users.password_by) in the language now; a row from before these ids, as it was written."""
    if by == BY_SELF:
        return i18n.t("server.password_by.self")
    if by == BY_PASSPHRASE:
        return i18n.t("server.password_by.passphrase")
    if by == BY_COMMAND_LINE:
        return i18n.t("cli.admin.by_command_line")
    if by.startswith(BY_ROLE):
        return roles.label(by[len(BY_ROLE):])
    return by


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
    lang: str = ""  # the language it chose (i18n.LANGS); "": none yet, the browser's (server/lang.py)

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
        return deleted_label() if self.deleted else label_of(self.name, self.username)

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
                "department_label": department_label(self.department), "role": self.role, "role_label": roles.label(self.role), "tags": sorted(self.tags), "expires": self.expires,
                "lang": self.lang}

    def full(self) -> dict:
        """The fields the 用户 page shows: `state` (可以用, or the reason it is not usable; 未设密码 for the built-in
        administrator account without a password) and `state_tip`."""
        problem = self.usable_now()
        state = i18n.t("account.state.no_password") if self.no_password else problem.text if problem else i18n.t("account.state.usable")
        tip = i18n.t("account.state.no_password_tip") if self.no_password else state
        return {**self.public(), "owner": self.owner, "all_nodes": self.allowed is None, "enabled": self.enabled, "deleted": self.deleted, "created": self.created,
                "last_login": self.last_login, "password_set": self.password_set, "password_by": password_by_text(self.password_by),
                "no_password": self.no_password, "usable_now": problem is None, "state": state, "state_tip": tip}


def _user(r) -> User:
    return User(id=r["id"], username=r["username"], name=r["name"], department=r["department"], role=r["role"],
                tags=frozenset(json_of(r["tags"])), expires=r["expires"], enabled=bool(r["enabled"]),
                deleted=r["deleted"], created=r["created"], last_login=r["last_login"],
                password_set=r["password_set"], password_by=r["password_by"],
                no_password=r["id"] == ADMIN_ID and not r["hash"], lang=r["lang"] if "lang" in r.keys() else "")


def set_lang(user_id: int, lang: str) -> None:
    """The language the account chose (one of i18n.LANGS, or "" to follow the browser): what everything is said to
    it in, wherever it logs in (server/lang.py)."""
    from . import i18n

    chosen = i18n.normal(lang) or ""
    if lang and not chosen:
        raise Invalid(Msg("E-ACCOUNT-BADLANG", lang=str(lang)[:20], langs=list(i18n.LANGS)))
    with db().write() as c:
        c.execute("UPDATE users SET lang = ? WHERE id = ?", (chosen, user_id))


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
             # 队列优先 (queue_first): only on this listing, which only the 用户 page reads; never on User, so none of
             # an account's own answers (public, full) can carry it
             "queue_first": bool(r["queue_first"]),
             "presence": here.get(r["id"], nobody)} for r in rows]


def new_password(text: str, what: object = None) -> str:
    """A new password as it is kept (its scrypt hash), or Invalid when the rules refuse it (rule_problem). The one place
    a new password is hashed, before any write transaction: scrypt takes a tenth of a second, and every other write of
    the process would wait that long behind it."""
    if problem := rule_problem(text, what):
        raise Invalid(problem)
    return hash_secret(text)


def create(username: str, hashed: str, name: str, department: str, expires: float, allowed: object,
           role: str = roles.DEFAULT, by: str = "") -> User:
    """Create an account (server/users.py checks which roles the caller may create: lab2shot/roles.py). `hashed`: its
    first password, from new_password (the caller's write transaction, registration's, may hold this one). `by`: who
    set its first password (BY_SELF, by_role(the creator's role): password_by)."""
    username, role = check_username(username), roles.check(role)
    name, department, given = check_name(name), check_department(department), check_tags(allowed)
    if expires <= now():
        raise Invalid(Msg("E-ACCOUNT-EXPIRYPAST"))
    if by_username(username) or username in RESERVED_USERNAMES:  # admin / adapter are the other two template folders (site/library.py)
        raise Invalid(Msg("E-ACCOUNT-USERNAMETAKEN", username=username))
    # A deleted account keeps its username until it is purged: what is kept under the username (its templates,
    # work/users/<username>/, site/library.py) is still the deleted account's, and purging it removes that folder.
    if db().row("SELECT 1 FROM users WHERE username = ?", (username,)):
        raise Invalid(Msg("E-ACCOUNT-USERNAMEDELETED", username=username))
    t = now()
    with db().write() as c:
        uid = _new_id(c)
        c.execute("INSERT INTO users (id, username, hash, name, department, role, tags, expires, enabled, created, "
                  "password_set, password_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                  (uid, username, hashed, name, department, role, json_text(sorted(given)), expires, t, t, by))
    return get(uid)


REVISION = "users.revision"  # meta: counts every change of how an account is named or whether it is there (revision)


def _changed(c) -> None:
    """(Inside the write that changes an account's name, department, state, or removes it) one more revision."""
    c.execute("INSERT INTO meta (key, value) VALUES (?, '1') ON CONFLICT (key) DO UPDATE SET "
              "value = CAST(value AS INTEGER) + 1", (REVISION,))


def revision() -> str:
    """How many times accounts changed what an answer may say of them (a template's author: library.author_of): part
    of the key of an answer kept by its key (server/revisions.py), so a renamed or removed account is not named on."""
    r = db().row("SELECT value FROM meta WHERE key = ?", (REVISION,))
    return r["value"] if r else "0"


TOP_ID = "users.top_id"  # meta: the largest id an account permanently deleted had (purge)


def _new_id(c) -> int:
    """A new account's id: above every id any account has had, a permanently deleted one's too. SQLite alone would
    give the id of a purged newest account again, and whatever is still named by it (rows kept of it, counts in
    memory) would then be the new account's."""
    top = c.execute("SELECT MAX(id) FROM users").fetchone()[0] or 0
    gone = c.execute("SELECT value FROM meta WHERE key = ?", (TOP_ID,)).fetchone()
    return max(top, int(gone[0]) if gone else 0) + 1


def update(user_id: int, *, name: str | None = None, department: str | None = None, expires: float | None = None,
           enabled: bool | None = None, allowed: object = None, role: str | None = None,
           queue_first: bool | None = None) -> User:
    """Change an account (for the built-in administrator account, only its name, department and 队列优先). Disabling it or
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
    if queue_first is not None:
        changes["queue_first"] = int(bool(queue_first))
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
            _changed(c)
    return get(user_id)


@dataclass(frozen=True)
class Shift:
    """What moving an account's expiry did (shift_expiry): before and after (None: never expires), and whether it
    moved at all (`why` not: never — the account never expires; deleted — it is gone; none — no days)."""

    old: float | None
    new: float | None
    why: str = ""

    @property
    def applied(self) -> bool:
        return not self.why


def queue_first(user_id: int) -> bool:
    """队列优先 (Queue priority): the account's tasks, once submitted, wait ahead of every waiting task of an account
    without it (farm/queue.py in_order); nothing running is stopped. An administrator's setting, shown and changed only
    on the 用户 page (listing, update); False for no such account."""
    r = db().row("SELECT queue_first FROM users WHERE id = ?", (user_id,))
    return bool(r is not None and r["queue_first"])


def quota_gb(user_id: int) -> float | None:
    """The account's own disk quota as kept (users.quota_gb), None when it has none of its own (the setting's default
    applies) or there is no such account. The rule of quotas is lab2shot/server/quota.py's; the users table is here."""
    r = db().row("SELECT quota_gb FROM users WHERE id = ?", (user_id,))
    return None if r is None or r["quota_gb"] is None else float(r["quota_gb"])


def set_quota_gb(user_id: int, gb: float | None) -> None:
    """Give the account its own quota (None: back to the setting's default); joins the caller's transaction if any."""
    with db().write() as c:
        c.execute("UPDATE users SET quota_gb = ? WHERE id = ?", (None if gb is None else float(gb), user_id))


def shift_expiry(c, user_id: int | None, days: float) -> Shift:
    """(Inside the caller's write `c`) move an account's expiry by `days` (negative: back), from now when it has
    already passed and never to before now: the one place a reward of time is given or taken back (site/feedback.py rate).
    An account that never expires (the built-in administrator), a deleted or permanently deleted one, or no days: only
    said, nothing moves."""
    r = c.execute("SELECT expires, deleted FROM users WHERE id = ?", (user_id,)).fetchone() if user_id is not None else None
    if r is None or r[1] is not None:
        return Shift(None if r is None else r[0], None if r is None else r[0], "deleted")
    old = r[0]
    if old is None:
        return Shift(None, None, "never")
    if not days:
        return Shift(old, old, "none")
    t = now()
    new = max(old, t) + days * 86400 if days > 0 else max(t, old + days * 86400)
    if new == old:
        return Shift(old, old, "none")
    c.execute("UPDATE users SET expires = ? WHERE id = ?", (new, user_id))
    _changed(c)  # 已过期 ↔ 可以用 is how the account is said
    return Shift(old, new)


def set_password(user_id: int, new: str, by: str, keep: str | None = None) -> None:
    """Set a new password (by: who, an id of password_by: BY_SELF, BY_PASSPHRASE, BY_COMMAND_LINE, by_role). Every session of the account ends except `keep` (the token of
    the browser that made the change)."""
    hashed = new_password(new, i18n.t("account.secret.new_password"))
    with db().write() as c:
        c.execute("UPDATE users SET hash = ?, password_set = ?, password_by = ? WHERE id = ?",
                  (hashed, now(), by, user_id))
        c.execute("DELETE FROM sessions WHERE user_id = ? AND token != ?", (user_id, sha(keep) if keep else ""))


def password_hash(user_id: int) -> str:
    r = db().row("SELECT hash FROM users WHERE id = ?", (user_id,))
    return r["hash"] if r else ""


def switch_off_to_delete(user_id: int) -> User:
    """The first step of deleting an account (lab2shot/site/account_removal.py delete, which stops its jobs and removes its
    outputs next): never the built-in administrator account; it is switched off and its sessions end, so nothing new
    comes from it."""
    u = get(user_id)
    if u.owner:
        raise Invalid(Msg("E-ACCOUNT-ADMINDELETE", username=u.username))
    if u.deleted:
        raise NotFound(Msg("E-ACCOUNT-NOUSER"))
    with db().write() as c:
        c.execute("UPDATE users SET enabled = 0 WHERE id = ?", (user_id,))
        c.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    return u


def mark_deleted(user_id: int) -> None:
    """The last step of deleting an account: marked deleted (its job records and feedback stay)."""
    with db().write() as c:
        c.execute("UPDATE users SET deleted = ? WHERE id = ?", (now(), user_id))
        _changed(c)


# Every table naming an account, and what permanent deletion does with it (PURGE): one list, which `lab2shot check
# purge` holds the schema to (a table with user_id, and a column that says by whom something was done: `by`, `*_by`,
# with its account id beside it, must be on it), and purge works through. privacy.md promises that what is kept no
# longer leads to the person, so a kept row loses, besides user_id, every other thing in it that says who or where from
# (`forget`). What is found is found by the account's id, never by its words: two people may have one name, and a
# username may be a part of any other word, so a row is changed only when its id says it is this account's:
#   keeps  records of what was done: kept for the statistics and the administrator (LEFT JOIN users shows a missing
#          account as 「已删除的用户」, farm/usage.py _uses), without the account's name or where it worked from
#   counts counts of the whole site: the account's rows are added to account 0's (no account has id 0), so the site's
#          totals stay as they were
#   goes   ON DELETE CASCADE: removed with the account row, as the confirmation dialog says
#   names  no user_id: who did something is a name with its account id beside it (`named`: the name's column and the
#          id's, written together from an Actor): the name says DELETED where the id is this account's
# Node graphs saved on the server are files (work/users/<username>/templates/, lab2shot/site/library.py), deleted with the
# account's folder; its task folders, its cache and its uploads are deleted with it (farm/disk.py forget_account), so
# the job records kept afterwards no longer have a graph. A preset template saved since ids names who saved it by id
# (site/library.py author_id), shown as 「已删除的用户」 once the account is gone; a file saved before holds a name, and
# being the repository's templates/, it is left as it is. The account's id is never given again (_new_id).

ACTOR_KEYS = ("who", "role")  # an audit line's words for whoever did it (server/access.py actor), never for whom


def _forget_jobs(c, u: User) -> None:
    # the record's client is the account and what its requests showed (farm/clients.py full: address, computer, OS
    # user): only the application it came from stays
    c.execute("UPDATE jobs SET user_id = NULL, record = json_set(record, '$.client', "
              "json_object('app', json_extract(record, '$.client.app'))) WHERE user_id = ?", (u.id,))


def _forget_consents(c, u: User) -> None:
    # who accepted a licence (extensions/manual.py accept: farm/clients.py full): only the application stays
    c.execute("UPDATE consents SET user_id = NULL, record = json_set(record, '$.who', "
              "json_object('app', json_extract(record, '$.who.app'))) WHERE user_id = ?", (u.id,))


def _forget_feedback(c, u: User) -> None:
    """Its feedback is kept without saying whose (user_id, the sender the server noted in its bundle), and no feedback's
    bundle names it any more: a bundle holds what the server's log said when it was sent (site/feedback.py submit), whoever
    sent it, and a line of that log naming this account goes from all of them: its username, its label, or where it
    logged in from (an address, a computer's name: login_log, read here before _forget_logins forgets them; a request's
    line names only its address). The bundles are files of site/feedback.py, cleaned by it (feedback.forget_in_bundles, run
    by account_removal.purge before this with the pattern `marked` gives); here the rows."""
    c.execute("UPDATE feedback SET user_id = NULL WHERE user_id = ?", (u.id,))


def marked(u: User) -> re.Pattern:
    """What names this account in a text: its username, its labels, and where it logged in from (an address, a
    computer's name: login_log, read before purge forgets them; a request's line names only its address)."""
    places = {v for row in db().rows("SELECT ip, hostname FROM login_log WHERE user_id = ?", (u.id,))
              for v in (row["ip"], row["hostname"]) if v}
    return re.compile("|".join([rf"(?<![\w.-]){re.escape(u.username)}(?![\w-])", *(re.escape(x) for x in sorted(labels_of(u.name, u.username))),
                                *(rf"(?<![\w.:-]){re.escape(v)}(?![\w:-]|\.\w)" for v in sorted(places))]))


def _forget_rewards(c, u: User) -> None:
    c.execute("UPDATE feedback_rewards SET user_id = NULL WHERE user_id = ?", (u.id,))


def unmarked(value, marked: re.Pattern):
    """`value` (a JSON value) without what `marked` finds: a line of a list of them (a log's) gone, any other text with
    it said as DELETED."""
    if isinstance(value, list):
        return [unmarked(v, marked) for v in value if not (isinstance(v, str) and marked.search(v))]
    if isinstance(value, dict):
        return {k: unmarked(v, marked) for k, v in value.items()}
    return marked.sub(kept_deleted().replace("\\", "\\\\"), value) if isinstance(value, str) else value


def _forget_logins(c, u: User) -> None:
    # also the tries that typed its username before it resolved to the account (a username is one account's)
    c.execute("UPDATE login_log SET user_id = NULL, username = ?, ip = '', agent = '', device = '', device_id = '', "
              "hostname = '' WHERE user_id = ? OR username = ?", (kept_deleted(), u.id, u.username))


def _forget_actions(c, u: User) -> None:
    """Its own actions (user_id: their `who` is it), and every action about it (target_id, which audit writes from the
    account done to; a row from before that, its `username`): there, whatever names it but the words for who did it
    (ACTOR_KEYS: another administrator, who may share its name)."""
    labels = labels_of(u.name, u.username)
    for row_id, by, about, params in c.execute(
            "SELECT id, user_id, target_id, params FROM admin_actions WHERE user_id = ? OR target_id = ? OR "
            "(target_id IS NULL AND json_extract(params, '$.username') = ?)", (u.id, u.id, u.username)).fetchall():
        said = json_of(params, {})
        if not isinstance(said, dict):
            continue
        if by == u.id and said.get("who") in (*labels, u.name, u.username):
            said["who"] = kept_deleted()
        if about == u.id or said.get("username") == u.username:
            said = {k: v if k in ACTOR_KEYS else _unnamed(v, u) for k, v in said.items()}
        c.execute("UPDATE admin_actions SET user_id = ?, target_id = ?, params = ? WHERE id = ?",
                  (None if by == u.id else by, None if about == u.id else about, json_text(said), row_id))


def _unnamed(value, u: User):
    """`value` (a JSON value about this account) with what names it said as DELETED: its label, its username (whole
    texts: a username may be a part of other words) and its name, also inside a text (a change written as name=张三)."""
    if isinstance(value, str):
        if value == u.username or value in labels_of(u.name, u.username):
            return kept_deleted()
        return value.replace(u.name, kept_deleted()) if u.name else value
    if isinstance(value, list):
        return [_unnamed(v, u) for v in value]
    if isinstance(value, dict):
        return {k: _unnamed(v, u) for k, v in value.items()}
    return value


def _forget_notice(c, u: User) -> None:
    # the notice's `by` (meta server.notice: server/notice.py), with its account id beside it
    row = c.execute("SELECT value FROM meta WHERE key = 'server.notice'").fetchone()
    notice = json_of(row[0], {}) if row else {}
    if isinstance(notice, dict) and notice.get("by_id") == u.id:
        c.execute("UPDATE meta SET value = ? WHERE key = 'server.notice'",
                  (json_text({**notice, "by": kept_deleted(), "by_id": None}),))


def _merge_traffic(c, u: User) -> None:
    c.execute("INSERT INTO traffic (user_id, day, bytes) SELECT 0, day, bytes FROM traffic WHERE user_id = ? "
              "ON CONFLICT (user_id, day) DO UPDATE SET bytes = bytes + excluded.bytes", (u.id,))
    c.execute("DELETE FROM traffic WHERE user_id = ?", (u.id,))


@dataclass(frozen=True)
class Purged:
    how: str  # keeps, counts, goes, names (above)
    forget: Callable[..., None] | None = None  # (connection, account): what purge does with its rows
    # (name column, its account id column): who did something; the name says DELETED where the id is this account's
    named: tuple[tuple[str, str], ...] = ()


PURGE = {
    "jobs": Purged("keeps", _forget_jobs),
    "consents": Purged("keeps", _forget_consents),
    "feedback": Purged("keeps", _forget_feedback, named=(("updated_by", "updated_by_id"), ("replied_by", "replied_by_id"),
                                                         ("rated_by", "rated_by_id"))),
    # the reward ledger (site/feedback.py rate): kept for the record, without whose account it was
    "feedback_rewards": Purged("keeps", _forget_rewards, named=(("by", "by_id"),)),
    "login_log": Purged("keeps", _forget_logins),  # after feedback: _forget_feedback reads where it logged in from
    "admin_actions": Purged("keeps", _forget_actions),
    "traffic": Purged("counts", _merge_traffic),
    **{t: Purged("goes") for t in ("sessions", "tasks", "task_group_names", "registrations", "terms_agreed")},
    "invites": Purged("names", named=(("created_by", "created_by_id"),)),
    "terms_versions": Purged("names", named=(("by", "by_id"),)),
    "role_rights": Purged("names", named=(("updated_by", "updated_by_id"),)),
    "meta": Purged("names", _forget_notice),
}


def purgeable(user_id: int) -> User:
    """The account permanent deletion (lab2shot/site/account_removal.py purge) may go on with: one already deleted (deletion
    and permanent deletion are two separate steps, to prevent accidental loss), never the built-in administrator
    account (ADMIN_ID)."""
    u = get(user_id)
    if u.owner:
        raise Invalid(Msg("E-ACCOUNT-ADMINDELETE", username=u.username))
    if not u.deleted:
        raise Invalid(Msg("E-ACCOUNT-NOTDELETED", username=u.username))
    return u


def kept_counts(user_id: int) -> dict[str, int]:
    """The rows permanent deletion keeps of an account, per table (PURGE keeps)."""
    return {t: db().row(f"SELECT COUNT(*) AS n FROM {t} WHERE user_id = ?", (user_id,))["n"]
            for t, p in PURGE.items() if p.how == "keeps"}


def purge_rows(u: User) -> None:
    """The database's part of permanent deletion (account_removal.purge, after the account's files are gone): the
    account row is removed and the username may be reused (not before: create). Job records, feedback, login records
    and the admin action log are kept but no longer identify the account (PURGE keeps: nor where it worked from); the
    statistics show them as 「已删除的用户」, and what it was sent stays in the site's traffic totals (PURGE counts)."""
    from . import traffic

    traffic.flush()  # what it sent that is still counted in memory, into the rows moved below
    with db().write() as c:
        for table, p in PURGE.items():
            if p.forget is not None:
                p.forget(c, u)
            for column, by_id in p.named:
                c.execute(f"UPDATE {table} SET {column} = ?, {by_id} = NULL WHERE {by_id} = ?", (kept_deleted(), u.id))
        c.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = "
                  "MAX(CAST(value AS INTEGER), CAST(excluded.value AS INTEGER))", (TOP_ID, str(u.id)))
        c.execute("DELETE FROM users WHERE id = ?", (u.id,))
        _changed(c)


def login(username: str, password: str) -> User | None:
    """The account these credentials open, or None. An unknown username and a wrong password give the same result and
    take the same time."""
    u = by_username(username) if USERNAME.match(username.strip().lower()) else None
    if u is None:
        dummy_check(password)
        return None
    return u if matches(password, password_hash(u.id)) else None


# ------------------------------------------------------------------ sessions


# The kinds of session (sessions.kind, login_log.kind): a browser, a client (a DCC plugin or `lab2shot login`), the
# command line on this machine (machine_token), a plugin's embedded window; the word the 用户 page shows each by is
# account.session.<kind> (session_kind)
SESSION_KINDS = ("web", "client", "machine", "embedded")


def session_kind(kind: str, lang: str | None = None) -> str:
    return i18n.t(f"account.session.{kind}", in_lang=lang) if kind in SESSION_KINDS else kind


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
                    _end_children(c, r["token"], t)
        c.execute("INSERT INTO sessions (token, user_id, kind, created, expires, admin_until, seen, ip, agent, "
                  "device_id, device, hostname) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                  (sha(token), user.id, kind, t, t + life, admin_until, t, ip[:100], agent[:300], device_id[:100],
                   device, hostname[:100]))
        c.execute("UPDATE users SET last_login = ? WHERE id = ?", (t, user.id))
        log_login(user.username, user.id, True, "", kind, ip, agent, device, device_id, hostname, ended)
    return token, Session(sha(token), user, kind, t + life, admin_until)


def _end_children(c, parent: str, t: float) -> None:
    """(In a write) the embedded windows under the plugin login `parent` (its token column) end with it, kept a while
    like a displaced session so the window can say why (kicked_info)."""
    c.execute("UPDATE sessions SET expires = ?, replaced_at = ?, replaced_by = ? WHERE parent = ? AND kind = 'embedded' "
              "AND replaced_at IS NULL", (t, t, json_text({"at": t, "ip": "", "device": "", "kind": "embedded"}), parent))


def kicked_info(token: str) -> dict | None:
    """Why `token` no longer works when another login ended it (start()): {"at", "ip", "device", "kind"} of that
    login. None when it was never a session or was not ended by a login (guessed, expired, logged out, account
    deleted). An embedded window's login whose plugin login is gone (logged out, displaced, lapsed) says so with
    kind "embedded": the window then asks to be opened again from the DCC, never for a password (a login typed there
    would be a browser's, and would end the account's own browser login)."""
    if not token or len(token) > 100:
        return None
    r = db().row("SELECT s.kind, s.replaced_by, s.replaced_at, s.created, p.kind AS p_kind, p.expires AS p_expires, "
                 "p.replaced_at AS p_replaced FROM sessions s LEFT JOIN sessions p ON p.token = s.parent WHERE s.token = ?",
                 (sha(token),))
    if r is None:
        return None
    if r["kind"] == "embedded":
        if r["replaced_at"] is not None or r["p_kind"] != "client" or r["p_replaced"] is not None or r["p_expires"] <= now():
            return {"at": r["replaced_at"] or r["created"], "ip": "", "device": "", "kind": "embedded"}
        return None
    return json_of(r["replaced_by"]) if r["replaced_at"] is not None else None


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
    return session_by_key(sha(token))


def session_by_key(key: str) -> Session | None:
    """session() by the stored key of a token (its sha256): for a ticket that remembers whose login asked for it
    (server/auth.py Tickets keeps no plaintext token)."""
    r = db().row("SELECT s.kind, s.expires AS s_expires, s.admin_until, s.seen, s.device AS s_device, "
                 "s.hostname AS s_hostname, p.kind AS p_kind, p.expires AS p_expires, p.replaced_at AS p_replaced, "
                 "p.user_id AS p_user, u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                 "LEFT JOIN sessions p ON p.token = s.parent AND s.parent != '' WHERE s.token = ?", (key,))
    t = now()
    if r is None or r["s_expires"] <= t:
        return None
    embedded = r["kind"] == "embedded"
    # an embedded window's login lives only while the plugin login it hangs under does (start_embedded)
    if embedded and (r["p_kind"] != "client" or r["p_replaced"] is not None or r["p_expires"] <= t or r["p_user"] != r["id"]):
        return None
    user = _user(r)
    if user.usable_now():
        return None
    expires = min(r["s_expires"], r["p_expires"]) if embedded else r["s_expires"]
    if r["kind"] != "machine":  # 在线: this request, in memory only (the machine token is nobody's presence)
        _ACTIVE[key] = _Active(t, user.id, r["kind"], r["s_device"] or "", r["s_hostname"] or "")
    if r["kind"] != "machine" and t - r["seen"] > SEEN_EVERY_S:  # in use: renew from now
        # start() writes only web / client / machine (machine is excluded above). The renewal length must depend on
        # the kind; otherwise a client token would shrink from TOKEN_S to SESSION_S on its first renewal. An embedded
        # window's never outlasts its parent.
        expires = min(t + EMBED_S, r["p_expires"]) if embedded else t + (TOKEN_S if r["kind"] == "client" else SESSION_S)
        with db().write() as c:
            c.execute("UPDATE sessions SET seen = ?, expires = ? WHERE token = ?", (t, expires, key))
    # an embedded window holds no rights beyond the editor, whatever the account's role (its parent, a client, never does)
    return Session(key, user, r["kind"], expires, 0.0 if embedded else r["admin_until"])


def start_embedded(parent: Session, ip: str = "", agent: str = "") -> tuple[str, Session]:
    """A login for a DCC plugin's embedded web window, under the plugin's own login `parent` (kind "client" only:
    server/auth.py redeems a ticket only a client token could ask for). It ends no other session and no web or client
    login ends it; it ends with its parent (session(), _end_children). No rights beyond the editor; it lasts EMBED_S
    from its last use and never past its parent. Not in login_log: no password was typed (the ticket's redemption is
    the server log's, server/auth.py). Returns (token, session)."""
    if parent.kind != "client":
        raise ValueError("an embedded window's login hangs under a client's")
    token = secrets.token_urlsafe(32)
    t = now()
    expires = min(t + EMBED_S, parent.expires)
    device = session_kind("embedded", i18n.LOG_LANG)  # kept in the records: the server's English
    with db().write() as c:
        c.execute("INSERT INTO sessions (token, user_id, kind, created, expires, admin_until, seen, ip, agent, device, parent) "
                  "VALUES (?, ?, 'embedded', ?, ?, 0, ?, ?, ?, ?, ?)",
                  (sha(token), parent.user.id, t, expires, t, ip[:100], agent[:300], device, parent.token))
    return token, Session(sha(token), parent.user, "embedded", expires, 0.0)


def end(token: str) -> None:
    """Log out: the session is gone, and the embedded windows under it (a plugin's) end with it."""
    with db().write() as c:
        c.execute("DELETE FROM sessions WHERE token = ?", (sha(token),))
        _end_children(c, sha(token), now())


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


def online(manages: Callable[[int, str], bool]) -> dict:
    """Accounts online now (presence(): a request within ONLINE_S; the machine token never counts), with browsers
    and clients counted separately (start()): a count of each and who/where, for the admin overview's 在线 tile and
    tooltip. `manages(account id, role)`: whether whoever asks manages that account (server/access.py manages): where
    another is online (its device, its computer) is said only of those it does."""
    here = {i: p["online"] for i, p in presence().items() if p["online"]}
    if not here:
        return {"count": 0, "browser": 0, "client": 0, "who": {"browser": [], "client": []}, "window_s": ONLINE_S}
    marks = ",".join("?" * len(here))  # placeholders only: the ids are the server's own integers
    users = db().rows(f"SELECT id, name, username, department, role FROM users WHERE id IN ({marks})", tuple(here))

    def line(r, w: dict) -> str:
        who = label_of(r["name"], r["username"]) + (f" · {r['department']}" if r["department"] else "")
        if not manages(r["id"], r["role"]):
            return who
        at = f" · {w['hostname']}" if w["kind"] == "client" and w["hostname"] else ""
        return f"{who} · {w['device']}{at}"

    browser = [line(r, w) for r in users for w in here[r["id"]] if w["kind"] == "web"]
    client = [line(r, w) for r in users for w in here[r["id"]] if w["kind"] == "client"]
    return {"count": len(users), "browser": len(browser), "client": len(client),
            "who": {"browser": browser, "client": client}, "window_s": ONLINE_S}


def logins_in(p: Periods, manages: Callable[[int, str], bool]) -> dict:
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
    return {"online": online(manages),
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
        # an embedded window's login whose plugin login is long gone (it ended it: _end_children; or it was removed)
        c.execute("DELETE FROM sessions WHERE kind = 'embedded' AND replaced_at IS NULL AND parent NOT IN (SELECT token FROM sessions)")
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


def reason_text(reason: str) -> str:
    """A login_log reason as people read it: a message code (N-LOGIN-REASON…, an account problem's own) said in the
    language now; anything else (a row from before codes were kept) as written."""
    if reason and i18n.is_message_key(reason):
        try:
            return Msg(reason).text
        except Exception:  # a code no catalogue has any more, or one wanting parameters: as written
            return reason
    return reason


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
    return [{"at": r["at"], "ok": bool(r["ok"]), "reason": reason_text(r["reason"]), "kind": r["kind"], "ip": r["ip"],
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
    db().set_meta(PASSPHRASE, {"hash": new_password(new, i18n.t("account.secret.passphrase")), "set": now()})


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
                  "VALUES (?, ?, 'machine', ?, ?, ?, ?, '', ?)",
                  (sha(plain), ADMIN_ID, t, t + MACHINE_S, t + MACHINE_S, t, i18n.t("account.device.machine", in_lang=i18n.LOG_LANG)))
    os.replace(fresh, path)
    return plain
