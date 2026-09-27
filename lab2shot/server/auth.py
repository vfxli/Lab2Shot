"""Logging in over HTTP: who a request is (its account, lab2shot/accounts.py), and the guards on guessing passwords.

The server may be reached from the internet (a port-forwarding tunnel), where every request can arrive from this
machine's own address: an address never proves who asks. Only a password does.

A request is its account's when it carries one of:
  - the browser's cookie (COOKIE), given at login: HttpOnly, SameSite=Strict, Secure over HTTPS;
  - a client's token (Authorization: Bearer ...), given to a DCC plugin or the command line at /api/auth/token and
    kept in ~/.lab2shot/;
  - this machine's token (MACHINE_HEADER): the command line on the server machine, acting as the administrator;
    taken only from a loopback connection that no proxy forwarded (local_request), never as a cookie or bearer.
Its account must be usable now (enabled, not expired, not deleted): checked on every request, so disabling a user or
their expiry ends their sessions at once. The administrator's rights on a browser last accounts.ADMIN_S from typing
the password; after that the admin page asks for it again and the editor goes on.

Wrong passwords cost more and more (Limiter): after FREE of them from one client each further try waits longer,
CLIENT_LOCK within WINDOW_S locks that client out, GLOBAL_LOCK from everyone locks every check for WINDOW_S (behind a
tunnel every client may look alike, so the global count is what really stops guessing). A new administrator password
set on this machine clears the counts: that is how the administrator gets back in during a lockout. A wrong username and a
wrong password get the same answer, in the same time (no one learns which accounts exist). Every failure, and
everything else that looks like probing (server/access.py), goes into the suspicious-activity list of the admin page
(Watch).
"""

from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request, Response
from pydantic import BaseModel

from .routes import Access, Router
from .. import accounts, logs
from ..accounts import Session, User, now
from ..config import settings
from ..errors import Invalid, NotSignedIn, TooManyTries
from ..messages import Msg

log = logs.get("auth")

COOKIE = "lab2shot"
MACHINE_HEADER = "x-lab2shot-machine"

FREE = 3  # wrong passwords from one client before it has to wait
CLIENT_LOCK = 10  # wrong passwords from one client within WINDOW_S: it is locked out until the window passes
GLOBAL_LOCK = 30  # wrong passwords from everyone within WINDOW_S: every check stops for WINDOW_S
WINDOW_S = 15 * 60
MAX_WAIT_S = 60
WRONG = Msg("E-LOGIN-WRONG")  # the same for a username that is not there


# ------------------------------------------------------------------ who asks


def _cookie(request: Request) -> str:
    token = request.cookies.get(COOKIE, "")
    return token if 0 < len(token) <= 100 else ""


def _bearer(request: Request) -> str:
    kind, _, token = request.headers.get("authorization", "").partition(" ")
    return token.strip() if kind.lower() == "bearer" and 0 < len(token.strip()) <= 100 else ""


def token_of(request: Request) -> str:
    return _bearer(request) or _cookie(request)


def by_header(request: Request) -> bool:
    """Does this request carry its credential in a header (a client's bearer token, this machine's token)? A page of
    another site can't make a browser send one (it would need CORS, which this server never allows), so such a request
    is never a cross-site forgery; one that relies on the cookie (or on nothing: logging in) may be."""
    return bool(_bearer(request) or request.headers.get(MACHINE_HEADER))


LOOPBACK = {"127.0.0.1", "::1"}
PROXY_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "forwarded", "x-real-ip")


def local_request(request: Request) -> bool:
    """A connection from this machine's own loopback address that no proxy or tunnel says it forwarded."""
    host = request.client.host if request.client else ""
    return host in LOOPBACK and not any(h in request.headers for h in PROXY_HEADERS)


def credential(request: Request) -> str:
    """The token this request carries: this machine's own (MACHINE_HEADER, only from local_request) first, else a
    client's bearer token or a browser's cookie. Looked up the same way (accounts.session) whichever it is."""
    machine_token = request.headers.get(MACHINE_HEADER, "")
    return (machine_token if local_request(request) else "") if machine_token else token_of(request)


def session(request: Request) -> Session | None:
    """The live session this request carries, cached once per request: for a request whose account may change while
    it is still open (a long-lived stream: server/farm.py's job-events), call accounts.session(credential(request))
    directly instead, so each check is fresh."""
    if "lab2shot_session" in request.scope:
        return request.scope["lab2shot_session"]
    found = accounts.session(credential(request))
    if found is not None and found.kind == "machine" and not (request.headers.get(MACHINE_HEADER) and local_request(request)):
        found = None  # the machine token presented any other way (a cookie, a bearer) opens nothing
    request.scope["lab2shot_session"] = found
    return found


def machine(request: Request) -> bool:
    """Is this request the command line on this machine (its machine-token session, accounts.machine_token)?"""
    s = session(request)
    return s is not None and s.kind == "machine"


def user(request: Request) -> User | None:
    s = session(request)
    return s.user if s else None


def me(request: Request) -> User:
    """The account asking (the route table lets only logged-in requests reach a user's route)."""
    u = user(request)
    if u is None:
        raise NotSignedIn(Msg("E-LOGIN-REQUIRED"))
    return u


def label(request: Request) -> str:
    """How the account asking is named in what it writes (an audit line, the notice's `by`): its name,
    or its username without one. The one place for it."""
    u = me(request)
    return u.name or u.username


def can(request: Request, capability: str) -> bool:
    """Does this request's session hold `capability` now (lab2shot/roles.py)?"""
    s = session(request)
    return s is not None and s.can(capability)


def https(request: Request) -> bool:
    """Served over HTTPS: by this server, or by the tunnel in front of it (a false claim only makes the cookie one its
    own browser won't send back over http)."""
    forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    return request.url.scheme == "https" or forwarded == "https"


def who(request: Request) -> str:
    """The client, as far as the connection tells: its address, and the first address a proxy says it forwarded (it
    may lie: that only splits the per-client count, the global one still counts every try)."""
    ip = request.client.host if request.client else "?"
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()[:64]
    return f"{ip}（转发自 {forwarded}）" if forwarded else ip


def details(request: Request, declared: dict | None = None) -> dict:
    """What a request shows about where it comes from, kept with what it does for the administrator (never who it
    is: that is the account): the address, the User-Agent, and what the client says of itself (a DCC plugin's
    application, computer name and OS user; a browser's platform and screen)."""
    said = {k: str(v)[:200] for k, v in (declared or {}).items() if isinstance(v, str | int | float)}
    return {"ip": who(request), "user_agent": request.headers.get("user-agent", "")[:400], **said}


def client_key(request: Request) -> str:
    """Whose requests these are, for counting: the session's token when it opens a live session, else its address.

    只认真开得了会话的令牌：随手编一个 Bearer 就换一个桶的话，匿名洪水每个请求换一个令牌，永远碰不到 429，
    还把 Watch.per / Rate.buckets 撑到没边。没有会话的一律按地址算（隧道后面大家一个地址，所以匿名的不封、
    只限速：见 Watch）。`session` 每个请求只查一次（缓存在 scope 里），这里不多花一次数据库。"""
    s = session(request)
    return f"s:{s.token[:16]}" if s is not None else f"a:{who(request)}"


# ------------------------------------------------------------------ what looks like probing


WATCH_KEPT = 1000  # suspicious events remembered (the newest)
BLOCK_AFTER = 40  # suspicious events of one session within BLOCK_WINDOW_S: it is blocked for BLOCK_S
BLOCK_WINDOW_S = 10 * 60
BLOCK_S = 30 * 60
PER_KEPT = 10_000  # sessions counted at once; above it the ones with nothing inside the window are forgotten


@dataclass
class Watch:
    """What looks like probing, for the admin page's 安全: failed logins, refused routes, odd paths, too many
    requests. A session that trips BLOCK_AFTER of them within BLOCK_WINDOW_S is blocked for BLOCK_S (a client without
    a session is not: behind a tunnel its address may be everyone's, the administrator's too)."""

    events: collections.deque = field(default_factory=lambda: collections.deque(maxlen=WATCH_KEPT))
    per: dict[str, list[float]] = field(default_factory=dict)
    blocked: dict[str, dict] = field(default_factory=dict)  # client key -> {"until", "who", "why"}
    counts: collections.Counter = field(default_factory=collections.Counter)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, request: Request, kind: str, detail: str = "", counts: bool = True) -> None:
        """`counts` False: 记下来给后台的「安全」看，但不算进封禁的 BLOCK_AFTER 条。

        给页面自己发出来的、打到一个这台服务器没有的接口上的请求用（升级窗口里网页新、服务器旧，或者反过来）：
        那不是探测，是版本对不上；算进去会把正在用的人锁在门外 BLOCK_S。"""
        t, key, whom = now(), client_key(request), who(request)
        u = request.scope.get("lab2shot_session")
        account = u.user.username if u else ""
        with self.lock:
            self.events.append({"t": t, "kind": kind, "detail": detail[:300], "who": whom, "client": key, "user": account,
                                "path": request.url.path[:200], "method": request.method, "counted": counts})
            self.counts[kind] += 1
            if counts and key.startswith("s:"):
                tries = [x for x in self.per.get(key, []) if t - x < BLOCK_WINDOW_S] + [t]
                self.per[key] = tries
                if len(self.per) > PER_KEPT:  # 只留窗口内还有记录的（不然一天换一个令牌的客户端把这张表越攒越大）
                    self.per = {k: v for k, v in self.per.items() if v and t - v[-1] < BLOCK_WINDOW_S}
                    self.blocked = {k: v for k, v in self.blocked.items() if v["until"] > t}
                if len(tries) >= BLOCK_AFTER and key not in self.blocked:
                    self.blocked[key] = {"until": t + BLOCK_S, "who": whom, "user": account,
                                         "why": Msg("W-LOGIN-BLOCKED", minutes=BLOCK_WINDOW_S // 60, count=len(tries)).text}
                    logs.say(log, Msg("W-LOGIN-PROBEBLOCKED", account=account, whom=whom, minutes=BLOCK_S // 60, window=BLOCK_WINDOW_S // 60, count=len(tries)))
        logs.say(log, Msg("W-LOGIN-SUSPICIOUS", account=account, whom=whom, kind=kind, method=request.method, path=request.url.path[:200], detail=detail[:200] or "-"))

    def is_blocked(self, request: Request) -> float:
        """Seconds this client is still blocked (0: it is not)."""
        key = client_key(request)
        with self.lock:
            b = self.blocked.get(key)
            if b and b["until"] <= now():
                del self.blocked[key]
                return 0.0
            return b["until"] - now() if b else 0.0

    def unblock(self, key: str) -> None:
        with self.lock:
            self.blocked.pop(key, None)
            self.per.pop(key, None)

    def view(self, limit: int = 300) -> dict:
        t = now()
        with self.lock:
            return {"events": list(self.events)[-limit:][::-1], "counts": dict(self.counts),
                    "blocked": [{"client": k, **v} for k, v in self.blocked.items() if v["until"] > t]}


RATE_PER_S = 200.0
RATE_BURST = 2000.0


@dataclass
class Rate:
    """Requests per client (client_key): a bucket of BURST that refills at PER_S a second — far above what the pages
    ask for while playing a shot or uploading a sequence, far below a flood."""

    buckets: dict[str, list[float]] = field(default_factory=dict)  # key -> [tokens, last time]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, key: str, burst: float | None = None, per_s: float | None = None) -> bool:
        """One request of `key` (a client, or a client on one route with a limit of its own: its declared Limit, server/routes.py)."""
        burst, per_s = RATE_BURST if burst is None else burst, RATE_PER_S if per_s is None else per_s
        t = now()
        with self.lock:
            tokens, last = self.buckets.get(key, (burst, t))
            tokens = min(burst, tokens + (t - last) * per_s)
            if tokens < 1:
                self.buckets[key] = [tokens, t]
                return False
            self.buckets[key] = [tokens - 1, t]
            if len(self.buckets) > 10_000:  # forget the idle ones
                self.buckets = {k: v for k, v in self.buckets.items() if t - v[1] < 600}
            return True


@dataclass
class Limiter:
    """Wrong passwords at one work folder's server: per client and from everyone, within WINDOW_S."""

    per: dict[str, list[float]] = field(default_factory=dict)
    everyone: list[float] = field(default_factory=list)
    locked_until: float = 0.0  # every check, after GLOBAL_LOCK
    seen: str = ""  # the administrator's password these counts are about: a new one clears them
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _fresh(self, t: float) -> None:
        stored = accounts.password_hash(accounts.ADMIN_ID)
        if stored != self.seen:
            self.per.clear()
            self.everyone.clear()
            self.locked_until, self.seen = 0.0, stored
        self.everyone = [x for x in self.everyone if t - x < WINDOW_S]
        for k in list(self.per):
            self.per[k] = [x for x in self.per[k] if t - x < WINDOW_S]
            if not self.per[k]:
                del self.per[k]

    def wait(self, client: str) -> float:
        """Seconds before `client` may try again (0: now)."""
        t = now()
        with self.lock:
            self._fresh(t)
            if self.locked_until > t:
                return self.locked_until - t
            tries = self.per.get(client, [])
            if len(tries) >= CLIENT_LOCK:
                return tries[-CLIENT_LOCK] + WINDOW_S - t
            if len(tries) >= FREE:
                return max(0.0, tries[-1] + min(2 ** (len(tries) - FREE + 1), MAX_WAIT_S) - t)
            return 0.0

    def failed(self, client: str) -> tuple[int, int, bool]:
        """Count a wrong one: (this client's, everyone's, whether everything is locked now)."""
        t = now()
        with self.lock:
            self.per.setdefault(client, []).append(t)
            self.everyone.append(t)
            n, total = len(self.per[client]), len(self.everyone)
            locked = total >= GLOBAL_LOCK and self.locked_until <= t
            if locked:
                self.locked_until = t + WINDOW_S
            return n, total, locked

    def passed(self, client: str) -> None:
        with self.lock:
            self.per.pop(client, None)

    def clear_lock(self) -> int:
        """本机命令行解开全局锁（server/users.py 的 POST /api/admin/security/unlock）：锁解开、每个来源的计数清零，
        不用像 `_fresh` 那样换掉管理员密码。返回清掉了几次失败（留底里写的那个数）。"""
        with self.lock:
            cleared = len(self.everyone)
            self.per.clear()
            self.everyone.clear()
            self.locked_until = 0.0
            return cleared


@dataclass
class Streams:
    """Event streams open per account (server/routes.py MAX_STREAMS). A stream costs this server no thread (it waits
    on the event loop), but it does hold a connection, so one account may not open them without end. A new one over
    the limit is refused and says so; the ones already open are never cut — a page following a job it started must not
    lose it because another tab opened one too."""

    open: dict[int, int] = field(default_factory=dict)  # account -> how many it has open now
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, user_id: int, most: int) -> bool:
        with self.lock:
            if self.open.get(user_id, 0) >= most:
                return False
            self.open[user_id] = self.open.get(user_id, 0) + 1
            return True

    def give_back(self, user_id: int) -> None:
        with self.lock:
            left = self.open.get(user_id, 0) - 1
            if left > 0:
                self.open[user_id] = left
            else:
                self.open.pop(user_id, None)


@dataclass
class Guards:
    """The counters of one work folder's server."""

    limiter: Limiter = field(default_factory=Limiter)
    watch: Watch = field(default_factory=Watch)
    rate: Rate = field(default_factory=Rate)
    streams: Streams = field(default_factory=Streams)


_guards: dict[Path, Guards] = {}
_guards_lock = threading.Lock()


def guards() -> Guards:
    with _guards_lock:
        return _guards.setdefault(settings().work_dir, Guards())


def _wait(seconds: float) -> Msg:
    """输错太多：还要等多久。"""
    if seconds < 90:
        return Msg("E-LOGIN-WAITSECONDS", seconds=int(seconds) + 1)
    return Msg("E-LOGIN-WAITMINUTES", minutes=int(seconds / 60) + 1)


_checking = threading.Lock()  # one check at a time: tries sent together still wait their turn and count


def guarded(request: Request, what: str, check, wrong: Msg | None = None, subject: str = ""):
    """Run a check of a secret (`check()`: what it found, falsy when wrong), counting a wrong one; TooManyTries while
    this client (or everyone) must wait, NotSignedIn (`wrong`, else E-LOGIN-WRONGSECRET about `what`, the word for the secret) when it is wrong.
    `subject`: whose secret (the username tried, the account changing its password): the per-client count is per
    (client, subject) — behind a tunnel a whole company shares one address, and one colleague's ten wrong passwords
    must not lock everyone out for 15 minutes. Everyone's count (GLOBAL_LOCK) still counts every try."""
    client = f"{request.client.host if request.client else '?'}|{subject[:64]}"  # never X-Forwarded-For: anyone sends it, and a new one each try would be a new bucket
    g = guards()
    with _checking:
        wait = g.limiter.wait(client)
        if wait > 0:
            g.watch.note(request, "试错太多被挡", what)
            raise TooManyTries(_wait(wait))
        found = check()
        if not found:
            n, total, locked = g.limiter.failed(client)
            g.watch.note(request, "登录失败", Msg("W-LOGIN-FAILED", what=what, count=n, total=total, minutes=WINDOW_S // 60).text)
            if locked:
                logs.say(log, Msg("W-LOGIN-LOCKED", minutes=WINDOW_S // 60, count=total))
            raise NotSignedIn(wrong or Msg("E-LOGIN-WRONGSECRET", what=what))
        g.limiter.passed(client)
        return found


# ------------------------------------------------------------------ routes (server/access.py: open, or the user's)

router = Router(prefix="/api/auth", tags=["登录"])


def state_of(s: Session | None) -> dict:
    """What a page is told about its browser: logged in or not, the account, what applies to it on the admin side
    (`applies`: server/available.py session, the one availability answer) and until when its rights last; for the
    administrator's own account (User.owner), whether the default password is still in use and whether the 口令 is set."""
    from . import available

    if s is None:
        return {"user": None, "applies": available.session(None).json()}
    out = {"user": s.user.public(), "expires": s.user.expires, "admin_until": s.admin_until, "applies": available.session(s).json()}
    if s.user.owner:
        out |= {"passphrase": accounts.passphrase() is not None}
    return out


def kicked_detail(info: dict) -> Msg:
    """The message the kicked page shows (server/access.py's 401, and /api/auth/state): when, from where, what."""
    import time as _time

    when = _time.strftime("%m-%d %H:%M", _time.localtime(info["at"]))
    return Msg("E-LOGIN-KICKED", when=when, ip=info["ip"], device=info["device"])


@router.get("/state", access=Access.open("登录了没有、哪个账号、是不是管理员"), summary="这个浏览器登录了没有：哪个账号、到期时间、是不是有管理员权限；管理员另外知道是否设了口令；被别处的登录顶掉时，说明什么时候、从哪、什么设备")
def state(request: Request) -> dict:
    s = session(request)
    out = state_of(s)
    if s is None and (info := accounts.kicked_info(credential(request))):
        kicked = kicked_detail(info)
        out["kicked"] = {**info, "detail": kicked.text, "code": kicked.code}
    return out


def _set_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(COOKIE, token, max_age=accounts.SESSION_S, path="/", httponly=True, samesite="strict", secure=https(request))


def _check_login(request: Request, username: str, password: str, kind: str, device_id: str = "", hostname: str = "",
                 app: str = "") -> User:
    """The account a username and password open, or the failure logged (login_log) and raised: rate-limited, wrong
    (the same answer either way: no user enumeration), or not usable now (expired, disabled — only whoever knows the
    password learns this)."""
    agent = request.headers.get("user-agent", "")
    device, ip = accounts.device_label(kind, agent, app), who(request)

    def logged(ok: bool, reason: str, user_id: int | None) -> None:
        accounts.log_login(username, user_id, ok, reason, kind, ip, agent, device, device_id, hostname, [])

    existing = accounts.by_username(username)
    if existing is not None and existing.owner and existing.no_password:  # a new installation: nobody may log in yet
        logged(False, "管理员还没设密码", existing.id)
        raise NotSignedIn(Msg("E-LOGIN-NOPASSWORD", user=existing.username))
    try:
        u = guarded(request, "密码", lambda: accounts.login(username, password), WRONG, subject=username)
    except TooManyTries:
        existing = accounts.by_username(username)
        logged(False, "试错太多，暂时锁住", existing.id if existing else None)
        raise
    except NotSignedIn:
        existing = accounts.by_username(username)
        logged(False, "密码不对" if existing else "用户名不存在", existing.id if existing else None)
        raise
    if problem := u.usable_now():
        logged(False, problem.text, u.id)
        raise NotSignedIn(problem)
    return u


class Login(BaseModel):
    username: str
    password: str
    device_id: str = ""  # kept in the browser's localStorage, sent at every login: this browser, on this computer


@router.post("/login", access=Access.open("登录（输错多了要等）"), summary="登录：用户名和密码对了，这个浏览器换一个新的登录凭证（cookie），30 天内不用再登录；管理员另外有三天的管理权限。同一个账号别的浏览器上的登录会被顶掉。用户名不对和密码不对是同一句回答；输错太多次要等一会儿")
def login(req: Login, request: Request, response: Response) -> dict:
    u = _check_login(request, req.username, req.password, "web", req.device_id)
    token, s = accounts.start(u, "web", who(request), request.headers.get("user-agent", ""), replaces=_cookie(request),
                              device_id=req.device_id)
    _set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-IN", user=u.username, where=who(request)))
    return state_of(s)


class TokenRequest(Login):
    app: str = ""  # which application asks (maya, houdini, nuke, cli, script ...)
    hostname: str = ""  # the computer's name, when the client can tell (a browser cannot)


@router.post("/token", access=Access.open("DCC 插件和命令行登录，拿令牌（输错多了要等）"), summary="DCC 插件和命令行登录：用户名和密码对了，给一个长期有效的令牌（放在请求头 Authorization: Bearer 里）；同一个账号别的 DCC 插件或命令行的登录会被顶掉（浏览器不受影响）；账号停用或过期时一起失效")
def token(req: TokenRequest, request: Request) -> dict:
    u = _check_login(request, req.username, req.password, "client", req.device_id, req.hostname, req.app)
    agent = f"{req.app[:40]} · {request.headers.get('user-agent', '')}"
    got, s = accounts.start(u, "client", who(request), agent, device_id=req.device_id, hostname=req.hostname, app=req.app)
    logs.say(log, Msg("I-LOGIN-TOKEN", user=u.username, app=req.app, where=who(request)) if req.app else Msg("I-LOGIN-CLIENT", user=u.username, where=who(request)))
    return {"token": got, "expires": s.expires, "user": u.public()}


@router.post("/logout", access=Access.open("退出"), summary="退出登录：这个浏览器（或客户端的令牌）不再能用，要重新登录")
def logout(request: Request, response: Response) -> dict:
    s = session(request)
    if token := token_of(request):
        accounts.end(token)
    if s is not None:
        logs.say(log, Msg("I-LOGIN-OUT", user=s.user.username, where=who(request)))
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict", secure=https(request))
    request.scope["lab2shot_session"] = None
    return state_of(None)


class PasswordChange(BaseModel):
    current: str
    new: str


@router.post("/password", access=Access.user("改自己的密码：要现在的密码"), summary="改自己的密码：要现在的密码；改完别处的登录都退出，这个浏览器接着用")
def change_password(req: PasswordChange, request: Request, response: Response) -> dict:
    u = me(request)
    if problem := accounts.rule_problem(req.new, "新密码"):
        raise Invalid(problem)
    guarded(request, "现在的密码", lambda: accounts.matches(req.current, accounts.password_hash(u.id)), subject=str(u.id))
    accounts.set_password(u.id, req.new, "本人")
    token, s = accounts.start(accounts.get(u.id), "web", who(request), request.headers.get("user-agent", ""))
    _set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-PASSWORDCHANGED", user=u.username, where=who(request)))
    return state_of(s)


class Recover(BaseModel):
    passphrase: str
    new: str


@router.post("/recover", access=Access.open("管理员忘了密码：用主人的口令设新密码（输错多了要等）"), summary="管理员忘了密码：用主人在服务器上设的口令设一个新的管理员密码，并且登录")
def recover(req: Recover, request: Request, response: Response) -> dict:
    if problem := accounts.rule_problem(req.new, "新密码"):
        raise Invalid(problem)
    kept = accounts.passphrase()
    if kept is None:
        raise Invalid(Msg("E-LOGIN-NOPASSPHRASE"))
    guarded(request, "口令", lambda: accounts.matches(req.passphrase, kept["hash"]), subject="passphrase")
    accounts.set_password(accounts.ADMIN_ID, req.new, "口令")  # the counts of wrong tries start again with it
    token, s = accounts.start(accounts.admin(), "web", who(request), request.headers.get("user-agent", ""), replaces=_cookie(request))
    _set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-RECOVERED", where=who(request)))
    return state_of(s)
