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

Wrong passwords cost more and more (Limiter), counted per account and per client, never over everyone: nothing a
stranger does can lock every login. A try from a device that has logged in to the account before (its random device id,
accounts.known_device) is counted on its own only; any other try on an account waits longer and longer once strangers
have got it wrong SUBJECT_FREE times, up to MAX_WAIT_S from all of them together, so a guess at one account stays slow
however many addresses it comes from. Where addresses tell clients apart (a trusted proxy's, client_source), one address
is also held to its own counts, on each account and over all of them. A new administrator password set on this machine
clears the counts, and so does `lab2shot admin unlock`. A wrong username and a wrong password get the same answer, in
the same time (no one learns which accounts exist). Every failure, and everything else that looks like probing
(server/access.py), goes into the suspicious-activity list of the admin page (Watch).
"""

from __future__ import annotations

import base64
import collections
import functools
import hashlib
import hmac
import ipaddress
import json
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request, Response
from pydantic import BaseModel

from .routes import Access, Router
from .. import accounts, logs
from ..accounts import Session, User, now
from ..config import proxy_networks, settings
from ..errors import Invalid, NotSignedIn, TooManyTries
from ..messages import Msg

log = logs.get("auth")

COOKIE = "lab2shot"
MACHINE_HEADER = "x-lab2shot-machine"

FREE = 3  # wrong ones from one client (a known device, or an address) on one account before it has to wait
CLIENT_LOCK = 10  # wrong ones from one client on one account within WINDOW_S: it is locked out until the window passes
ADDRESS_FREE = 20  # wrong ones from one address over every account before it has to wait (only where addresses tell apart)
ADDRESS_LOCK = 60  # ... within WINDOW_S: that address is locked out until the window passes
SUBJECT_FREE = 10  # wrong ones on one account from every unknown device together before each further one waits
WINDOW_S = 15 * 60
MAX_WAIT_S = 60  # the longest wait between tries; the per-account one never becomes a lock
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
    """Served over HTTPS: by this server, or by a trusted proxy in front of it that says so (client_source)."""
    return request.url.scheme == "https" or client_source(request).proto == "https"


def host(request: Request) -> str:
    """The host the browser asked for: a trusted proxy's X-Forwarded-Host (client_source), else the Host header."""
    return client_source(request).host or request.headers.get("host", "").lower()


def host_known(request: Request) -> bool:
    """Was this request sent to one of this server's own names? Against DNS rebinding: a page of another site whose name
    now points at this server sends its requests with that name as the Host, and would otherwise pass for this site
    (access.same_site compares Origin with Host). An address always is one (rebinding needs a name); a name must be
    one tls.host_names gives (localhost, this machine's host name, setting server.names). A request that came through
    a trusted proxy is the proxy's to route: it is only asked which name it forwarded for (host)."""
    from urllib.parse import urlsplit

    from .tls import host_names

    if client_source(request).via:
        return True
    try:
        name = urlsplit(f"//{request.headers.get('host', '')}").hostname or ""
    except ValueError:
        return False
    return _address(name) is not None or name.rstrip(".") in host_names()


@functools.lru_cache(maxsize=4)
def _trusted(listed: str) -> tuple:
    return tuple(proxy_networks(listed))


def _address(text: str):
    try:
        return ipaddress.ip_address(text.strip().strip("[]"))
    except ValueError:
        return None


@dataclass(frozen=True)
class Source:
    """Where a request comes from, as far as this server can tell (client_source)."""

    ip: str  # the client's address: what every count, limit and record uses
    apart: bool  # it tells this client apart from others: False for loopback, and for a trusted proxy's own address
    via: str = ""  # the trusted proxy it came through ("" none)
    ignored: bool = False  # it carried forwarded headers that did not count (its connection is no trusted proxy)
    proto: str = ""  # a trusted proxy's X-Forwarded-Proto (lower case; "" none)
    host: str = ""  # a trusted proxy's X-Forwarded-Host (lower case; "" none)


def client_source(request: Request) -> Source:
    """Where a request comes from: the one place every count, limit and record of the server takes the address from.

    The connection's own address, unless it comes from a proxy listed in 可信代理 (server.trusted_proxies): then the
    address that proxy says it forwarded — X-Forwarded-For read from the right, each hop a listed proxy skipped, the
    first that is not one is the client (whatever a client writes to the left of it was only passed along); without
    X-Forwarded-For, X-Real-IP. The same rule holds for every forwarded header: X-Forwarded-Proto (https()) and
    X-Forwarded-Host (host()) are taken only from a trusted proxy too, the value the nearest one set (the last).
    From any other connection none of them counts: anyone can send them, and a new address in each request would be
    a new bucket for every per-client count. The server itself runs without uvicorn's proxy handling
    (server/restart.py proxy_headers=False), so request.client is always the connection.

    Two addresses tell nobody apart (`apart` False): this machine's loopback, which is what every request looks like
    when a tunnel on this machine only passes the connection through (frp TCP to this server's own port: there is
    no header to read, and uvicorn has no PROXY protocol), and a trusted proxy's own address, when it forwarded none.
    Per-client limits that would lump everyone into one (registration's per-address limits, the wrong invite code
    count) leave those out; the site-wide ones still count."""
    peer = request.client.host if request.client else ""
    trusted = _trusted(str(settings()["server.trusted_proxies"]))
    here = _address(peer)
    headers = any(h in request.headers for h in ("x-forwarded-for", "x-real-ip", "x-forwarded-proto", "x-forwarded-host"))
    if here is None:
        return Source(peer or "?", False, ignored=headers)
    if not any(here in net for net in trusted):
        return Source(str(here), not here.is_loopback, ignored=headers)
    hops = [h for h in (x.strip() for x in ",".join(request.headers.getlist("x-forwarded-for")).split(",")) if h]
    if not hops and (real := request.headers.get("x-real-ip", "").strip()):
        hops = [real]
    client, apart = here, False
    for hop in reversed(hops):
        ip = _address(hop)
        if ip is None:  # not an address: the proxy passed along what a client wrote, so it names no one
            break
        client = ip
        if not any(ip in net for net in trusted):
            apart = not ip.is_loopback
            break
    def last(name: str) -> str:
        return ",".join(request.headers.getlist(name)).split(",")[-1].strip().lower()[:200]

    return Source(str(client), apart, via=str(here), proto=last("x-forwarded-proto"), host=last("x-forwarded-host"))


def client_ip(request: Request) -> str:
    """The client's address (client_source)."""
    return client_source(request).ip


def who(request: Request) -> str:
    """The client, for what the server records and shows (the login log, the suspicious-activity list, a job's
    details): its address (client_ip)."""
    return client_ip(request)


def details(request: Request, declared: dict | None = None) -> dict:
    """What a request shows about where it comes from, kept with what it does for the administrator (never who it
    is: that is the account): the address, the User-Agent, and what the client says of itself (a DCC plugin's
    application, computer name and OS user; a browser's platform and screen)."""
    said = {k: str(v)[:200] for k, v in (declared or {}).items() if isinstance(v, str | int | float)}
    return {"ip": who(request), "user_agent": request.headers.get("user-agent", "")[:400], **said}


def client_key(request: Request) -> str:
    """Whose requests these are, for counting: the session's token when it opens a live session, else its address.

    Only a token that opens a live session counts: if any made-up Bearer got a bucket of its own, an anonymous flood
    would send a new token with every request, never reach 429, and grow Watch.per / Rate.buckets without end.
    Everything without a session is counted by address (behind a tunnel everyone shares one, so anonymous clients are
    only rate-limited, never blocked: see Watch). `session` is looked up once per request (cached in the scope), so
    this costs no extra database read."""
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
        """`counts` False: recorded for the admin page's 「安全」, but not counted toward BLOCK_AFTER.

        For a logged-in request from a page of this site to an endpoint this server does not have (server/access.py
        refusal): during an upgrade the page may be newer than the server, or older, and counting that would lock a
        person who is working out for BLOCK_S."""
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
                if len(self.per) > PER_KEPT:  # keep only keys with events inside the window (clients changing tokens would grow it)
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


RULES = {"k": (FREE, CLIENT_LOCK), "c": (FREE, CLIENT_LOCK), "a": (ADDRESS_FREE, ADDRESS_LOCK), "s": (SUBJECT_FREE, None)}
COUNTED_AS = {"k": "登录过的这台设备", "c": "这个来源", "s": "从没登录过的设备一共"}  # a wrong one's first key, in the list
LIMITER_KEPT = 20_000  # keys counted at once; above it the ones longest untouched are forgotten


@dataclass
class Limiter:
    """Wrong secrets at one work folder's server, counted within WINDOW_S under the keys of a try (keys): the try waits
    for the longest wait among them, and a wrong one counts under each. The kinds of key (RULES: when waiting starts,
    and when it becomes a lock):
      - "k:<known>|<subject>": a try from a known device (one that has logged in to this account before, or a live
        session of it: the caller says which, `known`) is counted under this alone. Whatever strangers do, the account's
        owner on their own device is never slowed by it;
      - "c:<address>|<subject>" and "a:<address>" (one address on one account, and over every account): only where the
        address tells clients apart (client_source) — behind a TCP tunnel every client is loopback, and a count of it
        would be everyone's;
      - "s:<subject>": one account from every unknown device together (`per_subject`). It only slows, up to MAX_WAIT_S
        between tries, never locks: from as many addresses as they like, strangers get one guess per MAX_WAIT_S at an
        account, and cannot shut anyone out of it.
    Nothing counts every try together: no number of wrong ones locks everyone out. `per_subject` False for a secret
    that is not an account's (the invite codes of registering, Guards.invites): one count for everyone there would only
    let a stranger slow registering for all."""

    per_subject: bool = True
    per: dict[str, list[float]] = field(default_factory=dict)
    seen: str = ""  # the administrator's password these counts are about: a new one clears them
    lock: threading.Lock = field(default_factory=threading.Lock)

    def keys(self, source: Source, subject: str, known: str) -> list[str]:
        if known:
            return [f"k:{known[:100]}|{subject}"]
        out = [f"c:{source.ip}|{subject}", f"a:{source.ip}"] if source.apart else []
        return out + [f"s:{subject}"] if self.per_subject else out

    def _fresh(self, t: float) -> None:
        stored = accounts.password_hash(accounts.ADMIN_ID)
        if stored != self.seen:
            self.per.clear()
            self.seen = stored
        for k in list(self.per):
            self.per[k] = [x for x in self.per[k] if t - x < WINDOW_S]
            if not self.per[k]:
                del self.per[k]
        if len(self.per) > LIMITER_KEPT:  # a flood of made-up usernames: forget the ones longest untouched
            kept = sorted(self.per, key=lambda k: self.per[k][-1])[-LIMITER_KEPT // 2:]
            self.per = {k: self.per[k] for k in kept}

    def _wait(self, key: str, t: float) -> float:
        free, lock = RULES[key[0]]
        tries = self.per.get(key, [])
        if lock is not None and len(tries) >= lock:
            return tries[-lock] + WINDOW_S - t
        if len(tries) >= free:
            return max(0.0, tries[-1] + min(2 ** (len(tries) - free + 1), MAX_WAIT_S) - t)
        return 0.0

    def wait(self, keys: list[str]) -> float:
        """Seconds before a try under these keys may be checked (0: now)."""
        t = now()
        with self.lock:
            self._fresh(t)
            return max((self._wait(k, t) for k in keys), default=0.0)

    def failed(self, keys: list[str]) -> dict[str, int]:
        """Count a wrong one under each key: how many each has now."""
        t = now()
        with self.lock:
            for k in keys:
                self.per[k] = (self.per.get(k, []) + [t])[-ADDRESS_LOCK:]
            return {k: len(self.per[k]) for k in keys}

    def passed(self, keys: list[str]) -> None:
        """The right secret: its client's and its account's counts start over (not its address's: one account of its
        own that it logs in to now and then must not wipe what an address tried on the others)."""
        with self.lock:
            for k in keys:
                if not k.startswith("a:"):
                    self.per.pop(k, None)

    def clear(self) -> int:
        """Forget every count, for the local command line (POST /api/admin/security/unlock, server/users.py), without
        replacing the administrator's password as `_fresh` does. Returns how many failures were cleared (the number the
        audit line records): each failure is counted under exactly one "k:" or "s:" key."""
        with self.lock:
            cleared = sum(len(v) for k, v in self.per.items() if k[0] in "ks")
            self.per.clear()
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


CHALLENGE_S = 15 * 60  # a proof-of-work challenge is good for this long after it was given
CHALLENGE_KEPT = 200_000  # answered challenges remembered at once (~30 MB); beyond it none is given until the oldest lapse
ANSWER_MAX = 16  # digits of an answer


@dataclass
class Challenges:
    """Proof of work (工作量证明): a challenge this server signed, which a browser answers by trying numbers until
    sha256("<nonce>:<number>") starts with `bits` zero bits (webui/src/platform/powWorker.ts: about a second or two),
    and which the server then checks with one hash. Signed (HMAC with a key of this run: a restart only asks the page
    for a new one), short-lived (CHALLENGE_S), and single-use: a challenge is used up the first time it is redeemed
    with its answer, and every later attempt needs a new one (a wrong answer registers nothing either). What else a
    form needs to carry unforged rides in it (`carry`: the decoy field's name, and the moment it was given, for the
    minimum time to fill a form). No outside captcha service is involved."""

    key: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    used: dict[str, float] = field(default_factory=dict)  # nonce -> until when it would have been good
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _sign(self, body: str) -> str:
        return base64.urlsafe_b64encode(hmac.new(self.key, body.encode(), hashlib.sha256).digest()).decode().rstrip("=")

    def issue(self, bits: int, carry: dict) -> dict:
        """A new challenge: the token to send back, and what the page needs to answer it."""
        t = now()
        with self.lock:
            self.used = {n: until for n, until in self.used.items() if until > t}
            if len(self.used) >= CHALLENGE_KEPT:
                raise TooManyTries(Msg("E-CHALLENGE-BUSY"))
        nonce = secrets.token_hex(16)
        body = base64.urlsafe_b64encode(json.dumps({"n": nonce, "t": t, "b": bits, **carry}).encode()).decode().rstrip("=")
        return {"token": f"{body}.{self._sign(body)}", "nonce": nonce, "bits": bits, **carry}

    def redeem(self, token: str, answer: str) -> dict:
        """Use a challenge up with its answer: what it carries, with "t" the moment it was given, or
        E-CHALLENGE-STALE when it is not this server's, lapsed, answered wrong, or used before. Only an answered
        challenge is remembered as used: remembering unanswered ones would let anyone fill the memory for free."""
        body, _, mac = str(token or "")[:1000].partition(".")
        if not body or not hmac.compare_digest(mac.encode(), self._sign(body).encode()):
            raise Invalid(Msg("E-CHALLENGE-STALE"))
        said = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        answer = str(answer or "")
        digest = hashlib.sha256(f"{said['n']}:{answer}".encode()).digest()
        if not (answer.isascii() and answer.isdigit() and len(answer) <= ANSWER_MAX) or int.from_bytes(digest, "big") >> (256 - said["b"]):
            raise Invalid(Msg("E-CHALLENGE-STALE"))
        t = now()
        with self.lock:
            if said["n"] in self.used or t - said["t"] > CHALLENGE_S:
                raise Invalid(Msg("E-CHALLENGE-STALE"))
            self.used[said["n"]] = said["t"] + CHALLENGE_S
        return said


@dataclass
class Guards:
    """The counters of one work folder's server."""

    limiter: Limiter = field(default_factory=Limiter)
    invites: Limiter = field(default_factory=lambda: Limiter(per_subject=False))  # wrong invite codes (server/register.py)
    challenges: Challenges = field(default_factory=Challenges)
    watch: Watch = field(default_factory=Watch)
    rate: Rate = field(default_factory=Rate)
    streams: Streams = field(default_factory=Streams)


_guards: dict[Path, Guards] = {}
_guards_lock = threading.Lock()


def guards() -> Guards:
    with _guards_lock:
        return _guards.setdefault(settings().work_dir, Guards())


def _wait(seconds: float) -> Msg:
    """Too many wrong tries: how long to wait."""
    if seconds < 90:
        return Msg("E-LOGIN-WAITSECONDS", seconds=int(seconds) + 1)
    return Msg("E-LOGIN-WAITMINUTES", minutes=int(seconds / 60) + 1)


_checking = threading.Lock()  # one check at a time: tries sent together still wait their turn and count


def guarded(request: Request, what: str, check, wrong: Msg | None = None, subject: str = "", limiter: Limiter | None = None,
            kind: str = "登录失败", known: str = ""):
    """Run a check of a secret (`check()`: what it found, falsy when wrong), counting a wrong one; TooManyTries while
    this try must wait (Limiter), NotSignedIn (`wrong`, else E-LOGIN-WRONGSECRET about `what`, the word for the secret)
    when it is wrong. `subject`: whose secret (the username tried, the account changing its password); `known`: the
    device (or session) this try comes from, when it is one the account has used before ('' a stranger: Limiter.keys).
    `limiter`: whose counts (the passwords' by default; Guards.invites for invite codes), `kind`: how the
    suspicious-activity list names a wrong one."""
    g = guards()
    counts = g.limiter if limiter is None else limiter
    keys = counts.keys(client_source(request), subject[:64], known)
    with _checking:
        wait = counts.wait(keys)
        if wait > 0:
            g.watch.note(request, "试错太多被挡", what)
            raise TooManyTries(_wait(wait))
        found = check()
        if not found:
            got = counts.failed(keys)
            if keys:
                g.watch.note(request, kind, Msg("W-LOGIN-FAILED", what=what, whose=COUNTED_AS[keys[0][0]], count=got[keys[0]],
                                                minutes=WINDOW_S // 60).text)
            if got.get(f"s:{subject[:64]}") == SUBJECT_FREE:
                logs.say(log, Msg("W-LOGIN-SLOWED", what=what, subject=subject[:64], minutes=WINDOW_S // 60, count=SUBJECT_FREE,
                                  wait=MAX_WAIT_S))
            raise NotSignedIn(wrong or Msg("E-LOGIN-WRONGSECRET", what=what))
        counts.passed(keys)
        return found


# ------------------------------------------------------------------ routes (server/access.py: open, or the user's)

router = Router(prefix="/api/auth", tags=["登录"])


def state_of(s: Session | None) -> dict:
    """What a page is told about its browser: logged in or not, the account, what applies to it on the admin side
    (`applies`: server/available.py session, the one availability answer) and until when its rights last; for the
    administrator's own account (User.owner), whether the 口令 is set.
    `settings_pages`: the admin side list's 设置 band (available.settings_pages), for a signed-in page to lay out.
    `terms`: the version of the 用户协议 and 隐私政策 the account must agree to before it goes on (lab2shot/terms owed)."""
    from .. import terms
    from . import available

    if s is None:
        return {"user": None, "applies": available.session(None).json()}
    out = {"user": s.user.public(), "expires": s.user.expires, "admin_until": s.admin_until, "applies": available.session(s).json(),
           "settings_pages": available.settings_pages(), "terms": terms.owed(s.user)}
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


def browser_https(request: Request) -> bool:
    """Does the browser behind this request talk HTTPS? What https() knows, or the page's own Origin says so: a browser
    always sends it on the requests that set the cookie (logging in, registering, a new password) and a page cannot
    write it, so a proxy that forgot X-Forwarded-Proto still gets a Secure cookie."""
    from urllib.parse import urlsplit

    return https(request) or urlsplit(request.headers.get("origin", "")).scheme == "https"


def set_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(COOKIE, token, max_age=accounts.SESSION_S, path="/", httponly=True, samesite="strict",
                        secure=browser_https(request))


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
        known = device_id if existing is not None and accounts.known_device(existing.id, device_id) else ""
        u = guarded(request, "密码", lambda: accounts.login(username, password), WRONG, subject=username.strip().lower(),
                    known=known)  # the subject as accounts.login reads it: "Admin " is the same account as "admin"
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
    set_cookie(request, response, token)
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
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict", secure=browser_https(request))
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
    guarded(request, "现在的密码", lambda: accounts.matches(req.current, accounts.password_hash(u.id)), subject=str(u.id),
            known=f"session {session(request).token[:16]}")  # only a live session of the account gets here
    accounts.set_password(u.id, req.new, "本人")
    token, s = accounts.start(accounts.get(u.id), "web", who(request), request.headers.get("user-agent", ""))
    set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-PASSWORDCHANGED", user=u.username, where=who(request)))
    return state_of(s)


class Recover(BaseModel):
    passphrase: str
    new: str
    device_id: str = ""  # as at login: a browser the administrator has logged in from is not slowed by strangers' tries


@router.post("/recover", access=Access.open("管理员忘了密码：用主人的口令设新密码（输错多了要等）"), summary="管理员忘了密码：用主人在服务器上设的口令设一个新的管理员密码，并且登录")
def recover(req: Recover, request: Request, response: Response) -> dict:
    if problem := accounts.rule_problem(req.new, "新密码"):
        raise Invalid(problem)
    kept = accounts.passphrase()
    if kept is None:
        raise Invalid(Msg("E-LOGIN-NOPASSPHRASE"))
    guarded(request, "口令", lambda: accounts.matches(req.passphrase, kept["hash"]), subject="passphrase",
            known=req.device_id if accounts.known_device(accounts.ADMIN_ID, req.device_id) else "")
    accounts.set_password(accounts.ADMIN_ID, req.new, "口令")  # the counts of wrong tries start again with it
    token, s = accounts.start(accounts.admin(), "web", who(request), request.headers.get("user-agent", ""), replaces=_cookie(request))
    set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-RECOVERED", where=who(request)))
    return state_of(s)
