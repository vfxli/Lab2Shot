"""Logging in over HTTP: who a request is (its account, lab2shot/accounts.py), and the guards on guessing passwords.

The server may be reached from the internet (a port-forwarding tunnel), where every request can arrive from this
machine's own address: an address never proves who asks. Only a password does.

A request is its account's when it carries one of:
  - the browser's cookie (COOKIE), given at login: HttpOnly, SameSite=Strict, Secure over HTTPS;
  - a client's token (Authorization: Bearer ...), given to a DCC plugin or the command line at /api/auth/token and
    kept in ~/.lab2shot/;
  - a DCC plugin's embedded web window's cookie (kind "embedded"): the plugin asks for a one-time ticket with its own
    token (POST /api/auth/embed), its window opens /embed with it and gets a cookie of its own, under the plugin's
    login (accounts.start_embedded: it ends with it, displaces no one, has no rights beyond the editor). See Tickets;
  - this machine's token (MACHINE_HEADER): the command line on the server machine, acting as the administrator;
    taken only from a loopback connection that no proxy forwarded (local_request), never as a cookie or bearer. What
    guards it is its secrecy (accounts.machine_token: a 0600 file in the work folder, 256 random bits): behind a TCP
    tunnel on this machine every request is loopback and carries no forwarding header, so the address check then
    tells nothing apart.
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

from .repeats import Repeats
from .routes import Access, Body, Limit, Router
from .wire import SECRETS
from .words import Word
from .. import accounts, i18n, logs
from ..accounts import Session, User, now
from ..config import proxy_networks, settings
from ..errors import Forbidden, Invalid, MessageError, NotSignedIn, TooManyTries
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
    """A connection from this machine's own loopback address that no proxy or tunnel says it forwarded. It keeps the
    machine token from being taken through an HTTP proxy that says what it forwards; a tunnel that only passes the
    connection through (frp TCP) looks local here, so the token's secrecy is what guards it there."""
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


def signed_in(request: Request) -> Session:
    """The live session of a request to an account's route (the guard lets none through without one): checked outright,
    never by an assert (python -O drops those)."""
    s = session(request)
    if s is None:
        raise NotSignedIn(Msg("E-LOGIN-REQUIRED"))
    return s


def me(request: Request) -> User:
    """The account asking (the route table lets only logged-in requests reach a user's route)."""
    u = user(request)
    if u is None:
        raise NotSignedIn(Msg("E-LOGIN-REQUIRED"))
    return u


def actor(request: Request) -> accounts.Actor:
    """The account asking, as what it does is recorded (an audit line's `who`, a record's `by`): the one place for it."""
    return accounts.Actor.of(me(request))


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


def who(request: Request) -> str:
    """The client, for what the server records and shows (the login log, the suspicious-activity list, a job's
    details), and what every count and limit is kept by: its address (client_source)."""
    return client_source(request).ip


def details(request: Request, declared: dict | None = None) -> dict:
    """What a request shows about where it comes from, kept with what it does for the administrator (never who it
    is: that is the account): the address, the User-Agent, and what the client says of itself (a DCC plugin's
    application, computer name and OS user; a browser's platform and screen)."""
    said = {k: str(v)[:200] for k, v in (declared or {}).items() if isinstance(v, str | int | float)}
    return {"ip": who(request), "user_agent": request.headers.get("user-agent", "")[:400], **said}


def client_key(request: Request) -> str:
    """Whose requests these are, for counting (auth.Rate, Watch): the session's token when it opens a live session;
    without one, where the request comes from (client_source) when that tells clients apart, else its connection.

    Only a token that opens a live session counts: if any made-up Bearer got a bucket of its own, an anonymous flood
    would send a new token with every request, never reach 429, and grow Watch.per / Rate.buckets without end.
    Behind a TCP tunnel on this machine every stranger is its loopback address: counted by address they would share
    one bucket, and one of them flooding the login would lock every other out. The connection is then the only thing
    that is one client's own, so it is what is counted; guessing passwords is held back apart from this, per account
    (Limiter), and anonymous clients are only rate-limited, never blocked (Watch). `session` is looked up once per
    request (cached in the scope), so this costs no extra database read."""
    s = session(request)
    if s is not None:
        return f"s:{s.token[:16]}"
    source = client_source(request)
    if source.apart or request.client is None:
        return f"a:{source.ip}"
    return f"c:{request.client.host}:{request.client.port}"


# ------------------------------------------------------------------ what looks like probing


SUSPICIOUS_S = 600  # the same kind of suspicious request from one address within this long: one log line, counted (Watch)


def _said_again(said: Msg, count: int, last: float) -> None:
    logs.say(log, Msg("W-LOGIN-SUSPICIOUSAGAIN", line=said.text, count=count))


_suspicious = Repeats(SUSPICIOUS_S, lambda key, t, said: logs.say(log, said) or said, _said_again)

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

    def note(self, request: Request, kind: str, detail: str | Word | Msg = "", counts: bool = True) -> None:
        """`counts` False: recorded for the admin page's 「安全」, but not counted toward BLOCK_AFTER.

        For a logged-in request from a page of this site to an endpoint this server does not have (server/access.py
        refusal): during an upgrade the page may be newer than the server, or older, and counting that would lock a
        person who is working out for BLOCK_S."""
        t, key, whom = now(), client_key(request), who(request)
        u = request.scope.get("lab2shot_session")
        account = u.user.username if u else ""
        with self.lock:
            kept = detail if isinstance(detail, (Word, Msg)) else detail[:300]  # a word or a message: said when shown (view)
            self.events.append({"t": t, "kind": kind, "detail": kept, "who": whom, "client": key, "user": account,
                                "path": request.url.path[:200], "method": request.method, "counted": counts})
            self.counts[kind] += 1
            if counts and u is not None:
                tries = [x for x in self.per.get(key, []) if t - x < BLOCK_WINDOW_S] + [t]
                self.per[key] = tries
                if len(self.per) > PER_KEPT:  # keep only keys with events inside the window (clients changing tokens would grow it)
                    self.per = {k: v for k, v in self.per.items() if v and t - v[-1] < BLOCK_WINDOW_S}
                    self.blocked = {k: v for k, v in self.blocked.items() if v["until"] > t}
                if len(tries) >= BLOCK_AFTER and key not in self.blocked:
                    self.blocked[key] = {"until": t + BLOCK_S, "who": whom, "user": account,
                                         "why": Msg("W-LOGIN-BLOCKED", minutes=BLOCK_WINDOW_S // 60, count=len(tries))}
                    logs.say(log, Msg("W-LOGIN-PROBEBLOCKED", account=account, whom=whom, minutes=BLOCK_S // 60, window=BLOCK_WINDOW_S // 60, count=len(tries)))
        # in the log once per address and kind within SUSPICIOUS_S, the rest counted (server/repeats.py): a flood of odd
        # requests, from however many connections, never rolls the log over the records it keeps
        _suspicious.happened((whom, kind), Msg("W-LOGIN-SUSPICIOUS", account=account, whom=whom, kind=kind, method=request.method,
                                              path=request.url.path[:200], detail=kept or "-"))

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
            events, counts = list(self.events)[-limit:][::-1], dict(self.counts)
            blocked = [{"client": k, **v} for k, v in self.blocked.items() if v["until"] > t]
        # kinds are ids (server.watch.<kind> their words), details and reasons words or messages: said in the language now
        return {"events": [{**e, "kind": watch_kind(e["kind"]), "detail": str(e["detail"])} for e in events],
                "counts": {watch_kind(k): n for k, n in counts.items()},
                "blocked": [{**b, "why": str(b["why"])} for b in blocked]}


def watch_kind(kind: str) -> str:
    """A kind of suspicious activity (Watch.note: an id) in words."""
    return i18n.t(f"server.watch.{kind}")


RATE_PER_S = 200.0
RATE_BURST = 2000.0


RATE_KEPT = 10_000  # buckets remembered at once; past it the longest unused go first


@dataclass
class Rate:
    """Requests per client (client_key): a bucket of BURST that refills at PER_S a second — far above what the pages
    ask for while playing a shot or uploading a sequence, far below a flood. At most RATE_KEPT buckets, the least
    recently used forgotten first, a few at a time: behind a tunnel every connection is a client of its own, and a
    stranger opening connection after connection must not make each request pay for the whole table. A bucket
    forgotten is full again, which is where a client that has been quiet longest would be anyway."""

    buckets: collections.OrderedDict = field(default_factory=collections.OrderedDict)  # key -> [tokens, last time]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, key: str, burst: float | None = None, per_s: float | None = None) -> bool:
        """One request of `key` (a client, or a client on one route with a limit of its own: its declared Limit, server/routes.py)."""
        burst, per_s = RATE_BURST if burst is None else burst, RATE_PER_S if per_s is None else per_s
        t = now()
        with self.lock:
            tokens, last = self.buckets.pop(key, (burst, t))
            tokens = min(burst, tokens + (t - last) * per_s)
            ok = tokens >= 1
            self.buckets[key] = [tokens - 1 if ok else tokens, t]  # the newest at the end
            while len(self.buckets) > RATE_KEPT:
                self.buckets.popitem(last=False)
            return ok


Key = tuple[str, str]  # a Limiter count: its kind (RULES) and whose it is
RULES = {"k": (FREE, CLIENT_LOCK), "c": (FREE, CLIENT_LOCK), "a": (ADDRESS_FREE, ADDRESS_LOCK), "s": (SUBJECT_FREE, None)}
COUNTED_AS = {k: Word(f"server.counted_as.{k}") for k in ("k", "c", "s")}  # a wrong one's first key, in the list
LIMITER_KEPT = 20_000  # keys counted at once; above it the ones longest untouched are forgotten


@dataclass
class Limiter:
    """Wrong secrets at one work folder's server, counted within WINDOW_S under the keys of a try (keys): the try waits
    for the longest wait among them, and a wrong one counts under each. The kinds of key (RULES: when waiting starts,
    and when it becomes a lock):
      - ("k", "<known>|<subject>"): a try from a known device (one that has logged in to this account before, or a live
        session of it: the caller says which, `known`) is counted under this alone. Whatever strangers do, the account's
        owner on their own device is never slowed by it;
      - ("c", "<address>|<subject>") and ("a", "<address>") (one address on one account, and over every account): only where the
        address tells clients apart (client_source) — behind a TCP tunnel every client is loopback, and a count of it
        would be everyone's;
      - ("s", "<subject>"): one account from every unknown device together (`per_subject`). It only slows, up to MAX_WAIT_S
        between tries, never locks: from as many addresses as they like, strangers get one guess per MAX_WAIT_S at an
        account, and cannot shut anyone out of it.
    Nothing counts every try together: no number of wrong ones locks everyone out. `per_subject` False for a secret
    that is not an account's (the invite codes of registering, Guards.invites): one count for everyone there would only
    let a stranger slow registering for all."""

    per_subject: bool = True
    per: dict[Key, list[float]] = field(default_factory=dict)
    seen: str = ""  # the administrator's password these counts are about: a new one clears them
    checking: set[Key] = field(default_factory=set)  # the keys of the tries being checked now (begin)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def keys(self, source: Source, subject: str, known: str) -> list[Key]:
        if known:
            return [("k", f"{known[:100]}|{subject}")]
        out = [("c", f"{source.ip}|{subject}"), ("a", source.ip)] if source.apart else []
        return out + [("s", subject)] if self.per_subject else out

    def begin(self, keys: list[Key]) -> bool:
        """Start checking a try under these keys; False when a try sharing one of them is being checked now. Until
        `end`, a try that shares a key is refused, never queued: it would see the counts before this one's."""
        with self.lock:
            if not self.checking.isdisjoint(keys):
                return False
            self.checking.update(keys)
            return True

    def end(self, keys: list[Key]) -> None:
        with self.lock:
            self.checking.difference_update(keys)

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

    def _wait(self, key: Key, t: float) -> float:
        free, lock = RULES[key[0]]
        tries = self.per.get(key, [])
        if lock is not None and len(tries) >= lock:
            return tries[-lock] + WINDOW_S - t
        if len(tries) >= free:
            return max(0.0, tries[-1] + min(2 ** (len(tries) - free + 1), MAX_WAIT_S) - t)
        return 0.0

    def wait(self, keys: list[Key]) -> float:
        """Seconds before a try under these keys may be checked (0: now)."""
        t = now()
        with self.lock:
            self._fresh(t)
            return max((self._wait(k, t) for k in keys), default=0.0)

    def failed(self, keys: list[Key]) -> dict[Key, int]:
        """Count a wrong one under each key: how many each has now."""
        t = now()
        with self.lock:
            for k in keys:
                self.per[k] = (self.per.get(k, []) + [t])[-ADDRESS_LOCK:]
            return {k: len(self.per[k]) for k in keys}

    def passed(self, keys: list[Key]) -> None:
        """The right secret: its client's and its account's counts start over (not its address's: one account of its
        own that it logs in to now and then must not wipe what an address tried on the others)."""
        with self.lock:
            for k in keys:
                if k[0] != "a":
                    self.per.pop(k, None)

    def clear(self) -> int:
        """Forget every count, for the local command line (POST /api/admin/security/unlock, server/users.py), without
        replacing the administrator's password as `_fresh` does. Returns how many failures were cleared (the number the
        audit line records): each failure is counted under exactly one "k" or "s" key."""
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


TICKET_S = 60  # an embedded window's ticket is good this long after it was given, and for one use
TICKETS_KEPT = 10_000  # tickets outstanding at once over every account; past it none is given until some lapse
TICKETS_PER_LOGIN = 10  # outstanding at once for one plugin login


@dataclass(frozen=True)
class Ticket:
    """What a ticket stands for: the plugin login (its token's key: accounts.session_by_key) that asked, its account,
    and until when it may be used."""

    parent: str
    user_id: int
    until: float


@dataclass
class Tickets:
    """One-time tickets for a DCC plugin's embedded web window (POST /api/auth/embed gives one, GET /embed takes it):
    in this server's memory only, as their sha256 (a restart forgets them all: the plugin asks again), each good for
    TICKET_S and for one use — taken out of the table the moment it is used, whether it then opens a login or not. A
    ticket is said in the log only by its first six characters (prefix)."""

    kept: dict[str, Ticket] = field(default_factory=dict)  # sha256 of the ticket -> what it stands for
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _fresh(self, t: float) -> None:
        self.kept = {k: v for k, v in self.kept.items() if v.until > t}

    def issue(self, parent: Session) -> str:
        t = now()
        with self.lock:
            self._fresh(t)
            if len(self.kept) >= TICKETS_KEPT or sum(v.parent == parent.token for v in self.kept.values()) >= TICKETS_PER_LOGIN:
                raise TooManyTries(Msg("E-EMBED-TOOMANY", seconds=TICKET_S))
            ticket = secrets.token_urlsafe(32)
            self.kept[accounts.sha(ticket)] = Ticket(parent.token, parent.user.id, t + TICKET_S)
        return ticket

    def peek(self, ticket: str) -> Ticket | None:
        """What a ticket stands for while it is still good, leaving it there (the page that asks which account to keep
        has not used it)."""
        if not ticket or len(ticket) > 100:
            return None
        with self.lock:
            found = self.kept.get(accounts.sha(ticket))
            return found if found is not None and found.until > now() else None

    def take(self, ticket: str) -> Ticket | None:
        """Use a ticket up: what it stood for, or None when it is not one, lapsed or used before. Taken out either way."""
        if not ticket or len(ticket) > 100:
            return None
        with self.lock:
            found = self.kept.pop(accounts.sha(ticket), None)
            return found if found is not None and found.until > now() else None


def prefix(ticket: str) -> str:
    """How the log names a ticket: its first six characters, never the rest."""
    return (ticket or "")[:6]


@dataclass
class Guards:
    """The counters of one work folder's server."""

    tickets: Tickets = field(default_factory=Tickets)
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


def guarded(request: Request, what: str, check, wrong: Msg | None = None, subject: str = "", limiter: Limiter | None = None,
            kind: str = "login_failed", known: str = "", refuse: type[MessageError] = NotSignedIn):
    """Run a check of a secret (`check()`: what it found, falsy when wrong), counting a wrong one; TooManyTries while
    this try must wait (Limiter), NotSignedIn (`wrong`, else E-LOGIN-WRONGSECRET about `what`, the word for the secret)
    when it is wrong. `subject`: whose secret, each kind apart (user:<the username tried>, uid:<the account changing
    its password>, passphrase: the 口令; a username may be any word); `known`: the
    device (or session) this try comes from, when it is one the account has used before ('' a stranger: Limiter.keys).
    `limiter`: whose counts (the passwords' by default; Guards.invites for invite codes), `kind`: how the
    suspicious-activity list names a wrong one. `refuse`: the error a wrong one raises (NotSignedIn, 401: not logged
    in; Forbidden, 403, for a secret a live login is asked for again, which stays logged in)."""
    g = guards()
    counts = g.limiter if limiter is None else limiter
    keys = counts.keys(client_source(request), subject[:64], known)
    # Tries that share a key are checked one at a time, so each sees what the ones before it counted; one that arrives
    # while another is being checked is told to wait a second rather than kept waiting on a thread. Tries with no key
    # in common (other accounts; behind a tunnel, whoever sends them) are checked side by side.
    if not counts.begin(keys):
        raise TooManyTries(_wait(0))
    try:
        wait = counts.wait(keys)
        if wait > 0:
            g.watch.note(request, "too_many_tries", what)
            raise TooManyTries(_wait(wait))
        found = check()
        if not found:
            got = counts.failed(keys)
            if keys:
                g.watch.note(request, kind, Msg("W-LOGIN-FAILED", what=what, whose=COUNTED_AS[keys[0][0]], count=got[keys[0]],
                                                minutes=WINDOW_S // 60))
            if got.get(("s", subject[:64])) == SUBJECT_FREE:
                logs.say(log, Msg("W-LOGIN-SLOWED", what=what, subject=subject[:64], minutes=WINDOW_S // 60, count=SUBJECT_FREE,
                                  wait=MAX_WAIT_S))
            raise refuse(wrong or Msg("E-LOGIN-WRONGSECRET", what=what))
        counts.passed(keys)
        return found
    finally:
        counts.end(keys)


# ------------------------------------------------------------------ routes (server/access.py: open, or the user's)

router = Router(prefix="/api/auth", tags=["Login"])


def state_of(s: Session | None) -> dict:
    """What a page is told about its browser: logged in or not, the account, what applies to it on the admin side
    (`applies`: server/available.py session, the one availability answer) and until when its rights last; for the
    administrator's own account (User.owner), whether the 口令 is set.
    `settings_pages`: the admin side list's 设置 band (available.settings_pages), for a signed-in page to lay out.
    `terms`: the version of the 用户协议 and 隐私政策 the account must agree to before it goes on (lab2shot/terms owed)."""
    from .. import terms
    from . import available

    from .. import i18n

    if s is None:  # `lang`: what this request is said in (server/lang.py), the page's language from here on
        return {"user": None, "applies": available.session(None).json(), "lang": i18n.current()}
    out = {"user": s.user.public(), "expires": s.user.expires, "admin_until": s.admin_until, "applies": available.session(s).json(),
           "settings_pages": available.settings_pages(), "terms": terms.owed(s.user), "lang": i18n.current()}
    if s.user.owner:
        out |= {"passphrase": accounts.passphrase() is not None}
    return out


def kicked_detail(info: dict) -> Msg:
    """The message the kicked page shows (server/access.py's 401, and /api/auth/state): when, from where, what. An
    embedded window whose plugin login ended (accounts.kicked_info kind "embedded") is told to open it again from
    the DCC."""
    import time as _time

    if info.get("kind") == "embedded":
        return Msg("E-EMBED-ENDED")

    when = _time.strftime("%m-%d %H:%M", _time.localtime(info["at"]))
    return Msg("E-LOGIN-KICKED", when=when, ip=info["ip"], device=info["device"])


@router.get("/state", access=Access.open("Whether logged in, which account, whether an administrator"), summary="Whether this browser is logged in: which account, when it expires, whether it has admin rights; an "
                                                                                                         "administrator also learns whether a passphrase is set; when replaced by a login elsewhere, says when, from "
                                                                                                         "where and on what device")
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
        logged(False, "N-LOGIN-REASONNOPASSWORD", existing.id)
        raise NotSignedIn(Msg("E-LOGIN-NOPASSWORD", user=existing.username))
    try:
        known = device_id if existing is not None and accounts.known_device(existing.id, device_id) else ""
        u = guarded(request, Word("server.secret.password"), lambda: accounts.login(username, password), WRONG, subject=f"user:{username.strip().lower()}",
                    known=known)  # the subject as accounts.login reads it: "Admin " is the same account as "admin"
    except TooManyTries:
        existing = accounts.by_username(username)
        logged(False, "N-LOGIN-REASONLOCKED", existing.id if existing else None)
        raise
    except NotSignedIn:
        existing = accounts.by_username(username)
        logged(False, "N-LOGIN-REASONWRONGPASSWORD" if existing else "N-LOGIN-REASONNOUSER", existing.id if existing else None)
        raise
    if problem := u.usable_now():
        logged(False, problem.code, u.id)
        raise NotSignedIn(problem)
    return u


class Login(Body):
    username: str
    password: str
    device_id: str = ""  # kept in the browser's localStorage, sent at every login: this browser, on this computer


# Logging in has a rate of its own, per client (client_key: behind a tunnel, per connection), so that what fills the
# shared count, or another client's flood, never keeps anyone from logging in. At this rate one client's password
# checks never outrun the SECRETS lane (wire.LANE_THREADS).
LOGIN = Limit(burst=30, per_s=5)


@router.post("/login", access=Access.open("Log in (wait after too many wrong tries)", limit=LOGIN, lane=SECRETS), summary="Log in: with the right username and password this browser gets a new login credential (cookie) good for 30 "
                                                                                                                "days; an administrator also gets admin rights for three days. A login of the same account in another browser "
                                                                                                                "is replaced. A wrong username and a wrong password get the same answer; after too many wrong tries, wait a "
                                                                                                                "while")
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


@router.post("/token", access=Access.open("DCC plugin and command line login, getting a token (wait after too many wrong tries)", limit=LOGIN, lane=SECRETS), summary="DCC plugin and command line login: with the right username and password, a long-lived token (sent in the "
                                                                                                                                                  "header Authorization: Bearer); another DCC plugin or command line login of the same account is replaced "
                                                                                                                                                  "(browsers are not affected); it ends when the account is disabled or expires")
def token(req: TokenRequest, request: Request) -> dict:
    u = _check_login(request, req.username, req.password, "client", req.device_id, req.hostname, req.app)
    agent = f"{req.app[:40]} · {request.headers.get('user-agent', '')}"
    got, s = accounts.start(u, "client", who(request), agent, device_id=req.device_id, hostname=req.hostname, app=req.app)
    logs.say(log, Msg("I-LOGIN-TOKEN", user=u.username, app=req.app, where=who(request)) if req.app else Msg("I-LOGIN-CLIENT", user=u.username, where=who(request)))
    return {"token": got, "expires": s.expires, "user": u.public()}


@router.post("/logout", access=Access.open("Log out"), summary="Log out: this browser (or the client's token) can no longer be used until it logs in again")
def logout(request: Request, response: Response) -> dict:
    s = session(request)
    if token := token_of(request):
        accounts.end(token)
    if s is not None:
        logs.say(log, Msg("I-LOGIN-OUT", user=s.user.username, where=who(request)))
    response.delete_cookie(COOKIE, path="/", httponly=True, samesite="strict", secure=browser_https(request))
    request.scope["lab2shot_session"] = None
    return state_of(None)


class PasswordChange(Body):
    current: str
    new: str


@router.post("/password", access=Access.user("Change your own password: needs the current one", lane=SECRETS), summary="Change your own password: needs the current one; logins elsewhere are ended, this browser stays logged in")
def change_password(req: PasswordChange, request: Request, response: Response) -> dict:
    u = me(request)
    if problem := accounts.rule_problem(req.new, Word("server.secret.new_password")):
        raise Invalid(problem)
    guarded(request, Word("server.secret.current_password"), lambda: accounts.matches(req.current, accounts.password_hash(u.id)), subject=f"uid:{u.id}",
            known=f"session {session(request).token[:16]}", refuse=Forbidden)  # only a live session of the account gets here
    accounts.set_password(u.id, req.new, accounts.BY_SELF)
    token, s = accounts.start(accounts.get(u.id), "web", who(request), request.headers.get("user-agent", ""))
    set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-PASSWORDCHANGED", user=u.username, where=who(request)))
    return state_of(s)


class Recover(Body):
    passphrase: str
    new: str
    device_id: str = ""  # as at login: a browser the administrator has logged in from is not slowed by strangers' tries


@router.post("/recover", access=Access.open("Administrator forgot the password: set a new one with the owner's passphrase (wait after too many wrong tries)", limit=LOGIN, lane=SECRETS), summary="Administrator forgot the password: set a new administrator password with the passphrase the owner set on the "
                                                                                                                                                                  "server, and log in")
def recover(req: Recover, request: Request, response: Response) -> dict:
    if problem := accounts.rule_problem(req.new, Word("server.secret.new_password")):
        raise Invalid(problem)
    kept = accounts.passphrase()
    if kept is None:
        raise Invalid(Msg("E-LOGIN-NOPASSPHRASE"))
    guarded(request, Word("server.secret.passphrase"), lambda: accounts.matches(req.passphrase, kept["hash"]), subject="passphrase:",
            known=req.device_id if accounts.known_device(accounts.ADMIN_ID, req.device_id) else "")
    accounts.set_password(accounts.ADMIN_ID, req.new, accounts.BY_PASSPHRASE)  # the counts of wrong tries start again with it
    token, s = accounts.start(accounts.admin(), "web", who(request), request.headers.get("user-agent", ""), replaces=_cookie(request))
    set_cookie(request, response, token)
    logs.say(log, Msg("I-LOGIN-RECOVERED", where=who(request)))
    return state_of(s)


# ------------------------------------------------------------------ a DCC plugin's embedded web window (Tickets)


EMBED = Limit(burst=10, per_s=1)  # tickets one plugin login may ask for: a window opened now and then
_FRAGMENT_SAFE = "=&-_.~!$'()*+,;:@/?%"


def embed_target(to: object) -> str | None:
    """Where an embedded window may be sent after its ticket is taken: only the editor of this site ("/"), with an
    optional fragment (#job=…&focus=…, read by the page itself). Anything else — another site, "//host", a backslash,
    a scheme, a query, a path other than "/", whitespace or control characters — is None: never an open redirect.
    The answer is built from the parts checked here, the fragment percent-encoded, never the text as given."""
    from urllib.parse import quote, urlsplit

    if not isinstance(to, str) or not to.startswith("/") or to.startswith("//") or len(to) > 1000:
        return None
    if any(ord(c) < 0x21 or ord(c) == 0x7F or c == "\\" for c in to):
        return None
    try:
        parts = urlsplit(to)
    except ValueError:
        return None
    if parts.scheme or parts.netloc or parts.query or parts.path != "/":
        return None
    return "/" + (f"#{quote(parts.fragment, safe=_FRAGMENT_SAFE)}" if parts.fragment else "")


class EmbedRequest(Body):
    to: str = "/"  # where the window goes once logged in (embed_target): "/#job=<id>&focus=<node>"


@router.post("/embed", access=Access.user("DCC plugin asks for a one-time ticket for an embedded window (plugin tokens only)", limit=EMBED),
             summary="One-time ticket a DCC plugin asks for before opening an embedded web window (only with the plugin's token, "
                     "Authorization: Bearer): valid for 60 seconds and one use; opening url (/embed?ticket=...&to=...) with it logs "
                     "the window in as this account's embedded window, replacing no login; it ends with the plugin's token when that "
                     "logs out or is replaced. to may only be an editor address of this site (/ plus #...)")
def embed_ticket(req: EmbedRequest, request: Request) -> dict:
    from urllib.parse import urlencode

    s = signed_in(request)
    if s.kind != "client" or not _bearer(request):  # a browser's cookie, an embedded window's, the machine token: no
        guards().watch.note(request, "embed_not_plugin", s.kind)
        raise Forbidden(Msg("E-EMBED-CLIENTONLY"))
    where = embed_target(req.to)
    if where is None:
        raise Invalid(Msg("E-EMBED-BADTARGET"))
    ticket = guards().tickets.issue(s)
    logs.say(log, Msg("I-LOGIN-EMBEDTICKET", user=s.user.username, ticket=prefix(ticket), where=who(request)))
    return {"ticket": ticket, "expires_in": TICKET_S, "url": "/embed?" + urlencode({"ticket": ticket, "to": where})}


embed_pages = Router()  # GET /embed: a page address outside /api/ (server/app.py includes it before the page's own)


def _embed_page(status: int, title: Msg, body: Msg, actions: tuple[tuple[str, str, str], ...] = ()) -> Response:
    """The few words /embed says when it does not go on to the editor (a ticket used or lapsed, a refused target,
    another account in this browser): a page of its own, plain HTML with no script (the guard's CSP holds), its words
    from the message catalogue. `actions`: (label, address, what it does) links."""
    import html

    links = "".join(f'<a class="b{" p" if n == 0 else ""}" href="{html.escape(href)}">{html.escape(label)}</a>'
                    f'<span class="w">{html.escape(what)}</span>' for n, (label, href, what) in enumerate(actions))
    page = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f"<title>Lab2Shot · {html.escape(title.text)}</title><style>"
        "body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0e0f12;"
        "color:#e6e7ea;font:14px/1.6 system-ui,-apple-system,'PingFang SC','Microsoft YaHei',sans-serif}"
        "main{max-width:520px;padding:28px 32px;border:1px solid #2a2d34;border-radius:10px;background:#16181d}"
        "h1{font-size:17px;margin:0 0 10px}p{margin:0 0 18px;color:#b4b7bf}"
        ".b{display:inline-block;padding:6px 14px;border-radius:6px;border:1px solid #3a3e47;color:#e6e7ea;"
        "text-decoration:none;margin:0 8px 4px 0}.b.p{background:#2f6fed;border-color:#2f6fed}"
        ".w{display:block;color:#8a8e98;font-size:12px;margin:0 0 12px}"
        f"</style></head><body><main data-lab2shot-embed=\"{html.escape(body.code)}\"><h1>{html.escape(title.text)}</h1>"
        f"<p>{html.escape(body.text)}</p>{links}</main></body></html>")
    return Response(page, status_code=status, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


def _embed_refused(request: Request, status: int, body: Msg, noted: str, ticket: str) -> Response:
    guards().watch.note(request, noted, prefix(ticket))
    logs.say(log, Msg("W-LOGIN-EMBEDREFUSED", why=body.text, ticket=prefix(ticket), where=who(request)))
    return _embed_page(status, Msg("N-EMBED-TITLE"), body)


@embed_pages.get("/embed", access=Access.page("A DCC plugin's embedded window logs in with a one-time ticket, then opens the editor"), include_in_schema=False)
def embed(request: Request, ticket: str = "", to: str = "/", switch: int = 0) -> Response:
    """Take an embedded window's ticket (Tickets) and send it on to `to` (embed_target), logged in as the plugin's
    account under the plugin's login (accounts.start_embedded):
      - a browser already holding a live login of the same account keeps it untouched (a system browser opened as a
        fallback: the user's own login is never replaced); the ticket is used up all the same;
      - one holding another account's login is never overwritten unasked: a page says so and offers the choice; only
        `switch=1` (its button, a click on this site's own page or an address typed or opened by the DCC) logs that
        one out here and goes on;
      - a request another site's page made (Sec-Fetch-Site cross-site / same-site) is refused, its ticket used up: a
        link of another site must never log a browser into an account it did not choose (login CSRF; the cookie being
        SameSite=Strict, such a request would not even show the login this browser has).
    The ticket is in the log only by its first six characters."""
    from urllib.parse import quote

    from fastapi.responses import RedirectResponse

    where = embed_target(to)
    if where is None:
        return _embed_refused(request, 400, Msg("E-EMBED-BADTARGET"), "embed_bad_target", ticket)
    fetched = request.headers.get("sec-fetch-site", "").lower()
    if fetched in ("cross-site", "same-site"):
        guards().tickets.take(ticket)
        return _embed_refused(request, 403, Msg("E-EMBED-CROSSSITE"), "embed_cross_site", ticket)
    found = guards().tickets.peek(ticket)
    if found is None:
        return _embed_refused(request, 401, Msg("E-EMBED-TICKET", seconds=TICKET_S), "embed_bad_ticket", ticket)
    here = session(request) if _cookie(request) else None  # this browser's own login (never a bearer: a browser sends none)
    if here is not None and here.user.id == found.user_id:
        guards().tickets.take(ticket)
        logs.say(log, Msg("I-LOGIN-EMBEDKEPT", user=here.user.username, ticket=prefix(ticket), where=who(request)))
        return RedirectResponse(where, status_code=303, headers={"Cache-Control": "no-store"})
    if here is not None and not switch:
        other = accounts.get(found.user_id)
        again = "/embed?ticket=" + quote(ticket, safe="") + "&to=" + quote(where, safe="") + "&switch=1"
        return _embed_page(409, Msg("N-EMBED-OTHERTITLE"),
                           Msg("E-EMBED-OTHERACCOUNT", here=here.user.label, plugin=other.label, seconds=TICKET_S),
                           ((Msg("N-EMBED-SWITCH", plugin=other.label).text, again, Msg("N-EMBED-SWITCHWHAT", here=here.user.label).text),
                            (Msg("N-EMBED-KEEP", here=here.user.label).text, "/", Msg("N-EMBED-KEEPWHAT").text)))
    taken = guards().tickets.take(ticket)
    parent = accounts.session_by_key(taken.parent) if taken is not None else None
    if taken is None or parent is None or parent.kind != "client" or parent.user.id != taken.user_id:
        return _embed_refused(request, 401, Msg("E-EMBED-TICKET", seconds=TICKET_S), "embed_bad_ticket", ticket)
    if here is not None:  # switch=1: this browser's login of another account, ended at the user's choice
        accounts.end(_cookie(request))
        logs.say(log, Msg("I-LOGIN-OUT", user=here.user.username, where=who(request)))
    token, s = accounts.start_embedded(parent, who(request), request.headers.get("user-agent", ""))
    response = RedirectResponse(where, status_code=303, headers={"Cache-Control": "no-store"})
    response.set_cookie(COOKIE, token, max_age=max(1, int(s.expires - now())), path="/", httponly=True, samesite="strict",
                        secure=https(request))
    logs.say(log, Msg("I-LOGIN-EMBED", user=parent.user.username, ticket=prefix(ticket), where=who(request)))
    return response
