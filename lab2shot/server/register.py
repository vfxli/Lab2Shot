"""Registering oneself over HTTP (lab2shot/registration.py): the login page's 「注册」, open to anyone while 开放注册
is on.

What the HTTP side adds to registration's own limits, all checked here on the server, so calling the routes directly
skips none of them:

  - proof of work (auth.Challenges): the page asks for a challenge when the form opens, a Web Worker answers it while
    the person types (webui/src/platform/powWorker.ts, POW_BITS: about a second or two), and the registration carries
    the answer; a challenge is signed, lasts auth.CHALLENGE_S and is used up by the first registration that carries its
    answer, so every attempt that gets any further (a username tried, a code guessed) costs a new one;
  - the minimum time to fill the form (MIN_FILL_S from the challenge's moment, which rides signed in it) and a decoy
    field (its name rides in the challenge too; the page never shows the field, so a person leaves it empty);
  - request rates per client (the routes' own Limit, auth.Rate) and per network (NET_BURST, NET_PER_S);
  - wrong invite codes counted per client like wrong passwords (auth.guarded with Guards.invites).

Behind a tunnel that only passes the connection through (frp TCP to this server's own port), every request is this
machine's loopback (auth.client_source: it tells nobody apart). The per-network rate, the wrong-code count and
registration's per-address limits then leave it out, so they never become one limit for everyone; the routes' own
rates then are one rate for everyone, which is why they are set for a crowd arriving at once (a person needs one
challenge and one registration), and the site-wide limits, invite codes and the proof of work hold as they are.

Every refusal of the decoy, the timing or a challenge is noted in the suspicious-activity list (auth.Watch), the
same list failed logins go to. A registration that got through logs the new account in, like a login."""

from __future__ import annotations

import secrets

from fastapi import Request, Response
from pydantic import BaseModel

from .routes import Access, Limit, Router
from .. import accounts, logs, registration, roles
from ..config import settings
from ..database import db
from ..errors import Forbidden, Invalid, TooManyTries
from ..messages import Msg
from . import auth, quota
from .access import audit
from .users import day_text

log = logs.get("auth")

router = Router(prefix="/api/auth", tags=["登录"])

POW_BITS = 21  # zero bits the answer's hash starts with: 2^21 tries expected, about 1.5 s in a browser's worker (1.4 M tries a second)
MIN_FILL_S = 4.0  # a registration sent sooner after its challenge was given is not a person typing
TRAPS = ("website", "homepage", "blog", "referrer")  # names the decoy field may have: a bot filling every field fills it
NET_BURST, NET_PER_S = 20.0, 1 / 30  # registrations one network (/24, /64) may send at once and then per second


@router.get("/register", access=Access.open("登录页：现在能不能自己注册、要不要邀请码、有哪些环节"), summary="自己注册：现在开没开放注册、要不要邀请码、注册页上可选的环节；关着时只回答没开放")
def info() -> dict:
    c = registration.counts()
    if not c["open"]:
        return {"open": False}
    return {"open": True, "invite": c["invite"], "stages": accounts.departments(), "paused": bool(c["paused"]),
            "min_fill_s": MIN_FILL_S}


@router.post("/register/challenge", access=Access.open("注册页：领一道工作量证明题（浏览器算一两秒）", limit=Limit(burst=20, per_s=1 / 3)),
             summary="注册前领一道工作量证明题：浏览器找一个数，让 sha256(nonce:数) 以 bits 个 0 位开头，注册时带回；题目有签名、15 分钟内有效、只能用一次")
def challenge() -> dict:
    if not settings()["register.open"]:
        raise Forbidden(Msg("E-REGISTER-CLOSED"))
    return auth.guards().challenges.issue(POW_BITS, {"trap": secrets.choice(TRAPS)})


class Register(BaseModel):
    username: str
    name: str
    department: str  # 环节
    password: str
    again: str
    invite: str = ""
    challenge: str  # the token from /register/challenge
    answer: str  # the number that solves it
    trap: str = ""  # the decoy field: a person never sees it, so it stays empty
    terms: int = 0  # the version of the 用户协议 and 隐私政策 the form showed with its box ticked (0: not ticked)
    device_id: str = ""  # as at login: this browser, on this computer


def _refused(request: Request, what: str, message: Msg) -> Invalid:
    auth.guards().watch.note(request, "注册被挡", what)
    return Invalid(message)


@router.post("/register", access=Access.open("自己注册一个账号，注册好直接登录（有工作量证明、诱饵字段、次数上限）", limit=Limit(burst=10, per_s=1 / 6)),
             summary="自己注册：用户名、中文名、环节、密码两遍，勾选同意的用户协议和隐私政策的版本号，开了邀请码验证时再加邀请码，外加工作量证明的题和答案；规则和管理员建账号完全一样，角色一律普通用户，建好这个浏览器直接登录")
def register(req: Register, request: Request, response: Response) -> dict:
    if not settings()["register.open"]:
        raise Forbidden(Msg("E-REGISTER-CLOSED"))
    g, src = auth.guards(), auth.client_source(request)
    ip = src.ip
    if src.apart and not g.rate.take(f"register {registration.net_of(ip)}", NET_BURST, NET_PER_S):
        g.watch.note(request, "请求太频繁", "注册")
        raise TooManyTries(Msg("E-ACCESS-TOOFAST"))
    said = g.challenges.redeem(req.challenge, req.answer)  # used up here, whatever follows
    if req.trap:
        raise _refused(request, "填了诱饵字段", Msg("E-REGISTER-FAILED"))
    if accounts.now() - said["t"] < MIN_FILL_S:
        raise _refused(request, "填得太快", Msg("E-REGISTER-TOOFAST"))
    if settings()["register.invite"] and req.invite.strip():
        if src.apart:  # wrong codes counted per client, like wrong passwords
            auth.guarded(request, "邀请码", lambda: registration.invite_usable(req.invite), Msg("E-REGISTER-INVITE"),
                         subject="invite", limiter=g.invites, kind="邀请码不对")
        elif not registration.invite_usable(req.invite):  # everyone looks alike: a count would stop them all
            raise _refused(request, "邀请码不对", Msg("E-REGISTER-INVITE"))
    with db().write():  # the account, its quota and the code's use: all or nothing
        u, used = registration.register(req.username, req.name, req.department, req.password, req.again, req.invite,
                                        req.terms, ip, src.apart)
        quota.set_limit(u.id, float(settings()["register.quota_gb"]))
    token, s = accounts.start(u, "web", ip, request.headers.get("user-agent", ""), replaces=auth.token_of(request),
                              device_id=req.device_id)
    auth.set_cookie(request, response, token)
    invite = registration.invite_label(used)
    logs.say(log, Msg("I-REGISTER-DONE", user=u.username, where=ip, invite=invite))
    audit(Msg("I-AUDIT-SELFREGISTERED", username=u.username, name=u.name, department=u.department,
              expires=day_text(u.expires), quota=settings()["register.quota_gb"], tags=registration.tags_label(u.tags),
              invite=invite, ip=ip, role=roles.label(u.role)),
          session=s, method="POST", path=str(request.url.path))
    return auth.state_of(s)
