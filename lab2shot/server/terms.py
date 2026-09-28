"""用户协议与隐私政策 over HTTP (lab2shot/terms): reading them, agreeing to them, and the administrator's copy.

    GET  /api/terms           anyone (the login and registration pages link to them): both texts as a person reads them
    POST /api/auth/terms      the account asking agrees to the version its page showed; under /api/auth/, which the
                              guard answers before an account agreed (server/access.py), like logging in and out
    /api/admin/terms          the texts as written, to read and edit (terms.edit): save a copy, or go back to the
                              program's own texts; either, when the texts change, is a new version everyone agrees to"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel

from .. import logs, terms
from ..messages import Msg
from . import auth
from .routes import Access, Router

log = logs.get("auth")

router = Router(prefix="/api", tags=["登录"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


def _public(t: terms.Terms) -> dict:
    return {"version": t.version, "at": t.at, "documents": t.shown()}


@router.get("/terms", access=Access.open("登录页、注册页：用户协议和隐私政策"), summary="用户协议和隐私政策：现在的版本号、生效时间和两份文字（里面提到的保留天数等按现在的设置填好）")
def read() -> dict:
    return _public(terms.current())


class Agree(BaseModel):
    version: int  # the version the page showed


@router.post("/auth/terms", access=Access.user("同意用户协议和隐私政策（改过以后第一次用之前）"), summary="同意用户协议和隐私政策：带上页面显示的版本号；这期间文字又改了就要重新读过再同意")
def agree(req: Agree, request: Request) -> dict:
    u = auth.me(request)
    ip = auth.client_ip(request)
    terms.agree(u.id, req.version, ip)
    logs.say(log, Msg("I-TERMS-AGREED", user=u.username, version=req.version, where=ip))
    return auth.state_of(auth.session(request))


def _view(t: terms.Terms) -> dict:
    """The texts as written, with what the placeholders say now and how many accounts agreed to this version."""
    return {"version": t.version, "at": t.at, "by": t.by, "edited": t.edited, "most": terms.MOST,
            "documents": [{"id": doc, "title": title, "text": t.texts[doc]} for doc, title in terms.DOCS.items()],
            "fills": terms.fills(), **terms.agreed_count(t.version)}


@admin.get("/terms", access=Access.admin("terms.edit"), summary="用户协议和隐私政策（编辑用）：两份原文、版本号、谁什么时候改的、可以写的占位符和现在的值、多少账号已经同意了这一版")
def admin_read() -> dict:
    return _view(terms.current())


class Texts(BaseModel):
    agreement: str
    privacy: str


@admin.put("/terms", access=Access.admin("terms.edit"), summary="改用户协议和隐私政策：两份一起存（纯文字，每份最多 2 万字）；和现在不一样就是新的一版，所有账号下次使用前要重新同意")
def admin_save(req: Texts, request: Request) -> dict:
    return _view(terms.edit({"agreement": req.agreement, "privacy": req.privacy}, auth.label(request)))


@admin.delete("/terms", access=Access.admin("terms.edit"), summary="用户协议和隐私政策恢复成程序自带的文字：和现在不一样就是新的一版，所有账号下次使用前要重新同意")
def admin_reset(request: Request) -> dict:
    return _view(terms.reset(auth.label(request)))
