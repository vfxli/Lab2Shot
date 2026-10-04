"""The Terms of Service and Privacy Policy over HTTP (lab2shot/terms): reading them, agreeing to them, and the
administrator's copy, in every interface language.

    GET  /api/terms           anyone (the login and registration pages link to them): both texts as a person reads them,
                              in the request's language
    POST /api/auth/terms      the account asking agrees to the version its page showed (in the request's language); under /api/auth/, which the
                              guard answers before an account agreed (server/access.py), like logging in and out
    /api/admin/terms          the texts as written, every language, to read and edit (terms.edit): save a copy, or go back to the
                              program's own texts; either, when the texts change, is a new version everyone agrees to"""

from __future__ import annotations

from fastapi import Request

from .. import i18n, logs, terms
from ..messages import Msg
from . import auth
from .routes import Access, Body, Router

log = logs.get("auth")

router = Router(prefix="/api", tags=["login"])
admin = Router(prefix="/api/admin", tags=["admin"])


def _public(t: terms.Terms) -> dict:
    return {"version": t.version, "at": t.at, "documents": t.shown()}


@router.get("/terms", access=Access.open("Login and registration pages: the Terms of Service and Privacy Policy"), summary="The Terms of Service and Privacy Policy: the current version, when it took effect, and both texts in the request's language (the retention days and such they mention filled in from the current settings)")
def read() -> dict:
    return _public(terms.current())


class Agree(Body):
    version: int  # the version the page showed


@router.post("/auth/terms", access=Access.user("Agree to the Terms of Service and Privacy Policy (before first use after a change)"), summary="Agree to the Terms of Service and Privacy Policy, read in the request's language: send the version the page showed; if the texts changed meanwhile, read them again before agreeing")
def agree(req: Agree, request: Request) -> dict:
    u = auth.me(request)
    ip = auth.who(request)
    terms.agree(u.id, req.version, ip, i18n.current())
    logs.say(log, Msg("I-TERMS-AGREED", user=u.username, version=req.version, where=ip))
    return auth.state_of(auth.session(request))


def _view(t: terms.Terms) -> dict:
    """The texts as written, with what the placeholders say now and how many accounts agreed to this version."""
    return {"version": t.version, "at": t.at, "by": t.by, "edited": t.edited, "most": terms.MOST,
            "documents": [{"id": doc, "title": terms.title(doc), "texts": {lang: t.texts[lang][doc] for lang in t.texts}}
                          for doc in terms.DOCS],
            "fills": terms.fills(), **terms.agreed_count(t.version)}


@admin.get("/terms", access=Access.admin("terms.edit"), summary="The Terms of Service and Privacy Policy for editing: both texts as written in every language, the version, who changed it and when, the placeholders that may be written and what they say now, how many accounts agreed to this version")
def admin_read() -> dict:
    return _view(terms.current())


class Texts(Body):
    agreement: dict[str, str]  # lang -> text (every language of lab2shot/i18n LANGS)
    privacy: dict[str, str]


@admin.put("/terms", access=Access.admin("terms.edit"), summary="Change the Terms of Service and Privacy Policy: both texts in every language saved together (plain text, at most 20,000 characters each); if they differ from now it is a new version, which every account agrees to before its next use")
def admin_save(req: Texts, request: Request) -> dict:
    return _view(terms.edit({"agreement": req.agreement, "privacy": req.privacy}, auth.actor(request)))


@admin.delete("/terms", access=Access.admin("terms.edit"), summary="Restore the Terms of Service and Privacy Policy that come with the program: if they differ from now it is a new version, which every account agrees to before its next use")
def admin_reset(request: Request) -> dict:
    return _view(terms.reset(auth.actor(request)))
