"""The language of a request: what everything said in answer to it is said in (lab2shot/i18n).

In order, the first that names one of i18n.LANGS:

    1. the X-Lab2Shot-Lang header      a client that says its own (a DCC plugin, the command line)
    2. the account's choice            users.lang, set from the account menu (PUT /api/me/lang); "" none yet
    3. the `lang` cookie               a browser's choice before a login, or without one
    4. Accept-Language                 the browser's own languages
    5. i18n.DEFAULT                    zh

The guard (server/access.py Guard) sets it on the request's context before anything is said, so a refusal, a route's
answer, an error and a stream all speak it; every answer says which (Content-Language) and that it depends on these
(Vary). An answer kept and served again (routes.py's keyed answers) is kept per language."""

from __future__ import annotations


from fastapi import Response
from pydantic import Field
from starlette.requests import Request

from .. import i18n
from .routes import Access, Body, Router

VARY = f"{i18n.HEADER}, Cookie, Accept-Language"
COOKIE_DAYS = 400  # what browsers keep a cookie for at most
SCOPE_KEY = "lab2shot_lang"


def before_login(request: Request) -> str:
    """The language a request asks for without its account: header, cookie, Accept-Language, zh."""
    return (i18n.normal(request.headers.get(i18n.HEADER)) or i18n.normal(request.cookies.get(i18n.COOKIE))
            or i18n.from_accept_language(request.headers.get("accept-language")) or i18n.DEFAULT)


def of(request: Request, session) -> str:
    """The request's language, its account's choice in its place (see the module doc)."""
    said = i18n.normal(request.headers.get(i18n.HEADER))
    if said:
        return said
    chosen = i18n.normal(getattr(session.user, "lang", "")) if session is not None else None
    return chosen or before_login(request)


def set_cookie(response, lang: str, secure: bool) -> None:
    """Remember a browser's language (readable by the page: the login page reads it before it asks anything)."""
    response.set_cookie(i18n.COOKIE, lang, max_age=COOKIE_DAYS * 86400, path="/", samesite="lax", secure=secure,
                        httponly=False)


# ------------------------------------------------------------------ the account's choice

router = Router(prefix="/api")


class LangRequest(Body):
    lang: str = Field("", max_length=16)  # one of i18n.LANGS; "": follow the browser again


@router.put("/me/lang", access=Access.user("The account's interface language"),
            summary="Choose the language everything is said in for this account (zh, en; empty: the browser's); "
                    "this browser remembers it for its login page too (the lang cookie)")
def put_lang(req: LangRequest, request: Request, response: Response) -> dict:
    from .. import accounts
    from . import auth

    accounts.set_lang(auth.me(request).id, req.lang)
    chosen = i18n.normal(req.lang) or before_login(request)
    if i18n.normal(req.lang):
        set_cookie(response, chosen, secure=auth.https(request))
    else:
        response.delete_cookie(i18n.COOKIE, path="/")
    i18n.set_current(chosen)  # this answer, and whatever it carries, in the new one
    return {"lang": chosen}


# ------------------------------------------------------------------ saying messages again

SAID_MOST = 500  # messages one request may ask for (the page's log keeps 500 entries)
class SaidRequest(Body):
    # each {code, params}: a message as an answer or an event carried it (Msg.json's params, or its args)
    messages: list[dict] = Field(default_factory=list, max_length=SAID_MOST)


@router.post("/said", access=Access.user("Messages said again in your language"),
             summary="The words of messages the server said before (their codes and parameters), in the language of this "
                     "request: what the page's log shows after the language is switched; one that cannot be said again "
                     "(a code no catalogue has, parameters not its template's) is null")
def said_again(req: SaidRequest) -> dict:
    from ..messages import again

    out: list[str | None] = []
    for m in req.messages:
        # a node pointed at, a parameter's name...: kept by its key ({"said": ...}, messages.wire), said again here
        params = m.get("params") if isinstance(m, dict) else None
        msg = again({"code": m.get("code"), "params": params}) if isinstance(m, dict) else None
        out.append(msg.text if msg is not None else None)
    return {"texts": out}
