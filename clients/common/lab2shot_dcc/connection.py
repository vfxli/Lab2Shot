"""The connection to a Lab2Shot server, on top of the transport (lab2shot_client).

- The address is what the user typed (nothing is filled in for them), kept in the plugin's settings for next time.
  Every URL is that address plus an API path: a server behind a reverse proxy, under a path prefix, on any port, works
  as typed; nothing the server answers is used as an address.
- HTTPS: the system's certificates, and besides them the authority that came with the download (only a server that
  serves HTTPS with its own authority ships one). Never "trust anything".
- Time limits: connecting 10 s (CONNECT_S, checked before each step that talks to the server), an answer 60 s
  (READ_S; the tool list 180 s, TOOLS_READ_S: a server just started may still be working it out); files go up and
  down in chunks, each chunk under the read limit. A dropped line is retried by the transport a bounded number of
  times, waiting longer each time.
- Cancelling: every session has the transport's own `cancel_event`. Once it is set, the next request, chunk read or
  written, or retry wait raises the transport's Cancelled — an upload or download in the middle stops there, and no
  retry takes it for a dropped line.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import urllib.parse

from . import paths

CONNECT_S = 10.0
READ_S = 60.0
TOOLS_READ_S = 180.0  # the tool list: a server just started may still be working out the tools' signatures
SETTINGS = "settings.json"


# the transport's own: once a session's cancel_event is set, the request or file chunk in progress raises it
Cancelled = paths.client_module().Cancelled


def settings_file() -> str:
    return os.path.join(paths.user_dir(), SETTINGS)


def read_settings() -> dict:
    try:
        with open(settings_file(), encoding="utf-8") as f:
            got = json.load(f)
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def write_settings(**changes) -> None:
    data = {**read_settings(), **changes}
    tmp = settings_file() + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, settings_file())


LANG_AUTO = "auto"  # the plugin's language setting when its user chose none: the DCC's own (Host.ui_language)


def language_setting() -> str:
    """What the user chose in the plugin's settings: "zh", "en", or LANG_AUTO."""
    got = read_settings().get("lang", LANG_AUTO)
    return got if got in (*paths.client_module().LANGS, LANG_AUTO) else LANG_AUTO


def language(host) -> str:
    """The plugin's language: the user's choice, else the DCC's (English only when its interface names another
    language; a DCC that says nothing gets Chinese, the product's default, as a terminal that says nothing does:
    lab2shot_client.language_of_environment)."""
    chosen = language_setting()
    if chosen != LANG_AUTO:
        return chosen
    try:
        said = str(host.ui_language() or "")
    except Exception:  # noqa: BLE001 - a host that cannot say: the default
        said = ""
    return "en" if said and not said.lower().startswith("zh") else "zh"


def apply_language(host) -> str:
    """Make the plugin's language the transport's (its own words, and what the server says: X-Lab2Shot-Lang)."""
    lang = language(host)
    paths.client_module().set_lang(lang)
    return lang


def normalise(address: str) -> str:
    """The address as typed, made a base URL: https:// when no scheme was given, no trailing slash."""
    text = str(address or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "https://" + text
    parts = urllib.parse.urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(paths.text("dcc.connection.bad_address", address=address))
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def reachable(server: str, timeout: float = CONNECT_S) -> str:
    """"" when a connection to the server opens within `timeout`; else why not."""
    parts = urllib.parse.urlsplit(server)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        with socket.create_connection((parts.hostname, port), timeout=timeout):
            return ""
    except OSError as exc:
        return paths.text("dcc.connection.unreachable", server=server, error=exc.strerror or exc)


TRANSFER_TRIES = 3  # the transport's own resumes of one file, kept short under the job's retries (jobs.py)


def session(server: str, app: str, **kw):
    """A transport for `server` that can be cancelled (`.cancel_event`, the transport's public hook), with the
    plugin's time limits and the shipped certificate authority (if any)."""
    import threading

    client = paths.client_module()
    return client.Lab2Shot(server, app=app, timeout=READ_S, cafile=paths.bundled_ca(), cancel_event=threading.Event(),
                           transfer_tries=TRANSFER_TRIES, **kw)


class Connection:
    """The plugin's one connection: the server the user chose and whether this machine is logged in there."""

    def __init__(self, app: str):
        self.app = app
        self.server = read_settings().get("server", "")
        self.user: dict | None = None

    def tokens(self) -> dict:
        client = paths.client_module()
        try:
            with open(client.tokens_file(), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def has_token(self) -> bool:
        return bool(self.server and self.tokens().get(self.server))

    def session(self):
        """A fresh, cancellable transport (one per job: cancelling one job never stops another)."""
        if not self.server:
            raise RuntimeError(paths.text("dcc.connection.not_connected"))
        return session(self.server, self.app)

    def login(self, address: str, username: str, password: str) -> dict:
        """Log in (any thread but the main one: it talks to the server); keeps the address for next time."""
        server = normalise(address)
        why = reachable(server)
        if why:
            raise RuntimeError(why)
        lab = session(server, self.app)
        user = lab.login(username, password)
        self.server, self.user = server, user
        write_settings(server=server)
        return user

    def logout(self) -> None:
        if self.server:
            try:
                session(self.server, self.app).logout()
            finally:
                self.user = None
