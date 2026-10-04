"""The top bar's 「DCC 插件」: which DCC plugins this server offers, and each one's download.

A plugin is the folder clients/<dcc>/ of this repository, offered once it holds a `plugin.json` (its name, the file
that says how to install it, the folders the zip leaves out): a new DCC's folder is listed and downloadable as it is,
nothing here names it. PLANNED only says which DCCs the dialog lists as not offered yet while they have no plugin. The
download is packed from that folder at request time, with the shared client (lab2shot/client.py, every plugin's one way
to talk to the server) added as `lab2shot_client.py` next to the plugin's own code: the zip is always this server's
version of both, never a copy that could drift. Plugins are small, so the zip is made in memory and nothing is written
on the server.

The parameter interface's condition rules (engine/conditions.py, standard library only) come along as the plugin's
`conditions` says, beside the client, for the same reason. What every DCC shares (clients/common/: the plugin
framework, the same for every DCC) comes along where the plugin's `common` says (a folder inside the zip). When this server itself serves HTTPS with the authority it made
(server/tls.py: setting server.https), that authority's certificate comes along too, where the plugin's
`certificate` says, so the plugin trusts it on top of the system's; a server behind a reverse proxy with a bought
certificate (running HTTP itself) adds none. The certificate is read from work/tls/ at the moment of the download:
it never lives in the repository.

Who may: the setting plugins.download (off by default while the plugins are being tested) opens both routes to every
signed-in user; off, they are the administrators' and deputy administrators' only, for internal testing (one subject,
server/available.py ACTIONS["plugins.download"], decides it for these routes and for the top bar alike), and the list
says `open: false` so the dialog marks them as not open to users. A plugin already installed connects whatever it says:
only the download is held back.
"""

from __future__ import annotations

import io
import json
import zipfile

from fastapi import Request, Response

from .. import __version__
from ..config import ROOT
from ..errors import Forbidden, NotFound
from ..messages import Msg
from ..roles import SessionFacts
from . import auth, available
from .routes import Access, Router

router = Router(prefix="/api", tags=["Settings"])

CLIENTS = ROOT / "clients"
CLIENT_PY = ROOT / "lab2shot" / "client.py"
COMMON = CLIENTS / "common"
# the parameter interface's condition rules (Hide When / Disable When), standard library only: a plugin builds its
# panel by the very rules the server and the page judge them by (engine/conditions.py), never a copy of its own
CONDITIONS_PY = ROOT / "lab2shot" / "engine" / "conditions.py"
# DCCs the dialog lists as 「尚未提供」 while their folder has no plugin.json (a display list only: one that has a plugin is
# listed by its plugin.json, whether it is here or not)
PLANNED = (("houdini", "Houdini"), ("nuke", "Nuke"))
SKIPPED = {"__pycache__", ".pytest_cache", ".DS_Store"}


def _inside(path: str, default: str) -> str:
    """A relative path inside the zip's one top folder, from a plugin's own plugin.json (never one that climbs out)."""
    target = str(path or default).replace("\\", "/").strip("/")
    return default if not target or ".." in target.split("/") else target


def _authority() -> bytes | None:
    """This server's own certificate authority, when the server itself serves HTTPS with it (else None)."""
    from ..config import settings
    from . import tls

    try:
        if not settings()["server.https"]:  # as this server runs now (a proxy in front terminating HTTPS: off)
            return None
        f = tls.authority_file()
        return f.read_bytes() if f is not None else None
    except OSError:
        return None


def _may(request: Request) -> None:
    """Whether this session may see and download the plugins now (see the module docstring); refused otherwise."""
    if not available.ACTIONS["plugins.download"].holds(SessionFacts.of(auth.session(request))):
        raise Forbidden(Msg("E-PLUGIN-CLOSED"))


def _open() -> bool:
    """The setting plugins.download: the plugins are open to every user (False: to the administrators only)."""
    from ..config import settings

    return bool(settings()["plugins.download"])


def _manifest(dcc: str) -> dict | None:
    """The plugin's own description (clients/<dcc>/plugin.json), or None while this DCC has no plugin."""
    path = CLIENTS / dcc / "plugin.json"
    if not path.is_file():
        return None
    try:
        got = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return got if isinstance(got, dict) else None


@router.get("/plugins", access=Access.user("Top bar DCC Plugins: which applications' plugins are provided"),
            summary="DCC plugins: whether this server provides the plugin for Maya, Houdini and Nuke, with the download address of "
                    "each one provided")
def plugins(request: Request) -> dict:
    _may(request)
    offered = {f.name: m for f in sorted(CLIENTS.iterdir()) if f.is_dir() and (m := _manifest(f.name)) is not None}
    out = [{"id": dcc, "label": str(m.get("name") or dcc), "available": True, "url": f"/api/plugins/{dcc}.zip"}
           for dcc, m in offered.items()]
    out += [{"id": dcc, "label": label, "available": False, "url": ""} for dcc, label in PLANNED if dcc not in offered]
    return {"plugins": out, "open": _open()}


@router.get("/plugins/{dcc}.zip", access=Access.user("Top bar DCC Plugins: download a plugin"),
            summary="Download a DCC plugin: the plugin folder and the shared client in one zip, with install instructions inside")
def download(dcc: str, request: Request) -> Response:
    _may(request)
    m = _manifest(dcc) if dcc.isidentifier() and dcc != "common" else None  # a folder name, never a path
    if m is None:
        raise NotFound(Msg("E-PLUGIN-NONE", dcc=dict(PLANNED).get(dcc, dcc[:40])))
    base = CLIENTS / dcc
    skipped = SKIPPED | set(m.get("exclude") or ())
    top = f"lab2shot_{dcc}_{__version__}"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(base)
            if path.is_file() and not (set(rel.parts) & skipped) and rel.name != "plugin.json":
                z.write(path, f"{top}/{rel.as_posix()}")
        # where the plugin imports the shared client from, inside its folder: a relative path that cannot climb out of
        # the zip's one top folder once unpacked
        z.write(CLIENT_PY, f"{top}/{_inside(m.get('client'), 'lab2shot_client.py')}")
        if m.get("conditions"):
            z.write(CONDITIONS_PY, f"{top}/{_inside(m['conditions'], 'lab2shot_conditions.py')}")
        if m.get("common"):  # the framework every DCC's plugin shares, as it is in clients/common/
            where = _inside(m["common"], "common")
            for path in sorted(COMMON.rglob("*")):
                rel = path.relative_to(COMMON)
                if path.is_file() and not (set(rel.parts) & skipped):
                    z.write(path, f"{top}/{where}/{rel.as_posix()}")
        if m.get("certificate") and (ca := _authority()) is not None:
            z.writestr(f"{top}/{_inside(m['certificate'], 'lab2shot_ca.pem')}", ca)
        z.writestr(f"{top}/VERSION", __version__ + "\n")
    return Response(buffer.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{top}.zip"'})
