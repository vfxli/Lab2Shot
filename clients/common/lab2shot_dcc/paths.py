"""Where the plugin keeps its own things, and the two modules that ship beside the package.

- The user's plugin folder: %USERPROFILE%/lab2shot on Windows, ~/lab2shot elsewhere (logs, the plugin's settings,
  temporary exports, and the login token: the transport's token file is pointed here, so a plugin writes nowhere else
  outside the DCC project).
- lab2shot_client.py (the transport) and lab2shot_conditions.py (the parameter interface's condition rules) sit next
  to this package in the plugin download (server/plugins.py puts them there); while developing in the repository
  they are lab2shot/client.py and lab2shot/engine/conditions.py.
- A certificate authority shipped with the download (lab2shot_ca.pem next to this package) exists only when the
  server that made the download serves HTTPS with its own authority; it is trusted on top of the system's.
"""

from __future__ import annotations

import importlib.util
import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
BESIDE = os.path.dirname(PACKAGE_DIR)  # the folder holding this package (the plugin's scripts folder)
CA_NAME = "lab2shot_ca.pem"


def user_dir() -> str:
    """The plugin's own folder for this user (made when missing)."""
    base = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    folder = os.path.join(base, "lab2shot")
    os.makedirs(folder, exist_ok=True)
    return folder


def bundled_ca() -> str | None:
    """The certificate authority that came with the download, or None (a server with a public certificate)."""
    for folder in (BESIDE, PACKAGE_DIR):
        path = os.path.join(folder, CA_NAME)
        if os.path.isfile(path):
            return path
    return None


def _load(name: str, candidates: list[str]):
    if name in sys.modules:
        return sys.modules[name]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
    raise ImportError(name)


def _repo() -> str:
    # clients/common/lab2shot_dcc -> the repository root, when running from the repository
    return os.path.dirname(os.path.dirname(BESIDE))


def client_module():
    """lab2shot_client (the transport). A plugin keeps the transport's token and device id in the plugin's own user
    folder (user_dir), the one place a plugin writes outside the DCC project."""
    module = _load("lab2shot_client", [os.path.join(BESIDE, "lab2shot_client.py"),
                                       os.path.join(_repo(), "lab2shot", "client.py")])
    if not getattr(module, "_lab2shot_dcc_paths", False):
        module.tokens_file = lambda: os.path.join(user_dir(), "tokens.json")
        module.device_id_file = lambda: os.path.join(user_dir(), "device.json")
        module._lab2shot_dcc_paths = True
    return module


def conditions_module():
    """lab2shot_conditions (Hide When / Disable When, the server's own rules). Its sentences are said in the plugin's
    language by the transport's words (its `say` and `joined` hooks set to `text`: the keys are the page's
    ui.conditions.*, generated into the transport by tools/messages_client.py), not by the lab2shot package, which a
    DCC does not have, and whose language would not be the plugin's if it had."""
    module = _load("lab2shot_conditions", [os.path.join(BESIDE, "lab2shot_conditions.py"),
                                           os.path.join(_repo(), "lab2shot", "engine", "conditions.py")])
    if not getattr(module, "_lab2shot_dcc_paths", False):
        module.say = text
        module.joined = lambda items: text("list.sep").join(items)
        module._lab2shot_dcc_paths = True
    return module


def pick(word) -> str:
    """A template's own text in the plugin's language: a string as it is (a user's, one language), or {zh, en} (a
    built-in card's) — the language now, else the other one written (lab2shot.i18n.pick's rule, the web's too)."""
    if not isinstance(word, dict):
        return "" if word is None else str(word)
    lang = client_module().get_lang()
    return str(word.get(lang) or next((v for v in word.values() if v), ""))


def text(key: str, **params) -> str:
    """One of the plugin's words (dcc.*, lab2shot/i18n/<lang>/dcc.toml) or the client's messages, in the plugin's
    language (connection.apply_language), from the transport's generated table (tools/messages_client.py)."""
    return client_module().text(key, **params)
