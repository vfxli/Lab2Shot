"""`lab2shot check routes`, `lab2shot check layers`, `lab2shot check tests`: the architecture's rules, held as code.

    routes   every route of the server is declared (server/routes.py DECLARED); a user route whose path or query names
             one account's data (OWNED_PARAMS: a packet, a job, a task, a saved graph), or whose request body does
             (its declared `body_ids`; a body field named like account data, OWNED_BODY_FIELDS, must be declared there,
             and a declared one must be a field of the body), declares whose it is (`owned=`, server/owners.py); an
             admin route naming one is owned too, or needs the right to other people's data (data.others), unless what
             it names is no account's (ADMIN_NOT_ACCOUNT_DATA)
    layers   every import in lab2shot/ (at the top of a module and inside functions alike) goes down the layer table
             (LAYERS): a module imports from its own layer or a lower one, never a higher one; an extension
             (adapters/<name>/, its upstream repo/ left out) imports from lab2shot only through lab2shot.sdk
    digests  a key or a content digest is made by lab2shot/io/digest.py, the one implementation; hashlib is used
             directly only where it is not a key (DIGEST_ALLOWED: passwords, tokens, signatures, streaming an upload's
             bytes, an ETag, the standalone client)
    tests    the unit tests: tests/ (the core's, standard-library unittest) and the page's (webui/src/**/*.test.ts, by
             node); the DCC clients' pure-Python ones run in `check dcc` (cli/check_dcc.py UNIT_TESTS)
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from .. import i18n

# ------------------------------------------------------------------ routes

# route parameters that name one account's data (a packet, a job, a task, a saved graph)
OWNED_PARAMS = frozenset({"fp", "job_id", "task_id", "gid"})
# request body fields named like one account's data: those names, and their lists (fps, job_ids / jobs, ...). A route
# reading such a field declares it (Access.body_ids) and the owner that checks or filters it; a field of account data
# under another name (a template's `id`, a node's `inputs`) is declared the same way, by the route itself
OWNED_BODY_FIELDS = OWNED_PARAMS | {p + "s" for p in OWNED_PARAMS} | {p[:-3] + "s" for p in OWNED_PARAMS if p.endswith("_id")}
# admin routes whose `job_id` names no account's data: an installation (lab2shot/installer), not a user's job
ADMIN_NOT_ACCOUNT_DATA = ("/api/admin/installs/",)
OTHERS_RIGHT = "data.others"


def _api_routes(routes, prefix: str = ""):
    """Every FastAPI route of the app, with the prefix it is included under (routers included lazily too)."""
    from fastapi.routing import APIRoute

    for r in routes:
        if hasattr(r, "original_router"):
            yield from _api_routes(r.original_router.routes, prefix + r.include_context.prefix)
        elif isinstance(r, APIRoute):
            yield prefix + r.path, r


def _body_fields(r) -> set[str]:
    """The field names of a route's request body: each body parameter's model fields (a body that is no model: its name)."""
    out = set()
    for b in r.dependant.body_params:
        fields = getattr(b.field_info.annotation, "model_fields", None)
        out |= set(fields) if fields else {b.name}
    return out


def route_problems(app, declared: dict) -> tuple[int, list[str]]:
    """(routes looked at, what is wrong) for the app's routes against their declarations."""
    from ..server.routes import declared_as

    n, out = 0, []
    for path, r in _api_routes(app.routes):
        names = {p.name for p in (*r.dependant.path_params, *r.dependant.query_params)}
        body = _body_fields(r)
        for method in sorted(r.methods):
            n += 1
            key = f"{declared_as(method)} {path}"
            access = declared.get(key)
            if access is None:
                out.append(i18n.t("cli.check.routes.undeclared", route=key))
                continue
            if unsaid := sorted((body & OWNED_BODY_FIELDS) - set(access.body_ids)):
                out.append(i18n.t("cli.check.routes.body_undeclared", route=key, fields=", ".join(unsaid)))
            if unknown := sorted(set(access.body_ids) - body):
                out.append(i18n.t("cli.check.routes.body_unknown", route=key, fields=", ".join(unknown)))
            named = sorted(names & OWNED_PARAMS) + sorted(set(access.body_ids) & body)
            if not named or access.owned is not None:
                continue
            if access.level == "user":
                out.append(i18n.t("cli.check.routes.unowned", route=key, params=", ".join(named)))
            elif access.level == "admin" and access.needs != OTHERS_RIGHT and not path.startswith(ADMIN_NOT_ACCOUNT_DATA):
                out.append(i18n.t("cli.check.routes.admin_unowned", route=key, params=", ".join(named),
                                  right=OTHERS_RIGHT))
    return n, out


def check_routes(r) -> None:
    from ..server.app import app
    from ..server.routes import DECLARED

    n, problems = route_problems(app, DECLARED)
    for p in problems:
        r.bad(p)
    if not problems:
        r.ok(i18n.t("cli.check.routes.ok", routes=n))


# ------------------------------------------------------------------ layers

# The layers of lab2shot, lowest first: a unit (a subpackage or a module right under lab2shot/) imports only from its own
# layer or a lower one. Units of one layer are one concern and may use each other.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # what everything stands on: words, messages, settings, files, logs
    ("base", ("__init__", "errors", "messages", "i18n", "config", "io", "text", "process", "logs", "workdir",
              "periods", "availability", "releases", "progress", "categories")),
    ("store", ("database", "traffic")),  # the database and what is counted straight into it
    ("data", ("data", "serving", "recent", "ops")),  # data types, packets, the cache
    # the node system: node types, formats, the extension registry and its contract (sdk, adapters loader)
    ("nodes", ("nodes", "formats", "sdk", "extensions", "adapters")),
    ("engine", ("engine",)),  # graphs, templates, cooking
    ("accounts", ("accounts", "roles")),  # who: accounts, sessions, rights
    ("work", ("farm", "transfer", "view", "installer")),  # running it: the queue, files in and out, the view, installs
    # the site's own things that belong to accounts
    ("site", ("site", "terms", "analysis")),
    ("front", ("server", "cli", "client")),  # the ways in: HTTP and the command line
)
# files that sit in another layer than their unit's (most specific first): the catalogue lint is a check of the command line
LAYER_FILES = {"lab2shot/i18n/lint.py": "front"}
SDK = "lab2shot.sdk"


def layer_of(rel: str, unit: str) -> int | None:
    names = [name for name, _ in LAYERS]
    if rel in LAYER_FILES:
        return names.index(LAYER_FILES[rel])
    for i, (_, units) in enumerate(LAYERS):
        if unit in units:
            return i
    return None


def _module_of(rel: Path) -> tuple[str, str]:
    """(dotted module, its package) of a file under the repository root."""
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        mod = ".".join(parts[:-1])
        return mod, mod
    mod = ".".join(parts)
    return mod, mod.rsplit(".", 1)[0]


def imports_of(path: Path, rel: Path) -> list[tuple[int, str]]:
    """(line, dotted module) of every lab2shot import in a file, top level and inside functions alike; `from . import x`
    as the module it names when x is a module of the package."""
    _, package = _module_of(rel)
    out = []
    for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom):
            if n.level:
                base = package.split(".")
                base = base[:len(base) - (n.level - 1)]
                mod = ".".join(base + ([n.module] if n.module else []))
            else:
                mod = n.module or ""
            if mod == "lab2shot" or (n.level and not n.module):
                out += [(n.lineno, f"{mod}.{a.name}") for a in n.names]
            elif _ours(mod):
                out.append((n.lineno, mod))
        elif isinstance(n, ast.Import):
            out += [(n.lineno, a.name) for a in n.names if _ours(a.name)]
    return out


def _ours(module: str) -> bool:
    return module == "lab2shot" or module.startswith("lab2shot.")


def _unit(module: str) -> str:
    parts = module.split(".")
    if parts[0] != "lab2shot":
        return ""
    return parts[1] if len(parts) > 1 and parts[1] != "__version__" else "__init__"


def layer_problems(root: Path) -> tuple[int, list[str]]:
    """(imports looked at, what goes up the layer table or past the sdk)."""
    names = [name for name, _ in LAYERS]
    n, out = 0, []
    for path in sorted((root / "lab2shot").rglob("*.py")):
        rel = path.relative_to(root)
        if "__pycache__" in rel.parts:
            continue
        mod, _ = _module_of(rel)
        unit = _unit(mod)
        mine = layer_of(rel.as_posix(), unit)
        if mine is None:
            out.append(i18n.t("cli.check.layers.unplaced", unit=unit, where=rel.as_posix()))
            continue
        for line, target in imports_of(path, rel):
            n += 1
            other = _unit(target)
            if other == unit:
                continue
            theirs = layer_of("", other)
            if theirs is None:
                out.append(i18n.t("cli.check.layers.unplaced", unit=other, where=f"{rel.as_posix()}:{line}"))
            elif theirs > mine:
                out.append(i18n.t("cli.check.layers.upward", where=f"{rel.as_posix()}:{line}", module=target,
                                  low=names[mine], high=names[theirs]))
    for path in sorted((root / "adapters").glob("*/**/*.py")):
        rel = path.relative_to(root)
        if "repo" in rel.parts or "__pycache__" in rel.parts:
            continue
        for line, target in imports_of(path, rel):
            if target.startswith("lab2shot.") and target != SDK and not target.startswith(SDK + "."):
                n += 1
                out.append(i18n.t("cli.check.layers.past_sdk", where=f"{rel.as_posix()}:{line}", module=target))
    return n, out


def check_layers(r, root: Path) -> None:
    n, problems = layer_problems(root)
    for p in problems:
        r.bad(p)
    if not problems:
        r.ok(i18n.t("cli.check.layers.ok", imports=n, layers=len(LAYERS)))


# ------------------------------------------------------------------ digests

DIGEST_ALLOWED = {
    "lab2shot/io/digest.py": "the one implementation",
    "lab2shot/accounts.py": "passwords (scrypt) and session tokens",
    "lab2shot/site/registration.py": "an invitation code kept as its hash (a secret)",
    "lab2shot/server/auth.py": "signatures (HMAC) and the proof of work",
    "lab2shot/transfer/uploads.py": "an upload's bytes hashed as they arrive, part by part",
    "lab2shot/server/wire.py": "an answer's ETag (BLAKE2, made on every answer)",
    "lab2shot/client.py": "the standalone client imports nothing of lab2shot",
}


def digest_problems(root: Path) -> tuple[int, list[str]]:
    n, out = 0, []
    for path in sorted((root / "lab2shot").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in rel:
            continue
        n += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        uses = sorted({node.lineno for node in ast.walk(tree)
                       if (isinstance(node, ast.Import) and any(a.name == "hashlib" for a in node.names))
                       or (isinstance(node, ast.ImportFrom) and node.module == "hashlib")})
        if uses and rel not in DIGEST_ALLOWED:
            out.append(i18n.t("cli.check.digests.direct", where=f"{rel}:{uses[0]}"))
    return n, out


def check_digests(r, root: Path) -> None:
    n, problems = digest_problems(root)
    for p in problems:
        r.bad(p)
    if not problems:
        r.ok(i18n.t("cli.check.digests.ok", files=n, allowed=len(DIGEST_ALLOWED)))


# ------------------------------------------------------------------ tests

def test_commands(root: Path) -> list[tuple[str, list[str], Path]]:
    """(what, command, where) of the core's and the page's unit tests (no DCC, no server, no GPU)."""
    out = [("tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-q"], root)]
    page = sorted((root / "webui" / "src").rglob("*.test.ts"))
    if page:
        out.append(("webui", ["node", "--test", *[p.relative_to(root / "webui").as_posix() for p in page]], root / "webui"))
    return out


def check_tests(r, root: Path) -> None:
    for what, command, where in test_commands(root):
        done = subprocess.run(command, cwd=where, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if done.returncode == 0:
            r.ok(i18n.t("cli.check.tests.passed", suite=what))
        else:
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
            r.bad(i18n.t("cli.check.tests.failed", suite=what, code=done.returncode, tail=tail))
