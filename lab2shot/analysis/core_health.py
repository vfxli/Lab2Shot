"""Core health (核心健康): measures whether the core stays small as projects are added, and reports what may be the
same thing implemented twice.

- Concept count: data types, core nodes, families, parameter widget kinds, contract checks, usage-check kinds and
  handle kinds. Counted from the source text, so a historical commit (read through git for the trend) is counted
  exactly like the working tree.
- Sameness, as refactoring candidates (never merged automatically; each candidate carries its evidence): types stored
  and checked identically that differ only by label, core nodes whose ports and parameters largely overlap, duplicated
  code (a token-based detector over lab2shot/, adapters/ and worker_sdk/), and functions repeated across adapters that
  belong in the worker SDK.
- Special cases: names of third-party projects and their formats in the core's code (target: 0).
- Rules: the concept counts must equal the checked-in baseline (core_baseline.toml, changed deliberately with a
  reason); duplicated code above the threshold is not allowed unless core_allow.toml lists it with a reason.
"""

from __future__ import annotations

import ast
import io
import json
import keyword
import re
import subprocess
import time
import tokenize
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT

BASELINE_FILE = Path(__file__).with_name("core_baseline.toml")
ALLOW_FILE = Path(__file__).with_name("core_allow.toml")

# Concepts in report order: id -> label.
CONCEPTS = {
    "types": "数据类型",
    "core_nodes": "核心节点",
    "families": "节点家族",
    "widgets": "参数控件种类",
    "contracts": "契约检查",
    "expects": "用法检查种类",
    "handles": "手柄种类",
    "scopes": "作用域种类",
    "special_cases": "核心里的特例",
}


@dataclass(frozen=True)
class CoreSettings:
    clone_tokens: int = 100  # minimum length (tokens, names normalised) of a clone
    helper_tokens: int = 40  # minimum length of a function repeated across adapters to be reported as an SDK candidate
    node_overlap: float = 0.6  # Jaccard overlap of ports and parameters above which two core nodes are reported
    trend_points: int = 24  # number of days of history in the trend

    @staticmethod
    def load(path: Path | None = None) -> CoreSettings:
        """Load the [core] table of a settings file. Returns the defaults when `path` is None; no settings file ships
        with the analysis."""
        if path is None:
            return CoreSettings()
        c = tomllib.loads(path.read_text(encoding="utf-8")).get("core", {})
        d = CoreSettings()
        return CoreSettings(int(c.get("clone_tokens", d.clone_tokens)), int(c.get("helper_tokens", d.helper_tokens)),
                            float(c.get("node_overlap", d.node_overlap)),
                            int(c.get("trend_points", d.trend_points)))


# ------------------------------------------------------------------ the source, now or at a commit


def _wanted(path: str) -> bool:
    return (path.endswith(".py") and path.startswith(("lab2shot/", "adapters/", "worker_sdk/")) and "/.venv/" not in path) or \
           (path.startswith("webui/src/") and path.endswith((".ts", ".tsx", ".css")))


def working_tree() -> dict[str, str]:
    """Map each counted file's relative POSIX path to its text in the current checkout."""
    out = {}
    for base in ("lab2shot", "adapters", "worker_sdk", "webui/src"):
        for p in (ROOT / base).rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and _wanted(rel) and "__pycache__" not in rel and "node_modules" not in rel:
                out[rel] = p.read_text(encoding="utf-8", errors="replace")
    return out


def at_commit(rev: str) -> dict[str, str]:
    """Map each counted file's path to its text at commit `rev`, read with a single `git cat-file --batch` process."""
    listing = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", rev], capture_output=True, text=True, check=True).stdout
    blobs = []
    for line in listing.splitlines():
        meta, _, path = line.partition("\t")
        if meta.split()[1] == "blob" and _wanted(path):
            blobs.append((path, meta.split()[2]))
    proc = subprocess.run(["git", "-C", str(ROOT), "cat-file", "--batch"], input="".join(f"{sha}\n" for _, sha in blobs).encode(),
                          capture_output=True, check=True)
    data, pos, out = proc.stdout, 0, {}
    for path, _ in blobs:
        header_end = data.index(b"\n", pos)
        size = int(data[pos:header_end].split()[2])
        out[path] = data[header_end + 1:header_end + 1 + size].decode("utf-8", errors="replace")
        pos = header_end + 1 + size + 1
    return out


# ------------------------------------------------------------------ concepts and lines


def _projects(files: dict[str, str]) -> set[str]:
    return {p.split("/")[1] for p, text in files.items() if re.fullmatch(r"adapters/[^/]+/extension\.py", p) and "EXTENSION" in text}


def concepts(files: dict[str, str], special: int | None = None) -> dict[str, int]:
    """Count the core's concepts from source text, so that historical commits are counted the same way."""
    def text(path: str) -> str:
        return files.get(path, "")

    core_nodes = {m for p, t in files.items() if p.startswith("lab2shot/") and p.endswith(".py")  # includes the core's format modules
                  for m in re.findall(r'^\s+id = "(core\.\w+)"', t, re.M)}
    widgets = {m for p, t in files.items() if p.endswith(".py") and p.startswith(("lab2shot/", "adapters/"))
               for m in re.findall(r'(?:widget\s*=\s*|"widget":\s*)"(\w+)"', t)}
    handles = re.search(r"HANDLE_KINDS[^=]*=\s*\{(.*?)\n\}", text("lab2shot/nodes/handles.py"), re.S)
    # scope kinds (engine/scopes.py SCOPE_KINDS); 0 for older commits without that file
    scope_kinds = re.search(r"SCOPE_KINDS\s*=\s*\(([^)]*)\)", text("lab2shot/engine/scopes.py"))
    # Families: in older commits, every NodeDef subclass in nodes/results.py. That file was later split into
    # lab2shot/nodes/families/*.py with the same set of families, so commits that still have results.py are counted
    # there and newer ones sum the same pattern over the family files.
    if "lab2shot/nodes/results.py" in files:
        families = len(re.findall(r"^class \w+\(NodeDef\)", text("lab2shot/nodes/results.py"), re.M))
    else:
        # Families subclass WorkerNode (defined in families/base.py, which is not itself a family) or, if not yet
        # migrated, NodeDef.
        family_files = [p for p in files if re.fullmatch(r"lab2shot/nodes/families/(?!__init__\.py|base\.py)\w+\.py", p)]
        # A family may also mix in a small declaration class (e.g. Matting in families/matte.py); only the family is counted.
        families = sum(len(re.findall(r"^class \w+\([^)]*\b(?:NodeDef|WorkerNode)\b[^)]*\)", text(p), re.M)) for p in family_files)
    types_path = "lab2shot/data/types.py" if "lab2shot/data/types.py" in files else "lab2shot/nodes/types.py"
    contracts_path = "lab2shot/data/contracts.py" if "lab2shot/data/contracts.py" in files else "lab2shot/nodes/contracts.py"
    out = {
        "types": len(set(re.findall(r'DataType\("([\w.]+)"', text(types_path)))),
        "core_nodes": len(core_nodes),
        "families": families,
        "widgets": len(widgets),
        "contracts": len(re.findall(r"^def _check_\w+", text(contracts_path), re.M)),
        "expects": len(re.findall(r"^class \w+\(Expect\)", text("lab2shot/nodes/expects.py"), re.M)),
        "handles": len(re.findall(r'^\s+"(\w+)":', handles.group(1), re.M)) if handles else 0,
        "scopes": len(re.findall(r'"(\w+)"', scope_kinds.group(1))) if scope_kinds else 0,
    }
    if special is not None:
        out["special_cases"] = special
    return out


# ------------------------------------------------------------------ special cases in the core


# Files exempt from the special-case check: the manual-downloads registry, which names user-supplied assets (body
# models, the SDK a format module builds on) by design.
REGISTRIES = {"lab2shot/extensions/manual.py"}


def extension_formats(files: dict[str, str]) -> set[str]:
    """Return the file suffixes handled only by extensions, as declared by their format modules (SUFFIXES = (...))."""
    return {s.lower() for p, t in files.items() if p.startswith("adapters/") and p.endswith("/nodes.py")
            for group in re.findall(r"^SUFFIXES\s*=\s*\(([^)]*)\)", t, re.M) for s in re.findall(r'"(\.[\w.]+)"', group)}


def special_cases(files: dict[str, str], settings: CoreSettings) -> list[dict]:
    """Find places in the core's code (lab2shot/, excluding this analysis and the REGISTRIES files) that name a third-party
    project, one of its node types, or a format only an extension handles, either in a string literal or in an
    identifier. Comments and docstrings are ignored; each line is reported at most once."""
    projects = _projects(files)
    prefixes = {m.split(".")[0] for p, t in files.items() if p.startswith("adapters/") and p.endswith("/nodes.py")
                for m in re.findall(r'^\s+id = "([\w.]+)"', t, re.M)}
    names = {n.lower() for n in projects | prefixes}
    formats = extension_formats(files)
    hits: dict[tuple[str, int], list[str]] = {}
    for path, text in sorted(files.items()):
        if not path.startswith("lab2shot/") or not path.endswith(".py") or path.startswith("lab2shot/analysis/") or path in REGISTRIES:
            continue
        for kind, value, line in _code_tokens(text):
            if kind == tokenize.STRING:
                try:
                    s = str(ast.literal_eval(value)).lower()
                except (ValueError, SyntaxError):
                    continue
                head = s.split(".")[0]
                if s in names or (head in names and "." in s and " " not in s) or s in formats or any(s.endswith(f) and len(s) < 12 for f in formats):
                    hits.setdefault((path, line), []).append(value)
            elif kind == tokenize.NAME and any(part in names for part in value.lower().split("_")):
                hits.setdefault((path, line), []).append(value)
    return [{"path": path, "line": line, "what": "、".join(dict.fromkeys(what))} for (path, line), what in hits.items()]


# ------------------------------------------------------------------ tokens


def _code_tokens(text: str) -> list[tuple[int, str, int]]:
    """Return (token type, text, line) for the code tokens, excluding comments, docstrings and other bare string
    statements, and blank lines. NEWLINE, INDENT and DEDENT are kept to preserve structure."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return []
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}
    sig = [t for t in toks if t.type not in skip]
    out = []
    for i, t in enumerate(sig):
        if t.type == tokenize.STRING and (i == 0 or sig[i - 1].type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)) \
                and (i + 1 == len(sig) or sig[i + 1].type == tokenize.NEWLINE):
            continue  # a docstring
        out.append((t.type, t.string, t.start[0]))
    return out


def _normalised(tokens: list[tuple[int, str, int]]) -> list[str]:
    """Replace identifiers with N so that renamed copies still match; keywords, operators, strings and numbers are kept."""
    marks = {tokenize.NEWLINE: "⏎", tokenize.INDENT: "→", tokenize.DEDENT: "←"}
    return [marks.get(k) or ("N" if k == tokenize.NAME and not keyword.iskeyword(v) else v) for k, v, _ in tokens]


# ------------------------------------------------------------------ sameness


@dataclass(frozen=True)
class Clone:
    a: str
    a_lines: tuple[int, int]
    b: str
    b_lines: tuple[int, int]
    tokens: int

    @property
    def files(self) -> tuple[str, str]:
        return (self.a, self.b)


def clones(files: dict[str, str], min_tokens: int) -> list[Clone]:
    """Find token sequences of at least `min_tokens` (names normalised) that occur twice, across files or at
    non-overlapping positions within one file. Consecutive windows at the same offset are merged into one clone."""
    seqs: dict[str, tuple[list[str], list[int]]] = {}
    for path, text in files.items():
        if path.endswith(".py"):
            toks = _code_tokens(text)
            seqs[path] = (_normalised(toks), [line for _, _, line in toks])
    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for path, (vals, _) in seqs.items():
        for i in range(len(vals) - min_tokens + 1):
            index[hash(tuple(vals[i:i + min_tokens]))].append((path, i))
    diagonals: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for occ in index.values():
        if 1 < len(occ) <= 20:  # a sequence repeated widely is an idiom, not a copy
            for x in occ:
                for y in occ:
                    if x < y and (x[0] != y[0] or y[1] - x[1] >= min_tokens):
                        diagonals[(x[0], y[0], y[1] - x[1])].append(x[1])
    found = []
    for (a, b, offset), starts in diagonals.items():
        starts.sort()
        run_start = prev = starts[0]
        for s in starts[1:] + [None]:
            if s is not None and s == prev + 1:
                prev = s
                continue
            end = prev + min_tokens - 1
            la, lb = seqs[a][1], seqs[b][1]
            found.append(Clone(a, (la[run_start], la[end]), b, (lb[run_start + offset], lb[end + offset]), end - run_start + 1))
            if s is not None:
                run_start = prev = s
    return sorted(found, key=lambda c: -c.tokens)


def adapter_helpers(files: dict[str, str], min_tokens: int) -> list[dict]:
    """Find functions (names normalised, docstrings excluded) that are identical across several adapters and should
    move to the worker SDK. `sdk` lists matching definitions already in the SDK."""
    groups: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    for path, text in files.items():
        if not path.endswith(".py") or not path.startswith(("adapters/", "worker_sdk/")):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) else node.body
                code = ast.unparse(ast.Module(body=[*body], type_ignores=[])) if body else ""
                toks = _normalised(_code_tokens(code))
                if len(toks) >= min_tokens:
                    groups[hash((len(node.args.args), tuple(toks)))].append((path, node.name, node.lineno))
    out = []
    for places in groups.values():
        adapters = {p.split("/")[1] for p, _, _ in places if p.startswith("adapters/")}
        if len(adapters) >= 2:
            sdk = [f"{p}:{line} {name}" for p, name, line in places if p.startswith("worker_sdk/")]
            out.append({"places": [f"{p}:{line} {name}" for p, name, line in sorted(places) if p.startswith("adapters/")],
                        "adapters": sorted(adapters), "sdk": sdk})
    return sorted(out, key=lambda g: -len(g["adapters"]))


def type_twins() -> list[dict]:
    """Find data types that share storage and checks (container, channel count, required metadata, stage roles, 3D
    kinds, and the ports that accept exactly that type) and differ only by label; such types may be one type."""
    from ..data.contracts import META
    from ..data.types import DATA_TYPES, SCENE_KINDS, channels_of
    from ..nodes import node_types

    asked: dict[str, set[str]] = defaultdict(set)
    for t in node_types().values():
        for p in t.inputs:
            for alt in p.type.split("|"):
                asked[alt].add(f"{t.id}.{p.name}")
    sig: dict[tuple, list[str]] = defaultdict(list)
    for tid, d in DATA_TYPES.items():
        key = (tid.split(".")[0], channels_of(tid) or None, tuple(sorted((k, tuple(sorted(v)) if v else None) for k, v in META.get(tid, {}).items())),
               d.in_2d, d.in_3d, tuple(sorted(k for k, kind in SCENE_KINDS.items() if kind.type == tid)), tuple(sorted(asked.get(tid, ()))))
        sig[key].append(tid)
    return [{"types": ids, "labels": [DATA_TYPES[t].label for t in ids],
             "evidence": f"容器 {k[0]}，通道 {k[1] or '—'} 条，说明书要写的 {', '.join(m for m, _ in k[2]) or '没有'}，"
                         f"专门收它的输入口 {len(k[6])} 个"} for k, ids in sig.items() if len(ids) > 1]


def node_overlaps(threshold: float) -> list[dict]:
    """Find pairs of core nodes whose inputs, outputs and parameters overlap with Jaccard similarity >= `threshold`."""
    from ..nodes import node_types
    from ..nodes.applies import all_outputs

    feats = {}
    for t in node_types().values():
        if t.runtime == "core" and not t.id.startswith(("sample.", "test.")):
            feats[t.id] = ({f"入 {p.type}" for p in t.inputs} | {f"出 {p.type}" for p in all_outputs(t)}
                           | {f"参数 {n}" for n in t.Params.model_fields}, t.label)
    out = []
    ids = sorted(feats)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            fa, fb = feats[a][0], feats[b][0]
            if not fa or not fb:
                continue
            j = len(fa & fb) / len(fa | fb)
            if j >= threshold:
                out.append({"nodes": [feats[a][1], feats[b][1]], "ids": [a, b], "similarity": round(j, 2), "shared": sorted(fa & fb)})
    return sorted(out, key=lambda o: -o["similarity"])


# ------------------------------------------------------------------ baseline and allowlist


def baseline(path: Path = BASELINE_FILE) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def allowed_clones(path: Path = ALLOW_FILE) -> list[dict]:
    """Return the [[clones]] entries of the allowlist, each with `files` (the two file paths) and `reason`."""
    try:
        return tomllib.loads(path.read_text(encoding="utf-8")).get("clones", [])
    except OSError:
        return []


def unexplained(found: list[Clone], allow: list[dict]) -> list[Clone]:
    pairs = {tuple(sorted(a["files"])) for a in allow if a.get("reason")}
    return [c for c in found if tuple(sorted(c.files)) not in pairs]


def baseline_problems(now: dict[str, int], base: dict) -> list[str]:
    """Describe how the concept counts differ from the checked-in baseline, which must be updated deliberately."""
    want = base.get("concepts", {})
    out = []
    if not str(base.get("reason", "")).strip():
        out.append("core_baseline.toml 要写一行 reason：这次为什么加了或减了核心概念")
    for key, label in CONCEPTS.items():
        if key not in now:
            continue
        if key not in want:
            out.append(f"{label}（{key}）不在基线里：现在 {now[key]}")
        elif now[key] > want[key]:
            out.append(f"{label}多了：基线 {want[key]}，现在 {now[key]}。确实要加就改 core_baseline.toml 的数并写明理由；本质相同的先合并")
        elif now[key] < want[key]:
            out.append(f"{label}少了：基线 {want[key]}，现在 {now[key]}。把 core_baseline.toml 改小，锁住这次重构")
    return out


# ------------------------------------------------------------------ the trend


def trend(cache_file: Path, settings: CoreSettings, now: dict | None = None) -> list[dict]:
    """Return concept and project counts for the last commit of each of the last `trend_points` days of first-parent
    history, followed by the working tree (`now`). Each commit is read through git once and cached in `cache_file`."""
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    try:
        log = subprocess.run(["git", "-C", str(ROOT), "log", "--first-parent", "--format=%H %ct", "HEAD"],
                             capture_output=True, text=True, check=True, timeout=30).stdout.split("\n")
    except (OSError, subprocess.SubprocessError):
        log = []
    by_day: dict[str, tuple[str, int]] = {}
    for line in log:
        if not line.strip():
            continue
        sha, t = line.split()
        day = time.strftime("%Y-%m-%d", time.localtime(int(t)))
        by_day.setdefault(day, (sha, int(t)))  # log is newest first, so this keeps the day's last commit
    days = sorted(by_day)[-settings.trend_points:]
    rows = []
    grown = False
    for day in days:
        sha, t = by_day[day]
        key = sha
        if key not in cache:
            files = at_commit(sha)
            cache[key] = {"concepts": concepts(files, len(special_cases(files, settings))), "projects": len(_projects(files))}
            grown = True
        rows.append({"date": day, "rev": sha[:7], "time": t, **cache[key]})
    if grown:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache), encoding="utf-8")
    if now:
        rows.append({"date": "现在", "rev": "", "time": time.time(), **now})
    return rows


# ------------------------------------------------------------------ the whole section


@dataclass
class Health:
    concepts: dict[str, int]
    projects: int
    special: list[dict]
    twins: list[dict]
    overlaps: list[dict]
    clones: list[Clone]
    helpers: list[dict]
    trend: list[dict] = field(default_factory=list)


def measure(settings: CoreSettings, cache_file: Path | None = None) -> Health:
    files = working_tree()
    special = special_cases(files, settings)
    now_concepts = concepts(files, len(special))
    projects = len(_projects(files))
    rows = trend(cache_file, settings, {"concepts": now_concepts, "projects": projects}) if cache_file else []
    return Health(now_concepts, projects, special, type_twins(), node_overlaps(settings.node_overlap),
                  clones(files, settings.clone_tokens), adapter_helpers(files, settings.helper_tokens), rows)


def concept_count(counts: dict[str, int]) -> int:
    """Return the total concept count, excluding special cases (held to their target of 0, not a concept)."""
    return sum(v for k, v in counts.items() if k != "special_cases")


def section(h: Health) -> dict:
    """Build the core-health section for the page: the concept count and its breakdown, the suspected duplicates, and
    the trend of the count."""
    allow = allowed_clones()
    reasons = {tuple(sorted(a["files"])): a.get("reason", "") for a in allow}
    duplicates = []
    for t in h.twins:
        duplicates.append({"kind": "类型", "what": " / ".join(f"{label} {tid}" for tid, label in zip(t["types"], t["labels"], strict=True)), "evidence": t["evidence"]})
    for o in h.overlaps:
        duplicates.append({"kind": "核心节点", "what": " / ".join(o["nodes"]), "evidence": f"端口和参数重合 {o['similarity']:.0%}：{'、'.join(o['shared'])}"})
    for c in h.clones:
        why = reasons.get(tuple(sorted(c.files)), "")
        duplicates.append({"kind": "重复代码", "what": f"{c.a}:{c.a_lines[0]}-{c.a_lines[1]} ≈ {c.b}:{c.b_lines[0]}-{c.b_lines[1]}",
                           "evidence": f"{c.tokens} 个记号一样（名字不计）" + (f"；允许：{why}" if why else "")})
    for g in h.helpers:
        duplicates.append({"kind": "该进 SDK", "what": "、".join(g["places"]), "evidence": f"{len(g['adapters'])} 个扩展包里一样的函数"
                           + (f"；SDK 里已经有：{'、'.join(g['sdk'])}" if g["sdk"] else "")})
    return {"concepts": concept_count(h.concepts), "projects": h.projects,
            "by_concept": [{"id": k, "label": v, "count": h.concepts.get(k, 0)} for k, v in CONCEPTS.items() if k != "special_cases"],
            "duplicates": duplicates,
            "trend": [{"date": r["date"], "rev": r["rev"], "projects": r["projects"], "concepts": concept_count(r["concepts"])} for r in h.trend]}
