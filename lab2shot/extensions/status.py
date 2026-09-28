"""An extension's install status, computed in one place for every reader: nodes (usable or not, and why), the
admin page, the command line and the installer's own checks. A pure look: nothing is installed here.

A weight is "ok", "missing" (not downloaded yet), "pending" (a gated download waiting for access), or for an item the
user downloads by hand (lab2shot.extensions.manual) "manual" (not there yet) or "consent" (there, waiting for the user
to accept its licence). Weights with an `option` are needed only when a node's parameter picks that option: they
never block the rest.

Ready means usable: installed, finished for the spec this code builds (the fingerprint the installer recorded equals
the one this code computes: installer/plan.py built_for, env_fingerprint, the single judgement), required weights
present, and the installer's self-check passed for that build (lab2shot/installer/run.py step_selfcheck). A card
lights up only then.
"""

from __future__ import annotations

from ..messages import Msg
from . import manual
from .spec import Extension

MANUAL_STATUS = {"ready": "ok", "consent": "consent", "unrecognised": "manual", "missing": "manual"}


def orphan_weight_keys(ext: Extension) -> list[str]:
    """Weight keys recorded in the install record but no longer present in this code.

    A weight's `key` is its identity on disk (`third_party/<project>/install_state.json` records installed items by
    key). After a key is renamed, the file is still present but its status becomes "missing", the extension is no
    longer ready and its nodes are unavailable on the page, without any notice, while tens of GB of weights remain
    untouched. This function makes the condition explicit: orphan keys are listed and shown on the extension page."""
    return sorted(set(ext.install_state().get("weights", {})) - {w.key for w in ext.weights})


def weight_rows(ext: Extension) -> list[dict]:
    """One row per weight: what it is, how it is obtained and its state."""
    state = ext.install_state().get("weights", {})
    rows = []
    for w in ext.weights:
        row = {"key": w.key, "kind": w.kind, "note": w.note, "gated": w.gated, "optional": w.option is not None,
               "status": state.get(w.key, "missing"), "notice": w.notice}
        if w.kind == "manual":
            item = manual.item_of(ext, w.source)
            row.update(item=item.key, title=item.title, page=item.page, status=MANUAL_STATUS[manual.state(item.key)])
        else:
            row["page"] = w.page
        rows.append(row)
    return rows


def missing_requirements(ext: Extension) -> list[str]:
    """The extensions `ext` builds on (Extension.requires) that did not load whole, directly or through theirs."""
    from .registry import broken_extensions, extensions

    exts, broken = extensions(), broken_extensions()
    return [r for r in ext.requires if r not in exts or r in broken or missing_requirements(exts[r])]


def built(ext: Extension) -> tuple[str | None, str]:
    """(the fingerprint the live environment was finished for, None when there is none; the one this code builds):
    ready needs the two to be equal (installer/plan.py built_for, env_fingerprint)."""
    from ..installer.plan import built_for, env_fingerprint

    return built_for(ext.paths), env_fingerprint(ext)


def selfcheck(ext: Extension, current: str) -> tuple[str, dict]:
    """("passed" / "failed" / "none", its record) of the installer's self-check for the build `current` (a record for
    another build is none)."""
    record = ext.install_state().get("selfcheck") or {}
    if record.get("fingerprint") != current:
        return "none", record
    return ("passed" if record.get("ok") else "failed"), record


def extension_status(ext: Extension) -> dict:
    """installed / ready / what blocks it, for one extension; `label`: its state in a few words (a badge); `reason`
    the text of `message` (what blocks it: {code, level, text, params}; None when ready). An environment not finished
    for the spec this code builds (built from other code, or with no finished build recorded) is not ready
    (E-EXT-OUTDATED): its nodes are unavailable until it is installed again."""
    rows = weight_rows(ext)
    lacking_ext = missing_requirements(ext)
    required = [r for r in rows if not r["optional"]]
    installed = ext.paths.python.exists()
    recorded, current = built(ext)
    stale = installed and recorded != current  # not finished for this code's spec (another spec, or never finished)
    by_hand = [r for r in required if r["kind"] == "manual" and r["status"] != "ok"]
    lacking = [r for r in by_hand if r["status"] == "manual"]
    pending = [r for r in required if r["status"] == "pending"]
    check, record = selfcheck(ext, current)
    weights_ok = all(r["status"] == "ok" for r in required)
    # files upstream reads from hard-coded paths are requirements too (EnvSpec.places): having all weights downloaded
    # does not mean they are in place. With one such file missing, upstream would run the preceding steps (COLMAP,
    # matting, tens of thousands of training steps) for over an hour before failing when it reads the file.
    # This is a framework-level rule: each extension declares its list; the core only verifies it and knows no
    # project's file names
    unplaced = _unplaced(ext) if installed else []
    orphans = orphan_weight_keys(ext)  # renamed keys: file present, record mismatched (see orphan_weight_keys)
    # the card lights only on a passed self-check of this build; one not checked yet is 未自检 (installing it again
    # runs the self-check: installer/plan.py ALWAYS)
    ready = installed and not stale and weights_ok and not unplaced and not lacking_ext and check == "passed"
    if orphans and not weights_ok:
        # weights incomplete and the record holds unmatched names: almost certainly a renamed key; state it directly
        # instead of prompting a 20 GB re-download
        reason = Msg("E-EXT-WEIGHTKEYCHANGED", title=ext.title, name=ext.name, keys=orphans)
        label = "安装记录对不上"
    elif lacking_ext:
        reason = Msg("E-EXT-NEEDSEXT", title=ext.title, needs=lacking_ext)
        label = f"缺少扩展包 {'、'.join(lacking_ext)}"
    elif stale:
        reason, label = Msg("E-EXT-OUTDATED", title=ext.title, name=ext.name), "需要重装"
    elif ready:
        reason, label = None, "已就绪"
    elif lacking:
        needs = [manual.explain(r["item"], "missing") for r in lacking]
        reason = needs[0] if len(needs) == 1 else Msg("E-EXT-NEEDSMANUAL", count=len(needs), needs=needs)
        label = f"缺少 {'、'.join(r['title'] for r in lacking)}"
    elif by_hand:
        reason, label = manual.explain(by_hand[0]["item"], "consent"), "等同意许可协议"
    elif not installed:
        reason, label = Msg("E-EXT-NOTINSTALLED", title=ext.title, name=ext.name), "未安装"
    elif pending:
        reason, label = Msg("E-EXT-GATED", page=pending[0]["page"]), "模型未齐"
    elif not weights_ok:
        missing = [r["key"] for r in required if r["status"] != "ok"]
        reason, label = Msg("E-EXT-WEIGHTSMISSING", weights=missing), "模型未齐"
    elif unplaced:
        reason, label = Msg("E-EXT-NOTPLACED", title=ext.title, name=ext.name, count=len(unplaced), files=unplaced[:3]), "文件没摆到位"
    elif check == "failed":
        said = (record.get("message") or {}).get("text", "")
        reason, label = Msg("E-EXT-SELFCHECKFAILED", title=ext.title, detail=said), "自检未过"
    else:
        reason, label = Msg("E-EXT-NOSELFCHECK", title=ext.title, name=ext.name), "未自检"
    return {"installed": installed, "ready": ready, "reason": reason.text if reason else "",
            "message": reason.json() if reason else None, "label": label, "weights": rows, "selfcheck": check,
            "needs_manual": any(r["kind"] == "manual" for r in rows), "needs_request": any(r["gated"] for r in rows),
            "manual": [{"key": r["item"], "title": r["title"], "status": r["status"]} for r in by_hand]}


def public(status: dict) -> dict:
    """What an account that does not install sees of a status: a card that is not ready shows only 「未安装」, with
    no reason, hand downloads, access requests or weight states."""
    return {**status, "label": "已就绪" if status["ready"] else "未安装", "reason": "", "message": None, "manual": [],
            "needs_manual": False, "needs_request": False,
            "weights": [{k: v for k, v in r.items() if k in ("key", "kind", "note", "optional", "notice")} for r in status["weights"]]}


def _unplaced(ext: Extension) -> list[str]:
    """Files declared to be placed into the checkout (EnvSpec.places) that are currently missing. Read-only; when the
    check cannot read, nothing counts as missing: the purpose is to stop a job before it starts, not to mark the
    extension unusable because of a disk problem."""
    from ..installer.place import missing

    try:
        return missing(ext)
    except Exception:  # noqa: BLE001
        return []
