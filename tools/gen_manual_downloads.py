"""Generate docs/manual-downloads.md from the extension declarations.

The document lists every file the installer cannot fetch by itself (items of lab2shot.extensions.manual) and every
weight whose Hugging Face repository requires an approved access request. Run after changing either:

    uv run python tools/gen_manual_downloads.py
"""

from __future__ import annotations

from pathlib import Path

from lab2shot import i18n
from lab2shot.extensions import extensions, manual

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "manual-downloads.md"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    """The document, in Chinese (the one docs/ keeps)."""
    with i18n.using("zh"):
        return _render()


def _render() -> str:
    t = i18n.t
    needs = manual.needed_by()
    lines = [
        t("cli.manual_downloads.title"),
        "",
        t("cli.manual_downloads.generated"),
        "",
        t("cli.manual_downloads.intro"),
        "",
        t("cli.manual_downloads.kind_manual"),
        t("cli.manual_downloads.kind_gated"),
        "",
        t("cli.manual_downloads.when"),
        "",
        t("cli.manual_downloads.manual_heading"),
        "",
        t("cli.manual_downloads.steps"),
        "",
        t("cli.manual_downloads.manual_step1"),
        t("cli.manual_downloads.manual_step2"),
        t("cli.manual_downloads.manual_step3"),
        t("cli.manual_downloads.manual_step4"),
        "",
        t("cli.manual_downloads.body_models"),
        "",
    ]
    for key, item in manual.items().items():
        users = i18n.separator().join(e["title"] for e in needs.get(key, [])) or "—"
        lines += [
            f"### {item.title}",
            "",
            t("cli.manual_downloads.item_header"),
            "| --- | --- |",
            t("cli.manual_downloads.item_what", what=_cell(item.what)),
            t("cli.manual_downloads.item_users", users=_cell(users)),
            t("cli.manual_downloads.item_page", page=item.page),
            t("cli.manual_downloads.item_download", download=_cell(item.download)),
            t("cli.manual_downloads.item_file", filename=item.filename),
            t("cli.manual_downloads.item_note", note=_cell(item.note)),
            "",
        ]
    lines += [
        t("cli.manual_downloads.gated_heading"),
        "",
        t("cli.manual_downloads.steps"),
        "",
        t("cli.manual_downloads.gated_step1"),
        t("cli.manual_downloads.gated_step2"),
        t("cli.manual_downloads.gated_step3"),
        t("cli.manual_downloads.gated_step4"),
        "",
        t("cli.manual_downloads.gated_header"),
        "| --- | --- | --- |",
    ]
    for ext in sorted(extensions().values(), key=lambda e: e.title.lower()):
        for w in ext.weights:
            if w.gated:
                lines.append(f"| {_cell(ext.title)} | {w.key} | {w.page} |")
    lines += [
        "",
        t("cli.manual_downloads.network_heading"),
        "",
        t("cli.manual_downloads.network"),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
