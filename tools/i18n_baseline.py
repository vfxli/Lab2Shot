"""Lower lab2shot/i18n/baseline.toml to what is there now (lab2shot check i18n holds every count to it; it never goes up).

    uv run python tools/i18n_baseline.py          # lower every count that went down (a count that went up stays: check says so)
    uv run python tools/i18n_baseline.py --init   # write what is there now, whatever was there (only when the rules change)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from lab2shot.i18n import lint

    now = lint.counts(ROOT)
    old = None if "--init" in sys.argv else lint.baseline()
    lint.BASELINE.write_text(lint.write_baseline(now, old), encoding="utf-8")
    print(f"wrote {lint.BASELINE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
