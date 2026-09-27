"""The installer's self-check, run inside a freshly built extension environment: `python -m lab2shot_worker.selfcheck`.

On the CPU (the installer hides the GPUs), with no footage: imports the modules named in LAB2SHOT_SELFCHECK_MODULES (a
JSON list: the worker SDK, the pinned torch / torchvision, what the extension declares in EnvSpec.imports) and, when
torch is among them, sums a tiny tensor. Prints one line, `LAB2SHOT_SELFCHECK {json}`: the Python version, torch's
version, and every module that failed with why (lab2shot/installer/run.py reads it; nothing else is printed there).
"""

from __future__ import annotations

import importlib
import json
import os
import sys


def main() -> int:
    modules = json.loads(os.environ.get("LAB2SHOT_SELFCHECK_MODULES", "[]"))
    failed: dict[str, str] = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except ImportError as exc:
            # A shared library that is loaded by its package through torch.ops (xformers._C, torchvision._C) is not a
            # Python extension module and cannot be imported directly; that is not a defect of the environment.
            if "does not define module export function" not in str(exc):
                failed[name] = f"{type(exc).__name__}: {exc}"[:300]
        except BaseException as exc:  # noqa: BLE001 - whatever a broken package raises is the answer
            failed[name] = f"{type(exc).__name__}: {exc}"[:300]
    out: dict = {"python": sys.version.split()[0], "failed": failed}
    if "torch" in modules and "torch" not in failed:
        import torch

        out["torch"] = torch.__version__
        try:
            out["cpu"] = float(torch.ones(3).sum()) == 3.0
        except BaseException as exc:  # noqa: BLE001
            failed["torch.ones(3).sum()"] = str(exc)[:300]
    print("LAB2SHOT_SELFCHECK " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
