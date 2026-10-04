"""Install (or remove) a DCC plugin on this machine's Windows side while developing, exactly as the server's download
packs it (lab2shot/server/plugins.py: the same zip, unpacked here), instead of copying files by hand.

    uv run python tools/deploy_dcc.py nuke             # ~/.nuke/lab2shot/ + a marked block in ~/.nuke/init.py
    uv run python tools/deploy_dcc.py nuke --remove    # both taken away again
    uv run python tools/deploy_dcc.py maya [--remove]  # ~/Documents/maya/modules/lab2shot.mod + lab2shot/

What it touches, and nothing else: the plugin's own folder (written beside as <name>.new, swapped in whole, the old one
removed: a DCC holding files open makes it stop before anything is half written), and for Nuke the user's init.py:
only the block between the Lab2Shot marks, appended when missing (the file backed up as init.py.lab2shot.bak first,
written aside and renamed). The user's other plugins, settings and files are never touched.

`--home`: the Windows user folder as this machine sees it (default: %USERPROFILE% asked from Windows, /mnt/c/...).
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BEGIN = "# >>> Lab2Shot (managed block: delete from this line to the <<< line to uninstall)"
END = "# <<< Lab2Shot"
BLOCK = [BEGIN, "import nuke", "nuke.pluginAddPath('./lab2shot')", END]


def windows_home() -> Path:
    got = subprocess.run(["cmd.exe", "/c", "echo %USERPROFILE%"], capture_output=True, text=True, cwd="/mnt/c",
                         timeout=30).stdout.strip()
    return Path(subprocess.run(["wslpath", got], capture_output=True, text=True, timeout=30).stdout.strip())


def packed(dcc: str) -> zipfile.ZipFile:
    """The plugin as the server's download packs it."""
    from lab2shot.server.plugins import download

    return zipfile.ZipFile(io.BytesIO(download(dcc).body))


def swap_in(source: Path, target: Path) -> None:
    """`source` becomes `target`: copied beside it first, then renamed into place; the old one removed after."""
    fresh = target.with_name(target.name + ".new")
    old = target.with_name(target.name + ".old")
    for leftover in (fresh, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    shutil.copytree(source, fresh)
    try:
        if target.exists():
            target.rename(old)
        fresh.rename(target)
    except OSError as exc:
        if old.exists() and not target.exists():
            old.rename(target)
        shutil.rmtree(fresh, ignore_errors=True)
        sys.exit(f"{target} is in use (close the DCC and deploy again): {exc}")
    if old.exists():
        shutil.rmtree(old, ignore_errors=True)


def _write_aside(path: Path, text: str, newline: str) -> None:
    tmp = path.with_name(path.name + ".lab2shot.part")
    with open(tmp, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)
    os.replace(tmp, path)


def _newline(raw: bytes) -> str:
    return "\r\n" if b"\r\n" in raw else "\n"


def add_block(init: Path) -> str:
    raw = init.read_bytes() if init.exists() else b""
    text = raw.decode("utf-8")
    if BEGIN in text:
        return "init.py: the Lab2Shot block is there already"
    if init.exists():
        shutil.copy2(init, init.with_name(init.name + ".lab2shot.bak"))
    lines = text.splitlines()
    lines += ([""] if lines and lines[-1].strip() else []) + BLOCK
    _write_aside(init, "\n".join(lines) + "\n", _newline(raw) if raw else "\n")
    return "init.py: Lab2Shot block added" + (" (backup: init.py.lab2shot.bak)" if raw else " (new file)")


def remove_block(init: Path) -> str:
    if not init.exists():
        return "init.py: none"
    raw = init.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    if BEGIN not in lines:
        return "init.py: no Lab2Shot block"
    start = lines.index(BEGIN)
    end = lines.index(END, start) if END in lines[start:] else len(lines) - 1
    before, after = lines[:start], lines[end + 1:]
    if before and not before[-1].strip() and not after:  # the blank line add_block put before the block
        before = before[:-1]
    kept = before + after
    backup = init.with_name(init.name + ".lab2shot.bak")
    if backup.exists() and backup.read_bytes().decode("utf-8").splitlines() == kept:
        os.replace(backup, init)  # the user's file exactly as it was before us (its own line ends and last newline)
        return "init.py: Lab2Shot block removed (restored from init.py.lab2shot.bak)"
    if not kept:
        init.unlink()
        return "init.py: removed (it held only the Lab2Shot block)"
    _write_aside(init, "\n".join(kept) + "\n", _newline(raw))
    return "init.py: Lab2Shot block removed"


def deploy_nuke(home: Path, remove: bool) -> list[str]:
    dot = home / ".nuke"
    target = dot / "lab2shot"
    if remove:
        out = []
        if target.exists():
            shutil.rmtree(target)
            out.append(f"removed {target}")
        out.append(remove_block(dot / "init.py"))
        return out
    dot.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, packed("nuke") as z:
        z.extractall(tmp)
        top = next(Path(tmp).iterdir())
        source = top / "lab2shot"
        shutil.copy2(top / "VERSION", source / "VERSION")
        swap_in(source, target)
    return [f"installed {target}", add_block(dot / "init.py")]


def deploy_maya(home: Path, remove: bool) -> list[str]:
    modules = home / "Documents" / "maya" / "modules"
    target, mod = modules / "lab2shot", modules / "lab2shot.mod"
    if remove:
        out = []
        if target.exists():
            shutil.rmtree(target)
            out.append(f"removed {target}")
        if mod.exists():
            mod.unlink()
            out.append(f"removed {mod}")
        return out
    modules.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp, packed("maya") as z:
        z.extractall(tmp)
        top = next(Path(tmp).iterdir())
        shutil.copy2(top / "VERSION", top / "lab2shot" / "VERSION")
        swap_in(top / "lab2shot", target)
        shutil.copy2(top / "lab2shot.mod", mod)
    return [f"installed {target}", f"installed {mod}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dcc", choices=("nuke", "maya"))
    ap.add_argument("--remove", action="store_true", help="take the plugin away again")
    ap.add_argument("--home", type=Path, help="the Windows user folder as seen here (default: %%USERPROFILE%%)")
    args = ap.parse_args()
    home = args.home or windows_home()
    if not home.is_dir():
        sys.exit(f"no such folder: {home}")
    for line in (deploy_nuke if args.dcc == "nuke" else deploy_maya)(home, args.remove):
        print(line)


if __name__ == "__main__":
    main()
