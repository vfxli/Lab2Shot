"""FBX (.fbx): read cameras, models and skinned characters out of FBX files, write them for Maya and Unreal.

The Autodesk FBX SDK is not on any package index: the user downloads it and accepts Autodesk's licence in
「手动下载」 (the item FBX_SDK and its installer FbxSdk are declared in this file), which installs it under
third_party/_fbx_sdk/<version>/.
Installing this extension compiles a small pybind11 module (fbxio.cpp, build.py) against that SDK into the
extension's own conda-forge environment: Python 3.12, numpy, and the libxml2 2.x the SDK links to (the machine's
libxml2 is a newer, incompatible one). The SDK is linked in statically, so the worker needs nothing else of it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from lab2shot.messages import Msg
from lab2shot.sdk import (BASIC, THIRD_PARTY_DIR, EnvSpec, Extension, GitSource, Install, LicenseInfo, ManualError, ManualItem,
                          manual_weight, open_member, unwrap_licence)

PYBIND11_URL = "https://github.com/pybind/pybind11.git"
PYBIND11_COMMIT = "97bf890db679505a14dfe547a5e77bb2bd05dc90"  # tag v3.1.0: the headers the module is built with

FBX_ROOT = THIRD_PARTY_DIR / "_fbx_sdk"  # <version>/: what Autodesk's installer installs, after the user accepted
TITLE = "Autodesk FBX SDK"


class FbxSdk(Install):
    """The Autodesk FBX SDK for Linux: a tar.gz holding Autodesk's installer program (fbx<version>_fbxsdk_linux),
    which prints its END USER LICENSE AGREEMENT and installs only when it is answered "yes". Lab2Shot shows that
    text on the page, and runs the installer answering "yes" only when the user clicked 同意并安装.
    Installing the SDK is this extension's own code; the core only provides the consent mechanism."""
    consent = True
    PROGRAM = re.compile(r"(?:^|/)fbx(\d{4})(\d)(\d*)_fbxsdk_linux$", re.IGNORECASE)
    AGREE_PROMPT = b"To continue installing the software"

    def installed(self) -> list[str]:
        """Installed versions, newest first."""
        if not FBX_ROOT.is_dir():
            return []
        found = [p.name for p in FBX_ROOT.iterdir() if (p / "include" / "fbxsdk.h").is_file() and "." in p.name
                 and not p.name.endswith(".part")]
        return sorted(found, key=lambda v: [int(x) if x.isdigit() else 0 for x in v.split(".")], reverse=True)

    def ready(self) -> bool:
        return bool(self.installed())

    def path(self) -> Path | None:
        versions = self.installed()
        return FBX_ROOT / versions[0] if versions else None

    def has(self, version: str) -> bool:
        return version in self.installed()

    def version(self, names: list[str]) -> str:
        m = next((m for m in map(self.PROGRAM.search, names) if m), None)
        return ".".join(x for x in m.groups() if x) if m else ""

    def licence(self, file: Path, names: list[str], scratch: Path) -> str:
        target = scratch / "target"
        output = self._run(file, names, scratch, target, b"no\n")  # declined: the installer prints it and stops
        if target.exists() and any(target.iterdir()):
            raise ManualError(Msg("E-MANUAL-LICENCEINSTALLED"))
        at = output.find(self.AGREE_PROMPT)
        if at < 0:
            raise ManualError(Msg("E-MANUAL-NOLICENCE"))
        return unwrap_licence(output[:at])

    def accept(self, file: Path, names: list[str], scratch: Path) -> None:
        version = self.version(names)
        final = FBX_ROOT / version
        part = FBX_ROOT / f"{version}.part"
        shutil.rmtree(part, ignore_errors=True)
        # yes: the licence (the user's own click); y: extract into the folder given; n: do not page the ReadMe
        output = self._run(file, names, scratch, part, b"yes\ny\nn\n", timeout=900)
        if not (part / "include" / "fbxsdk.h").is_file():
            shutil.rmtree(part, ignore_errors=True)
            last = [line for line in output.decode("cp1252", "replace").splitlines() if line.strip()][-3:]
            raise ManualError(Msg("E-MANUAL-FBXFAILED", title=TITLE, detail=" / ".join(last)))
        shutil.rmtree(final, ignore_errors=True)
        part.rename(final)

    def _run(self, file: Path, names: list[str], scratch: Path, target: Path, answers: bytes, timeout: int = 120) -> bytes:
        """Autodesk's installer program from the file, run into `target` with these answers on its input. Its output
        reaches a pipe only when it ends (fully buffered), so the answers are given up front, in the order it asks."""
        member = next(n for n in names if self.PROGRAM.search(n))
        program = scratch / Path(member).name
        with open_member(file, member) as src, program.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        program.chmod(0o755)
        target.mkdir(parents=True, exist_ok=True)
        (scratch / "tmp").mkdir(exist_ok=True)
        try:
            done = subprocess.run([str(program), "-w", str(scratch / "tmp"), str(target)], input=answers, cwd=scratch,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ManualError(Msg("E-MANUAL-FBXRUN", title=TITLE, detail=str(exc))) from exc
        return done.stdout


# The file the user downloads from Autodesk and accepts the licence of: declared by this extension, never named by the core
FBX_SDK = ManualItem(
    key="fbx_sdk", title=TITLE, what="读写 FBX 文件的开发库", page="https://aps.autodesk.com/developer/overview/fbx-sdk",
    download="FBX SDK 的 Linux 版（gcc）", filename="fbx2020310_fbxsdk_gcc_linux.tar.gz",
    note="下载前后都需要同意 Autodesk 的许可协议", markers=("fbx*_fbxsdk_linux",), install=FbxSdk(),
    alone=("fbx2020310_fbxsdk_linux",), looks_like=("*fbx*sdk*",),
    hint=Msg("W-MANUAL-FBXPLATFORM"))


class Fbx(Extension):
    name = "fbx"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Autodesk FBX SDK"
    summary = "Autodesk 的免费 C++ SDK：用它编写插件、转换器和应用，借 FBX 技术转换和交换三维资产"
    format_module = True  # reads and writes scene formats
    homepage = "https://aps.autodesk.com/developer/overview/fbx-sdk"
    source = GitSource(url=PYBIND11_URL, commit=PYBIND11_COMMIT)
    license = LicenseInfo(
        tag=BASIC,
        name="Autodesk FBX SDK License",
        url="https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-license",
        summary=(
            "FBX SDK 属于 Autodesk，由你在「手动下载」里看过并同意它的许可协议后安装（协议 1.1.1 条允许用于开发、研究、"
            "内部、教育或商业用途；SDK 本身不得再分发）。编译用的 pybind11 为 BSD-3。安装时编译一个小模块，"
            "需要本机 C++ 编译器（config/local.toml 的 build 段 cxx）"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # conda-forge: the SDK's libfbxsdk needs libxml2.so.2 (libxml2 2.x); numpy for the worker
        conda=("python=3.12.14", "libxml2=2.13.9", "numpy=2.5.3"),
        build="build.py",
        build_files=("fbxio.cpp",),
    )
    manual_items = (FBX_SDK,)
    weights = (manual_weight(FBX_SDK),)


EXTENSION = Fbx()
