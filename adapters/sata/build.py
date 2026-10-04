"""EnvSpec.build for SATA：没有编译工作；权重放置由 `lab2shot ext place` 完成。

SATA 的 `sata/mypath.py` 从自己文件的位置推出 `RESULT_DIR = <repo>/result`，
`make_load_model` 只认这个路径（`result/<模型>/config.yaml`）。权重由安装器下到
扩展的 weights 目录（Weight dest="" 使 HF 快照落在 `weights/result/<模型>/`），
`lab2shot ext place sata`（本脚本带 `place` 参数）用符号链接放回检出，不复制
两份数百 MB 的 checkpoint。EnvSpec.places 登记 `repo/result/<模型>/config.yaml`，
所以 `lab2shot ext adopt` 会检查、缺失时提示运行 place。
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

if len(sys.argv) <= 1 or sys.argv[1] != "place":
    # 安装的「编译」步骤在权重下载之前跑，SATA 无编译，什么都不做；
    # 权重就位后由 `lab2shot ext place sata` 做放置。
    raise SystemExit(0)

root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
installed = root / "weights" / "result"
into = repo / "result"

if not installed.is_dir() or not any((p / "config.yaml").exists() for p in installed.iterdir()):
    raise SystemExit(f"SATA weights are not in {installed} (finish the install's download step first)")
into.parent.mkdir(parents=True, exist_ok=True)
if into.is_symlink() or into.is_file():
    into.unlink()
elif into.is_dir():
    shutil.rmtree(into)
into.symlink_to(installed, target_is_directory=True)
models = ", ".join(sorted(p.name for p in installed.iterdir() if (p / "config.yaml").exists()))
print(f"SATA weights placed at {into} -> {installed} (models {models})")
