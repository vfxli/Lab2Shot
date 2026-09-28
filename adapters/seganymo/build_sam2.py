"""EnvSpec.build for SegAnyMo：编出它自带那份 SAM 2 的 CUDA 核 `_C`，放回检出里。

为什么非编不可：SAM 2 的视频那条路默认开着填洞（`repo/sam2/sam2/build_sam.py` 的 `++model.fill_hole_area=8`），
它靠 `_C.get_connected_componnets` 做连通域标记。这个核加载不上时，`sam2/utils/transforms.py` 的 `except`
会原样返回并只打一句警告：遮罩不填洞、静默降质。

两条硬约束：
1. `SAM2_BUILD_ALLOW_ERRORS=0`。SAM 2 自己的默认值是 1：编不过就当没事继续装，装完看着没问题、组件却是缺的。
2. 原仓库不动：先把 `repo/sam2` 拷到 `<扩展根>/build/sam2` 再编（和 `adapters/tram/build_droid.py`
   一样的做法），编好只把那一个 `_C*.so` 放回检出。放回去是因为 worker 的 PYTHONPATH 指的是
   `repo/sam2`（`extension.py` 的 `worker_env`），不是 site-packages。这一条在 `EnvSpec.places` 里登记着，
   `lab2shot ext place seganymo` 可以只做「放回去」这一步。

编出来覆盖哪些架构由 `cuda_build_env` 定：安装器传来的编译目标（设置「编译目标架构」与扩展声明的
env_archs 两边都有的），与编译这台机器插的是什么卡无关；这里不写死架构。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env

ext_root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])

INTO = repo / "sam2" / "sam2"          # worker 从这里 import sam2


def built() -> Path | None:
    got = sorted(INTO.glob("_C*.so"))
    return got[0] if got else None


if len(sys.argv) > 1 and sys.argv[1] == "place":
    # 只把已经编好的放回去：不下载、不编译、不碰环境（EnvSpec.places 的约定）
    kept = ext_root / "build" / "_C.so"
    if not kept.exists():
        raise SystemExit("没有编好的 _C.so 可放回（先跑一次完整安装）")
    shutil.copy2(kept, INTO / "_C.so")
    raise SystemExit(0)

build = ext_root / "build" / "sam2"
if build.exists():
    shutil.rmtree(build)
shutil.copytree(repo / "sam2", build,
                ignore=lambda folder, names: [n for n in names if n in (".git", "checkpoints", "demo", "notebooks")])

# `cuda_build_env` 已经按安装器传来的编译目标架构设好 TORCH_CUDA_ARCH_LIST（worker_sdk build.cuda_build_env），
# 所以这里不自己写架构。
env = cuda_build_env(prefix) | {
    "SAM2_BUILD_CUDA": "1",
    "SAM2_BUILD_ALLOW_ERRORS": "0",   # 编不过就当场失败，绝不静默跳过
}
subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=build, env=env, check=True)

made = sorted((build / "sam2").glob("_C*.so"))
if not made:
    raise SystemExit("SAM 2 的 _C 没编出来（setup.py 退出码是 0，但没有 .so）")
(ext_root / "build").mkdir(parents=True, exist_ok=True)
shutil.copy2(made[0], ext_root / "build" / "_C.so")   # 留一份，place 用
shutil.copy2(made[0], INTO / made[0].name)
shutil.rmtree(build)
print(f"SAM 2 的 _C 编好了：{INTO / made[0].name}（架构 {env['TORCH_CUDA_ARCH_LIST']}）")
