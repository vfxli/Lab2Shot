"""EnvSpec.build for GVHMR: compile pytorch3d from source, without its Pulsar renderer.

Meta never published a prebuilt pytorch3d wheel for this torch/CUDA/Python combination,
so it has to build from source. A plain `pip install --no-build-isolation` of it
(EnvSpec.compiled) compiles but fails to *link* when the build has more than one
`-gencode` target (sm_89 for the RTX 4090 and sm_120 for the RTX 5090): pytorch3d's
Pulsar renderer calls its CUDA kernels as explicit C++ templates across several
separately-compiled .cu files, and nvcc's default non-relocatable device code cannot
resolve those cross-file template calls, so the link fails with "undefined reference"
to symbols that do exist (linked with hidden visibility). `NVCC_FLAGS=-rdc=true`
(relocatable device code, pytorch3d's own setup.py hook) makes it link, but then fails
at import time ("undefined symbol: __cudaRegisterLinkedBinary_..."): relocatable device
code needs an extra `nvcc -dlink` step that pytorch3d's setup.py never asks torch's
CUDAExtension for (torch supports it only via a `dlink=True` kwarg setup.py does not pass).

GVHMR itself only imports `pytorch3d.transforms` (rotation conversions) and
`pytorch3d.ops.knn` (nothing under `pytorch3d.renderer.points.pulsar`), so instead of
teaching pytorch3d's build about `-dlink`, this build removes Pulsar entirely. Only the
throwaway copy is edited (deleted at the end of this script); the installer's pinned
checkout is never written (WHAM's build_dpvo.py patches its DPVO copy the same way).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env

# The commit the checkout is pinned at (extension.py extra_sources: PYTORCH3D). Not read
# here, but this script's content is part of the environment's build fingerprint (installer/plan.py env_fingerprint):
# a line removed here makes every installed GVHMR read as 需要重装, so it stays, and must change with the pin.
PYTORCH3D_COMMIT = "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba"  # v0.7.9

# ext.cpp's Pulsar Renderer includes and pybind registration block: identified by their
# first/last lines (exact text, so a change upstream fails loudly here instead of
# silently leaving Pulsar half-registered), everything between and including them is
# dropped. `./pulsar/global.h` (just above these, not part of either span) stays
# included: EPS / MAX_FLOAT / MAX_INT / ... used by ext.cpp's own "Constants" section
# right after are #defined there, not by the Renderer.
EXT_CPP_SPANS = (
    ('#include "./pulsar/pytorch/renderer.h"\n', '#include "./pulsar/pytorch/tensor_util.h"\n'),
    ('  // Pulsar.\n', '      &pulsar::pytorch::sphere_ids_from_result_info_nograd);\n'),
)


def _drop_span(text: str, first_line: str, last_line: str) -> str:
    start = text.index(first_line)
    end = text.index(last_line, start) + len(last_line)
    return text[:start] + text[end:]


ext_root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])

build = ext_root / "build" / "pytorch3d"
if build.exists():
    shutil.rmtree(build)
build.parent.mkdir(parents=True, exist_ok=True)
# the installer's checkout (extension.py extra_sources: PYTORCH3D), copied: the sources are patched below
# and the checkout itself is never written (a changed checkout would read as 「原始代码被改过」)
shutil.copytree(Path(os.environ["LAB2SHOT_EXTRA_PYTORCH3D"]), build, ignore=shutil.ignore_patterns(".git"))

ext_cpp = build / "pytorch3d" / "csrc" / "ext.cpp"
text = ext_cpp.read_text()
for first_line, last_line in EXT_CPP_SPANS:
    text = _drop_span(text, first_line, last_line)
ext_cpp.write_text(text)
# Only the Renderer's own compiled units (its .cpp/.cu, picked up by setup.py's
# recursive glob): pulsar/global.h, constants.h and logging.h stay (ext.cpp's
# "Constants" section right after the removed span still needs them), as does
# pulsar/include/ (headers the removed .cu/.cpp used to include, never compiled
# themselves).
pulsar = build / "pytorch3d" / "csrc" / "pulsar"
for sub in ("gpu", "host", "pytorch"):
    shutil.rmtree(pulsar / sub)
(pulsar / "warnings.cpp").unlink()

env = cuda_build_env(prefix)

subprocess.run(
    ["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", "--no-cache", str(build)],
    env=env, check=True,
)
shutil.rmtree(build.parent)  # everything needed is in site-packages now
