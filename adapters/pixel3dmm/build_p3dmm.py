"""EnvSpec.build for Pixel3DMM, run with the environment's own Python.

Three things, two of which upstream's install_preprocessing_pipeline.sh does inside
the checkouts and this does next to them:

1. the code base env_paths expects, composed out of symlinks (codebase.py);
2. pytorch3d, compiled from the pinned checkout with the environment's CUDA 13.2
   compiler, minus its point renderer "pulsar" — pulsar's explicit template
   instantiations produce no host symbols with a CUDA 13 nvcc, so `_C` fails
   to link. Only its header constants (`_C.EPS`, `_C.MAX_UINT`, read at import time by
   pytorch3d.renderer.points.pulsar) are kept; `_C.PulsarRenderer` is gone, and
   nothing here renders points. What Pixel3DMM uses is knn_points / knn_gather,
   load_obj and Meshes, none of it pulsar;
3. PIPNet's FaceBoxes NMS, a Cython module its `sh make.sh` builds in place.

nvdiffrast is not here: from v0.4.0 it compiles its CUDA extension at pip install
(EnvSpec.compiled), not on the first cook, which is what a worker with no compiler
and no network needs.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from lab2shot_worker.build import cuda_build_env, run

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codebase import compose  # noqa: E402  (the adapter's own module, next to this script)

ext_root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
env = cuda_build_env(prefix)

print("Composing the Pixel3DMM code tree (symlinks, the checkouts are not modified)", flush=True)
lay = compose(ext_root)

# ------------------------------------------------------------------ pytorch3d without pulsar
build = ext_root / "build" / "pytorch3d"
if build.exists():
    shutil.rmtree(build)
shutil.copytree(ext_root / "pytorch3d", build, ignore=shutil.ignore_patterns(".git"))
pulsar = build / "pytorch3d" / "csrc" / "pulsar"
for source in [*pulsar.rglob("*.cu"), *pulsar.rglob("*.cpp")]:  # the headers stay: ext.cpp reads its constants
    source.unlink()
ext_cpp = build / "pytorch3d" / "csrc" / "ext.cpp"
text = ext_cpp.read_text()
text = text.replace('#include "./pulsar/pytorch/renderer.h"\n', "").replace('#include "./pulsar/pytorch/tensor_util.h"\n', "")
# from the "// Pulsar." comment down to the binding of pulsar_sphere_ids_from_result_info_nograd; the
# "// Constants." block after it stays (pytorch3d.renderer.points.pulsar reads EPS and MAX_UINT at import)
text, dropped = re.subn(r"\n *// Pulsar\..*?sphere_ids_from_result_info_nograd\);\n", "\n", text, flags=re.S)
if dropped != 1:
    raise SystemExit(f"pytorch3d ext.cpp: expected one pulsar binding block, found {dropped}")
ext_cpp.write_text(text)
print("Compiling pytorch3d (without the pulsar point renderer, which does not build with CUDA 13)", flush=True)
run_env = {**env, "MAX_JOBS": env.get("MAX_JOBS", "4")}
subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", "--no-cache",
                str(build)], env=run_env, check=True)
shutil.rmtree(build.parent)  # it is in site-packages now

# ------------------------------------------------------------------ PIPNet's NMS (its FaceBoxesV2/utils/make.sh)
# Its Cython source still writes np.int and np.float, gone from numpy since 1.24: the fixed source is written into
# our composed tree (the checkout keeps its own), and compiled from there.
nms = lay.faceboxes / "utils" / "nms" / "cpu_nms.pyx"
source = (lay.pipnet_repo / "FaceBoxesV2" / "utils" / "nms" / "cpu_nms.pyx").read_text()
# np.int / np.float were aliases of the builtins, so the builtins are the faithful replacement
# (dtype=int is the platform's default int, what np.int_t in the same file declares)
fixed, changed = re.subn(r"\bnp\.(int|float)\b(?![0-9_])", r"\1", source)
if changed != 2:  # np.float in the signature, np.int in the dtype: upstream is pinned, so this is exactly two
    raise SystemExit(f"PIPNet cpu_nms.pyx: expected 2 removed numpy aliases, found {changed}")
if nms.is_symlink():
    nms.unlink()
nms.write_text(fixed)
print("Compiling PIPNet's FaceBoxes NMS (Cython; np.int / np.float, removed in numpy 1.24, replaced)", flush=True)
run([sys.executable, "build.py", "build_ext", "--inplace"], cwd=lay.faceboxes / "utils")
print("Pixel3DMM environment ready", flush=True)
