"""Build VidMap's native library against the isolated COLMAP 4.2 prefix."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

repo = Path(os.environ["LAB2SHOT_EXT_REPO"])
root = Path(os.environ["LAB2SHOT_EXT_ROOT"])
prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
build = root / "build" / ("vidmap-" + str(os.getpid()))
shutil.copytree(repo, build, ignore=shutil.ignore_patterns(".git", "__pycache__"))
# ceres-solver (conda-forge GPU build) declares find_dependency(CUDAToolkit 12.9.86) REQUIRED; the
# machine's CUDA 12.9.86 toolkit satisfies it, and its libs are only needed at link time —
# VidMap's geometric solver itself runs on the CPU (COLMAP 4.2 cpu build).
cuda_home = Path("/usr/local/cuda-12.9")
assert (cuda_home / "version.json").is_file(), "system CUDA 12.9 toolkit missing"
env = dict(os.environ, CMAKE_PREFIX_PATH=str(prefix),
           CUDAToolkit_ROOT=str(cuda_home), CUDA_HOME=str(cuda_home),
           PATH=f"{cuda_home / 'bin'}:{os.environ.get('PATH', '')}")
subprocess.run(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation", "--no-deps", str(build)],
               env=env, cwd=build, check=True)

# conda-forge's libtiff 4.7.2 was compiled against a 12-bit libjpeg (jpeg12_*@LIBJPEG_8.0)
# but conda-forge's libjpeg-turbo 3.2.0 has no 12-bit variant, so libtiff.so.6 fails to
# resolve jpeg12_write_raw_data & co. VidMap/COLMAP never decode 12-bit JPEG, so a stub
# library that only satisfies the linker is enough; worker_env preloads it (LD_PRELOAD).
shim = root / "jpeg12_shim.so"  # in the extension root, shared by the live and self-check environments
src = root / "jpeg12_shim.c"
mapf = root / "jpeg12_shim.map"
src.write_text(
    "__asm__(\".symver jpeg12_read_raw_data_stub,jpeg12_read_raw_data@LIBJPEG_8.0\");\n"
    "__asm__(\".symver jpeg12_read_scanlines_stub,jpeg12_read_scanlines@LIBJPEG_8.0\");\n"
    "__asm__(\".symver jpeg12_write_raw_data_stub,jpeg12_write_raw_data@LIBJPEG_8.0\");\n"
    "__asm__(\".symver jpeg12_write_scanlines_stub,jpeg12_write_scanlines@LIBJPEG_8.0\");\n"
    "void jpeg12_read_raw_data_stub(void) {}\n"
    "void jpeg12_read_scanlines_stub(void) {}\n"
    "void jpeg12_write_raw_data_stub(void) {}\n"
    "void jpeg12_write_scanlines_stub(void) {}\n",
    encoding="utf-8",
)
mapf.write_text(
    "LIBJPEG_8.0 {\n  global:\n    jpeg12_read_raw_data;\n    jpeg12_read_scanlines;\n"
    "    jpeg12_write_raw_data;\n    jpeg12_write_scanlines;\n};\n",
    encoding="utf-8",
)
cc = os.environ.get("CC") or "gcc"
subprocess.run([cc, "-shared", "-fPIC", "-o", str(shim), str(src),
                f"-Wl,--version-script={mapf}"], check=True)
print("jpeg12_shim.so built")
