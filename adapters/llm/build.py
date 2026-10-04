"""EnvSpec.build for LLM: llama-cpp-python 按 GGML_CUDA 从源码编译进环境。

pip 的 llama-cpp-python 是 sdist（内嵌一份 llama.cpp 源码，随版本钉住），CUDA 后端由 CMAKE_ARGS 打开；
scikit-build-core 把 CMAKE_ARGS 里的定义原样传给 CMake。编译用扩展环境自己的 pip CUDA 工具链
（EnvSpec.cuda_toolkit，13.2：头文件第一条与 glibc >= 2.43 兼容，机器上的 CUDA 12.8 头文件编不过
C23 的 rsqrt/cospi），host 编译器用安装器配的 CC/CXX（gcc-14：nvcc 13.2 接受的最后一代）。
架构取安装器选的（LAB2SHOT_CUDA_ARCHS，build.archs 设置收窄而来），不在这台机器的卡上猜。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from lab2shot_shared.gpu_arch import ARCHS_ENV
from lab2shot_worker.build import cuda_build_env

prefix = Path(os.environ["LAB2SHOT_EXT_PREFIX"])
env = cuda_build_env(prefix)  # CUDA_HOME / PATH 指到本环境的 pip nvcc；没有时它自己报 E-WORKER-NONVCC

# TORCH_CUDA_ARCH_LIST 形（"8.9;12.0"）转 CMake 的 CMAKE_CUDA_ARCHITECTURES（"89;120"）
caps = os.environ.get(ARCHS_ENV) or "8.9;12.0"
archs = ";".join(c.replace(".", "") for c in caps.split(";") if c)
nvcc = Path(env["CUDA_HOME"]) / "bin" / "nvcc"
env["CMAKE_ARGS"] = (f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={archs} -DCMAKE_CUDA_COMPILER={nvcc}"
                     f" -DCMAKE_CUDA_FLAGS=-ccbin={os.environ.get('CXX', 'g++')}")
subprocess.check_call(["uv", "pip", "install", "--python", sys.executable, "--no-build-isolation",
                       "llama-cpp-python==0.3.36"], env=env)
subprocess.check_call([sys.executable, "-c", "import llama_cpp"], env=env)
print(f"llama-cpp-python built (GGML_CUDA, {archs})")
