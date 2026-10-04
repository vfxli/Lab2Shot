"""LLM: llama.cpp + GGUF 的文字组件（翻译）。

一个共用扩展按 GGUF 权重表换模型：推理引擎是 llama.cpp（MIT），经 pinned 的
llama-cpp-python 在 worker 进程内调用（它内嵌的 llama.cpp 就是本扩展锁定的那份，
见 docs.md「我们怎么接的」）；权重是四份 GGUF——Hy-MT2-1.8B 与 Qwen3.5-9B 各
Q4_K_M / Q8_0 两个量化档，供质量 / 显存取舍。模型自己的许可都在权重条目里：
Hy-MT2 Apache-2.0、Qwen3.5-9B Apache-2.0（官方模型卡），都可商用。
"""

from __future__ import annotations

from lab2shot.sdk import CUDA_13_2_TOOLKIT, COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_COMMIT = "b56f34ab130b35c038ffa0acc81cb998a8b5f87c"  # 2026-10-01 master：hunyuan-dense 与 qwen35 架构都在

# Hy-MT2 官方仓（提示词格式的出处，official 引文核对它；模型本身不从它加载）
HY_MT2_URL = "https://github.com/Tencent-Hunyuan/Hy-MT2.git"
HY_MT2_COMMIT = "ff1903ecaa724e10951a23c16817a2413c752b35"

# 四份 GGUF（下载暂存 downloads/v3/weights 已核过 sha256；安装器按声明校验，文件已在目标位置时不重复下载）
HY_MT2_GGUF = "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/a0c709d9fac510f2c807aa3af52872340dc37a4a"
QWEN35_GGUF = "https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF/resolve/182be2fd6c7bc44887d88a91cb03ff009cc9f549"

# GGUF 权重表：节点「模型」参数的每个选项一份。采样参数照官方推荐照抄——Hy-MT2：1.8B/7B 推荐
# temp 0.7 / top_p 0.6 / top_k 20 / rep 1.05 / max_tokens 4096、无默认 system prompt（README）；
# Qwen3.5-9B：Instruct（非思考）模式通用任务 temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 /
# presence_penalty 1.5 / rep 1.0 / max_tokens 32768（官方模型卡 Best Practices）。
# n_ctx 按 max_tokens 加上提示词余量：Hy-MT2 4096 + 余量 → 8192；Qwen3.5 32768 + 余量 → 36864
# （它的 KV 是混合 SSM 结构，占得少）。
WEIGHTS = {
    "hy_mt2_q4": Weight(
        key="hy-mt2-q4", kind="url", source=f"{HY_MT2_GGUF}/Hy-MT2-1.8B-Q4_K_M.gguf",
        dest="hy_mt2/Hy-MT2-1.8B-Q4_K_M.gguf",
        sha256="dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699",
        option=("model", "hy_mt2_q4"),
    ),
    "hy_mt2_q8": Weight(
        key="hy-mt2-q8", kind="url", source=f"{HY_MT2_GGUF}/Hy-MT2-1.8B-Q8_0.gguf",
        dest="hy_mt2/Hy-MT2-1.8B-Q8_0.gguf",
        sha256="5c3fe0b1408a5ceb0143184ef247b11b579c525f4b02b060e6c851bb76fef1a4",
        option=("model", "hy_mt2_q8"),
    ),
    "qwen35_q4": Weight(
        key="qwen35-q4", kind="url", source=f"{QWEN35_GGUF}/Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        dest="qwen35_9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf",
        sha256="d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43",
        option=("model", "qwen35_q4"),
    ),
    "qwen35_q8": Weight(
        key="qwen35-q8", kind="url", source=f"{QWEN35_GGUF}/Qwen_Qwen3.5-9B-Q8_0.gguf",
        dest="qwen35_9b/Qwen_Qwen3.5-9B-Q8_0.gguf",
        sha256="b58fe056b5435070240de259f3f981aa38fee96825bbd78c088d5fd90e46f2b5",
        option=("model", "qwen35_q8"),
    ),
}


class Llm(Extension):
    name = "llm"
    sdk = 2
    homepage = "https://github.com/ggml-org/llama.cpp"
    source = GitSource(url=LLAMA_CPP_URL, commit=LLAMA_CPP_COMMIT)
    extra_sources = {"hy_mt2": GitSource(url=HY_MT2_URL, commit=HY_MT2_COMMIT)}
    import_repo = None
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/ggml-org/llama.cpp/blob/master/LICENSE",
    )
    generative = False
    env = EnvSpec(
        python="3.12",
        torch=(),  # llama.cpp 自带全套推理，不需要 torch：worker 也因此按 CPU 记账（显存数字见 docs.md 的实测）
        cuda_toolkit=CUDA_13_2_TOOLKIT,  # pip nvcc 13.2：头文件与 glibc >= 2.43 兼容的第一版（build.py 用）
        build="build.py",  # llama-cpp-python 按 GGML_CUDA 从源码编译（CMAKE_ARGS 在脚本里）
        imports=("llama_cpp",),  # 自检：包装库装上了、能 import
    )
    disk_gb = 4.0  # 环境 + 编译中间产物约 2 GB，比默认的 3 GB 略多
    weights = tuple(WEIGHTS.values())


EXTENSION = Llm()
