"""NVIDIA Cosmos DiffusionRenderer: video inverse rendering (G-buffers: base colour,
normal, depth, roughness, metallic) and forward rendering (relight a shot with an HDRI)."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

# github.com/nv-tlabs/cosmos1-diffusion-renderer redirects here (same repo, renamed). The pinned commit
# differs from the initial release only in README / requirements / model URLs, not in code.
REPO_URL = "https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer.git"
REPO_COMMIT = "0f3e2dc435032ecbad654c2fc2153df85384b138"

# weight key -> (Hugging Face repo, file, LFS sha256 or None for small git files; pinned by REVISIONS).
# Stored as weights/<repo name>/<file>, the layout upstream's --checkpoint_dir expects.
# Upstream's download script also fetches nothing else (T5 and the guardrail models are
# not used by DiffusionRenderer); the tokenizer's autoencoder.jit / model.pt are unused.
INVERSE = "nvidia/Diffusion_Renderer_Inverse_Cosmos_7B"
FORWARD = "nvidia/Diffusion_Renderer_Forward_Cosmos_7B"
TOKENIZER = "nvidia/Cosmos-Tokenize1-CV8x8x8-720p"
REVISIONS = {
    INVERSE: "41f63abccfb764b2516ff85b98af529c2662e2df",
    FORWARD: "7d5f9b81affa0def046cd39bf02a8ca13a011442",
    TOKENIZER: "b6af495317c76f287a4131e9299936b1533f5f9f",
}
FILES = {
    "inverse-7b": (INVERSE, "model.pt", "5b9f671b484e6ca4b13cc412d2e4d47338ec96008ddddae68afc1a0431a6d3ee"),
    "inverse-7b-config": (INVERSE, "config.json", None),
    "forward-7b": (FORWARD, "model.pt", "56b6dd02d0653da94dc311b80d43ea606a111212438fc9525f292937f650ef98"),
    "forward-7b-config": (FORWARD, "config.json", None),
    "tokenizer-encoder": (TOKENIZER, "encoder.jit", "4f08348acac20e1b16283be6a9e3836b54bf556665155989fbb4a9710c20cb7a"),
    "tokenizer-decoder": (TOKENIZER, "decoder.jit", "955610ca0900509d50e7d5f824751cbad3ad262ab014841a5d046d0ec2a6c7a0"),
    "tokenizer-mean-std": (TOKENIZER, "mean_std.pt", "a863149f5393b724ef3729dbbf9f13d796833ac23d36baf76c02ac1a71bad19e"),
    "tokenizer-image-mean-std": (TOKENIZER, "image_mean_std.pt", "916fe37d596d9e81c32abc358fd6cbec7b512bd06caa975c7bc02085d4035f44"),
    "tokenizer-config": (TOKENIZER, "config.json", None),
}
NOTES = {
    INVERSE: "逆渲染 7B（拆 G-buffer），28.9 GB，NVIDIA Open Model License",
    FORWARD: "正向渲染 7B（按 HDRI 重打光），28.9 GB，NVIDIA Open Model License",
    TOKENIZER: "Cosmos 视频编解码器 CV8x8x8-720p，NVIDIA Open Model License",
}


def _weights() -> tuple[Weight, ...]:
    return tuple(
        hf_file(repo, REVISIONS[repo], name, key=key, dest=f"{repo.split('/')[1]}/{name}", note=NOTES[repo], sha256=sha or "",
                # its licence (NVIDIA Open Model License) requires crediting it once the model is actually there —
                # on the one real model file, not its small config.json, so the notice waits for the download
                notice="Built on NVIDIA Cosmos" if name != "config.json" else "")
        for key, (repo, name, sha) in FILES.items()
    )


class DiffusionRenderer(Extension):
    name = "diffusionrenderer"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Cosmos DiffusionRenderer"
    summary = "在一个统一框架里同时解逆向渲染和正向渲染这一对问题的神经方法"
    homepage = "https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/"
    source = GitSource(url=REPO_URL, commit=REPO_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0 代码 · NVIDIA Open Model License 权重",
        url="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        summary=(
            "可商用。代码（cosmos-transfer1-diffusion-renderer）为 Apache-2.0；"
            "逆渲染、正向渲染两个 7B 模型和 Cosmos-Tokenize1 编解码器为 NVIDIA Open Model License"
            "（2025-10-24 版）：免费、可商用、可改、可分发，输出归使用者，NVIDIA 不主张所有权；"
            "分发模型时须附许可证和 “Licensed by NVIDIA Corporation under the NVIDIA Open Model License” 声明，"
            "对外提供用到 Cosmos 模型的产品/服务时须在网站、界面或文档写明 “Built on NVIDIA Cosmos”；"
            "绕过或削弱模型自带的安全护栏、对任何人提起该模型的专利/版权诉讼，许可自动终止"
            "（DiffusionRenderer 官方推理本身就不带护栏模型，本扩展按官方方式运行）；须遵守 NVIDIA 可信 AI 条款。"
            "不安装 nvdiffrast（NVIDIA 非商用）：HDRI 投影用 PyTorch 重写；不安装 transformer-engine（用 PyTorch 等价实现）"
        ),
    )
    worker_modules = ("shims.py",)
    env = EnvSpec(
        python="3.10",
        # Upstream pins torch 2.6.0 / CUDA 12.4 only for TransformerEngine 1.12, which is not installed
        # (shims.py); the model code and the TorchScript tokenizer run unchanged on the torch the
        # hamer / smirk extensions already use (Python 3.10, shared uv cache).
        torch=("torch==2.8.0", "torchvision==0.23.0"),
        torch_backend="cu128",
    )
    weights = _weights()

    def worker_env(self) -> dict[str, str]:
        # The HDRI input is an EXR, read with OpenCV (off by default in opencv-python).
        return {"OPENCV_IO_ENABLE_OPENEXR": "1"}


EXTENSION = DiffusionRenderer()
