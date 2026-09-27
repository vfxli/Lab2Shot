"""DiffusionLight-Turbo: HDR environment map (light probe) from one frame, by
inpainting a chrome ball with SDXL + LoRA + depth ControlNet at several exposures."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

DIFFUSIONLIGHT_URL = "https://github.com/DiffusionLight/DiffusionLight-Turbo.git"
# 2025-07-03, the official research repo of DiffusionLight-Turbo (the conference
# DiffusionLight repo's successor). Its models/ folder holds both LoRAs the default
# "turbo_swapping" algorithm uses (Turbo LoRA + Exposure LoRA, MIT).
DIFFUSIONLIGHT_COMMIT = "6e8b0f1ad664f5e11281d80482ae19a490927cc4"


# Hugging Face repo -> (pinned revision, {file: sha256}): model files carry the LFS sha256, small git files
# (configs, tokenizers) are pinned by the revision alone. Only what the fp16 pipeline loads: the SDXL VAE is
# replaced by the fp16-fix VAE (upstream from_sdxl use_fixed_vae=True), so SDXL's own VAE is not downloaded.
# Stored as weights/<repo name>/<file>.
HF_FILES = {
    "stabilityai/stable-diffusion-xl-base-1.0": (
        "462165984030d82259a11f4367a4eed129e94a7b",
        {
            "LICENSE.md": "",
            "model_index.json": "",
            "scheduler/scheduler_config.json": "",
            "text_encoder/config.json": "",
            "text_encoder/model.fp16.safetensors": "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
            "text_encoder_2/config.json": "",
            "text_encoder_2/model.fp16.safetensors": "ec310df2af79c318e24d20511b601a591ca8cd4f1fce1d8dff822a356bcdb1f4",
            "tokenizer/merges.txt": "",
            "tokenizer/special_tokens_map.json": "",
            "tokenizer/tokenizer_config.json": "",
            "tokenizer/vocab.json": "",
            "tokenizer_2/merges.txt": "",
            "tokenizer_2/special_tokens_map.json": "",
            "tokenizer_2/tokenizer_config.json": "",
            "tokenizer_2/vocab.json": "",
            "unet/config.json": "",
            "unet/diffusion_pytorch_model.fp16.safetensors": "83e012a805b84c7ca28e5646747c90a243c65c8ba4f070e2d7ddc9d74661e139",
        },
    ),
    "diffusers/controlnet-depth-sdxl-1.0": (
        "17bb97973f29801224cd66f192c5ffacf82648b4",
        {
            "config.json": "",
            "diffusion_pytorch_model.fp16.safetensors": "66a6813e6bd7270ecfe68206a59ddd605a011ae85321188376605c66e0a4f303",
        },
    ),
    "madebyollin/sdxl-vae-fp16-fix": (
        "207b116dae70ace3637169f1ddd2434b91b3a8cd",
        {
            "config.json": "",
            "diffusion_pytorch_model.safetensors": "1b909373b28f2137098b0fd9dbc6f97f8410854f31f84ddc9fa04b077b0ace2c",
        },
    ),
    # Upstream's depth estimator for the ControlNet condition (relighting/argument.py
    # DEPTH_ESTIMATOR). Apache-2.0: no Depth Anything / non-commercial depth model involved.
    "Intel/dpt-hybrid-midas": (
        "11eaf7a1cf4bd70740697dbc216f98980c0aeb03",
        {
            "config.json": "",
            "preprocessor_config.json": "",
            "pytorch_model.bin": "b6c4d44f9d96ca3fa76dd3bbb153989a60b4ad5526559f3c598562a368d687ec",
        },
    ),
}
# Short component names for the weight keys (the install list shows "<tag>/<file>").
TAGS = {
    "stabilityai/stable-diffusion-xl-base-1.0": "sdxl",
    "diffusers/controlnet-depth-sdxl-1.0": "controlnet-depth",
    "madebyollin/sdxl-vae-fp16-fix": "vae-fp16-fix",
    "Intel/dpt-hybrid-midas": "dpt-hybrid",
}
NOTES = {
    "stabilityai/stable-diffusion-xl-base-1.0": "SDXL 1.0 基础模型 fp16（CreativeML Open RAIL++-M）",
    "diffusers/controlnet-depth-sdxl-1.0": "SDXL 深度 ControlNet fp16（CreativeML Open RAIL++-M）",
    "madebyollin/sdxl-vae-fp16-fix": "SDXL-VAE-FP16-Fix（MIT）",
    "Intel/dpt-hybrid-midas": "DPT-Hybrid MiDaS 深度估计，给 ControlNet 做条件（Apache-2.0）",
}


def _weights() -> tuple[Weight, ...]:
    out = []
    for repo, (rev, files) in HF_FILES.items():
        name = repo.split("/")[1]
        for f, sha256 in files.items():
            out.append(
                hf_file(
                    repo, rev, f,
                    key=f"{TAGS[repo]}/{f}",
                    dest=f"{name}/{f}",
                    note=NOTES[repo],
                    sha256=sha256,
                )
            )
    return tuple(out)


class DiffusionLight(Extension):
    name = "diffusionlight"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "DiffusionLight-Turbo"
    summary = "把光照估计改写成「在画面里补画一个镜面铬球」的问题，从一张低动态范围（LDR）画面估计光照"
    homepage = "https://diffusionlight.github.io/turbo/"
    source = GitSource(url=DIFFUSIONLIGHT_URL, commit=DIFFUSIONLIGHT_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT 代码/LoRA · SDXL 与 ControlNet 为 CreativeML Open RAIL++-M",
        url="https://github.com/DiffusionLight/DiffusionLight-Turbo/blob/main/LICENSE",
        summary=(
            "可商用，但须遵守 OpenRAIL++-M 的使用限制。逐项："
            "DiffusionLight-Turbo 代码及仓库自带的 Turbo LoRA、曝光 LoRA 为 MIT（Hugging Face 上同名 LoRA 模型卡也是 MIT）；"
            "SDXL 1.0 基础模型、SDXL 深度 ControlNet（diffusers）为 CreativeML Open RAIL++-M："
            "允许商用和分发生成结果，但不得用于许可证附件 A 列出的禁止用途（违法、伤害未成年人、虚假信息、歧视等），"
            "再分发模型时须附带同样的限制；"
            "SDXL-VAE-FP16-Fix 为 MIT；给 ControlNet 做深度条件的是 Intel DPT-Hybrid MiDaS，Apache-2.0"
            "（不是 Depth Anything，没有非商用深度模型）。"
            "不依赖 nvdiffrast，不安装 xformers"
        ),
    )
    # 上游钉的 torch 2.0.1+cu118 只编译到 sm_50-90，没有 sm_120（Blackwell）核心，所以用 torch 2.8.0+cu128：
    # diffusers 0.23 子类化的管线代码是纯 Python（scaled_dot_product_attention 从 torch 2.0 起就有），
    # 在新 torch 上原样可用，diffusers / transformers / accelerate 的版本不用改。装在 .venv-ada-blackwell。
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/diffusionlight/.venv-ada-blackwell
    env = EnvSpec(
        python="3.11",
        # 2.8.0 (not 2.7.x): the version several other extensions in this tree pin, so this
        # install reuses their cached wheels instead of a fresh multi-GB fetch.
        torch=("torch==2.8.0",),
        torch_backend="cu128",
    )
    weights = _weights()

    def worker_env(self) -> dict[str, str]:
        cache = self.paths.root / "cache"
        return {
            # Its models live in the extension cache (downloaded there at install time).
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
        }


EXTENSION = DiffusionLight()
