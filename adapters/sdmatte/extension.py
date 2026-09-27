"""SDMatte (vivo Camera Research + Shanghai University, ICCV 2025): interactive matting built on a
Stable Diffusion U-Net — a rough mask or a box points at the subject, no trimap needed. MIT."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

SDMATTE_URL = "https://github.com/vivoCameraResearch/SDMatte.git"
SDMATTE_COMMIT = "c3ea6949f270d253539a2bda2ad290e127934de9"  # main

WEIGHTS_REPO = "LongfeiHuang/SDMatte"
WEIGHTS_REVISION = "3936e1b68222c5bbfb59cc6970b74c5b14a09a68"
CHECKPOINT_SHA256 = "d3465a9b7a75dbbf348be603831f5cf417aee6b0e84ae7ef96dac710c9e2de46"
# the diffusers layout the model is built from (`load_weight=False`: the shapes come from these, the numbers
# from SDMatte.pth, so none of Stable Diffusion's own weight files is downloaded)
CONFIGS = ("unet/config.json", "vae/config.json", "text_encoder/config.json", "scheduler/scheduler_config.json",
           "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json", "tokenizer/vocab.json",
           "tokenizer/merges.txt")


class SDMatte(Extension):
    name = "sdmatte"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SDMatte"
    summary = "扩散模型驱动的交互式抠像，把文字驱动的交互换成视觉提示驱动的交互；可商用"
    homepage = "https://github.com/vivoCameraResearch/SDMatte"
    source = GitSource(url=SDMATTE_URL, commit=SDMATTE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT（代码和权重）",
        url="https://github.com/vivoCameraResearch/SDMatte/blob/main/LICENSE",
        summary=(
            "可商用：仓库是 MIT，Hugging Face 上的权重页也写 license: mit。"
            "网络结构从 Stable Diffusion 2 改来，但我们只下载作者自己训练的 SDMatte.pth，"
            "不下载、也不装 Stability AI 的任何权重文件。"
            "训练数据里有 Composition-1K、RefMatte 等只许研究用的数据集，严格的商业交付前建议做一次法务确认"
        ),
    )
    import_repo = ""
    env = EnvSpec(
        python="3.12",
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
        # SDMatte.pth is a pickled detectron2 checkpoint (torch.save of {"model": ..., "trainer": ...})
        pickled_checkpoints=True,
        imports=("diffusers", "transformers"),
    )
    weights = (
        hf_file(WEIGHTS_REPO, WEIGHTS_REVISION, "SDMatte.pth", key="sdmatte", dest="SDMatte/SDMatte.pth",
                sha256=CHECKPOINT_SHA256,
                note="SDMatte 权重（12 GB，MIT）：作者训练的 U-Net、VAE 和文字编码器，Stable Diffusion 的原权重一个都不下载"),
        *(hf_file(WEIGHTS_REPO, WEIGHTS_REVISION, name, key=f"config/{name}", dest=f"SDMatte/{name}",
                  note="SDMatte 的网络配置") for name in CONFIGS),
    )


EXTENSION = SDMatte()
