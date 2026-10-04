"""SDMatte (vivo Camera Research + Shanghai University, ICCV 2025): interactive matting built on a
Stable Diffusion U-Net — a rough mask or a box points at the subject, no trimap needed. MIT."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

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
    homepage = "https://github.com/vivoCameraResearch/SDMatte"
    source = GitSource(url=SDMATTE_URL, commit=SDMATTE_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/vivoCameraResearch/SDMatte/blob/main/LICENSE",
    )
    generative = False  # processes footage, not tagged 生成式扩散: only QwenImage (diffusers) carries it
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
                sha256=CHECKPOINT_SHA256),
        # each config's note: extension.sdmatte.weight."config/<name>".note (i18n/<lang>.toml; one per CONFIGS entry)
        *(hf_file(WEIGHTS_REPO, WEIGHTS_REVISION, name, key=f"config/{name}", dest=f"SDMatte/{name}")
          for name in CONFIGS),
    )


EXTENSION = SDMatte()
