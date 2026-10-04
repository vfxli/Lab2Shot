"""Diffusers: the shared extension for generative image models run through Hugging Face Diffusers.

First model: Qwen-Image 2.1 (QwenImage21Pipeline) — text-to-image, single-image editing and multi-reference
composition in one pipeline, with native RGBA transparency. The pipeline code is the pinned diffusers checkout
(imported from it, not pip-installed); the weights are the official Qwen/Qwen-Image-2.1 snapshot at a pinned
revision.
"""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

# The pipeline this adapter runs, at the commit it was read and checked against (QwenImage21Pipeline with the
# Qwen3-VL text encoder, the 64-channel RGBA VAE and FlowMatchEulerDiscreteScheduler).
DIFFUSERS_URL = "https://github.com/huggingface/diffusers.git"
DIFFUSERS_COMMIT = "578c9b2c6636ab2424a0e56186268b83623656b2"  # src/diffusers 0.41.0.dev0, main

# The official model snapshot (Hugging Face Qwen/Qwen-Image-2.1): every file the pipeline's from_pretrained reads,
# at a pinned revision. Large safetensors carry their sha256 (checked by the installer); small git files
# (config.json, the tokenizer, model_index.json) are pinned by the revision alone.
REPO = "Qwen/Qwen-Image-2.1"
REVISION = "d26bb61231c349cf6b7896fa83353113880e1ba3"

# file in the snapshot -> LFS sha256 ("" a small git file, pinned by the revision). Stored as
# weights/Qwen-Image-2.1/<file>, the folder layout from_pretrained reads.
SHARDS = {
    "model_index.json": "",
    "scheduler/scheduler_config.json": "",
    "processor/added_tokens.json": "",
    "processor/chat_template.jinja": "",
    "processor/merges.txt": "",
    "processor/preprocessor_config.json": "",
    "processor/special_tokens_map.json": "",
    "processor/tokenizer.json": "",
    "processor/tokenizer_config.json": "",
    "processor/video_preprocessor_config.json": "",
    "processor/vocab.json": "",
    "text_encoder/config.json": "",
    "text_encoder/generation_config.json": "",
    "text_encoder/model.safetensors.index.json": "",
    "text_encoder/model-00001-of-00004.safetensors": "dde00291b5f7fb92013895310a3da0ddba78674df9f10d505d375243dc01fc6f",
    "text_encoder/model-00002-of-00004.safetensors": "9047faccc0a6d98496a52d55f27be1c94a9c259d1e283fbea0128d054a948d42",
    "text_encoder/model-00003-of-00004.safetensors": "8c54187654c0176b73ae73785bf791dc9a14c9df7fb4310083a09d42048cb57e",
    "text_encoder/model-00004-of-00004.safetensors": "5311532aaaeae3259eb6a7b2c600636be1159adf7ded35f53579f7d0e7d43cdd",
    "transformer/config.json": "",
    "transformer/diffusion_pytorch_model.safetensors.index.json": "",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": "9e6bc2d641e67bf277895ea8777141044a38f3edb7101bc469b2961dd7c36b4b",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors": "3aaf234dcbe128530479735854a346b5e3e66283b7c11db56f836bbd1c13ebaa",
    "vae/config.json": "",
    "vae/diffusion_pytorch_model.safetensors": "a07a1b7c4ee2966a1b3bdc37de9b4f983d56937e46619f709a80b6e490675417",
}


def _weights() -> tuple:
    return tuple(
        hf_file(REPO, REVISION, name, key=f"qwen-image-2.1/{name}", dest=f"Qwen-Image-2.1/{name}", sha256=sha,
                said=("extension.weight.hf_choice", (("repo", REPO), ("choice", "qwen-image-2.1"))))
        for name, sha in SHARDS.items()
    )


class Diffusers(Extension):
    name = "diffusers"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    homepage = "https://github.com/huggingface/diffusers"
    source = GitSource(url=DIFFUSERS_URL, commit=DIFFUSERS_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        url="https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE",
    )
    generative = True  # runs a generative diffusion model (nodes/tags.py GENERATIVE)
    # The pipeline runs straight from the pinned checkout (romav2's pattern): what `import diffusers` gives is the
    # code that was read and checked, and the official cites name its real files and lines.
    import_repo = "src"
    env = EnvSpec(
        python="3.11",
        # The same torch the other GPU adapters on this machine use (runs on the RTX 4090 and the RTX 5090, cu128).
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu128",
        # what the pipeline imports at inference time: the Qwen3-VL text encoder (transformers >= 5.17 — the version
        # the pipeline's hidden-state workaround names, 5.18, ships the proper fix), cpu offload (accelerate), and the
        # plain image/numeric stack. No peft (LoRA loading only), no opencv (every picture reaches the worker as PNG).
        imports=("diffusers", "transformers", "accelerate"),
    )
    weights = _weights()


EXTENSION = Diffusers()
