"""LTX-2.5 Alpha Gen (Lightricks): an IC-LoRA on the LTX-2.5 22B video model that pulls an alpha matte from an
ordinary RGB shot. It takes no mask and no prompt: the model decides on its own what the foreground is.

It runs in the shared LTX base (adapters/ltx: code, environment, base model files, ltx_runtime.py; runs_in). This
extension adds only its LoRA and the fixed matte recipe in worker.py."""

from __future__ import annotations

from adapters.ltx.extension import LICENSE_URL
from lab2shot.sdk import COMMERCIAL, Extension, LicenseInfo, hf_file

LORA_REPO, LORA_REV = "Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen", "6184df14b1b560bf9b6447d3ee126bd7e4a88513"
LORA_FILE = "ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"  # the released checkpoint (training step 25000)
LORA_SHA256 = "d9e143f979e0756f0c83d772e6a8890f4c6b933db1f7d93fc06e25c179f3fd36"


class AlphaGen(Extension):
    name = "alphagen"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "LTX-2.5 Alpha Gen"
    homepage = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"
    license = LicenseInfo(tag=COMMERCIAL, url=LICENSE_URL)
    generative = False  # processes footage, not tagged 生成式扩散: only QwenImage (diffusers) carries it
    runs_in = "ltx"  # its code, environment, base model files and worker runtime (ltx_runtime.py)
    weights = (hf_file(LORA_REPO, LORA_REV, LORA_FILE, key="alphagen_lora", sha256=LORA_SHA256, gated=True),)


EXTENSION = AlphaGen()
