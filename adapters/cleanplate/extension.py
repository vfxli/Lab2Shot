"""LTX-2.5 Clean Plate (Lightricks): an IC-LoRA on the LTX-2.5 22B video model that removes people and vehicles from a
shot and rebuilds the background behind them, giving a clean plate of the same scene, frame for frame.

It runs in the shared LTX base (adapters/ltx: code, environment, base model files, ltx_runtime.py; runs_in). This
extension adds only its LoRA and the fixed recipe in worker.py."""

from __future__ import annotations

from adapters.ltx.extension import LICENSE_URL
from lab2shot.sdk import COMMERCIAL, Extension, LicenseInfo, hf_file

LORA_REPO, LORA_REV = "Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate", "5403b2b77e9994e0ae2e549e9933d06c9e8df26f"
LORA_FILE = "ltx-2.5-22b-ic-lora-clean-plate-1.0.safetensors"
LORA_SHA256 = "27f4f7ba552d3bf7273dd817291123a5934f17e7698ca9cd8945b079a5358393"


class CleanPlate(Extension):
    name = "cleanplate"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "LTX-2.5 Clean Plate"
    homepage = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate"
    license = LicenseInfo(tag=COMMERCIAL, url=LICENSE_URL)
    generative = False  # processes footage, not tagged 生成式扩散: only QwenImage (diffusers) carries it
    runs_in = "ltx"  # its code, environment, base model files and worker runtime (ltx_runtime.py)
    weights = (hf_file(LORA_REPO, LORA_REV, LORA_FILE, key="cleanplate_lora", sha256=LORA_SHA256, gated=True),)


EXTENSION = CleanPlate()
