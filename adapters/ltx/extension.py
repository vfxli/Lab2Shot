"""LTX-2.5 (Lightricks): the base that the LTX IC-LoRA extensions run on. It has no nodes of its own.

It holds what every LTX feature shares, so that each one only adds its LoRA and its fixed recipe:
* the pinned upstream code (Lightricks/LTX-2: ltx-core + ltx-pipelines, put on PYTHONPATH, not pip-installed);
* the one environment those packages need (requirements.txt here);
* the base model files: the distilled 22B transformer, the Gemma 4 text encoder and the video VAE;
* the worker-side runtime, ltx_runtime.py: a feature's whole job (run_feature) from its LoRA and fixed recipe.

A feature extension (alphagen, ...) declares `runs_in = "ltx"` and nothing of the code or environment: its workers
run in this environment with this worker_env, installing it installs this one first, and it downloads only its own
LoRA. This extension is a base: it is not listed on its own (Extension.is_base)."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

LTX_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_COMMIT = "9ec55f9f22798a3198d9c923856824821bc3317e"  # 2026-10-02 (main), ltx-core / ltx-pipelines 1.4.2
SOURCE = GitSource(url=LTX_URL, commit=LTX_COMMIT)
LICENSE_URL = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x"

# ltx-core asks for torch~=2.7 and transformers >=5.8,<5.15 (5.15 breaks its Gemma 4 encoder). torch 2.10 + cu128
# covers both cards (sm_89, sm_120). The optional natten / ltx-kernels extras are not used.
ENV = EnvSpec(
    python="3.12",
    torch=("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0"),
    torch_backend="cu128",
    imports=("ltx_core", "ltx_pipelines", "cv2"),
)

BASE_REPO, BASE_REV = "Lightricks/LTX-2.5", "2356ce76915d6c48d313d7e8b25900e1dd3abaa8"
# (key, file in the repository, sha256): what IC-LoRA inference on the distilled model reads. Not downloaded: the
# audio VAE, the spatial / temporal upscalers, the duration head and the dev transformer (stage 2, audio and the
# full-model pipelines are not used).
BASE_FILES = (
    ("ltx_transformer", "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
     "31eb3cad89b9e54e99dd3baf286f70825ac4f6c660a70d9184d895be76d7bff4"),
    ("ltx_text_encoder", "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
     "ef7243612fdae7a75cb4d5cee9433e81380675fb6c213bd98ae74a9cd16561d1"),
    ("ltx_video_vae", "vae/ltx-2.5-video-vae-bf16.safetensors",
     "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3"),
)
# the access request of Lightricks/LTX-2.5 must have been accepted on Hugging Face
BASE_WEIGHTS = tuple(hf_file(BASE_REPO, BASE_REV, file, key=key, sha256=sha, gated=True) for key, file, sha in BASE_FILES)


class Ltx(Extension):
    name = "ltx"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "LTX-2.5"
    homepage = "https://github.com/Lightricks/LTX-2"
    source = SOURCE
    # LTX-2.x Community License: free, commercial use included, for entities under USD 10 million annual revenue
    license = LicenseInfo(tag=COMMERCIAL, url=LICENSE_URL)
    generative = False  # processes footage, not tagged 生成式扩散: only QwenImage (diffusers) carries it
    env = ENV
    worker_modules = ("ltx_runtime.py",)
    weights = BASE_WEIGHTS

    def worker_env(self) -> dict[str, str]:
        """Every feature's worker (Extension.runs_in): the upstream packages imported from the pinned checkout (src
        layout), not pip-installed; the base model files where this extension installed them (ltx_runtime BASE_ENV)."""
        import os

        packages = self.paths.repo / "packages"
        return {"PYTHONPATH": os.pathsep.join(str(packages / p / "src") for p in ("ltx-core", "ltx-pipelines")),
                "LAB2SHOT_LTX_WEIGHTS": str(self.paths.weights)}


EXTENSION = Ltx()
