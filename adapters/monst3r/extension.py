"""MonST3R (UC Berkeley / Google DeepMind / UC Merced): DUSt3R fine-tuned on dynamic
scenes. Point maps of frame pairs + global alignment with optical flow (SEA-RAFT) and
moving-object masks (its own flow check refined by SAM 2.1) give each frame's camera,
one focal length and depth, with people moving in the shot.
Same environment as CUT3R (its own copy of the pins and requirements); the worker uses CUT3R's DUSt3R model input
(adapters/cut3r/dust3r_input.py), so this extension requires cut3r."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

MONST3R_URL = "https://github.com/Junyi42/monst3r.git"
# main (last commit: "set batchify as default for faster inference").
MONST3R_COMMIT = "574cc77ad278bad582f470e5382624e01f8769a7"

MODEL_REPO, MODEL_REV = "Junyi42/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt", "1ea2a9eade23a10898ec73be3fd44b5aa4ea0255"
# weights key -> (Hugging Face repo, pinned revision, file, dest in weights/, sha256); notes: extension.monst3r.weight.<key>.note
FILES = {
    "monst3r-config": (MODEL_REPO, MODEL_REV, "config.json", "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt/config.json",
                       "fccd9c31cacd66a9a7a511e1debf8ddc43525d252b2378ffdff584ac8e237071"),
    "monst3r": (MODEL_REPO, MODEL_REV, "model.safetensors", "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt/model.safetensors",
                "53089f3bf16ef8d546d9c00232474f2b42490ce359986a1795795c6dc07a3e57"),
    # Optical flow for the flow loss and the moving-object masks (the checkpoint MonST3R uses).
    "sea-raft-spring-M": ("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", "eb97ef34ba5d856c3fa2cdcd073150c057ac8b69",
                          "model.safetensors", "Tartan-C-T-TSKH-spring540x960-M/model.safetensors",
                          "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102"),
    # Refines the moving-object masks through the shot (MonST3R's sam2_mask_refine).
    "sam2.1-hiera-large": ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610",
                           "sam2.1_hiera_large.pt", "sam2.1-hiera-large/sam2.1_hiera_large.pt",
                           "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"),
}


class MonST3R(Extension):
    name = "monst3r"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MonST3R"
    homepage = "https://monst3r-project.github.io/"
    source = GitSource(url=MONST3R_URL, commit=MONST3R_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/Junyi42/monst3r/blob/main/LICENSE",
    )
    generative = False
    # Same environment as CUT3R, pins and requirements.txt copied (upstream: Python 3.11 + PyTorch for CUDA 12.1).
    env = EnvSpec(python="3.12", torch=("torch==2.10.0", "torchvision==0.25.0"), torch_backend="cu128")
    requires = ("cut3r",)  # the worker imports cut3r's dust3r_input.py
    submodules = ("croco",)  # dust3r imports it from repo/croco
    weights = tuple(
        hf_file(
            repo, rev, filename,
            key=key,
            dest=dest,
            sha256=sha,
        )
        for key, (repo, rev, filename, dest, sha) in FILES.items()
    )


EXTENSION = MonST3R()
