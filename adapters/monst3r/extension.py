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
# weights key -> (Hugging Face repo, pinned revision, file, dest in weights/, sha256, note)
FILES = {
    "monst3r-config": (MODEL_REPO, MODEL_REV, "config.json", "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt/config.json",
                       "fccd9c31cacd66a9a7a511e1debf8ddc43525d252b2378ffdff584ac8e237071", "MonST3R 网络配置"),
    "monst3r": (MODEL_REPO, MODEL_REV, "model.safetensors", "MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt/model.safetensors",
                "53089f3bf16ef8d546d9c00232474f2b42490ce359986a1795795c6dc07a3e57",
                "MonST3R ViT-L 512 DPT 权重（2.3 GB，CC BY-NC-SA 4.0 非商用）"),
    # Optical flow for the flow loss and the moving-object masks (the checkpoint MonST3R uses).
    "sea-raft-spring-M": ("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", "eb97ef34ba5d856c3fa2cdcd073150c057ac8b69",
                          "model.safetensors", "Tartan-C-T-TSKH-spring540x960-M/model.safetensors",
                          "cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102",
                          "SEA-RAFT 光流 Tartan-C-T-TSKH-spring540x960-M（79 MB，BSD-3-Clause）"),
    # Refines the moving-object masks through the shot (MonST3R's sam2_mask_refine).
    "sam2.1-hiera-large": ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610",
                           "sam2.1_hiera_large.pt", "sam2.1-hiera-large/sam2.1_hiera_large.pt",
                           "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
                           "SAM 2.1 Hiera-L，细化运动物体遮罩（898 MB，Apache-2.0）"),
}


class MonST3R(Extension):
    name = "monst3r"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MonST3R"
    summary = "处理动态视频，基本以前馈的方式给出随时间变化的动态点云，以及每帧的相机位姿和内参"
    homepage = "https://monst3r-project.github.io/"
    source = GitSource(url=MONST3R_URL, commit=MONST3R_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC BY-NC-SA 4.0（代码和权重）；附带 SEA-RAFT（BSD-3-Clause）和 SAM 2.1（Apache-2.0）",
        url="https://github.com/Junyi42/monst3r/blob/main/LICENSE",
        summary=(
            "非商用：MonST3R 代码和权重都按 CC BY-NC-SA 4.0 发布（署名、禁止商用、改编后须以相同许可发布），"
            "它基于 Naver DUSt3R / CroCo（同为 CC BY-NC-SA 4.0；CroCo 的 pos_embed.py / blocks.py 另含 Meta MAE 的 CC BY-NC 4.0 "
            "部分和 timm 的 Apache-2.0 部分），权重由 DUSt3R 权重微调而来。"
            "另外用到的两个模型可商用：光流 SEA-RAFT（代码和权重 BSD-3-Clause，普林斯顿）、"
            "SAM 2.1 Hiera-L（代码和权重 Apache-2.0，Meta；其中连通域代码 cc_torch 为 BSD-3-Clause）。"
            "整体按最严的 CC BY-NC-SA 4.0 对待。没有 SMPL、nvdiffrast 等其他依赖"
        ),
    )
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
            note=note,
        )
        for key, (repo, rev, filename, dest, sha, note) in FILES.items()
    )


EXTENSION = MonST3R()
