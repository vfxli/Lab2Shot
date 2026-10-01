"""CUT3R (Continuous 3D Perception Model, UC Berkeley / Google DeepMind): a recurrent
network that reads a video frame by frame and keeps a 3D state, giving each frame's
camera, focal length and metric depth, also with people moving in the shot.
Both this worker and MonST3R's use the worker SDK's reconstruction helpers (lab2shot_worker.recon) and this
adapter's DUSt3R model input (dust3r_input.py; MonST3R requires this extension for it)."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

CUT3R_URL = "https://github.com/CUT3R/CUT3R.git"
# main @ 2025-08-27 (last commit; inference code unchanged since the 2025-03 release).
CUT3R_COMMIT = "8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf"

# 512 px (long side, any aspect ratio), DPT head, trained on sequences of 4-64 views.
# The authors publish it on Google Drive only (README "Download Checkpoints"); the
# sha256 is of the downloaded file (3 173 761 006 bytes).
CHECKPOINT_ID = "1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD"
CHECKPOINT = "cut3r_512_dpt_4_64.pth"
CHECKPOINT_SHA256 = "45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103"


class Cut3r(Extension):
    name = "cut3r"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "CUT3R"
    summary = "一个统一框架，能解一大类三维任务；核心是带状态的循环模型，每来一帧新观测就更新一次自己的状态"
    homepage = "https://cut3r.github.io/"
    source = GitSource(url=CUT3R_URL, commit=CUT3R_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC BY-NC-SA 4.0（代码和权重）",
        url="https://github.com/CUT3R/CUT3R/blob/main/LICENSE",
        summary=(
            "非商用：CUT3R 代码和 cut3r_512_dpt_4_64 权重都按 CC BY-NC-SA 4.0 发布"
            "（署名、禁止商用、改编后须以相同许可发布）。"
            "代码里带的 Naver DUSt3R / CroCo 同为 CC BY-NC-SA 4.0；CroCo 的 pos_embed.py / blocks.py 另含 "
            "Meta MAE 的 CC BY-NC 4.0 部分和 timm 的 Apache-2.0 部分。没有其他模型或非商用依赖（不需要 SMPL、nvdiffrast 等）。"
            "「记忆更新」的 TTT3R 模式是 TTT3R（Inception3D，MIT）的记忆更新规则，由本扩展的 worker 按它的代码重写几行实现，"
            "不另外下载；它用的仍是 CUT3R 的权重，所以整体仍然非商用"
        ),
    )
    # Upstream: Python 3.11 + PyTorch for CUDA 12.1. The models are plain PyTorch (the
    # optional cuRoPE kernel is replaced by the same formula in PyTorch, see worker.py),
    # so the same torch as VGGT / Pi3 is used (cu128 wheels include sm_89).
    # Same environment as monst3r.
    env = EnvSpec(
        python="3.12",
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    worker_modules = ("dust3r_input.py",)
    weights = (
        Weight(
            key="cut3r_512_dpt_4_64",
            kind="url",
            source=f"https://drive.usercontent.google.com/download?id={CHECKPOINT_ID}&export=download&confirm=t",
            dest=CHECKPOINT,
            sha256=CHECKPOINT_SHA256,
            note=f"{CHECKPOINT}（3.2 GB，Google Drive，CC BY-NC-SA 4.0 非商用）",
        ),
    )


EXTENSION = Cut3r()
