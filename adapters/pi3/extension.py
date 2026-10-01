"""Pi3 (π³, Shanghai AI Lab): feed-forward multi-view reconstruction (cameras + depth +
points of a whole shot in one network pass), used to cross-check camera solves.
Same environment as the VGGT extension (its own copy of the pins and requirements); the worker uses the SDK driver
(lab2shot_worker.feedforward)."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

from .pi3_models import MODELS

PI3_URL = "https://github.com/yyfz/Pi3.git"
# main: Pi3 + Pi3X (conv head, metric scale, depth-normal edge filter).
PI3_COMMIT = "9fa3ddb3f8d53041f8b2738df404f62223bbaa7b"


class Pi3(Extension):
    name = "pi3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Pi3 (π³)"
    summary = "不需要固定基准视角的前馈网络，从一组无序的画面预测仿射不变的相机位姿和尺度不变的逐帧点图"
    homepage = "https://github.com/yyfz/Pi3"
    source = GitSource(url=PI3_URL, commit=PI3_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="BSD-3-Clause（代码）+ CC BY-NC 4.0（Pi3 / Pi3X 权重）",
        url="https://github.com/yyfz/Pi3#-license",
        summary=(
            "非商用：Pi3 和 Pi3X 权重均为 CC BY-NC 4.0（严格非商用，再分发须保留此限制）。"
            "代码主体 BSD-3-Clause 可商用，但其中的 RoPE 位置编码文件 pi3/models/layers/pos_embed.py 来自 Naver DUSt3R/CroCo，"
            "是 CC BY-NC-SA 4.0 非商用；DINOv2 部分 Apache-2.0，PRoPE 部分 MIT。"
            "（Pi3 的 Hugging Face 模型卡标签写的是 bsd-2-clause，但正文写明商用须联系作者，以仓库 README 的许可表为准）"
        ),
    )
    # Same environment as VGGT, pins and requirements.txt copied (both are plain PyTorch;
    # upstream pins torch 2.5.1; pi3's plyfile / gradio extras are not needed).
    import_repo = ""
    worker_modules = ("pi3_models.py",)
    env = EnvSpec(python="3.12", torch=("torch==2.10.0", "torchvision==0.25.0"), torch_backend="cu128")
    weights = tuple(
        hf_file(
            repo, rev, filename,
            key=repo.split("/")[1],
            sha256=sha,
            note=f"{repo}（{size / 1e9:.1f} GB，CC BY-NC 4.0 非商用）",
        )
        for repo, rev, filename, size, sha in MODELS.values()
    )


EXTENSION = Pi3()
