"""Meta VGGT: feed-forward multi-view reconstruction (cameras + depth + points of a
whole shot in one network pass), used to cross-check camera solves."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

from .vggt_models import MODELS

VGGT_URL = "https://github.com/facebookresearch/vggt.git"
# main @ 2026-05-18: includes the 2026-05-15 aggregator fix that keeps only the
# layers the heads read (2-3x more frames in the same GPU memory).
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"


class Vggt(Extension):
    name = "vggt"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Meta VGGT"
    summary = ("前馈网络，从一张、几张或几百张画面里几秒钟内推断出场景的全部关键三维属性：相机内外参、点图、"
               "深度图和三维点轨迹")
    homepage = "https://github.com/facebookresearch/vggt"
    source = GitSource(url=VGGT_URL, commit=VGGT_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="VGGT License（代码）；权重：原版 CC BY-NC 4.0（非商用）/ Commercial 版 VGGT License",
        url="https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt",
        summary=(
            "代码是 Meta 的 VGGT License（2025-07-29 版）：允许商用、修改和再分发（再分发须附带许可证原文），"
            "发表论文须注明使用了 VGGT，须遵守附带的可接受使用政策（禁止军事、战争、核工业、间谍、武器、关键基础设施操作、"
            "欺诈冒充等用途），对 Meta 提起知识产权诉讼则许可终止，无任何担保。"
            "权重两份：VGGT-1B 原版（默认）为 CC BY-NC 4.0，非商用，仅限研究；"
            "VGGT-1B-Commercial 按同一份 VGGT License 发布，可以商用（禁军事），"
            "需要先在 https://huggingface.co/facebook/VGGT-1B-Commercial 申请访问（人工审批），效果与原版接近"
        ),
    )
    import_repo = ""
    worker_modules = ("vggt_models.py",)
    env = EnvSpec(
        python="3.12",
        # Upstream pins torch 2.3.1; the model is plain PyTorch (SDPA attention, no
        # compiled ops) and runs unchanged on 2.10. cu128 wheels include sm_89.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        hf_file(
            *MODELS["original"][:3],
            key="VGGT-1B",
            sha256=MODELS["original"][4],
            note="VGGT-1B 原版权重（5.0 GB，CC BY-NC 4.0 非商用）",
        ),
        Weight(
            key="VGGT-1B-Commercial",
            kind="hf",
            source=MODELS["commercial"][0],
            revision=MODELS["commercial"][1],
            dest="VGGT-1B-Commercial",
            files=("model.safetensors", "LICENSE", "README.md", "config.json"),
            gated=True,
            option=("model", "commercial"),
            note="VGGT-1B-Commercial 权重（5.0 GB，可商用）；需先在 https://huggingface.co/facebook/VGGT-1B-Commercial 申请访问",
        ),
    )


EXTENSION = Vggt()
