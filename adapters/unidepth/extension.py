"""UniDepth V2 (ETH Zurich): per-frame monocular metric depth, point map and
camera intrinsics; accepts a known focal length as a condition.
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_weights, downloads

UNIDEPTH_URL = "https://github.com/lpiccinelli-eth/UniDepth.git"
UNIDEPTH_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"  # 2025-05-18 (main)

# model param -> (Hugging Face repo, pinned revision, {file: sha256}, note). The model cards
# carry no licence of their own: the repository's CC BY-NC 4.0 applies.
MODELS = {
    "unidepth-v2-vitl14": (
        downloads.UNIDEPTH_V2_L.repo,  # ViPE runs this model too: pinned once (extensions/downloads.py)
        downloads.UNIDEPTH_V2_L.revision,
        {d.filename: d.sha256 for d in (downloads.UNIDEPTH_V2_L_CONFIG, downloads.UNIDEPTH_V2_L)},
        "ViT-L 1.4 GB",
    ),
    "unidepth-v2-vitb14": (
        "lpiccinelli/unidepth-v2-vitb14",
        "6830c15e415da7f94babbe0e83b46e1a04bc28ea",
        {"config.json": "5f16045f79d323ea3d457b1b25eead9d9772e932df12bc6e5711fd78422db3cc",
         "model.safetensors": "871289bd6464fc1a8088fcd2a9bbe39f7746aa40cfa562c608f8e928bf053597"},
        "ViT-B 0.46 GB",
    ),
    "unidepth-v2-vits14": (
        "lpiccinelli/unidepth-v2-vits14",
        "038c238f06c87b6c2f5b3749fd51fbf442b1f218",
        {"config.json": "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe",
         "model.safetensors": "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"},
        "ViT-S 0.14 GB",
    ),
}



class UniDepth(Extension):
    name = "unidepth"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UniDepth V2"
    summary = "跨领域只凭单张画面预测真实尺度的三维点，自带一个可自我提示的相机模块；非商用"
    homepage = "https://github.com/lpiccinelli-eth/UniDepth"
    source = GitSource(url=UNIDEPTH_URL, commit=UNIDEPTH_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="CC-BY-NC-4.0（代码和权重，非商用）",
        url="https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE",
        summary=(
            "非商用：代码为 CC BY-NC 4.0（仓库 LICENSE 与源码文件头）；"
            "Hugging Face 上的 unidepth-v2-vitl14 / vitb14 / vits14 权重的模型卡没有写许可证，按仓库许可证 CC BY-NC 4.0 对待，"
            "结果不能用于商业项目。主干网络 DINOv2 结构来自 Meta（Apache-2.0），其权重已包含在 UniDepth 检查点里，不另外下载"
        ),
    )
    # Same torch as Video Depth Anything (shared uv cache); upstream asks for torch>=2.4 and runs unchanged on 2.9.
    # Attention goes through torch SDPA (xformers is optional upstream and not installed).
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128")
    weights = hf_weights(MODELS, lambda key: "CC BY-NC 4.0，非商用")


EXTENSION = UniDepth()
