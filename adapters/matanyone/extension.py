"""MatAnyone 2 (S-Lab, NTU): temporally stable video matting of people, guided by
a mask on one frame, propagated through the shot with a memory of past frames."""

from __future__ import annotations


from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

MATANYONE2_URL = "https://github.com/pq-yang/MatAnyone2.git"
MATANYONE2_COMMIT = "0079197acd6d16a741f71558809c06c586c579e0"  # main

# The checkpoint the official inference script downloads (GitHub release v1.0.0).
CHECKPOINT = "matanyone2.pth"
CHECKPOINT_URL = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
CHECKPOINT_SHA256 = "5e9821e4087231427376b437c85bb6e072b41e582314f06fd524f75bc4af5914"


class MatAnyone(Extension):
    name = "matanyone"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "MatAnyone 2"
    summary = (
        "用一个学出来的抠像质量评估器（MQE），在没有真值的情况下判断 alpha 的语义和边界质量，"
        "把视频抠像的数据规模做上去"
    )
    homepage = "https://github.com/pq-yang/MatAnyone2"
    source = GitSource(url=MATANYONE2_URL, commit=MATANYONE2_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="NTU S-Lab License 1.0（非商用）",
        url="https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt",
        summary=(
            "非商用：代码和权重（matanyone2.pth，作者 GitHub 发布页）都按 S-Lab License 1.0 发布，"
            "只允许非商业用途的使用和再分发（须保留版权声明和免责声明），商用须联系作者（南洋理工 S-Lab）另行授权。"
            "模型结构基于 Cutie（MIT）。不需要其他权重：ResNet 主干的 ImageNet 预训练权重已包含在 checkpoint 里，不另外下载"
        ),
    )
    import_repo = ""
    env = EnvSpec(
        python="3.12",
        # Upstream's uv.lock: torch 2.10.0 from the cu128 index (includes sm_89). No compiled ops.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="matanyone2",
            kind="url",
            source=CHECKPOINT_URL,
            dest=CHECKPOINT,
            note="MatAnyone 2 权重（135 MB，S-Lab License 1.0，非商用）",
            sha256=CHECKPOINT_SHA256,
        ),
    )


EXTENSION = MatAnyone()
