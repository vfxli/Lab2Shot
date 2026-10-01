"""Depth Anything 3 (ByteDance Seed): monocular metric depth per frame, and
multi-view reconstruction of a whole shot (consistent depth + camera per frame).

Code Apache-2.0. Weights: DA3NESTED-GIANT-LARGE-1.1 and DA3-LARGE-1.1 are
CC BY-NC 4.0 (non-commercial); DA3-BASE and DA3METRIC-LARGE are Apache-2.0.
"""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_weights, downloads

DA3_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"  # main

# checkpoint -> (Hugging Face repo, pinned revision, {file: sha256}, note).
# Not downloaded: DA3-GIANT-1.1 (its model.safetensors is byte-identical to the
# deprecated DA3-GIANT, sha256 1e47a083...; the retrained Giant 1.1 only ships inside
# DA3NESTED-GIANT-LARGE-1.1), DA3-SMALL (Apache-2.0, weakest), DA3MONO-LARGE
# (Apache-2.0, relative depth only). The nested checkpoint's metric branch is
# identical to DA3METRIC-LARGE.
MODELS = {
    "da3nested-giant-large-1.1": (  # Track4World's backbone too: pinned once (extensions/downloads.py)
        downloads.DA3NESTED_MODEL.repo,
        downloads.DA3NESTED_MODEL.revision,
        {d.filename: d.sha256 for d in (downloads.DA3NESTED_CONFIG, downloads.DA3NESTED_MODEL)},
        "Giant 1.1 + Metric-Large，1.40B 参数，6.8 GB",
    ),
    "da3-large-1.1": (
        "depth-anything/DA3-LARGE-1.1",
        "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        {"config.json": "744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5",
         "model.safetensors": "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"},
        "0.35B 参数，1.6 GB",
    ),
    "da3-base": (
        "depth-anything/DA3-BASE",
        "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        {"config.json": "5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5",
         "model.safetensors": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"},
        "0.12B 参数，0.54 GB",
    ),
    "da3metric-large": (  # ViPE's metric depth too: its model file pinned once (extensions/downloads.py)
        downloads.DA3METRIC_MODEL.repo,
        downloads.DA3METRIC_MODEL.revision,
        {"config.json": "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53",
         downloads.DA3METRIC_MODEL.filename: downloads.DA3METRIC_MODEL.sha256},
        "0.35B 参数，1.3 GB",
    ),
}
# 「模型」选哪些是非商用：节点的 OptionTrait 从这里生成（licence_traits）
OPTION_LICENCES = {"model": {"da3nested-giant-large-1.1": NONCOMMERCIAL, "da3-large-1.1": NONCOMMERCIAL}}


class DepthAnything3(Extension):
    name = "depthanything3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Depth Anything 3"
    summary = "从任意视觉输入预测空间一致的几何，相机位姿已知或未知都可以"
    homepage = "https://github.com/ByteDance-Seed/Depth-Anything-3"
    source = GitSource(url=DA3_URL, commit=DA3_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0（代码、DA3-BASE、DA3METRIC-LARGE）+ CC-BY-NC-4.0（DA3NESTED-GIANT-LARGE-1.1、DA3-LARGE-1.1，非商用）",
        url="https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE",
        summary=(
            "代码 Apache-2.0。DA3NESTED-GIANT-LARGE-1.1（模型卡 CC BY-NC 4.0）为非商用；"
            "DA3-LARGE-1.1 在官方 README 的模型表里是 CC BY-NC 4.0，但它的 Hugging Face 模型卡写着 Apache-2.0"
            "（同系列 DA3-LARGE 的模型卡 2025-11-19 已从 Apache-2.0 改成 CC BY-NC 4.0），按非商用对待；"
            "DA3-BASE 和 DA3METRIC-LARGE 为 Apache-2.0，可商用（二者组合的米制结果可用于商业项目）。"
            "作者声明所有模型只用公开学术数据集训练。另外 DA3-SMALL、DA3MONO-LARGE 也是 Apache-2.0，本扩展没有下载"
        ),
    )
    # the same torch as UniDepth (shared uv cache; pinned here too)
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128")
    weights = hf_weights(MODELS, lambda key: "CC BY-NC 4.0，非商用" if key in OPTION_LICENCES["model"] else "Apache-2.0")


EXTENSION = DepthAnything3()
