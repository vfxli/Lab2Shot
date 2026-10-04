"""Depth Anything 3 (ByteDance Seed): monocular metric depth per frame, and
multi-view reconstruction of a whole shot (consistent depth + camera per frame).

Code Apache-2.0. Weights: DA3NESTED-GIANT-LARGE-1.1 and DA3-LARGE-1.1 are
CC BY-NC 4.0 (non-commercial); DA3-BASE and DA3METRIC-LARGE are Apache-2.0.
"""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_weights

DA3_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"  # main

# checkpoint -> (Hugging Face repo, pinned revision, {file: sha256}); what each is: extension.depthanything3.weight.<checkpoint>.note.
# Not downloaded: DA3-GIANT-1.1 (its model.safetensors is byte-identical to the
# deprecated DA3-GIANT, sha256 1e47a083...; the retrained Giant 1.1 only ships inside
# DA3NESTED-GIANT-LARGE-1.1), DA3-SMALL (Apache-2.0, weakest), DA3MONO-LARGE
# (Apache-2.0, relative depth only). The nested checkpoint's metric branch is
# identical to DA3METRIC-LARGE.
MODELS = {
    "da3nested-giant-large-1.1": (  # Track4World's backbone too (it pins the same files in its own declaration)
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        "b2359bdf726fb44ef62acca04d629dcf158053e7",
        {"config.json": "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468",
         "model.safetensors": "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c"},
    ),
    "da3-large-1.1": (
        "depth-anything/DA3-LARGE-1.1",
        "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c",
        {"config.json": "744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5",
         "model.safetensors": "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"},
    ),
    "da3-base": (
        "depth-anything/DA3-BASE",
        "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
        {"config.json": "5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5",
         "model.safetensors": "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"},
    ),
    "da3metric-large": (  # ViPE's metric depth too (it pins the same model file in its own declaration)
        "depth-anything/DA3METRIC-LARGE",
        "4010e39f3634a45bc60553321fb49fb760bd594e",
        {"config.json": "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53",
         "model.safetensors": "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"},
    ),
}
# 「模型」选哪些是非商用：节点的 OptionTrait 从这里生成（licence_traits）
OPTION_LICENCES = {"model": {"da3nested-giant-large-1.1": NONCOMMERCIAL, "da3-large-1.1": NONCOMMERCIAL}}


class DepthAnything3(Extension):
    name = "depthanything3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Depth Anything 3"
    homepage = "https://github.com/ByteDance-Seed/Depth-Anything-3"
    source = GitSource(url=DA3_URL, commit=DA3_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE",
    )
    generative = False
    # the same torch as UniDepth (shared uv cache; pinned here too)
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128")
    weights = hf_weights(MODELS)


EXTENSION = DepthAnything3()
