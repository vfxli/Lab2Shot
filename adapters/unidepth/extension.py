"""UniDepth V2 (ETH Zurich): per-frame monocular metric depth, point map and
camera intrinsics; accepts a known focal length as a condition.
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_weights

UNIDEPTH_URL = "https://github.com/lpiccinelli-eth/UniDepth.git"
UNIDEPTH_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"  # 2025-05-18 (main)

# model param -> (Hugging Face repo, pinned revision, {file: sha256}, note). The model cards
# carry no licence of their own: the repository's CC BY-NC 4.0 applies.
MODELS = {
    "unidepth-v2-vitl14": (
        "lpiccinelli/unidepth-v2-vitl14",  # ViPE runs this model too and pins the same files in its own declaration
        "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9",
        {"config.json": "09eb0ea8de53a6c9a1d428ac98c79847fe2602ea417701261f3f628099a30816",
         "model.safetensors": "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6"},
    ),
    "unidepth-v2-vitb14": (
        "lpiccinelli/unidepth-v2-vitb14",
        "6830c15e415da7f94babbe0e83b46e1a04bc28ea",
        {"config.json": "5f16045f79d323ea3d457b1b25eead9d9772e932df12bc6e5711fd78422db3cc",
         "model.safetensors": "871289bd6464fc1a8088fcd2a9bbe39f7746aa40cfa562c608f8e928bf053597"},
    ),
    "unidepth-v2-vits14": (
        "lpiccinelli/unidepth-v2-vits14",
        "038c238f06c87b6c2f5b3749fd51fbf442b1f218",
        {"config.json": "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe",
         "model.safetensors": "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"},
    ),
}



class UniDepth(Extension):
    name = "unidepth"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UniDepth V2"
    homepage = "https://github.com/lpiccinelli-eth/UniDepth"
    source = GitSource(url=UNIDEPTH_URL, commit=UNIDEPTH_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE",
    )
    generative = False
    # Same torch as Video Depth Anything (shared uv cache); upstream asks for torch>=2.4 and runs unchanged on 2.9.
    # Attention goes through torch SDPA (xformers is optional upstream and not installed).
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128")
    weights = hf_weights(MODELS)


EXTENSION = UniDepth()
