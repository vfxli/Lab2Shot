"""UniK3D (ETH Zurich): per-frame monocular metric 3D for any lens (pinhole,
fisheye, 360°): per-pixel camera rays + distance -> point map, depth, and the
network's own spherical-harmonics camera model.

Same author and dependency list as UniDepth: the same environment (its own copy of
UniDepth's pins and requirements).
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_weights

UNIK3D_URL = "https://github.com/lpiccinelli-eth/UniK3D.git"
UNIK3D_COMMIT = "29f7862f3c3ee79c53ef0153dfc901ce511df575"  # 2025-09-14 (main)

# model param -> (Hugging Face repo, pinned revision, {file: sha256}, note).
# The model cards carry no licence of their own: the repository's CC BY-NC-SA 4.0 applies.
MODELS = {
    "unik3d-vitl": (
        "lpiccinelli/unik3d-vitl",
        "008aa54308b8bc142f74c9c7b098833aaede06bd",
        {"config.json": "2cb006c03081f593b31d90238891cd5e0def149f9edfe7fa033be22588645b21",
         "model.safetensors": "890be79699dd68abb3c9a3aab24b528bf09d97eca4dd073e3e83c0a2028566b2"},
    ),
    "unik3d-vitb": (
        "lpiccinelli/unik3d-vitb",
        "c65bb3a8a4fa981a06ee9286ce847aebe7f66fe2",
        {"config.json": "28d8451ac80ad237f3773579d2202cebd9767701fa087c72dc70caad9188f5d4",
         "model.safetensors": "6fe560cd967256c73232c64d2fec53ea92b99e5bbc89e86f64131b40660f424f"},
    ),
    "unik3d-vits": (
        "lpiccinelli/unik3d-vits",
        "80a99bfeb8d983ca6af27125fd83c88b891e75ad",
        {"config.json": "5b5b7d5c6884fdb1c5f2fbf58dfaa6beb81d5ccff3dbcead4632f81eb32ad131",
         "model.safetensors": "9f977d058961146582931c136cf2ce44964cff3153dae7823dd1209223489bc6"},
    ),
}


class UniK3D(Extension):
    name = "unik3d"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UniK3D"
    homepage = "https://github.com/lpiccinelli-eth/UniK3D"
    source = GitSource(url=UNIK3D_URL, commit=UNIK3D_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        url="https://github.com/lpiccinelli-eth/UniK3D/blob/main/LICENSE",
    )
    generative = False
    # the same torch as UniDepth (shared uv cache; pinned here too)
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128")
    weights = hf_weights(MODELS)


EXTENSION = UniK3D()
