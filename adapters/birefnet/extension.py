"""BiRefNet: high-resolution matte of the main subject(s) of each frame
("matte anything": people, objects, animals; no prompt, the model picks the subject).

All five checkpoints share one architecture (Swin-L backbone) and the official
model code at the pinned commit; only the weights differ.
"""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

BIREFNET_URL = "https://github.com/ZhengPeng7/BiRefNet.git"
BIREFNET_COMMIT = "ebcc0bc8ec7fe919cec829f2dea656b3078acddc"  # main

# model param -> (Hugging Face repo, pinned revision, LFS sha256 of model.safetensors,
# native input size: square side, or None = any shape (the frame's own size)).
# Every model card (README.md at these revisions) says license: mit. The matting
# checkpoint is stored in FP32 (885 MB), the others in FP16 (444 MB).
MODELS = {
    "general": (
        "ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154", 1024,
    ),
    "matting": (
        "ZhengPeng7/BiRefNet-matting", "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9",
        "a9875de5b1e6c8eb5fdaa8c727a82927ce442cdc87ba3abee6a77e6fa46c25bb", 1024,
    ),
    "hr": (
        "ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1",
        "9d678bafec0b0019fbb073b7fd02f05ede25dc4b15254f23b2fb0be333200c0d", 2048,
    ),
    "hr_matting": (
        "ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
        "a5a4de698739ea5e0e8bbab28e1b293dde95092b87a442d566cbc585c53cef55", 2048,
    ),
    "dynamic": (
        "ZhengPeng7/BiRefNet_dynamic", "280306042f57b7a33854319da62fd86aaa89ec4c",
        "e3d2e4884e51ff30f0cd630edc6b1e41b06b7f23a0a2a5169f7b7cb33a711c2d", None,
    ),
}


class BiRefNet(Extension):
    name = "birefnet"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "BiRefNet"
    summary = "用于高分辨率二分图像分割（DIS）的双边参考框架，由定位模块和带双边参考的还原模块组成"
    homepage = "https://github.com/ZhengPeng7/BiRefNet"
    source = GitSource(url=BIREFNET_URL, commit=BIREFNET_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT",
        url="https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE",
        summary=(
            "代码 MIT；五个权重（BiRefNet、BiRefNet-matting、BiRefNet_HR、BiRefNet_HR-matting、BiRefNet_dynamic）"
            "的 Hugging Face 模型卡均标 MIT，可商用。注意：训练数据含 DIS5K、P3M-10k、AM-2k、Distinctions-646 等"
            "仅限学术研究的数据集，作者仍以 MIT 发布权重；严格的商业项目交付前建议法务确认。"
            "（briaai 的 RMBG-2.0 同为 BiRefNet 结构但权重非商用，本扩展不下载）"
        ),
    )
    env = EnvSpec(
        python="3.11",
        # Upstream asks for torch>=2.5; the model code (Swin-L, torchvision deform_conv2d,
        # SDPA) runs unchanged on 2.9.0.
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu128",
    )
    weights = tuple(
        hf_file(
            repo, revision, "model.safetensors",
            key=key,
            dest=f"{repo}/model.safetensors",
            note=f"{repo}（MIT，Swin-L，{'任意尺寸' if size is None else f'{size}×{size}'}）",
            sha256=sha256,  # the LFS object: the URL is pinned to a revision, the bytes are checked too
        )
        for key, (repo, revision, sha256, size) in MODELS.items()
    )


EXTENSION = BiRefNet()
