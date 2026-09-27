"""Google DeepMind TAPNext++: long-range online 2D point tracking (any point,
through occlusions, re-detection after the point comes back into view)."""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

TAPNET_URL = "https://github.com/google-deepmind/tapnet.git"
TAPNET_COMMIT = "c2cbab81cc06092b5f05bfe2da7bfec54e2079c9"  # TAPNext++ 512 checkpoint support

# The two TAPNext++ PyTorch checkpoints linked from tapnet/tapnextpp/README.md and
# colabs/torch_tapnextpp_demo.ipynb. Google Cloud Storage URLs carry no revision,
# so the installer checks the sha256 (GCS md5 pjiOC5EfLjZE+vqKmhGN+w== / 7qT/+gQ/TyhQPBMNgmsRbA==).
# model -> (file in weights/, URL, bytes, sha256, input resolution it was fine-tuned at)
CHECKPOINTS = {
    "512": (
        "tapnextpp_512.ckpt",
        "https://storage.googleapis.com/gresearch/tapnextpp/tapnextpp_512.ckpt",
        2532283010,
        "6cd0e793fdcface3063d63f8ed3819bcf74c2c0468fe1fef85acee4de2f3609f",
        512,
    ),
    "256": (
        "tapnextpp_ckpt.pt",
        "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt",
        2532282370,
        "cb96a43444ccb4fbdb25d800b88c7ba196179a526e78f01b021a16b1c1eff6da",
        256,
    ),
}


class TAPNext(Extension):
    name = "tapnext"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "TAPNext++"
    summary = ("把「跟住任意一个点」写成「预测下一个 token」；TAPNext++ 微调后稳定跟点的时长长 40 倍，"
               "能跟过遮挡")
    homepage = "https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapnextpp"
    source = GitSource(url=TAPNET_URL, commit=TAPNET_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="Apache-2.0",
        url="https://github.com/google-deepmind/tapnet/blob/main/LICENSE",
        summary=(
            "代码 Apache-2.0，可商用；TAPNext++ 两个权重（tapnextpp_512.ckpt、tapnextpp_ckpt.pt）"
            "按仓库 README 同样是 Apache-2.0，可商用。只用到仓库里的 PyTorch 推理代码（torch、torchvision、einops），"
            "不装 JAX / recurrentgemma；训练数据集（Kubric、PointOdyssey）不随扩展分发"
        ),
    )
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (no compiled ops): the same torch build as other extensions
        # (shared uv cache); sm_89 kernels are in the cu130 wheels.
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
    )
    weights = tuple(
        Weight(key=f"tapnextpp_{model}", kind="url", source=url, dest=filename,
               note=f"TAPNext++ {res}x{res} 输入（Apache-2.0，{size / 1e9:.1f} GB）", sha256=sha256)
        for model, (filename, url, size, sha256, res) in CHECKPOINTS.items()
    )


EXTENSION = TAPNext()
