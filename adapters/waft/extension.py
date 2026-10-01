"""WAFT (Princeton Vision & Learning Lab, ICLR 2026): Warping-Alone Field Transforms, a RAFT-like optical-flow network
that warps high-resolution features instead of building a cost volume. Two frames in, the flow between them and a
per-pixel uncertainty out."""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

WAFT_URL = "https://github.com/princeton-vl/WAFT.git"
WAFT_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"  # 2026-03, waftv2 (the default branch; has the a1 models)

# The authors' Google Drive (model zoo, folder a1): tar-c-t.pth, the checkpoint the README recommends "for
# downstream applications" (WAFT-a1, Depth Anything V2 ViT-S features, trained TartanAir -> Chairs -> Things).
# Drive files carry no revision: the installer checks the sha256.
CHECKPOINT = "tar-c-t.pth"
CHECKPOINT_ID = "1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC"
CHECKPOINT_SHA256 = "9f4b24f48b3937eca690a12b73bc3190effde6d4d4c87db01998fe63d846397f"  # 257 MB


class Waft(Extension):
    name = "waft"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "WAFT"
    summary = "光流方法，和 RAFT 相似，但把代价体换成高分辨率的扭曲，精度更好、显存更省"
    homepage = "https://github.com/princeton-vl/WAFT"
    source = GitSource(url=WAFT_URL, commit=WAFT_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="BSD-3-Clause（代码）；权重没有写明许可（非商用）",
        url="https://github.com/princeton-vl/WAFT/blob/waftv2/LICENSE",
        summary=(
            "非商用：代码 BSD-3 本身可商用，但权重放在作者的 Google Drive 上，没有单独写许可证，而且是用 Sintel、KITTI、Spring、"
            "TartanAir 等只许研究使用的数据集训练的，所以按非商用处理。骨干 Depth Anything V2 ViT-S 本身是 Apache-2.0"
        ),
    )
    import_repo = ""  # WAFT's `model`, `utils` and `thirdparty` packages from the pinned repo
    env = EnvSpec(
        python="3.12",
        # Pure PyTorch (xformers is optional upstream and not installed): the same torch build as other extensions.
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
        pickled_checkpoints=True,  # tar-c-t.pth is a plain torch.save state dict (pinned, sha256-checked)
    )
    weights = (
        Weight(key="waft_a1_tar_c_t", kind="url",
               source=f"https://drive.usercontent.google.com/download?id={CHECKPOINT_ID}&export=download&confirm=t",
               dest=CHECKPOINT, sha256=CHECKPOINT_SHA256,
               note="WAFT-a1 tar-c-t（257 MB，作者 Google Drive；README 推荐用于实际应用）"),
    )


EXTENSION = Waft()
