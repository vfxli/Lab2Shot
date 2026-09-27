"""SegAnyMo (Segment Any Motion in Videos, CVPR 2025): masks of what really moves in the shot."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, CUDA_13_2_TOOLKIT, EnvSpec, Extension, GitSource, LicenseInfo, Weight, hf_file

# main (the last commit: README). The repo carries its own SAM 2 copy (sam2/, PNG frames allowed) and
# its own PyTorch TAPIR (preproc/tapnet_torch); neither submodule it records is needed at run time.
SEGANYMO = GitSource(url="https://github.com/nnanhuang/SegAnyMo.git", commit="6fe8fe2874ccb47169d87d78ea3b2a59e203e5ac")
# DINOv2 network code for the DINO features (upstream loads it through torch.hub from GitHub at run time): the commit
# SegAnyMo records for its preproc/dinov2 submodule, pinned and local.
DINOV2 = GitSource(url="https://github.com/facebookresearch/dinov2.git", commit="e1277af2ba9496fbadf7aec6eba56e8d882d1e35")

MOSEG_REPO, MOSEG_REVISION = "Changearthmore/moseg", "5ce31c82a80df8ec00ed1e7b1baee72d2b3eecc5"
SAM2_REPO, SAM2_REVISION = "facebook/sam2-hiera-large", "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251"
DAV2_REPO, DAV2_REVISION = "depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"
DAV2_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


class SegAnyMo(Extension):
    name = "seganymo"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SegAnyMo"
    summary = "运动物体分割：把长距离轨迹的运动线索和 DINO 的语义特征结合起来，再用 SAM2 把结果加密到像素级"
    homepage = "https://motion-seg.github.io/"
    source = SEGANYMO
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="MIT",
        url="https://github.com/nnanhuang/SegAnyMo/blob/main/LICENSE",
        summary=(
            "代码 MIT，可以商用；运动分类模型 moseg.pth 由作者随代码发布（Hugging Face 上没有另写许可证）。"
            "用到的其他模型也都可以商用：SAM 2 和 DINOv2 是 Apache-2.0，BootsTAPIR 代码 Apache-2.0、权重 CC-BY-4.0（署名 Google DeepMind），"
            "深度用 Depth Anything V2 Small（Apache-2.0），不用原作者的 Large（CC-BY-NC）。"
            "训练数据有 Kubric、Dynamic Replica、HOI4D、Waymo 等研究数据集，严格的商业交付前建议做一次法务确认"
        ),
    )
    # SAM 2 的 `_C`（填洞去斑用的连通域核）必须在这个环境里编出来（build_sam2.py）：SAM 2 视频那条路默认开着
    # 填洞（`repo/sam2/sam2/build_sam.py` 的 `++model.fill_hole_area=8`），核加载不上时 `sam2/utils/transforms.py`
    # 的 `except` 原样返回、只打一句警告，静默降质：遮罩看着正常但没填洞。一个在旧 torch 下编好、留在检出里的
    # `_C` 换了 torch 之后会 `undefined symbol`，两张卡都加载不上，所以换 torch 必须重编。
    #
    # 为什么要 cu130：这台机器的 glibc >= 2.43，只有 CUDA 13.2 的头文件编得过
    # （`lab2shot/extensions/spec.py` 的 CUDA_13_2_TOOLKIT 那段）；而 torch 的 cpp_extension 要求
    # nvcc 的 CUDA 版本和编译 torch 用的对得上，所以 torch 也跟到 cu130。和 TRAM 同一条路。
    # 走不通的两条：系统 CUDA 12.9（gcc 15 超上限；指到 gcc-14 又撞 glibc 的
    # `rsqrt`/`cospi`/`sinpi` exception specification）、pip 的 CUDA 12.9（那个 wheel 里只有 ptxas，没有 nvcc）。
    #
    # `build_sam2.py` 里那条 `SAM2_BUILD_ALLOW_ERRORS=0` 不许改回去：SAM 2 自己的默认值是 1，编不过就当没事
    # 继续装，装完看着没问题、组件却缺了。
    env_archs = ("sm_89", "sm_120")  # Ada and Blackwell: third_party/seganymo/.venv-ada-blackwell
    env = EnvSpec(
        python="3.12",
        # cu130（不是 cu128）：SAM 2 的 `_C` 要在这台机器上编出来，见 env_archs 前面的说明
        torch=("torch==2.9.0", "torchvision==0.24.0"),
        torch_backend="cu130",
        cuda_toolkit=CUDA_13_2_TOOLKIT,  # 环境自己的 nvcc，不碰机器上的工具链
        build="build_sam2.py",           # 编 SAM 2 的 _C，放回检出（见那个脚本的说明）
        places=("repo/sam2/sam2/_C.so",),
    )
    extra_sources = {"dinov2": DINOV2}
    weights = (
        hf_file(MOSEG_REPO, MOSEG_REVISION, "moseg.pth", key="moseg", sha256="705445fbf6b2e42cd4e946390c5c6891046e70add8e8377e38f5368f503591d9",
                note="运动分类模型 moseg（14 MB，随 MIT 代码发布）"),
        hf_file(SAM2_REPO, SAM2_REVISION, "sam2_hiera_large.pt", key="sam2_hiera_large",
                sha256="7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b",
                note="SAM 2 Hiera-L（0.9 GB，Apache-2.0）：把运动的点连成整块物体遮罩并整段跟踪"),
        Weight(key="bootstapir", kind="url", source="https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt",
               dest="bootstapir/bootstapir_checkpoint_v2.pt", sha256="8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b",
               note="BootsTAPIR 点跟踪（0.2 GB，CC-BY-4.0）"),
        # torch.hub's own checkpoint folder under TORCH_HOME: the DINOv2 hub entry finds it there instead of downloading
        Weight(key="dinov2_vitb14", kind="url", source="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
               dest="torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
               sha256="0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73", note="DINOv2 ViT-B/14（0.3 GB，Apache-2.0）"),
        *(hf_file(DAV2_REPO, DAV2_REVISION, f, key=f"dav2_small_{f.split('.')[0]}",
                  sha256="3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70" if f == "model.safetensors" else "",
                  note="Depth Anything V2 Small（0.1 GB，Apache-2.0）：运动分类用的深度" if f == "model.safetensors" else "")
          for f in DAV2_FILES),
    )

    def worker_env(self) -> dict[str, str]:
        # the SAM 2 copy is the package folder sam2/sam2 (with its hydra configs next to it in sam2/sam2_configs)
        return {"PYTHONPATH": f"{self.paths.repo}:{self.paths.repo / 'sam2'}:{self.paths.repo / 'preproc'}",
                "LAB2SHOT_DINOV2_DIR": str(self.paths.root / "dinov2")}


EXTENSION = SegAnyMo()
