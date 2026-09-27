"""Meta SAM 3: video segmentation + tracking from a text concept or per-person boxes."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, InstallError, LicenseInfo, Msg, Weight

SAM3_URL = "https://github.com/facebookresearch/sam3.git"
# main: SAM 3 + SAM 3.1 code, close_session memory fixes.
SAM3_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"

# SAM 3 (Nov 2025) checkpoint: detector + tracker in one file. It serves both
# prompt types: the text concept runs the full detector + tracker video model,
# per-person boxes run SAM 3's tracker alone (SAM 2 style instance tracking).
# SAM 3.1 (facebook/sam3.1, Object Multiplex) is not used: its predictor wants
# FlashAttention 3 (Hopper only) and has no box-per-object prompt.
CHECKPOINT = "sam3/sam3.pt"


class Sam3(Extension):
    name = "sam3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SAM 3"
    summary = "可提示分割的统一基础模型，画面和视频都管：用文字，或点、框、遮罩这类视觉提示，检出、分割并跟住物体"
    homepage = "https://github.com/facebookresearch/sam3"
    source = GitSource(url=SAM3_URL, commit=SAM3_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="SAM License",
        url="https://github.com/facebookresearch/sam3/blob/main/LICENSE",
        summary=(
            "代码和权重同一份 SAM License：允许商用、修改和再分发（再分发须附带许可证原文）；"
            "发表论文须注明使用了 SAM 3；禁止军事、战争、核工业、间谍、武器等用途，须遵守美国等出口管制和制裁；"
            "不得逆向工程；对 Meta 提起知识产权诉讼则许可自动终止；无任何担保。"
            "权重在 Hugging Face 上需先申请访问"
        ),
    )
    import_repo = ""
    env = EnvSpec(
        python="3.12",
        # Upstream README's torch; cu128 wheels include sm_89 (RTX 4090). No compiled ops.
        torch=("torch==2.10.0", "torchvision==0.25.0"),
        torch_backend="cu128",
    )
    weights = (
        Weight(
            key="sam3",
            kind="hf",
            source="facebook/sam3",
            # 装好并自检过的快照
            revision="3c879f39826c281e95690f02c7821c4de09afae7",
            dest="sam3",
            files=("sam3.pt", "LICENSE"),
            gated=True,
            note="SAM 3 检测+跟踪权重（3.4 GB，SAM License）；需先在 https://huggingface.co/facebook/sam3 申请访问",
        ),
    )


    def post_install(self, run, paths) -> None:
        ckpt = paths.weights / CHECKPOINT
        if not ckpt.is_file() or ckpt.stat().st_size < 3_000_000_000:
            raise InstallError(Msg("E-SAM3-WEIGHTSINCOMPLETE", path=str(ckpt)))
        # The SAM License must travel with the weights (and with anything built on them).
        if not (paths.weights / "sam3" / "LICENSE").is_file():
            raise InstallError(Msg("E-SAM3-NOLICENSE"))


EXTENSION = Sam3()
