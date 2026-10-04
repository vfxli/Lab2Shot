"""Meta SAM 3: video segmentation + tracking from a text concept or per-person boxes."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, InstallError, LicenseInfo, Msg, Weight

SAM3_URL = "https://github.com/facebookresearch/sam3.git"
# main: SAM 3 + SAM 3.1 code, close_session memory fixes.
SAM3_COMMIT = "660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7"

# Two checkpoints, one per prompt type:
# - text concept: SAM 3.1 (facebook/sam3.1, Object Multiplex) detector + tracker video model, FlashAttention 3 off
#   (use_fa3=False: FA3 is Hopper-only; this runs on Ada / Blackwell). Same SAM License file as sam3.
# - per-person boxes: SAM 3 (Nov 2025) tracker alone (SAM 2 style instance tracking): SAM 3.1 has no
#   box-per-object prompt (a box there is an example of a kind of object, not "this person").
CHECKPOINT = "sam3/sam3.pt"
CHECKPOINT_31 = "sam3.1/sam3.1_multiplex.pt"


class Sam3(Extension):
    name = "sam3"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "SAM 3"
    homepage = "https://github.com/facebookresearch/sam3"
    source = GitSource(url=SAM3_URL, commit=SAM3_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/facebookresearch/sam3/blob/main/LICENSE",
    )
    generative = False
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
        ),
        Weight(
            key="sam31",
            kind="hf",
            source="facebook/sam3.1",
            revision="daa63191845a41281374e725f4c9e51c7a824460",
            dest="sam3.1",
            files=("sam3.1_multiplex.pt", "LICENSE"),
            gated=True,
        ),
    )


    def post_install(self, run, paths) -> None:
        for name in (CHECKPOINT, CHECKPOINT_31):
            ckpt = paths.weights / name
            if not ckpt.is_file() or ckpt.stat().st_size < 3_000_000_000:
                raise InstallError(Msg("E-SAM3-WEIGHTSINCOMPLETE", path=str(ckpt)))
            # The SAM License must travel with the weights (and with anything built on them).
            if not (ckpt.parent / "LICENSE").is_file():
                raise InstallError(Msg("E-SAM3-NOLICENSE"))


EXTENSION = Sam3()
