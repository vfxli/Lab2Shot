"""Two-stage Transformer (Qin, Zheng, Zhou, SIGGRAPH Asia 2022): motion in-betweening with a Context Transformer and a
Detail Transformer on the LaFAN1 skeleton (22 joints, 30 fps). Deterministic: the same keys give the same motion.

The released weights are trained on LaFAN1 (Ubisoft, CC BY-NC-ND 4.0): research only. The LaFAN1 dataset itself is
installed too: the worker takes the skeleton (bone offsets) and a standing reference pose from it.
"""

from __future__ import annotations

from lab2shot.sdk import RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, Weight

TST_URL = "https://github.com/victorqin/motion_inbetweening.git"
TST_COMMIT = "fa9b6dc5f0791fd28bfccb6783e6bfd26d578515"  # 2023-06-07 (license added)
LAFAN1_COMMIT = "94084601bacdf9cc3764b5c73daaeccae6035fac"  # ubisoft-laforge-animation-dataset master


class TwoStageTransformer(Extension):
    name = "two_stage_transformer"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Two-stage Transformer"
    homepage = "https://github.com/victorqin/motion_inbetweening"
    source = GitSource(url=TST_URL, commit=TST_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,  # its weights are research only: stricter than 非商用 (nodes/tags.py)
        uses=("LaFAN1",),
        url="https://github.com/victorqin/motion_inbetweening/blob/master/LICENSE.txt",
    )
    generative = False
    import_repo = "packages"  # the motion_inbetween package, imported from the pinned repository
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",
        pickled_checkpoints=True,  # upstream's checkpoints hold the optimizer state next to the weights
    )
    weights = (
        Weight(key="tst", kind="zip", dest="experiments",
               source="https://github.com/victorqin/motion_inbetweening/releases/download/v1.0.0/pre-treained.zip",
               sha256="8445c588880bcbff091a614ef391fd9845dcb700f7a96e4d17160e656956058b"),
        Weight(key="lafan1", kind="zip", dest="lafan1",
               source=("https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/"
                       f"{LAFAN1_COMMIT}/lafan1/lafan1.zip"),
               sha256="ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977"),
    )


EXTENSION = TwoStageTransformer()
