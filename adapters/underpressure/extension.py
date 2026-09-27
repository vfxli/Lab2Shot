"""UnderPressure (InterDigital R&D France, SCA 2022): a small 1D convolutional network that estimates the vertical
ground reaction forces (vGRFs) under both feet from motion alone, derives reliable foot-contact labels from them, and
cleans footskate with an optimisation-based IK that keeps the contacts consistent with those forces.

Nothing is downloaded: the trained network is `pretrained.tar` inside the pinned repository (4 MB), and the reference
skeleton the motion is retargeted onto comes from the repository's own `footskate_samples/0.pt`. The environment is
torch alone — the whole project is numpy-free torch and the Python standard library (panda3d is only for its viewer,
which Lab2Shot does not use).
"""

from __future__ import annotations

from lab2shot.sdk import EnvSpec, Extension, GitSource, LicenseInfo, RESEARCH

UNDERPRESSURE_URL = "https://github.com/InterDigitalInc/UnderPressure.git"
UNDERPRESSURE_COMMIT = "e7ab73e7466444f262ec1c55f2f26afe51168796"  # 2023-05-14, the repository's latest


class UnderPressure(Extension):
    name = "underpressure"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "UnderPressure"
    summary = "从动捕动作估计两只脚的地面反作用力，据此判断脚接触，再用优化式 IK 清掉脚滑；仅限研究"
    homepage = "https://github.com/InterDigitalInc/UnderPressure"
    source = GitSource(url=UNDERPRESSURE_URL, commit=UNDERPRESSURE_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        name="InterDigital Limited Software Evaluation License",
        url="https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt",
        summary=("仅限研究。InterDigital 的评估许可只允许「fundamental research work」，明文排除一切商业用途"
                 "（包括放进任何提供给第三方的产品或服务里，不论收不收费）。发表论文要注明"
                 "「UnderPressure is an InterDigital product」并引用原文。代码和预训练网络都在这一条许可下，"
                 "Lab2Shot 不修改原仓库的任何文件。"),
    )
    import_repo = ""  # the repository's modules (anim, data, models, footskate, util) are imported from its own folder
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",  # RTX 4090 and RTX 5090 (sm_120) both run this build
        imports=("torch",),
    )
    weights = ()  # the trained network ships inside the repository (repo/pretrained.tar, 4 MB)


EXTENSION = UnderPressure()
