"""StableMotion (Simon Fraser University + Electronic Arts, SIGGRAPH Asia 2025): a motion-cleanup diffusion model
trained on raw, unpaired mocap. It reads a whole window of motion, says frame by frame whether that frame is broken
(a quality label it predicts in an extra feature channel), and then inpaints exactly the broken frames while keeping
the good ones untouched.

Two things it needs are not in its git repository:

* the checkpoint (StableMotion-BrokenAMASS, `stablemotion_ckpt_seed3407.tar.gz`, 270 MB). Its authors put it on a
  OneDrive share, which hands the file out only after a browser handshake, so the installer cannot fetch it: it is a
  hand-downloaded item (lab2shot.extensions.manual "stablemotion") the user puts into downloads/.
* the normalizer: the per-channel statistics of the BrokenAMASS training features. Upstream produces them by
  preprocessing all of AMASS, which needs a registered AMASS download and the SMPL+H body model, and even then the
  corruption pass is seeded differently from the released one. The two vectors ship inside the upstream repository
  itself (`dataset/meta_.../mean.pt` / `std.pt`, 232 numbers each, checked into git), so the worker reads them from
  the pinned checkout; the 233rd (label) channel is 0.5 / 0.5, what upstream's own `add_label_channel` writes.

The model is a 233-channel DiT over 100-frame windows at 20 fps in the canonicalised global-SMPL-RIFKE representation
(232 motion features + 1 quality label); everything about that representation comes from upstream's own
`data_loaders.amasstools.globsmplrifke_feats`, never rewritten here.
"""

from __future__ import annotations

from lab2shot.sdk import EnvSpec, Extension, ExtensionWeights, GitSource, LicenseInfo, ManualItem, RESEARCH, manual_weight

STABLEMOTION_URL = "https://github.com/Murrol/StableMotion.git"
STABLEMOTION_COMMIT = "45d8836ce5cd7ae70f0065fdac672debfdfd6066"  # "Removed duplicate model"


# The weights are an archive on the authors' OneDrive that the program cannot download: the user places it in
# downloads/ and the core recognises it and installs it into this extension's weights/. The item is declared by this
# extension; the core names no assets and only provides the mechanism.
CHECKPOINT = ManualItem(
    key="stablemotion", title="StableMotion 权重", what="动捕清理模型 StableMotion-BrokenAMASS（270 MB 的压缩包）",
    page="https://github.com/Murrol/StableMotion#pretrained-checkpoint-stablemotion-brokenamass",
    download="README 里「Pretrained Checkpoint」那一节的 OneDrive 链接",
    filename="stablemotion_ckpt_seed3407.tar.gz",
    note="作者把权重放在 OneDrive 网盘，要浏览器点过才给文件，程序下不了；权重是在 AMASS 上训练的，只许学术研究",
    markers=("ema001000000.pt", "model001000000.pt"),
    install=ExtensionWeights("stablemotion", ("ema001000000.pt", "args.json")),
    alone=("ema001000000.pt",), looks_like=("*stablemotion*",), noncommercial=True)


class StableMotion(Extension):
    name = "stablemotion"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "StableMotion"
    summary = "用没有配对的坏数据训练动捕清理模型：一个既能判别又能生成的扩散模型，识别并修复坏掉的帧；仅限研究"
    homepage = "https://yxmu.foo/stablemotion-page/"
    source = GitSource(url=STABLEMOTION_URL, commit=STABLEMOTION_COMMIT)
    license = LicenseInfo(
        tag=RESEARCH,
        uses=("AMASS",),
        name="MIT（代码）+ AMASS 学术许可（权重）",
        url="https://github.com/Murrol/StableMotion/blob/main/LICENSE",
        summary=("仅限研究。代码是 MIT，可以随便用；但发布的权重 StableMotion-BrokenAMASS 是在 AMASS 上训练的，"
                 "AMASS 的许可只允许非商业的学术研究，所以这个节点按更严的一档标（宁严勿松）。"
                 "要商用就得按官方 README 说的，拿自己的动捕数据重新训练一个模型。"),
    )
    import_repo = ""  # upstream's packages (model, diffusion, utils, data_loaders) are imported from its own folder
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",  # RTX 4090 and RTX 5090 (sm_120) both run this build
        pickled_checkpoints=True,  # upstream's checkpoint is a pickled torch.save: loaded with weights_only off
        imports=("diffusers", "einops"),
    )
    # The normalisation statistics are not downloaded: the upstream repository contains `dataset/meta_.../mean.pt` and
    # `std.pt` (232 numbers each, checked into git); the 233rd is the label channel's 0.5, as upstream add_label_channel writes
    manual_items = (CHECKPOINT,)
    weights = (
        manual_weight(CHECKPOINT),  # the checkpoint archive: OneDrive, browser download only (see the module docstring)
    )


EXTENSION = StableMotion()
