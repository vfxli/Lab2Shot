"""Sketch2Anim (Lei Zhong et al., ACM TOG 44(4) / SIGGRAPH 2025): a storyboard drawn as a stick figure plus a root
path becomes a full 3D body animation.

The released code (verified against the repository, not the paper) is a MotionLCM / MLD latent
diffusion model over HumanML3D motions with a ControlNet conditioned on the 2D projection of a
trajectory, a pose-aware denoiser conditioned on the 2D projection of one key pose, and a sentence-T5
text embedding. Inference is plain PyTorch: `bpy` appears only in `mld/render/blender/`, the optional video
renderer, which is never called here, so Blender is not required. The Blender plugin of the project page is not released
(Readme "TODO List": Blender Plugin, evaluation code and training are all still unticked; inference and the
pretrained model are ticked and are what this adapter uses).

The model's own conventions, which the worker converts from: 22 HumanML3D joints (the SMPL body joints), metres,
Y up, 20 fps, at most 196 frames. The 2D sketch lives in the same metric world seen through an orthographic
camera rotated by Rx(20°)·Ry(30°) (upstream `utils.py project2D`), which is why the node's canvas coordinates are
fitted to a standing body's size before they are handed over.
"""

from __future__ import annotations

from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight

# The project-page repository is named Sketch2Anim (a fork of the nerfies template); the code is in Sketch2Animation.
# The URL must not be derived from the paper title.
SKETCH2ANIM_URL = "https://github.com/zhongleilz/Sketch2Animation.git"
SKETCH2ANIM_COMMIT = "5b781ee253437ad4aafe8ebf70aee5b111175b86"  # "Modify some codes" (latest commit)
# The official weights are in a public Google Drive folder (the Readme's PRETRAINED_WEIGHTS link); there is no HF mirror
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
T5 = ("sentence-transformers/sentence-t5-large", "1b36eb48a1df42a07ffd02234d25abfbf3e3f3cb")
T5_FILES = ("config.json", "config_sentence_transformers.json", "modules.json", "sentence_bert_config.json",
            "special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json",
            "model.safetensors", "1_Pooling/config.json", "2_Dense/config.json",
            # the Dense module of sentence-transformers 2.7.0 reads only pytorch_model.bin, not safetensors
            "2_Dense/pytorch_model.bin")


class Sketch2Anim(Extension):
    name = "sketch2anim"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Sketch2Anim"
    summary = "把分镜草图变成三维动画：用关键姿势、关节轨迹和动作描述控制三维动作生成，二维草图经神经映射对到三维"
    homepage = "https://zhongleilz.github.io/Sketch2Anim/"
    source = GitSource(url=SKETCH2ANIM_URL, commit=SKETCH2ANIM_COMMIT)
    license = LicenseInfo(
        tag=NONCOMMERCIAL,
        name="MIT（代码）+ HumanML3D / AMASS 非商用（权重）",
        url="https://github.com/zhongleilz/Sketch2Animation/blob/main/LICENSE",
        summary=(
            "非商用。代码是 MIT（作者 Lei Zhong 2025，原文允许商用、允许修改和再发布），但官方预训练权重用 HumanML3D 训练，"
            "HumanML3D 的动作来自 AMASS，AMASS 的许可只许学术研究和非商业用途，所以权重和生成出来的动作只能用于研究和评估。"
            "文字编码用的 sentence-t5-large 是 Apache-2.0（可商用），不改变上面的结论。要商用得用有授权的动捕数据重新训练"
        ),
    )
    import_repo = ""  # upstream is not a package: the worker puts the repository root on sys.path (top-level mld / common / visualization)
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",
        pickled_checkpoints=True,  # the official .ckpt is a Lightning checkpoint with optimizer state, not a plain state_dict
    )
    weights = (
        Weight(key="adapter", kind="url", source=DRIVE.format("161wMAVzFCnqzPHsSaRgsZqWob0uEyMvM"),
               dest="checkpoints/adapter.ckpt",
               sha256="5f9b3a0e69aad4240a9395956cf3614eeaa143a45fe7c0411c37b6c515b8944c",
               note="Sketch2Anim 官方权重 adapter.ckpt（HumanML3D 训练，非商用），357 MB"),
        Weight(key="pretrain", kind="url", source=DRIVE.format("1XLVncH3Ed7G7sW7HzOdL7xA9Ou7G_Gfm"),
               dest="checkpoints/pretrain_22joint_combine_adapter.ckpt",
               sha256="251ec98958c2170741444d26c3afc71d917e4c9ad7fc1d8d67ee37ca82d5207b",
               note="Sketch2Anim 官方发布的预训练底座 pretrain_22joint_combine_adapter.ckpt（非商用），303 MB；推理只加载 adapter.ckpt，这一份不读"),
        Weight(key="t5", kind="hf", source=T5[0], revision=T5[1], dest="sentence-t5-large", files=T5_FILES,
               note="sentence-t5-large（Apache-2.0）：把那句英文描述编码成条件向量，1.3 GB"),
    )

    def worker_env(self) -> dict[str, str]:
        return {"HF_HUB_OFFLINE": "1"}  # the sentence model is read from the installed folder only, without network access


EXTENSION = Sketch2Anim()
