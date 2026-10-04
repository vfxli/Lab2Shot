"""Kimodo (NVIDIA, 2026): kinematic motion diffusion trained on 700 hours of licensed optical motion capture (Bones
Rigplay), controlled by full-body keyframes, end-effector targets and optional text. Lab2Shot uses all seven released
weights (30 fps, Y up, metres) on three skeletons: SOMA 30 joints (four weights), SMPL-X 22 joints, Unitree G1 34
joints. With an animation wired in the keys are the constraint and the motion comes back on that rig; with nothing
wired the constraints are empty — what upstream's `constraint_lst` documents as "unconstrained generation" — and the
node gives a fresh skeletal animation on the model's own skeleton.

Text is optional: without a prompt Kimodo generates from the keys alone (its benchmark's "constraints without text"
cases), and the text encoder is never loaded. With a prompt, LLM2Vec on Meta Llama 3 8B Instruct (gated on Hugging
Face) encodes it once on the CPU (TEXT_ENCODER_DEVICE=cpu, about 16 GB of memory, as Kimodo supports it); the node
caches the embedding by the prompt, so the same prompt is never encoded twice.
"""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, RESEARCH, EnvSpec, Extension, GitSource, LicenseInfo, Weight

from .bodies import BODIES

KIMODO_URL = "https://github.com/nv-tlabs/kimodo.git"
KIMODO_COMMIT = "1aece8c124d73d255ceff5086d983b844c9f4e94"  # main (README with ARDY)
# 官方放出来的七个权重，三副骨架（节点的「模型」就是这七个选项）。每个一行：参数值 -> (Hugging Face 仓库, 锁定的提交,
# 骨架)；说明在 extension.kimodo.weight.<key>.note，面板上的名字在 node."kimodo.motion".param.model.option.<值>。这是核心这边唯一的一张表：安装（下面的 weights）、节点的「模型」选项与骨架（nodes.py）都从它来；
# worker 不查表，节点把权重的文件夹名随任务交给它（nodes.py prepare 的 checkpoint）。
# 权重的 key（下面 KEYS）是记在硬盘上的身份：`third_party/<项目>/install_state.json` 按 key 记「这个权重装好了」，
# 改了 key 等于把已经装好的权重变成「没下载」，页面上 Kimodo 会变成「权重没下载完」、节点不可用。已经在用的 key 不许改。
MODELS = {
    "rp": ("nvidia/Kimodo-SOMA-RP-v1.1", "6c9233af1180b8151e3c4703477104af5dce9dd5", "soma"),
    "seed": ("nvidia/Kimodo-SOMA-SEED-v1.1", "aae3af194322c60d21bc44062b64c3fec912be50", "soma"),
    "rp_v1": ("nvidia/Kimodo-SOMA-RP-v1", "defbe1f34f5fd031eb0ccfe19f3f92a9880c287b", "soma"),
    "seed_v1": ("nvidia/Kimodo-SOMA-SEED-v1", "388f2be4dc9f1a778c73697e6fb68b5ddb57b828", "soma"),
    "smplx": ("nvidia/Kimodo-SMPLX-RP-v1", "1419ba56b734c48bbafb41fefa84088ca94583b5", "smplx"),
    "g1": ("nvidia/Kimodo-G1-RP-v1", "3020ad8c419c244e0429d360163730c63c4ed011", "g1"),
    "g1_seed": ("nvidia/Kimodo-G1-SEED-v1", "5e6f2c7e18c2ab834c8d7983b9dcce701e5c6097", "g1"),
}
GATED = {"smplx"}  # 受限仓库：要用户本人在 Hugging Face 页面点同意
# 七个权重的许可不一样（每个仓库模型卡 `third_party/kimodo/weights/<仓库>/README.md` 第 2–4 行的 `license_name`）：
# 六个是 `nvidia-open-model-license`（可商用），只有 Kimodo-SMPLX-RP-v1 是
# `nvidia-internal-scientific-research-and-development-model-license`（只限研究）。许可只是标签，如实标出，不因此少接一档。
# 「模型」的哪些取值受许可限制、到哪一档（nodes/tags.py 的等级）：SMPL-X 权重只限研究。节点的 OptionTrait 从这里生成
OPTION_LICENCES = {"model": {"smplx": RESEARCH}}
# 每个权重在安装记录里的名字：一旦有人装过就不能再改（见 MODELS 上面那段说明）
KEYS = {"rp": "soma-rp", "seed": "soma-seed", "rp_v1": "soma-rp-v1", "seed_v1": "soma-seed-v1",
        "smplx": "smplx-rp-v1", "g1": "g1-rp-v1", "g1_seed": "g1-seed-v1"}
# the text encoder Kimodo's load_model builds (TEXT_ENCODER_PRESETS["llm2vec"]): two LLM2Vec adapters on Llama 3
LLM2VEC = {"mntp": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
           "supervised": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7")}
LLAMA = ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2")
LLAMA_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "LICENSE", "USE_POLICY.md", *(f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)))


class Kimodo(Extension):
    name = "kimodo"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Kimodo"
    homepage = "https://research.nvidia.com/labs/sil/projects/kimodo/"
    source = GitSource(url=KIMODO_URL, commit=KIMODO_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/nv-tlabs/kimodo/blob/main/LICENSE",
    )
    generative = False
    import_repo = ""  # the kimodo package, imported from the pinned repository
    env = EnvSpec(
        python="3.11",
        torch=("torch==2.8.0",),
        torch_backend="cu128",
        build="build.py",  # MotionCorrection (C++ / pybind11 / Eigen): Kimodo's foot-skate and constraint clean-up
    )
    # the core 「标准人」's SOMA 30 and G1 34 skeletons, matching this node's own (bodies.py); their data is put in place
    # by post_install below
    standard_bodies = BODIES
    weights = (
        # 七个权重各一行，按「模型」这个参数取用；默认的那个（Rigplay）不带 option，装了扩展就要有。
        # key 用 KEYS 那张表定死：硬盘上的安装记录按它们记着
        *(Weight(key=KEYS[choice], kind="hf", source=repo, revision=rev, dest=repo.split("/")[-1],
                 gated=choice in GATED, option=None if choice == "rp" else ("model", choice))
          for choice, (repo, rev, _skeleton) in MODELS.items()),
        # needed by the text job alone (its parameters: text = True), so only for a 文字描述
        *(Weight(key=f"llm2vec-{k}", kind="hf", source=repo, revision=rev, dest=repo, option=("text", True)) for k, (repo, rev) in LLM2VEC.items()),
        Weight(key="llama3-8b-instruct", kind="hf", source=LLAMA[0], revision=LLAMA[1], dest=LLAMA[0], files=LLAMA_FILES,
               gated=True, option=("text", True)),
    )

    def worker_env(self) -> dict[str, str]:
        weights = str(self.paths.weights)
        return {
            "CHECKPOINT_DIR": weights,  # load_model finds Kimodo-* here, never on the Hub
            "LOCAL_CACHE": "True",
            "TEXT_ENCODER_MODE": "local",  # never the text encoder web service Kimodo tries first
            "TEXT_ENCODERS_DIR": weights,  # the LLM2Vec adapters as local folders
            # Llama 3 8B in system memory (Kimodo's small-VRAM setting) unless the text job finds room on the GPU
            # (worker.py text_device sets it per job)
            "TEXT_ENCODER_DEVICE": "cpu",
        }

    def post_install(self, run, paths) -> None:
        """The SOMA / G1 skin meshes this extension's 「标准人」 bodies (bodies.py) read live in the core's body-model
        area (third_party/_body_models/), not in this repository: nothing in the main process reads the research repo.
        Copy the SOMA skin and build the G1 body there from the pinned checkout's assets."""
        from lab2shot_shared import body_models

        run([str(paths.python), str(self.adapter_dir / "skin_data.py"), str(paths.repo), str(body_models.ROOT)])


EXTENSION = Kimodo()
