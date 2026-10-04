"""Pinned official SATA extension; inference is isolated from the core."""
from lab2shot.sdk import Extension, EnvSpec, GitSource, LicenseInfo, Weight, COMMERCIAL

class Sata(Extension):
    name = "sata"
    sdk = 2
    title = "SATA"
    homepage = "https://github.com/zzysteve/SATA"
    source = GitSource("https://github.com/zzysteve/SATA.git", "9858f4df2322bc796c5dcf56e9c8eae01969518c")
    license = LicenseInfo(tag=COMMERCIAL, url="https://github.com/zzysteve/SATA/blob/main/LICENSE")
    generative = False
    submodules = ('src/fairmotion',)
    import_repo = None  # worker_env lists every official package root together
    env = EnvSpec(python="3.11", torch=("torch==2.8.0",), torch_backend="cu128",
                  pickled_checkpoints=True, build="build.py", imports=("numpy",),
                  places=tuple(f"repo/result/{name}/config.yaml" for name in
                               ("rvq_human", "vae_human", "vae_animo", "vae_merge")))
    weights = (Weight(key="retarget-models", kind="hf", source="SteveZh/sata_models", revision="2eedae4e9a1d94db5bde075f4b852608add807e3", dest="", files=("result/*",)),
               # 官方语义骨架图的关节文本特征用 T5-base（src/sata/utils/bvh2joint.py），
               # 768 维平均池化；钉住 revision 的完整快照，tokenizer 一并离线
               Weight(key="t5-base", kind="hf", source="google-t5/t5-base", revision="a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1", dest="t5-base",
                      files=("config.json", "model.safetensors", "spiece.model", "tokenizer.json")))

    def worker_env(self):
        import os
        paths = [str(self.paths.repo / "src")]
        paths.append(str(self.paths.repo / "src/fairmotion"))
        return {"PYTHONPATH": os.pathsep.join(paths)}

EXTENSION = Sata()
