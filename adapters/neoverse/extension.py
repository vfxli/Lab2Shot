from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file


class NeoVerse(Extension):
    name = "neoverse"
    sdk = 2
    title = "NeoVerse"
    import_repo = ""  # 上游 `diffsynth` 包在检出根目录，worker 经 PYTHONPATH 导入（repo/diffsynth/...）
    homepage = "https://github.com/IamCreateAI/NeoVerse"
    source = GitSource(url=homepage + ".git", commit="bb97880d3f6d5cb06b2fcccc8ca889e0988daa4d")
    license = LicenseInfo(tag=COMMERCIAL, url=homepage + "/blob/main/LICENSE.txt")
    generative = False
    # torch-scatter 用 PyG 预编译 wheel（源码构建在 glibc/CUDA 12.8 上 cospi/sinpi noexcept 冲突，
    # 与 wham/unirig/tram 同法），见 requirements.txt
    env = EnvSpec(python="3.11", torch=("torch==2.7.1", "torchvision==0.22.1"), torch_backend="cu128",
                  imports=("gsplat", "torch_scatter"))
    weights = (hf_file("Yuppie1204/NeoVerse", "d6d6990358b088c960175243a33a45ba38ecf1de", "reconstructor.ckpt",
                       dest="reconstructor.ckpt", key="reconstructor", sha256="fd87ca703cf026c19e345e08f68292ae5a43e34d77ff20c7b49b78907e10f5c9"),)


EXTENSION = NeoVerse()
