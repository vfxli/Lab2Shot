"""Official Depth Anything 3 streaming pipeline, isolated from the core."""
from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, downloads, hf_file

# DINOv2's network code at the commit torch.hub would fetch (Depth Anything 3's backbone)
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
# The DA3NESTED checkpoint the pipeline streams with
DA3NESTED_REPO, DA3NESTED_REVISION = "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3NESTED_CONFIG = downloads.hf(DA3NESTED_REPO, DA3NESTED_REVISION, "config.json",
                                "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468")
DA3NESTED_MODEL = downloads.hf(DA3NESTED_REPO, DA3NESTED_REVISION, "model.safetensors",
                               "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c")


class DA3Long(Extension):
    name = "da3long"
    sdk = 2
    title = "Depth Anything 3 Streaming"
    homepage = "https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main/da3_streaming"
    source = GitSource("https://github.com/ByteDance-Seed/Depth-Anything-3.git", "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4")
    submodules = ("da3_streaming/loop_utils/salad",)
    extra_sources = {"dinov2": GitSource("https://github.com/facebookresearch/dinov2.git", DINOV2_COMMIT)}
    import_repo = "src"
    license = LicenseInfo(url=homepage, tag=NONCOMMERCIAL)
    generative = False
    env = EnvSpec(python="3.11", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu128",
                  imports=("faiss", "pypose", "numba", "depth_anything_3.api"))
    weights = (
        DA3NESTED_MODEL.weight(key="model", dest="model.safetensors"),
        DA3NESTED_CONFIG.weight(key="config", dest="config.json"),
        Weight("salad", "url", "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
               "dino_salad.ckpt", option=("loops", True)),
    )


EXTENSION = DA3Long()
