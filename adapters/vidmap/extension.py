"""VidMap's official neural frontend and native COLMAP mapping extension."""
from lab2shot.sdk import NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, Weight, downloads, hf_file

# The Depth Anything 3 checkpoint and GeoCalib's pinhole model it runs
DA3NESTED_REPO, DA3NESTED_REVISION = "depth-anything/DA3NESTED-GIANT-LARGE-1.1", "b2359bdf726fb44ef62acca04d629dcf158053e7"
DA3NESTED_CONFIG = downloads.hf(DA3NESTED_REPO, DA3NESTED_REVISION, "config.json",
                                "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468")
DA3NESTED_MODEL = downloads.hf(DA3NESTED_REPO, DA3NESTED_REVISION, "model.safetensors",
                               "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c")
GEOCALIB_PINHOLE = downloads.Download("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
                                      "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac")


class VidMap(Extension):
    name = "vidmap"
    sdk = 2
    title = "VidMap"
    homepage = "https://github.com/cvg/vidmap"
    source = GitSource(homepage + ".git", "f090773550e6116371e99ce9350291e881b929ef")
    submodules = ("third_party/Depth-Anything-3", "third_party/MegaLoc", "third_party/RoMaV2")
    extra_sources = {"dinov3": GitSource("https://github.com/facebookresearch/dinov3.git", "adc254450203739c8149213a7a69d8d905b4fcfa")}
    import_repo = ""
    license = LicenseInfo(url=homepage + "/blob/main/LICENSE", tag=NONCOMMERCIAL)
    generative = False
    # PyTorch 2.14 / TorchVision 0.29 as the README tests (README.md:61-63): the frontend's default torch.compile path
    # (aot_compile + load_compiled_function) does not exist in 2.10, and inductor fails on RoMaV2 there
    env = EnvSpec(python="3.12", torch=("torch==2.14.0", "torchvision==0.29.0"), torch_backend="cu130",
                  # conda-forge 的 colmap 4.2.0 cpu 包链接 libOpenImageIO.so.3.1，但其依赖表漏列了 openimageio；
                  # openimageio 3.1 的 config.cmake 又 find_dependency(fmt) 和 TBB；colmap 依赖的
                  # ceres-solver 2.2.0 是 GPU build（gpugplh…），其 config.cmake find_dependency(CUDAToolkit 12.9.86)
                  # REQUIRED。CUDAToolkit 由 build.py 指向系统 CUDA 12.9.86（CUDAToolkit_ROOT），不装 conda
                  # cuda-toolkit（其完整工具链与 colmap 的 krb5/libxcb 约束冲突，无法解析）。
                  # libtiff 4.7.2 编译启用了 12-bit JPEG（jpeg12_* 符号）而 conda-forge libjpeg-turbo 无 12-bit，
                  # 由 build.py 编译 jpeg12_shim.so（stub 符号）并在 worker_env 预加载解决。
                  conda=("colmap=4.2.0=cpu_*", "openimageio=3.1.*", "fmt=11.*", "tbb", "cmake=4.4.3", "ninja=1.13.2", "pybind11=3.0.2"),
                  build="build.py", compiled_cuda=False, imports=("pycolmap", "vidmap_native._core", "vidmap.frontend.runner"),
                  pickled_checkpoints=True)

    def worker_env(self):
        import os
        # conda-forge libtiff 4.7.2 needs jpeg12_*@LIBJPEG_8.0 that its libjpeg-turbo lacks;
        # the stub built by build.py satisfies the linker (12-bit JPEG is never used at runtime).
        shim = self.paths.root / "jpeg12_shim.so"
        # the frontend's compiled RoMaV2 / DA3 graphs (frontend/models/compiled_graph.py) default to ~/.cache/vidmap;
        # they belong in the extension's own cache, keyed there by torch/GPU/source identity
        cache = self.paths.root / "cache" / "vidmap"
        env = {f"VIDMAP_{ns.upper()}_CACHE_DIR": str(cache / ns) for ns in ("romav2", "da3")}
        return env | ({"LD_PRELOAD": str(shim)} if shim.is_file() else {})
    weights = (
        DA3NESTED_MODEL.weight(key="da3", dest="da3/model.safetensors"),
        DA3NESTED_CONFIG.weight(key="da3-config", dest="da3/config.json"),
        GEOCALIB_PINHOLE.weight(key="geocalib", dest="torch/hub/geocalib/pinhole.tar"),
        Weight("roma", "url", "https://github.com/Parskatt/RoMaV2/releases/download/v2.0.1/romav2.0.1.pt",
               "torch/hub/checkpoints/romav2.0.1.pt", sha256="1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7"),
        Weight("aliked", "url", "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth",
               "torch/hub/checkpoints/aliked-n16.pth", sha256="5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"),
        hf_file("gberton/MegaLoc", "7cb9f7970d366fdf059963d04d372e503e8e9df9", "model.safetensors", key="megaloc",
                dest="megaloc/model.safetensors", sha256="d4f9f2bcb60018f91eb6a8e061ed054fd55654e10c2569cf13841ea986ffb4f8"),
    )


EXTENSION = VidMap()
