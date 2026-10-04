from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

REV = "d78a16c91c7a56488894a1c8de4f5c7cc28aa8b0"


class HYWorld2(Extension):
    name = "hyworld2"
    sdk = 2
    title = "HY-World 2.0 / WorldMirror 2.0"
    import_repo = ""  # 上游 `hyworld2` 包在检出根目录，worker 经 PYTHONPATH 导入（repo/hyworld2/...）
    homepage = "https://github.com/Tencent-Hunyuan/HY-World-2.0"
    source = GitSource(url=homepage + ".git", commit="df9988efb87bfc0f4947eb3889411cf957478b06")
    license = LicenseInfo(tag=COMMERCIAL, url=homepage + "/blob/main/License.txt")
    generative = False
    env = EnvSpec(python="3.11", torch=("torch==2.7.1", "torchvision==0.22.1"), torch_backend="cu128",
                  compiled=("flash-attn==2.8.3.post1",), imports=("hyworld2.worldrecon.pipeline", "gsplat"))
    weights = (
        hf_file("tencent/HY-World-2.0", REV, "HY-WorldMirror-2.0/config.json", dest="HY-WorldMirror-2.0/config.json", key="config"),
        hf_file("tencent/HY-World-2.0", REV, "HY-WorldMirror-2.0/model.safetensors", dest="HY-WorldMirror-2.0/model.safetensors", key="model",
                sha256="9fff06539d3d9e85338d7de1ffb5afffb7739fa5bf4d62b4b7c319b5ecdde54f"),
        # 官方重建管线的天空分割器（compute_sky_mask source="auto" 会读工作目录里的 skyseg.onnx）
        hf_file("JianyuanWang/skyseg", "3ba8c6df1d9ba9ff26f637c7ba9568ac11a9aa7f", "skyseg.onnx", dest="skyseg.onnx", key="skyseg",
                sha256="ab9c34c64c3d821220a2886a4a06da4642ffa14d5b30e8d5339056a089aa1d39"),
    )


EXTENSION = HYWorld2()
