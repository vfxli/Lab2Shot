"""COLMAP: classic structure-from-motion camera solve (per-frame camera, lens,
sparse point cloud) for shots whose camera moves (parallax)."""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo

COLMAP_URL = "https://github.com/colmap/colmap.git"
COLMAP_COMMIT = "be5e29168d4aff238409d60424812df66aac919f"  # tag 4.2.0 (= the pycolmap 4.2.0 wheels)


from .lens import GROUP


# 哪些选项用到非商用部分：「显卡提取特征」开着时用 SiftGPU（仅限教育和研究）。节点的 OptionTrait 从这里生成
OPTION_LICENCES = {"sift_gpu": {True: NONCOMMERCIAL}}

class Colmap(Extension):
    name = "colmap"
    lens_groups = (GROUP,)  # 镜头模型清单（lens.py）；「LensDistortion」按它列
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "COLMAP"
    homepage = "https://github.com/colmap/colmap"
    source = GitSource(url=COLMAP_URL, commit=COLMAP_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/colmap/colmap/blob/main/COPYING.txt",
    )
    generative = False
    # No torch: COLMAP is C++. The wheel and its CUDA 12 runtime come from requirements.txt.
    env = EnvSpec(python="3.12")

    def worker_env(self) -> dict[str, str]:
        # Nothing is downloaded at run time: SIFT + sequential/exhaustive matching and
        # both mappers need no model files (loop detection's vocabulary tree and the
        # ALIKED/LightGlue features, which would download, are not used).
        return {"XDG_CACHE_HOME": str(self.paths.root / "cache")}


EXTENSION = Colmap()
