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
    summary = "通用的运动恢复结构（SfM）加多视图立体（MVS）流程，带图形界面和命令行，能处理有序和无序的画面集合"
    homepage = "https://github.com/colmap/colmap"
    source = GitSource(url=COLMAP_URL, commit=COLMAP_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        name="BSD-3-Clause（选 GPU 提取特征时用到的 SiftGPU 为非商用许可）",
        url="https://github.com/colmap/colmap/blob/main/COPYING.txt",
        summary=(
            "COLMAP 本身 BSD-3，可商用；安装的是 COLMAP 官方发布的 pycolmap-cuda12 4.2.0 wheel。"
            "默认用 CPU 提取 SIFT 特征（VLFeat，BSD-2），整条流程可商用；"
            "“GPU 提取特征”选项用到 SiftGPU（北卡大学版权，仅限教育、研究和非营利用途），打开后属非商用。"
            "SIFT 专利（US 6,711,293）已于 2020 年到期，没有专利问题。"
            "wheel 编译时关闭了 LSD（AGPL-3）、CGAL（GPL）和 ONNX 特征（ALIKED/LightGlue）；"
            "但静态链接了 SuiteSparse CHOLMOD（含 GPL-2.0+ 的 Supernodal 模块），自己使用不受限，"
            "只有把这个 wheel 打包再分发时才需遵守 GPL。其余依赖 Ceres/PoseLib/faiss/OpenImageIO/Eigen 等均为宽松许可"
        ),
    )
    # No torch: COLMAP is C++. The wheel and its CUDA 12 runtime come from requirements.txt.
    env = EnvSpec(python="3.12")

    def worker_env(self) -> dict[str, str]:
        # Nothing is downloaded at run time: SIFT + sequential/exhaustive matching and
        # both mappers need no model files (loop detection's vocabulary tree and the
        # ALIKED/LightGlue features, which would download, are not used).
        return {"XDG_CACHE_HOME": str(self.paths.root / "cache")}


EXTENSION = Colmap()
