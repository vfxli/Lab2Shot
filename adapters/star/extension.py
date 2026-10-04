"""Pinned official STaR extension; inference is isolated from the core."""
from lab2shot.sdk import Extension, EnvSpec, GitSource, LicenseInfo, Weight, COMMERCIAL, CUDA_13_2_TOOLKIT, ManualItem, ExtensionWeights, manual_weight

STATS = tuple(f"mixamo_quat_{name}.npy" for name in (
    "local_motion_mean", "local_motion_std", "mean", "std", "global_motion_mean", "global_motion_std",
    "shape_mean_xyz", "shape_std_xyz"))
ASSETS = ManualItem(key="star-statistics",
    page="https://github.com/XiaohangYang829/STaR",
    filename="star_statistics.tar.gz",
    markers=("model.t7", "mixamo_quat_mean.npy"), looks_like=("*star_statistics*",),
    install=ExtensionWeights("star", ("model.t7", *STATS)))

class Star(Extension):
    name = "star"
    sdk = 2
    title = "STaR"
    homepage = "https://github.com/XiaohangYang829/STaR"
    source = GitSource("https://github.com/XiaohangYang829/STaR.git", "727495359f52b3a4e6f5a98f6d425c1ea9b94b1c")
    license = LicenseInfo(tag=COMMERCIAL, url="https://github.com/XiaohangYang829/STaR/blob/main/LICENSE")
    generative = False
    submodules = ()
    import_repo = None  # worker_env lists every official package root together
    env = EnvSpec(python="3.11", torch=("torch==2.9.0",), torch_backend="cu130",
                  cuda_toolkit=CUDA_13_2_TOOLKIT,  # pointnet2 编译需 nvcc；pip nvcc 13.2 与 glibc >= 2.43 兼容（同 tram/seganymo）
                  pickled_checkpoints=True, build="build.py",
                  # 不 import method.network：它一 import 就把 Chamfer 损失 JIT 编进上游检出的 submodules/ChamferDistancePytorch/tmp
                  # （训练才用，worker 不走它），自检会把上游弄脏。point_module 一样要 pointnet2_ops 能加载
                  imports=("pointnet2_ops", "method.point_module"))
    manual_items = (ASSETS,)
    weights = (manual_weight(ASSETS),)

    def worker_env(self):
        import os
        paths = [str(self.paths.repo / "")]
        paths.append(str(self.paths.repo / "submodules"))
        return {"PYTHONPATH": os.pathsep.join(paths)}

EXTENSION = Star()
