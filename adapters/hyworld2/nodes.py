from typing import Literal
from lab2shot.sdk import GaussianReconstruction, Measured, Official, P, Cost, measured_param


class WorldMirror(GaussianReconstruction):
    """HY-World 2.0 的重建部分就是 WorldMirror 2.0（官方仓 worldrecon/ 下的同一个模型），所以只有这一个节点：
    世界生成那一半不接，这里不加载它的任何权重。"""

    id = "hyworld2.reconstruct"
    version = 3  # 3：官方默认 952，显卡放不下时 worker 降档；2：轴向与单位烘进高斯本体（不再挂翻转变换），相机扣除外扩边缘
    runtime = "hyworld2"
    category = "gaussian"
    official = Official(cite=("third_party/hyworld2/repo/hyworld2/worldrecon/pipeline.py:587-640",
                              "third_party/hyworld2/repo/hyworld2/worldrecon/hyworldmirror/models/models/worldmirror.py:373-384",
                              "third_party/hyworld2/repo/hyworld2/worldrecon/hyworldmirror/models/models/rasterization.py:216-229"),
                        takes={"image": "imgs"}, gives={"gaussians": "splats", "camera": "camera_poses"})
    cost = Cost(gpu=True, vram_gb=18.6, vram_full_gb=50, note=True)  # RTX 4090 实测（整卡峰值减空闲）：1920×1080、32 视角，952 放不下自动降到 518；满档 952、32 视角按 worker 的显存公式（worker.py resolution_vram_gb）约 50 GB，这里的卡都放不下

    class Params(GaussianReconstruction.Params):
        # 官方默认 target_size=952（worldrecon/pipeline.py:393）；视频官方默认取 32 帧（inference_utils.py:129）
        resolution: Literal[280, 518, 952] = P(952, group="solve")
        # 显存随视角数近乎线性增长：只给测过的档（measured_param）。32 是官方视频默认、实测档；更少视角不会更多显存
        max_views: Literal[8, 16, 32] = measured_param({8: Measured(below=16), 16: Measured(below=32), 32: Measured(gb=18.6)},
                                                      default=32, group="solve")

    @classmethod
    def vram_full(cls, params: dict) -> float:
        """满档（按所设分辨率不降档）的显存随视角数变：同 worker.py resolution_vram_gb 的公式和常数（两处必须一致：
        满档时 worker 按这个数挑档，见 fitting_resolution）。接了线、还不知道的取最大一档。"""
        size = params.get("resolution")
        views = params.get("max_views")
        size = size if isinstance(size, int) else 952
        views = views if isinstance(views, int) else 32
        return 4.8 + 13.3 * (size / 518) ** 2 * views / 32


NODES = (WorldMirror,)
