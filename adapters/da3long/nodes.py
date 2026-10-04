"""DA3 streaming produces the same depth/camera contract as the whole-shot family."""
from lab2shot.sdk import Cost, Confidence, Licence, Official, P, WholeShotDepthCamera, WholeShotParams


class Reconstruct(WholeShotDepthCamera):
    id = "da3long.reconstruct"
    version = 2  # 2：官方默认 120/60，显卡放不下时 worker 降档
    metric = True  # DA3NESTED-GIANT-LARGE-1.1 is in metres, chunks Sim3-aligned to the first (DepthCamera.metric)
    runtime = "da3long"
    confidence = Confidence("exp_plus_one")
    official = Official(
        cite=["third_party/da3long/repo/da3_streaming/da3_streaming.py:132-200",
              "third_party/da3long/repo/da3_streaming/da3_streaming.py:207-305",
              "third_party/da3long/repo/da3_streaming/da3_streaming.py:714-796"],
        takes={"image": "image_dir"}, gives={"depth": "depth", "camera": "intrinsics"})
    cost = Cost(gpu=True, vram_gb=21, vram_full_gb=30, ram_gb=13, note=True)  # RTX 4090 实测：1920×1080、130 帧，自动降到每段 60 帧，整卡峰值减空闲 19.3 GB；worker 按调度给的显存挑档（worker.py chunk_vram_gb）：21 GB 时 16:9 每段 60 帧，满档 30 GB 时 4:3 每段 120 帧（约 29.7 GB）
    licence = Licence(note=True)

    class Params(WholeShotParams):
        # 官方默认 120 帧一段、重叠 60（da3_streaming/configs/base_config.yaml）；显存放不下时 worker 自动降到放得下的
        # 一档并在节点上说明（README 实测：504x378 下 120 帧 28.3 GB、60 帧 21.2 GB）
        max_frames: int = P(120, ge=4, le=120, group="solve")
        overlap: int = P(60, ge=2, le=119, group="solve")
        loops: bool = P(True, group="solve")


NODES = (Reconstruct,)
