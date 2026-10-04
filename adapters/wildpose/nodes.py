"""WildPose's real outputs: all-frame camera, keyframe refined depth."""
from lab2shot.sdk import Cost, Job, LensWholeShotParams, Licence, Official, P, WholeShotDepthCamera, window_of


class Reconstruct(WholeShotDepthCamera):
    id = "wildpose.reconstruct"
    metric = True  # approximate: MoGe-2 metric depth initialises it, no explicit scale term in the final BA (DepthCamera.metric)
    runtime = "wildpose"
    version = 5  # 5：没填焦距在提交前就拦下（plan_refusals）；4：处理分辨率按长边取（与 MASt3R-512 一致），主点按画面中心在送去像素里的位置；3：度量输出不再被「尺度」缩放；2：处理画面的宽高都取 16 的倍数（模型按 16 像素切块，宽画面的高取 8 的倍数会失败）
    on_node = ("focal_mm", "step")
    min_frames, min_frames_step = 8, "step"
    official = Official(
        cite=["third_party/wildpose/repo/src/utils/datasets.py:19-143",
              "third_party/wildpose/repo/src/slam.py:96-179",
              "third_party/wildpose/repo/src/depth_video.py:270-292"],
        takes={"image": "color_data"}, gives={"depth": "depths", "camera": "traj_est_not_align"})
    cost = Cost(gpu=True, vram_gb=13, ram_gb=6, measured_on="RTX 5090 32 GB", note=True)
    licence = Licence(note=True)

    class Params(LensWholeShotParams):
        resolution: int = P(512, ge=256, le=768, group="solve")
        buffer: int = P(300, ge=16, le=2000, group="solve")
        final_ba: bool = P(True, group="solve")

    @classmethod
    def plan_refusals(cls, params, comes):
        """No focal length typed and none wired into it: known before the cook, so 「计算」 is greyed with the reason
        (B-WILDPOSE-NOFOCAL). A focal length wired from another node is known only once it cooks: prepare refuses
        then (E-WILDPOSE-NOFOCAL)."""
        from lab2shot.sdk import Msg
        if params.get("focal_mm") is not None or comes(cls.param_port("focal_mm").name):
            return []
        return [(Msg("B-WILDPOSE-NOFOCAL"), "image")]

    @classmethod
    def prepare(cls, ctx):
        from lab2shot.sdk import Invalid, Msg
        job = super().prepare(ctx)
        if not job.lens.given:
            raise Invalid(Msg("E-WILDPOSE-NOFOCAL"))
        # the picture's centre in the pixels sent: on an undistorted plate's canvas (overscan) not the canvas centre
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})


NODES = (Reconstruct,)
