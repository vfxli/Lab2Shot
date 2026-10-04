"""Nodes provided by the HaMeR extension (MIT code; MANO and the training data research only)."""

from __future__ import annotations

from lab2shot.sdk import Official, CameraLensParams, Keypoints2D, P, WorldHumans, Cost, people_port, window_of


class Hands(WorldHumans):
    id = "hamer.hand_solve"
    version = 2  # 2：位移按画面中心（principal_px）反推，不按画布中心（带不对称 overscan 时手不再整体偏移）
    # 引的是官方脚本的入口：
    # demo.py 的 argparse 只收一个图片文件夹（`--img_folder`）和两个参数（`--rescale_factor`、
    # `--body_detector vitdet`），main 里画面（img_cv2）进去，ViTPose+ 的手部关键点（vitposes['keypoints']）、
    # MANO 参数（pred_mano_params）和网格（out['pred_vertices']）出来。
    official = Official(
        cite=("third_party/hamer/repo/demo.py:22-33",
              "third_party/hamer/repo/demo.py:44-157",
              "third_party/hamer/repo/hamer/models/hamer.py:100-135",
              "third_party/hamer/repo/vitpose_model.py:60-72"),  # 官方 ViTPoseModel.predict_pose(image, det_results, …)：收人框
        takes={"image": "img_cv2", "boxes": "det_results"},
        gives={"character": "pred_mano_params", "keypoints": "vitposes['keypoints']"},
        ours={"camera": ""},  # node."hamer.hand_solve".official.ours.camera
    )
    # 公开基准上的实测（接不接、接什么的差别；没有相机这一条：这个节点没有相机输入，摆进相机的世界是核心节点
    # 「相机空间转换」的事，那条链没测过）：
    #   focal_mm：实测（FreiHAND、DexYCB 8 个镜头）：不给 Focal Length 时 HaMeR 自己假设的 Focal Length 偏 40% 以上，手在镜头里的位置差 42 cm；填真实 Focal Length 降到 3 cm（8 个全变好）；接 AnyCalib 估的 Focal Length 降到 6 cm（7 好 1 差）
    on_node = ("focal_mm", "rescale_factor")
    inputs = WorldHumans.inputs + (people_port(optional=True, each=False),)  # 框可选：official 注 ⓪
    # 每帧单独找手、单独重建，机位无关；手要在画面里几十像素以上；Focal Length 对手的位置影响最大
    runtime = "hamer"
    # 找手就是靠 ViTPose+ 在画面上找的这 21 个点（demo.py），除了框手之外也交出来
    keypoints = Keypoints2D("")  # what they are: node."hamer.hand_solve".port.keypoints.help
    camera_to_worker = None
    plate_camera = True  # 「相机」：放手用的原点静止针孔相机（families/humans.py plate_camera），不填 Focal Length 按 50 mm
    default_focal_mm = 50.0  # HaMeR's own default lens (5000 px at 256) puts hands ~100 m away
    # RTX 4090 上量得的显存和速度
    cost = Cost(gpu=True, vram_gb=5.9, seconds_per_frame=0.07)

    class Params(CameraLensParams):
        rescale_factor: float = P(2.0, ge=1.0, le=4.0, group="people")

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px（同 gvhmr）：去畸变画面带 overscan 时送去的是整块
        画布，画面中心不在画布中心；放手用的针孔相机主点在画面中心，位移按它反推，不按画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})


NODES = (Hands,)
