"""TRAM 扩展提供的节点（代码为 MIT；SMPL、SPEC 和 DEVA 部分仅限研究用途）。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import rgb_port, Official, measured_param, Port, WorldHumans, WorldHumansParams, Cost, Measured, people_port, window_of


class Solve(WorldHumans):
    id = "tram.solve"
    version = 2  # 2：主点按画面中心在送去像素里的位置；VIMO 骨干逐帧只算一次（数值与逐窗重算一致到浮点误差）
    # 引用官方脚本入口，而非上游流程内部 VIMO 推理函数的签名：
    # estimate_camera.py 的 argparse 接受 `--video`（一段画面）和 `--static_camera`，
    # 人物框 boxes_ / masks_ / tracks_ 由脚本自行通过 detect_segment_track 获得（:38-39）；
    # estimate_humans.py 的 argparse 接受 `--video` 和 `--max_humans`，框从 tracks.npy 读取（:32, :49）。
    official = Official(
        # 引用多处：官方相机解算脚本、官方人体解算脚本，以及写出顶点的官方可视化代码
        # （Official.cite 支持多条引用，符号出现在任一引用范围内即可）
        cite=("third_party/tram/repo/scripts/estimate_camera.py:15-57",
              "third_party/tram/repo/scripts/estimate_humans.py:14-52",
              "third_party/tram/repo/lib/models/hmr_vimo.py:134-150",   # 官方 HMR_VIMO.inference(imgfiles, boxes, …)：接受每个人的逐帧框
              "third_party/tram/repo/scripts/emdb/run_smpl.py:59-63",   # 官方以外部（真值）框调用该函数的示例
              "third_party/tram/repo/lib/vis/traj.py:125-135"),
        takes={"image": "imgfiles", "boxes": "boxes"},
        gives={"character": "pred_rotmat", "camera": "world_cam_R"},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 8 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 31%（7 好 0 差）；接 AnyCalib 估的 Focal Length 更差（2 好 6 差）
    on_node = ("focal_mm", "static_camera")
    # VIMO 每次处理 16 帧（worker.py MIN_FRAMES）。帧数不足的输入在提交前拒绝，
    # 避免排队、启动进程并加载模型后才在 worker 中报错，浪费 GPU
    min_frames = 16
    # docs.md：自带 DROID-SLAM 相机解算，最适合移动镜头；固定机位有专用开关；
    # 纯摇镜头和长焦压缩镜头的尺度可能不准；填写真实 Focal Length 可使位置误差降低 31%（3DPW 8 个镜头，7 好 0 差）。
    # 「人物框」为可选输入：官方 HMR_VIMO.inference 接受每个人的逐帧框（official 注 ⓪）；未接入时由上游使用
    # ViTDet + SAM + DEVA 检测人物。不提供「相机」输入口（上游 VIMO 仅使用 Focal Length 和主点，相机是其自身解算的产物，见注 ①）
    inputs = (rgb_port(), people_port(optional=True, each=False))  # 人物框为可选输入，见 official 注 ⓪
    runtime = "tram"
    # 上游只向 VIMO 输入 Focal Length（img_focal）和主点，不使用逐帧外参，对应三档中「不使用相机动画、仅使用 Focal Length」
    # 一档（families/humans.py camera_to_worker）。该设置同时移除家族添加的「相机」输入口，
    # 并使「已知 Focal Length」「Filmback」成为普通参数（没有相机口，也就不存在接入相机后置灰的情况）
    camera_to_worker = None
    # 本方法自行解算相机：上游用遮挡人物后的 DROID-SLAM 解出整段轨迹
    # （third_party/tram/repo/scripts/estimate_camera.py:45-53 的 world_cam_R / world_cam_T / img_focal），
    # 属于流程的产物，因此「相机」输出口为官方输出，予以保留（families/humans.py solves_camera）。
    # 其他人体 / 手 / 脸解算器不具备这一点：其相机只能由人物位置反推，因此它们没有「相机」输出口。
    solves_camera = True
    smpl_body = "smpl"  # worker 解算的人体模型（worker.py save_person）
    # vram_gb：在 RTX 4090 上测得（docs.md）
    cost = Cost(gpu=True, vram_gb=8.5, seconds_per_frame=1.0)

    # 继承 WorldHumansParams 仅为使用「固定机位」（上游 --static_camera）。其中的「相机旋转」对本节点无用（camera_to_worker=None，
    # 相机由上游自行解算），家族据此声明移除该参数，并将「固定机位」的提示替换为不涉及该输入的表述（families/humans.py _without_camera_rotate）
    class Params(WorldHumansParams):
        max_people: Literal[1, 2, 5, 20] = measured_param(
            {1: Measured(below=20), 2: Measured(below=20), 5: Measured(below=20), 20: Measured(gb=8.5)},
            default=20, group="people")

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px：去畸变画面带 overscan 时，送去的是整块画布，
        画面中心不在画布中心（不对称 overscan 时更明显），worker 的内参主点用它而不用画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})


NODES = (Solve,)
