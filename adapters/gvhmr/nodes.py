"""GVHMR 扩展提供的节点（仅限研究用途）。"""

from __future__ import annotations

from lab2shot.sdk import (Official, Keypoints2D, P, WorldHumans, WorldHumansParams, follow_camera_param, Cost, Licence, people_port,
                          All, Not, Param, window_of)

class Solve(WorldHumans):
    id = "gvhmr.solve"
    version = 2  # 2：不接人物框时逐帧持续跟踪（编号不再逐帧重置）；主点按画面中心在送去像素里的位置；半分辨率用 INTER_AREA
    # 引用的是官方脚本入口（tools/demo/demo.py 的 argparse 与 main 流程），而非模型函数签名：
    # 命令行接受 --video（以及 --static_cam / --use_dpvo 两个开关和 Focal Length 数值 --f_mm），
    # 输出 pred 中的 smpl_params_global / smpl_params_incam，关键点为其保存的 paths.vitpose。
    # 上游将相机逐帧旋转转换为 cam_angvel 输入网络（demo.py:197 compute_cam_angvel(R_w2c)），
    # 官方流程以 DPVO / SimpleVO 解算 R_w2c；外部接入的相机旋转在同一位置替换（worker.py）。
    official = Official(
        cite=("third_party/gvhmr/repo/tools/demo/demo.py:39-60",     # 入口：--video / --static_cam / --f_mm
              "third_party/gvhmr/repo/tools/demo/demo.py:106-116",   # 框：由 Tracker 生成，不是输入
              "third_party/gvhmr/repo/tools/demo/demo.py:190-201",   # 网络输入：kp2d / cam_angvel / f_imgseq
              "third_party/gvhmr/repo/tools/demo/demo.py:210-218",   # 输出：smpl_params_incam
              "third_party/gvhmr/repo/tools/demo/demo.py:255-262"),  # 输出：smpl_params_global
        # `camera_rotate`：「相机旋转」参数的输入口（families/humans.py WorldHumansParams），上游据此计算 cam_angvel。
        # 本节点没有名为 camera 的输入口，此处的键与参数名一致
        takes={"image": "--video", "camera_rotate": "cam_angvel", "boxes": "bbx_xys"},
        # 「参照相机」：上游将同一人体同时输出在相机空间和世界空间（demo.py:215 smpl_params_incam、
        # :259 smpl_params_global，均保存于 :327 paths.hmr4d_results），两者之间的刚性变换即为该相机，
        # 由这两份官方输出唯一确定
        gives={"character": "smpl_params_global", "keypoints": "kp2d", "ref_camera": "smpl_params_incam"},
        ours={"camera": ""},  # node."gvhmr.solve".official.ours.camera
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 66%（5 好 0 差）；接 AnyCalib 估的 Focal Length 更差（2 好 4 差）
    #   camera_rotate：实测（3DPW 6 个镜头，接的是整台 ViPE 相机、上游只用其旋转）：人在镜头里的位置误差少 40%（4 好 2 差）；接真实相机少 47%
    on_node = ("focal_mm", "static_camera")
    inputs = WorldHumans.inputs + (people_port(optional=True, each=False),)  # 人物框为可选输入，见 official 注 ①
    # 支持移动机位与固定机位（固定机位有专用开关）；相机仅估计旋转，纯摇镜头最可靠；人物须连续出现 16 帧以上
    runtime = "gvhmr"
    # 上游仅使用相机逐帧旋转（转换为 cam_angvel 输入网络），不使用位移。因此没有「相机」输入口，旋转接到「相机旋转」参数上
    # （families/humans.py）
    camera_to_worker = "rotation"
    smpl_body = "smplx"  # worker 解算的人体模型（worker.py save_person）；仅作声明，不增加输出口（families/humans.py smpl_body）
    # 「参照相机」输出口：结果所在重力世界中对应的相机，仅供「相机空间转换」作为参照，不能作为成品相机使用
    # （见 families/humans.py reference_camera 的说明及 official 注 ③）。
    reference_camera = True
    # 「相机」输出口：放人用的那台原点静止针孔相机（相机空间那一份人所在的相机；焦距按「Focal Length」，留空用它自己的默认），
    # 模板把它当「解算器的相机」一路：「相机空间转换」把世界里的人从参照相机搬到它（families/humans.py plate_camera）
    plate_camera = True
    # 上游流程的第一步计算 ViTPose 的 17 个关键点（worker crops_features）；除输入网络外也作为输出提供
    keypoints = Keypoints2D("")  # what they are: node."gvhmr.solve".port.keypoints.help
    min_frames = 16  # 时序网络的窗口长度（worker MIN_FRAMES）：不足时在计算前拒绝，并报告实际帧数
    # 显存与速度在 RTX 4090 上测得
    cost = Cost(gpu=True, vram_gb=5.5, seconds_per_frame=0.25)
    licence = Licence(note=True)

    class Params(WorldHumansParams):
        follow_camera: bool = follow_camera_param()
        vo_step: int = P(8, ge=1, le=30, group="camera", applies=All(Not(Param("camera_rotate").wired()), Param("static_camera").one_of(False)))

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px：去畸变画面带 overscan 时，送去的是整块画布，
        画面中心不在画布中心（不对称 overscan 时更明显），worker 的内参主点用它而不用画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})


NODES = (Solve,)
