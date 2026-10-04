"""WHAM 扩展提供的节点（代码为 MIT；SMPL 与训练数据仅限研究用途）。"""

from __future__ import annotations

from lab2shot.sdk import Official, Keypoints2D, P, WorldHumans, WorldHumansParams, follow_camera_param, Cost, window_of


class Solve(WorldHumans):
    id = "wham.solve"
    version = 2  # 2：相机角速度按每个人自己的帧截取；主点按画面中心在送去像素里的位置
    # 引用官方脚本入口（demo.py 的 argparse 与 main 流程），而非模型函数签名：
    # 命令行接受 --video（以及 --calib / --estimate_local_only / --run_smplify 等开关），
    # 检测和跟踪由脚本自行完成并写入 tracking_results（其中包含 keypoints），DPVO 相机写入 slam_results，
    # 经 CustomDataset 转换为 cam_angvel 输入网络，结果为 results[_id] 中的各项。
    # 接入相机的旋转放在 slam_results 的位置（worker.py），与上游流程一致。
    official = Official(
        cite=("third_party/wham/repo/demo.py:180-231",   # 入口：--video / --calib / --run_smplify
              "third_party/wham/repo/demo.py:49-75",     # 框：由 DetectionModel 生成，不是输入
              "third_party/wham/repo/demo.py:108-170"),  # 输入 cam_angvel；输出 keypoints / pose_world / verts
        # `camera_rotate`：「相机旋转」参数的输入口（families/humans.py WorldHumansParams），
        # 上游据此计算 cam_angvel。节点上没有名为 camera 的输入口，此处的键与参数名一致
        takes={"image": "--video", "camera_rotate": "cam_angvel"},
        ours={"camera": ""},  # node."wham.solve".official.ours.camera
        gives={"keypoints": "tracking_results[_id]['keypoints']", "character": "pose_world",
               # 「参照相机」：上游将同一人同时输出在相机空间和世界空间（demo.py:162-167 的
               # results[_id]['pose'] / ['trans'] 为相机空间，['pose_world'] / ['trans_world'] 为世界空间），
               # 两者之间的刚性变换即该相机，由这两份官方输出唯一确定；两份都须输出，不能只输出世界空间一份
               "ref_camera": "trans_cam"},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 36%（4 好 2 差）；接 AnyCalib 估的 Focal Length 更差（2 好 4 差）
    #   camera_rotate：实测（3DPW 6 个镜头，接的是整台 ViPE 相机、上游只用其旋转）：人在镜头里的位置误差少 40%（5 好 1 差）；接真实相机少 31%
    on_node = ("focal_mm", "static_camera")
    # 相机仅估计旋转，位置由人物反推；固定机位有专用开关；人物须连续出现 30 帧以上；
    # 填写真实 Focal Length 可使人物位置误差降低 36%，接入 ViPE 相机可降低 40%（3DPW 6 个镜头）
    runtime = "wham"
    # 上游仅使用相机逐帧旋转（转换为 cam_angvel 输入网络），不使用位移，
    # 可在 third_party/wham/repo 中核实。因此没有「相机」输入口，旋转接到「相机旋转」参数上（families/humans.py）
    camera_to_worker = "rotation"
    smpl_body = "smpl"  # worker 解算的人体模型（worker.py save_person）；仅作声明，不增加输出口（families/humans.py smpl_body）
    # 「参照相机」输出口：结果所在世界中对应的相机，仅供「相机空间转换」作为参照，不能作为成品相机使用
    # （见 families/humans.py reference_camera 的说明及 official 注 ②）。
    reference_camera = True
    # 「相机」输出口：放人用的那台原点静止针孔相机（相机空间那一份人所在的相机；焦距按「Focal Length」，留空用它自己的默认），
    # 模板把它当「解算器的相机」一路：「相机空间转换」把世界里的人从参照相机搬到它（families/humans.py plate_camera）
    plate_camera = True
    # 官方流程的第一步计算 ViTPose 的 17 个关键点（worker detect_and_track），并作为关键点输出
    keypoints = Keypoints2D("")  # what they are: node."wham.solve".port.keypoints.help
    # 上游 MINIMUM_FRMAES：少于 30 帧的轨迹会被丢弃（worker.py MIN_FRAMES）。在提交前拒绝，
    # 避免排队、启动进程并加载模型后才在 worker 中报错（GVHMR / TRAM 的 16 帧同理）
    min_frames = 30
    # vram_gb：在 RTX 4090 上测得
    cost = Cost(gpu=True, vram_gb=2.9, seconds_per_frame=0.375)

    class Params(WorldHumansParams):
        follow_camera: bool = follow_camera_param()
        smplify: bool = P(False, group="people")

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px：去畸变画面带 overscan 时，送去的是整块画布，
        画面中心不在画布中心（不对称 overscan 时更明显），worker 的内参主点用它而不用画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})


NODES = (Solve,)
