"""WHAM 扩展提供的节点（代码为 MIT；SMPL 与训练数据仅限研究用途）。"""

from __future__ import annotations

from lab2shot.sdk import Official, Keypoints2D, P, WorldHumans, WorldHumansParams, follow_camera_param, Cost


class Solve(WorldHumans):
    id = "wham.solve"
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
        ours={"camera": "放人用的针孔相机：原点、不动，焦距 = 节点的「Focal Length」（留空用解算器自己估的 / 默认的），主点在画面中心（families/humans.py plate_camera）"},
        gives={"keypoints": "tracking_results[_id]['keypoints']", "character": "pose_world",
               # 「参照相机」：上游将同一人同时输出在相机空间和世界空间（demo.py:162-167 的
               # results[_id]['pose'] / ['trans'] 为相机空间，['pose_world'] / ['trans_world'] 为世界空间），
               # 两者之间的刚性变换即该相机，由这两份官方输出唯一确定；两份都须输出，不能只输出世界空间一份
               "ref_camera": "trans_cam"},
        note="① **「人物框」不是它的输入，框是上游自己在内部检测的**："
             "官方脚本只收一段画面（demo.py:182 `--video`），框由它自己的 YOLOv8x 检测 + ViTPose 跟踪出"
             "（demo.py:65 `detector.track(img, fps, length)`，lib/models/preproc/detector.py DetectionModel）。"
             "所以节点没有「人物框」输入口：把接进来的框塞进上游同一个结构 detector.tracking_results['bbox'] 里、"
             "顶替掉它自己的检测，是我们自己开的口，不是官方的输入。要只解其中某一个人，在上游把别人从画面里去掉，再接进「RGB」口："
             "「ViTDet 人物框」→「选人」→「人物框转遮罩」→「图像合成」。"
             "② **「参照相机」**：WHAM 交出来的人在**它自己的"
             "世界**里（原点在这个人起点的髋部），那个世界里配着一台相机，人和实拍对得上就是靠它。"
             "参照相机不是成品相机：和你自己的相机一起接进核心节点「相机空间转换」（接到「来源相机」），"
             "算一个修正挂到人身上；直接当镜头相机用，人会不在画面里、位置不对。"
             "**它由官方的两份输出唯一确定**：上游把同一个人同时输出在相机里（demo.py:162-163 "
             "`results[_id]['pose'] / ['trans']`）和世界里（:164-165 `['pose_world'] / ['trans_world']`），"
             "两者之间的刚性关系就是那台相机（worker.py wh.camera_from_body）；它的**旋转**来自上游自己的 "
             "DPVO（demo.py:56 `SLAMModel(...)`、:76 `slam.process()`、:88 `joblib.dump(slam_results, ...)`）。"
             "这个口必须有：没有它，模板上的对齐只剩相机空间那一档，相机的运动会被加两遍。"
             "要一台**能用**的相机，还是从真正解相机的节点（ViPE、TRAM）或「导入 USD」接。"
             "③ 「蒙皮角色」是 CG 形态，官方吐的是 SMPL 参数 + 顶点。"
             "④ cam_angvel 只用相机的**旋转**；官方的 --calib（demo.py:189）只是内参，不是相机。",
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
    keypoints = Keypoints2D("ViTPose 在画面上找的全身 17 个点（鼻子、双眼、双耳、肩、肘、腕、髋、膝、踝），WHAM 解算之前就是按它们找的人")
    # 上游 MINIMUM_FRMAES：少于 30 帧的轨迹会被丢弃（worker.py MIN_FRAMES）。在提交前拒绝，
    # 避免排队、启动进程并加载模型后才在 worker 中报错（GVHMR / TRAM 的 16 帧同理）
    min_frames = 30
    # vram_gb：在 RTX 4090 上测得
    cost = Cost(gpu=True, vram_gb=2.9, seconds_per_frame=0.375)

    class Params(WorldHumansParams):
        follow_camera: bool = follow_camera_param()
        smplify: bool = P(False, label="SMPLify 细化", group="人物")


NODES = (Solve,)
