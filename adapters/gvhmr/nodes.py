"""GVHMR 扩展提供的节点（仅限研究用途）。"""

from __future__ import annotations

from lab2shot.sdk import (Official, Keypoints2D, P, WorldHumans, WorldHumansParams, follow_camera_param, Cost, Licence, people_port,
                          All, Not, Param)

class Solve(WorldHumans):
    id = "gvhmr.solve"
    # 引用的是官方脚本入口（tools/demo/demo.py 的 argparse 与 main 流程），而非模型函数签名：
    # 命令行接受 --video（以及 --static_cam / --use_dpvo / --f_mm 三个开关和一个 Focal Length 数值），
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
        note="① **「人物框」是可选输入，直接交给官方函数**（规则：官方函数接受人物框就直接把框交给它；不接受框的项目，在节点图上走「人物框转遮罩」）：官方包里 "
             "`VitPoseExtractor.extract(video_path, bbx_xys)`（hmr4d/utils/preproc/vitpose.py:23）、"
             "`Extractor.extract_video_features(video_path, bbx_xys)`（vitfeat_extractor.py:65）和 `DemoPL.predict(data)` 的 "
             "`data[\"bbx_xys\"]`（gvhmr_pl_demo.py:16-35）收的就是逐帧的框，官方 demo 自己也是先用 YOLOv8x 跟踪出框"
             "（demo.py:109-110 `Tracker().get_one_track`）再喂给这几处。接了框：按框解这几个人，不再跑跟踪器；"
             "不接：照官方 demo 的路自己跟踪（官方只取最大的一条轨迹，我们解每一条够长的）。"
             "② 「2D 关键点」在上游是**预处理阶段的产物**：`demo.py:126 vitpose_extractor.extract(video_path, bbx_xys)`，"
             "存成 paths.vitpose，再当作 kp2d 喂给模型——所以它是官方算出来的东西，只是官方自己又拿去当输入。"
             "③ **「参照相机」**：GVHMR 交出来的人在**它自己的"
             "重力世界**里（原点在这个人起点的髋部），那个世界里配着一台相机，人和实拍对得上就是靠它。"
             "参照相机不是成品相机：和你自己的相机一起接进核心节点「相机空间转换」（接到「来源相机」），"
             "算一个修正挂到人身上；直接当镜头相机用，人会不在画面里、位置不对。"
             "**它由官方的两套 SMPL 参数唯一确定**：pred 里同时有 smpl_params_incam（demo.py:215，相机空间）"
             "和 smpl_params_global（:259，世界），两者之间的刚性关系就是那台相机（worker.py:246 "
             "wh.camera_from_body）；它的**旋转**来自上游自己的 VO（demo.py:149-150 SimpleVO / :186 R_w2c）。"
             "这个口必须有：没有它，模板上的对齐只剩相机空间那一档，相机的运动会被加两遍。"
             "要一台**能用**的相机，还是从真正解相机的节点（ViPE、TRAM）或「导入 USD」接。"
             "④ 「人物」（蒙皮角色）和「网格」是 CG 形态：官方吐的是 SMPL-X 参数，网格要另外跑一次 body model。"
             "⑤ cam_angvel 只用到相机的**旋转**，官方从来不吃相机的位移。",
    )
    # 公开基准实测结果（3DPW，6 个镜头），用作输入的提示
    measured = {
        "focal_mm": "实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 66%（5 好 0 差）；接 AnyCalib 估的 Focal Length 更差（2 好 4 差）",
        # measured 按输入口名和参数名挂载提示（nodes/base.py describe）；「相机旋转」是参数，键为参数名。
        # 该数据在接入完整 ViPE 相机时测得，上游仅使用其旋转
        "camera_rotate": "实测（3DPW 6 个镜头，接的是整台 ViPE 相机、上游只用其旋转）：人在镜头里的位置误差少 40%（4 好 2 差）；接真实相机少 47%",
    }
    on_node = ("focal_mm", "static_camera")
    inputs = WorldHumans.inputs + (people_port(optional=True, each=False),)  # 人物框为可选输入，见 official 注 ①
    # 支持移动机位与固定机位（固定机位有专用开关）；相机仅估计旋转，纯摇镜头最可靠；人物须连续出现 16 帧以上
    runtime = "gvhmr"
    # 上游仅使用相机逐帧旋转（转换为 cam_angvel 输入网络），不使用位移。输入口提示按此档位生成（kit/cameras.py TAKES_CAMERA）
    camera_to_worker = "rotation"
    smpl_body = "smplx"  # worker 解算的人体模型（worker.py save_person）；仅作声明，不增加输出口（families/humans.py smpl_body）
    # 「参照相机」输出口：结果所在重力世界中对应的相机，仅供「相机空间转换」作为参照，不能作为成品相机使用
    # （见 families/humans.py reference_camera 的说明及 official 注 ③）。
    reference_camera = True
    # 上游流程的第一步计算 ViTPose 的 17 个关键点（worker crops_features）；除输入网络外也作为输出提供
    keypoints = Keypoints2D("ViTPose 在画面上找的全身 17 个点（鼻子、双眼、双耳、肩、肘、腕、髋、膝、踝），GVHMR 解算之前就是按它们找的人")
    min_frames = 16  # 时序网络的窗口长度（worker MIN_FRAMES）：不足时在计算前拒绝，并报告实际帧数
    # 显存与速度在 RTX 4090 上测得
    cost = Cost(gpu=True, vram_gb=5.5, seconds_per_frame=0.25)
    licence = Licence(note="GVHMR 代码和权重仅限非商业科研（浙江大学许可证）；SMPL-X 人体模型需自行注册下载，仅限非商用。")

    class Params(WorldHumansParams):
        follow_camera: bool = follow_camera_param()
        vo_step: int = P(8, label="相机估计间隔", help="估计相机转动时每隔几帧算一次，其余插值：8 适合一般镜头；甩镜、快速摇镜调小（2–4），更准更慢。接了「相机旋转」就不用估了", ge=1, le=30, group="相机", applies=All(Not(Param("camera_rotate").wired()), Param("static_camera").one_of(False)))


NODES = (Solve,)
