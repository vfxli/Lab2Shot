"""TRAM 扩展提供的节点（代码为 MIT；SMPL、SPEC 和 DEVA 部分仅限研究用途）。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import Official, measured_param, Port, WorldHumans, WorldHumansParams, Cost, Licence, Measured, people_port


class Solve(WorldHumans):
    id = "tram.solve"
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
        note="⓪ **「人物框」是可选输入，直接交给官方函数**（规则：官方函数接受人物框就直接把框交给它；不接受框的项目，在节点图上走「人物框转遮罩」）：官方包的 "
             "`HMR_VIMO.inference(imgfiles, boxes, valid=, frame=, …)`（lib/models/hmr_vimo.py:134）收的就是一个人逐帧的框，"
             "官方自己的 EMDB 评测脚本就是拿外部真值框调它的（scripts/emdb/run_smpl.py:59-63）。官方 demo 不传框只是因为它"
             "自己用 ViTDet + SAM + DEVA 检人跟踪（estimate_camera.py:38-39 → tracks.npy → estimate_humans.py:32）。"
             "接了框：VIMO 按框解这几个人；上游的检测 + 分割照旧跑一遍——SLAM 要的人物遮罩来自它（worker.py solve_camera）。"
             "不接框：照官方 demo 的路自己检人跟踪，解所有轨迹（上限「最多人数」）。"
             "① **没有「相机」输入口**：VIMO 的签名里只有 img_focal 和 "
             "img_center（内参），官方的相机是它自己用遮人的 DROID-SLAM 解出来的"
             "（scripts/estimate_camera.py:45-47），是这条流程的**产物**、不是它的输入。"
             "接一台解好的相机（ViPE / 3DE）只能让 worker **跳过** TRAM 自己的"
             "相机解算、把人按那台相机摆进它的世界——数据一个字节都不进上游的推理，所以不是官方输入。"
             "内参（焦距、主点）不是相机输入，完整的内参加外参才算相机；img_focal 对应的是「Focal Length」参数，不是相机口，"
             "`camera_to_worker` 走「不吃相机动画、只用 Focal Length」那一档。要按自己的相机摆人，走显式的小工具节点。"
             "② 「相机」**输出**口是官方的，只是写在另一个文件：`scripts/estimate_camera.py:49-53` 的 "
             "world_cam_R / world_cam_T / img_focal，所以这个声明引了几条上游位置（见 cite）。"
             "③ 「人物」（蒙皮角色）和「网格」是 CG 形态，装的还是官方那份东西："
             "「人物」= VIMO 吐的 SMPL 参数（pred_rotmat 每帧关节旋转 + pred_shape + pred_trans，"
             "hmr_vimo.py:167-172）装成一副带蒙皮和权重的角色——SMPL / SMPL-X / MANO / "
             "FLAME / MHR 这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型；"
             "「网格」= 官方自己拿这些参数跑 SMPL body model 得到的顶点（lib/vis/traj.py:128-134 的 pred_vert）。",
    )
    # 公开基准实测结果，用作输入的提示
    measured = {
        "focal_mm": "实测（3DPW 8 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 31%（7 好 0 差）；接 AnyCalib 估的 Focal Length 更差（2 好 6 差）",
    }
    on_node = ("focal_mm", "static_camera")
    # VIMO 每次处理 16 帧（worker.py MIN_FRAMES）。帧数不足的输入在提交前拒绝，
    # 避免排队、启动进程并加载模型后才在 worker 中报错，浪费 GPU
    min_frames = 16
    # docs.md：自带 DROID-SLAM 相机解算，最适合移动镜头；固定机位有专用开关；
    # 纯摇镜头和长焦压缩镜头的尺度可能不准；填写真实 Focal Length 可使位置误差降低 31%（3DPW 8 个镜头，7 好 0 差）。
    # 「人物框」为可选输入：官方 HMR_VIMO.inference 接受每个人的逐帧框（official 注 ⓪）；未接入时由上游使用
    # ViTDet + SAM + DEVA 检测人物。不提供「相机」输入口（上游 VIMO 仅使用 Focal Length 和主点，相机是其自身解算的产物，见注 ①）
    inputs = (Port("image", "image.3", "RGB"), people_port(optional=True, each=False))  # 人物框为可选输入，见 official 注 ⓪
    runtime = "tram"
    # 上游只向 VIMO 输入 Focal Length（img_focal）和主点，不使用逐帧外参，对应三档中「不使用相机动画、仅使用 Focal Length」
    # 一档（families/humans.py camera_to_worker）。该设置同时移除家族添加的「相机」输入口，
    # 并使「Focal Length」「Filmback」成为普通参数（没有相机口，也就不存在接入相机后置灰的情况）
    camera_to_worker = None
    # 本方法自行解算相机：上游用遮挡人物后的 DROID-SLAM 解出整段轨迹
    # （third_party/tram/repo/scripts/estimate_camera.py:45-53 的 world_cam_R / world_cam_T / img_focal），
    # 属于流程的产物，因此「相机」输出口为官方输出，予以保留（families/humans.py solves_camera）。
    # 其他人体 / 手 / 脸解算器不具备这一点：其相机由人物位置反推，相应输出口已移除。
    solves_camera = True
    smpl_body = "smpl"  # worker 解算的人体模型（worker.py save_person）
    # vram_gb：在 RTX 4090 上测得（docs.md）
    cost = Cost(gpu=True, vram_gb=8.5, seconds_per_frame=1.0)
    licence = Licence(note="代码 MIT，但 SMPL 人体模型、SPEC 相机标定（马普所）和 DEVA 跟踪器（CC BY-NC-SA）都只能研究用，整体按非商用对待。")

    # 继承 WorldHumansParams 仅为使用「固定机位」（上游 --static_camera）。其中的「相机旋转」对本节点无用（camera_to_worker=None，
    # 相机由上游自行解算），家族据此声明移除该参数，并将「固定机位」的提示替换为不涉及该输入的表述（families/humans.py _without_camera_rotate）
    class Params(WorldHumansParams):
        max_people: Literal[1, 2, 5, 20] = measured_param(
            "最多人数", {1: Measured("最快", below=20), 2: Measured("人越少越快", below=20), 5: Measured("人越少越快", below=20), 20: Measured("1080×1920 120 帧全流程约 2 分钟、内存约 9 GB", gb=8.5)},
            default=20, group="人物",
            help="解几个人（上游 --max_humans）。自己检人时跟踪时间最长的优先；接了「人物框」就按框里的顺序（最显眼的在前）取前几个。"
                 "只要主角选 1 最快；群戏调大")


NODES = (Solve,)
