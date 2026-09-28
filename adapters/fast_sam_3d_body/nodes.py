"""Nodes provided by the Fast SAM 3D Body extension: the same solve as SAM 3D Body, faster."""

from __future__ import annotations

from adapters.sam_3d_body.nodes import Solve  # this extension requires sam_3d_body (Extension.requires)
from lab2shot.sdk import Official, Cost, Licence


class FastSolve(Solve):
    """Same inputs, parameters and outputs as SAM 3D Body 全身动作 (its worker writes the same raw results)."""

    id = "fast_sam_3d_body.solve"
    # 这个扩展有自己的上游仓库（third_party/fast_sam_3d_body/repo），不能沿用 SAM 3D Body 那份声明：
    # 入口同名但多一个 hand_box_source 参数，输出 dict 也少一项（没有 mhr_model_params），
    # 而且它自己的 demo 默认用 YOLO11 检人，不是 ViTDet。cite 引的是官方脚本的入口，不是模型函数的签名。
    official = Official(
        cite=("third_party/fast_sam_3d_body/repo/demo.py:23-49",      # main()：画面文件夹 + 自己建的 HumanDetector
              "third_party/fast_sam_3d_body/repo/demo.py:86-91",      # 调用处：只传 image_path，没有框
              "third_party/fast_sam_3d_body/repo/demo.py:116-121",    # argparse：--image_folder
              "third_party/fast_sam_3d_body/repo/demo_human.py:623-629",  # 它自己的 demo：--detector 默认 yolo
              "third_party/fast_sam_3d_body/repo/sam_3d_body/sam_3d_body_estimator.py:194-205",  # 官方 process_one_image(img, bboxes=None, …)：收框
              "third_party/fast_sam_3d_body/repo/sam_3d_body/sam_3d_body_estimator.py:375-392"),  # 它吐出来的每一项
        takes={"image": "image_folder", "boxes": "bboxes"},
        gives={"character": "body_pose_params"},
        note="⓪ **「人物框」是可选输入，直接交给官方函数**（规则：官方函数接受人物框就直接把框交给它；不接受框的项目，在节点图上走「人物框转遮罩」）："
             "这个仓库自己的 `process_one_image(img, bboxes=None, …)`（sam_3d_body_estimator.py:194-205，**官方包的函数**）"
             "收框；官方 demo 不传框只是因为它自己建了检测器（demo.py:44-49；demo_human.py `--detector` 默认 `yolo`，"
             "三档 vitdet / yolo / yolo_pose，:623-629）。接了框就按框解；没接时 worker 用 `yolo_pose` 这一档自己检："
             "**一趟同时给出框和手腕**（手部裁切走同一份 YOLO-Pose 手腕，run_publisher.py:305 "
             "`hand_box_source=\"yolo_pose\"`）。接了框而且开着「手部精修」时，YOLO-Pose 仍跑一遍只为拿手腕。"
             "① **官方的整套 MHR 参数就是「蒙皮角色」这个口**，和 SAM 3D Body 一模一样（两个节点共用 "
             "adapters/sam_3d_body/sam3dbody.py 的 solve_person / evaluate / rig_data，逐项怎么对上见那边的 "
             "official 注 ①；SMPL / SMPL-X / MANO / FLAME / MHR 这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，"
             "装成「蒙皮角色」，不另立数据类型）：global_rot + body_pose_params + "
             "hand_pose_params + scale_params + shape_params 原样送回 MHR 算出骨架每帧的关节矩阵（npz joint_world），"
             "shape_params 还给出静止网格（npz rest_vertices）和蒙皮权重（npz skin_weights / bind_world）；"
             "pred_joint_coords / pred_global_rots / pred_pose_raw 都是同一副骨架、同一份姿势的另一种写法。"
             "**一个参数都没丢。** `expr_params` 这个扩展的上游同样自己置零"
             "（`sam_3d_body/models/heads/mhr_head.py:442、508、901 pred_face = pred[...] * 0`），所以没有表情 "
             "blendShape——官方就不出。`character` 这里写的是 body_pose_params 而不是 SAM 3D Body 那边的 "
             "mhr_model_params：这个扩展的输出 dict 里没有 mhr_model_params 那一项（见上面的注释）。"
             "上游还给出 `pred_keypoints_2d`（画面上的点，不是骨架，同样**装不进蒙皮角色**），目前没有对应的输出口。"
             "② cam_int（sam_3d_body_estimator.py:199）只是内参：内参（焦距、主点）不是相机输入，完整的内参加外参才算相机。"
             "节点没有「相机」输出口：官方没有解出相机，按 focal_length + pred_cam_t 拼出来的相机是我们自己造的输出，"
             "我们自己造出来的输出口不留。要把人摆进某台相机的世界，接核心节点「相机空间转换」。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 49%（5 好 1 差）；接 AnyCalib 估的 Focal Length 反而多 166%（6 个全变差）
    # 接法和 SAM 3D Body 全身动作一样：结果在相机空间，要摆进世界接「相机空间转换」
    runtime = "fast_sam_3d_body"
    # vram_gb: RTX 4090
    cost = Cost(gpu=True, vram_gb=4.2, seconds_per_frame=0.2, note='Fast SAM 3D Body 的实测')
    licence = Licence(note="代码 MIT，但沿用的 SAM 3D Body 部分和权重是 SAM License（可商用，禁止军事用途，需附许可证并在论文中注明）；"
        "用来找手腕的 YOLO11-Pose 是 AGPL-3.0（开源传染性许可，分发或对外提供服务时有开源义务）。")


NODES = (FastSolve,)
