"""Nodes provided by the Fast SAM 3D Body extension: the same solve as SAM 3D Body, faster."""

from __future__ import annotations

from adapters.sam_3d_body.nodes import Solve  # this extension requires sam_3d_body (Extension.requires)
from lab2shot.sdk import Official, Cost


class FastSolve(Solve):
    """Same inputs, parameters and outputs as sam_3d_body.solve (its worker writes the same raw results)."""

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
        ours={"camera": ""},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 49%（5 好 1 差）；接 AnyCalib 估的 Focal Length 反而多 166%（6 个全变差）
    # 接法和 SAM 3D Body 全身动作一样：结果在相机空间，要摆进世界接「相机空间转换」
    runtime = "fast_sam_3d_body"
    # vram_gb: RTX 4090
    cost = Cost(gpu=True, vram_gb=4.2, seconds_per_frame=0.2, note=True)


NODES = (FastSolve,)
