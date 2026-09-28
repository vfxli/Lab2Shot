"""Nodes provided by the HaMeR extension (MIT code; MANO and the training data research only)."""

from __future__ import annotations

from lab2shot.sdk import Official, CameraLensParams, Keypoints2D, P, WorldHumans, Cost, Licence, people_port


class Hands(WorldHumans):
    id = "hamer.hands"
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
        note="⓪ **「人物框」是可选输入，直接交给官方代码**（规则：官方函数接受人物框就直接把框交给它；不接受框的项目，在节点图上走「人物框转遮罩」）：官方仓库的 "
             "`ViTPoseModel.predict_pose(image, det_results)`（vitpose_model.py:60-72）收的就是人框（xyxy + 分数），"
             "官方 demo 自己也是先用 ViTDet 检出人框（demo.py:47-55、:80-87）再喂给它，由它找出手；包内的 "
             "`ViTDetDataset(cfg, img, boxes, right)` 收的是手框加左右手标记（vitdet_dataset.py:16-25）。"
             "接了框：不再自己检人，按框找手；不接：照官方 demo 的路自己检（同一个 ViTDet-H）。"
             "上游逐张图片跑、不认人，我们交的是一整段镜头里每只手一个文件，所以逐帧的框按 IoU 连成轨迹"
             "（lab2shot_worker.tracking，和「ViTDet 人物框」节点同一份实现）——那是我们的用法，不是它的输入。"
             "① **官方的 MANO 参数就是「蒙皮角色」这个口**，没有另立类型、也没有另加口（SMPL / SMPL-X / MANO / FLAME / MHR "
             "这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型）。逐项对上（worker.py hand_track）："
             "`hamer/models/hamer.py:108 output['pred_mano_params']` 的 global_orient + hand_pose 变成"
             "「蒙皮角色」骨架每帧的关节旋转（npz local_rotations，root 是 global_orient）；betas 变成这只手的"
             "静止网格和静止骨架（npz rest_vertices / rest_joints，按 betas 形变过），配 MANO 自己的蒙皮权重"
             "（npz skin_weights）；`:131 output['pred_keypoints_3d']` 是同一批 MANO 参数跑出来的 21 个点"
             "（16 个骨架关节 + 5 个指尖顶点），我们 worker 用同一个 smplx MANOLayer + joint_map 重算了一遍"
             "（worker.py hand_track 的 tips），骨架关节在「蒙皮角色」里、指尖在网格上，**一个数都没丢**。"
             "MANO 没有表情，所以「蒙皮角色」上没有 blendShape。"
             "② `character -> pred_mano_params` 这个符号写在 `hamer/models/hamer.py:108`、不在 demo.py 里，"
             "所以这个声明引了三条上游位置（官方入口的 argparse、官方 main、模型吐 MANO 参数那一段）。"
             "③ **节点没有「相机」输出口**（我们自己造出来的输出口不留）："
             "官方只有 `demo.py:143 scaled_focal_length`（一个 Focal Length，内参而已），它把手放在这个 Focal Length 的相机空间里，"
             "并没有解出一台相机。手就留在相机空间；要摆进某台相机的世界，接核心节点「相机空间转换」"
             "（core.camera_space）。",
    )
    # 公开基准上的实测（接不接、接什么的差别；没有相机这一条：这个节点没有相机输入，摆进相机的世界是核心节点
    # 「相机空间转换」的事，那条链没测过）：
    #   focal_mm：实测（FreiHAND、DexYCB 8 个镜头）：不给 Focal Length 时 HaMeR 自己假设的 Focal Length 偏 40% 以上，手在镜头里的位置差 42 cm；填真实 Focal Length 降到 3 cm（8 个全变好）；接 AnyCalib 估的 Focal Length 降到 6 cm（7 好 1 差）
    on_node = ("focal_mm", "rescale_factor")
    inputs = WorldHumans.inputs + (people_port(optional=True, each=False),)  # 框可选：official 注 ⓪
    # 每帧单独找手、单独重建，机位无关；手要在画面里几十像素以上；Focal Length 对手的位置影响最大
    runtime = "hamer"
    # 找手就是靠 ViTPose+ 在画面上找的这 21 个点（demo.py），除了框手之外也交出来
    keypoints = Keypoints2D("ViTPose+ 在画面上找的每只手 21 个点（手腕 + 五指的指节和指尖），HaMeR 就是靠它们找到手的")
    camera_to_worker = None
    default_focal_mm = 50.0  # HaMeR's own default lens (5000 px at 256) puts hands ~100 m away
    # RTX 4090 上量得的显存和速度
    cost = Cost(gpu=True, vram_gb=5.9, seconds_per_frame=0.07)
    licence = Licence(note="代码 MIT，但 MANO 手部模型只能研究用、不可再分发，权重用非商用数据训练，整体按非商用对待。")

    class Params(CameraLensParams):
        rescale_factor: float = P(2.0, label="手部框放大", ge=1.0, le=4.0, group="人物")


NODES = (Hands,)
