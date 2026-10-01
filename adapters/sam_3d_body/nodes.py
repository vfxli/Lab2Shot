"""Nodes provided by the SAM 3D Body extension."""

from __future__ import annotations

from lab2shot.sdk import (rgb_port, Official, PERSON_ID, CameraLensParams, WorkerNode, NodeParams, P, Port, character_of_model, people_port,
                          WorldHumans, focal_param, write_boxes, Cost)


class DetectPeople(WorkerNode):
    id = "sam_3d_body.detect_people"
    # 官方的检测器函数本身：一张画面进去，一批框出来（SAM 3D Body 的 tools/build_detector.py）
    official = Official(
        cite="third_party/sam_3d_body/repo/tools/build_detector.py:100-137",
        takes={"image": "img"},
        gives={"boxes": "boxes"},
    )
    version = 2
    # 逐帧检测，机位无关；画面里要有人，没有人就什么都框不出来
    on_node = ("threshold",)
    inputs = (rgb_port(),)
    outputs = (Port("boxes", "boxes", "人物框"),)
    runtime = "sam_3d_body"
    # 显存没有单独量过：按 ViTDet 检测器的量级估的保守值
    cost = Cost(gpu=True, vram_gb=3.0, vram_measured=False, note="显存是按 ViTDet 检测器量级估的保守值")

    class Params(NodeParams):
        threshold: float = P(0.8, label="检测阈值", ge=0.0, le=1.0, group="检测", widget="slider")

    @classmethod
    def convert(cls, ctx, raw, job):
        import json

        image = job.plate
        people = json.loads(raw.file("boxes.json").read_text(encoding="utf-8"))["people"]
        if not people:  # nobody in the shot: 人物框 with nobody in them, what comes after decides (Port.takes_empty)
            ctx.say("N-SAM3DBODY-NOPEOPLEFOUND", threshold=float(ctx.params["threshold"]))
        m = image.meta
        return {"boxes": write_boxes(ctx.outputs["boxes"], people, m["width"], m["height"], m["frames"])}


class Solve(WorldHumans):
    """A world-humans node whose worker keeps each person in the camera's frame with the MHR rig's own joints (raw
    person files: frames, joint_world [F,J,4,4] and vertices [F,V,3] in cm in the GL camera frame, bind_world,
    rest_vertices, faces, uv, uv_faces, parents, joint_names, skin_indices, skin_weights). 结果就留在相机空间里：
    官方没有相机进出，要摆进世界接核心节点「相机空间转换」。"""

    id = "sam_3d_body.solve"
    # 引的是官方脚本的入口 demo.py，不只是模型函数的签名：它收 `--image_folder` 一个画面文件夹，自己建
    # HumanDetector 检人，调 process_one_image 时一个框都不传；结果那一份（MHR 的参数、顶点、2D/3D 关键点、
    # focal_length、pred_cam_t）写在 estimator 里，所以后面的 cite 指它的输出 dict
    # （Official 允许几条 cite：上游常把「吃什么」和「吐什么」写在两个文件里）。
    official = Official(
        cite=("third_party/sam_3d_body/repo/demo.py:23-49",       # main()：画面文件夹 + 自己建的 HumanDetector
              "third_party/sam_3d_body/repo/demo.py:87-92",       # 调用处：只传 image_path，没有框
              "third_party/sam_3d_body/repo/demo.py:116-121",     # argparse：--image_folder
              "third_party/sam_3d_body/repo/sam_3d_body/sam_3d_body_estimator.py:64-75",    # 官方包的 process_one_image(img, bboxes=None, …)
              "third_party/sam_3d_body/repo/sam_3d_body/sam_3d_body_estimator.py:194-216"),  # 它吐出来的每一项
        takes={"image": "image_folder", "boxes": "bboxes"},
        gives={"character": "mhr_model_params"},
        ours={"camera": "放人用的针孔相机：原点、不动，焦距 = 节点的「Focal Length」（留空用解算器自己估的 / 默认的），主点在画面中心（families/humans.py plate_camera）"},
        note="⓪ **「人物框」是可选输入，直接交给官方函数**（规则：官方函数接受人物框就直接把框交给它；"
             "不接受框的项目，在节点图上走「人物框转遮罩」）。官方包里的 "
             "`SAM3DBodyEstimator.process_one_image(img, bboxes=None, …)`（sam_3d_body_estimator.py:64-75，"
             "**是官方包的函数，不是我们写的**）本来就收框；官方 demo 不传框（demo.py:88-91）只是因为它手里没有别的框，"
             "所以自己建了 HumanDetector（:44-49）。「ViTDet 人物框」用的就是那同一个检测器，接进来的仍是官方的框，"
             "中间多的只有「选人」这一步（→ 这个口）。不接框时 worker 走 demo 的路自己检人。"
             "遮罩那条链（「人物框转遮罩」→「图像合成」）留给官方函数**不**收框的项目。"
             "① **官方的整套 MHR 参数就是「蒙皮角色」这个口**，没有另立类型、也没有另加口（SMPL / SMPL-X / MANO / FLAME / MHR "
             "这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型；具体多少个点、"
             "多少节骨架不影响这一点）。逐项对上（sam3dbody.py solve_person / "
             "evaluate / rig_data）：`global_rot` + `body_pose_params` + `hand_pose_params` + `scale_params` + "
             "`shape_params` 原样送回 MHR（head.mhr_forward → head.mhr）算出每帧的骨架状态，变成「蒙皮角色」骨架"
             "每帧的关节矩阵（npz joint_world）；`mhr_model_params` 就是这一步 mhr_forward 返回的那一份中间参数，"
             "我们用的是同一个函数的同一份输出，不是另一份数据；`shape_params` 还变成这个人的静止网格"
             "（npz rest_vertices），配 MHR 自己的蒙皮权重和绑定姿势（npz skin_weights / skin_indices / bind_world）；"
             "`pred_joint_coords` 和 `pred_global_rots`（= out['joint_global_rots']）就是同一副骨架的关节位置和"
             "全局旋转，在 joint_world 的平移和旋转部分里；`pred_pose_raw` 是 body_pose_params 转 euler 之前的"
             "同一份姿势，不是另一份数据。**一个参数都没丢。**"
             "`expr_params` 是**上游自己永远置零的**（`sam_3d_body/models/heads/mhr_head.py:316 "
             "pred_face = pred[:, count : count + self.num_face_comps] * 0`，同一段 :306-307 连下巴也置零），"
             "所以「蒙皮角色」上没有表情 blendShape——不是我们丢的，是官方就不出。"
             "`mask` 只有调用方自己传遮罩进去、或者打开 use_mask 让 SAM2 现算时才有"
             "（sam_3d_body_estimator.py:135-153），我们两样都不做（worker.py node_solve 只传 bboxes 和 "
             "cam_int），所以它在我们这条路上永远是 None。"
             "上游还给出 `pred_keypoints_2d`（官方自己找的 2D 关键点，worker 已经存进 npz），目前没有对应的输出口；"
             "**这一项装不进蒙皮角色**（它是画面上的点，不是骨架）。"
             "② 官方只给 focal_length（内参）和 pred_cam_t（人在相机里的位移），从来没解出一台相机。「相机」口交出的是放人用的那台"
             "原点静止针孔相机（焦距就是用的那个），登记在 ours：不是解出来的，三维视图透过它看背板、模板把它当「解算器的相机」一路。"
             "要一台真的相机，从解相机的节点（ViPE、TRAM）接，或者用「导入 USD」；要把人摆进那台相机的世界，接核心节点「相机空间转换」（core.camera_space）。"
             "③ cam_int（sam_3d_body_estimator.py:69）只是内参：内参（焦距、主点）不是相机输入，完整的内参加外参才算相机；"
             "它对应的是「已知 Focal Length」「Filmback」两个参数，不是相机输入。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   focal_mm：实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 61%（5 好 1 差）；接 AnyCalib 估的 Focal Length 反而多 150%（6 个全变差）
    # 结果在相机空间，运动镜头要先解出相机、再用核心节点「相机空间转换」摆进世界；单帧估计 + 平滑
    on_node = ("focal_mm", "smoothing", "hand_refine")
    # 「人物框」可选：接了就按框解（官方的 process_one_image 本来就收 bboxes），不接就自己检出画面里所有人
    inputs = (rgb_port(), people_port(optional=True, each=False))
    runtime = "sam_3d_body"
    camera_to_worker = "focal"  # per-frame focal lengths: zooms are followed
    plate_camera = True  # 「相机」：放人用的原点静止针孔相机（families/humans.py plate_camera）
    # 显存没有单独量过：沿用同一模型家族 Fast SAM 3D Body 的数字
    cost = Cost(gpu=True, vram_gb=4.2, seconds_per_frame=0.6, vram_measured=False, note="显存沿用同一模型家族 Fast SAM 3D Body 的实测，SAM 3D Body 本身还没量过")

    class Params(CameraLensParams):
        # 节点上没有「相机」进出口（官方不吃整台相机、也不出相机），所以 Focal Length 就是一个普通参数。
        # per_frame：可以一帧一个值（跟变焦）
        focal_mm: float | None = focal_param(per_frame=True)
        hand_refine: bool = P(True, label="手部精修", group="质量")
        lock_shape: bool = P(True, label="锁定体型", group="质量")
        smoothing: float = P(0.5, label="平滑强度", ge=0.0, le=1.0, group="质量", widget="slider")

    @classmethod
    def stage_info(cls, result: dict, lens) -> dict:
        return {"extension": cls.runtime, "world": result["world"], "focal_source": result["camera"]["source"], "lens": lens.said}

    @classmethod
    def person_character(cls, person: dict, d, place):
        """MHR 的人：worker 写的是关节世界矩阵和静止网格（不是家族的 SMPL 式数组），已经在 GL 相机空间里（place 不用）。
        交的只有蒙皮角色，没有点缓存的「网格」口：拆静止的蒙皮人、提取骨架动画是别的节点的事。"""
        return f"person_{person['id']:02d}", _character(d, person), None


def _character(data, person: dict):
    """One solved person's npz as a USD character. All of it stays in the camera's own space: 官方给的就是相机空间的
    结果，节点不拿一台相机把它摆进世界（那一步是核心节点「相机空间转换」）。"""
    import numpy as np

    frames = [int(f) for f in data["frames"]]
    return character_of_model(
        data["joint_names"], data["parents"], bind_world=data["bind_world"],
        anim_world=np.asarray(data["joint_world"], np.float64), rest_points=data["rest_vertices"], faces=data["faces"],
        joint_indices=data["skin_indices"], joint_weights=data["skin_weights"], uv=data["uv"], uv_faces=data["uv_faces"],
        custom_data={PERSON_ID: person["id"], "lab2shot:frames_detected": len(frames)},
    )


NODES = (DetectPeople, Solve)
