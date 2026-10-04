"""Nodes provided by the SAM 3D Body extension."""

from __future__ import annotations

from lab2shot.sdk import (rgb_port, Official, PERSON_ID, CameraLensParams, WorkerNode, NodeParams, P, Port, character_of_model, people_port,
                          WorldHumans, focal_param, write_boxes, Cost, window_of)


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
    outputs = (Port("boxes", "boxes"),)
    runtime = "sam_3d_body"
    # 显存没有单独量过：按 ViTDet 检测器的量级估的保守值
    cost = Cost(gpu=True, vram_gb=3.0, vram_measured=False, note=True)

    class Params(NodeParams):
        threshold: float = P(0.8, ge=0.0, le=1.0, group="detection", widget="slider")

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
    version = 4  # 4：撤回 3 的绑定平移（求出的模板本来就在原点地面上，悬空来自「相机空间转换」，改在重定向预处理里处理），只为让 3 的结果重算；2：主点按画面中心在送去像素里的位置（带 overscan 的画布不再用画布中心）
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
        ours={"camera": ""},
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
    cost = Cost(gpu=True, vram_gb=4.2, seconds_per_frame=0.6, vram_measured=False, note=True)

    class Params(CameraLensParams):
        # 节点上没有「相机」进出口（官方不吃整台相机、也不出相机），所以 Focal Length 就是一个普通参数。
        # per_frame：可以一帧一个值（跟变焦）
        focal_mm: float | None = focal_param(per_frame=True)
        hand_refine: bool = P(True, group="quality")
        lock_shape: bool = P(True, group="quality")
        smoothing: float = P(0.5, ge=0.0, le=1.0, group="quality", widget="slider")

    @classmethod
    def prepare(cls, ctx):
        """另交给 worker「画面中心在送去的像素里的位置」principal_px：去畸变画面带 overscan 时，送去的是整块画布，
        画面中心不在画布中心（不对称 overscan 时更明显），worker 的内参主点用它而不用画布的一半。"""
        job = super().prepare(ctx)
        window = window_of(job.plate)
        (x, y), (w, h) = window.offset, window.plate
        return job.with_(extra={"principal_px": [x + w / 2.0, y + h / 2.0]})

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
