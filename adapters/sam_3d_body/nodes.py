"""Nodes provided by the SAM 3D Body extension."""

from __future__ import annotations

from lab2shot.sdk import (Official, PERSON_ID, ROOT_PATH, SCENE_FILE, CameraLensParams, NodeDef, NodeParams, P, Port, character_of_model, people_port,
                          SkinnedCharacter, WorldHumans, create_stage, focal_param, save_stage,
                          scene_packet, write_boxes, write_character, write_mesh, Cost)


class DetectPeople(NodeDef):
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
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("boxes", "boxes", "人物框"),)
    runtime = "sam_3d_body"
    # 显存没有单独量过：按 ViTDet 检测器的量级估的保守值
    cost = Cost(gpu=True, vram_gb=3.0, vram_measured=False, note="显存是按 ViTDet 检测器量级估的保守值")

    class Params(NodeParams):
        threshold: float = P(0.8, label="检测阈值", help="认定「这是一个人」的把握门槛（0–1）。人被漏掉（远处、遮挡、背影）就调低到 0.5–0.6；把雕像、海报当成人就调高", ge=0.0, le=1.0, group="检测", widget="slider")

    @classmethod
    def cook(cls, ctx):
        import json

        image = ctx.input("image")
        raw = ctx.run_worker(image)
        people = json.loads((raw / "boxes.json").read_text(encoding="utf-8"))["people"]
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
        note="⓪ **「人物框」是可选输入，直接交给官方函数**（规则：官方函数接受人物框就直接把框交给它；"
             "不接受框的项目，在节点图上走「人物框转遮罩」）。官方包里的 "
             "`SAM3DBodyEstimator.process_one_image(img, bboxes=None, …)`（sam_3d_body_estimator.py:64-75，"
             "**是官方包的函数，不是我们写的**）本来就收框；官方 demo 不传框（demo.py:88-91）只是因为它手里没有别的框，"
             "所以自己建了 HumanDetector（:44-49）。「ViTDet 人物框」用的就是那同一个检测器，接进来的仍是官方的框，"
             "中间多的只有「选人」这一步（→ 这个口）。不接框时 worker 走 demo 的路自己检人。"
             "遮罩那条链（「人物框转遮罩」→「图像合成」）留给官方函数**不**收框的项目。"
             "① **官方的整套 MHR 参数就是「人物」这个口**，没有另立类型、也没有另加口（SMPL / SMPL-X / MANO / FLAME / MHR "
             "这类参数化人体就是「蒙皮 + 权重 + 骨架动画」，装成「蒙皮角色」，不另立数据类型；具体多少个点、"
             "多少节骨架不影响这一点）。逐项对上（sam3dbody.py solve_person / "
             "evaluate / rig_data）：`global_rot` + `body_pose_params` + `hand_pose_params` + `scale_params` + "
             "`shape_params` 原样送回 MHR（head.mhr_forward → head.mhr）算出每帧的骨架状态，变成「人物」骨架"
             "每帧的关节矩阵（npz joint_world）；`mhr_model_params` 就是这一步 mhr_forward 返回的那一份中间参数，"
             "我们用的是同一个函数的同一份输出，不是另一份数据；`shape_params` 还变成这个人的静止网格"
             "（npz rest_vertices），配 MHR 自己的蒙皮权重和绑定姿势（npz skin_weights / skin_indices / bind_world）；"
             "`pred_joint_coords` 和 `pred_global_rots`（= out['joint_global_rots']）就是同一副骨架的关节位置和"
             "全局旋转，在 joint_world 的平移和旋转部分里；`pred_pose_raw` 是 body_pose_params 转 euler 之前的"
             "同一份姿势，不是另一份数据。**一个参数都没丢。**"
             "`expr_params` 是**上游自己永远置零的**（`sam_3d_body/models/heads/mhr_head.py:316 "
             "pred_face = pred[:, count : count + self.num_face_comps] * 0`，同一段 :306-307 连下巴也置零），"
             "所以「人物」上没有表情 blendShape——不是我们丢的，是官方就不出。"
             "`mask` 只有调用方自己传遮罩进去、或者打开 use_mask 让 SAM2 现算时才有"
             "（sam_3d_body_estimator.py:135-153），我们两样都不做（worker.py node_solve 只传 bboxes 和 "
             "cam_int），所以它在我们这条路上永远是 None。"
             "上游还给出 `pred_keypoints_2d`（官方自己找的 2D 关键点，worker 已经存进 npz），目前没有对应的输出口；"
             "**这一项装不进蒙皮角色**（它是画面上的点，不是骨架）。"
             "② **节点没有「相机」输出口**（我们自己造出来的输出口不留）：官方只给 focal_length（内参）和 pred_cam_t（人在相机里的位移），从来没解出一台相机；按这两样拼出来的相机是我们造的。要一台相机，从真正解相机的节点（ViPE、TRAM）接，或者用「导入 USD」自己导入；要把人摆进那台相机的世界，接核心节点「相机空间转换」（core.camera_space）。"
             "③ cam_int（sam_3d_body_estimator.py:69）只是内参：内参（焦距、主点）不是相机输入，完整的内参加外参才算相机；"
             "它对应的是「Focal Length」「Filmback」两个参数，不是相机输入。",
    )
    # Benchmark figures shown as the inputs' tooltips.
    measured = {
        "focal_mm": "实测（3DPW 6 个镜头）：填真实 Focal Length，人在镜头里的位置误差少 61%（5 好 1 差）；接 AnyCalib 估的 Focal Length 反而多 150%（6 个全变差）",
    }
    # 结果在相机空间，运动镜头要先解出相机、再用核心节点「相机空间转换」摆进世界；单帧估计 + 平滑
    on_node = ("focal_mm", "smoothing", "hand_refine")
    # 「人物框」可选：接了就按框解（官方的 process_one_image 本来就收 bboxes），不接就自己检出画面里所有人
    inputs = (Port("image", "image.3", "RGB"), people_port(optional=True, each=False))
    runtime = "sam_3d_body"
    camera_to_worker = "focal"  # per-frame focal lengths: zooms are followed
    # 显存没有单独量过：沿用同一模型家族 Fast SAM 3D Body 的数字
    cost = Cost(gpu=True, vram_gb=4.2, seconds_per_frame=0.6, vram_measured=False, note="显存沿用同一模型家族 Fast SAM 3D Body 的实测，SAM 3D Body 本身还没量过")

    class Params(CameraLensParams):
        # a zoom: a focal length wired one per frame is followed frame by frame
        # 节点上没有「相机」进出口（官方不吃整台相机、也不出相机），所以 Focal Length 就是一个普通参数。
        # per_frame：可以一帧一个值（跟变焦）
        focal_mm: float | None = focal_param(per_frame=True)
        hand_refine: bool = P(True, label="手部精修", help="用专门的手部模型细化手指，手势更准，每帧慢约 30%；手不重要或看不清时可以关", group="质量")
        lock_shape: bool = P(True, label="锁定体型", help="整段镜头同一个人用同一个体型（高矮胖瘦不随帧变化），推荐打开；关闭则每帧体型单独估计，会忽胖忽瘦", group="质量")
        smoothing: float = P(0.5, label="平滑强度", help="减少动作抖动：0 不平滑（保留全部细节，会抖），0.5 适中，1 很平滑（快速动作会变软、脚可能滑）", ge=0.0, le=1.0, group="质量", widget="slider")

    @classmethod
    def convert(cls, ctx, raw, job):
        image, used = job.plate, job.lens
        frames = image.meta["frames"]
        w, h = image.meta["width"], image.meta["height"]
        result = raw.result()
        info = {"extension": cls.runtime, "focal_source": result["camera"]["source"], "lens": used.said}
        ctx.stage("写出 USD")
        # 只交蒙皮角色一样，没有点缓存的「网格」口：输出的人一定是骨架加蒙皮网格的形式，DCC 里才能二次修正；
        # 从这份数据里拆出静止的蒙皮人、提取骨架动画是别的节点的事
        stage = create_stage(frames, info)
        # 几乎什么都没解出来的人不交：整段只真正解出一两帧、其余靠补的人，会在相机前留下一个超大的人或远处一具
        # 骨架。判据和参数在家族上（`WorldHumans.people_solved_enough`）：这个节点自己写了 convert，
        # 走不到家族那一句，所以在这里调同一个函数
        keep = cls.people_solved_enough(ctx, raw, result["people"])
        for person in keep:
            name = f"person_{person['id']:02d}"
            character, own, _vertices = _character(raw, person)
            write_character(stage, name, character, own, shot=frames)  # hidden on the frames it was not solved on: no ghost sliding between solves
        save_stage(stage, ctx.outputs["character"] / SCENE_FILE)
        people = [p["id"] for p in keep]
        character = scene_packet(ctx.outputs["character"], frames, "scene.character", people=people, width=w, height=h)
        # 没有「相机」输出：官方只给 focal_length 和 pred_cam_t，人就留在相机空间里。
        # 要摆进某台相机的世界，图上接「相机空间转换」
        return {"character": character}


def _character(raw, person: dict):
    """One solved person as a USD character, the frames it covers and its exact vertices on them [F,V,3].
    All of it stays in the camera's own space: 官方给的就是相机空间的结果，节点不拿一台相机把它摆进世界
    （那一步是核心节点「相机空间转换」）。"""
    import numpy as np

    data = raw.arrays(person["file"])
    frames = [int(f) for f in data["frames"]]
    joint_world = np.asarray(data["joint_world"], np.float64)
    vertices = np.asarray(data["vertices"], np.float64)
    return character_of_model(
        data["joint_names"], data["parents"], bind_world=data["bind_world"],
        anim_world=joint_world, rest_points=data["rest_vertices"], faces=data["faces"], joint_indices=data["skin_indices"],
        joint_weights=data["skin_weights"], uv=data["uv"], uv_faces=data["uv_faces"],
        custom_data={PERSON_ID: person["id"], "lab2shot:frames_detected": len(frames)},
    ), frames, vertices


NODES = (DetectPeople, Solve)
