"""Nodes provided by the MediaPipe Face extension (Apache-2.0 code and models)."""

from __future__ import annotations

from typing import Literal


from lab2shot.sdk import (rgb_port, Official, SCENE_FILE, Msg, NothingToCook, WorkerNode, NodeParams, P, Port,
                          create_stage, curves_packet, save_stage, scene_packet, tracks_packet,
                          write_mesh, Param, Cost)


class Face(WorkerNode):
    id = "mediapipe_face.face"
    # 引的是官方 Tasks API 自己定义的结果类型 FaceLandmarkerResult：一张画面进去（detect(image)，同文件
    # 3189-3193），出来 face_landmarks / face_blendshapes / facial_transformation_matrixes 三样，没有别的。
    official = Official(
        cite="third_party/mediapipe_face/repo/mediapipe/tasks/python/vision/face_landmarker.py:2893-2905",
        takes={"image": "image"},
        gives={"landmarks": "face_landmarks", "expressions": "face_blendshapes",
               "head": "facial_transformation_matrixes"},
        note="① 「头部」口给的是官方那张 canonical face 网格（上游自带的资源）按 "
             "facial_transformation_matrixes 逐帧摆位，网格本身和矩阵都是官方的。② 没有「相机」输出口："
             "官方的矩阵是相对它自己假设的一台虚拟相机（63° 垂直视场）说的，FaceLandmarkerResult "
             "里没有相机这一项；worker 按那个视场造的相机（worker.py VIRTUAL_VFOV_DEG）"
             "只用来把头部网格摆进世界，不是官方结果，不交出去。要把头放进某台相机的世界，"
             "接核心节点「相机空间转换」（core.camera_space）。③ image 的确切出处是同文件 `3189-3193 def "
             "detect(self, image: image_lib.Image) -> FaceLandmarkerResult`。",
    )
    lens = "pinhole"  # treats the plate as a lens without distortion: says it needs undistorted plates
    on_node = ("max_faces", "mode")
    # 只用 CPU：113 帧的镜头整段约 2.9 秒
    cost = Cost(seconds_per_frame=0.025, note="只用 CPU")
    # 没有「相机」输入口，也没有「相机」输出口：FaceLandmarkerResult 里没有相机，头部矩阵是相对官方那台
    # 虚拟相机（63° 垂直视场）说的。要把脸放进某台相机的世界，接核心节点「相机空间转换」
    inputs = (rgb_port(),)
    outputs = (
        Port("landmarks", "tracks2d", "面部关键点"),
        Port("expressions", "curves", "表情曲线"),
        Port("head", "scene.model", "头部"),  # the face mask mesh, moved by the head's transform every frame
    )
    runtime = "mediapipe_face"

    class Params(NodeParams):
        max_faces: int = P(1, label="最多几张脸", ge=1, le=8, group="检测")
        mode: Literal["video", "image"] = P(
            "video", label="方式", group="检测",
            option_labels={"video": "视频", "image": "逐张"},
        )
        threshold: float = P(0.5, label="检测阈值", ge=0.0, le=1.0, group="检测", widget="slider")
        min_tracking_confidence: float = P(0.5, label="跟住门槛", ge=0.0, le=1.0, group="检测", widget="slider", applies=Param("mode").one_of("video"))
        min_presence_confidence: float = P(0.5, label="存在阈值", ge=0.0, le=1.0, group="检测", widget="slider",
                                           applies=Param("mode").one_of("video"))

    @classmethod
    def convert(cls, ctx, raw, job):
        import numpy as np
        from pxr import Gf, UsdGeom

        image = job.plate
        frames = image.meta["frames"]
        w, h = image.meta["width"], image.meta["height"]
        d = raw.arrays("faces.npz")
        found = d["found"]  # [F, N]
        if not found.any():  # no face in the whole shot: not an error, nothing to give (engine/cook.py)
            raise NothingToCook(Msg("N-MEDIAPIPEFACE-NOFACE", frames=len(frames)))
        n_faces = found.shape[1]
        slots = [i for i in range(n_faces) if found[:, i].any()]
        missing = int((~found[:, slots]).sum())
        if missing:
            ctx.say("W-MEDIAPIPEFACE-MISSING", count=missing, total=len(frames) * len(slots))

        # 2D landmarks: one track per landmark (per face)
        lm = np.nan_to_num(d["landmarks"][..., :2]).transpose(1, 2, 0, 3)  # [N, 478, F, 2]
        tracks = np.concatenate([lm[i] for i in slots])
        visible = np.concatenate([np.repeat(found[None, :, i], lm.shape[1], 0) for i in slots])
        prefix = (lambda i: f"face{i + 1}_") if len(slots) > 1 else (lambda i: "")
        names = [f"{prefix(i)}lm{k:03d}" for i in slots for k in range(lm.shape[1])]
        landmarks = tracks_packet(ctx.outputs["landmarks"], frames, w, h, tracks, visible,
                                  np.full(len(tracks), frames[0]), names, extension="mediapipe_face")

        # blendshape curves, holding the last good value where the face was lost
        curve_names, values = [], []
        for i in slots:
            bs = _hold(d["blendshapes"][:, i], found[:, i])
            curve_names += [f"{prefix(i)}{n}" for n in d["blendshape_names"]]
            values.append(bs)
        curves = curves_packet(ctx.outputs["expressions"], frames, curve_names, np.concatenate(values, 1), extension="mediapipe_face")

        # head: MediaPipe's matrices are canonical face (cm) -> its virtual camera (GL axes, like a USD camera),
        # 相对官方那台虚拟相机（63° 垂直视场，worker.py VIRTUAL_VFOV_DEG）给的，节点原样交出，不按别的相机
        # 换算远近、也不造一台相机出来。要把头摆进某台相机的世界，在图上接核心节点「相机空间转换」
        canon = raw.arrays("canonical_face.npz")
        stage = create_stage(frames, {"extension": "mediapipe_face"})
        for i in slots:
            mats = _hold(d["matrices"][:, i].reshape(len(frames), 16), found[:, i]).reshape(-1, 4, 4).astype(np.float64)
            xf = UsdGeom.Xform.Define(stage, f"/shot/face_{i + 1:02d}")
            op = xf.AddTransformOp()
            for f, m in zip(frames, mats):
                op.Set(Gf.Matrix4d(m.T.tolist()), f)
            tri = canon["triangles"]
            write_mesh(stage, f"/shot/face_{i + 1:02d}/mask", canon["vertices"], tri, uv=canon["uvs"], uv_faces=tri)
        save_stage(stage, ctx.outputs["head"] / SCENE_FILE)
        head = scene_packet(ctx.outputs["head"], frames, "scene.model", width=w, height=h)
        return {"landmarks": landmarks, "expressions": curves, "head": head}


def _hold(values, found):
    """Per-frame values [F, ...]: frames without a face take the nearest earlier (or, at the start, later) good value."""
    import numpy as np

    out = np.array(values, np.float32)
    good = np.flatnonzero(found)
    idx = np.searchsorted(good, np.arange(len(out)), side="right") - 1
    idx = np.where(idx < 0, 0, idx)
    return out[good[idx]]


NODES = (Face,)
