"""Sketch2Anim 扩展提供的节点。"""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (DEFAULT_FPS, Official, SCENE_FILE, Cost, Invalid, Job, Licence, Measured, MissingFrames, Msg, NodeParams, P,
                          Port, RawOutput,
                          WorkerNode, create_stage, measured_param, rig_of_model,
                          save_stage, scene_packet, write_rig)

MODEL_FPS = 20.0  # HumanML3D：发布权重的生成帧率（worker.py 重采样到镜头帧率）
MAX_MODEL_FRAMES = 196  # cfg.DATASET.SAMPLER.MAX_LEN
MIN_MODEL_FRAMES = 40  # cfg.DATASET.SAMPLER.MIN_LEN：短于此长度的动作不在训练范围内
# 视角：草图是经 Rx(俯角)·Ry(偏角) 旋转的正交相机所见的人体（上游 utils.py rotate_pose）。训练时俯角在 0–30°、
# 偏角在 -45–45° 之间随机采样（dataset.py 469-473），因此仅提供以下三档，均在训练范围内
VIEWS = {"front": (15.0, 0.0), "side30": (20.0, 30.0), "side45": (20.0, 45.0)}



def figures_in(sketch) -> list[tuple[int, list[tuple[float, float]]]]:
    """接入的草图（tracks2d）→ [(帧号, 18 个关节的 x y)]，仅包含实际绘制过的帧。

    约定与「手画简笔画」`core.draw_figure` 的输出一致：每个关节一条轨迹，
    顺序为 `nodes/handles.py FIGURE_JOINTS`，绘制过的帧 `visible` 为真。"""
    import numpy as np

    from lab2shot.sdk import FIGURE_JOINTS, read_tracks

    data = read_tracks(sketch)
    tracks, visible = np.asarray(data["tracks"], np.float64), np.asarray(data["visible"], bool)
    # 输入必须是一副火柴人：18 个关节，顺序为 nodes/handles.py FIGURE_JOINTS。
    # 其他跟踪点（点跟踪器输出、面部关键点等）点数不符，在此处报错，避免计算中途失败
    if len(tracks) != len(FIGURE_JOINTS):
        raise Invalid(Msg("E-SKETCH2ANIM-NOTFIGURE", count=len(tracks), want=len(FIGURE_JOINTS)))
    frames = list(sketch.meta["frames"])
    out = []
    for k, frame in enumerate(frames):
        if not visible[:, k].all():      # 该帧未画满整副火柴人，不作为关键姿势
            continue
        out.append((int(frame), [(float(x), float(y)) for x, y in tracks[:, k]]))
    return out


class Sketch2AnimMotion(WorkerNode):
    """草图（火柴人的关节坐标）→ 模型生成的完整人体动作。

    草图通过输入口接入，不在本节点上绘制：绘制由「手画简笔画」`core.draw_figure` 完成，本节点只负责解算。
    每个火柴人是一个关键姿势，多个火柴人的髋关节连线即人体的行进路线。模型坐标系的相关处理
    （米制单位、20 帧/秒、草图归一化）均在 worker.py 中；本侧发送像素坐标，并将返回的骨架转换为带 CG 骨骼名
    和 CG 关节轴向的 USD 骨架动画（data/skeleton.py rig_of_model，与「SMPL 转骨架动画」及各解算器的角色
    使用同一转换，Maya 中的重定向可以识别）。"""

    id = "sketch2anim.motion"
    # 引用官方 demo_kp_traj_2d.py 传给模型的 batch 及其输出：
    # 输入关键姿势（pose / pose_2d）、轨迹（hint / hint_2d）和一句文字（text），输出 joints_pred。
    official = Official(
        cite="third_party/sketch2anim/repo/demo_kp_traj_2d.py:253-305",
        # 上游的网络输入为 `batch['pose_2d']`（火柴人的关节坐标）和 `batch['hint_2d']`（髋关节连线构成的路线），
        # 两者由同一份草图计算得到；另有 `batch['text']` 英文描述（节点的「提示词」参数）。不读取画面像素。
        takes={"sketch": "pose_2d"},   # hint_2d（髋关节路线）由同一份草图计算，位于相同代码行
        gives={"skeleton": "joints_pred"},
        note="草图是这个节点的一个输入口，不是解算器身上的参数：解算器只负责解算，画简笔画、拆网格图这类小工具"
             "都是单独的节点（「官方的输入等于解算器的输入，加任何东西都是显式的其他小工具」）。"
             "草图有三种来源，各是一个节点，输出都接进这个节点的「草图」口："
             "棋盘格图（手动指定几行几列）拆成序列图、直接输入序列图、自己画简笔画。"
             "\n"
             "画火柴人的是「手画简笔画」`core.draw_figure`（它有个**可选**的「图像」口当底图），"
             "底图可以来自「拆网格图」`core.split_grid`（棋盘格图）或「读取序列」。"
             "帧范围由这个节点自己的两个参数说了算（起始帧号 / 结束帧号，"
             "和骨骼动作家族 `FreeMotionParams` 同一套说法）。**没有「帧率」**：帧率只在输出设置节点上出现。",
    )
    # 单次最长 196 个模型帧（20 帧/秒下为 9.8 秒）
    runtime = "sketch2anim"
    # 唯一的输入口对应上游的输入：火柴人的关节坐标。上游 `third_party/sketch2anim/repo/demo_kp_traj_2d.py:253-262`
    # 的网络输入为 `batch['pose_2d'] / ['hint_2d'] / ['text']`，即关节坐标、轨迹和英文描述，不读取画面像素。
    # 火柴人不在本节点上绘制（节点图上应能看出草图来源，并可替换来源）：草图由「手画简笔画」
    # （`core.draw_figure`）绘制后接入，其底图可来自「拆网格图」（`core.split_grid`）或「读取序列」。
    inputs = (Port("sketch", "tracks2d", "草图"),)
    outputs = (Port("skeleton", "scene.skeleton", "骨架动画"),)
    on_node = ("prompt", "sketch_view")
    missing_frames = MissingFrames.FAIL
    # vram_gb 与耗时在 RTX 4090 上测得（tests/integration/test_sketch2anim.py 的输出）
    licence = Licence(note="Sketch2Anim 的代码是 MIT，但官方权重用 HumanML3D 训练，HumanML3D 的动作来自 AMASS，"
                           "AMASS 只许学术研究和非商业用途：生成出来的动作只能用于研究和评估。")
    cost = Cost(gpu=True, vram_gb=1.6, whole="一整段一次生成，不是逐帧的活：4 秒的动作去噪 0.5 秒，头一次还要读模型 6 秒")

    class Params(NodeParams):
        # 帧范围：本节点不读取画面，长度由这两个参数决定，含义与骨骼动作家族的 FreeMotionParams 相同。
        # 不提供「帧率」参数（帧率仅在输出设置节点上出现）：模型帧率为 20 帧/秒（HumanML3D，
        # worker.py MODEL_FPS = 20.0），重采样后的帧数由起止帧号决定
        start_frame: int = P(1001, label="起始帧号", group="时间", worker=False,
                             help="生成出来的第一帧是第几帧，影视习惯从 1001 开始。和镜头的帧号对齐，交到 DCC 里时间轴就对得上")
        end_frame: int = P(1120, label="结束帧号", group="时间", worker=False,
                           help="生成到第几帧为止（含这一帧）")
        prompt: str = P("a person walks forward.", label="提示词", group="草图", lines=4,
                        help="一句英文，说清这是什么动作，最好以「A person …」开头，如「A person jumps over a box」。"
                             "模型的文字编码器只认英文（sentence-t5）。和画出来的姿势矛盾时，"
                             "「贴合草图」大就听草图的，「贴合描述」大就听文字的")
        sketch_view: Literal["front", "side30", "side45"] = P(
            "side30", label="草图视角", group="草图",
            option_labels={"front": "正面", "side30": "侧前 30", "side45": "侧前 45"},
            help="画的这个火柴人是从哪个角度看身体的：正面、向左转 30 度、向左转 45 度（都带 15–20 度俯视）。"
                 "选错了生成出来的动作朝向会歪。三档都在模型训练过的角度范围里")
        text_guidance: float = P(7.5, label="贴合描述", group="模型", ge=1.0, le=15.0,
                                 help="模型往那句文字上靠的力度（官方 guidance_scale，默认 7.5）。"
                                      "动作和描述不像就调高；动作僵硬、幅度变小就调低")
        control: float = P(1.0, label="贴合草图", group="模型", ge=0.0, le=2.0,
                           help="模型往画出来的姿势和路线上靠的力度（官方 control_scale，默认 1）。"
                                "生成的姿势和画的差太远就调高；动作别扭、像被硬拽就调低")
        # RTX 4090 实测（4 秒动作，显存均为 1.6 GB）：2 步 0.45 秒、4 步 0.51 秒、8 步 0.56 秒
        steps: Literal[2, 4, 8] = measured_param(
            "去噪步数", {2: Measured("官方默认：4 秒的动作 0.45 秒", flat=True), 4: Measured("慢一成，动作略稳", flat=True),
                     8: Measured("慢两成，提升很小", flat=True)}, default=2, group="模型",
            help="去噪步数。这是个 LCM 模型，官方配置就是 2 步，再多提升很小、时间成倍涨")
        seed: int = P(1234, label="随机种子", group="模型", ge=0,
                      help="同一个种子得到同一段动作。不满意就换一个数字重算，挑一个最好的")
        foot_lock: bool = P(True, label="脚锁定", group="结果",
                            help="上游自带的去脚滑：判断脚踩地的那几帧，把脚钉在原地再解关节旋转。"
                                 "脚本来就该滑的动作（滑冰、拖步）关掉")

    @classmethod
    def prepare(cls, ctx):
        import numpy as np

        # 动作长度等于节点设置的帧范围（上游不读取画面，长度由参数决定）
        fps = DEFAULT_FPS  # 模型帧率为 20 帧/秒，按此默认帧率重采样；节点不提供帧率参数
        first, last = int(ctx.params["start_frame"]), int(ctx.params["end_frame"])
        if last < first:
            raise Invalid(Msg("E-SKETCH2ANIM-BADRANGE", start=first, end=last))
        shot = list(range(first, last + 1))
        frames = len(shot)
        length = max(int(round(frames / fps * MODEL_FPS)), 1)
        if length > MAX_MODEL_FRAMES:
            raise Invalid(Msg("E-SKETCH2ANIM-TOOLONG", frames=frames, most=int(MAX_MODEL_FRAMES / MODEL_FPS * fps),
                              fps=fps, seconds=round(MAX_MODEL_FRAMES / MODEL_FPS, 1)))
        if length < MIN_MODEL_FRAMES:  # 短于训练中的最短动作：仍然计算，但给出警告
            ctx.say("W-SKETCH2ANIM-SHORT", frames=frames, least=int(MIN_MODEL_FRAMES / MODEL_FPS * fps), fps=fps)
        # 从接入的草图读取（由「手画简笔画」`core.draw_figure` 绘制）：
        # tracks2d 中每个关节一条轨迹，仅绘制过的帧 `visible` 为真
        drawn = figures_in(ctx.input("sketch"))
        if not drawn:
            raise Invalid(Msg("E-SKETCH2ANIM-NOPOSE"))
        # 绘制火柴人的帧号 -> 模型帧序号；位于镜头范围外的帧对齐到最近的端点
        keys, xy = [], []
        for frame, joints in drawn:
            k = int(round((frame - first) / fps * MODEL_FPS))
            k = min(max(k, 0), length - 1)
            if k in keys:  # 同一模型帧上有两个火柴人时，以后者为准，并对前者给出提示
                ctx.say("N-SKETCH2ANIM-SAMEFRAME", frame=frame, port="sketch")
                xy[keys.index(k)] = joints
                continue
            keys.append(k)
            xy.append(joints)
        angle_x, angle_y = VIEWS[ctx.params["sketch_view"]]
        sketch = ctx.work / "sketch.npz"
        np.savez(sketch, length=np.int64(length), key_frames=np.asarray(keys, np.int64),
                 key_xy=np.asarray(xy, np.float32))
        # 无底板（plate=None）：本节点不读取画面，帧范围由上述参数决定（通过 notes["frames"] 传递）
        return Job(None, inputs={"sketch": sketch},
                   extra={"angle_x": angle_x, "angle_y": angle_y, "fps": fps, "out_frames": frames},
                   notes={"frames": shot, "keys": keys})

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job):
        import numpy as np

        from lab2shot_shared import motion as mo

        d = raw.arrays("motion.npz")
        info = raw.result()
        names = [str(n) for n in d["names"]]
        parents = np.asarray(d["parents"], np.int64)
        offsets, rot, root = d["offsets"], np.asarray(d["rotations"], np.float64), d["root"]
        frames = job.notes["frames"]
        if len(rot) != len(frames):  # worker.resample 输出的帧数应与此一致，不一致表示 worker 与节点版本不同步
            raise Invalid(Msg("E-SKETCH2ANIM-FRAMES", made=len(rot), want=len(frames)))
        ctx.stage("写骨架动画")

        def stack(rotations, translations):
            m = np.zeros((*rotations.shape[:-2], 4, 4))
            m[..., :3, :3], m[..., :3, 3], m[..., 3, 3] = rotations, translations, 1.0
            return m

        bind_local = stack(np.repeat(np.eye(3)[None], len(names), 0), np.asarray(offsets, np.float64))
        bind_local[0, :3, 3] = 0.0  # 静止姿势位于原点：根关节的位移属于动画，而非骨骼长度
        anim_local = stack(rot, np.repeat(np.asarray(offsets, np.float64)[None], len(rot), 0))
        anim_local[:, 0, :3, 3] = root
        names_cg, bind_cg, anim_cg = rig_of_model(names, parents, mo.world_from_local(bind_local, parents),
                                                  mo.world_from_local(anim_local, parents))
        out = ctx.outputs["skeleton"]
        stage = create_stage(frames, {"extension": cls.runtime, "sketch2anim": info})
        write_rig(stage, "sketch_01", names_cg, parents, bind_cg, anim_cg, frames,
                  custom_data={"generated_by": "Lab2Shot Sketch2Anim"})
        save_stage(stage, out / SCENE_FILE)
        # 向使用者报告绘制姿势与生成结果的偏差，以及 IK 关节拟合的精度
        ctx.say("I-SKETCH2ANIM-DONE", keys=len(job.notes["keys"]), frames=len(frames),
                key_error=info["key_error_cm"], ik_error=info["ik_error_cm"], seconds=info["seconds"],
                generate=info["generate_seconds"])
        if info["path_error_cm"] >= 0:  # 仅在绘制了多个火柴人时才有路线可比较
            ctx.say("I-SKETCH2ANIM-PATH", error=info["path_error_cm"])
        packet = scene_packet(out, frames, "scene.skeleton", people=["sketch_01"])
        packet.meta["sketch2anim"] = info  # 本次的耗时、显存和误差随数据保存，可在面板的「数据信息」中查看
        return {"skeleton": packet}


NODES = (Sketch2AnimMotion,)
