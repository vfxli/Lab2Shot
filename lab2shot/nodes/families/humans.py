"""从画面解算人物（身体、手、脸）和相机：SMPL 系列的全身方法。"""

from __future__ import annotations

import numpy as np

from ...data.camera import CameraSamples
from ...errors import Invalid, NothingToCook
from ...messages import Msg
from ...data.packet import Packet
from ...data.payloads import SCENE_FILE, scene_packet
from ...data.skeleton import body_character, model_regions
from ...data.units import M_TO_CM
from ..base import P, Port
from ..lens import CameraLensParams, without_camera_conditions
from .base import Job, RawOutput, WorkerNode
from ..kit.cameras import camera_port, plate_lens, rotation_is_still, send_camera, send_rotation, solved_camera
from ..kit.ports import static_camera_param
from .tracks import Keypoints2D, PersonKeypoints, keypoints2d
from ..applies import Cost


class WorldHumansParams(CameraLensParams):
    """全身方法的参数：相机是否为固定机位，以及「旋转」档方法所接受的相机旋转。"""

    static_camera: bool = static_camera_param()
    # 相机旋转通过一条显式连线提供，而不是接入整台相机：只接受旋转的上游（GVHMR 的 `cam_angvel` 只取 R_w2c 的旋转部分，
    # WHAM 只取轨迹中的四元数）会丢弃位移，接入整台相机会使使用者误以为位移也被使用。
    # 从「拆分相机」的「旋转」端口连线，在图上可见。表示形式（欧拉角 / 四元数 / 矩阵）由节点内部转换，
    # 连线上传递的是项目统一的三维曲线单位：逐帧 XYZ 欧拉角，单位为度。
    # 默认值为 (0,0,0) 而非可空：向量参数只有不可空时才能接线（`core.transform` 的「移动」「旋转」写法相同，
    # 判据见 `nodes/base.py param_port`）。「未接线」与「接入全 0 的连线」可由 `ctx.values` 区分。
    camera_rotate: tuple[float, float, float] = P(
        (0.0, 0.0, 0.0), label="相机旋转", unit="°", group="镜头", widget="vec3", per_frame=True, worker=False, wired=True)


def _with_least_frames(params: type) -> type:
    """为每个人体解算节点添加「最少解出帧数」参数。

    该参数加在家族上而不是由各适配器自行编写：这是所有人体解算器的共同需求，新接入的项目自动具备。

    与 `without_camera_conditions` 一样新建模型类，而不就地修改 FieldInfo：继承同一 Params 的节点共用
    同一个 FieldInfo 对象，就地修改会影响其他节点。"""
    from pydantic import create_model

    # `worker=False`：这是节点侧的筛选条件，不是解算参数；不发送给 worker，调整时复用模型的原始结果
    # （nodes/base.py），不会重新解算。若发送给 worker 则会进入任务指纹，每次修改都会重跑数分钟的 GPU 计算。
    field = P(None, label="最少解出帧数", group="人物", ge=2, placeholder="自动", worker=False)
    return create_model(f"{params.__name__}WithLeastFrames", __base__=params, least_frames=(int | None, field))


def _without_camera_rotate(params: type) -> type:
    """从不接受旋转的节点上移除「相机旋转」，并将「固定机位」替换为不含该连线的版本。

    `WorldHumansParams.camera_rotate` 是只接受旋转的档位（GVHMR / WHAM，`camera_to_worker == "rotation"`）的端口。
    TRAM 继承同一 Params 只是为了使用「固定机位」，其 worker 从不读取 `camera_rotate`（`camera_to_worker=None`，
    上游自行用 DROID-SLAM 解算整台相机）；若参数留在面板上仍可接线，但接入任何内容都不起作用，使用者也无法察觉。
    「固定机位」上「接入「相机旋转」时置灰」的条件随之失去意义，替换为 `static_camera_param(rotation_wire=False)`。

    与 `without_camera_conditions` / `_with_least_frames` 一样新建模型类，而不就地修改：继承同一 Params 的
    节点共用同一个 FieldInfo。pydantic 的 create_model 只能添加字段不能删除，因此从 NodeParams 开始逐个复制其余字段。"""
    import copy

    from pydantic import create_model

    from ..base import NodeParams

    if "camera_rotate" not in params.model_fields:
        return params
    fields: dict = {}
    for name, field in params.model_fields.items():
        if name == "camera_rotate":
            continue
        if name == "static_camera":
            fields[name] = (bool, static_camera_param(rotation_wire=False))
            continue
        fields[name] = (field.annotation, copy.deepcopy(field))
    return create_model(f"{params.__name__}WithoutCameraRotate", __base__=NodeParams, **fields)


def _person_said(person: dict) -> str:
    """节点上对该人物的称呼。各实现的条目结构不同（save_person 提供 name + id，
    sam3dbody.py write_people 只提供 id，SMIRK / Pixel3DMM 只提供 name），因此只在此处统一识别。"""
    name = person.get("name")
    if name:
        return str(name)
    return f"人物 {int(person['id']) + 1}" if "id" in person else "这个人"


class WorldHumans(WorkerNode):
    """从画面得到人物（身体、手、脸）和相机；可连接已知相机，填写或连线提供的 Focal Length 优先于相机的镜头
    （nodes/lens.py）。
    camera_to_worker 表示方法从连接的相机获取的内容（camera_in.npz）："camera" 为整台相机；
    "rotation" 仅为逐帧旋转（GVHMR / WHAM：上游将其转为 cam_angvel 并丢弃平移）；
    "focal" 为逐帧焦距（SAM 3D Body：支持变焦；逐帧连线提供的 Focal Length 同理）；
    None 仅将镜头焦距作为 focal_px 传入（HaMeR、SMIRK、TRAM、Pixel3DMM：上游自行解算相机）。
    只有官方函数本身接受人物框时才提供「人物框」输入（节点在其 `inputs` 中添加 `people_port()`，
    见下方 `inputs` 的说明）；未提供时由方法自身的检测器在画面中查找人物。

    原始数据约定（lab2shot_worker.world_humans）：raw/result.json 包含 "people" [{name, file}]、"space"（"world"：
    以米为单位、+Y 向上的世界，为方法自身的重力对齐世界或节点交给 worker 的输入相机世界；"camera"：每帧的
    OpenCV 相机坐标）、"world"，可选 "camera" {...}；每人一个 npz（body_character 的字段；缺少时报错并指明字段）；
    raw/camera.npz 包含 frames、focal_px、cam_to_world（OpenCV，米），同时解算镜头中心的方法还会写出
    principal_px [F,2]（画面像素）；仅由 `solves_camera` 节点读取。

    每个人只输出一项：蒙皮角色（骨架 + 逐帧动画 + 蒙在骨架上的网格），可在 DCC 中二次修正。不提供点缓存「网格」端口：
    上游均不单独输出网格（GVHMR / WHAM 保存 SMPL 参数；HaMeR 默认不导出网格；Pixel3DMM 导出的是由同一套 FLAME
    参数烘焙的顶点，与蒙皮角色逐点差异小于 1 毫米），拆分蒙皮和提取骨架动画由独立的核心节点完成。
    FLAME / SMPL 模型自带的分区（头皮 / 脸 / 脖子 / 嘴唇 / 鼻子 / 左右耳等）挂在「蒙皮角色」上
    （`io/usd.py write_character` 的 `subsets`）。

    相机空间的结果通过连接的相机逐帧放置，未连接时位于原点处相机的前方。相机输出（仅当 `solves_camera` 时）
    为方法自身解算的相机。Job.notes：无。

    `smpl_body`（"smpl" / "smplh" / "smplx"，方法的模型不属于 SMPL 家族时为 ""）：worker 解算的身体类型。
    它不增加单独的「SMPL 人体」输出：SMPL / SMPL-X / MANO / FLAME / MHR 均为「蒙皮 + 权重 + 骨骼动画（+ 表情）」，
    正是蒙皮角色类型；该声明只记录 worker 解算的身体类型，参数位于角色内部。

    `keypoints`（Keypoints2D）：方法自身的检测器在解算之前于画面上找到的 2D 关键点（ViTPose 的 17 个身体点、
    手部的 21 个点）。声明了该项的节点具有「2D 关键点」输出，其 worker 将 keypoints_2d [F,K,3]（x、y 为画面像素
    坐标，及置信度）和 keypoint_names [K] 写入每个人的 npz（lab2shot_worker.world_humans save_person）。"""
    lens = "pinhole"  # 将画面视为无畸变镜头拍摄：声明需要去畸变的画面

    on_node = ("focal_mm",)
    # 家族本身没有「人物框」输入端口，由节点按上游函数是否接受人物框自行添加：GVHMR（bbx_xys）、TRAM（HMR_VIMO.inference
    # 的 boxes）、HaMeR（predict_pose 的 det_results）、SAM 3D Body 两个节点（process_one_image 的 bboxes）的上游函数
    # 均接受人物框，各自在 `inputs` 中添加了 `people_port(optional=True)`，接入后将人物框交给 worker（prepare 中的
    # `ctx.input_files("boxes")`）；WHAM 的入口只接受 `--video`，因此没有该端口。
    # 上游不接受人物框的节点若只需计算某一人，使用显式的节点链：
    #     人物检测 → 选人 → 人物框转遮罩 → 图像合成（留下）→ 「图像」端口
    # 相乘后画面中只剩该人，上游自身的检测器便只会找到此人。
    inputs = (Port("image", "image.3", "RGB"), camera_port())
    # 每个人一项输出：蒙皮角色（骨架 + 动画 + 蒙在骨架上的网格，可在 DCC 中二次修正），见类的文档字符串
    main = "character"
    outputs = (Port("character", "scene.character", "蒙皮角色"), Port("camera", "scene.camera", "相机"))
    cost = Cost(gpu=True)
    keypoints: Keypoints2D | None = None  # 方法自身检测到的 2D 关键点；None 表示不输出
    smpl_body: str = ""  # 其 worker 所解算的 SMPL 家族身体（"" 表示不属于 SMPL 家族）：仅用于说明解算的是哪种身体
    camera_to_worker: str | None = "camera"
    # 上游是否解算相机。默认为 False：人体 / 手 / 脸方法大多不解算相机，只给出弱透视偏移
    # （SMIRK 的 outputs['cam'] 是 224 裁切图上的三个数）或一个 Focal Length（HaMeR 的 scaled_focal_length）；
    # 由人物位置反推相机并挂在「相机」端口上，会使使用者误以为是解算器的结果。只有声明为 True 的节点才有该端口
    # （TRAM：使用遮挡人物的 DROID-SLAM 实际解算相机）。
    solves_camera: bool = False
    # 该段结果自身的「参照相机」：声明为 True 的节点增加一个 `ref_camera` 输出端口。
    #
    # GVHMR / WHAM 输出的人物位于其自身的世界（原点为该人物起始位置的髋部），该世界配有一台相机，
    # 人物与实拍画面的对齐依赖于它。要使人物在使用者的相机下同样对齐，需用这两台相机计算一个常量修正并施加到人物上。
    # 该步骤由显式的核心节点「相机空间转换」`core.camera_space` 完成（「参照相机」接其「来源相机」），
    # 两台相机在节点图上各有一条连线，使用者可以看到所用的相机。
    #
    # 它与 `solves_camera` 是两回事：
    #   · `solves_camera`：上游自行解算的成品相机（TRAM 的 DROID-SLAM、Pixel3DMM 的面部相机），
    #     端口名为「相机」，使用者可以交付，也可以作为相机接给其他节点；
    #   · `reference_camera`：仅供「相机空间转换」作为「来源相机」使用，端口名为「参照相机」，端口提示中固定写明
    #     此用途。它由上游的两份输出唯一确定（同一身体在相机空间中 + 在世界中）。
    # 两者同时声明为 True 是矛盾的（确实解算出成品相机时应输出成品相机），__init_subclass__ 会立即拒绝。
    reference_camera: bool = False
    default_focal_mm: float | None = None  # 既未提供焦距也未提供相机时假定的镜头（None：使用方法自身的镜头）

    def __init_subclass__(cls, **kw):
        # 该端口定义在家族上：声明了 keypoints 的节点自动增加「2D 关键点」，适配器中只需一行声明
        if cls.keypoints is not None and not any(p.name == "keypoints" for p in cls.outputs):
            cls.outputs = (*cls.outputs, cls.keypoints.port(cls))
        # 「相机」输入端口只在上游确实接受整台相机（`camera_to_worker == "camera"`）时存在。
        # `"rotation"`（GVHMR / WHAM）只接受旋转，通过「相机旋转」参数连线（见 WorldHumansParams）；
        # `"focal"`（SAM 3D Body / Fast SAM 3D Body）和 `None`（HaMeR / SMIRK / TRAM / Pixel3DMM）只接受 Focal Length，
        # 那是「Focal Length」参数，不是相机。将相机空间的结果放入使用者的相机世界是一个显式步骤：
        # 核心节点「相机空间转换」（core.camera_space）。
        # 上游不解算相机时不应存在「相机」输出端口（见上方 solves_camera 声明）。
        if not cls.solves_camera:
            cls.outputs = tuple(p for p in cls.outputs if p.name != "camera")
        # 「参照相机」（声明见 reference_camera）：仅供「相机空间转换」作为参照，不是成品相机
        if cls.reference_camera:
            if cls.solves_camera:
                # 扩展包编写错误（导入时即抛出，不面向使用者）：英文信息保留在代码中，面向使用者的文字才进入消息目录
                raise TypeError(f"{cls.__name__}: declare solves_camera or reference_camera, not both - a method "
                                f"that really solves a camera gives it on its `camera` output, and must not "
                                f"give a second `ref_camera` as well")
            if not any(p.name == "ref_camera" for p in cls.outputs):
                cls.outputs = (*cls.outputs, Port(
                    "ref_camera", "scene.camera", "参照相机",
                    help="这不是成品相机，别交付、别当解算用的相机接给别人。它是这一段结果在"
                         "它自己的世界里配着的那一台：人和实拍对得上就是靠它。"
                         "把它接进「相机空间转换」的「来源相机」，你自己那台接「目标相机」，"
                         "那个节点把人搬到你那台相机的世界里对上画面。"
                         "要一台能用的相机，从真正解相机的节点（ViPE、TRAM）或者「导入 USD」接"))
        # 只有 `"camera"` 档保留「相机」输入端口。`"rotation"` 也不保留：上游只接受旋转和 Focal Length，位移会被丢弃，
        # 接入整台相机属于使用者无法察觉的隐式提取；旋转通过其自身的参数端口，从「拆分相机」显式连线。
        if cls.camera_to_worker not in ("camera",):
            cls.inputs = tuple(p for p in cls.inputs if p.name != "camera")
            # 端口移除后，镜头两个参数上「接入相机时置灰」的条件也需一并移除：
            # 没有相机端口的节点，Focal Length 和 Filmback 为两个普通参数（nodes/lens.py LensParams 档）。
            # 新建参数类而不就地清除 applies：所有继承 CameraLensParams 的节点共用同一个 FieldInfo，
            # 就地修改会一并清除其他节点的置灰条件。
            cls.Params = without_camera_conditions(cls.Params)
        # 「相机旋转」参数只属于接受旋转的档位：其他档位（TRAM 的 None）的 worker 不读取它，参数不应出现在面板上（见 _without_camera_rotate 的说明）
        if cls.camera_to_worker != "rotation":
            cls.Params = _without_camera_rotate(cls.Params)
        # 「最少解出帧数」加在家族上：每个人体解算节点都具备，新接入的项目自动具备（见 _with_least_frames 的说明）。
        # 已自行声明同名参数的节点保持不变。
        if "least_frames" not in cls.Params.model_fields:
            cls.Params = _with_least_frames(cls.Params)
        # 结构上不存在既接受相机又输出相机的节点：不解算相机的节点没有「相机」输出端口，确实解算相机的节点（TRAM、Pixel3DMM）
        # `camera_to_worker` 为 None，没有相机输入端口。因此不需要「接入相机时将相机输出置灰」一类逻辑；
        # 图中需要该相机时，一律从其来源连接（ViPE 或「导入 USD」）。
        super().__init_subclass__(**kw)

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        # 只有 `camera_to_worker == "camera"` 的节点有「相机」输入口（见 __init_subclass__），其余的没有可读的相机
        camera = ctx.input("camera") if cls.camera_to_worker == "camera" else None
        frames = image.meta["frames"]
        lens = plate_lens(ctx, image, default_mm=cls.default_focal_mm)
        extra = {"focal_px": lens.focal_px}
        inputs: dict = {}
        # 只有上游函数本身接受人物框的节点才有「人物框」端口（见 inputs 的说明）。
        # 接入时将 boxes.json 交给 worker，worker 按框解算而不再自行检测；未接入时按上游 demo 的流程自行检测。
        if any(p.name == "boxes" for p in cls.inputs):
            inputs.update(ctx.input_files("boxes"))
        if camera is not None and cls.camera_to_worker == "camera":
            inputs["camera"] = send_camera(ctx, camera, frames, lens.focal_at(frames))
            if "static_camera" in ctx.params:  # 由相机决定是否为固定机位
                extra["static_camera"] = CameraSamples.from_packet(camera, frames).is_still()
        elif cls.camera_to_worker == "rotation" and ctx.values.get("camera_rotate") is not None:
            # 只使用旋转的档位（GVHMR / WHAM）：从「相机旋转」连线获取逐帧欧拉角，
            # 在此转换为 worker 所需的 cam_to_world，位移一律为 0（上游本就不使用位移）。
            # 表示形式的转换在节点内部完成，连线上传递的是项目单位：逐帧 XYZ 欧拉角，单位为度。
            inputs["camera"] = send_rotation(ctx, ctx.values["camera_rotate"], frames, lens.focal_at(frames))
            # 「固定机位」的提示说明「接入「相机旋转」时由该连线决定：不旋转则按固定机位解算」（kit/ports.py
            # static_camera_param）：参数在面板上已置灰，其值必须由该连线决定，与上方整台相机档位的 is_still() 相同。
            # 缺少此项时，GVHMR 的 static_cam 和 WHAM 的固定机位后处理在固定机位上永远不会启用。
            if "static_camera" in ctx.params:
                extra["static_camera"] = rotation_is_still(ctx.values["camera_rotate"], frames)
        elif (camera is not None and cls.camera_to_worker) or (cls.camera_to_worker == "focal" and lens.per_frame):
            inputs["camera"] = send_camera(ctx, None, frames, lens.focal_at(frames))  # 每帧的焦距（接入变焦数据时逐帧不同）
        if "camera" in inputs:
            extra["focal_px"] = None
        return Job(image, extra=extra, inputs=inputs, lens=lens, camera=camera)

    @classmethod
    def least_solved_frames(cls, own: int, asked: int | None) -> int:
        """该方法输出一个人物时，实际解出的帧至少应有多少。

        `own`：该人物输出段的长度（其 `frames`，即 USD 中其可见的帧）；
        `asked`：使用者在「最少解出帧数」中填写的值，留空时为 None。

        2 是不受参数控制的下限：只解出 1 帧的人物无法插值出动作，`lab2shot_shared/poses.py interpolate_poses`
        在两端保持（held at the ends），单帧没有前后可供插值，整段会被保持为该帧的位置，
        表现为一个贴在镜头上、整段静止的巨大人物。因此参数只能比该下限更严格（`ge=2`），不能更宽松。

        自动档（`asked is None`）：取该方法自身声明的下限 `NodeDef.min_frames`
        （GVHMR 16、TRAM 16、Pixel3DMM 16，即上游自身的时序窗口），未声明时取该人物自身输出段长度的四分之一。

        按该人物自身的输出段计算，而非按整段镜头计算：一张脸只在 16 帧中出现、其余帧不可见
        （家族的 convert 会将其在其他帧设为 invisible），属于如实的短结果，不应因镜头较长而丢弃。
        该规则拦截的是声称有 60 帧、实际 59 帧为插值得到的人物。"""
        if asked is not None:
            return max(2, int(asked))
        return max(2, cls.min_frames or own // 4)

    @classmethod
    def people_solved_enough(cls, ctx, raw: RawOutput, people: list[dict]) -> list[dict]:
        """几乎没有解出任何帧的人物不予输出（否则会出现贴在相机上的巨大人物或远处孤立的骨架）。

        一份实现，由两处调用：家族自身的 `convert`（`smirk.face`、`pixel3dmm.face` 在其结果上追加输出，同样经过它），
        以及自行实现 `convert` 的 `sam_3d_body.solve`（`fast_sam_3d_body.solve` 继承之）。
        判据见 `least_solved_frames`，参数位于节点上（「最少解出帧数」，由家族自动添加）。

        未写出 `solved` 时立即报错，不静默放行（`E-HUMANS-NOSOLVED`）：原始数据约定
        （`lab2shot_worker/world_humans.py` 模块开头）要求每个人的 npz 都明确记录实际解出的帧；
        若缺少时按全部解出处理，新接入的项目遗漏该字段时，这一漏洞会无声出现。

        漏检一两帧本身没有问题：没解出的帧上这个人不显示，不插值（照常给出 `N-SAM3DBODY-UNSOLVEDFRAMES`）。
        此处拦截的只是几乎整段都没解出的情况。"""
        kept: list[dict] = []
        for person in people:
            d = raw.arrays(person["file"])
            if "solved" not in set(d):
                raise Invalid(Msg("E-HUMANS-NOSOLVED", node=cls.label, file=person["file"]))
            own, solved = len(d["frames"]), len(d["solved"])
            least = cls.least_solved_frames(own, ctx.params.get("least_frames"))
            if solved < least:
                ctx.say("N-HUMANS-THINPERSON", who=_person_said(person), own=own, solved=solved, least=least)
            else:
                kept.append(person)
        # 检测到的数量、输出的数量、去除的数量：汇总为一条，点击节点右下角的感叹号即可查看
        ctx.say("I-HUMANS-PEOPLE", found=len(people), kept=len(kept), dropped=len(people) - len(kept))
        if people and not kept:  # 全部被去除：空结果不属于错误，输出空结果并说明原因
            raise NothingToCook(Msg("N-HUMANS-NOBODYLEFT", found=len(people)))
        return kept

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        from ...data.units import CV_TO_GL
        from ...io import usd

        image, camera, lens = job.plate, job.camera, job.lens
        frames = image.meta["frames"]
        w, h = image.meta["width"], image.meta["height"]
        result = raw.result()
        flip = np.diag([*CV_TO_GL, 1.0])
        place_all = None
        if result["space"] == "camera":
            place_all = (CameraSamples.from_packet(camera, frames).cam_to_world @ flip if camera is not None
                         else np.repeat(flip[None], len(frames), 0))
        index = {f: i for i, f in enumerate(frames)}
        ctx.stage("写出 USD 人物和相机")
        stage = usd.create_stage(frames, {"extension": cls.runtime, "world": result["world"]})
        names, found = [], []
        for person in ctx.each(cls.people_solved_enough(ctx, raw, result["people"])):
            name, d = person["name"], raw.arrays(person["file"])
            own = [int(f) for f in d["frames"]]
            if cls.keypoints is not None and {"keypoints_2d", "keypoint_names"} <= set(d):
                kp = np.asarray(d["keypoints_2d"], np.float32)  # [F,K,3]：x、y 像素坐标及置信度
                found.append(PersonKeypoints(name, own, kp[..., :2], kp[..., 2] if kp.shape[-1] > 2 else None,
                                             tuple(str(n) for n in d["keypoint_names"])))
            place = place_all[[index[f] for f in own]] if place_all is not None else None
            character = body_character(d, place)
            # 模型自带的分区（FLAME / SMPL 的头皮、脸、脖子、嘴唇、鼻子、左右耳等）随蒙皮角色输出
            usd.write_character(stage, name, character, own, subsets=model_regions(d, character.faces), shot=frames)  # 未解出的帧上不可见
            names.append(name)
        usd.save_stage(stage, ctx.outputs["character"] / SCENE_FILE)
        character = scene_packet(ctx.outputs["character"], frames, "scene.character", people=names, width=w, height=h)
        out = {"character": character}
        # 输出相机的两种节点使用同一段代码：端口名称不同，读取的都是 raw/camera.npz。
        # `camera`（solves_camera）是上游解算的成品相机；`ref_camera`（reference_camera）仅供「相机空间转换」
        # 作为参照，两项声明互斥（由 __init_subclass__ 保证）。
        camera_port_name = "camera" if cls.solves_camera else ("ref_camera" if cls.reference_camera else "")
        if camera_port_name:
            # 输出的始终是上游自身解算的相机，不存在「接入相机时原样透传」的分支：有「相机」输出端口的节点
            # （`tram.solve`、`pixel3dmm.face`）均为 `camera_to_worker=None`，没有相机输入端口，`job.camera` 始终为 None。
            # 需要将某台相机透传到输出时，由使用者从其来源（ViPE / 「导入 USD」）自行连线。
            c = raw.arrays("camera.npz")
            poses = np.asarray(c["cam_to_world"], np.float64) @ flip  # GL 相机轴向
            if place_all is not None:
                poses = flip @ poses  # OpenCV（相机空间）世界：世界坐标同样翻转
            poses[:, :3, 3] *= M_TO_CM
            info = result["camera"] if isinstance(result.get("camera"), dict) else {}
            # principal_px：只有同时解算镜头中心的方法才会写出（如在画面裁切区域内拟合的面部跟踪器）；
            # 未写出时镜头中心为画面中心。
            out[camera_port_name] = solved_camera(
                ctx, image, c["frames"], c["focal_px"], poses, filmback_mm=lens.filmback_mm,
                principal_px=c["principal_px"] if "principal_px" in c.files else None,
                port=camera_port_name,
                info={"extension": cls.runtime, **info, "lens": lens.said})
        if cls.keypoints is not None and "keypoints" in ctx.wanted:  # 仅在需要时写出
            out.update(keypoints2d(ctx, "keypoints", image, found, extension=cls.runtime))
        return out

