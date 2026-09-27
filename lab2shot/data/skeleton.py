"""SMPL 系列人体模型的蒙皮（WorldHumans 及将其放入世界坐标系的各家族）：将 worker 输出的逐人 npz 转换为
USD SkinnedCharacter（蒙皮角色）或其逐帧精确顶点。

统一入口为 `character_of_model`：无论人体模型如何命名和定向其关节，Lab2Shot 输出的骨架均使用 CG 骨骼名
（data/joints.py cg_names，Mixamo 命名）和 CG 关节朝向（下方的 cg_orientations），动画师打开交付文件时
看到的是正确的名称和轴向。

骨架本身（SMPL 系列的关节定义，以及参数与关节到世界变换之间的相互转换）由 lab2shot_shared.smpl 实现，
核心与 worker 共用同一份实现。"""

from __future__ import annotations

import numpy as np

from .units import M_TO_CM


# 静止姿势中角色的朝向。人体模型的静止姿势均为标准 T / A 姿势：头朝 +Y，面朝 +Z，角色左手方向为 +X
# （SMPL、SMPL-X、MANO、FLAME 均采用此约定，与本项目内部的「厘米、Y 轴向上」标准一致）。
CG_FORWARD = np.array([0.0, 0.0, 1.0])


def cg_orientations(parents, bind_world: np.ndarray, anim_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 CG 约定规范化关节朝向，返回修改后的 (bind_world, anim_world)。

    ML 人体模型的每个关节都与世界轴向一致（SMPL 静止姿势中所有关节朝向均为单位矩阵）。DCC 中则不同：
    骨骼的主轴沿骨骼本身，绕主轴旋转即骨骼自转（roll），绕另外两轴旋转即弯曲；HumanIK、MotionBuilder、
    UE 的重定向也按此读取骨骼扭转。若不规范化，交付的骨架在 Maya 中每根骨骼的局部轴都朝向世界方向，
    通道编辑器中的曲线难以阅读，也无法镜像。

    采用的约定（Maya「Orient Joint」默认的 XYZ + 世界次轴，每根骨骼一个固定值）：
      * +X 指向子关节（aim 轴，沿骨骼方向）；
      * +Z 尽量贴近角色朝向 CG_FORWARD；骨骼本身朝前或朝后时（脚趾）改用世界上方向，
        使两根参考轴中总有一根与骨骼夹角不小于 45°，避免退化（与重定向中对齐静止姿势使用同一规则和
        同一函数 lab2shot_shared.motion.aim_frame）；
      * +Y = Z × X，右手系；
      * 分叉关节（髋、胸）和末端关节（无子关节）没有唯一的骨骼方向，沿用父关节的朝向（相当于 Maya 中
        将末端关节的 jointOrient 归零）；无父关节且分叉的根关节保持世界轴向。

    动画保持不变：每个关节的静止姿势右乘一个固定旋转 `fix`，逐帧关节矩阵右乘同一个 `fix`。关节位置
    逐帧不变，蒙皮所用的 anim @ inv(bind) 也逐位不变，改变的仅是每根骨骼自身轴的朝向。
    """
    from lab2shot_shared.motion import UP, aim_frame, orthonormal

    parents = np.asarray(parents, np.int64)
    bind = np.asarray(bind_world, np.float64)
    pos = bind[:, :3, 3]
    kids: dict[int, list[int]] = {}
    for j, up in enumerate(parents):
        kids.setdefault(int(up), []).append(j)
    old = orthonormal(bind[:, :3, :3])
    new = old.copy()
    for j in range(len(parents)):  # 父关节先于子关节：处理子关节时父关节已完成
        child = kids.get(j, ())
        bone = pos[child[0]] - pos[j] if len(child) == 1 else np.zeros(3)
        if (length := float(np.linalg.norm(bone))) > 1e-9:
            aim = bone / length
            new[j] = aim_frame(aim, UP if abs(aim @ CG_FORWARD) > abs(aim @ UP) else CG_FORWARD)
        else:
            new[j] = new[parents[j]] if parents[j] >= 0 else np.eye(3)
    fix = np.zeros((len(parents), 4, 4))
    fix[:, 3, 3] = 1.0
    fix[:, :3, :3] = np.swapaxes(old, 1, 2) @ new  # inv(旧朝向) @ 新朝向：只改变轴向，不改变位置
    return bind @ fix, np.asarray(anim_world, np.float64) @ fix


def rig_of_model(joint_names, parents, bind_world: np.ndarray, anim_world: np.ndarray,
                 lone_side: str | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """将模型自身的骨架转换为 CG 约定：返回 (CG 骨骼名, 调整朝向后的 bind_world, 调整朝向后的 anim_world)。
    骨骼名由 joints.cg_names（Mixamo 命名）生成，关节轴由 cg_orientations 规范化。

    这是整个项目中唯一执行该转换的位置：蒙皮角色经由 character_of_model，仅含骨骼的交付（「SMPL 转骨架动画」）
    直接调用此处，两条路径得到的名称和轴向完全一致，同一人物在两种交付物中可以对应。
    `lone_side` ("l" / "r")：仅解算一只手且关节名不含左右的模型（MANO）由此声明是哪只手。"""
    from .joints import cg_names

    bind, anim = cg_orientations(parents, bind_world, anim_world)
    return cg_names([str(n) for n in joint_names], parents, lone_side), bind, anim


def character_of_model(joint_names, parents, *, lone_side: str | None = None, **rest):
    """将人体模型自身的骨架（SMPL / SMPL-X / MHR / MANO / FLAME 等）转换为可供 DCC 使用的蒙皮角色：骨骼使用
    CG 名称（joints.cg_names，Mixamo 命名）和 CG 关节朝向（cg_orientations）。
    这是所有家族共用的唯一入口，关节命名方式不同的新模型无需额外代码即可正确处理。
    `lone_side` ("l" / "r")：仅解算一只手或一侧、且关节名不含左右的模型（MANO）通过此参数传递其声明。"""
    from ..io.usd import SkinnedCharacter
    from .joints import cg_names

    if rest.get("bind_world") is not None and rest.get("anim_world") is not None:
        names, rest["bind_world"], rest["anim_world"] = rig_of_model(
            joint_names, parents, rest["bind_world"], rest["anim_world"], lone_side)
    else:
        names = cg_names([str(n) for n in joint_names], parents, lone_side)
    return SkinnedCharacter(joint_names=names, parents=parents, **rest)


def body_character(d, place: np.ndarray | None = None, top_k: int = 4):
    """person_<id>.npz（米）-> 以厘米为单位的 USD SkinnedCharacter（蒙皮角色）：骨架由模型的关节旋转驱动，
    网格以模型的权重蒙皮（另含其 blend shape，如 FLAME 表情）。`place` [F,4,4]（厘米）将每帧结果移入场景
    （用于相机空间的结果）。"""
    from lab2shot_shared.smpl import world_of

    rest_j = d["rest_joints"].astype(np.float64) * M_TO_CM
    parents = d["parents"].astype(np.int64)
    root = (d["transl"].astype(np.float64) + d["root_rest"].astype(np.float64)) * M_TO_CM
    anim = world_of(parents, rest_j, d["local_rotations"].astype(np.float64), root)
    if place is not None:
        anim = place[:, None] @ anim
    bind = np.repeat(np.eye(4)[None], len(rest_j), 0)
    bind[:, :3, 3] = rest_j
    weights = d["skin_weights"].astype(np.float64)
    idx = np.argsort(-weights, axis=1)[:, :top_k]
    w = np.take_along_axis(weights, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    shapes = "blendshapes" in d  # npz 或普通 dict
    side = str(d["side"]).lower()[:1] if "side" in d else None  # 单手模型在关节旁声明是哪只手（MANO）
    return character_of_model(
        d["joint_names"], parents, lone_side=side, bind_world=bind, anim_world=anim,
        rest_points=d["rest_vertices"].astype(np.float64) * M_TO_CM, faces=d["faces"], joint_indices=idx, joint_weights=w,
        custom_data={"body_model": str(d["body_model"])},
        blendshape_names=[str(n) for n in d["blendshape_names"]] if shapes else [],
        blendshape_offsets=d["blendshapes"].astype(np.float64) * M_TO_CM if shapes else None,
        blendshape_weights=d["blendshape_weights"] if shapes else None,
    )


def model_regions(d, faces) -> dict[str, np.ndarray]:
    """将人体 / 面部模型自带的区域转换为网格上的分区（每个区域 -> 其面片）：
    FLAME 的 `scalp`、`face`、`neck`、`lips`、`nose`、`left_ear` 等，共 14 块。

    依据：FLAME 是参数化模型，所有人的头模顶点编号和面片连接完全相同，只有顶点位置不同。
    因此「第 1312 号顶点位于头皮」对任何人都成立：面部解算将某人的头部拟合为 FLAME 后，这些编号自动
    落在其头皮上，不受头骨形状差异影响（NeuralHaircut 的 cut_scalp.py 第一步同样读取一块固定的模板头皮）。

    区域表由模型作者提供（FLAME 官方下载中的 `FLAME_masks.pkl`），不属于任何头发项目；
    模型未下载或该模型没有区域表时不写入任何内容（分区是附加信息，缺失不影响网格本身）。

    区域表以顶点给出，分区需要面片：三个顶点都在该区域内的面才属于该区域（边界上的面不会同时计入两块）。
    下游「按分区取出」按名称取出一块（头发解算需要 `scalp`：若输入整个头部，脸和脖子上的顶点也会被视为
    发根，发丝会生长到脸上）。"""
    import pickle

    from lab2shot_shared import body_models

    path = body_models.masks_file(str(d["body_model"])) if "body_model" in d else None
    if path is None:
        return {}
    try:
        masks = pickle.loads(path.read_bytes(), encoding="latin1")
    except Exception:  # 文件损坏不应导致整次计算失败：分区是附加信息
        return {}
    tri = np.asarray(faces, np.int64).reshape(len(faces), -1)
    out = {}
    for name, verts in masks.items():
        inside = np.zeros(int(tri.max()) + 1, bool)
        keep = np.asarray(verts, np.int64)
        inside[keep[keep <= tri.max()]] = True
        picked = np.flatnonzero(inside[tri].all(axis=1))
        if len(picked):
            out[str(name)] = picked
    return out


def body_vertices(d, place: np.ndarray | None = None) -> np.ndarray:
    """person_<id>.npz 的逐帧精确顶点 [F,V,3]（厘米，模型自身的网格，包含修正形变），摆放方式与其角色相同。"""
    cache = d["vertices"].astype(np.float64) * M_TO_CM
    return cache if place is None else cache @ np.swapaxes(place[:, :3, :3], 1, 2) + place[:, None, :3, 3]


# ------------------------------------------------------------------ 标准人（核心节点「标准人」所用的身体）


# 使用的身体模型：SMPL-X（`lab2shot_shared.body_models` 表中的 id）。仅此一种，不作为参数：
# SMPL 的中性体型为 .pkl（需要 chumpy 才能读取），SMPL-X 为 .npz，且 SMPL-X 是 SMPL 的超集
# （相同的 22 个身体关节，另加手指、下巴和眼睛），一种身体即可覆盖两种用途。
STANDARD_BODY = "smplx"


def neutral_body(height_cm: float | None = None):
    """中性体型、T-pose、带蒙皮和骨架的人体（`io/usd.py SkinnedCharacter`），厘米，Y 轴向上。

    网格和权重并非本项目生成：读取使用者自行下载并同意许可的身体模型文件
    （SMPL-X 的 `SMPLX_NEUTRAL.npz`，位置见 `lab2shot_shared.body_models`）。使用其静止姿势网格
    `v_template`、官方蒙皮权重 `weights`、官方面片 `f`，关节位置按官方 `J_regressor` 从静止网格回归得到；
    这四项均由模型作者提供，本项目不自行放置任何顶点。

    骨骼名、关节轴和蒙皮取舍经由本文件已有的统一入口 `body_character`
    （解算器输出的每个蒙皮角色都经由它），因此「标准人」与 GVHMR / WHAM / SAM 3D Body 输出的人物
    在骨骼命名、关节朝向和每点影响骨骼数上完全一致，可被重定向和 HumanIK 识别。
    此处不引入新的数学计算：姿势全部为零旋转，`body_character` 的结果即静止姿势本身。

    `height_cm`：将整个人体（关节与顶点）等比缩放到该身高；None 表示使用模型自身的身高。
    返回 (角色, 向使用者报告的数值)。
    """
    from lab2shot_shared import body_models, smpl as S

    from ..errors import Invalid
    from ..messages import Msg

    path = body_models.find(STANDARD_BODY)
    if path is None:
        m = body_models.MODELS[STANDARD_BODY]
        raise Invalid(Msg("E-BODY-NOMODEL", model=m.title, file=m.files[0],
                          folder=str(body_models.ROOT / STANDARD_BODY)))
    body = S.body(STANDARD_BODY)
    with np.load(path, allow_pickle=True) as f:
        template = np.asarray(f["v_template"], np.float64)  # 米，模型自身的静止姿势
        regressor = np.asarray(f["J_regressor"], np.float64)
        weights = np.asarray(f["weights"], np.float64)
        faces = np.asarray(f["f"], np.int64)
    if weights.shape[1] != body.joints:  # 文件与关节表不一致时停止，不得以错误的骨架蒙皮
        raise Invalid(Msg("E-BODY-JOINTCOUNT", model=body.title, file=path.name,
                          got=int(weights.shape[1]), want=body.joints))
    joints = regressor @ template  # 官方关节回归：静止网格 -> 静止姿势中各关节的位置
    tall = float(template[:, 1].max() - template[:, 1].min())  # 模型自身的身高，米
    scale = 1.0 if not height_cm else float(height_cm) / (tall * M_TO_CM)
    # 脚底位于 y = 0：身体模型的原点在骨盆，直接输出时人体会整体位于地面下约 130 厘米。
    # DCC 中的角色资产均站在地面上（「自动落地」也按此计算），因此将整个人体（关节与顶点）上移，
    # 使最低顶点（脚底）位于 0。上移的是静止姿势，动作由其他节点提供，关节间的相对偏移不变
    joints, template = joints * scale, template * scale
    lift = -float(template[:, 1].min())
    joints[:, 1] += lift
    template[:, 1] += lift
    character = body_character({
        "rest_joints": joints, "parents": np.asarray(body.parents, np.int64),
        "transl": np.zeros((1, 3)), "root_rest": joints[0],  # 根关节位于静止位置：单帧，无位移
        # 所有关节为单位旋转（`world_of` 接受旋转矩阵 [F,J,3,3]），即模型的静止姿势（T-pose）本身
        "local_rotations": np.tile(np.eye(3), (1, body.joints, 1, 1)),
        "skin_weights": weights, "rest_vertices": template, "faces": faces,
        "joint_names": list(body.names), "body_model": STANDARD_BODY,
    })
    return character, {"model": body.title, "joints": body.joints, "vertices": len(template),
                       "faces": len(faces), "height_cm": round(tall * M_TO_CM * scale, 1)}
