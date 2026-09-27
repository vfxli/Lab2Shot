"""各家族对画面相机的共同处理：接入的相机原样透传、解算的相机写出、需要发送给 worker 的相机发送，以及节点为该画面
使用的镜头（Focal Length / Filmback）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from lab2shot_worker.recon import interpolate_poses

from ...data import units
from ...data.camera import CameraSamples, carried_lens
from ...data.packet import Packet
from ...data.payloads import SCENE_FILE
from ...availability import Because, Not
from ..applies import Wired
from ..expects import SameShot
from ..base import Port
from ..lens import Lens, lens


def pass_camera(ctx, camera: Packet) -> Packet:
    """将接入的相机原样作为本节点的「相机」输出：其文件不做任何修改（帧、帧率、Focal Length、Filmback 均保持原样，
    使用处按帧号对齐，打包进场景时同样如此）。

    此处不按节点上填写的 Focal Length 重建相机：使用者接入的相机（如 3DE 跟踪、已质检的相机）应原样使用，
    若被修改，使用者无从得知。接入相机时这两个参数在面板上置灰（nodes/lens.py CAMERA_HAS_LENS），此处只有复制一条路径。"""
    import shutil

    shutil.copyfile(camera.path(SCENE_FILE), ctx.outputs["camera"] / SCENE_FILE)
    return Packet(ctx.outputs["camera"], camera.type, dict(camera.meta))


def solved_camera(ctx, image: Packet, solved: list[int], focal_px, cam_to_world, *, filmback_mm: float = units.FILMBACK_MM,
                  info: dict | None = None, fy_px=None, principal_px=None, lens: dict | None = None,
                  port: str = "camera", **meta) -> Packet:
    """将解算得到的相机作为本节点的相机输出，覆盖画面的每一帧：`focal_px`（画面宽度下的像素焦距，单个值或每个解算帧
    一个值）和 `cam_to_world`（USD，厘米，每个解算帧一个）位于 `solved` 帧上，其他帧插值得到（并给出警告）。方法提供
    其余内参时一并写出：`fy_px` 和 `principal_px`（(cx, cy) 或每个解算帧一个，像素），按同样方式插值；`lens` 为其解算
    出的畸变。相机与其解算所基于的画面一致：画面所在的画布作为扩边；方法未解算镜头时采用画面携带的镜头，即
    「LensDistortion」从画面中去除的镜头，使交付的相机在合成和 3DE 中可以恢复畸变。"""
    frames = image.meta["frames"]
    solved = [int(f) for f in solved]
    missing = len(set(frames) - set(solved))
    if missing:
        ctx.say("N-CAMERA-INTERPOLATED", count=missing)
    def along(values, width: int = 0):
        """由单个值扩展为每个画面帧一个，或由每个解算帧一个扩展（每项 `width` 列）。"""
        shape = (len(solved), width) if width else (len(solved),)
        v = np.broadcast_to(np.asarray(values, np.float64), shape)
        return np.stack([np.interp(frames, solved, v[:, k]) for k in range(width)], -1) if width else np.interp(frames, solved, v)

    mats = interpolate_poses(dict(zip(solved, np.asarray(cam_to_world, np.float64))), frames)
    from ...data.windows import Window

    window = Window.of(image.meta)  # 解算所基于的画面框及其周围的画布
    carried = lens or carried_lens(image.meta.get("lens"))  # 方法解算出的镜头优先于画面携带的镜头
    # 背板是相机的一项属性（DCC 中相机背板即相机的一个参数）：解算出的相机记录其所基于的画面，
    # 三维视图通过该相机观察时以此画面为背板（按其自身镜头去畸变，view/proxy.py through_picture_file）。
    # 在此处统一记录，每个解算相机的节点自动具备，无需扩展额外编写。
    samples = CameraSamples.solved(frames, window.width, window.height, along(focal_px), mats,
                                   filmback_mm=filmback_mm, info=info, fy_px=None if fy_px is None else along(fy_px),
                                   principal_px=None if principal_px is None else along(principal_px, 2),
                                   lens=carried, overscan=window.overscan, plate=image.fingerprint)
    # `port`：输出端口。绝大多数节点为「相机」；人体家族中仅供「对齐到相机」作参照的相机
    # 为 `ref_camera`（families/humans.py reference_camera）。两种端口读取同一份 raw/camera.npz，写出逻辑只在此处。
    return samples.write(ctx.outputs[port], **meta)


# 求解或测量镜头的节点对其两个 ST-map 输出的命名：各处统一
STMAP_PORTS = ("undistort_stmap", "distort_stmap")


def lens_stmaps(ctx, lens_meta: dict, image: Packet, ports: tuple[str, str] = STMAP_PORTS,
                overscan: str | float = "auto") -> dict:
    """节点求解或测量得到的镜头的两张 ST-map，作为其自身输出（求得畸变的节点应输出能恢复畸变的映射，而不只是系数）。

    `overscan`（选项与「LensDistortion」相同）："auto" 将其放在能容纳所有画面像素矫正后位置的画布上，使画面边角
    不丢失、去畸变图没有空边；"none" 保持在画面框上。整个镜头共用一张，除非镜头在其中变化（变焦）。
    镜头没有可输出的畸变（或本身已是 ST-map）时返回 {}；只计算有连线的端口。"""
    import numpy as np
    from lab2shot_worker.files import write_exr

    from ...data import lens_models as L
    from ...data.payloads import UNIT, channel_names, map_packet, still_packet

    lens = L.Lens.from_meta(lens_meta)
    if lens is None or not any(port in ctx.wanted for port in ports):
        return {}
    frames = image.meta["frames"]
    keys = frames if lens.animated else [None]
    canvas, capped = L.fit_canvas(lens, keys, overscan)
    pairs = {k: L.lens_stmaps(lens, canvas, k) for k in ctx.each(keys)}
    if capped:
        ctx.say("W-LENS-OVERSCANCAP", cap=L.OVERSCAN_CAP, outside=max(p.outside for p in pairs.values()))
    out = {}
    from ...data.windows import Window

    for port, direction in zip(ports, ("undistort", "distort"), strict=True):
        if port not in ctx.wanted:
            continue
        # 去畸变图覆盖画布（画面框为其显示窗口）；加畸变图表示每个画面像素的落点，覆盖画面框本身
        window = canvas if direction == "undistort" else Window(*canvas.plate)
        files = {}
        for key, pair in pairs.items():
            frame = frames[0] if key is None else key
            files[frame] = ctx.outputs[port] / f"frame.{frame}.exr"
            values = pair.undistort if direction == "undistort" else pair.distort
            write_exr(files[frame], values.astype(np.float32), channel_names(2), half=False,
                      windows=window.exr_windows)
        if None in pairs:
            packet = still_packet(ctx.outputs[port], files[frames[0]], frames, *window.plate, channels=2,
                                  value_range=UNIT, window=window)
        else:
            packet = map_packet(ctx.outputs[port], 2, files, *window.plate, UNIT, window)
        packet.meta["direction"] = direction
        out[port] = packet
    return out


def sent_fov_x_deg(lens: Lens, image: Packet) -> float | None:
    """发送给 worker 的像素的水平视场角：覆盖数据包保留的全部像素（其数据窗口；去畸变画面的画布宽于其画面框）。
    焦距未知时为 None。"""
    from ...data import units
    from ...data.windows import Window

    if lens.focal_px is None:
        return None
    return float(units.fov_x_deg(lens.focal_px, Window.of(image.meta).canvas[0]))


    # 方法对接入相机的使用程度，以一句话写在相机端口的悬停提示上（哪些解算器使用整台相机、哪些只使用
    # Focal Length、哪些只使用旋转，使用者应能在端口上直接看出）。
    # 四档即家族声明 `camera_to_worker` 的四个取值，因此「节点如何使用」与「提示如何描述」始终同源。
TAKES_CAMERA = {
    "camera": "接上相机：解出来的人对齐到这台相机的世界，逐帧位姿和 Focal Length 解算器都用上。"
              "接了之后节点上的「Focal Length」「Filmback」变灰——相机自带，交付出去的也是你这台相机，一个字节不动",
    # "rotation"：GVHMR 和 WHAM 的上游只取相机每帧的旋转并丢弃位移：
    # GVHMR：third_party/gvhmr/repo/tools/demo/demo.py compute_cam_angvel(R_w2c)；
    # WHAM：third_party/wham/repo/lib/data/datasets/dataset_custom.py 只取轨迹中的四元数。
    # 直接接入相机会使使用者误以为位移也被使用，因此该档没有「相机」输入端口（families/humans.py __init_subclass__
    # 已移除该端口），旋转通过「相机旋转」参数的接线端口，从「拆分相机」的「旋转」端口连接。
    # 这句话不会出现在任何端口上；保留在四档表中只为说明该档使用相机的哪一部分。
    "rotation": "解算器只用相机**每帧的旋转**（相机怎么转），位移不参与解算。这一档没有「相机」输入口："
                "从「拆分相机」的「旋转」口接到参数「相机旋转」上，Focal Length 同样从「拆分相机」接给「已知 Focal Length」",
    "focal": "接上相机：解算器只用它的 Focal Length（逐帧，跟变焦），位姿不参与解算，但人会按这台相机摆进世界；"
             "交付出去的也是你这台相机，一个字节不动。接了之后节点上的「Focal Length」「Filmback」变灰",
    None: "接上相机：这个方法不吃相机动画，只用它的 Focal Length（它没有「世界」这个概念）。"
          "接了之后节点上的「Focal Length」「Filmback」变灰——相机自带",
}


def camera_port(optional: bool = True, takes: str | None = "camera") -> Port:
    """画面相机的输入端口：提供其镜头和每帧的位置。
    `takes`：方法对其使用的程度（家族的 `camera_to_worker`），写在端口的悬停提示中（TAKES_CAMERA）。"""
    return Port("camera", "scene.camera", "相机", optional=optional, expects=(SameShot(),), help=TAKES_CAMERA[takes])


def send_camera(ctx, camera: Packet | None, frames: list[int], focal_px: np.ndarray) -> Path:
    """worker 读取的画面相机（lab2shot_worker.recon.load_camera）：每帧节点所用的焦距（画面宽度下的像素值：
    镜头的 focal_at，显式填写的 Focal Length 优先于相机的焦距），以及提供 `camera` 时相机在其自身世界中的位置
    （Y 向上），OpenCV 相机轴向，单位为米。未提供相机时只有焦距（接入的变焦数据）。"""
    fields = {"frames": np.asarray(frames), "focal_px": np.asarray(focal_px, np.float64)}
    if camera is not None:
        fields["cam_to_world"] = CameraSamples.from_packet(camera, frames).opencv_m()
    path = ctx.work / "camera_in.npz"
    np.savez(path, **fields)
    return path


def send_rotation(ctx, rotate, frames: list[int], focal_px) -> Path:
    """只接受旋转的档位（GVHMR / WHAM）读取的 camera_in.npz：逐帧 Focal Length，加上由「相机旋转」连线计算出的
    cam_to_world，位移恒为 0。

    位移为 0 而不附带位移，是因为上游本身不使用位移：GVHMR（`third_party/gvhmr/repo/tools/demo/demo.py`）输入网络的是
    `compute_cam_angvel(R_w2c)`，只有旋转；WHAM（`third_party/wham/repo/lib/data/datasets/dataset_custom.py`）只取
    轨迹中的四元数，丢弃位移。用接入相机的位移放置结果属于使用者看不到的隐式操作，因此旋转通过图上
    可见的连线提供；要将人物放入某台相机的世界，需在图上另接「对齐到相机」。

    `rotate`：`ctx.values["camera_rotate"]`，即「拆分相机」输出的「朝向」值：逐帧 XYZ 欧拉角，单位为度，采用本项目的
    朝向约定（`data/scene.py xyz_euler_deg` 的输出）。此处将其转回旋转矩阵，再换为 worker 读取的 OpenCV 轴向和米
    （与 `send_camera` 使用同一 `units.usd_poses_to_opencv_m`）。欧拉角、四元数、矩阵之间的转换在节点内部完成。"""
    from ...data.scene import euler_xyz_matrix

    per = rotate.at(frames).reshape(-1, 3)
    poses = np.repeat(np.eye(4)[None], len(frames), 0)
    for i, deg in enumerate(per):
        poses[i, :3, :3] = euler_xyz_matrix(deg)
    path = ctx.work / "camera_in.npz"
    np.savez(path, frames=np.asarray(frames), focal_px=np.asarray(focal_px, np.float64),
             cam_to_world=units.usd_poses_to_opencv_m(poses), rotation_only=True)
    # `rotation_only`：告知 worker 这不是一台完整相机（recon.InputCamera.rotation_only）。位移恒为 0，
    # 因此不得用它将人物整体移入该相机的世界。
    return path


def rotation_is_still(rotate, frames, tolerance_deg: float = 0.01) -> bool:
    """「相机旋转」连线上的相机在整段中没有旋转（每帧 XYZ 欧拉角与第一帧之差不超过 `tolerance_deg` 度）。

    这是 `CameraSamples.is_still()` 在只接受旋转档位上的对应实现（后者针对整台相机：位置差不超过 0.05 cm、
    朝向差不超过 0.01°；此处没有位置，只看朝向，门槛相同）。「固定机位」参数的提示（kit/ports.py
    static_camera_param）说明「接入「相机旋转」时由该连线决定：不旋转则按固定机位解算」，只接受旋转的档位
    （GVHMR / WHAM）依靠本函数实现该行为，否则固定机位仍会按手持方式解算。"""
    per = np.asarray(rotate.at(frames), np.float64).reshape(-1, 3)
    return bool(len(per) == 0 or np.abs(per - per[:1]).max() <= tolerance_deg)


def plate_lens(ctx, image: Packet, frames: list[int] | None = None, default_mm: float | None = None) -> Lens:
    """节点为其画面采用的镜头（nodes/lens.py：取其 Focal Length，否则取接入相机的镜头，否则取方法自身的镜头）。"""
    return lens(ctx, image.meta["width"], frames or image.meta["frames"], default_mm=default_mm)


def lens_note(focal_px: float, width: int, filmback_mm: float) -> str:
    """用一句话说明解算得到的焦距：与视图及输出相机上标注的数值相同。

    标注的是真实 Focal Length（focal_px ÷ 画面宽 × Filmback，即 `scene.camera` 的 `focal_mm`）及其对应的 Filmback，
    而非摄影中按对角线折算的「35mm 等效」：Maya / Nuke / 3DE 中相机即由「Focal Length + Filmback」两个数描述，
    两种算法并列出现时使用者会误读为同一数值。Focal Length（px）保留在括号中，它是上游 COLMAP / AnyCalib / GeoCalib
    自身解算的原始值。"""
    return f"Focal Length {units.focal_mm(focal_px, filmback_mm, width):.3g} mm（Filmback {filmback_mm:g} mm，{focal_px:.0f} px）"
