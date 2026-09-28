"""NVIDIA ViPE worker：相机解算与深度计算。运行于 third_party/vipe/.venv，使用编译好的
`vipe` 包；不导入 Lab2Shot 核心。

    python worker.py <job.json>

一个 worker 提供两条处理路径，分别对应串联的两个节点：

「ViPE 相机解算」（mode = pose_only / pose_only_long）运行到 SLAM 为止，输出三项：
  raw/camera.npz       每帧 camera-to-world 以及整段共用的 Focal Length；
  raw/slam_points.npz  SLAM 建立的三维点，按关键帧分块（上游的 slam_map）；
  raw/instance_<帧>.npz + result.json 的 instance_phrases
                       解算前分割出的运动物体（GroundingDINO + SAM + DeAOT，即官方 save_artifacts 保存的两项）。

「ViPE 深度图」（model = default / dav3）不重新解算：将接入的相机和点云还原为上游所需的
`SLAMOutput` 结构，只运行官方流水线中位姿之后的 post 处理器
（`third_party/vipe/repo/vipe/pipeline/default.py:88-98`），输出
  raw/depth_<帧>.npz  float32 [H, W]，沿相机 Z 轴的距离（米），输入分辨率，另含 "kind"。
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fail, progress, read_frame, save_npz, say, serve, stage
from lab2shot_worker.frame_io import Writer
from lab2shot_worker.recon import load_camera
from lab2shot_worker.run import Run

NODES = ("vipe.camera_solve", "vipe.depth")  # 「ViPE 相机解算」、「ViPE 深度图」：同一 worker 的两条路径（见模块说明）

# 每个允许的预设可加载的内容：流水线类、SLAM 关键帧深度、逐帧深度（None 表示仅相机）。
# 下方组合出的配置会与此核对，确保不会运行其他模型（UniK3D、其他预设的模型）。
PRESETS = {
    "pose_only": (".PoseOnlyAnnotationPipeline", "moge2-l", None),
    "pose_only_long": (".PoseOnlyLongAnnotationPipeline", "moge2-l", None),
    "default": (".DefaultAnnotationPipeline", "unidepth-l", "adaptive_unidepth-l_svda"),
    "dav3": (".DefaultAnnotationPipeline", "dav3", "mvd_dav3"),
}
MASKING = ["GeoCalib pinhole", "GroundingDINO Swin-T", "BERT base uncased", "SAM ViT-B", "DeAOT R50", "DROID-SLAM"]
MODELS = {
    "pose_only": MASKING + ["MoGe-2 ViT-L"],
    "pose_only_long": MASKING + ["MoGe-2 ViT-L"],
    "default": MASKING + ["UniDepth-V2 ViT-L", "Prior-Depth-Anything ViT-B (Depth-Anything-V2-Base)",
                          "Video-Depth-Anything-Small"],
    "dav3": MASKING + ["Depth-Anything-3 Metric-Large", "Depth-Anything-3 Giant"],
}
LICENSE = {
    "pose_only": "commercial use OK",
    "pose_only_long": "commercial use OK",
    "default": "NON-COMMERCIAL: UniDepth-V2 (CC-BY-NC-4.0), Depth-Anything-V2-Base (CC-BY-NC-4.0)",
    "dav3": "NON-COMMERCIAL: Depth-Anything-3 Giant (CC-BY-NC-4.0)",
}
DEPTH_SOURCE = {
    "default": (
        "ViPE adaptive depth: Video-Depth-Anything-Small relative depth for the whole shot, aligned per frame "
        "(momentum-smoothed scale/shift in inverse depth) to a metric prompt: Prior-Depth-Anything completing the "
        "projected SLAM map when it covers the frame, otherwise UniDepth-V2-L. Sky/far pixels are capped at 1000 m"
    ),
    "dav3": (
        "ViPE multi-view depth: Depth-Anything-3 Giant on sliding windows of 10 frames (3 overlapping, cross-faded) "
        "plus neighbouring SLAM keyframes, conditioned on the solved cameras and focal, so the scale follows the SLAM "
        "(approximately metric) solve"
    ),
}
INSTANCE_SOURCE = (
    "ViPE moving-object instances: GroundingDINO finds the phrases (person, animal, vehicle, ball, balloon, gun, pet, "
    "car, bus) plus sky, SAM turns each box into a mask and DeAOT tracks it through the shot; one id per object, kept "
    "the same on every frame. ViPE segments them to mask the solve, so they are what the camera solve ignored"
)
CONVENTION = "OpenCV camera: +X right, +Y down, +Z forward; cam_to_world maps camera to world (world = first frame's camera, approximately metric)"


def make_stream(paths: list[Path], fps: float, name: str):
    """将任务的标准化 PNG 按任务帧顺序（而非按文件名排序）封装为 ViPE 流。"""
    from vipe.streams.base import VideoFrame, VideoStream

    class _Stream(VideoStream):
        def __init__(self) -> None:
            super().__init__()
            self._height, self._width = read_frame(paths[0]).shape[:2]

        def frame_size(self) -> tuple[int, int]:
            return (self._height, self._width)

        def fps(self) -> float:
            return fps

        def name(self) -> str:
            return name

        def __len__(self) -> int:
            return len(paths)

        def __iter__(self):
            for i, path in enumerate(paths):
                rgb = torch.as_tensor(read_frame(path, "float32")).cuda()
                yield VideoFrame(raw_frame_idx=i, rgb=rgb)

    return _Stream()


def make_progress_processor(total: int):
    """在 SLAM 消费帧时报告进度（运行于 ViPE 的预取线程）。"""

    class _Progress:
        n_passes_required = 1

        def __init__(self) -> None:
            self.done = 0

        def update_fps(self, fps):
            return fps

        def update_frame_size(self, size):
            return size

        def update_attributes(self, attributes):
            return attributes

        def update_iterator(self, iterator, pass_idx):
            for frame_idx, frame in enumerate(iterator):
                yield self(frame_idx, frame)

        def __call__(self, frame_idx, frame):
            self.done += 1
            if self.done % 4 == 0 or self.done == total:
                progress(self.done, total, "跟踪")
            return frame

    return _Progress()


def hydra_value(text: str) -> str:
    r"""返回写入 Hydra 覆盖串时的值形式：加引号。

    Hydra 的覆盖语法有独立的词法器，值中出现其无法识别的字符（中文、空格、逗号、冒号等）时
    会抛出 `LexerNoViableAltException`，且不指明具体字符。
    输出路径位于工作文件夹下，工作文件夹由使用者指定（`LAB2SHOT_WORK_DIR` 或安装时选择的位置），
    路径含中文（如「D:\项目\Lab2Shot」）时会失败。加引号后 Hydra 将其作为字符串原样接受，不经词法器解析。

    反斜杠和单引号须转义（Windows 路径使用反斜杠）。"""
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def compose_config(mode: str, out_dir: Path, fixed_focal: bool):
    from vipe.config import parse_typed_config

    overrides = [
        f"pipeline={mode}",
        f"pipeline.output.path={hydra_value(str(out_dir))}",
        "pipeline.output.save_artifacts=false",
        "pipeline.output.save_viz=false",
        "pipeline.slam.visualize=false",
    ]
    if fixed_focal:
        # "gt" 表示内参已知：bundle adjustment 保持焦距固定。
        overrides.append("pipeline.init.intrinsics=gt")
    cfg = parse_typed_config("default", hydra_args=overrides)

    pipe = cfg.pipeline.to_dictconfig()
    instance = str(pipe.instance)
    depth = str(pipe.slam.keyframe_depth)
    want_instance, want_depth, want_post = PRESETS[mode]
    post = str(pipe.post.depth_align_model) if want_post is not None else None
    if not instance.endswith(want_instance) or depth != want_depth or post != want_post:
        fail("E-VIPE-PRESET", preset=str(mode), instance=instance, depth=depth, post=str(post))
    return cfg


def use_local_checkpoints(hf_dir: Path, repo_dir: Path) -> None:
    """default / dav3：UniDepth、Prior-Depth-Anything 和 DA3 从 Hugging Face Hub 加载检查点；
    此处将其指向 weights/hf（由安装器固定版本）。"""
    import vipe.priors.depth.dav3.model as dav3_model

    # 上游打包缺陷：vipe/priors/depth/dav3/model/dinov2/ 没有 __init__.py，构建出的包中缺少该目录
    # （仅 editable 安装可见），因此从编译该包所用的固定版本检出中获取。
    dav3_model.__path__.append(str(repo_dir / "vipe" / "priors" / "depth" / "dav3" / "model"))
    local = {
        "lpiccinelli/unidepth-v2-vitl14": hf_dir / "unidepth-v2-vitl14",
        "Rain729/Prior-Depth-Anything": hf_dir / "prior-depth-anything",
        "depth-anything/DA3METRIC-LARGE": hf_dir / "DA3METRIC-LARGE",
        "depth-anything/DA3-GIANT": hf_dir / "DA3-GIANT",
    }

    def path(repo_id: str, filename: str) -> str:
        folder = local.get(repo_id)
        if folder is None or not (folder / filename).is_file():
            raise RuntimeError(f"ViPE tried to download {repo_id}/{filename}, which the installer did not provide")
        return str(folder / filename)

    import vipe.priors.depth.priorda.priorda as priorda
    from vipe.priors.depth.dav3.api import DepthAnything3
    from vipe.priors.depth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2

    # PyTorchModelHubMixin：包含 config.json + model.safetensors 的本地文件夹
    unidepth_from_pretrained = UniDepthV2.from_pretrained
    UniDepthV2.from_pretrained = classmethod(
        lambda cls, repo_id, **kw: unidepth_from_pretrained(str(Path(path(repo_id, "config.json")).parent), **kw)
    )
    da3_from_pretrained = DepthAnything3.from_pretrained.__func__

    def da3(cls, repo_id, *, model_name=None, weights_path=None, **kw):
        return da3_from_pretrained(
            cls, repo_id, model_name=model_name, weights_path=weights_path or path(repo_id, "model.safetensors"), **kw
        )

    DepthAnything3.from_pretrained = classmethod(da3)
    priorda.hf_hub_download = lambda repo_id, filename, **kw: path(repo_id, filename)
    # Video-Depth-Anything-Small 经 torch.hub 从 TORCH_HOME/hub/checkpoints 加载。


def save_depth(depth_stream, mode: str, frame_numbers: list[int], raw: Path, height: int, width: int) -> dict:
    """逐帧运行预设的深度处理器 -> raw/depth_<frame>.npz。

    运动物体编号图不在此处输出：它是解算阶段的产物（`default.py:70-78` 的 `_add_init_processors`），
    由「ViPE 相机解算」输出（save_instances），再经「物体分割」连线接入本节点。"""
    import torch.nn.functional as F

    stage("估计深度")
    # 直接消费流的生成器：外层的缓存包装会把每一帧（RGB、遮罩、深度）保留在内存中直到结束。
    frames_iter = depth_stream.iterator if getattr(depth_stream, "iterator", None) is not None else iter(depth_stream)
    n = len(frame_numbers)
    writer = Writer(threads=2, max_pending=8)
    medians: list[float] = []
    lo, hi = math.inf, -math.inf
    count = 0
    for i, frame in enumerate(frames_iter):
        if i >= n:
            raise RuntimeError(f"ViPE returned more than {n} depth frames")
        depth = frame.metric_depth
        if depth is None:
            raise RuntimeError(f"ViPE returned no depth for frame {frame_numbers[i]}")
        depth = depth.float()
        if tuple(depth.shape) != (height, width):
            depth = F.interpolate(depth[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        depth = depth.cpu().numpy().astype(np.float32)
        valid = depth[depth > 0]
        medians.append(float(np.median(valid)) if valid.size else 0.0)
        if valid.size:
            lo, hi = min(lo, float(valid.min())), max(hi, float(depth.max()))
        writer.npz(raw / f"depth_{frame_numbers[i]}.npz", depth=depth, kind=np.str_("metric_depth_m"))
        count += 1
        progress(count, n, "深度")
    writer.close()
    if count != n:
        raise RuntimeError(f"ViPE returned {count} depth frames for {n} frames")
    return {
        "depth": "depth_<frame>.npz: depth float32 [height, width], kind",
        "depth_kind": "metric_depth_m",
        "depth_units": "meters (camera Z depth, not ray length; same approximate metric scale as the camera)",
        "depth_source": DEPTH_SOURCE[mode],
        "depth_range": [lo, hi],
        "depth_median_per_frame": medians,
    }


def save_slam_points(slam, raw: Path) -> dict:
    """SLAM 建立的三维点 -> raw/slam_points.npz（OpenCV 世界坐标，米，按关键帧分块）。

    上游将其存放在 `SLAMOutput.slam_map` 中，按关键帧分块
    （`third_party/vipe/repo/vipe/slam/interface.py:27-38`：`dense_disp_xyz` 为所有点的拼接，
    `dense_disp_packinfo` 记录各块起止，`dense_disp_frame_inds` 记录各块对应的流内帧序号）。
    官方同样单独保存该数据（`pose_only.py:104-106` 的 save_slam_map）。
    此处原样输出，供「ViPE 深度图」还原：深度计算时逐帧将其投影到画面上作为提示
    （`processors.py:297`、`:320` 的 project_map）。"""
    if slam.slam_map is None:
        fail("E-VIPE-NOSLAMMAP")
    m = slam.slam_map
    xyz, rgb = [], []
    for k in range(len(m.dense_disp_frame_inds)):
        p, c = m.get_dense_disp_pcd(k, -1)
        xyz.append(p.detach().cpu().numpy().astype(np.float64))
        rgb.append(c.detach().cpu().numpy().astype(np.float32))
    save_npz(
        raw / "slam_points.npz",
        xyz=np.concatenate(xyz) if xyz else np.zeros((0, 3)),
        rgb=np.concatenate(rgb) if rgb else np.zeros((0, 3), np.float32),
        counts=np.asarray([len(p) for p in xyz], np.int64),
        frames=np.asarray(list(m.dense_disp_frame_inds), np.int64),  # 流内序号，而非原始帧号
    )
    return {"points": "slam_points.npz: xyz [M,3] metres in the camera world, rgb [M,3] 0..1, "
                      "counts [K] points per keyframe, frames [K] the keyframe's index in the shot",
            "points_source": "ViPE's SLAM map (slam_output.slam_map): the dense-disparity points DROID-SLAM "
                             "back-projected at each keyframe, in the solved camera's world",
            "points_keyframes": len(xyz), "points_total": int(sum(len(p) for p in xyz))}


def save_instances(stream, frame_numbers: list[int], raw: Path, height: int, width: int) -> dict:
    """解算时分割出的运动物体实例 -> raw/instance_<帧>.npz + 词表。

    GroundingDINO + SAM + DeAOT 是解算阶段的处理器（`default.py:70-78` 的 `_add_init_processors`），
    在 SLAM 之前已运行完毕，帧仍在缓存中（`CachedVideoStream.data`），因此此处不重新运行任何模型，
    仅读取已计算的编号图。官方 save_artifacts 保存的即这两项（`vipe/utils/io.py:359-380`）。"""
    import torch.nn.functional as F

    phrases: dict[int, str] = {}
    written = 0
    writer = Writer(threads=2, max_pending=8)
    for i, frame in enumerate(stream):
        if i >= len(frame_numbers):
            break
        instance = frame.instance
        if instance is not None:
            inst = instance.float()
            if tuple(inst.shape) != (height, width):
                # 编号不得插值：两个编号之间的值会成为一个不存在的第三个物体
                inst = F.interpolate(inst[None, None], size=(height, width), mode="nearest")[0, 0]
            writer.npz(raw / f"instance_{frame_numbers[i]}.npz", instance=inst.cpu().numpy().astype(np.uint8))
            written += 1
        if frame.instance_phrases:  # 上游在整段镜头范围内合并（io.py:372-381）
            phrases.update({int(k): str(v) for k, v in frame.instance_phrases.items()})
    writer.close()
    return {
        "instance": "instance_<frame>.npz: instance uint8 [height, width], 0 = background, else the object's id",
        "instance_frames": written,
        "instance_phrases": {str(k): v for k, v in sorted(phrases.items())},
        "instance_source": INSTANCE_SOURCE,
    }


def quaternion_of(rotations: np.ndarray) -> np.ndarray:
    """旋转矩阵 [N,3,3] -> 四元数 [N,4]，顺序为 (x, y, z, w)，与 lietorch 的 SE3 一致
    （`third_party/vipe/repo/vipe/ext/lietorch/groups.py:281` 的 id_elem = [0,0,0,0,0,0,1]，最后一位为 w）。"""
    m = np.asarray(rotations, np.float64)
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = np.zeros((len(m), 4), np.float64)
    big = t > 0
    if big.any():  # 迹为正时的数值稳定分支
        s = np.sqrt(t[big] + 1.0) * 2
        q[big, 3] = 0.25 * s
        q[big, 0] = (m[big, 2, 1] - m[big, 1, 2]) / s
        q[big, 1] = (m[big, 0, 2] - m[big, 2, 0]) / s
        q[big, 2] = (m[big, 1, 0] - m[big, 0, 1]) / s
    for i in np.flatnonzero(~big):  # 否则以最大对角元为主元（Shepperd 方法）
        a = int(np.argmax([m[i, 0, 0], m[i, 1, 1], m[i, 2, 2]]))
        b, c = (a + 1) % 3, (a + 2) % 3
        s = np.sqrt(1.0 + m[i, a, a] - m[i, b, b] - m[i, c, c]) * 2
        q[i, a] = 0.25 * s
        q[i, b] = (m[i, b, a] + m[i, a, b]) / s
        q[i, c] = (m[i, c, a] + m[i, a, c]) / s
        q[i, 3] = (m[i, c, b] - m[i, b, c]) / s
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rebuild_slam_output(job, n_frames: int, width: int, height: int):
    """接入的「相机」和「点云」 -> 上游 post 阶段所需的 SLAMOutput。

    该函数不重新运行 SLAM，而是将已解算的结果还原为上游结构。深度处理器只从 slam_output 读取三项：
    `trajectory`（每帧位姿）、`intrinsics`（内参）、`slam_map`（三维点）
    （由 `third_party/vipe/repo/vipe/pipeline/default.py:88-98` 组装，
    `processors.py:223` 读取 trajectory，`:279`/`:297`/`:320` 读取 slam_map，`:403` 读取 dense_disp_frame_inds）。
    `backend_graph` 仅在 `secondary_keyframe=True` 时使用（`processors.py:430-437`），
    上游从不开启该选项（`default.py:96` 未传入，`processors.py:385` 默认 False），因此无需还原。"""
    from vipe.ext.lietorch import SE3
    from vipe.slam.interface import SLAMMap, SLAMOutput

    cam = load_camera(job)
    if cam is None:
        fail("E-VIPE-NOCAMERA")
    c2w = np.asarray(cam.cam_to_world, np.float64)
    if len(c2w) != n_frames:
        fail("E-VIPE-CAMFRAMES", camera=len(c2w), frames=n_frames)
    pose = np.concatenate([c2w[:, :3, 3], quaternion_of(c2w[:, :3, :3])], axis=1)
    traj = SE3(torch.as_tensor(pose, dtype=torch.float32, device="cuda"))
    rig = SE3(torch.as_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda"))
    focal = float(np.median(np.asarray(cam.focal_px, np.float64)))
    intrinsics = torch.as_tensor([[focal, focal, width / 2, height / 2]], dtype=torch.float32, device="cuda")

    pts = np.load(job.inputs["points"])
    counts = np.asarray(pts["counts"], np.int64)
    if not counts.size or int(counts.sum()) == 0:
        fail("E-VIPE-NOPOINTS")
    starts = np.cumsum(counts) - counts
    packinfo = torch.as_tensor(np.stack([starts, counts], -1).reshape(len(counts), 1, 2), dtype=torch.long, device="cuda")
    slam_map = SLAMMap(
        dense_disp_xyz=torch.as_tensor(np.asarray(pts["xyz"], np.float32), device="cuda"),
        dense_disp_rgb=torch.as_tensor(np.asarray(pts["rgb"], np.float32), device="cuda"),
        dense_disp_packinfo=packinfo,
        dense_disp_frame_inds=[int(f) for f in np.asarray(pts["frames"], np.int64)],
    )
    return SLAMOutput(trajectory=traj, intrinsics=intrinsics, rig=rig, slam_map=slam_map), focal


def given_objects(job, frame_numbers: list[int]):
    """流处理器：将「物体分割」输入的编号图转换为每帧的 instance / mask / camera_type。

    上游在解算阶段用 `TrackAnythingProcessor` 计算编号图，并同时计算可用像素：
    `frame.mask = erode(instance == 0 | sky_mask, 5)`（`third_party/vipe/repo/vipe/pipeline/processors.py:162-168`）。
    深度计算时读取该 mask（`processors.py:326` 和 `:341-343`）。
    解算节点已输出编号图，因此此处按上游同一行公式重新计算 mask，不重新运行三个模型。
    输入为 None（未接入「物体分割」）时 mask 为空；上游两处均为 `if frame.mask is not None`，计算照常进行。

    编号图采用项目中向 worker 传图的统一方式：一份 JSON 清单（`{"frames": {帧号: 路径}}`，由 `job.listing` 读取，
    路径相对清单所在文件夹），每帧一张单通道图，像素值即编号。天空编号由参数 `sky` 提供（节点侧根据类别表计算）。"""
    from lab2shot_worker.files import read_mask
    from vipe.streams.base import FrameAttribute, VideoFrame
    from vipe.utils.cameras import CameraType
    from vipe.utils.morph import erode

    paths = job.listing("objects")
    sky = {int(i) for i in (job.params.get("sky") or [])}

    class _Given:
        n_passes_required = 1

        def update_fps(self, fps):
            return fps

        def update_frame_size(self, size):
            return size

        def update_attributes(self, attributes):
            got = {FrameAttribute.CAMERA_TYPE}
            return attributes | (got | {FrameAttribute.INSTANCE, FrameAttribute.MASK} if paths else got)

        def update_iterator(self, iterator, pass_idx):
            for frame_idx, frame in enumerate(iterator):
                yield self(frame_idx, frame)

        def __call__(self, frame_idx, frame):
            frame.camera_type = CameraType.PINHOLE  # 仅支持针孔（节点的 lens = "pinhole"）
            number = frame_numbers[frame_idx] if frame_idx < len(frame_numbers) else None
            path = paths.get(number)
            if path is None:
                return frame
            ids = np.rint(read_mask(path)).astype(np.int64)  # 单通道图，像素值即编号
            instance = torch.as_tensor(ids, device=frame.rgb.device)
            frame.instance = instance.to(torch.uint8)
            # 天空编号也须告知上游：深度对齐时上游会计算 frame.sky_mask 以排除天空
            # （processors.py:341-343 的 `& (~frame.sky_mask)`），而 sky_mask 由
            # instance_phrases 中的 SKY_PROMPT 计算得到（streams/base.py:263-267）
            frame.instance_phrases = {int(i): VideoFrame.SKY_PROMPT for i in sky}
            keep = instance == 0
            for i in sky:  # 天空不属于运动物体：上游将其排除在遮挡之外（streams/base.py:263-267）
                keep |= instance == i
            frame.mask = erode(keep, 5)
            return frame

    return _Given()


def run_depth(run: Run, pipeline, mode: str, raw: Path, frames, frame_numbers: list[int]) -> None:
    """「ViPE 深度图」：不解算，仅运行官方流水线中位姿之后的阶段（`default.py:88-98` 的 post 处理器）。"""
    from vipe.streams.base import ProcessedVideoStream

    job = run.job
    height, width = job.height, job.width
    run.stage("准备深度")
    slam, focal = rebuild_slam_output(job, len(frames), width, height)
    stream = make_stream([p for _, p in frames], job.fps, Path(job.data["id"]).name)
    given = ProcessedVideoStream(stream, [given_objects(job, frame_numbers), make_progress_processor(len(frames))])
    # 与上游对 slam_stream 的处理一致：post 处理器接在其后（default.py:135-138）
    out_stream = pipeline._add_post_processors(0, given, slam)
    info = save_depth(out_stream, mode, frame_numbers, raw, height, width)
    run.finish(
        frame_numbers,
        kind="depth",
        convention=CONVENTION,
        units="meters (camera Z depth, not ray length; the input camera's scale)",
        mode=mode,
        focal_px=focal,
        focal_source="camera",
        width=width,
        height=height,
        fps=job.fps,
        models=MODELS[mode],
        license=LICENSE[mode],
        **info,
    )


def main(job_path: str) -> None:
    run = Run.start(job_path, NODES, "ViPE")
    job, params = run.job, run.params
    # 「ViPE 相机解算」的参数名为 mode（标准 / 长镜头），「ViPE 深度图」的参数名为 model（深度模型）。
    # 两个节点共用本 worker，四个取值在 PRESETS / MODELS / LICENSE 三张表中使用同一组键
    mode = params.get("mode") or params["model"]
    with_depth = PRESETS[mode][2] is not None
    # 「ViPE 相机解算」的 Focal Length 位于参数中；「ViPE 深度图」接入相机，Focal Length 位于 camera_in.npz
    # （每个值只有一个来源，见 nodes.py ViPE.run 的说明）
    focal_px_given = params.get("focal_px")

    if with_depth:
        use_local_checkpoints(Path(os.environ.get("VIPE_HF_DIR") or job.weights_dir / "hf"), job.repo_dir)
    import vipe.utils.logging as vipe_logging
    from vipe.pipeline import make_pipeline
    from vipe.priors.track_anything.groundingdino.config import config as gdino_config

    vipe_logging.disable_progress_bar = True  # tqdm 进度条会淹没日志，进度由本 worker 自行报告
    vipe_logging.configure_logging()
    # GroundingDINO 的文本编码器从 weights/ 读取，而非 Hugging Face 缓存。
    gdino_config.text_encoder_type = os.environ.get("VIPE_BERT_DIR", "bert-base-uncased")

    shot = run.frames()
    frames, frame_numbers = shot.pairs, shot.numbers
    raw = job.raw_dir

    run.stage("准备 ViPE")
    cfg = compose_config(mode, raw, focal_px_given is not None)
    pipeline = make_pipeline(cfg.pipeline)
    if with_depth:
        # 「ViPE 深度图」：相机和三维点由输入接入，此处不重新解算，仅运行 post 阶段
        run_depth(run, pipeline, mode, raw, frames, frame_numbers)
        return

    # 返回的帧携带解算阶段分割出的运动物体编号图（取自缓存，不重新运行模型），
    # SLAM 的输出在传往 post 阶段时截获。ViPE 自身不写盘（compose_config 已关闭 save_artifacts）
    slam_outputs: list = []
    pipeline.return_output_streams = True
    add_post = pipeline._add_post_processors

    def add_post_processors(view_idx, video_stream, slam_output):
        slam_outputs.append(slam_output)
        return add_post(view_idx, video_stream, slam_output)

    pipeline._add_post_processors = add_post_processors

    height, width = job.height, job.width
    focal_geocalib: list[float] = []
    add_init = pipeline._add_init_processors

    def add_init_processors(video_stream):
        from vipe.pipeline.processors import IntrinsicEstimationProcessor

        stream = add_init(video_stream)
        intrinsics = stream.processors[0]  # GeoCalib，已在采样帧上运行
        assert isinstance(intrinsics, IntrinsicEstimationProcessor), type(intrinsics)
        fov_y = intrinsics.fov_y
        focal_geocalib.append(height / (2 * math.tan(fov_y / 2)))
        if focal_px_given is not None:
            intrinsics.fov_y = 2 * math.atan(height / (2 * focal_px_given))
        stream.processors.append(make_progress_processor(len(frames)))
        run.stage("解算相机")
        return stream

    pipeline._add_init_processors = add_init_processors

    stream = make_stream([p for _, p in frames], job.fps, Path(job.data["id"]).name)
    from vipe.streams.base import ProcessedVideoStream

    video = ProcessedVideoStream(stream, [])
    if mode == "pose_only":
        run.stage("读取序列")
        video = video.cache(desc="Reading frames")  # 与 `vipe infer` 的行为一致
    # 长镜头模式流式读取帧（任务的 PNG 数量已知，GeoCalib 的随机访问由 ViPE 自身缓存），
    # 可在内存中少保留一份全分辨率副本。
    run.stage("估计 Focal Length / 加载模型")
    t_solve = time.time()
    output = pipeline.run(video)
    run.frame_seconds.extend([(time.time() - t_solve) / len(frames)] * len(frames))  # 模型加载与整段解算的耗时，平均分摊到每帧
    slam = slam_outputs[0]
    run.stage("写出运动物体分割")
    extra = save_instances(output.output_streams[0], frame_numbers, raw, height, width)
    extra.update(save_slam_points(slam, raw))

    traj = slam.get_view_trajectory(0) if slam.rig is not None else slam.trajectory
    cam_to_world = traj.matrix().double().cpu().numpy()
    if cam_to_world.shape[0] != len(frames):
        raise RuntimeError(f"ViPE returned {cam_to_world.shape[0]} poses for {len(frames)} frames")
    fx, fy, cx, cy = (float(v) for v in slam.intrinsics[0][:4].cpu())
    focal_px = focal_px_given if focal_px_given is not None else 0.5 * (fx + fy)
    if focal_px_given is None and focal_geocalib and not 0.75 < focal_px / focal_geocalib[0] < 1.33:
        # bundle adjustment 仅在相机有位移（视差）时才能确定焦距；固定机位或纯摇镜头下焦距会漂移。
        say("W-VIPE-FOCALJUMP", solved=focal_px, single=float(focal_geocalib[0]), diff=abs(focal_px / focal_geocalib[0] - 1))
    if abs(cx - width / 2) > 1 or abs(cy - height / 2) > 1:
        say("W-VIPE-PRINCIPAL", cx=cx, cy=cy, center_x=width / 2, center_y=height / 2)

    save_npz(
        raw / "camera.npz",
        frames=np.asarray(frame_numbers, dtype=np.int64),
        focal_px=np.float64(focal_px),
        fx=np.float64(fx),
        fy=np.float64(fy),
        cx=np.float64(cx),
        cy=np.float64(cy),
        width=np.int64(width),
        height=np.int64(height),
        cam_to_world=cam_to_world,
    )

    # 相对第一帧的漂移：固定机位的镜头应接近零。
    rel = np.linalg.inv(cam_to_world[0]) @ cam_to_world
    translation = np.linalg.norm(rel[:, :3, 3], axis=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(rel[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    run.finish(
        frame_numbers,
        kind="camera",
        camera="camera.npz",
        convention=CONVENTION,
        units="meters (ViPE's metric scale from MoGe-2; approximate)",
        mode=mode,
        focal_px=focal_px,
        focal_source="user" if focal_px_given is not None else "vipe",
        focal_px_geocalib_init=focal_geocalib[0] if focal_geocalib else None,
        intrinsics={"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        width=width,
        height=height,
        fps=job.fps,
        drift={
            "max_translation": float(translation.max()),
            "max_rotation_deg": float(angle.max()),
        },
        models=MODELS[mode],
        **extra,
    )


if __name__ == "__main__":
    serve(main)
