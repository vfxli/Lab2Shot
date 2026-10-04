"""The adapter API: all an extension uses of Lab2Shot, in one place.

An adapter's main-environment modules (adapters/<name>/extension.py, nodes.py and their helpers) import from here, the
worker SDK (lab2shot_worker) and their own folder, nothing else of Lab2Shot (the layering rule), so the
core can change behind this without touching them. It only re-exports: each name is defined where the core keeps it.
Workers use the worker SDK, never this.

SDK_API is its version. An extension says which version it is written for (Extension.sdk); one written for another
is left out with the reason (lab2shot/adapters.py), never half-working.

A node that runs a worker subclasses WorkerNode (or a family): prepare(ctx) -> Job, convert(ctx, raw, job) with raw a
RawOutput. A node with its own cook runs its worker through ctx.run_worker(image, extra=, inputs=) and reads the folder
through RawOutput(folder, MissingFrames....); ctx.input_files("boxes", ...) gives the files of connected inputs to send.
"""

from __future__ import annotations

from ..data.payloads import (
    SCENE_FILE,
    UNIT,
    curves_packet,
    scene_curves_packet,
    image_files,
    image_packet,
    points_packet,
    scene_packet,
    still_packet,
    read_tracks,
    tracks_packet,
    write_boxes
)
from ..data.camera import CameraSamples
# 一个「模型」口的 3D 数据是每帧变形的点缓存：口上写 kinds=(DEFORMING,)，
# 只写得了静止模型的格式就在接线时拒绝它（data/types.py，nodes/output.py Writes）
from ..data.types import DEFORMING
from ..data.lens_models import COLMAP_MODELS, PINHOLE_MODELS, MODELS as LENS_TABLE, distorts, external_camera, distortion, model_labels as lens_model_labels
from ..data.packet import Packet
from ..data.scene import pack
from ..data.evaluate import scene_points
from ..data.scene import open_scene
from ..data.scene_arrays import Axes, scene_arrays
from ..extensions.manual import (ExtensionWeights, Install, ManualError, ManualItem, body_model_weight, manual_weight,
                                 unwrap_licence)
from ..io.files import open_member
from ..config import THIRD_PARTY_DIR
from ..nodes.tags import BASIC, COMMERCIAL, NONCOMMERCIAL, RESEARCH, label as licence_label
from ..extensions.spec import (
    CUDA_13_2_TOOLKIT,
    EnvSpec,
    Extension,
    GitSource,
    InstallError,
    LicenseInfo,
    Weight,
    hf_file,
    hf_weights,
)
from ..extensions import downloads  # tools to declare pinned downloads (Download, hf, pip_git, archive_zip)
from ..io.color import working_space
from ..io.images import read_named
from ..data.units import (usd_points_to_m, CV_TO_GL, DEFAULT_FPS, DEFAULT_WIDTH, FILMBACK_MM, M_TO_CM, focal_mm,
                          opencv_points_to_usd, opencv_poses_to_usd, to_cm)
from ..io.usd import (
    PERSON_ID,
    ROOT_PATH,
    SkinnedCharacter,
    create_stage,
    save_stage,
    write_character,
    write_mesh,
    write_rig,
)
from ..data.skeleton import character_of_model, merge_weights, rig_of_model, scaled_to_ground
from ..data.standard_bodies import StandardBody
from ..data.contracts import KEEPS, NEW_PICTURE, PLATE_FRAME, STMAP_SHAPE, Shape, warped_by
from ..nodes.base import (EITHER, NodeDef, NodeParams, P, Port, Reads, empty_packet, fp16_param, parse_corners, parse_figures,
                          say_bad_entries)
from ..availability import All, AnyOf, Because, Not
from ..nodes.applies import (Cost, Fact, Licence, OptionTrait, Param, Wired, WiredPicture, WiredType, fact, incoming,
                             licence_traits)
from ..nodes.core.geometry import points_from_depth
from ..nodes.expects import DistinctNames, FrameCount, HighDynamicRange, OwnCamera, SameShot
from ..nodes.handles import FIGURE_JOINTS, Handle, figure_handle
from ..nodes.core.mask import class_word
from ..nodes.clipboard import Pasteable, put_clipboard
from ..nodes.official import Official  # 每个三方节点声明「官方的口 + 上游行号」
from ..nodes.lens import (CAMERA_HAS_LENS, NO_LENS, CameraLensParams, SolvedLensParams, LensParams, focal_param,
                         packed_lens, unpacked_lens,
                         LensGroup, GroupModel, lens_groups, table_of, lens_identity, core_group)
from ..data.values import LENS
from ..nodes.formats import WorkerImport, import_file_param, selection_param, selection_ports
from ..nodes.output import Format, OutputSettings, Writes, fps_param, name_param
from ..nodes.kit.cameras import send_camera
from ..nodes.kit.retarget_needs import GENERIC, JointNeeds
from ..nodes.families import (
    Confidence,
    GaussianReconstruction,
    DiffusionImage,
    DiffusionImageParams,
    LINEAR,
    NEAREST,
    NORMALIZE,
    MOTION,
    camera_normals,
    family_points,
    native_points_of,
    points_params,
    Job,
    MissingFrames,
    RawOutput,
    WorkerNode,
    AutoRig,
    AutoRigParams,
    GuidedMatte,
    MatteNode,
    Keypoints2D,
    Matting,
    CleanupParams,
    DetectCleanupParams,
    body_joints,
    RigMotion,
    RigRetarget,
    RigRetargetParams,
    motion_fps_param,
    MotionGenParams,
    FreeMotionParams,
    PartMap,
    ModelJoint,
    LensCalibration,
    LensWholeShotParams,
    LightProbe,
    LlmText,
    LightProbeParams,
    PerFrameDepthCamera,
    OpticalFlow,
    OpticalFlowParams,
    PointTracker,
    PointTracker3D,
    PointTracks3DParams,
    WholeShotDepthCamera,
    WholeShotParams,
    Segmentation,
    TrackParams,
    WorldHumans,
    WorldHumansParams,
    basecolor_map,
    basecolor_port,
    normal_port,
    rgb_port,
    values_port,
    camera_port,
    clamped_flow_side,
    conf_threshold_param,
    correspondence,
    depth_maps,
    flow_resolution_param,
    follow_camera_param,
    foreground_entry,
    frame_maps,
    humanoid_joints,
    lens_note,
    lens_stmaps,
    loops_param,
    mapping_param,
    matte,
    skeleton_param,
    Measured,
    max_frames_param,
    measured_param,
    pass_camera,
    people_port,
    plate_lens,
    plate_mask_port,
    point_size_param,
    precision_level_param,
    resolution_param,
    solved_camera,
    track_queries,
    tracks,
    unit_cm_param,
)
from ..data.payloads import window_of
from ..data.values import FLOAT, VECTOR, read as read_value, value_list_packet, value_packet
from ..errors import Invalid, NothingToCook
from ..messages import Msg

SDK_API = 2  # 2: the worker-node template (WorkerNode: prepare -> Job -> convert(ctx, RawOutput, Job)), Confidence

__all__ = [
    "SDK_API",
    # messages (lab2shot/messages): an error a user reads carries one; its words are in adapters/<name>/i18n/<lang>.toml
    "Invalid", "Msg",
    # found nothing to give (no face in the shot): raised from a cook, the node's outputs are empty, its notice kept
    "NothingToCook",
    # the extension spec (extension.py)
    "CUDA_13_2_TOOLKIT", "EnvSpec", "Extension", "GitSource", "InstallError", "LicenseInfo", "Weight", "hf_file",
    "hf_weights", "downloads", "manual_weight", "body_model_weight", "ManualItem", "Install", "ExtensionWeights", "ManualError",
    "unwrap_licence", "open_member", "THIRD_PARTY_DIR",
    # licence classes (nodes/tags.py): LicenseInfo(tag=), and NodeDef.licence for a node that differs from its extension
    "BASIC", "COMMERCIAL", "NONCOMMERCIAL", "RESEARCH", "licence_label",
    # node definitions (nodes.py)
    "EITHER", "FIGURE_JOINTS", "class_word", "Handle", "NodeDef", "NodeParams", "P", "Port", "empty_packet", "figure_handle",
    "parse_corners", "parse_figures", "say_bad_entries", "fp16_param",
    "KEEPS", "NEW_PICTURE", "PLATE_FRAME", "STMAP_SHAPE", "Shape", "warped_by",
    # what applies (nodes/applies.py): when a parameter does something, what a node costs and whose licence it is, what a
    # choice changes about it
    "All", "AnyOf", "Because", "Cost", "Fact", "Licence", "Not", "OptionTrait", "licence_traits", "Param", "Wired", "WiredPicture", "WiredType", "fact", "incoming",
    # the lens (nodes/lens.py): its parameters, the rule that picks it and says where it came from
    "Official",
    "CAMERA_HAS_LENS", "CameraLensParams", "LensParams", "SolvedLensParams", "NO_LENS", "focal_param", "plate_lens",
    # Focal Length（px）→ Focal Length（mm）（「镜头标定」家族 nodes/families/lens_calibration.py 的一处算法）：AnyCalib、GeoCalib 的「Focal Length」口
    # 交出去之前都走它
    # another program's camera model read through the one model table (data/lens_models.py EXTERNAL_MODELS)
    "external_camera", "distortion", "COLMAP_MODELS", "PINHOLE_MODELS", "LENS_TABLE", "distorts",
    "packed_lens", "unpacked_lens", "LENS", "LensGroup", "GroupModel", "lens_groups", "table_of", "lens_identity", "core_group",
    # 镜头模型 id → 界面名：自己会估镜头的扩展（AnyCalib、COLMAP）用它给「拟合模型」的选项起名，
    # 和核心「LensDistortion」上的是同一份列表、同一个词
    "lens_model_labels",
    # basic values (nodes/values.py): a value output, a value read back
    "FLOAT", "VECTOR", "read_value", "value_list_packet", "value_packet",
    # what an input expects beyond its type (usage checks)
    "DistinctNames", "FrameCount", "HighDynamicRange", "OwnCamera", "SameShot",
    # format modules (nodes/formats.py): the import node read by the module's worker, its parameters, the file's units
    # and axes; the output-settings node (a file format), the scene arrays a writer's worker takes and the pack of
    # several scene wires into one
    "Axes", "WorkerImport", "import_file_param", "selection_param", "selection_ports", "Reads",
    "Format", "OutputSettings", "Writes", "fps_param", "name_param", "open_scene", "pack", "scene_arrays", "scene_curves_packet",
    "scene_points", "send_camera",
    # the worker-node template (nodes/families/base.py): prepare -> Job -> worker -> convert(ctx, RawOutput, Job)
    "Confidence", "Job", "MissingFrames", "RawOutput", "WorkerNode",
    # node families, their ports, parameters and helpers
    "AutoRig", "AutoRigParams", "GuidedMatte", "MatteNode", "LensCalibration", "Keypoints2D", "Matting", "LensWholeShotParams", "LightProbe", "LightProbeParams", "PerFrameDepthCamera",
    "OpticalFlow", "OpticalFlowParams", "PointTracker", "PointTracker3D", "PointTracks3DParams", "WholeShotDepthCamera",
    "WholeShotParams", "Segmentation", "TrackParams", "WorldHumans", "WorldHumansParams",
    "RigMotion", "RigRetarget", "RigRetargetParams", "JointNeeds", "GENERIC", "GaussianReconstruction",
    "DiffusionImage", "DiffusionImageParams", "motion_fps_param", "MotionGenParams", "FreeMotionParams", "PartMap", "ModelJoint", "humanoid_joints", "mapping_param", "skeleton_param", "LlmText",
    "CleanupParams", "DetectCleanupParams", "body_joints",
    "basecolor_port", "normal_port", "rgb_port", "values_port", "camera_port", "people_port", "plate_mask_port",
    "conf_threshold_param", "flow_resolution_param", "follow_camera_param", "loops_param", "Measured", "max_frames_param", "measured_param",
    "point_size_param", "precision_level_param", "resolution_param", "unit_cm_param",
    "basecolor_map", "camera_normals", "clamped_flow_side", "correspondence", "depth_maps", "family_points", "native_points_of", "foreground_entry", "frame_maps",
    "LINEAR", "NEAREST", "NORMALIZE", "MOTION",  # frame_maps 的第四项：这张结果图怎么重采样（nodes/kit/maps.py）
    "UNIT",  # 0..1：一张图的值就在这个范围里时声明出来（遮罩、alpha、置信度、粗糙度、金属度），
             # 视图就按它显示，不用按整段的 1%–99% 分位去猜（一张 0.2–0.8 的 alpha 会被拉成 0–1 来看）
    "lens_note", "lens_stmaps", "matte", "pass_camera", "points_params",
    # 「复制到 <软件>」：结果里带一段别的软件能粘贴的文字（nodes/clipboard.py）
    "Pasteable", "put_clipboard",
    "solved_camera", "track_queries", "tracks",
    # depth + camera -> a point cloud (nodes/core/geometry.py DepthToPoints's own logic, reused by nodes that carry a
    # point cloud port at the family level instead of one node solving it again)
    "points_from_depth",
    # packets: building and reading them
    "Packet", "SCENE_FILE", "CameraSamples", "window_of", "curves_packet", "image_files", "image_packet",
    "points_packet", "read_named", "read_tracks", "scene_curves_packet", "scene_packet", "still_packet", "tracks_packet", "write_boxes",
    # colour (io/color.py): the colour space of a picture made for a monitor, as the config in use names it

    "working_space",
    # scenes: writing USD, a camera's samples
    "CV_TO_GL", "DEFAULT_FPS", "focal_mm", "DEFAULT_WIDTH", "FILMBACK_MM", "M_TO_CM", "to_cm", "PERSON_ID", "ROOT_PATH", "SkinnedCharacter", "character_of_model", "rig_of_model",
    # a body for the core's 「标准人」 (Extension.standard_bodies, data/standard_bodies.py) and the two steps its skin
    # takes like the core's own: weights of dropped joints merged into kept ones, scaled and stood on the ground
    "StandardBody", "merge_weights", "scaled_to_ground", "write_rig", "create_stage", "opencv_points_to_usd", "opencv_poses_to_usd", "usd_points_to_m", "save_stage", "DEFORMING",
    "write_character", "write_mesh",
]
