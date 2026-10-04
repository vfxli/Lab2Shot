"""Node families: conversion of standard worker results into packets, and the node base classes shared by every node
of a kind.

Workers of different projects that perform the same task write the same raw contract (documented per function). Their
nodes therefore differ only in parameters, such as model choice and licence, and never in how results are written.
This module only re-exports; lab2shot.sdk imports the family names from here, so an extension's imports through
lab2shot.sdk remain stable when a family moves between these files."""

from __future__ import annotations

from .base import Job, MissingFrames, RawOutput, WorkerNode
from ..kit.cameras import camera_port, lens_note, lens_stmaps, pass_camera, plate_lens, solved_camera
from ..kit.confidence import Confidence
from .flow import (
    OpticalFlow,
    OpticalFlowParams,
    clamped_flow_side,
    correspondence,
)
from .depth_camera import DepthCamera, LensWholeShotParams, PerFrameDepthCamera, WholeShotDepthCamera, WholeShotParams
from .humans import WorldHumans, WorldHumansParams
from .lens_calibration import LensCalibration
from .llm_text import LlmText
from .rig_motion import CleanupParams, DetectCleanupParams, FreeMotionParams, MotionGenParams, RigMotion, motion_fps_param
from .rig_retarget import RigRetarget, RigRetargetParams
from .gaussian import GaussianReconstruction
from .image_generation import DiffusionImage, DiffusionImageParams
from ..kit.rig import ModelJoint, RigModel, body_joints, humanoid_joints, mapping_param, part_labels, skeleton_param
from ..kit.rig_map import PartMap
from .light import LightProbe, LightProbeParams
from ..kit.maps import (LINEAR, MOTION, NEAREST, NORMALIZE, RESAMPLING, basecolor_map, camera_normals, depth_maps, family_points, fit, native_points_of,
                   frame_maps, points_params)
from .matte import GuidedMatte, MatteNode, Matting, foreground_entry, matte
from ..kit.ports import (
    basecolor_port,
    normal_port,
    rgb_port,
    values_port,
    conf_threshold_param,
    flow_resolution_param,
    follow_camera_param,
    loops_param,
    Measured,
    max_frames_param,
    measured_param,
    people_port,
    plate_mask_port,
    point_size_param,
    precision_level_param,
    resolution_param,
    unit_cm_param,
)
from .rigging import AutoRig, AutoRigParams, Meshes, one_mesh, top_influences
from .segmentation import Segmentation
from .tracks import Keypoints2D, PersonKeypoints, PointTracker, TrackParams, keypoints2d, track_queries, tracks
from .tracks3d import PointTracker3D, PointTracks3DParams

__all__ = [
    "RigRetarget", "RigRetargetParams", "GaussianReconstruction",
    "DiffusionImage", "DiffusionImageParams",
    "CleanupParams", "Confidence", "DetectCleanupParams", "GuidedMatte", "MatteNode", "LensCalibration", "LlmText", "RigMotion", "motion_fps_param", "Keypoints2D", "Matting", "PersonKeypoints", "Job", "MotionGenParams", "FreeMotionParams", "LensWholeShotParams", "LightProbe", "AutoRig", "AutoRigParams",
    "LightProbeParams", "MissingFrames", "ModelJoint", "DepthCamera", "PerFrameDepthCamera", "OpticalFlow", "OpticalFlowParams", "PointTracker",
    "Segmentation", "PointTracker3D", "PointTracks3DParams", "RawOutput", "WholeShotDepthCamera", "WholeShotParams", "RigModel", "TrackParams", "WorkerNode",
    "WorldHumans", "WorldHumansParams",
    "basecolor_map", "basecolor_port", "normal_port", "rgb_port", "values_port", "body_joints", "camera_normals", "camera_port", "clamped_flow_side", "conf_threshold_param", "correspondence", "Meshes", "one_mesh", "top_influences",
    "depth_maps", "family_points", "fit", "native_points_of", "foreground_entry", "points_params",
    "LINEAR", "NEAREST", "NORMALIZE", "MOTION", "RESAMPLING",
    "flow_resolution_param", "follow_camera_param", "frame_maps", "humanoid_joints", "lens_note", "lens_stmaps", "loops_param",
    "PartMap", "mapping_param", "matte", "skeleton_param", "Measured", "max_frames_param", "measured_param", "part_labels", "pass_camera", "people_port", "plate_lens",
    "plate_mask_port", "point_size_param", "precision_level_param", "resolution_param", "solved_camera",
    "keypoints2d", "track_queries", "tracks", "unit_cm_param",
]
