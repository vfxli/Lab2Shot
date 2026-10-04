"""Core nodes: they run in the main environment and are shared by all extensions.

Each module corresponds to one menu area (for example, motion_vectors holds nodes derived from motion vectors) and lists
its node classes in NODES. No format node is here: the format modules in the core environment (image, data files,
Nuke, USD, PLY, BVH; lab2shot/formats, the same layer as the nodes) are not imported here: their import and
output-settings nodes are lab2shot/formats NODES, which the registry adds (nodes/registry.py core_nodes); format
modules provided by extensions add theirs as extensions do.
"""

from . import (camera, ensemble, ensemble_camera, ensemble_depth, ensemble_normal, ensemble_segment, flow, geometry, image, input, lens_distortion, mask, motion_vectors,
               output, people, retarget, scene, sketch, uv, values)

CORE_NODES = (input.NODES + values.NODES + flow.NODES + image.NODES + motion_vectors.NODES + camera.NODES + lens_distortion.NODES
              + mask.NODES + geometry.NODES + ensemble.NODES + ensemble_camera.NODES + ensemble_depth.NODES + ensemble_normal.NODES + ensemble_segment.NODES + uv.NODES + people.NODES + scene.NODES + retarget.NODES + sketch.NODES + output.NODES)

__all__ = ["CORE_NODES"]
