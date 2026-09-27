"""Core nodes: they run in the main environment and are shared by all extensions.

Each module corresponds to one menu area (for example, motion_vectors holds nodes derived from motion vectors) and lists
its node classes in NODES. The format modules in the core environment (USD, PLY, BVH; lab2shot/formats) contribute their
import and output-settings nodes. The output area lists 「输出」 first, followed by the core output-settings nodes (USD,
images, data files); format modules provided by extensions append theirs afterwards.
"""

from ...formats.bvh import nodes as bvh
from ...formats.ply import nodes as ply
from ...formats.usd import nodes as usd
from . import (camera, data_output, flow, geometry, image, image_output, input, lens_distortion, mask, motion_vectors,
               output, people, scene, sketch, uv, values)

CORE_NODES = (input.NODES + values.NODES + flow.NODES + image.NODES + motion_vectors.NODES + camera.NODES + lens_distortion.NODES
              + mask.NODES + geometry.NODES + uv.NODES + people.NODES + scene.NODES + sketch.NODES + output.NODES + usd.NODES + ply.NODES
              + bvh.NODES + image_output.NODES
              + data_output.NODES)

__all__ = ["CORE_NODES"]
