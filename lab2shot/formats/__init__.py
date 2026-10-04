"""The format modules that run in the core's environment: every import and output-settings node of the core lives here,
one module per format, and the registry adds them (nodes/registry.py core_nodes); the core's own nodes (nodes/core)
hold no format node. formats and nodes are one layer of cli/check_arch.py LAYERS ("nodes"): either may import the
other, and the core's nodes use a format's plain parsing where they read its text (nodes/core/lens_distortion.py: a
Nuke script, formats/nuke/parse.py). By convention the core's nodes never import a format's nodes.

    image       image sequences (EXR / PNG / JPG) and multi-layer EXR
    data_files  boxes, 2D tracks and curves for other software (JSON, CSV, 3DEqualizer, .chan, USD, Nuke text)
    nuke        Nuke scripts (.nk): the text a camera solve is delivered in and the one a tracking department hands over
    usd         Lab2Shot's own representation of 3D data
    ply, bvh    read into scene arrays by the core itself (lab2shot/nodes/formats.py ArraysImport)

The other formats live in their extensions. Every format that carries a scene follows lab2shot/nodes/formats.py."""

from .bvh import nodes as _bvh
from .data_files import nodes as _data_files
from .image import nodes as _image
from .nuke import nodes as _nuke
from .ply import nodes as _ply
from .usd import nodes as _usd

# the import and output-settings nodes (nodes/registry.py core_nodes), in the menu's order
NODES = _image.NODES + _data_files.NODES + _nuke.NODES + _usd.NODES + _ply.NODES + _bvh.NODES
