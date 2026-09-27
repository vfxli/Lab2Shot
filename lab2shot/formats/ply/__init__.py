"""PLY format support: the vertices of a scan or photogrammetry export (a 点云) and, when the file has faces, its mesh
(a 模型), read in the core environment into scene arrays (lab2shot_shared/scene_arrays.py) in the same way as other
array-based formats (lab2shot/nodes/formats.py ArraysImport). PLY records neither unit nor up axis; both are node
parameters."""

SUFFIXES = (".ply",)
