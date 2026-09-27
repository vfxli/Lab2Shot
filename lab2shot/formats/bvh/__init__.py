"""The BVH format module: a motion-capture skeleton and its motion (a 骨架动画), read in the core's environment into
scene arrays (lab2shot_shared/scene_arrays.py) like every format read into them (lab2shot/nodes/formats.py
ArraysImport). BVH records its frame time but not its unit: the node says it."""

SUFFIXES = (".bvh",)
