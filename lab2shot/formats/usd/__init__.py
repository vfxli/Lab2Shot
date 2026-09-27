"""USD format support. USD is Lab2Shot's native representation of 3D data, so this module runs in the core
environment. It implements the same format-module interface (lab2shot/nodes/formats.py) as other formats: an import
node, an output-settings node and a reader (reader.py)."""

SUFFIXES = (".usd", ".usda", ".usdc", ".usdz")
