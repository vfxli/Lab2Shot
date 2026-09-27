"""The format modules that run in the core's environment: USD (Lab2Shot's own representation), PLY and BVH (read into scene
arrays by the core itself, lab2shot/nodes/formats.py ArraysImport), and Nuke (.nk: the script a camera solve is
delivered in and the one a tracking department hands over; plain text, with no import or output node of its own).
The other formats live in their extensions. Every format that carries a scene follows lab2shot/nodes/formats.py."""
