"""Nuke format support: the text of a Nuke script (.nk), the form in which nodes are pasted into a comp.

Nuke nodes are plain text, so a result can be transferred as text: 「复制到 Nuke」 places the text of a node on the
clipboard for pasting into a comp. An output-settings node writes that text as a file of its result
(nodes/output.py OutputSettings.clipboard) and the server delivers it (server/packets.py clipboard); the core has no
knowledge of Nuke.

    script.py   the format itself: node blocks, animated knobs, names Nuke accepts
    camera.py   a camera as a Camera3
    tracks.py   tracked points as a Tracker4, a plane's four corners as a CornerPin2D
    nodes.py    「Nuke 相机输出设置」, the node that writes camera.py's text
"""
