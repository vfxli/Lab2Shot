"""The light-probe conventions, numpy only: the file names, the lat-long orientation, how a USD dome light takes the
map, luminance and chromaticities. The node side and the workers read them from here."""

from __future__ import annotations

import numpy as np

ENVMAP, PREVIEW = "envmap.exr", "preview.png"


LUMA = np.array([0.212671, 0.715160, 0.072169])  # Rec.709 luminance


ORIENTATION = {
    "frame": (
        "camera-relative: directions are in the camera frame of the chosen plate frame (not world, not gravity-aligned)"
    ),
    "projection": "equirectangular lat-long, width = 2 x height, full sphere",
    "pixel_to_direction": (
        "u = (x + 0.5) / W, v = (y + 0.5) / H (x = column from the left, y = row from the top); "
        "theta = 2*pi*u, phi = pi*v; direction in USD/OpenGL camera space "
        "(+X right, +Y up, -Z = viewing direction) = (-sin(phi)*sin(theta), cos(phi), sin(phi)*cos(theta))"
    ),
    "view_direction": (
        "u = 0.5, v = 0.5: the vertical centre line of the image (between columns W/2-1 and W/2, "
        "e.g. 511|512 at W=1024) at mid height faces exactly where the camera looks (the plate's centre)"
    ),
    "columns": "u = 0.25 camera left (-X), u = 0.5 forward (-Z), u = 0.75 camera right (+X), u = 0 and 1 (left/right edges) behind the camera (+Z)",
    "up": "top row = camera up (+Y, the plate's up direction); bottom row = camera down. A tilted or rolled camera tilts the map's horizon with it",
    "handedness": "not mirrored: it reads like a panorama seen from inside the sphere (what is left of the camera is left of the centre)",
}


USD_DOMELIGHT = {
    "spec": (
        "UsdLuxDomeLight / DomeLight_1 (poleAxis 'scene' on a Y-up stage) follow the OpenEXR lat-long spec: top pole +Y; "
        "longitude 0 (texture centre, u = 0.5) faces +Z; longitude +pi/2 (u = 0.25) faces +X; the left/right edges face -Z"
    ),
    "rotation": (
        "this map puts the camera's viewing direction at the texture centre, so the DomeLight's local-to-world "
        "rotation must be R_camera * RotY(180 deg): RotY(180) turns the texture centre from +Z to -Z (the USD camera's "
        "viewing axis) and u = 0.25 from +X to -X (camera left); R_camera (the camera's world rotation, same frame "
        "as the plate) then carries it into the scene. Translation does not matter for a dome"
    ),
    "usda_camera_at_identity": 'xformOpOrder = ["xformOp:rotateY"], float xformOp:rotateY = 180',
    "usda_general": (
        'xformOpOrder = ["xformOp:orient", "xformOp:rotateY"] with xformOp:orient = the camera\'s world rotation '
        "(USD applies the last op first: rotateY(180) in the camera frame, then the camera rotation)"
    ),
    "texture_format": "inputs:texture:format = \"latlong\"",
}


def latlong_directions(height: int, width: int) -> np.ndarray:
    """(H, W, 3) unit directions, USD/OpenGL camera space (+X right, +Y up, -Z forward), pixel centres.

    Same sphere as ball2envmap.py (theta = 2*pi*u across, phi = pi*v down, reflect
    vector (sin phi cos theta, sin phi sin theta, cos phi) with the camera at +X):
    its X is our +Z (towards the camera), its Y our -X, its Z our +Y.
    """
    theta = (np.arange(width) + 0.5) / width * 2 * np.pi
    phi = (np.arange(height) + 0.5) / height * np.pi
    theta, phi = np.meshgrid(theta, phi, indexing="xy")
    return np.stack([-np.sin(phi) * np.sin(theta), np.cos(phi), np.sin(phi) * np.cos(theta)], axis=-1)


REC709_CHROMATICITIES = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)  # Rec.709 primaries, D65 white
