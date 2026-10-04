"""The scene's context for a job — one rule for every DCC: the values of the fixed outside names (templates/
_conventions.md) the scene knows, put together from a few things the host reads (Host.frame_range, fps, linear_unit,
camera_lens). The rule is the framework's, never a host's: the server works in the bound picture's own frame numbers
(the DCC's frame + the picture's frame_offset), the picture's colour space is the input's, and the lens is the bound
camera's — or, with none bound, that of the camera the picture was taken from (a picture bound through its camera).

Called on the DCC's main thread (it asks the host)."""

from __future__ import annotations


def _picture(bindings: dict) -> dict | None:
    """The first bound picture (an image or a video input)."""
    return next((b for b in bindings.values() if str(b.get("type", "")).startswith(("image", "video"))), None)


def _camera(bindings: dict, picture: dict | None) -> dict | None:
    """The bound camera (one bound to a scene object); else the camera the bound picture hangs on, when the host
    remembered it on the picture's binding ("camera": its ref)."""
    bound = next((b for b in bindings.values() if b.get("type") == "scene.camera" and b.get("ref")), None)
    if bound is not None:
        return bound
    if picture and picture.get("camera"):
        return {"ref": picture["camera"], "type": "scene.camera"}
    return None


def scene_context(host, bindings: dict) -> dict:
    """{fps, unit, first_frame / last_frame (in the bound picture's own frame numbers, when it is a sequence),
    colorspace_in (the bound picture's), focal / filmback (mm, the camera's)}; a value the scene does not know is left
    out."""
    out: dict = {}
    fps = host.fps()
    if fps:
        out["fps"] = float(fps)
    unit = host.linear_unit()
    if unit:
        out["unit"] = unit
    picture = _picture(bindings)
    if picture and picture.get("sequence"):
        span = host.frame_range()
        if span:
            offset = int(picture.get("frame_offset") or 0)
            out["first_frame"], out["last_frame"] = int(span[0]) + offset, int(span[1]) + offset
    if picture and picture.get("colorspace"):
        out["colorspace_in"] = picture["colorspace"]
    camera = _camera(bindings, picture)
    lens = host.camera_lens(camera) if camera else None
    if lens:
        out["focal"], out["filmback"] = float(lens[0]), float(lens[1])
    return out
