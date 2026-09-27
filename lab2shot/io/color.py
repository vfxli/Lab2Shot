"""OCIO color management: a single working space, with conversion only at the input and output boundaries.

The working space is sRGB-encoded Rec.709, scene-referred, float, unclamped and never tone-mapped: the OCIO role
`texture_paint` (named "sRGB Encoded Rec.709 (sRGB)" in the ACES studio config). Every picture packet is in this space
from the moment a reader creates it; solvers receive it clamped to 8 bits (the format ML models are trained on); the
viewer displays it unchanged (a monitor is sRGB); output nodes convert it to the target file's colour space. No stage
in between carries or converts colour.

sRGB is used instead of ACEScg (Nuke's choice) because the core routes pictures between ML models, which all operate
in sRGB, and performs no lighting computation; ACEScg would require a transform at every model boundary and introduce
banding in every 8-bit proxy. Scene-referred sRGB is used instead of a display space because the transfer curve and
the primaries matrix are purely analytic: EXR highlights above 1 survive in float and an EXR written back is
bit-exact, whereas a display view (a tone map) is not invertible.

Two rules maintain this model:
- An SDR display-referred input (PNG, JPG, TIF, SDR video; sRGB or Rec.1886 transfer) is the working space bit for
  bit, i.e. exactly what a monitor displayed and what the models saw. Converting it through OCIO would apply the
  config's view transform in reverse (an inverse tone map), which is incorrect.
- A scene-referred input (an EXR in ACEScg / ACES2065-1 / linear, a log camera space) is converted by a colour space
  transform, never through a display view. HDR video (PQ, HLG) is the only input that is tone-mapped: the reader
  applies its HDR view in reverse and the SDR view forward, once.

Config precedence: the color.config setting (/admin settings page) > $OCIO > OCIO's built-in latest ACES studio config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import PyOpenColorIO as OCIO

from ..config import settings
from ..errors import Invalid
from ..messages import Msg
from . import images

BUILTIN_LATEST = "ocio://studio-config-latest"

# Colour spaces are looked up by a role every OCIO config defines or by a Color Interop Forum ID, never by the name a
# particular config gives a colour space ("ACEScg", "Raw"), so a studio's own config works unchanged.
DATA = "data"  # values, not colour: depth, masks, normals, motion
SCENE_LINEAR = "scene_linear"  # the working space of renders and composites
TEXTURE = "texture_paint"  # sRGB-encoded Rec.709, scene-referred: 8-bit pictures a model reads or paints
WORKING = TEXTURE  # the working space (see module docstring): picture packets, solver frames, the viewer
LINEAR_REC709 = "lin_rec709_scene"  # linear Rec.709 radiance (a light probe)
REC2100_PQ, REC2100_HLG = "rec2100_pq_display", "rec2100_hlg_display"
USE_DISPLAY_NAME = "<USE_DISPLAY_NAME>"  # a shared view's colour space is named after its display


@dataclass(frozen=True, eq=False)  # identity hash: usable as a cache key
class ColorConfig:
    config: OCIO.Config
    uri: str
    origin: str  # "user", "$OCIO" or "built-in"

    @property
    def name(self) -> str:
        return self.config.getName() or self.uri

    def colorspace(self, name: str) -> OCIO.ColorSpace:
        cs = self.config.getColorSpace(name)
        if cs is None:
            raise Invalid(Msg("E-COLOR-NOSPACE", config=self.name, space=name))
        return cs

    def colorspace_for_file(self, path: str) -> str:
        """Return the default for a reader's colour space parameter (「色彩空间」) for this file, determined by the file
        format only (the pixels are not analysed). An EXR maps to the config's scene_linear (ACEScg in an ACES config;
        OCIO's own file rule would give ACES2065-1, which renders are not in). Other files follow the config's file
        rules (PNG/JPG/TIF the sRGB display space, video Rec.1886). An image matched only by the default rule that
        stores float pixels (a Radiance .hdr: HDRI libraries ship radiance, never a display encoding) maps to the
        config's linear Rec.709 (by its interop ID). The page writes this value into the parameter when the file is
        selected (NodeDef.choices "default"), so the user can see and change it."""
        from .sequence import IMAGE_EXTS

        if Path(path).suffix.lower() == ".exr":
            return self.colorspace(SCENE_LINEAR).getName()
        name = self.config.getColorSpaceFromFilepath(str(path))[0]
        if Path(path).suffix.lower() in IMAGE_EXTS and self.config.filepathOnlyMatchesDefaultRule(str(path)) \
                and Path(path).is_file() and images.stores_float(path) and self.config.getColorSpace(LINEAR_REC709) is not None:
            return self.config.getColorSpace(LINEAR_REC709).getName()
        return name

    def display_view(self) -> tuple[str, str]:
        display = self.config.getDefaultDisplay()
        return display, self.config.getDefaultView(display)


@cache
def load_config(user: str | None = None) -> ColorConfig:
    uri = user if user is not None else settings()["color.config"]
    if uri:
        origin = "user"
    elif os.environ.get("OCIO"):
        uri, origin = os.environ["OCIO"], "$OCIO"
    else:
        uri, origin = BUILTIN_LATEST, "built-in"
    return ColorConfig(OCIO.Config.CreateFromFile(uri), uri, origin)


def is_display_referred(cs: OCIO.ColorSpace) -> bool:
    """Return True when pixel values are already display values (video, 8-bit sRGB)."""
    if cs.getReferenceSpaceType() == OCIO.REFERENCE_SPACE_DISPLAY:
        return True
    return cs.getEncoding() in ("sdr-video", "hdr-video")


def _hdr_display(cfg: ColorConfig, cs_name: str) -> tuple[str, str] | None:
    """Return the (display, view) whose output is the given HDR display colorspace, e.g. Rec.2100-HLG."""
    c = cfg.config
    for display in c.getDisplays():
        for view in c.getViews(display):
            if "HDR" not in view:
                continue
            name = c.getDisplayViewColorSpaceName(display, view)
            # shared views report <USE_DISPLAY_NAME>: the display colorspace is named after the display
            if name == cs_name or (name == USE_DISPLAY_NAME and display == cs_name):
                return display, view
    return None


def space(wanted: str, cfg: ColorConfig | None = None) -> str:
    """Return the name the active config gives a role or an interop ID (as carried by packets and written to file
    headers); raises E-COLOR-NOSPACE when the config has none."""
    return (cfg or load_config()).colorspace(wanted).getName()


def working_space(cfg: ColorConfig | None = None) -> str:
    """Return the config's name for the working space (see module docstring), as recorded by every picture packet."""
    return space(WORKING, cfg)


def is_working(cs_name: str, cfg: ColorConfig | None = None) -> bool:
    """Return True when pixels in `cs_name` are already in the working space without conversion: the working space
    itself, or an SDR display-referred space (sRGB or Rec.1886 transfer; first rule in the module docstring). Data
    spaces (depth, masks) are not colour and are likewise never converted."""
    c = cfg or load_config()
    cs = c.colorspace(cs_name)
    if cs.isData() or cs.getName() == c.colorspace(WORKING).getName():
        return True
    return is_display_referred(cs) and cs.getEncoding() != "hdr-video"


@cache
def _to_working_processor(cfg: ColorConfig, src: str) -> OCIO.CPUProcessor | None:
    """Processor from `src` to the working space; None when the pixels are already in it (is_working). For HDR
    video: the HDR view in reverse to scene-linear, then the SDR view forward. This is the only tone map in the
    pipeline and is applied once, in the reader."""
    if is_working(src, cfg):
        return None
    cs = cfg.colorspace(src)
    if cs.getEncoding() == "hdr-video":
        hdr = _hdr_display(cfg, src)
        if hdr is None:
            raise Invalid(Msg("E-COLOR-NOHDRVIEW", space=src))
        display, view = cfg.display_view()
        scene = space(SCENE_LINEAR, cfg)
        group = OCIO.GroupTransform()
        group.appendTransform(OCIO.DisplayViewTransform(src=scene, display=hdr[0], view=hdr[1], direction=OCIO.TRANSFORM_DIR_INVERSE))
        group.appendTransform(OCIO.DisplayViewTransform(src=scene, display=display, view=view))
        # the SDR view outputs the display space (sRGB / Rec.1886), which by the first rule is the working space
        return cfg.config.getProcessor(group).getDefaultCPUProcessor()
    tf = OCIO.ColorSpaceTransform(src=src, dst=working_space(cfg))
    return cfg.config.getProcessor(tf).getDefaultCPUProcessor()


def to_working(rgb: np.ndarray, cfg: ColorConfig, src: str) -> np.ndarray:
    """Convert float RGB in `src` to float RGB in the working space (unclamped; used on reader ingest). HDR video
    uses a baked 3D LUT (its encoded values lie in 0..1; about 10x faster than per-pixel ACES transforms, with no
    visible difference)."""
    img = np.ascontiguousarray(rgb, dtype=np.float32)
    proc = _to_working_processor(cfg, src)
    if proc is None:
        return img
    if cfg.colorspace(src).getEncoding() == "hdr-video":
        return apply_lut(img, _lut(cfg, src))
    img = img.copy()
    proc.applyRGB(img)
    return img


def to_working_picture(picture: np.ndarray, cfg: ColorConfig, src: str) -> np.ndarray:
    """Convert a picture in `src` (RGB, or premultiplied RGBA) to the working space, transforming the unpremultiplied
    colour (apply_premultiplied). Pixels already in the working space are returned unchanged."""
    proc = _to_working_processor(cfg, src)
    return np.ascontiguousarray(picture, dtype=np.float32) if proc is None else apply_premultiplied(proc, picture)


def to_srgb8(rgb: np.ndarray, cfg: ColorConfig, src: str) -> np.ndarray:
    """Convert float RGB in `src` to uint8 sRGB: the working space clamped to 0..1 and quantised. This is the input
    solvers receive and what the viewer displays. A packet already in the working space (every picture a reader
    creates) is only clamped."""
    return images.as_uint8(to_working(rgb, cfg, src))


@cache
def _from_working_processor(cfg: ColorConfig, dst: str) -> OCIO.CPUProcessor | None:
    """Processor from the working space to `dst` (used by output nodes writing files); None when `dst` needs no
    conversion (an SDR display space: the file receives the working pixels and the writer clamps them to its bit
    depth)."""
    if is_working(dst, cfg):
        return None
    tf = OCIO.ColorSpaceTransform(src=working_space(cfg), dst=dst)
    return cfg.config.getProcessor(tf).getDefaultCPUProcessor()


def convert_picture(picture: np.ndarray, cfg: ColorConfig, src: str, dst: str) -> np.ndarray:
    """Convert a picture in `src` to `dst` via the working space (used by output nodes writing files). `src` is the
    working space for every picture a reader creates, or the colour space recorded by an older packet."""
    return from_working(to_working_picture(picture, cfg, src), cfg, dst)


def from_working(picture: np.ndarray, cfg: ColorConfig, dst: str) -> np.ndarray:
    """Convert a picture in the working space (RGB, or premultiplied RGBA) to `dst`, transforming the unpremultiplied
    colour. Pixels are returned unchanged when `dst` needs no conversion."""
    proc = _from_working_processor(cfg, dst)
    return np.ascontiguousarray(picture, dtype=np.float32) if proc is None else apply_premultiplied(proc, picture)


LUT_SIZE = 65


@cache
def _lut(cfg: ColorConfig, src: str) -> np.ndarray:
    """Sample the to-working transform of an HDR video space on a LUT_SIZE^3 grid over 0..1 input."""
    g = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    grid = np.ascontiguousarray(np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3))
    _to_working_processor(cfg, src).applyRGB(grid)
    return grid.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear 3D LUT lookup for 0..1 RGB images."""
    n = lut.shape[0] - 1
    x = np.clip(img, 0.0, 1.0) * n
    i = np.minimum(x.astype(np.int32), n - 1)
    f = x - i
    r, g, b = i[..., 0], i[..., 1], i[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c00 = lut[r, g, b] * (1 - fr) + lut[r + 1, g, b] * fr
    c10 = lut[r, g + 1, b] * (1 - fr) + lut[r + 1, g + 1, b] * fr
    c01 = lut[r, g, b + 1] * (1 - fr) + lut[r + 1, g, b + 1] * fr
    c11 = lut[r, g + 1, b + 1] * (1 - fr) + lut[r + 1, g + 1, b + 1] * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return (c0 * (1 - fb) + c1 * fb).astype(np.float32)


def working_table(cfg: ColorConfig, src: str, size: int, lo: float, hi: float) -> np.ndarray | None:
    """Sample the to-working transform of `src` for a browser without OCIO (a local preview of a selected EXR,
    matching the server's display). The grid is size^3 over log2 of the linear value from lo to hi stops, red varying
    fastest, clipped to 0..1 (the monitor's range). Returns None when values are displayed as-is (already in the
    working space, or data)."""
    proc = _to_working_processor(cfg, src)
    if proc is None:
        return None
    g = np.exp2(np.linspace(lo, hi, size, dtype=np.float64)).astype(np.float32)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    grid = np.ascontiguousarray(np.stack([r, gg, b], -1).reshape(-1, 3))
    proc.applyRGB(grid)
    return np.clip(grid, 0.0, 1.0)


def video_colorspace(transfer: int, cfg: ColorConfig | None = None) -> str | None:
    """Return the config's display space for a video stream's transfer characteristic (ISO/IEC 23091-2: 16 PQ,
    18 HLG), looked up by interop ID; None for SDR or when the config has no such space."""
    wanted = {16: REC2100_PQ, 18: REC2100_HLG}.get(int(transfer))
    cs = (cfg or load_config()).config.getColorSpace(wanted) if wanted else None
    return cs.getName() if cs is not None else None


def apply_premultiplied(proc: OCIO.CPUProcessor, img: np.ndarray) -> np.ndarray:
    """Apply an OCIO processor to a picture (RGB, or premultiplied RGBA). RGBA colour is divided by alpha, transformed
    and multiplied again, as Nuke's Read and Write do, so the transform never operates on colour darkened by coverage;
    alpha is preserved."""
    if img.shape[-1] != 4:
        rgb = np.ascontiguousarray(img, dtype=np.float32).copy()
        proc.applyRGB(rgb)
        return rgb
    rgb = np.ascontiguousarray(images.unpremultiply(img))
    proc.applyRGB(rgb)
    return np.concatenate([rgb * img[..., 3:4], img[..., 3:4]], axis=-1).astype(np.float32)
