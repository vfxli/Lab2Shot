"""One interface over image sequences, single images and video files."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

from ..errors import Invalid
from ..messages import Msg
from . import NotThere, images
from .sequence import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    FrameSequence,
    find_sequence,
    sequence_of,
)

DEFAULT_START_FRAME = 1001


@dataclass
class Source:
    """Frames keyed by frame number; pixels in their file encoding (not color converted)."""

    kind: str  # "sequence" | "video" | "still"
    display: str  # human readable input, e.g. /plates/sh010.####.exr
    frames: tuple[int, ...]
    width: int
    height: int
    fps: float | None  # known for video only
    colorspace_hint: str  # a file path for OCIO file rules
    colorspace: str | None = None  # detected from the stream (HDR video); wins over file rules
    _sequence: FrameSequence | None = None
    _video: Path | None = None
    _still: Path | None = None
    _start: int = DEFAULT_START_FRAME
    _rotate_cw: int = 0  # phone videos: clockwise turns needed to stand the picture upright

    def missing(self) -> list[int]:
        """Frames a sequence skips between its first and last (a video or a still has none)."""
        return self._sequence.missing() if self._sequence is not None else []

    def files(self) -> dict[int, Path]:
        """Frame -> image file (a sequence, or a still as one frame); videos have none."""
        if self._sequence is not None:
            return {f: self._sequence.path(f) for f in self.frames}
        if self._still is not None:
            return {self.frames[0]: self._still}
        raise Invalid(Msg("E-VIDEO-NOFRAMEFILES"))

    def iter_frames(self, frames: list[int] | None = None, scale: float = 1.0) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (frame, float32 HxWx3) in order; `scale` resizes video frames (proxy resolution)."""
        wanted = sorted(set(frames)) if frames is not None else list(self.frames)
        unknown = set(wanted) - set(self.frames)
        if unknown:
            raise Invalid(Msg("E-SOURCE-FRAMES", frames=sorted(unknown)[:10]))
        if self._sequence is not None:
            for f in wanted:
                yield f, images.read_rgb(self._sequence.path(f))
            return
        if self._still is not None:
            yield wanted[0], images.read_rgb(self._still)
            return
        keep = set(wanted)
        with open_video(self._video) as container:
            stream = container.streams.video[0]
            cc = stream.codec_context
            deep = "10" in (cc.pix_fmt or "") or "12" in (cc.pix_fmt or "") or "16" in (cc.pix_fmt or "")
            src_space = av.video.reformatter.Colorspace.BT2020 if cc.colorspace in (9, 10) else av.video.reformatter.Colorspace.ITU709
            # decode at the stored (unrotated) size, then turn upright
            sw, sh = (self.height, self.width) if self._rotate_cw in (90, 270) else (self.width, self.height)
            w = max(2, int(round(sw * scale / 2)) * 2)
            h = max(2, int(round(sh * scale / 2)) * 2)
            for index, frame in enumerate(container.decode(video=0)):
                number = self._start + index
                if number > wanted[-1]:
                    break
                if number in keep:
                    fmt = "rgb48le" if deep else "rgb24"
                    out = frame.reformat(width=w, height=h, format=fmt, src_colorspace=src_space, interpolation="AREA")
                    rgb = out.to_ndarray().astype(np.float32) / (65535.0 if deep else 255.0)
                    if self._rotate_cw:
                        rgb = np.ascontiguousarray(np.rot90(rgb, k=-self._rotate_cw // 90))
                    yield number, rgb


# FFmpeg opens a video as what its content says, whatever its name: a playlist or a concat list "named" .mp4 would make
# it read other files of this server (or fetch addresses) into frames anyone can look at. Only the containers of
# VIDEO_EXTS, and only the file itself.
VIDEO_OPTIONS = {"format_whitelist": "mov,matroska,avi,mxf", "protocol_whitelist": "file"}


def open_video(path: Path) -> av.container.InputContainer:
    try:
        return av.open(str(path), options=VIDEO_OPTIONS)
    except av.error.FFmpegError as exc:
        raise Invalid(Msg("E-VIDEO-UNREADABLE", file=path.name)) from exc


def open_source(spec: str | Path, start_frame: int = DEFAULT_START_FRAME) -> Source:
    """Open a sequence pattern / folder / frame file, or a video file.

    Video frames are numbered from `start_frame` (pipeline default 1001).
    """
    path = Path(spec).expanduser()
    if path.suffix.lower() in VIDEO_EXTS:
        if not path.is_file():
            raise NotThere(Msg("E-VIDEO-NOTFOUND", path=str(path)))
        with open_video(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else None
            width, height = stream.codec_context.width, stream.codec_context.height
            from .color import video_colorspace

            colorspace = video_colorspace(stream.codec_context.color_trc or 0)
            # FFmpeg reports the display rotation counter-clockwise; turn the frames clockwise by its negative
            first = next(container.decode(video=0), None)
            count = stream.frames
            if not count:  # e.g. WebM stores no frame count: decode to count, from the start again
                container.seek(0)
                count = sum(1 for _ in container.decode(video=0))
            rotate_cw = int(round(-(getattr(first, "rotation", 0) or 0))) % 360 if first is not None else 0
            if rotate_cw in (90, 270):
                width, height = height, width
        return Source(
            kind="video",
            display=str(path),
            frames=tuple(range(start_frame, start_frame + count)),
            width=width,
            height=height,
            fps=fps,
            colorspace_hint=str(path),
            colorspace=colorspace,
            _video=path,
            _start=start_frame,
            _rotate_cw=rotate_cw,
        )

    if path.is_file() and path.suffix.lower() in IMAGE_EXTS and sequence_of(path) is None:
        # a picture without a frame number in its name (sequence.py): one frame at start_frame
        width, height = images.image_size(path)
        return Source(
            kind="still",
            display=str(path),
            frames=(start_frame,),
            width=width,
            height=height,
            fps=None,
            colorspace_hint=str(path),
            _still=path,
            _start=start_frame,
        )

    seq = find_sequence(path)
    suffix = seq.tail[seq.tail.rfind("."):].lower()  # the frame number may sit before more of the name: ####_left.png
    if suffix not in IMAGE_EXTS:
        raise Invalid(Msg("E-IMAGE-FORMAT", suffix=suffix or seq.tail))
    first = seq.path(seq.first)
    width, height = images.image_size(first)
    return Source(
        kind="sequence",
        display=seq.pattern,
        frames=seq.frames,
        width=width,
        height=height,
        fps=None,
        colorspace_hint=str(first),
        _sequence=seq,
    )
